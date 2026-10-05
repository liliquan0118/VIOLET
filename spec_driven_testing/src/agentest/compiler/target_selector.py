"""Select isolated primary and minimal-difference contrast coverage cells."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifacts import (
    CAP_COVERAGE_MATCH_MANIFEST,
    CAP_DECISION_CONFIGURATION_MODEL,
    artifact_document_for_capability,
    artifact_identity_view,
    content_sha256,
)
from .source_contracts import validate_resolved_spec_target


SELECTED_COVERAGE_TARGET_SCHEMA_VERSION = (
    "agentspectesting.selected-coverage-target/v0.1"
)
ISOLATED_ATOMIC_BRANCH_WITNESS = "isolated_atomic_branch_witness"


class TargetCellSelectionError(ValueError):
    """Raised when accepted cells cannot satisfy the requested policy uniquely."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = deepcopy(dict(details or {}))


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetCellSelectionError(
            "invalid_selector_input", f"{path} must be an object"
        )
    return value


def _artifact_identity(resolved_artifacts: Mapping[str, Any]) -> dict[str, Any]:
    return artifact_identity_view(resolved_artifacts)


def _verify_resolution_artifact_identity(
    resolved_target: Mapping[str, Any],
    current_identity: Mapping[str, Any],
) -> None:
    identities = [
        item
        for item in resolved_target.get("resolution_evidence") or []
        if isinstance(item, Mapping)
        and item.get("evidence_kind") == "accepted_artifact_identity"
    ]
    if len(identities) != 1:
        raise TargetCellSelectionError(
            "resolution_artifact_identity_not_unique",
            "ResolvedSpecTarget must contain exactly one accepted artifact identity",
            details={"identity_count": len(identities)},
        )
    resolved_identity = {
        "artifact_bundle_id": identities[0].get("artifact_bundle_id"),
        "compile_run_id": identities[0].get("compile_run_id"),
        "artifact_catalog_fingerprint": identities[0].get(
            "artifact_catalog_fingerprint"
        ),
        "artifact_hashes": deepcopy(identities[0].get("artifact_hashes") or {}),
    }
    if resolved_identity != current_identity:
        raise TargetCellSelectionError(
            "resolution_artifact_identity_mismatch",
            "ResolvedSpecTarget and selector artifacts do not share content identity",
            details={
                "resolved_identity": resolved_identity,
                "current_identity": deepcopy(dict(current_identity)),
            },
        )


def _artifact_document(
    resolved_artifacts: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    capability = {
        "coverage_match_manifest": CAP_COVERAGE_MATCH_MANIFEST,
        "decision_configuration_model": CAP_DECISION_CONFIGURATION_MODEL,
    }.get(name)
    if capability is None:
        raise TargetCellSelectionError(
            "unknown_artifact_role",
            f"target selector has no capability mapping for {name!r}",
        )
    return _mapping(
        artifact_document_for_capability(resolved_artifacts, capability),
        f"artifact-capability[{capability}]",
    )


def _unique_records(
    records: Any,
    predicate,
) -> list[Mapping[str, Any]]:
    if not isinstance(records, list):
        return []
    return [value for value in records if isinstance(value, Mapping) and predicate(value)]


def _decision_region(
    decision_model: Mapping[str, Any],
    family_id: str,
    focus_predicate_id: str,
) -> Mapping[str, Any]:
    matches = _unique_records(
        decision_model.get("operation_outcome_decision_regions"),
        lambda value: value.get("operation_policy_family_id") == family_id
        and focus_predicate_id in (value.get("decision_factor_predicate_ids") or []),
    )
    if len(matches) != 1:
        raise TargetCellSelectionError(
            "decision_region_not_unique",
            "focus predicate must belong to exactly one operation outcome region",
            details={
                "operation_policy_family_id": family_id,
                "focus_predicate_id": focus_predicate_id,
                "match_count": len(matches),
            },
        )
    return matches[0]


def _validate_focus_role(
    decision_model: Mapping[str, Any],
    family_id: str,
    focus_predicate_id: str,
) -> None:
    matches = _unique_records(
        decision_model.get("predicate_role_inventory"),
        lambda value: value.get("operation_policy_family_id") == family_id
        and value.get("predicate_id") == focus_predicate_id,
    )
    if len(matches) != 1:
        raise TargetCellSelectionError(
            "focus_predicate_role_not_unique",
            "focus predicate must have exactly one decision role record",
            details={"match_count": len(matches)},
        )
    roles = set(matches[0].get("roles") or [])
    if (
        "operation_outcome_decision_factor" not in roles
        or matches[0].get("included_in_combinatorial_denominator") is not True
    ):
        raise TargetCellSelectionError(
            "focus_predicate_not_selectable",
            "focus predicate is not a combinatorial operation outcome factor",
            details={
                "focus_predicate_id": focus_predicate_id,
                "roles": sorted(roles),
            },
        )


def _factor_assignment(
    cell: Mapping[str, Any],
    decision_factor_ids: set[str],
) -> dict[str, bool]:
    assignment = {}
    for item in cell.get("required_factor_values") or []:
        if not isinstance(item, Mapping):
            raise TargetCellSelectionError(
                "invalid_cell_factor_value",
                "required_factor_values must contain objects",
                details={"coverage_cell_id": cell.get("coverage_cell_id")},
            )
        predicate_id = item.get("predicate_id")
        truth = item.get("expected_truth_value")
        if predicate_id not in decision_factor_ids:
            raise TargetCellSelectionError(
                "cell_contains_foreign_decision_factor",
                "operation-outcome cell contains a factor outside its decision region",
                details={
                    "coverage_cell_id": cell.get("coverage_cell_id"),
                    "predicate_id": predicate_id,
                },
            )
        if not isinstance(truth, bool):
            raise TargetCellSelectionError(
                "cell_factor_truth_not_boolean",
                "isolated atomic selection requires boolean decision factors",
                details={
                    "coverage_cell_id": cell.get("coverage_cell_id"),
                    "predicate_id": predicate_id,
                },
            )
        if predicate_id in assignment:
            raise TargetCellSelectionError(
                "duplicate_cell_factor",
                "coverage cell assigns a decision factor more than once",
                details={
                    "coverage_cell_id": cell.get("coverage_cell_id"),
                    "predicate_id": predicate_id,
                },
            )
        assignment[predicate_id] = truth
    return assignment


def _full_assignment_cells(
    manifest: Mapping[str, Any],
    family_id: str,
    decision_region_id: str,
    decision_factor_ids: list[str],
) -> list[tuple[Mapping[str, Any], dict[str, bool]]]:
    factor_set = set(decision_factor_ids)
    result = []
    for cell in manifest.get("coverage_cells") or []:
        if not isinstance(cell, Mapping):
            continue
        if (
            cell.get("operation_policy_family_id") != family_id
            or cell.get("decision_region_id") != decision_region_id
            or cell.get("cell_kind") != "operation_outcome_configuration"
        ):
            continue
        assignment = _factor_assignment(cell, factor_set)
        dont_care = set(cell.get("dont_care_predicate_ids") or [])
        if dont_care & factor_set:
            continue
        if set(assignment) != factor_set:
            continue
        result.append((cell, assignment))
    return result


def _cell_view(
    cell: Mapping[str, Any],
    assignment: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "coverage_cell_id": cell.get("coverage_cell_id"),
        "decision_region_id": cell.get("decision_region_id"),
        "cell_kind": cell.get("cell_kind"),
        "expected_operation_decision": cell.get("expected_operation_decision"),
        "required_factor_values": [
            {
                "predicate_id": predicate_id,
                "expected_truth_value": assignment[predicate_id],
            }
            for predicate_id in sorted(assignment)
        ],
        "dont_care_predicate_ids": deepcopy(
            cell.get("dont_care_predicate_ids") or []
        ),
    }


def _selected_target_fingerprint_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(value))
    payload.pop("selected_target_fingerprint", None)
    return payload


def select_target_cells(
    resolved_target: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
    *,
    selection_policy: str = ISOLATED_ATOMIC_BRANCH_WITNESS,
) -> dict[str, Any]:
    """Select one isolated witness and its one-factor-flip contrast."""

    target = validate_resolved_spec_target(resolved_target)
    if target["resolution_status"] != "resolved_unique":
        raise TargetCellSelectionError(
            "resolved_target_not_unique",
            "cell selection requires resolution_status=resolved_unique",
            details={"resolution_status": target["resolution_status"]},
        )
    if selection_policy != ISOLATED_ATOMIC_BRANCH_WITNESS:
        raise TargetCellSelectionError(
            "unsupported_selection_policy",
            f"unsupported target cell selection policy {selection_policy!r}",
        )
    binding = _mapping(target.get("model_binding"), "$.model_binding")
    if binding.get("required_truth_value") is not True:
        raise TargetCellSelectionError(
            "isolated_policy_requires_positive_focus",
            "isolated atomic branch witness currently requires focus truth=true",
            details={
                "required_truth_value": binding.get("required_truth_value")
            },
        )

    current_identity = _artifact_identity(resolved_artifacts)
    _verify_resolution_artifact_identity(target, current_identity)
    manifest = _artifact_document(resolved_artifacts, "coverage_match_manifest")
    decision_model = _artifact_document(
        resolved_artifacts, "decision_configuration_model"
    )
    family_id = binding["operation_policy_family_id"]
    focus_id = binding["focus_predicate_id"]
    _validate_focus_role(decision_model, family_id, focus_id)
    region = _decision_region(decision_model, family_id, focus_id)
    decision_region_id = region.get("decision_region_id")
    factor_ids = region.get("decision_factor_predicate_ids") or []
    if (
        not isinstance(decision_region_id, str)
        or not decision_region_id
        or not isinstance(factor_ids, list)
        or not factor_ids
        or any(not isinstance(value, str) or not value for value in factor_ids)
        or len(set(factor_ids)) != len(factor_ids)
    ):
        raise TargetCellSelectionError(
            "invalid_decision_region",
            "decision region must provide a unique non-empty boolean factor list",
        )

    blocker = region.get("outer_blocker")
    blocker_id = None
    blocker_nonblocking_value = None
    if blocker is not None:
        blocker = _mapping(blocker, "$.decision_region.outer_blocker")
        blocker_id = blocker.get("predicate_id")
        blocking_value = blocker.get("blocking_truth_value")
        if blocker_id not in factor_ids or not isinstance(blocking_value, bool):
            raise TargetCellSelectionError(
                "invalid_outer_blocker",
                "outer blocker must reference one boolean decision factor",
            )
        blocker_nonblocking_value = not blocking_value
        if focus_id == blocker_id:
            raise TargetCellSelectionError(
                "focus_is_outer_blocker_not_supported",
                "positive eligibility isolation does not apply when focus is the outer blocker",
            )

    isolation_assignment = {predicate_id: False for predicate_id in factor_ids}
    isolation_assignment[focus_id] = True
    if blocker_id is not None:
        isolation_assignment[blocker_id] = blocker_nonblocking_value

    full_cells = _full_assignment_cells(
        manifest,
        family_id,
        decision_region_id,
        factor_ids,
    )
    primary_candidates = [
        (cell, assignment)
        for cell, assignment in full_cells
        if assignment == isolation_assignment
        and cell.get("expected_operation_decision")
        == binding["expected_operation_decision"]
    ]
    if len(primary_candidates) != 1:
        raise TargetCellSelectionError(
            "isolated_primary_cell_not_unique",
            "isolated assignment and accepted decision must select exactly one cell",
            details={
                "candidate_ids": sorted(
                    str(cell.get("coverage_cell_id"))
                    for cell, _ in primary_candidates
                ),
                "isolation_assignment": isolation_assignment,
                "expected_operation_decision": binding[
                    "expected_operation_decision"
                ],
            },
        )
    primary_cell, primary_assignment = primary_candidates[0]

    contrast_assignment = deepcopy(primary_assignment)
    contrast_assignment[focus_id] = not primary_assignment[focus_id]
    contrast_candidates = [
        (cell, assignment)
        for cell, assignment in full_cells
        if assignment == contrast_assignment
    ]
    if len(contrast_candidates) != 1:
        raise TargetCellSelectionError(
            "minimal_contrast_cell_not_unique",
            "one-factor-flip contrast must select exactly one accepted cell",
            details={
                "candidate_ids": sorted(
                    str(cell.get("coverage_cell_id"))
                    for cell, _ in contrast_candidates
                ),
                "contrast_assignment": contrast_assignment,
            },
        )
    contrast_cell, resolved_contrast_assignment = contrast_candidates[0]
    changed_factors = [
        predicate_id
        for predicate_id in factor_ids
        if primary_assignment[predicate_id]
        != resolved_contrast_assignment[predicate_id]
    ]
    if changed_factors != [focus_id]:
        raise TargetCellSelectionError(
            "contrast_not_focus_only",
            "minimal contrast must differ only on the focus predicate",
            details={"changed_factor_ids": changed_factors},
        )

    result = {
        "schema_version": SELECTED_COVERAGE_TARGET_SCHEMA_VERSION,
        "resolved_target_fingerprint": target["resolved_target_fingerprint"],
        "source_branch_id": target["source_branch_id"],
        "selection_policy": selection_policy,
        "artifact_identity": current_identity,
        "operation_policy_family_id": family_id,
        "focus_predicate_id": focus_id,
        "decision_region": {
            "decision_region_id": decision_region_id,
            "decision_factor_predicate_ids": deepcopy(factor_ids),
            "outer_blocker": deepcopy(region.get("outer_blocker")),
        },
        "primary_target": _cell_view(primary_cell, primary_assignment),
        "contrast_target": {
            **_cell_view(contrast_cell, resolved_contrast_assignment),
            "contrast_kind": "focus_truth_flip_minimal_difference",
            "changed_factor_values": [
                {
                    "predicate_id": focus_id,
                    "primary_truth_value": primary_assignment[focus_id],
                    "contrast_truth_value": resolved_contrast_assignment[focus_id],
                }
            ],
        },
        "selection_trace": {
            "full_assignment_cell_count": len(full_cells),
            "isolation_assignment": [
                {
                    "predicate_id": predicate_id,
                    "expected_truth_value": isolation_assignment[predicate_id],
                    "assignment_role": (
                        "focus"
                        if predicate_id == focus_id
                        else "outer_blocker_nonblocking"
                        if predicate_id == blocker_id
                        else "alternative_factor_inactive"
                    ),
                }
                for predicate_id in factor_ids
            ],
            "primary_candidate_ids": [primary_cell.get("coverage_cell_id")],
            "contrast_candidate_ids": [contrast_cell.get("coverage_cell_id")],
            "contrast_hamming_distance": 1,
        },
    }
    result["selected_target_fingerprint"] = content_sha256(
        _selected_target_fingerprint_payload(result)
    )
    return result


def validate_selected_coverage_target(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the deterministic fingerprint of a selected target result."""

    item = _mapping(value, "$selected_coverage_target")
    if item.get("schema_version") != SELECTED_COVERAGE_TARGET_SCHEMA_VERSION:
        raise TargetCellSelectionError(
            "unsupported_selected_target_schema",
            f"unsupported schema {item.get('schema_version')!r}",
        )
    expected = item.get("selected_target_fingerprint")
    if not isinstance(expected, str) or not expected:
        raise TargetCellSelectionError(
            "selected_target_fingerprint_missing",
            "selected target fingerprint must be present",
        )
    actual = content_sha256(_selected_target_fingerprint_payload(item))
    if actual != expected:
        raise TargetCellSelectionError(
            "selected_target_fingerprint_mismatch",
            "selected target content does not match its fingerprint",
            details={"expected": expected, "actual": actual},
        )
    return deepcopy(dict(item))


__all__ = [
    "ISOLATED_ATOMIC_BRANCH_WITNESS",
    "SELECTED_COVERAGE_TARGET_SCHEMA_VERSION",
    "TargetCellSelectionError",
    "select_target_cells",
    "validate_selected_coverage_target",
]
