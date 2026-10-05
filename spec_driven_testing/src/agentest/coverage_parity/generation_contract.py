"""Project uncovered coverage cells into dataset-independent generation contracts."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_DOMAINS,
    GAP_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
)


SCHEMA_VERSION = (
    "adequacy.policy_coverage_cell_generation_contracts.v1"
)


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _observation_requirements(expression: Any) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if isinstance(expression, dict):
        if expression.get("kind") == "observation_value_ref":
            values.append(
                {
                    "observation_value_id": expression.get(
                        "observation_value_id"
                    ),
                    "value_description": expression.get(
                        "value_description"
                    ),
                }
            )
        for child in expression.values():
            values.extend(_observation_requirements(child))
    elif isinstance(expression, list):
        for child in expression:
            values.extend(_observation_requirements(child))
    deduplicated = {}
    for value in values:
        value_id = value.get("observation_value_id")
        if isinstance(value_id, str):
            deduplicated[value_id] = value
    return [deduplicated[key] for key in sorted(deduplicated)]


def _literal_constraints(expression: Any) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if isinstance(expression, dict):
        kind = expression.get("kind")
        if kind == "comparison":
            values.append(
                {
                    "operator": expression.get("operator"),
                    "left": copy.deepcopy(expression.get("left")),
                    "right": copy.deepcopy(expression.get("right")),
                }
            )
        elif kind == "membership":
            values.append(
                {
                    "operator": expression.get("operator"),
                    "left": copy.deepcopy(expression.get("left")),
                    "right": copy.deepcopy(expression.get("right")),
                }
            )
        for child in expression.values():
            values.extend(_literal_constraints(child))
    elif isinstance(expression, list):
        for child in expression:
            values.extend(_literal_constraints(child))
    return values


def _normalized_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.casefold().split()).strip(" .,;:")


def _condition_temporal_role(
    factor: dict[str, Any] | None,
    path_contracts: list[dict[str, Any]],
) -> str:
    """Use existing relation roles to separate setup from result state."""
    surface = _normalized_text(
        factor.get("surface_text") if factor else None
    )
    if not surface:
        return "pre_or_activation_condition"
    for path in path_contracts:
        relation_monitors = (
            (path.get("oracle_monitors") or {}).get(
                "relation_monitors"
            )
            or []
        )
        for monitor in relation_monitors:
            if monitor.get("accounting_role") != "post_operation_state":
                continue
            activation = monitor.get("activation_condition") or {}
            if _normalized_text(activation.get("source_text")) == surface:
                return "post_operation_state"
    return "pre_or_activation_condition"


def _generation_readiness(
    *,
    cell: dict[str, Any],
    factor_contracts: list[dict[str, Any]],
    selector: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = []
    requirements = cell.get("required_factor_values") or []
    expected_by_predicate: dict[str, set[bool]] = {}
    for value in requirements:
        expected_by_predicate.setdefault(
            value["predicate_id"], set()
        ).add(value["expected_truth_value"])
    if any(len(values) > 1 for values in expected_by_predicate.values()):
        reasons.append("contradictory_required_predicate_truth_values")
    contract_ids = {
        value["predicate_id"] for value in factor_contracts
    }
    missing = sorted(set(expected_by_predicate) - contract_ids)
    if missing:
        reasons.append(
            "missing_factor_evaluator_contracts:" + ",".join(missing)
        )
    if not selector.get("operation_expressions"):
        reasons.append("missing_operation_expression")
    if cell["cell_kind"] == "unconditional_requirement" and not (
        cell.get("source_path_match_contract_ids")
        or cell.get("source_path_test_specification_ids")
    ):
        reasons.append("unconditional_cell_has_no_path_contract")
    if reasons:
        return "blocked", reasons
    if any(
        value.get("evaluation_class") != "mechanical"
        for value in factor_contracts
    ):
        return (
            "requires_bounded_semantic_fixture_design",
            ["one_or_more_predicates_are_not_purely_mechanical"],
        )
    return "ready_for_fixture_selection", []


def materialize_contracts(
    *,
    domain: str,
    manifest: dict[str, Any],
    gaps: dict[str, Any],
) -> dict[str, Any]:
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("domain") != domain
        or manifest.get("failures")
        or gaps.get("schema_version") != GAP_SCHEMA_VERSION
        or gaps.get("domain") != domain
        or gaps.get("failures")
    ):
        raise ValueError("Coverage-cell generation inputs are incompatible")
    cells = {
        value["coverage_cell_id"]: value
        for value in manifest.get("coverage_cells") or []
    }
    families = {
        value["operation_policy_family_id"]: value
        for value in (
            manifest.get("operation_family_match_manifests") or []
        )
    }
    contracts = []
    for gap in gaps.get("coverage_gap_records") or []:
        cell = cells[gap["coverage_cell_id"]]
        family = families[cell["operation_policy_family_id"]]
        factor_map = {
            value["predicate_id"]: value
            for value in family.get("factor_evaluator_contracts") or []
        }
        factor_contracts = [
            copy.deepcopy(factor_map[value["predicate_id"]])
            for value in cell.get("required_factor_values") or []
            if value["predicate_id"] in factor_map
        ]
        prerequisites = {
            value["prerequisite_monitor_id"]: value
            for value in family.get("prerequisite_monitors") or []
        }
        outcomes = {
            value["outcome_monitor_id"]: value
            for value in family.get("outcome_monitors") or []
        }
        paths = {
            value["path_match_contract_id"]: value
            for value in family.get("path_match_contracts") or []
        }
        references = (
            (
                "factor",
                [
                    value["predicate_id"]
                    for value in cell.get(
                        "required_factor_values"
                    )
                    or []
                ],
                set(factor_map),
            ),
            (
                "prerequisite",
                cell.get("candidate_prerequisite_monitor_ids") or [],
                set(prerequisites),
            ),
            (
                "outcome",
                cell.get("source_outcome_monitor_ids") or [],
                set(outcomes),
            ),
            (
                "path",
                cell.get("source_path_match_contract_ids") or [],
                set(paths),
            ),
        )
        for kind, required_ids, available_ids in references:
            missing = set(required_ids) - available_ids
            if missing:
                raise ValueError(
                    f"{cell['coverage_cell_id']} has missing {kind} "
                    f"references: {sorted(missing)}"
                )
        readiness, reasons = _generation_readiness(
            cell=cell,
            factor_contracts=factor_contracts,
            selector=family["operation_selector"],
        )
        required_paths = [
            copy.deepcopy(paths[value])
            for value in cell.get(
                "source_path_match_contract_ids"
            )
            or []
            if value in paths
        ]
        conditions = []
        for requirement in cell.get("required_factor_values") or []:
            factor = factor_map.get(requirement["predicate_id"])
            conditions.append(
                {
                    "predicate_id": requirement["predicate_id"],
                    "required_truth_value": requirement[
                        "expected_truth_value"
                    ],
                    "surface_text": (
                        factor.get("surface_text") if factor else None
                    ),
                    "evaluation_class": (
                        factor.get("evaluation_class") if factor else None
                    ),
                    "typed_expression": copy.deepcopy(
                        factor.get("typed_expression")
                        if factor
                        else None
                    ),
                    "runtime_binding": copy.deepcopy(
                        factor.get("runtime_binding")
                        if factor
                        else None
                    ),
                    "temporal_role": _condition_temporal_role(
                        factor, required_paths
                    ),
                    "required_observation_values": (
                        _observation_requirements(
                            factor.get("typed_expression")
                        )
                        if factor
                        else []
                    ),
                    "literal_constraints": (
                        _literal_constraints(
                            factor.get("typed_expression")
                        )
                        if factor
                        else []
                    ),
                }
            )
        contracts.append(
            {
                "test_generation_contract_id": (
                    f"CTGC::{gap['coverage_gap_id']}"
                ),
                "coverage_gap_id": gap["coverage_gap_id"],
                "target_coverage_cell_id": cell[
                    "coverage_cell_id"
                ],
                "cell_kind": cell["cell_kind"],
                "operation_policy_family_id": cell[
                    "operation_policy_family_id"
                ],
                "target_operation": copy.deepcopy(
                    family["operation_selector"]
                ),
                "required_conditions": conditions,
                "dont_care_predicate_ids": copy.deepcopy(
                    cell.get("dont_care_predicate_ids") or []
                ),
                "required_prerequisites": [
                    copy.deepcopy(prerequisites[value])
                    for value in cell.get(
                        "candidate_prerequisite_monitor_ids"
                    )
                    or []
                    if value in prerequisites
                ],
                "expected_operation_decision": cell.get(
                    "expected_operation_decision"
                ),
                "correctness_oracle_status": (
                    "coverage_only"
                    if (
                        cell["cell_kind"]
                        == "operation_outcome_configuration"
                        and cell.get("expected_operation_decision")
                        in {None, "unspecified"}
                    )
                    else "determinate"
                ),
                "required_outcome_monitors": [
                    copy.deepcopy(outcomes[value])
                    for value in cell.get(
                        "source_outcome_monitor_ids"
                    )
                    or []
                    if value in outcomes
                ],
                "required_path_match_contracts": [
                    copy.deepcopy(value) for value in required_paths
                ],
                "coverage_oracle": {
                    "target_coverage_cell_id": cell[
                        "coverage_cell_id"
                    ],
                    "required_condition_match_status": (
                        "condition_matched"
                    ),
                    "required_predicate_truth": [
                        {
                            "predicate_id": value["predicate_id"],
                            "expected_truth_value": value[
                                "expected_truth_value"
                            ],
                        }
                        for value in (
                            cell.get("required_factor_values") or []
                        )
                    ],
                },
                "generation_policy": {
                    "generate_one_target_cell_per_test": True,
                    "preserve_required_truth_values": True,
                    "dont_care_values_may_be_selected_for_feasibility": True,
                    "concrete_object_ids_are_selected_later": True,
                    "natural_language_must_not_add_constraints": True,
                },
                "generation_readiness": readiness,
                "readiness_reasons": reasons,
            }
        )
    readiness_counts = Counter(
        value["generation_readiness"] for value in contracts
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": domain,
        "test_generation_contracts": contracts,
        "failures": [],
        "summary": {
            "n_generation_contracts": len(contracts),
            "by_generation_readiness": dict(
                sorted(readiness_counts.items())
            ),
            "dataset_independent": True,
            "n_llm_calls": 0,
        },
    }


def materialize_from_files(
    *,
    domain: str,
    manifest_path: str | Path,
    gaps_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    result = materialize_contracts(
        domain=domain,
        manifest=_load(manifest_path),
        gaps=_load(gaps_path),
    )
    result["source"] = {
        "coverage_manifest": str(manifest_path),
        "coverage_gap_inventory": str(gaps_path),
    }
    _write(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=DEFAULT_DOMAINS)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gaps", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = materialize_from_files(
        domain=args.domain,
        manifest_path=args.manifest,
        gaps_path=args.gaps,
        output_path=args.output,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
