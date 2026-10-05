"""Compile and evaluate the Execution-to-Branch Given-validity gate.

The expected predicate assignment comes only from a compiled coverage target.
The observed assignment comes only from a bound Driver fixture and its runtime
pre-state check.  The gate therefore cannot make a failed fixture look valid by
copying the fixture's own ``expected_assignment`` field.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256
from ..compiler.then_atomization import ThenAtomizationError
from .bound import BoundOracleError, evaluate_bound_online_execution


CONTRACT_VERSION = "agentspectesting.branch-eligibility-contract/v0.1"
WITNESS_VERSION = "agentspectesting.branch-eligibility-witness/v0.1"
GATE_VERSION = "agentspectesting.branch-eligibility-gate/v0.1"
GATE_STATUSES = frozenset({"satisfied", "not_satisfied", "incomplete"})
_REQUIRED_RUNTIME_MILESTONES = (
    "fixture_valid",
    "target_operation_requested",
    "requesting_identity_bound",
    "target_entity_bound",
    "required_user_facts_available",
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _compiled_plan_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(value))
    payload.pop("compiled_plan_fingerprint", None)
    provenance = payload.get("provenance") or {}
    provenance.pop("artifact_bundle_path", None)
    for artifact in (provenance.get("artifacts") or {}).values():
        artifact.pop("path", None)
    return payload


def _bound_plan_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(value))
    payload.pop("bound_driver_plan_fingerprint", None)
    (payload.get("database_source") or {}).pop("database_path", None)
    return payload


def _validate_compiled_plan_identity(plan: Mapping[str, Any]) -> None:
    fingerprint = plan.get("compiled_plan_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != content_sha256(
        _compiled_plan_payload(plan)
    ):
        raise ThenAtomizationError("invalid compiled plan fingerprint")
    if not isinstance(plan.get("compiled_plan_id"), str) or not plan["compiled_plan_id"]:
        raise ThenAtomizationError("compiled plan ID is required")


def _target_candidates(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = []
    primary = plan.get("target")
    if isinstance(primary, Mapping):
        candidates.append(primary)
    for value in plan.get("contrast_targets") or []:
        candidates.append(_mapping(value, "$.contrast_targets[]"))
    return candidates


def _select_target(
    plan: Mapping[str, Any], *, target_name: str | None, coverage_cell_id: str | None
) -> Mapping[str, Any]:
    if not target_name and not coverage_cell_id:
        raise ThenAtomizationError("target_name or coverage_cell_id is required")
    matches = []
    for target in _target_candidates(plan):
        focal = _mapping(target.get("focal_configuration"), "$.target.focal_configuration")
        if target_name and target.get("target_name") != target_name:
            continue
        if coverage_cell_id and focal.get("coverage_cell_id") != coverage_cell_id:
            continue
        matches.append(target)
    if len(matches) != 1:
        raise ThenAtomizationError("eligibility target selector must resolve exactly once")
    return matches[0]


def _assignment(values: Any, *, expected_key: str, path: str) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise ThenAtomizationError(f"{path} must be a non-empty array")
    result = []
    seen = set()
    for index, raw in enumerate(values):
        item = _mapping(raw, f"{path}[{index}]")
        predicate_id = item.get("predicate_id")
        truth = item.get(expected_key)
        if not isinstance(predicate_id, str) or not predicate_id or predicate_id in seen:
            raise ThenAtomizationError(f"{path} predicate IDs must be unique")
        if not isinstance(truth, bool):
            raise ThenAtomizationError(f"{path} truth values must be Boolean")
        seen.add(predicate_id)
        result.append({"predicate_id": predicate_id, expected_key: truth})
    return sorted(result, key=lambda item: item["predicate_id"])


def compile_branch_eligibility_contract(
    compiled_plan: Mapping[str, Any],
    oracle_branch_id: str,
    *,
    target_name: str | None = None,
    coverage_cell_id: str | None = None,
) -> dict[str, Any]:
    """Freeze the selected branch's required Given assignment."""

    plan = deepcopy(dict(_mapping(compiled_plan, "$compiled_plan")))
    _validate_compiled_plan_identity(plan)
    if not isinstance(oracle_branch_id, str) or not oracle_branch_id:
        raise ThenAtomizationError("oracle_branch_id is required")
    target = _select_target(
        plan, target_name=target_name, coverage_cell_id=coverage_cell_id
    )
    primary = plan.get("target")
    if target is primary:
        source_branch_ids = {
            anchor.get("branch_id")
            for anchor in plan.get("source_anchors") or []
            if isinstance(anchor, Mapping) and anchor.get("branch_id")
        }
        if len(source_branch_ids) == 1 and oracle_branch_id not in source_branch_ids:
            raise ThenAtomizationError(
                "primary target branch conflicts with the compiled source anchor: "
                f"expected {next(iter(source_branch_ids))!r}, got {oracle_branch_id!r}"
            )
    focal = _mapping(target.get("focal_configuration"), "$.target.focal_configuration")
    cell_id = focal.get("coverage_cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        raise ThenAtomizationError("selected target has no coverage_cell_id")
    required = _assignment(
        focal.get("required_factor_values"),
        expected_key="expected_truth_value",
        path="$.target.focal_configuration.required_factor_values",
    )
    contract = {
        "schema_version": CONTRACT_VERSION,
        "branch_eligibility_contract_id": (
            f"{oracle_branch_id}::GIVEN::{content_sha256({'cell': cell_id, 'required': required})[:16]}"
        ),
        "oracle_branch_id": oracle_branch_id,
        "target_locator": {
            "target_name": target.get("target_name"),
            "coverage_cell_id": cell_id,
        },
        "required_assignment": required,
        "object_binding_policy": "exact_target_object",
        "source_compiled_plan_identity": {
            "compiled_plan_id": plan["compiled_plan_id"],
            "compiled_plan_fingerprint": plan["compiled_plan_fingerprint"],
        },
    }
    contract["branch_eligibility_contract_fingerprint"] = content_sha256(contract)
    return validate_branch_eligibility_contract(contract)


def validate_branch_eligibility_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$branch_eligibility_contract")))
    fingerprint = result.pop("branch_eligibility_contract_fingerprint", None)
    if result.get("schema_version") != CONTRACT_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid branch eligibility contract")
    _assignment(
        result.get("required_assignment"),
        expected_key="expected_truth_value",
        path="$.required_assignment",
    )
    if not isinstance(result.get("oracle_branch_id"), str) or not result["oracle_branch_id"]:
        raise ThenAtomizationError("eligibility contract branch is invalid")
    locator = _mapping(result.get("target_locator"), "$.target_locator")
    if not isinstance(locator.get("coverage_cell_id"), str) or not locator["coverage_cell_id"]:
        raise ThenAtomizationError("eligibility contract coverage cell is invalid")
    _mapping(result.get("source_compiled_plan_identity"), "$.source_compiled_plan_identity")
    result["branch_eligibility_contract_fingerprint"] = fingerprint
    return result


def build_branch_eligibility_witness(
    contract: Mapping[str, Any],
    bound_plan: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a trusted Driver fixture and runtime pre-state to a Given contract."""

    contract = validate_branch_eligibility_contract(contract)
    bound = deepcopy(dict(_mapping(bound_plan, "$bound_plan")))
    execution = deepcopy(dict(_mapping(execution, "$execution")))
    bound_fingerprint = bound.get("bound_driver_plan_fingerprint")
    if bound.get("schema_version") != "agentspectesting.bound-driver-plan/v0.1" or (
        not isinstance(bound_fingerprint, str)
        or bound_fingerprint != content_sha256(_bound_plan_payload(bound))
    ):
        raise ThenAtomizationError("invalid bound Driver plan")
    compiled_identity = _mapping(
        bound.get("compiled_plan_identity"), "$.bound_plan.compiled_plan_identity"
    )
    plan_lineage_matches = dict(compiled_identity) == contract[
        "source_compiled_plan_identity"
    ]
    fixture_id = execution.get("fixture_instance_id")
    fixtures = [
        item
        for item in bound.get("fixture_instances") or []
        if isinstance(item, Mapping) and item.get("fixture_instance_id") == fixture_id
    ]
    if len(fixtures) != 1:
        raise ThenAtomizationError("execution fixture must resolve exactly once")
    fixture = deepcopy(dict(fixtures[0]))
    actual_assignment = []
    seen = set()
    malformed = []
    for raw in fixture.get("predicate_witnesses") or []:
        if not isinstance(raw, Mapping):
            malformed.append("non_object_predicate_witness")
            continue
        predicate_id = raw.get("predicate_id")
        truth = raw.get("evaluated_truth_value")
        if not isinstance(predicate_id, str) or not predicate_id or predicate_id in seen:
            malformed.append("invalid_or_duplicate_predicate_id")
            continue
        seen.add(predicate_id)
        actual_assignment.append(
            {
                "predicate_id": predicate_id,
                "evaluated_truth_value": truth if isinstance(truth, bool) else None,
                "evidence": deepcopy(raw.get("evidence")),
            }
        )
        if not isinstance(truth, bool):
            malformed.append(f"non_boolean_truth:{predicate_id}")

    try:
        bound_oracle = evaluate_bound_online_execution(bound, execution)
        runtime_fixture_valid = bound_oracle["fixture_check"]["valid"] is True
        runtime_pre_state_mismatches = deepcopy(
            bound_oracle["fixture_check"].get("mismatches") or {}
        )
        artifact_checks = bound_oracle["artifact_checks"]["checks"]
        identity_keys = (
            "bound_plan_fingerprint_valid",
            "bound_plan_identity_matched",
            "execution_fixture_identity_matched",
            "reachability_fixture_identity_matched",
        )
        runtime_identity_closed = all(artifact_checks.get(key) is True for key in identity_keys)
        milestones = (
            bound_oracle.get("reachability_gate", {}).get("result", {}).get("milestones")
            or {}
        )
        runtime_given_evidence = {
            key: (milestones.get(key) or {}).get("status")
            for key in _REQUIRED_RUNTIME_MILESTONES
        }
        runtime_given_evidence_closed = all(
            status in {"satisfied", "not_applicable"}
            for status in runtime_given_evidence.values()
        )
        source_bound_oracle_fingerprint = bound_oracle["oracle_result_fingerprint"]
    except (BoundOracleError, KeyError, TypeError) as exc:
        runtime_fixture_valid = False
        runtime_pre_state_mismatches = {"bound_oracle": str(exc)}
        runtime_identity_closed = False
        runtime_given_evidence = {key: None for key in _REQUIRED_RUNTIME_MILESTONES}
        runtime_given_evidence_closed = False
        source_bound_oracle_fingerprint = None

    witness = {
        "schema_version": WITNESS_VERSION,
        "branch_eligibility_witness_id": (
            f"{contract['branch_eligibility_contract_id']}::WITNESS::{content_sha256(execution)[:16]}"
        ),
        "oracle_branch_id": contract["oracle_branch_id"],
        "fixture_instance_id": fixture_id,
        "coverage_cell_id": fixture.get("coverage_cell_id"),
        "actual_assignment": sorted(
            actual_assignment, key=lambda item: item["predicate_id"]
        ),
        "exact_target_object_binding": (
            (fixture.get("fixture_oracle_audit") or {}).get(
                "exact_target_object_binding"
            )
            is True
        ),
        "runtime_fixture_valid": runtime_fixture_valid,
        "runtime_pre_state_mismatches": runtime_pre_state_mismatches,
        "runtime_identity_closed": runtime_identity_closed,
        "runtime_given_evidence": runtime_given_evidence,
        "runtime_given_evidence_closed": runtime_given_evidence_closed,
        "plan_lineage_matches": plan_lineage_matches,
        "malformed_witness_reasons": sorted(set(malformed)),
        "source_branch_eligibility_contract_fingerprint": contract[
            "branch_eligibility_contract_fingerprint"
        ],
        "source_bound_driver_plan_identity": {
            "bound_driver_plan_id": bound.get("bound_driver_plan_id"),
            "bound_driver_plan_fingerprint": bound_fingerprint,
        },
        "source_bound_oracle_fingerprint": source_bound_oracle_fingerprint,
        "source_execution_fingerprint": content_sha256(execution),
    }
    witness["branch_eligibility_witness_fingerprint"] = content_sha256(witness)
    return validate_branch_eligibility_witness(witness)


def validate_branch_eligibility_witness(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$branch_eligibility_witness")))
    fingerprint = result.pop("branch_eligibility_witness_fingerprint", None)
    if result.get("schema_version") != WITNESS_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid branch eligibility witness")
    seen = set()
    for item in result.get("actual_assignment") or []:
        item = _mapping(item, "$.actual_assignment[]")
        predicate_id = item.get("predicate_id")
        if not isinstance(predicate_id, str) or not predicate_id or predicate_id in seen:
            raise ThenAtomizationError("actual assignment predicate IDs must be unique")
        seen.add(predicate_id)
        if item.get("evaluated_truth_value") not in {True, False, None}:
            raise ThenAtomizationError("actual assignment truth is invalid")
    result["branch_eligibility_witness_fingerprint"] = fingerprint
    return result


def evaluate_branch_eligibility(
    contract: Mapping[str, Any], witness: Mapping[str, Any]
) -> dict[str, Any]:
    contract = validate_branch_eligibility_contract(contract)
    witness = validate_branch_eligibility_witness(witness)
    if witness.get("oracle_branch_id") != contract["oracle_branch_id"]:
        raise ThenAtomizationError("eligibility branch lineage mismatch")
    if witness.get("source_branch_eligibility_contract_fingerprint") != contract[
        "branch_eligibility_contract_fingerprint"
    ]:
        raise ThenAtomizationError("eligibility contract/witness lineage mismatch")

    expected = {
        item["predicate_id"]: item["expected_truth_value"]
        for item in contract["required_assignment"]
    }
    actual = {
        item["predicate_id"]: item.get("evaluated_truth_value")
        for item in witness.get("actual_assignment") or []
    }
    missing = sorted(set(expected) - set(actual))
    unknown = sorted(key for key in expected if key in actual and actual[key] is None)
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in sorted(expected)
        if key in actual and actual[key] is not None and actual[key] != expected[key]
    }
    lineage_checks = {
        "coverage_cell_matches": witness.get("coverage_cell_id")
        == contract["target_locator"]["coverage_cell_id"],
        "compiled_plan_lineage_matches": witness.get("plan_lineage_matches") is True,
        "runtime_identity_closed": witness.get("runtime_identity_closed") is True,
        "exact_target_object_binding": witness.get("exact_target_object_binding") is True,
        "runtime_fixture_valid": witness.get("runtime_fixture_valid") is True,
        "runtime_given_evidence_closed": witness.get("runtime_given_evidence_closed")
        is True,
        "predicate_witnesses_well_formed": not witness.get(
            "malformed_witness_reasons"
        ),
    }
    if mismatches or lineage_checks["coverage_cell_matches"] is False:
        status = "not_satisfied"
        reason = "given_assignment_or_coverage_cell_mismatch"
    elif missing or unknown or not all(lineage_checks.values()):
        status = "incomplete"
        reason = "given_evidence_or_lineage_incomplete"
    else:
        status = "satisfied"
        reason = "all_required_given_predicates_verified"
    result = {
        "schema_version": GATE_VERSION,
        "branch_eligibility_gate_id": (
            f"{contract['branch_eligibility_contract_id']}::GATE::{witness['source_execution_fingerprint'][:16]}"
        ),
        "oracle_branch_id": contract["oracle_branch_id"],
        "status": status,
        "reason": reason,
        "required_predicate_count": len(expected),
        "matched_predicate_count": sum(
            key in actual and actual[key] is not None and actual[key] == value
            for key, value in expected.items()
        ),
        "missing_predicate_ids": missing,
        "unknown_predicate_ids": unknown,
        "predicate_mismatches": mismatches,
        "lineage_checks": lineage_checks,
        "runtime_pre_state_mismatches": deepcopy(
            witness.get("runtime_pre_state_mismatches") or {}
        ),
        "runtime_given_evidence": deepcopy(
            witness.get("runtime_given_evidence") or {}
        ),
        "source_branch_eligibility_contract_fingerprint": contract[
            "branch_eligibility_contract_fingerprint"
        ],
        "source_branch_eligibility_witness_fingerprint": witness[
            "branch_eligibility_witness_fingerprint"
        ],
        "source_execution_fingerprint": witness["source_execution_fingerprint"],
    }
    result["branch_eligibility_gate_fingerprint"] = content_sha256(result)
    return validate_branch_eligibility_gate(result)


def validate_branch_eligibility_gate(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$branch_eligibility_gate")))
    fingerprint = result.pop("branch_eligibility_gate_fingerprint", None)
    if result.get("schema_version") != GATE_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid branch eligibility gate")
    if result.get("status") not in GATE_STATUSES:
        raise ThenAtomizationError("invalid branch eligibility status")
    if not isinstance(result.get("source_execution_fingerprint"), str):
        raise ThenAtomizationError("eligibility gate execution lineage is missing")
    missing = result.get("missing_predicate_ids") or []
    unknown = result.get("unknown_predicate_ids") or []
    mismatches = _mapping(result.get("predicate_mismatches") or {}, "$.predicate_mismatches")
    lineage = _mapping(result.get("lineage_checks") or {}, "$.lineage_checks")
    if any(not isinstance(value, bool) for value in lineage.values()):
        raise ThenAtomizationError("eligibility lineage checks must be Boolean")
    coverage_mismatch = lineage.get("coverage_cell_matches") is False
    if mismatches or coverage_mismatch:
        expected_status = "not_satisfied"
    elif missing or unknown or not all(lineage.values()):
        expected_status = "incomplete"
    else:
        expected_status = "satisfied"
    if result.get("status") != expected_status:
        raise ThenAtomizationError("eligibility gate status contradicts its evidence")
    required_count = result.get("required_predicate_count")
    matched_count = result.get("matched_predicate_count")
    if (
        not isinstance(required_count, int)
        or isinstance(required_count, bool)
        or required_count < 0
        or not isinstance(matched_count, int)
        or isinstance(matched_count, bool)
        or not 0 <= matched_count <= required_count
    ):
        raise ThenAtomizationError("eligibility gate predicate counts are invalid")
    result["branch_eligibility_gate_fingerprint"] = fingerprint
    return result
