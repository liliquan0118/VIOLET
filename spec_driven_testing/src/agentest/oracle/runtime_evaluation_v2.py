"""Runtime Oracle v0.2: Given gate plus deterministic/semantic merge."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256
from ..compiler.then_atomization import ThenAtomizationError
from .branch_eligibility_v1 import validate_branch_eligibility_gate
from .runtime_evaluation_v1 import (
    evaluate_runtime_oracle_branch,
    validate_runtime_oracle_evaluation_set,
)
from .semantic_runtime_evaluation_v1 import (
    validate_semantic_runtime_evaluation_set,
)


RUNTIME_EVALUATION_SET_VERSION = "agentspectesting.runtime-oracle-evaluation-set/v0.2"
EFFECTIVE_RESULT_VERSION = "agentspectesting.effective-oracle-requirement-result/v0.2"
BRANCH_VERDICTS = frozenset({"pass", "fail", "incomplete", "invalid_test_case"})
REQUIREMENT_VERDICTS = frozenset({"pass", "fail", "unavailable"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _semantic_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("evaluator_contract_id")),
        str(record.get("binding_id")),
        str(record.get("requirement_id")),
    )


def _effective_record(
    deterministic: Mapping[str, Any], semantic: Mapping[str, Any] | None
) -> dict[str, Any]:
    deterministic_verdict = deterministic["verdict"]
    if deterministic_verdict in {"pass", "fail"}:
        verdict = deterministic_verdict
        source = "deterministic_evaluator"
        reason = "deterministic_result_is_conclusive"
        semantic_fingerprint = None
    elif semantic is not None:
        verdict = semantic["verdict"]
        source = "semantic_judge"
        reason = "deferred_requirement_resolved_by_semantic_judge"
        semantic_fingerprint = semantic["semantic_runtime_evaluation_fingerprint"]
    else:
        verdict = "unavailable"
        source = "unresolved"
        reason = "no_semantic_result_for_deferred_or_unavailable_requirement"
        semantic_fingerprint = None
    result = {
        "schema_version": EFFECTIVE_RESULT_VERSION,
        "effective_result_id": f"{deterministic['runtime_evaluation_id']}::V02",
        "evaluator_contract_id": deterministic["evaluator_contract_id"],
        "binding_id": deterministic["binding_id"],
        "requirement_id": deterministic["requirement_id"],
        "branch_id": deterministic["branch_id"],
        "verdict": verdict,
        "resolution_source": source,
        "resolution_reason": reason,
        "source_deterministic_evaluation_fingerprint": deterministic[
            "runtime_evaluation_fingerprint"
        ],
        "source_semantic_evaluation_fingerprint": semantic_fingerprint,
    }
    result["effective_result_fingerprint"] = content_sha256(result)
    return result


def evaluate_runtime_oracle_branch_v2(
    evaluator_contract_set: Mapping[str, Any],
    runtime_observation_binding_set: Mapping[str, Any],
    execution: Mapping[str, Any],
    branch_id: str,
    eligibility_gate: Mapping[str, Any],
    *,
    semantic_evaluation_set: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate the execution, then merge semantic results into v0.1 results."""

    execution = deepcopy(dict(_mapping(execution, "$execution")))
    execution_fingerprint = content_sha256(execution)
    gate = validate_branch_eligibility_gate(eligibility_gate)
    if gate.get("oracle_branch_id") != branch_id:
        raise ThenAtomizationError("eligibility gate branch mismatch")
    if gate.get("source_execution_fingerprint") != execution_fingerprint:
        raise ThenAtomizationError("eligibility gate execution mismatch")

    semantic = None
    if semantic_evaluation_set is not None:
        semantic = validate_semantic_runtime_evaluation_set(semantic_evaluation_set)
        if semantic.get("branch_id") != branch_id:
            raise ThenAtomizationError("semantic evaluation branch mismatch")
        if semantic.get("source_execution_fingerprint") != execution_fingerprint:
            raise ThenAtomizationError("semantic evaluation execution mismatch")

    deterministic = None
    effective = []
    counts: Counter[str] = Counter()
    gate_status = gate["status"]
    if gate_status == "satisfied":
        deterministic = validate_runtime_oracle_evaluation_set(
            evaluate_runtime_oracle_branch(
                evaluator_contract_set,
                runtime_observation_binding_set,
                execution,
                branch_id,
            )
        )
        if semantic is not None and (
            semantic.get("source_evaluator_contract_set_fingerprint")
            != deterministic.get("source_evaluator_contract_set_fingerprint")
            or semantic.get("source_binding_set_fingerprint")
            != deterministic.get("source_binding_set_fingerprint")
        ):
            raise ThenAtomizationError(
                "semantic evaluation uses different evaluator or binding contracts"
            )
        semantic_index = {}
        for record in (semantic or {}).get("requirement_results") or []:
            key = _semantic_key(record)
            if key in semantic_index:
                raise ThenAtomizationError("semantic requirement results are not unique")
            semantic_index[key] = record
        consumed = set()
        for record in deterministic["requirement_results"]:
            key = _semantic_key(record)
            semantic_record = semantic_index.get(key)
            if semantic_record is not None and record["verdict"] != "unavailable":
                raise ThenAtomizationError(
                    "semantic result cannot replace a conclusive deterministic result"
                )
            merged = _effective_record(record, semantic_record)
            effective.append(merged)
            counts[merged["verdict"]] += 1
            if semantic_record is not None:
                consumed.add(key)
        unexpected = sorted(set(semantic_index) - consumed)
        if unexpected:
            raise ThenAtomizationError(
                f"semantic results do not match deferred runtime requirements: {unexpected}"
            )
        if counts["fail"]:
            branch_verdict = "fail"
            reason = "at_least_one_effective_requirement_failed"
        elif counts["unavailable"]:
            branch_verdict = "incomplete"
            reason = "at_least_one_requirement_remains_unavailable"
        else:
            branch_verdict = "pass"
            reason = "all_effective_requirements_passed"
    elif gate_status == "not_satisfied":
        if semantic is not None:
            raise ThenAtomizationError(
                "semantic results are not consumable when the Given gate is not satisfied"
            )
        branch_verdict = "invalid_test_case"
        reason = "execution_does_not_belong_to_requested_given_branch"
    else:
        if semantic is not None:
            raise ThenAtomizationError(
                "semantic results are not consumable when Given validity is incomplete"
            )
        branch_verdict = "incomplete"
        reason = "execution_given_validity_could_not_be_established"

    result = {
        "schema_version": RUNTIME_EVALUATION_SET_VERSION,
        "runtime_evaluation_set_id": (
            f"{branch_id}::EXEC::{execution_fingerprint[:16]}::V02"
        ),
        "branch_id": branch_id,
        "execution_identity": {
            "candidate_id": execution.get("candidate_id"),
            "fixture_instance_id": execution.get("fixture_instance_id"),
            "variant_id": execution.get("variant_id"),
            "profile_id": execution.get("profile_id"),
            "execution_schema_version": execution.get("execution_schema_version")
            or execution.get("schema_version"),
            "execution_fingerprint": execution_fingerprint,
        },
        "eligibility_gate": deepcopy(gate),
        "deterministic_evaluation": deepcopy(deterministic),
        "semantic_evaluation": deepcopy(semantic),
        "effective_requirement_results": effective,
        "branch_verdict": branch_verdict,
        "aggregation": {
            "policy": "given_gate_then_fail_then_incomplete_then_pass/v0.2",
            "reason": reason,
            "requirement_verdict_counts": {
                verdict: counts[verdict] for verdict in sorted(REQUIREMENT_VERDICTS)
            },
        },
    }
    result["runtime_evaluation_set_fingerprint"] = content_sha256(result)
    return validate_runtime_oracle_evaluation_set_v2(result)


def validate_runtime_oracle_evaluation_set_v2(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$runtime_evaluation_set_v2")))
    fingerprint = result.pop("runtime_evaluation_set_fingerprint", None)
    if result.get("schema_version") != RUNTIME_EVALUATION_SET_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid runtime Oracle v0.2 evaluation set")
    if result.get("branch_verdict") not in BRANCH_VERDICTS:
        raise ThenAtomizationError("invalid runtime Oracle v0.2 branch verdict")
    gate = validate_branch_eligibility_gate(
        _mapping(result.get("eligibility_gate"), "$.eligibility_gate")
    )
    if gate.get("oracle_branch_id") != result.get("branch_id"):
        raise ThenAtomizationError("runtime Oracle v0.2 gate branch mismatch")
    identity = _mapping(result.get("execution_identity"), "$.execution_identity")
    if gate.get("source_execution_fingerprint") != identity.get("execution_fingerprint"):
        raise ThenAtomizationError("runtime Oracle v0.2 execution lineage mismatch")

    effective = result.get("effective_requirement_results")
    if not isinstance(effective, list):
        raise ThenAtomizationError("effective requirement results must be an array")
    counts: Counter[str] = Counter()
    ids = []
    for raw in effective:
        record = deepcopy(dict(_mapping(raw, "$.effective_requirement_results[]")))
        record_fingerprint = record.pop("effective_result_fingerprint", None)
        if record.get("schema_version") != EFFECTIVE_RESULT_VERSION or record_fingerprint != content_sha256(record):
            raise ThenAtomizationError("invalid effective Oracle requirement result")
        if record.get("branch_id") != result.get("branch_id"):
            raise ThenAtomizationError("effective Oracle result branch mismatch")
        if record.get("verdict") not in REQUIREMENT_VERDICTS:
            raise ThenAtomizationError("invalid effective Oracle requirement verdict")
        counts[record["verdict"]] += 1
        ids.append(record.get("effective_result_id"))
    if len(ids) != len(set(ids)):
        raise ThenAtomizationError("effective Oracle result IDs must be unique")
    expected_counts = {
        verdict: counts[verdict] for verdict in sorted(REQUIREMENT_VERDICTS)
    }
    aggregation = _mapping(result.get("aggregation"), "$.aggregation")
    if aggregation.get("requirement_verdict_counts") != expected_counts:
        raise ThenAtomizationError("runtime Oracle v0.2 verdict counts mismatch")

    if gate["status"] == "not_satisfied":
        expected_branch = "invalid_test_case"
    elif gate["status"] == "incomplete":
        expected_branch = "incomplete"
    else:
        expected_branch = (
            "fail" if counts["fail"] else "incomplete" if counts["unavailable"] else "pass"
        )
    if result.get("branch_verdict") != expected_branch:
        raise ThenAtomizationError("runtime Oracle v0.2 branch aggregation mismatch")
    if gate["status"] != "satisfied" and (
        result.get("deterministic_evaluation") is not None
        or result.get("semantic_evaluation") is not None
        or effective
    ):
        raise ThenAtomizationError("blocked Given gate must not contain Oracle evaluations")
    if result.get("deterministic_evaluation") is not None:
        deterministic = validate_runtime_oracle_evaluation_set(
            result["deterministic_evaluation"]
        )
        if deterministic.get("branch_id") != result.get("branch_id"):
            raise ThenAtomizationError("deterministic evaluation branch mismatch")
    if result.get("semantic_evaluation") is not None:
        semantic = validate_semantic_runtime_evaluation_set(
            result["semantic_evaluation"]
        )
        if semantic.get("branch_id") != result.get("branch_id"):
            raise ThenAtomizationError("semantic evaluation branch mismatch")
    result["runtime_evaluation_set_fingerprint"] = fingerprint
    return result
