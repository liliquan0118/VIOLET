"""Aggregate validated Semantic Judge responses into Oracle requirement verdicts."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256
from ..compiler.semantic_judge_contract_v1 import (
    validate_semantic_judge_contract_set,
)
from ..compiler.then_atomization import ThenAtomizationError
from .semantic_judge_v1 import (
    TASK_CLAIM_ATOMIZATION,
    TASK_CLAIM_SUPPORT,
    TASK_SUBJECTIVITY,
    attach_claim_atomization_response,
    build_claim_support_packets,
    build_initial_semantic_judge_packets,
    validate_claim_atomization_response,
    validate_claim_support_response,
    validate_subjectivity_response,
)


EVALUATION_SET_VERSION = "agentspectesting.semantic-runtime-evaluation-set/v0.1"
EVALUATION_VERSION = "agentspectesting.semantic-runtime-evaluation/v0.1"
VERDICTS = frozenset({"pass", "fail", "unavailable"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _response_index(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        result = deepcopy(dict(value))
    elif isinstance(value, list):
        result = {}
        for raw in value:
            record = _mapping(raw, f"{path}[]")
            packet_id = record.get("packet_id")
            if not isinstance(packet_id, str) or not packet_id or packet_id in result:
                raise ThenAtomizationError(f"{path} packet IDs must be unique")
            result[packet_id] = deepcopy(record.get("response"))
    else:
        raise ThenAtomizationError(f"{path} must be an object or array")
    if any(not isinstance(key, str) or not key for key in result):
        raise ThenAtomizationError(f"{path} packet IDs must be non-empty strings")
    return result


def _contract_for_branch(
    contracts: Mapping[str, Any], branch_id: str
) -> Mapping[str, Any]:
    matches = [
        item
        for item in contracts.get("contracts") or []
        if item.get("branch_id") == branch_id
    ]
    if len(matches) != 1:
        raise ThenAtomizationError(
            f"branch must resolve to one semantic judge contract: {branch_id!r}"
        )
    if matches[0].get("contract_status") != "ready":
        raise ThenAtomizationError("semantic judge contract is not ready")
    return matches[0]


def _stage_record(
    packet: Mapping[str, Any], status: str, response: Any, verdict: str, error: str | None
) -> dict[str, Any]:
    return {
        "packet_id": packet["packet_id"],
        "packet_fingerprint": packet["packet_fingerprint"],
        "task_name": packet["task_name"],
        "target_event": deepcopy(packet["target_event"]),
        "status": status,
        "verdict": verdict,
        "response": deepcopy(response),
        "error": error,
    }


def _aggregate(verdicts: list[str]) -> tuple[str, str]:
    if "fail" in verdicts:
        return "fail", "at_least_one_semantic_judgment_failed"
    if "unavailable" in verdicts:
        return "unavailable", "at_least_one_semantic_judgment_unavailable"
    return "pass", "all_semantic_judgments_passed"


def evaluate_semantic_judge_branch(
    semantic_judge_contract_set: Mapping[str, Any],
    execution: Mapping[str, Any],
    branch_id: str,
    agent_spec: Mapping[str, Any],
    *,
    initial_responses: Mapping[str, Any] | list[Mapping[str, Any]] | None,
    claim_support_responses: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate response lineage and aggregate one semantic requirement.

    Missing, invalid, and ``insufficient`` model results are data outcomes and
    become ``unavailable``.  Cross-run or unexpected packet IDs are structural
    errors and are rejected.
    """

    contracts = validate_semantic_judge_contract_set(semantic_judge_contract_set)
    execution = deepcopy(dict(_mapping(execution, "$execution")))
    contract = _contract_for_branch(contracts, branch_id)
    execution_fingerprint = content_sha256(execution)
    initial_packets = build_initial_semantic_judge_packets(
        contracts, execution, branch_id
    )
    initial_index = {packet["packet_id"]: packet for packet in initial_packets["packets"]}
    supplied_initial = _response_index(initial_responses, "$initial_responses")
    unexpected = sorted(set(supplied_initial) - set(initial_index))
    if unexpected:
        raise ThenAtomizationError(
            f"unexpected initial semantic response packet IDs: {unexpected}"
        )
    judge_kind = contract["program"]["judge_kind"]
    stage_records = []
    leaf_verdicts = []

    if judge_kind == "subjective_expression":
        if claim_support_responses:
            raise ThenAtomizationError("subjectivity evaluation does not accept claim responses")
        for packet_id, packet in initial_index.items():
            raw = supplied_initial.get(packet_id)
            if raw is None:
                stage_records.append(
                    _stage_record(packet, "missing", None, "unavailable", "response_missing")
                )
                leaf_verdicts.append("unavailable")
                continue
            try:
                checked = validate_subjectivity_response(packet, _mapping(raw, "$.response"))
                model_verdict = checked["verdict"]
                verdict = (
                    "fail"
                    if model_verdict == "violation"
                    else "pass"
                    if model_verdict == "no_violation"
                    else "unavailable"
                )
                stage_records.append(_stage_record(packet, "valid", checked, verdict, None))
            except (ThenAtomizationError, TypeError, KeyError) as exc:
                verdict = "unavailable"
                stage_records.append(
                    _stage_record(packet, "invalid", raw, verdict, str(exc))
                )
            leaf_verdicts.append(verdict)
    elif judge_kind == "claim_source_support":
        claim_results = []
        atomization_unavailable = False
        for packet_id, packet in initial_index.items():
            raw = supplied_initial.get(packet_id)
            if raw is None:
                stage_records.append(
                    _stage_record(packet, "missing", None, "unavailable", "response_missing")
                )
                atomization_unavailable = True
                continue
            try:
                checked = validate_claim_atomization_response(
                    packet, _mapping(raw, "$.response")
                )
                result = attach_claim_atomization_response(packet, checked)
                claim_results.append(result)
                stage_records.append(_stage_record(packet, "valid", checked, "pass", None))
            except (ThenAtomizationError, TypeError, KeyError) as exc:
                stage_records.append(
                    _stage_record(packet, "invalid", raw, "unavailable", str(exc))
                )
                atomization_unavailable = True

        support_packets = build_claim_support_packets(
            contracts, execution, branch_id, claim_results, agent_spec
        )
        support_index = {packet["packet_id"]: packet for packet in support_packets["packets"]}
        supplied_support = _response_index(
            claim_support_responses, "$claim_support_responses"
        )
        unexpected = sorted(set(supplied_support) - set(support_index))
        if unexpected:
            raise ThenAtomizationError(
                f"unexpected claim support response packet IDs: {unexpected}"
            )
        if atomization_unavailable:
            leaf_verdicts.append("unavailable")
        for packet_id, packet in support_index.items():
            raw = supplied_support.get(packet_id)
            if raw is None:
                verdict = "unavailable"
                stage_records.append(
                    _stage_record(packet, "missing", None, verdict, "response_missing")
                )
            else:
                try:
                    checked = validate_claim_support_response(
                        packet, _mapping(raw, "$.response")
                    )
                    model_verdict = checked["verdict"]
                    verdict = (
                        "pass"
                        if model_verdict == "supported"
                        else "fail"
                        if model_verdict == "unsupported"
                        else "unavailable"
                    )
                    stage_records.append(
                        _stage_record(packet, "valid", checked, verdict, None)
                    )
                except (ThenAtomizationError, TypeError, KeyError) as exc:
                    verdict = "unavailable"
                    stage_records.append(
                        _stage_record(packet, "invalid", raw, verdict, str(exc))
                    )
            leaf_verdicts.append(verdict)
        # A valid empty claim inventory is a vacuous pass.  The unavailable
        # flag above prevents an invalid/missing inventory from taking this path.
        if not support_index and not atomization_unavailable:
            leaf_verdicts.append("pass")
    else:
        raise ThenAtomizationError("unsupported semantic judge kind")

    verdict, reason = _aggregate(leaf_verdicts)
    evaluation = {
        "judge_kind": judge_kind,
        "verdict": verdict,
        "reason": reason,
        "stage_records": stage_records,
        "stage_verdict_counts": {
            key: Counter(record["verdict"] for record in stage_records)[key]
            for key in sorted(VERDICTS)
        },
    }
    record = {
        "schema_version": EVALUATION_VERSION,
        "semantic_runtime_evaluation_id": (
            f"{contract['semantic_judge_contract_id']}::SRE01::{execution_fingerprint[:16]}"
        ),
        "semantic_judge_contract_id": contract["semantic_judge_contract_id"],
        "evaluator_contract_id": contract["evaluator_contract_id"],
        "binding_id": contract["binding_id"],
        "requirement_id": contract["requirement_id"],
        "branch_id": branch_id,
        "verdict": verdict,
        "evaluation": evaluation,
        "source_execution_fingerprint": execution_fingerprint,
        "source_semantic_judge_contract_fingerprint": contract[
            "semantic_judge_contract_fingerprint"
        ],
        "source_initial_packet_set_fingerprint": initial_packets[
            "packet_set_fingerprint"
        ],
        "source_claim_support_packet_set_fingerprint": (
            support_packets["packet_set_fingerprint"]
            if judge_kind == "claim_source_support"
            else None
        ),
    }
    record["semantic_runtime_evaluation_fingerprint"] = content_sha256(record)
    result = {
        "schema_version": EVALUATION_SET_VERSION,
        "branch_id": branch_id,
        "source_execution_fingerprint": execution_fingerprint,
        "source_semantic_judge_contract_set_fingerprint": contracts[
            "semantic_judge_contract_set_fingerprint"
        ],
        "source_evaluator_contract_set_fingerprint": contracts[
            "source_evaluator_contract_set_fingerprint"
        ],
        "source_binding_set_fingerprint": contracts[
            "source_binding_set_fingerprint"
        ],
        "requirement_results": [record],
        "semantic_branch_verdict": verdict,
        "summary": {
            "requirement_count": 1,
            "verdict_counts": {
                key: Counter([verdict])[key] for key in sorted(VERDICTS)
            },
        },
    }
    result["semantic_runtime_evaluation_set_fingerprint"] = content_sha256(result)
    validate_semantic_runtime_evaluation_set(result)
    return result


def validate_semantic_runtime_evaluation_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$semantic_runtime_evaluation_set")))
    fingerprint = result.pop("semantic_runtime_evaluation_set_fingerprint", None)
    if result.get("schema_version") != EVALUATION_SET_VERSION or fingerprint != content_sha256(
        result
    ):
        raise ThenAtomizationError("invalid semantic runtime evaluation set")
    counts: Counter[str] = Counter()
    identities = []
    for raw in result.get("requirement_results") or []:
        record = deepcopy(dict(_mapping(raw, "$.requirement_results[]")))
        record_fingerprint = record.pop("semantic_runtime_evaluation_fingerprint", None)
        if record.get("schema_version") != EVALUATION_VERSION or record_fingerprint != content_sha256(
            record
        ):
            raise ThenAtomizationError("invalid semantic runtime evaluation")
        if record.get("branch_id") != result.get("branch_id"):
            raise ThenAtomizationError("semantic runtime branch mismatch")
        if record.get("source_execution_fingerprint") != result.get(
            "source_execution_fingerprint"
        ):
            raise ThenAtomizationError("semantic runtime execution mismatch")
        if record.get("verdict") not in VERDICTS:
            raise ThenAtomizationError("invalid semantic runtime verdict")
        counts[record["verdict"]] += 1
        identities.append(record.get("semantic_runtime_evaluation_id"))
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("semantic runtime evaluation IDs must be unique")
    expected_counts = {key: counts[key] for key in sorted(VERDICTS)}
    summary = _mapping(result.get("summary"), "$.summary")
    if summary.get("requirement_count") != len(identities) or summary.get(
        "verdict_counts"
    ) != expected_counts:
        raise ThenAtomizationError("semantic runtime evaluation summary mismatch")
    expected_branch = (
        "fail" if counts["fail"] else "unavailable" if counts["unavailable"] else "pass"
    )
    if result.get("semantic_branch_verdict") != expected_branch:
        raise ThenAtomizationError("semantic runtime branch aggregation mismatch")
    for field in (
        "source_semantic_judge_contract_set_fingerprint",
        "source_evaluator_contract_set_fingerprint",
        "source_binding_set_fingerprint",
    ):
        if not isinstance(result.get(field), str) or not result[field]:
            raise ThenAtomizationError(f"semantic runtime lineage is missing {field}")
    result["semantic_runtime_evaluation_set_fingerprint"] = fingerprint
    return result
