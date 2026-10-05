"""Materialize and score a closed semantic-judge calibration batch."""

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
    PACKET_VERSION,
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


SUITE_VERSION = "agentspectesting.semantic-judge-calibration-suite/v0.1"
BUNDLE_VERSION = "agentspectesting.semantic-judge-calibration-bundle/v0.1"
SCORE_VERSION = "agentspectesting.semantic-judge-calibration-score/v0.1"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _expected(value: Mapping[str, Any], task: str) -> dict[str, Any]:
    expected = deepcopy(dict(_mapping(value, "$.case.expected")))
    if task == TASK_CLAIM_ATOMIZATION:
        if set(expected) != {"claims"} or not isinstance(expected["claims"], list):
            raise ThenAtomizationError("atomization calibration requires claims")
        if any(not isinstance(item, str) or not item for item in expected["claims"]):
            raise ThenAtomizationError("expected claims must be non-empty strings")
    elif task == TASK_CLAIM_SUPPORT:
        if set(expected) != {"verdict"} or expected["verdict"] not in {
            "supported",
            "unsupported",
        }:
            raise ThenAtomizationError("support calibration verdict is invalid")
    elif task == TASK_SUBJECTIVITY:
        if set(expected) != {"verdict"} or expected["verdict"] not in {
            "violation",
            "no_violation",
        }:
            raise ThenAtomizationError("subjectivity calibration verdict is invalid")
    else:
        raise ThenAtomizationError("unsupported calibration task")
    return expected


def build_semantic_judge_calibration_bundle(
    suite: Mapping[str, Any],
    semantic_judge_contract_set: Mapping[str, Any],
    agent_spec: Mapping[str, Any],
) -> dict[str, Any]:
    suite = deepcopy(dict(_mapping(suite, "$calibration_suite")))
    contracts = validate_semantic_judge_contract_set(semantic_judge_contract_set)
    if suite.get("schema_version") != SUITE_VERSION:
        raise ThenAtomizationError("unsupported semantic judge calibration suite")
    if not isinstance(suite.get("suite_id"), str) or not suite["suite_id"]:
        raise ThenAtomizationError("calibration suite_id must be non-empty")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ThenAtomizationError("calibration cases must be non-empty")
    jobs = []
    case_ids = []
    counts: Counter[str] = Counter()
    for raw in cases:
        case = _mapping(raw, "$.cases[]")
        case_id = case.get("case_id")
        task = case.get("task_name")
        branch_id = case.get("branch_id")
        execution = _mapping(case.get("execution"), "$.case.execution")
        if not isinstance(case_id, str) or not case_id:
            raise ThenAtomizationError("calibration case_id must be non-empty")
        case_ids.append(case_id)
        expected = _expected(_mapping(case.get("expected"), "$.case.expected"), task)
        initial = build_initial_semantic_judge_packets(
            contracts, execution, branch_id
        )
        if len(initial["packets"]) != 1:
            raise ThenAtomizationError("each calibration case must target one message")
        if task == TASK_CLAIM_SUPPORT:
            claim_text = case.get("claim_text")
            initial_packet = initial["packets"][0]
            claim_result = attach_claim_atomization_response(
                initial_packet, {"claims": [{"text": claim_text}]}
            )
            support = build_claim_support_packets(
                contracts, execution, branch_id, [claim_result], agent_spec
            )
            if len(support["packets"]) != 1:
                raise ThenAtomizationError("support calibration must build one packet")
            packet = support["packets"][0]
        else:
            packet = initial["packets"][0]
            if packet.get("task_name") != task:
                raise ThenAtomizationError("calibration task and branch contract disagree")
        jobs.append(
            {
                "case_id": case_id,
                "task_name": task,
                "packet": deepcopy(packet),
                "expected": expected,
            }
        )
        counts[task] += 1
    if len(case_ids) != len(set(case_ids)):
        raise ThenAtomizationError("calibration case IDs must be unique")
    result = {
        "schema_version": BUNDLE_VERSION,
        "suite_id": suite["suite_id"],
        "source_suite_fingerprint": content_sha256(suite),
        "source_semantic_judge_contract_set_fingerprint": contracts[
            "semantic_judge_contract_set_fingerprint"
        ],
        "agent_spec_fingerprint": content_sha256(agent_spec),
        "jobs": jobs,
        "summary": {
            "job_count": len(jobs),
            "expected_model_calls": len(jobs),
            "task_counts": {task: counts[task] for task in sorted(counts)},
        },
    }
    result["calibration_bundle_fingerprint"] = content_sha256(result)
    validate_semantic_judge_calibration_bundle(result)
    return result


def validate_semantic_judge_calibration_bundle(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$calibration_bundle")))
    fingerprint = result.pop("calibration_bundle_fingerprint", None)
    if result.get("schema_version") != BUNDLE_VERSION or fingerprint != content_sha256(
        result
    ):
        raise ThenAtomizationError("invalid semantic judge calibration bundle")
    identities = []
    counts: Counter[str] = Counter()
    for raw in result.get("jobs") or []:
        job = _mapping(raw, "$.jobs[]")
        if set(job) != {"case_id", "task_name", "packet", "expected"}:
            raise ThenAtomizationError("calibration job fields are invalid")
        packet = _mapping(job.get("packet"), "$.job.packet")
        packet_copy = deepcopy(dict(packet))
        packet_fingerprint = packet_copy.pop("packet_fingerprint", None)
        if packet.get("schema_version") != PACKET_VERSION or packet_fingerprint != content_sha256(
            packet_copy
        ):
            raise ThenAtomizationError("invalid packet in calibration bundle")
        if packet.get("task_name") != job.get("task_name"):
            raise ThenAtomizationError("calibration job task mismatch")
        _expected(_mapping(job.get("expected"), "$.job.expected"), job["task_name"])
        identities.append(job.get("case_id"))
        counts[job["task_name"]] += 1
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("calibration job IDs must be unique")
    summary = _mapping(result.get("summary"), "$.summary")
    if summary.get("job_count") != len(identities) or summary.get(
        "expected_model_calls"
    ) != len(identities):
        raise ThenAtomizationError("calibration job count mismatch")
    if summary.get("task_counts") != {
        task: counts[task] for task in sorted(counts)
    }:
        raise ThenAtomizationError("calibration task counts mismatch")
    result["calibration_bundle_fingerprint"] = fingerprint
    return result


def validate_calibration_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    task = packet.get("task_name")
    if task == TASK_CLAIM_ATOMIZATION:
        return validate_claim_atomization_response(packet, response)
    if task == TASK_CLAIM_SUPPORT:
        return validate_claim_support_response(packet, response)
    if task == TASK_SUBJECTIVITY:
        return validate_subjectivity_response(packet, response)
    raise ThenAtomizationError("unsupported semantic calibration response task")


def score_semantic_judge_calibration(
    bundle: Mapping[str, Any], responses: Mapping[str, Any]
) -> dict[str, Any]:
    bundle = validate_semantic_judge_calibration_bundle(bundle)
    responses = _mapping(responses, "$calibration_responses")
    records = responses.get("responses")
    if not isinstance(records, list):
        raise ThenAtomizationError("calibration responses must be an array")
    response_index = {}
    for raw in records:
        record = _mapping(raw, "$.responses[]")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in response_index:
            raise ThenAtomizationError("calibration response case IDs must be unique")
        response_index[case_id] = record.get("response")
    results = []
    task_totals: Counter[str] = Counter()
    task_correct: Counter[str] = Counter()
    for job in bundle["jobs"]:
        case_id = job["case_id"]
        task = job["task_name"]
        task_totals[task] += 1
        raw_response = response_index.get(case_id)
        status = "missing"
        actual = None
        correct = False
        error = None
        if raw_response is not None:
            try:
                checked = validate_calibration_response(job["packet"], raw_response)
                status = "valid"
                if task == TASK_CLAIM_ATOMIZATION:
                    actual = {"claims": [item["text"] for item in checked["claims"]]}
                else:
                    actual = {"verdict": checked["verdict"]}
                correct = actual == job["expected"]
            except (ThenAtomizationError, TypeError, KeyError) as exc:
                status = "invalid"
                error = str(exc)
        if correct:
            task_correct[task] += 1
        results.append(
            {
                "case_id": case_id,
                "task_name": task,
                "status": status,
                "expected": deepcopy(job["expected"]),
                "actual": actual,
                "correct": correct,
                "error": error,
            }
        )
    total = len(results)
    correct_count = sum(item["correct"] for item in results)
    result = {
        "schema_version": SCORE_VERSION,
        "source_calibration_bundle_fingerprint": bundle[
            "calibration_bundle_fingerprint"
        ],
        "results": results,
        "summary": {
            "case_count": total,
            "correct_count": correct_count,
            "accuracy": correct_count / total if total else None,
            "by_task": {
                task: {
                    "case_count": task_totals[task],
                    "correct_count": task_correct[task],
                    "accuracy": task_correct[task] / task_totals[task],
                }
                for task in sorted(task_totals)
            },
        },
    }
    result["score_fingerprint"] = content_sha256(result)
    return result
