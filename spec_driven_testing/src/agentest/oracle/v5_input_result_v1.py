"""Offline v5 input-result adapter. A local check is never an overall test verdict."""

from copy import deepcopy
import json

from ..compiler.artifacts import content_sha256
from ..compiler.runtime_observation_binding_v1 import normalize_tau_messages
from ..compiler.then_atomization import ThenAtomizationError
from ..compiler.v5_input_conformance_v1 import validate_input_checks, evaluate_input_check
from ..compiler.v5_step3_intake_v1 import index, seal


SCHEMA = "agentspectesting.v5-partial-test-result/v0.1"
HANDOFF_SCHEMA = "agentspectesting.v5-input-result-handoff/v0.1"


def build_handoff(checks, *, sources):
    validate_input_checks(checks, *sources)
    branches = []
    for source in sources[2]["branches"]:
        bid = source["branch_id"]
        selected = [r for r in checks["checks"] if r["branch_id"] == bid]
        branches.append({"branch_id": bid,
                         "ready_input_check_ids": [r["assertion_id"] for r in selected if r["input_check_ready"]],
                         "unavailable_input_check_ids": [r["assertion_id"] for r in selected if not r["input_check_ready"]],
                         "input_conformance_status": "available" if any(r["input_check_ready"] for r in selected) else "not_prepared",
                         "gwt_coverage_status": "not_implemented", "task_completion_status": "not_implemented",
                         "full_test_ready": False})
    return seal({"schema_version": HANDOFF_SCHEMA, "result_schema_version": SCHEMA,
                 "input_check_set_fingerprint": checks["input_check_set_fingerprint"],
                 "execution_input": "saved_tau_messages_no_tool_dispatch",
                 "branches": branches, "legacy_branch_aggregation_allowed": False,
                 "summary": {"branch_count": len(branches), "branches_with_input_checks": sum(bool(b["ready_input_check_ids"]) for b in branches),
                             "full_tests_ready": 0, "external_llm_calls": 0, "target_agent_calls": 0}}, "handoff_fingerprint")


def evaluate_execution_input(checks, execution, branch_id, *, sources):
    """Consume saved messages with the existing normalizer; do not replay actions.

    Selected branch is workflow routing, not evidence of Given/When satisfaction.
    Execution fingerprints identify supplied bytes/content, not authenticity/completeness.
    """
    handoff = build_handoff(checks, sources=sources)
    branch = index(handoff["branches"], "branch_id").get(branch_id)
    if branch is None:
        raise ValueError("branch not in source handoff")
    if not isinstance(execution, dict) or not isinstance(execution.get("messages"), list):
        raise ValueError("execution must contain a messages array")
    for key in ("branch_id", "oracle_branch_id"):
        if execution.get(key) is not None and execution[key] != branch_id:
            raise ValueError("execution branch identity conflicts with selected source")
    try:
        encoded = json.dumps(execution, ensure_ascii=False, allow_nan=False)
        if json.loads(encoded) != execution:
            raise ValueError("execution changes under JSON round trip")
    except (ValueError, TypeError) as exc:
        raise ValueError("execution must be JSON data") from exc
    execution_fingerprint = content_sha256(execution)
    normalization_error = None
    events = []
    try:
        for message in execution["messages"]:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("unsupported saved message role or shape")
        events = normalize_tau_messages(execution["messages"])
    except (ThenAtomizationError, ValueError, TypeError) as exc:
        normalization_error = type(exc).__name__
    results = []
    if normalization_error is None:
        results = [evaluate_input_check(checks, aid, events, sources=sources) for aid in branch["ready_input_check_ids"]]
    counts = {v: sum(r["input_conformance_verdict"] == v for r in results)
              for v in ("pass", "fail", "evidence_error", "not_observed")}
    unavailable = branch["unavailable_input_check_ids"]
    verdict = None
    if normalization_error:
        status = "evidence_error"
    elif not branch["ready_input_check_ids"]:
        status = "not_prepared"
    else:
        status = "partial" if unavailable else "evaluated"
        if counts["fail"]:
            verdict = "fail"
        elif counts["evidence_error"]:
            verdict = "evidence_error"
        elif unavailable:
            verdict = None
        elif counts["not_observed"]:
            verdict = "not_observed"
        else:
            verdict = "pass"
    result = {"schema_version": SCHEMA, "branch_id": branch_id,
              "source_handoff_fingerprint": handoff["handoff_fingerprint"],
              "input_check_set_fingerprint": checks["input_check_set_fingerprint"],
              "execution_identity": {"execution_fingerprint": execution_fingerprint,
                                     **{k: deepcopy(execution.get(k)) for k in ("candidate_id", "fixture_instance_id", "variant_id", "profile_id")}},
              "evidence": {"normalizer": "normalize_tau_messages", "status": "normalization_error" if normalization_error else "normalized",
                           "error_type": normalization_error,
                           "canonical_event_count": None if normalization_error else len(events),
                           "canonical_event_fingerprint": None if normalization_error else content_sha256(events),
                           "authenticity": "not_assessed", "completeness": "not_assessed"},
              "input_conformance": {"status": status, "verdict": verdict, "check_results": results,
                                    "check_counts": counts, "unavailable_check_ids": unavailable,
                                    "scope": "selected_input_checks_in_supplied_messages_not_all_tool_constraints"},
              "gwt_coverage": {"status": "not_assessed", "given_verdict": None, "when_coverage": None, "then_verdict": None},
              "task_completion": {"status": "not_assessed", "verdict": None},
              "overall_result": {"status": "partial", "verdict": None,
                                 "reason": "gwt_coverage_and_task_completion_not_implemented"},
              "legacy_branch_aggregation_allowed": False,
              "execution_effects": {"tools_dispatched": 0, "external_llm_calls": 0, "target_agent_calls": 0}}
    return seal(result, "partial_result_fingerprint")


def validate_result(result, checks, execution, branch_id, *, sources):
    if result != evaluate_execution_input(checks, execution, branch_id, sources=sources):
        raise ValueError("partial result differs from source/execution reconstruction")
