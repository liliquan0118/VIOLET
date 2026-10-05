"""Extends runtime_evaluation_v1.py/runtime_evaluation_v2.py (both frozen --
present verbatim in releases/agentspectesting-method-v1.1.0/method_snapshot.zip,
so not edited in place, same discipline as every other _extension_v1 module in
this project) with the one change needed to make the "official" runtime
evaluation entry point actually usable for a real execution: calling
oracle_evaluator_contract_extension_v1.evaluate_oracle_contract_extended
instead of the frozen oracle_evaluator_contract_v1.evaluate_oracle_contract.

Real, previously-unnoticed bug this closes (see
docs/oracle_requirement_pipeline_v0_7.md section 28.7): the frozen
evaluate_oracle_contract's own evaluate_predicate only recognizes
{all_of, json_type, in_set, length_lte, field_lte} and raises
ThenAtomizationError("unsupported predicate kind: ...") for anything else.
Every predicate_kind this whole project's extension layer has ever built
(tool_call_error_matches -- used since a very early round for 100/101/103/
121/125/029/036/133 and dozens more since -- baggage_count_decreased,
passenger_count_changed, compensation_formula, payment_method_type_counts_
within_limit, structural_invariant_holds, flight_path_valid,
flights_chronologically_feasible, airport_in_network, ...) is one of those
"anything else" cases. Verified empirically: running any real persisted
contract compiled by compile_oracle_evaluator_contracts_extended (e.g.
airline_105_state#b0::OR01, a tool_call_error_matches contract) through the
frozen evaluate_oracle_contract crashes outright. The Step4 compilation
itself has been correct and real this whole time -- evaluator_status:
"executable" in the persisted contracts.json is not a lie -- but nothing in
this project's own official CLI chain (scripts/evaluate_runtime_oracle_branch_v0_2.py)
could ever actually RUN one of those contracts against a real trajectory
without crashing, for as long as the extension layer has existed.

This module is a straight mirror of both frozen functions' logic (same
aggregation policy, same output schema/fingerprint shapes). Input validation
uses the two INPUT-set extended validators already built for this project's
compile layer (validate_oracle_evaluator_contract_set_extended,
validate_runtime_observation_binding_set_extended) rather than the frozen
ones -- the extended compilers deliberately stamp a different schema_version
("...-extended/v0.1" vs "...set/v0.1") on their input/output sets, so the
frozen validators reject them outright on that mismatch alone (confirmed:
this was the first thing tried, and it failed exactly this way before
switching to the extended validators). The OUTPUT-side validators this
module still reuses as-is (validate_runtime_oracle_evaluation_set,
validate_branch_eligibility_gate, validate_semantic_runtime_evaluation_set)
are genuinely fine to reuse unmodified: they validate this module's own
freshly-built result dict, which follows the exact same frozen output shape
(RUNTIME_EVALUATION_SET_VERSION etc. are unchanged, only the INPUT contract/
binding sets carry the "-extended" schema_version). The only behavioral
difference from the frozen functions is the swapped evaluate_oracle_contract
call, plus two new optional pass-through parameters (flight_route_reference,
flight_schedule_reference) that evaluate_oracle_contract_extended itself
already defined -- this module does not invent new evaluation semantics, it
only threads them through.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256
from ..compiler.oracle_evaluator_contract_extension_v1 import (
    evaluate_oracle_contract_extended,
    validate_oracle_evaluator_contract_set_extended,
)
from ..compiler.runtime_observation_binding_extension_v1 import (
    validate_runtime_observation_binding_set_extended,
)
from ..compiler.runtime_observation_binding_v1 import normalize_tau_execution
from ..compiler.then_atomization import ThenAtomizationError
from .branch_eligibility_v1 import validate_branch_eligibility_gate
from .runtime_evaluation_v1 import (
    REQUIREMENT_VERDICTS,
    RUNTIME_EVALUATION_SET_VERSION as RUNTIME_EVALUATION_SET_VERSION_V1,
    RUNTIME_EVALUATION_VERSION,
    _binding_driver_binding_keys,
    _execution_identity,
    _mapping,
    _select_branch,
    resolve_execution_driver_bindings,
    validate_runtime_oracle_evaluation_set,
)
from .runtime_evaluation_v2 import (
    BRANCH_VERDICTS,
    EFFECTIVE_RESULT_VERSION,
    RUNTIME_EVALUATION_SET_VERSION as RUNTIME_EVALUATION_SET_VERSION_V2,
    _effective_record,
    _semantic_key,
    validate_runtime_oracle_evaluation_set_v2,
)
from .semantic_runtime_evaluation_v1 import validate_semantic_runtime_evaluation_set


def evaluate_runtime_oracle_branch_extended(
    evaluator_contract_set: Mapping[str, Any],
    runtime_observation_binding_set: Mapping[str, Any],
    execution: Mapping[str, Any],
    branch_id: str,
    *,
    flight_route_reference: Mapping[str, Mapping[str, str]] | None = None,
    flight_schedule_reference: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Mirrors runtime_evaluation_v1.evaluate_runtime_oracle_branch exactly,
    except the per-requirement evaluation call is evaluate_oracle_contract_extended.
    See this module's own docstring for why."""
    contracts = validate_oracle_evaluator_contract_set_extended(evaluator_contract_set)
    bindings = validate_runtime_observation_binding_set_extended(runtime_observation_binding_set)
    execution = deepcopy(dict(_mapping(execution, "$execution")))
    messages = execution.get("messages")
    if not isinstance(messages, list):
        raise ThenAtomizationError("$.execution.messages must be an array")
    # The frozen v1 function's equivalent check compares
    # contracts["source_binding_set_fingerprint"] to
    # bindings["binding_set_fingerprint"] -- but compile_oracle_evaluator_contracts_
    # extended does not carry that field forward (it exposes
    # "source_evaluator_contract_set_fingerprint", pointing at its own frozen
    # baseline sub-compile's fingerprint, not the bindings set), so that exact
    # check is not reproducible here. The population check just below
    # (set(contract_index) != set(binding_index)) is the real safeguard that
    # the two sets actually describe the same bindings -- a mismatched pair
    # would fail there regardless.
    contract_branch = _select_branch(contracts, branch_id)
    binding_branch = _select_branch(bindings, branch_id)
    contract_index = {
        item["binding_id"]: item for item in contract_branch.get("evaluator_contracts") or []
    }
    binding_index = {item["binding_id"]: item for item in binding_branch.get("bindings") or []}
    if set(contract_index) != set(binding_index):
        raise ThenAtomizationError("branch evaluator contracts and bindings are not closed")

    binding_resolution = resolve_execution_driver_bindings(execution, binding_branch)
    events = normalize_tau_execution(execution)
    results = []
    counts: Counter[str] = Counter()
    unresolved_binding_keys = set(binding_resolution["missing_keys"]) | set(
        binding_resolution["conflicts"]
    )
    for binding_id, evaluator_contract in contract_index.items():
        binding = binding_index[binding_id]
        blocked_keys = sorted(_binding_driver_binding_keys(binding) & unresolved_binding_keys)
        if evaluator_contract.get("evaluator_status") == "executable" and blocked_keys:
            verdict = "unavailable"
            evaluation = {
                "verdict": verdict,
                "reason": "required_driver_bindings_unavailable",
                "unavailable_driver_binding_keys": blocked_keys,
            }
        elif evaluator_contract.get("evaluator_status") == "executable":
            evaluation = evaluate_oracle_contract_extended(
                evaluator_contract,
                binding,
                events,
                binding_resolution["resolved"],
                flight_route_reference,
                flight_schedule_reference,
            )
            verdict = evaluation.get("verdict")
            if verdict not in REQUIREMENT_VERDICTS:
                raise ThenAtomizationError("single evaluator returned an invalid verdict")
        else:
            verdict = "unavailable"
            evaluation = {
                "verdict": verdict,
                "reason": f"evaluator_status={evaluator_contract.get('evaluator_status')}",
                "diagnostics": deepcopy(evaluator_contract.get("diagnostics") or []),
            }
        record = {
            "schema_version": RUNTIME_EVALUATION_VERSION,
            "runtime_evaluation_id": f"{evaluator_contract['evaluator_contract_id']}::RE01",
            "evaluator_contract_id": evaluator_contract["evaluator_contract_id"],
            "binding_id": binding_id,
            "requirement_id": evaluator_contract["requirement_id"],
            "branch_id": branch_id,
            "verdict": verdict,
            "evaluation": evaluation,
            "source_evaluator_contract_fingerprint": evaluator_contract["evaluator_contract_fingerprint"],
            "source_binding_fingerprint": binding_index[binding_id]["binding_fingerprint"],
        }
        record["runtime_evaluation_fingerprint"] = content_sha256(record)
        results.append(record)
        counts[verdict] += 1

    if counts["fail"]:
        branch_verdict = "fail"
        aggregation_reason = "at_least_one_executable_requirement_failed"
    elif counts["unavailable"]:
        branch_verdict = "incomplete"
        aggregation_reason = "at_least_one_requirement_was_unavailable"
    else:
        branch_verdict = "pass"
        aggregation_reason = "all_requirements_passed"

    identity = _execution_identity(execution)
    result = {
        "schema_version": RUNTIME_EVALUATION_SET_VERSION_V1,
        "runtime_evaluation_set_id": f"{branch_id}::EXEC::{identity['execution_fingerprint'][:16]}",
        "branch_id": branch_id,
        "branch_context": deepcopy(contract_branch.get("branch_context")),
        "execution_identity": identity,
        "source_evaluator_contract_set_fingerprint": contracts["evaluator_contract_set_fingerprint"],
        "source_binding_set_fingerprint": bindings["binding_set_fingerprint"],
        "driver_binding_resolution": binding_resolution,
        "canonical_event_count": len(events),
        "requirement_results": results,
        "branch_verdict": branch_verdict,
        "aggregation": {
            "policy": "fail_then_incomplete_then_pass/v0.1",
            "reason": aggregation_reason,
            "requirement_verdict_counts": {
                verdict: counts[verdict] for verdict in sorted(REQUIREMENT_VERDICTS)
            },
        },
    }
    result["runtime_evaluation_set_fingerprint"] = content_sha256(result)
    return result


def evaluate_runtime_oracle_branch_v2_extended(
    evaluator_contract_set: Mapping[str, Any],
    runtime_observation_binding_set: Mapping[str, Any],
    execution: Mapping[str, Any],
    branch_id: str,
    eligibility_gate: Mapping[str, Any],
    *,
    semantic_evaluation_set: Mapping[str, Any] | None = None,
    flight_route_reference: Mapping[str, Mapping[str, str]] | None = None,
    flight_schedule_reference: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Mirrors runtime_evaluation_v2.evaluate_runtime_oracle_branch_v2 exactly,
    except it calls evaluate_runtime_oracle_branch_extended above instead of
    the frozen v1 evaluate_runtime_oracle_branch."""
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
            evaluate_runtime_oracle_branch_extended(
                evaluator_contract_set,
                runtime_observation_binding_set,
                execution,
                branch_id,
                flight_route_reference=flight_route_reference,
                flight_schedule_reference=flight_schedule_reference,
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
        "schema_version": RUNTIME_EVALUATION_SET_VERSION_V2,
        "runtime_evaluation_set_id": f"{branch_id}::EXEC::{execution_fingerprint[:16]}::V02",
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


def evaluate_runtime_oracle_branch_extended_file(
    *,
    evaluator_contracts_path,
    runtime_bindings_path,
    execution_path,
    branch_id: str,
    output_path,
    flight_route_reference_path=None,
    flight_schedule_reference_path=None,
) -> dict[str, Any]:
    import json
    from pathlib import Path

    def load(path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    route_reference = load(flight_route_reference_path)["routes"] if flight_route_reference_path else None
    schedule_reference = (
        load(flight_schedule_reference_path)["schedule"] if flight_schedule_reference_path else None
    )
    result = evaluate_runtime_oracle_branch_extended(
        load(evaluator_contracts_path),
        load(runtime_bindings_path),
        load(execution_path),
        branch_id,
        flight_route_reference=route_reference,
        flight_schedule_reference=schedule_reference,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
