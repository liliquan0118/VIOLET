"""Evaluate one execution against all Oracle contracts for one GWT branch."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256
from ..compiler.oracle_evaluator_contract_v1 import (
    evaluate_oracle_contract,
    validate_oracle_evaluator_contract_set,
)
from ..compiler.runtime_observation_binding_v1 import (
    normalize_tau_execution,
    validate_runtime_observation_binding_set,
)
from ..compiler.then_atomization import ThenAtomizationError


RUNTIME_EVALUATION_SET_VERSION = (
    "agentspectesting.runtime-oracle-evaluation-set/v0.1"
)
RUNTIME_EVALUATION_VERSION = "agentspectesting.runtime-oracle-evaluation/v0.1"
VERDICTS = frozenset({"pass", "fail", "incomplete"})
REQUIREMENT_VERDICTS = frozenset({"pass", "fail", "unavailable"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _select_branch(document: Mapping[str, Any], branch_id: str) -> Mapping[str, Any]:
    matches = [
        branch
        for branch in document.get("branches") or []
        if branch.get("branch_id") == branch_id
    ]
    if len(matches) != 1:
        raise ThenAtomizationError(
            f"branch_id must resolve exactly once: {branch_id!r}"
        )
    return _mapping(matches[0], "$.branches[]")


def _binding_driver_binding_keys(binding: Mapping[str, Any]) -> set[str]:
    keys = set()
    runtime = binding.get("runtime_binding") or {}
    constraints = list(runtime.get("scope_constraints") or [])
    constraints.extend((runtime.get("right_event") or {}).get("scope_constraints") or [])
    for scope in constraints:
        reference = str(scope.get("expected_binding_ref") or "")
        prefix = "driver_bindings."
        if reference.startswith(prefix) and len(reference) > len(prefix):
            keys.add(reference[len(prefix) :])
    return keys


def _required_driver_binding_keys(binding_branch: Mapping[str, Any]) -> list[str]:
    keys = set()
    for binding in binding_branch.get("bindings") or []:
        keys.update(_binding_driver_binding_keys(binding))
    return sorted(keys)


def resolve_execution_driver_bindings(
    execution: Mapping[str, Any], binding_branch: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve only compiler-requested keys from Driver output or legacy pre-state."""

    required = _required_driver_binding_keys(binding_branch)
    emitted = execution.get("driver_bindings")
    if emitted is None:
        emitted = {}
    emitted = _mapping(emitted, "$.execution.driver_bindings")
    pre_state = execution.get("pre_state")
    if pre_state is None:
        pre_state = {}
    pre_state = _mapping(pre_state, "$.execution.pre_state")
    resolved = {}
    sources = {}
    conflicts = {}
    for key in required:
        emitted_has = key in emitted
        pre_state_has = key in pre_state
        if emitted_has and pre_state_has and emitted[key] != pre_state[key]:
            conflicts[key] = {
                "driver_bindings": deepcopy(emitted[key]),
                "pre_state": deepcopy(pre_state[key]),
            }
            continue
        if emitted_has:
            resolved[key] = deepcopy(emitted[key])
            sources[key] = "execution.driver_bindings"
        elif pre_state_has:
            resolved[key] = deepcopy(pre_state[key])
            sources[key] = "execution.pre_state_legacy_adapter"
    missing = sorted(set(required) - set(resolved) - set(conflicts))
    return {
        "required_keys": required,
        "resolved": resolved,
        "sources": sources,
        "missing_keys": missing,
        "conflicts": conflicts,
        "status": "resolved" if not missing and not conflicts else "incomplete",
    }


def _execution_identity(execution: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": execution.get("candidate_id"),
        "fixture_instance_id": execution.get("fixture_instance_id"),
        "variant_id": execution.get("variant_id"),
        "profile_id": execution.get("profile_id"),
        "execution_schema_version": execution.get("execution_schema_version")
        or execution.get("schema_version"),
        "execution_fingerprint": content_sha256(execution),
    }


def evaluate_runtime_oracle_branch(
    evaluator_contract_set: Mapping[str, Any],
    runtime_observation_binding_set: Mapping[str, Any],
    execution: Mapping[str, Any],
    branch_id: str,
) -> dict[str, Any]:
    contracts = validate_oracle_evaluator_contract_set(evaluator_contract_set)
    bindings = validate_runtime_observation_binding_set(
        runtime_observation_binding_set
    )
    execution = deepcopy(dict(_mapping(execution, "$execution")))
    messages = execution.get("messages")
    if not isinstance(messages, list):
        raise ThenAtomizationError("$.execution.messages must be an array")
    if contracts.get("source_binding_set_fingerprint") != bindings.get(
        "binding_set_fingerprint"
    ):
        raise ThenAtomizationError("evaluator set uses a different runtime binding set")

    contract_branch = _select_branch(contracts, branch_id)
    binding_branch = _select_branch(bindings, branch_id)
    contract_index = {
        item["binding_id"]: item
        for item in contract_branch.get("evaluator_contracts") or []
    }
    binding_index = {
        item["binding_id"]: item for item in binding_branch.get("bindings") or []
    }
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
        blocked_keys = sorted(
            _binding_driver_binding_keys(binding) & unresolved_binding_keys
        )
        if evaluator_contract.get("evaluator_status") == "executable" and blocked_keys:
            verdict = "unavailable"
            evaluation = {
                "verdict": verdict,
                "reason": "required_driver_bindings_unavailable",
                "unavailable_driver_binding_keys": blocked_keys,
            }
        elif evaluator_contract.get("evaluator_status") == "executable":
            evaluation = evaluate_oracle_contract(
                evaluator_contract,
                binding,
                events,
                binding_resolution["resolved"],
            )
            verdict = evaluation.get("verdict")
            if verdict not in REQUIREMENT_VERDICTS:
                raise ThenAtomizationError("single evaluator returned an invalid verdict")
        else:
            verdict = "unavailable"
            evaluation = {
                "verdict": verdict,
                "reason": (
                    f"evaluator_status={evaluator_contract.get('evaluator_status')}"
                ),
                "diagnostics": deepcopy(evaluator_contract.get("diagnostics") or []),
            }
        record = {
            "schema_version": RUNTIME_EVALUATION_VERSION,
            "runtime_evaluation_id": (
                f"{evaluator_contract['evaluator_contract_id']}::RE01"
            ),
            "evaluator_contract_id": evaluator_contract[
                "evaluator_contract_id"
            ],
            "binding_id": binding_id,
            "requirement_id": evaluator_contract["requirement_id"],
            "branch_id": branch_id,
            "verdict": verdict,
            "evaluation": evaluation,
            "source_evaluator_contract_fingerprint": evaluator_contract[
                "evaluator_contract_fingerprint"
            ],
            "source_binding_fingerprint": binding_index[binding_id][
                "binding_fingerprint"
            ],
        }
        record["runtime_evaluation_fingerprint"] = content_sha256(record)
        results.append(record)
        counts[verdict] += 1

    # A directly observed failure is conclusive even if another requirement is
    # deferred. Without a failure, every requirement must pass before the branch
    # may be called passing.
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
        "schema_version": RUNTIME_EVALUATION_SET_VERSION,
        "runtime_evaluation_set_id": (
            f"{branch_id}::EXEC::{identity['execution_fingerprint'][:16]}"
        ),
        "branch_id": branch_id,
        "branch_context": deepcopy(contract_branch.get("branch_context")),
        "execution_identity": identity,
        "source_evaluator_contract_set_fingerprint": contracts[
            "evaluator_contract_set_fingerprint"
        ],
        "source_binding_set_fingerprint": bindings["binding_set_fingerprint"],
        "driver_binding_resolution": binding_resolution,
        "canonical_event_count": len(events),
        "requirement_results": results,
        "branch_verdict": branch_verdict,
        "aggregation": {
            "policy": "fail_then_incomplete_then_pass/v0.1",
            "reason": aggregation_reason,
            "requirement_verdict_counts": {
                verdict: counts[verdict]
                for verdict in sorted(REQUIREMENT_VERDICTS)
            },
        },
    }
    result["runtime_evaluation_set_fingerprint"] = content_sha256(result)
    return result


def validate_runtime_oracle_evaluation_set(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$runtime_evaluation_set")))
    fingerprint = result.pop("runtime_evaluation_set_fingerprint", None)
    if (
        result.get("schema_version") != RUNTIME_EVALUATION_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid runtime Oracle evaluation set")
    if result.get("branch_verdict") not in VERDICTS:
        raise ThenAtomizationError("invalid runtime Oracle branch verdict")
    identities = []
    counts: Counter[str] = Counter()
    for raw in result.get("requirement_results") or []:
        record = deepcopy(dict(_mapping(raw, "$.requirement_results[]")))
        record_fingerprint = record.pop("runtime_evaluation_fingerprint", None)
        if (
            record.get("schema_version") != RUNTIME_EVALUATION_VERSION
            or record_fingerprint != content_sha256(record)
        ):
            raise ThenAtomizationError("invalid runtime Oracle requirement result")
        if record.get("branch_id") != result.get("branch_id"):
            raise ThenAtomizationError("runtime Oracle result branch mismatch")
        if record.get("verdict") not in REQUIREMENT_VERDICTS:
            raise ThenAtomizationError("invalid runtime Oracle requirement verdict")
        identities.append(record.get("runtime_evaluation_id"))
        counts[record["verdict"]] += 1
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("runtime Oracle evaluation IDs must be unique")
    expected_counts = {
        verdict: counts[verdict] for verdict in sorted(REQUIREMENT_VERDICTS)
    }
    aggregation = _mapping(result.get("aggregation"), "$.aggregation")
    if aggregation.get("requirement_verdict_counts") != expected_counts:
        raise ThenAtomizationError("runtime Oracle verdict counts mismatch")
    expected_branch = (
        "fail"
        if counts["fail"]
        else "incomplete"
        if counts["unavailable"]
        else "pass"
    )
    if result.get("branch_verdict") != expected_branch:
        raise ThenAtomizationError("runtime Oracle branch aggregation mismatch")
    result["runtime_evaluation_set_fingerprint"] = fingerprint
    return result


def evaluate_runtime_oracle_branch_file(
    *,
    evaluator_contracts_path: str | Path,
    runtime_bindings_path: str | Path,
    execution_path: str | Path,
    branch_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    result = evaluate_runtime_oracle_branch(
        load(evaluator_contracts_path),
        load(runtime_bindings_path),
        load(execution_path),
        branch_id,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
