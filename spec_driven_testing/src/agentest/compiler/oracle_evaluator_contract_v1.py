"""Compile and execute deterministic Oracle evaluator contracts."""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .oracle_requirement_acceptance_v1 import (
    validate_accepted_oracle_requirement_set,
)
from .runtime_observation_binding_v1 import (
    extract_bound_observations,
    validate_runtime_observation_binding_set,
)
from .then_atomization import ThenAtomizationError


EVALUATOR_SET_VERSION = "agentspectesting.oracle-evaluator-contract-set/v0.1"
EVALUATOR_VERSION = "agentspectesting.oracle-evaluator-contract/v0.1"
EVALUATOR_STATUSES = frozenset({"executable", "deferred", "needs_adjudication"})

_TYPE_NAMES = {
    "array": "array",
    "boolean": "boolean",
    "integer": "integer",
    "number": "number",
    "object": "object",
    "string": "string",
}
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_TYPE_RULE = re.compile(
    r"\bmust be (?:an? )?(string|integer|number|boolean|array|object)\b",
    re.IGNORECASE,
)
_AT_MOST_RULE = re.compile(
    r"\bat most (?P<limit>[0-9]+|[a-z]+) (?P<field>[a-z][a-z0-9_]*)\b",
    re.IGNORECASE,
)
_NOT_EXCEED_RULE = re.compile(
    r"^\s*(?:the\s+)?(?P<left>[a-z][a-z0-9_]*)\s+must\s+not\s+exceed\s+"
    r"(?:the\s+)?(?P<right>[a-z][a-z0-9_]*)\s*[.]?\s*$",
    re.IGNORECASE,
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _all_of(predicates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [deepcopy(dict(item)) for item in predicates]
    if len(values) == 1:
        return values[0]
    return {"predicate_kind": "all_of", "predicates": values}


def _number(text: str) -> int | None:
    lowered = text.casefold()
    if lowered.isdigit():
        return int(lowered)
    return _NUMBER_WORDS.get(lowered)


def _singular(text: str) -> str:
    lowered = text.casefold()
    if lowered.endswith("ies") and len(lowered) > 3:
        return lowered[:-3] + "y"
    if lowered.endswith("s") and len(lowered) > 1:
        return lowered[:-1]
    return lowered


def _tool_argument_predicate(
    requirement: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    endpoint = _mapping(contract.get("endpoint") or {}, "$.observation_contract.endpoint")
    parameter = contract.get("parameter")
    text = str(requirement.get("requirement_text") or "")
    predicates = []

    allowed_values = endpoint.get("allowed_values") or []
    if allowed_values:
        predicates.append(
            {"predicate_kind": "in_set", "values": deepcopy(allowed_values)}
        )
        endpoint_type = endpoint.get("type")
        if endpoint_type in _TYPE_NAMES:
            predicates.insert(
                0, {"predicate_kind": "json_type", "type": endpoint_type}
            )
        return _all_of(predicates), []

    type_match = _TYPE_RULE.search(text)
    if type_match:
        declared = type_match.group(1).casefold()
        endpoint_type = endpoint.get("type")
        if endpoint_type in _TYPE_NAMES and endpoint_type != declared:
            return None, [
                {
                    "code": "requirement_endpoint_type_conflict",
                    "requirement_type": declared,
                    "endpoint_type": endpoint_type,
                }
            ]
        return {"predicate_kind": "json_type", "type": declared}, []

    limit_match = _AT_MOST_RULE.search(text)
    if limit_match and isinstance(parameter, str):
        limit = _number(limit_match.group("limit"))
        named_field = limit_match.group("field")
        if limit is not None and _singular(named_field) == _singular(parameter):
            if endpoint.get("type") != "array":
                return None, [
                    {
                        "code": "cardinality_requires_array_endpoint",
                        "endpoint_type": endpoint.get("type"),
                    }
                ]
            return {
                "predicate_kind": "all_of",
                "predicates": [
                    {"predicate_kind": "json_type", "type": "array"},
                    {"predicate_kind": "length_lte", "value": limit},
                ],
            }, []

    return None, [{"code": "tool_argument_rule_not_in_supported_grammar"}]


def _tool_argument_constraint_predicate(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    text = str(contract.get("constraint_text") or "")
    match = _NOT_EXCEED_RULE.match(text)
    parameters = contract.get("parameters") or []
    if not match:
        return None, [{"code": "argument_constraint_not_in_supported_grammar"}]
    left = match.group("left")
    right = match.group("right")
    if left not in parameters or right not in parameters:
        return None, [
            {
                "code": "constraint_fields_not_in_observation_parameters",
                "constraint_fields": [left, right],
                "observation_parameters": deepcopy(parameters),
            }
        ]
    endpoints = _mapping(contract.get("endpoints") or {}, "$.observation_contract.endpoints")
    numeric_types = {"integer", "number"}
    endpoint_types = {name: (endpoints.get(name) or {}).get("type") for name in (left, right)}
    if any(value not in numeric_types for value in endpoint_types.values()):
        return None, [
            {"code": "ordered_comparison_requires_numeric_endpoints", "types": endpoint_types}
        ]
    return {
        "predicate_kind": "field_lte",
        "left_path": f"arguments.{left}",
        "right_path": f"arguments.{right}",
    }, []


def _compile_one(
    requirement: Mapping[str, Any], binding: Mapping[str, Any]
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    binding_status = binding.get("binding_status")
    if binding_status == "needs_adjudication":
        return (
            "needs_adjudication",
            {"evaluator_kind": "unavailable"},
            [{"code": "observation_binding_needs_adjudication"}],
        )
    if binding_status != "bound":
        return (
            "deferred",
            {"evaluator_kind": "unavailable"},
            [{"code": "observation_binding_not_executable", "binding_status": binding_status}],
        )

    expected = _mapping(binding.get("expected_observation"), "$.expected_observation")
    operator = expected.get("operator")
    if operator in {"present", "absent"}:
        return (
            "executable",
            {
                "evaluator_kind": "match_cardinality",
                "operator": operator,
                "match_count": {"comparator": "gt" if operator == "present" else "eq", "value": 0},
            },
            [],
        )

    if operator == "predicate_true":
        observation_contract = _mapping(
            requirement.get("observation_contract") or {}, "$.observation_contract"
        )
        if requirement.get("requirement_type") == "tool_argument":
            predicate, diagnostics = _tool_argument_predicate(
                requirement, observation_contract
            )
        elif requirement.get("requirement_type") == "tool_argument_constraint":
            predicate, diagnostics = _tool_argument_constraint_predicate(
                observation_contract
            )
        else:
            predicate, diagnostics = None, [
                {"code": "predicate_operator_requirement_type_mismatch"}
            ]
        if predicate is None:
            return "deferred", {"evaluator_kind": "unavailable"}, diagnostics
        return (
            "executable",
            {
                "evaluator_kind": "all_matches_predicate",
                "requires_observation": bool(expected.get("requires_observation")),
                "predicate": predicate,
            },
            diagnostics,
        )

    if operator == "relation_true":
        runtime = _mapping(binding.get("runtime_binding"), "$.runtime_binding")
        if (
            runtime.get("extractor_kind") != "temporal_relation"
            or runtime.get("relation") != "precedes"
        ):
            return (
                "deferred",
                {"evaluator_kind": "unavailable"},
                [{"code": "unsupported_temporal_binding"}],
            )
        return (
            "executable",
            {
                "evaluator_kind": "all_target_events_have_predecessor",
                "relation": "precedes",
                "requires_observation": bool(expected.get("requires_observation")),
            },
            [],
        )

    return (
        "deferred",
        {"evaluator_kind": "unavailable"},
        [{"code": "unsupported_expected_operator", "operator": operator}],
    )


def compile_oracle_evaluator_contracts(
    accepted_requirement_set: Mapping[str, Any],
    runtime_observation_binding_set: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = validate_accepted_oracle_requirement_set(accepted_requirement_set)
    bindings = validate_runtime_observation_binding_set(runtime_observation_binding_set)
    requirements = {
        item["requirement_id"]: item
        for branch in accepted["branches"]
        for item in branch["requirements"]
    }
    binding_index = {
        item["requirement_id"]: item
        for branch in bindings["branches"]
        for item in branch["bindings"]
    }
    if set(requirements) != set(binding_index):
        raise ThenAtomizationError("accepted requirements and runtime bindings are not closed")

    branches = []
    counts: Counter[str] = Counter()
    deferred_branches = []
    for source_branch in bindings["branches"]:
        contracts = []
        for binding in source_branch["bindings"]:
            requirement = requirements[binding["requirement_id"]]
            if binding.get("source_requirement_fingerprint") != requirement.get(
                "requirement_fingerprint"
            ):
                raise ThenAtomizationError(
                    f"runtime binding requirement lineage mismatch: {binding['requirement_id']}"
                )
            status, program, diagnostics = _compile_one(requirement, binding)
            record = {
                "schema_version": EVALUATOR_VERSION,
                "evaluator_contract_id": f"{binding['binding_id']}::EC01",
                "binding_id": binding["binding_id"],
                "requirement_id": binding["requirement_id"],
                "expectation_id": binding["expectation_id"],
                "branch_id": source_branch["branch_id"],
                "evaluator_status": status,
                "program": program,
                "expected_observation": deepcopy(binding["expected_observation"]),
                "diagnostics": diagnostics,
                "source_binding_fingerprint": binding["binding_fingerprint"],
                "source_requirement_fingerprint": requirement[
                    "requirement_fingerprint"
                ],
            }
            record["evaluator_contract_fingerprint"] = content_sha256(record)
            contracts.append(record)
            counts[status] += 1
        statuses = {item["evaluator_status"] for item in contracts}
        if "needs_adjudication" in statuses:
            branch_status = "needs_adjudication"
        elif statuses <= {"executable"}:
            branch_status = "executable"
        else:
            branch_status = "partially_executable"
            deferred_branches.append(source_branch["branch_id"])
        branches.append(
            {
                "branch_id": source_branch["branch_id"],
                "branch_context": deepcopy(source_branch["branch_context"]),
                "evaluator_contracts": contracts,
                "evaluator_status": branch_status,
            }
        )

    result = {
        "schema_version": EVALUATOR_SET_VERSION,
        "source_accepted_set_fingerprint": accepted["accepted_set_fingerprint"],
        "source_binding_set_fingerprint": bindings["binding_set_fingerprint"],
        "branches": branches,
        "summary": {
            "branch_count": len(branches),
            "evaluator_contract_count": sum(
                len(branch["evaluator_contracts"]) for branch in branches
            ),
            "evaluator_status_counts": {
                status: counts[status] for status in sorted(EVALUATOR_STATUSES)
            },
            "fully_executable_branch_count": sum(
                branch["evaluator_status"] == "executable" for branch in branches
            ),
            "partially_executable_branch_count": sum(
                branch["evaluator_status"] == "partially_executable"
                for branch in branches
            ),
            "adjudication_branch_count": sum(
                branch["evaluator_status"] == "needs_adjudication"
                for branch in branches
            ),
        },
        "deferred_branches": deferred_branches,
        "next_stage": "runtime_oracle_evaluation",
    }
    result["evaluator_contract_set_fingerprint"] = content_sha256(result)
    return result


def validate_oracle_evaluator_contract_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$evaluator_contract_set")))
    fingerprint = result.pop("evaluator_contract_set_fingerprint", None)
    if result.get("schema_version") != EVALUATOR_SET_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid Oracle evaluator contract set")
    identities = []
    counts: Counter[str] = Counter()
    for branch in result.get("branches") or []:
        for raw in branch.get("evaluator_contracts") or []:
            record = deepcopy(dict(_mapping(raw, "$.evaluator_contracts[]")))
            record_fingerprint = record.pop("evaluator_contract_fingerprint", None)
            if record.get("schema_version") != EVALUATOR_VERSION or record_fingerprint != content_sha256(record):
                raise ThenAtomizationError("invalid Oracle evaluator contract")
            if record.get("branch_id") != branch.get("branch_id"):
                raise ThenAtomizationError("Oracle evaluator contract branch mismatch")
            if record.get("evaluator_status") not in EVALUATOR_STATUSES:
                raise ThenAtomizationError("invalid Oracle evaluator status")
            identities.append(record.get("evaluator_contract_id"))
            counts[record["evaluator_status"]] += 1
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("Oracle evaluator contract IDs must be unique")
    summary = _mapping(result.get("summary"), "$.summary")
    if summary.get("evaluator_contract_count") != len(identities):
        raise ThenAtomizationError("Oracle evaluator contract count mismatch")
    expected = {status: counts[status] for status in sorted(EVALUATOR_STATUSES)}
    if summary.get("evaluator_status_counts") != expected:
        raise ThenAtomizationError("Oracle evaluator status counts mismatch")
    result["evaluator_contract_set_fingerprint"] = fingerprint
    return result


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, Mapping)
    return False


def evaluate_predicate(predicate: Mapping[str, Any], value: Any) -> dict[str, Any]:
    kind = predicate.get("predicate_kind")
    if kind == "all_of":
        children = [evaluate_predicate(item, value) for item in predicate.get("predicates") or []]
        return {"passed": all(item["passed"] for item in children), "children": children}
    if kind == "json_type":
        return {"passed": _json_type_matches(value, predicate.get("type")), "actual": value}
    if kind == "in_set":
        return {"passed": value in (predicate.get("values") or []), "actual": value}
    if kind == "length_lte":
        passed = hasattr(value, "__len__") and len(value) <= predicate.get("value")
        return {
            "passed": bool(passed),
            "actual_length": len(value) if hasattr(value, "__len__") else None,
        }
    if kind == "field_lte":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        left = value.get(predicate.get("left_path"))
        right = value.get(predicate.get("right_path"))
        numeric = all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in (left, right)
        )
        return {
            "passed": bool(numeric and left <= right),
            "left": left,
            "right": right,
        }
    raise ThenAtomizationError(f"unsupported predicate kind: {kind}")


def evaluate_oracle_contract(
    evaluator_contract: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one executable contract and retain direct observation evidence."""

    if evaluator_contract.get("evaluator_status") != "executable":
        return {
            "verdict": "unavailable",
            "reason": f"evaluator_status={evaluator_contract.get('evaluator_status')}",
        }
    if evaluator_contract.get("binding_id") != runtime_binding.get("binding_id"):
        raise ThenAtomizationError("evaluator contract and runtime binding do not match")
    if evaluator_contract.get("source_binding_fingerprint") != runtime_binding.get(
        "binding_fingerprint"
    ):
        raise ThenAtomizationError("evaluator contract uses a stale runtime binding")
    observation = extract_bound_observations(runtime_binding, events, driver_bindings)
    if observation.get("status") != "observed":
        return {"verdict": "unavailable", "observation": observation}

    matches = observation["matches"]
    program = _mapping(evaluator_contract.get("program"), "$.program")
    if program.get("evaluator_kind") == "match_cardinality":
        comparator = (program.get("match_count") or {}).get("comparator")
        passed = len(matches) > 0 if comparator == "gt" else len(matches) == 0
        return {
            "verdict": "pass" if passed else "fail",
            "match_count": len(matches),
            "observation": observation,
        }

    if program.get("evaluator_kind") == "all_matches_predicate":
        evaluations = [
            {
                "event_index": match.get("event_index"),
                **evaluate_predicate(program["predicate"], match.get("value")),
            }
            for match in matches
        ]
        missing_required = bool(program.get("requires_observation")) and not matches
        passed = not missing_required and all(item["passed"] for item in evaluations)
        return {
            "verdict": "pass" if passed else "fail",
            "match_count": len(matches),
            "missing_required_observation": missing_required,
            "predicate_evaluations": evaluations,
            "observation": observation,
        }
    if program.get("evaluator_kind") == "all_target_events_have_predecessor":
        target_count = int(observation.get("target_event_count") or 0)
        matched_count = int(observation.get("matched_target_count") or 0)
        missing_required = bool(program.get("requires_observation")) and target_count == 0
        passed = not missing_required and target_count == matched_count
        return {
            "verdict": "pass" if passed else "fail",
            "target_event_count": target_count,
            "matched_target_count": matched_count,
            "missing_required_observation": missing_required,
            "observation": observation,
        }
    raise ThenAtomizationError("unsupported evaluator program")


def compile_oracle_evaluator_contracts_file(
    *,
    accepted_requirements_path: str | Path,
    runtime_bindings_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    result = compile_oracle_evaluator_contracts(
        load(accepted_requirements_path), load(runtime_bindings_path)
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
