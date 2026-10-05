"""Compile expected oracle observations into canonical tau runtime bindings."""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .oracle_expectation_compiler_v1 import validate_oracle_expectation_set
from .oracle_requirement_acceptance_v1 import (
    validate_accepted_oracle_requirement_set,
)
from .then_atomization import ThenAtomizationError


PROFILE_VERSION = "agentspectesting.runtime-observation-profile/v0.1"
BINDING_SET_VERSION = "agentspectesting.runtime-observation-binding-set/v0.1"
BINDING_VERSION = "agentspectesting.runtime-observation-binding/v0.1"
BINDING_STATUSES = frozenset(
    {"bound", "partial", "semantic_deferred", "needs_adjudication"}
)

_TOKEN = re.compile(r"[a-z0-9]+")
_TOKEN_STOP = frozenset(
    {
        "a",
        "an",
        "in",
        "information",
        "of",
        "request",
        "requests",
        "the",
        "to",
        "user",
    }
)
_TOKEN_CANONICAL = {
    "booking": "book",
    "booked": "book",
    "change": "update",
    "changed": "update",
    "changing": "update",
    "edit": "update",
    "editing": "update",
    "edited": "update",
    "modify": "update",
    "modified": "update",
    "modifying": "update",
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def validate_runtime_observation_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = deepcopy(dict(_mapping(value, "$runtime_observation_profile")))
    if profile.get("schema_version") != PROFILE_VERSION:
        raise ThenAtomizationError("unsupported runtime observation profile")
    if not isinstance(profile.get("profile_id"), str) or not profile["profile_id"]:
        raise ThenAtomizationError("runtime observation profile_id must be non-empty")
    stream = _mapping(profile.get("canonical_event_stream"), "$.canonical_event_stream")
    if set(stream) != {"stream_id", "order_field", "source_message_field"}:
        raise ThenAtomizationError("canonical event stream fields are invalid")
    adapter = _mapping(profile.get("raw_message_adapter"), "$.raw_message_adapter")
    if set(adapter) != {"adapter_id", "messages_path"}:
        raise ThenAtomizationError("raw message adapter fields are invalid")
    if adapter.get("adapter_id") != "tau_message_sequence/v0.1":
        raise ThenAtomizationError("unsupported raw message adapter")
    contracts = _mapping(profile.get("event_contracts"), "$.event_contracts")
    required = {
        "assistant_tool_call",
        "assistant_message",
        "user_message",
        "tool_result",
    }
    if set(contracts) != required:
        raise ThenAtomizationError("runtime event contracts are incomplete")
    if profile.get("driver_binding_namespace") != "driver_bindings":
        raise ThenAtomizationError("unsupported driver binding namespace")
    scope_parameters = profile.get("scope_binding_parameters")
    if (
        not isinstance(scope_parameters, list)
        or any(not isinstance(item, str) or not item for item in scope_parameters)
        or len(scope_parameters) != len(set(scope_parameters))
    ):
        raise ThenAtomizationError("scope_binding_parameters must be unique strings")
    return profile


def _execution_contract_index(
    document: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    contracts = document.get("operation_family_contracts")
    if not isinstance(contracts, list):
        raise ThenAtomizationError("execution matching contracts are missing operation families")
    result = {}
    for raw in contracts:
        contract = _mapping(raw, "$.operation_family_contracts[]")
        family_id = contract.get("operation_policy_family_id")
        selector = _mapping(contract.get("operation_selector"), "$.operation_selector")
        tool_names = selector.get("verified_tool_names")
        if (
            not isinstance(family_id, str)
            or not family_id
            or family_id in result
            or not isinstance(tool_names, list)
            or any(not isinstance(item, str) or not item for item in tool_names)
        ):
            raise ThenAtomizationError("execution operation-family contract is invalid")
        result[family_id] = {
            "operation_family_contract_id": contract.get(
                "operation_family_contract_id"
            ),
            "verified_tool_names": list(tool_names),
            "operation_expressions": deepcopy(
                selector.get("operation_expressions") or []
            ),
        }
    return result


def _tokens(value: Any) -> set[str]:
    result = set()
    for token in _TOKEN.findall(str(value or "").casefold().replace("_", " ")):
        token = _TOKEN_CANONICAL.get(token, token)
        if token.endswith("ies") and len(token) > 3:
            token = token[:-3] + "y"
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            token = token[:-1]
        if token not in _TOKEN_STOP:
            result.add(token)
    return result


def _tool_catalog_index(document: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if document is None:
        return {}
    tools = document.get("tools")
    if not isinstance(tools, list):
        raise ThenAtomizationError("tool catalog must contain a tools array")
    result = {}
    for raw in tools:
        tool = _mapping(raw, "$.tools[]")
        name = tool.get("name")
        if not isinstance(name, str) or not name or name in result:
            raise ThenAtomizationError("tool catalog contains an invalid tool name")
        parameters = tool.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise ThenAtomizationError("tool parameters must be an object")
        if "properties" in parameters and isinstance(parameters["properties"], Mapping):
            parameters = parameters["properties"]
        required = tool.get("required_parameters")
        if required is None and isinstance(tool.get("parameters"), Mapping):
            required = tool["parameters"].get("required")
        result[name] = {
            "description": str(tool.get("description") or ""),
            "parameter_names": sorted(parameters),
            "required_parameters": list(required or []),
        }
    return result


def _semantic_unit_index(document: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if document is None:
        return {}
    units = document.get("semantic_units")
    if not isinstance(units, list):
        raise ThenAtomizationError("semantic unit model must contain semantic_units")
    result = {}
    for raw in units:
        unit = _mapping(raw, "$.semantic_units[]")
        unit_id = unit.get("semantic_unit_id")
        if not isinstance(unit_id, str) or not unit_id or unit_id in result:
            raise ThenAtomizationError("semantic unit model contains an invalid ID")
        result[unit_id] = deepcopy(dict(unit))
    return result


def _resolve_branch_operation_tool(
    branch_context: Mapping[str, Any],
    family_candidates: Sequence[Mapping[str, Any]],
    concrete_branch_tools: set[str],
    tool_catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    family_tools = {
        tool
        for family in family_candidates
        for tool in family.get("verified_tool_names") or []
    }
    exact = sorted(concrete_branch_tools & family_tools)
    if len(exact) == 1:
        return exact, {
            "resolver_kind": "same_branch_concrete_tool",
            "selected_tool": exact[0],
        }
    if not tool_catalog:
        return [], {"resolver_kind": "tool_catalog_unavailable"}
    when = ((branch_context.get("gwt") or {}).get("when") or "")
    query_tokens = _tokens(when)
    scores = []
    for tool_name in sorted(family_tools):
        tool = tool_catalog.get(tool_name)
        if tool is None:
            continue
        name_tokens = _tokens(tool_name)
        description_tokens = _tokens(tool.get("description"))
        parameter_tokens = {
            token
            for parameter in tool.get("parameter_names") or []
            for token in _tokens(parameter)
        }
        score = (
            3 * len(query_tokens & name_tokens)
            + 2 * len(query_tokens & description_tokens)
            + 2 * len(query_tokens & parameter_tokens)
        )
        scores.append(
            {
                "tool_name": tool_name,
                "score": score,
                "name_overlap": sorted(query_tokens & name_tokens),
                "description_overlap": sorted(query_tokens & description_tokens),
                "parameter_overlap": sorted(query_tokens & parameter_tokens),
            }
        )
    scores.sort(key=lambda item: (-item["score"], item["tool_name"]))
    selected = []
    if scores and scores[0]["score"] > 0:
        runner_up = scores[1]["score"] if len(scores) > 1 else 0
        if scores[0]["score"] - runner_up >= 2:
            selected = [scores[0]["tool_name"]]
    return selected, {
        "resolver_kind": "branch_when_tool_feature_overlap/v0.1",
        "query_tokens": sorted(query_tokens),
        "candidate_scores": scores,
        "minimum_unique_margin": 2,
        "selected_tool": selected[0] if selected else None,
    }


def _compile_semantic_left_event(
    semantic_unit_id: Any,
    semantic_units: Mapping[str, Mapping[str, Any]],
    selected_tool: str,
    tool_catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    unit = semantic_units.get(semantic_unit_id)
    if unit is None:
        return None, {"code": "semantic_unit_not_available"}
    action_text = str((unit.get("semantic_payload") or {}).get("direct_action_text") or "")
    normalized = " ".join(action_text.casefold().split())
    if "arguments of the selected operation" in normalized or "action details" in normalized:
        required_parameters = (tool_catalog.get(selected_tool) or {}).get(
            "required_parameters"
        ) or []
        if not required_parameters:
            return None, {"code": "selected_tool_required_parameters_unavailable"}
        return (
            {
                "binding_kind": "assistant_argument_disclosure",
                "event_filter": {"event_kind": "assistant_message"},
                "matcher": {
                    "kind": "contains_right_call_argument_values",
                    "required_argument_paths": [
                        f"arguments.{name}" for name in required_parameters
                    ],
                    "collection_policy": "all_scalar_leaves",
                    "match_policy": "one_message_contains_all_values",
                },
            },
            {"code": "compiled_argument_disclosure_grammar"},
        )
    if "explicit user confirmation" in normalized or "confirmation (yes)" in normalized:
        return (
            {
                "binding_kind": "driver_action_event",
                "event_filter": {
                    "event_kind": "user_message",
                    "field_equals": {"driver_action_kind": "confirm_operation"},
                },
            },
            {"code": "compiled_driver_confirmation_grammar"},
        )
    return None, {"code": "semantic_action_not_in_supported_event_grammar"}


def _scope_constraints(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "actual_path": f"arguments.{parameter}",
            "comparator": "equals",
            "expected_binding_ref": f"driver_bindings.{parameter}",
        }
        for parameter in contract.get("scope_parameters") or []
    ]


def _tool_event_binding(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "extractor_kind": "event_filter",
        "event_stream": "canonical_tau_events",
        "event_filter": {
            "event_kind": "assistant_tool_call",
            "field_equals": {"tool_name": contract.get("tool_name")},
        },
        "scope_constraints": _scope_constraints(contract),
        "projection": {"kind": "event"},
    }


def _binding_for_requirement(
    requirement: Mapping[str, Any],
    *,
    branch_context: Mapping[str, Any],
    branch_requirements: Sequence[Mapping[str, Any]],
    operation_families: Mapping[str, Mapping[str, Any]],
    tool_catalog: Mapping[str, Mapping[str, Any]],
    semantic_units: Mapping[str, Mapping[str, Any]],
    scope_binding_parameters: Sequence[str],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    kind = requirement.get("requirement_type")
    contract = _mapping(requirement.get("observation_contract") or {}, "$.observation_contract")
    resolution = requirement.get("resolution_hint") or {}
    diagnostics = []

    if requirement.get("runtime_binding_status") == "needs_adjudication":
        return (
            "needs_adjudication",
            {"extractor_kind": "unbound"},
            [{"code": "upstream_requirement_needs_adjudication"}],
        )

    if kind == "tool_call":
        return "bound", _tool_event_binding(contract), diagnostics

    if kind == "tool_argument":
        binding = _tool_event_binding(contract)
        binding["projection"] = {
            "kind": "field",
            "path": f"arguments.{contract.get('parameter')}",
        }
        return "bound", binding, diagnostics

    if kind == "tool_argument_constraint":
        binding = _tool_event_binding(contract)
        binding["projection"] = {
            "kind": "field_tuple",
            "paths": [f"arguments.{item}" for item in contract.get("parameters") or []],
        }
        binding["predicate_source"] = {
            "kind": "constraint_text",
            "text": contract.get("constraint_text"),
        }
        return "bound", binding, diagnostics

    if kind == "assistant_literal":
        return (
            "bound",
            {
                "extractor_kind": "event_filter",
                "event_stream": "canonical_tau_events",
                "event_filter": {"event_kind": "assistant_message"},
                "scope_constraints": [],
                "projection": {"kind": "field", "path": "content"},
                "matcher": {
                    "kind": "contains_literal",
                    "literal": contract.get("literal"),
                    "case_sensitive": True,
                },
            },
            diagnostics,
        )

    if resolution.get("resolver_kind") == "explicit_tool_call":
        resolved = {"tool_name": resolution.get("tool_name")}
        return "bound", _tool_event_binding(resolved), diagnostics

    if resolution.get("resolver_kind") == "explicit_assistant_literal":
        literal = resolution.get("literal")
        return (
            "bound",
            {
                "extractor_kind": "event_filter",
                "event_stream": "canonical_tau_events",
                "event_filter": {"event_kind": "assistant_message"},
                "scope_constraints": [],
                "projection": {"kind": "field", "path": "content"},
                "matcher": {
                    "kind": "contains_literal",
                    "literal": literal,
                    "case_sensitive": True,
                },
            },
            diagnostics,
        )

    if kind == "temporal_relation":
        refs = contract.get("runtime_contract_refs") or []
        family_ids = list(
            dict.fromkeys(
                ref.get("operation_policy_family_id")
                for ref in refs
                if isinstance(ref, Mapping)
                and isinstance(ref.get("operation_policy_family_id"), str)
            )
        )
        family_candidates = [
            {"operation_policy_family_id": family_id, **deepcopy(operation_families[family_id])}
            for family_id in family_ids
            if family_id in operation_families
        ]
        concrete_branch_tools = {
            (item.get("observation_contract") or {}).get("tool_name")
            for item in branch_requirements
            if item.get("requirement_type") == "tool_call"
            and item.get("runtime_binding_status") == "bound"
        }
        concrete_branch_tools.discard(None)
        family_tools = {
            tool for family in family_candidates for tool in family["verified_tool_names"]
        }
        exact_target_tools, operation_resolution = _resolve_branch_operation_tool(
            branch_context,
            family_candidates,
            concrete_branch_tools,
            tool_catalog,
        )
        right_event = {
            "binding_kind": (
                "exact_tool_call"
                if len(exact_target_tools) == 1
                else "operation_family_candidates"
            ),
            "operation_family_candidates": family_candidates,
            "exact_tool_names": exact_target_tools,
            "binding_status": "bound" if len(exact_target_tools) == 1 else "partial",
            "operation_resolution": operation_resolution,
        }
        if len(exact_target_tools) == 1:
            selected_tool = exact_target_tools[0]
            right_event["event_filter"] = {
                "event_kind": "assistant_tool_call",
                "field_equals": {"tool_name": selected_tool},
            }
            tool_parameters = set(
                (tool_catalog.get(selected_tool) or {}).get("parameter_names") or []
            )
            right_event["scope_constraints"] = [
                {
                    "actual_path": f"arguments.{parameter}",
                    "comparator": "equals",
                    "expected_binding_ref": f"driver_bindings.{parameter}",
                }
                for parameter in scope_binding_parameters
                if parameter in tool_parameters
            ]
        if right_event["binding_status"] == "partial":
            diagnostics.append(
                {
                    "code": "requested_operation_tool_not_uniquely_bound",
                    "candidate_tool_names": sorted(family_tools),
                }
            )
        left_event = None
        left_diagnostic = {"code": "right_operation_not_uniquely_bound"}
        if len(exact_target_tools) == 1:
            left_event, left_diagnostic = _compile_semantic_left_event(
                contract.get("left_event_semantic_unit_id"),
                semantic_units,
                exact_target_tools[0],
                tool_catalog,
            )
        diagnostics.append(left_diagnostic)
        fully_bound = right_event["binding_status"] == "bound" and left_event is not None
        return (
            "bound" if fully_bound else "partial",
            {
                "extractor_kind": "temporal_relation",
                "event_stream": "canonical_tau_events",
                "order_field": "event_index",
                "relation": contract.get("relation"),
                "left_event": left_event
                or {
                    "binding_kind": "semantic_event_deferred",
                    "semantic_unit_id": contract.get("left_event_semantic_unit_id"),
                    "requirement_text": requirement.get("requirement_text"),
                    "candidate_event_kinds": ["assistant_message", "user_message"],
                },
                "right_event": right_event,
            },
            diagnostics,
        )

    if kind == "semantic_requirement":
        semantic_tokens = _tokens(requirement.get("requirement_text"))
        semantic_normalized = " ".join(
            str(requirement.get("requirement_text") or "").casefold().split()
        )
        aliases = []
        for item in branch_requirements:
            if (
                item.get("requirement_type") != "tool_call"
                or item.get("runtime_binding_status") != "bound"
            ):
                continue
            tool_name = (item.get("observation_contract") or {}).get("tool_name")
            tool = tool_catalog.get(tool_name) or {}
            description_tokens = _tokens(tool.get("description"))
            description_normalized = " ".join(
                str(tool.get("description") or "").casefold().split()
            )
            exact_phrase = (
                len(semantic_normalized) >= 20
                and semantic_normalized in description_normalized
            )
            token_coverage = (
                len(semantic_tokens) >= 5 and semantic_tokens <= description_tokens
            )
            if exact_phrase or token_coverage:
                aliases.append(
                    (
                        item,
                        tool_name,
                        sorted(description_tokens),
                        (
                            "exact_normalized_phrase_in_tool_description/v0.1"
                            if exact_phrase
                            else "semantic_tokens_subset_of_tool_description/v0.1"
                        ),
                    )
                )
        if len(aliases) == 1:
            alias, tool_name, description_tokens, method = aliases[0]
            return (
                "bound",
                _tool_event_binding(alias["observation_contract"]),
                [
                    {
                        "code": "semantic_requirement_bound_to_same_branch_tool",
                        "aliased_requirement_id": alias["requirement_id"],
                        "method": method,
                        "semantic_tokens": sorted(semantic_tokens),
                        "tool_description_tokens": description_tokens,
                        "tool_name": tool_name,
                    }
                ],
            )
        source = requirement.get("source_candidates") or []
        semantic_ids = list(
            dict.fromkeys(
                (item.get("source_refs") or {}).get("semantic_unit_id")
                for item in source
                if isinstance((item.get("source_refs") or {}).get("semantic_unit_id"), str)
            )
        )
        return (
            "semantic_deferred",
            {
                "extractor_kind": "semantic_event_matcher_deferred",
                "event_stream": "canonical_tau_events",
                "semantic_unit_ids": semantic_ids,
                "requirement_text": requirement.get("requirement_text"),
                "candidate_event_kinds": ["assistant_message", "assistant_tool_call"],
            },
            [{"code": "no_exact_runtime_anchor"}],
        )

    return (
        "needs_adjudication",
        {"extractor_kind": "unbound"},
        [{"code": "unsupported_requirement_type", "requirement_type": kind}],
    )


def compile_runtime_observation_bindings(
    accepted_requirement_set: Mapping[str, Any],
    expectation_set: Mapping[str, Any],
    runtime_observation_profile: Mapping[str, Any],
    execution_matching_contracts: Mapping[str, Any],
    tool_catalog_document: Mapping[str, Any] | None = None,
    semantic_unit_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = validate_accepted_oracle_requirement_set(accepted_requirement_set)
    expectations = validate_oracle_expectation_set(expectation_set)
    profile = validate_runtime_observation_profile(runtime_observation_profile)
    operation_families = _execution_contract_index(execution_matching_contracts)
    tool_catalog = _tool_catalog_index(tool_catalog_document)
    semantic_units = _semantic_unit_index(semantic_unit_model)

    accepted_index = {
        requirement["requirement_id"]: requirement
        for branch in accepted["branches"]
        for requirement in branch["requirements"]
    }
    expectation_index = {
        expectation["requirement_id"]: expectation
        for branch in expectations["branches"]
        for expectation in branch["expectations"]
    }
    if set(accepted_index) != set(expectation_index):
        raise ThenAtomizationError("accepted requirements and expectations are not closed")
    for requirement_id, expectation in expectation_index.items():
        if expectation.get("source_requirement_fingerprint") != accepted_index[
            requirement_id
        ].get("requirement_fingerprint"):
            raise ThenAtomizationError(
                f"expectation requirement lineage mismatch: {requirement_id}"
            )

    branches = []
    status_counts: Counter[str] = Counter()
    unresolved_branches = []
    for accepted_branch in accepted["branches"]:
        branch_requirements = accepted_branch["requirements"]
        records = []
        for requirement in branch_requirements:
            expectation = expectation_index[requirement["requirement_id"]]
            status, runtime_binding, diagnostics = _binding_for_requirement(
                requirement,
                branch_context=accepted_branch["branch_context"],
                branch_requirements=branch_requirements,
                operation_families=operation_families,
                tool_catalog=tool_catalog,
                semantic_units=semantic_units,
                scope_binding_parameters=profile["scope_binding_parameters"],
            )
            record = {
                "schema_version": BINDING_VERSION,
                "binding_id": f"{requirement['requirement_id']}::RB01",
                "requirement_id": requirement["requirement_id"],
                "expectation_id": expectation["expectation_id"],
                "branch_id": accepted_branch["branch_id"],
                "binding_status": status,
                "runtime_binding": runtime_binding,
                "expected_observation": deepcopy(
                    expectation["expected_observation"]
                ),
                "diagnostics": diagnostics,
                "source_requirement_fingerprint": requirement[
                    "requirement_fingerprint"
                ],
                "source_expectation_fingerprint": expectation[
                    "expectation_fingerprint"
                ],
            }
            record["binding_fingerprint"] = content_sha256(record)
            records.append(record)
            status_counts[status] += 1
        branch_statuses = {record["binding_status"] for record in records}
        if "needs_adjudication" in branch_statuses:
            branch_status = "needs_adjudication"
        elif branch_statuses <= {"bound"}:
            branch_status = "bound"
        else:
            branch_status = "partially_bound"
            unresolved_branches.append(accepted_branch["branch_id"])
        branches.append(
            {
                "branch_id": accepted_branch["branch_id"],
                "branch_context": deepcopy(accepted_branch["branch_context"]),
                "bindings": records,
                "binding_status": branch_status,
                "potential_overlap_hints": deepcopy(
                    accepted_branch.get("potential_overlap_hints") or []
                ),
            }
        )

    binding_count = sum(len(branch["bindings"]) for branch in branches)
    result = {
        "schema_version": BINDING_SET_VERSION,
        "source_accepted_set_fingerprint": accepted["accepted_set_fingerprint"],
        "source_expectation_set_fingerprint": expectations[
            "expectation_set_fingerprint"
        ],
        "runtime_observation_profile_id": profile["profile_id"],
        "runtime_observation_profile_fingerprint": content_sha256(profile),
        "execution_matching_contracts_fingerprint": content_sha256(
            execution_matching_contracts
        ),
        "tool_catalog_fingerprint": (
            content_sha256(tool_catalog_document)
            if tool_catalog_document is not None
            else None
        ),
        "semantic_unit_model_fingerprint": (
            content_sha256(semantic_unit_model)
            if semantic_unit_model is not None
            else None
        ),
        "canonical_event_contract": deepcopy(profile["canonical_event_stream"]),
        "raw_message_adapter": deepcopy(profile["raw_message_adapter"]),
        "branches": branches,
        "summary": {
            "branch_count": len(branches),
            "binding_count": binding_count,
            "binding_status_counts": {
                status: status_counts[status] for status in sorted(BINDING_STATUSES)
            },
            "fully_bound_branch_count": sum(
                branch["binding_status"] == "bound" for branch in branches
            ),
            "partially_bound_branch_count": sum(
                branch["binding_status"] == "partially_bound" for branch in branches
            ),
            "adjudication_branch_count": sum(
                branch["binding_status"] == "needs_adjudication" for branch in branches
            ),
        },
        "unresolved_branches": unresolved_branches,
        "next_stage": "evaluator_contract_generation",
    }
    result["binding_set_fingerprint"] = content_sha256(result)
    return result


def validate_runtime_observation_binding_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$runtime_bindings")))
    fingerprint = result.pop("binding_set_fingerprint", None)
    if (
        result.get("schema_version") != BINDING_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid runtime observation binding set")
    identities = []
    statuses: Counter[str] = Counter()
    for branch in result.get("branches") or []:
        for raw in branch.get("bindings") or []:
            record = deepcopy(dict(_mapping(raw, "$.bindings[]")))
            record_fingerprint = record.pop("binding_fingerprint", None)
            if (
                record.get("schema_version") != BINDING_VERSION
                or record_fingerprint != content_sha256(record)
            ):
                raise ThenAtomizationError("invalid runtime observation binding")
            if record.get("branch_id") != branch.get("branch_id"):
                raise ThenAtomizationError("runtime observation binding branch mismatch")
            if record.get("binding_status") not in BINDING_STATUSES:
                raise ThenAtomizationError("invalid runtime observation binding status")
            identities.append(record.get("binding_id"))
            statuses[record["binding_status"]] += 1
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("runtime observation binding IDs must be unique")
    summary = _mapping(result.get("summary"), "$.summary")
    if summary.get("binding_count") != len(identities):
        raise ThenAtomizationError("runtime observation binding count mismatch")
    expected = {status: statuses[status] for status in sorted(BINDING_STATUSES)}
    if summary.get("binding_status_counts") != expected:
        raise ThenAtomizationError("runtime observation binding status counts mismatch")
    result["binding_set_fingerprint"] = fingerprint
    return result


def normalize_tau_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize saved tau messages into the canonical evaluator event stream."""

    events = []
    for message_index, raw_message in enumerate(messages):
        message = _mapping(raw_message, f"$.messages[{message_index}]")
        role = message.get("role")
        content = message.get("content")
        if role in {"assistant", "user"} and isinstance(content, str) and content.strip():
            events.append(
                {
                    "event_index": len(events),
                    "source_message_index": message_index,
                    "event_kind": f"{role}_message",
                    "content": content,
                }
            )
        if role == "assistant":
            for raw_call in message.get("tool_calls") or []:
                call = _mapping(raw_call, "$.message.tool_calls[]")
                function = call.get("function")
                nested = function if isinstance(function, Mapping) else {}
                name = call.get("name") or nested.get("name")
                arguments = call.get("arguments")
                if arguments is None:
                    arguments = nested.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ThenAtomizationError(
                            f"tau tool call arguments are invalid JSON: {exc}"
                        ) from exc
                if not isinstance(name, str) or not isinstance(arguments, Mapping):
                    raise ThenAtomizationError("tau assistant tool call is malformed")
                events.append(
                    {
                        "event_index": len(events),
                        "source_message_index": message_index,
                        "event_kind": "assistant_tool_call",
                        "tool_call_id": call.get("id"),
                        "tool_name": name,
                        "arguments": deepcopy(dict(arguments)),
                    }
                )
        if role == "tool":
            events.append(
                {
                    "event_index": len(events),
                    "source_message_index": message_index,
                    "event_kind": "tool_result",
                    "tool_call_id": message.get("id") or message.get("tool_call_id"),
                    "content": content,
                }
            )
    return events


def normalize_tau_execution(execution: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize messages and attach deterministic Driver action identities."""

    messages = execution.get("messages")
    if not isinstance(messages, list):
        raise ThenAtomizationError("$.execution.messages must be an array")
    events = normalize_tau_messages(messages)
    runtime_driver = execution.get("runtime_driver") or {}
    if not isinstance(runtime_driver, Mapping):
        raise ThenAtomizationError("$.execution.runtime_driver must be an object")
    actions = []
    transport = runtime_driver.get("transport_action_log")
    if isinstance(transport, list):
        for item in transport:
            if isinstance(item, Mapping):
                actions.append(
                    {
                        "action_kind": item.get("action_kind"),
                        "content": item.get("content"),
                    }
                )
    else:
        action_log = (
            ((runtime_driver.get("runtime_session") or {}).get("state") or {}).get(
                "action_log"
            )
            or []
        )
        for item in action_log:
            if not isinstance(item, Mapping):
                continue
            action = item.get("action") or {}
            realized = item.get("realized") or {}
            message = realized.get("message") or {}
            actions.append(
                {
                    "action_kind": action.get("action_kind"),
                    "content": message.get("content"),
                }
            )
    user_events = [event for event in events if event.get("event_kind") == "user_message"]
    cursor = 0
    for action in actions:
        content = action.get("content")
        kind = action.get("action_kind")
        if not isinstance(content, str) or not isinstance(kind, str):
            continue
        while cursor < len(user_events) and user_events[cursor].get("content") != content:
            cursor += 1
        if cursor >= len(user_events):
            break
        user_events[cursor]["driver_action_kind"] = kind
        cursor += 1
    return events


def _path(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _event_matches_filter(event: Mapping[str, Any], event_filter: Mapping[str, Any]) -> bool:
    if event.get("event_kind") != event_filter.get("event_kind"):
        return False
    return not any(
        _path(event, field) != expected
        for field, expected in (event_filter.get("field_equals") or {}).items()
    )


def _scope_matches(
    event: Mapping[str, Any],
    constraints: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    missing = []
    for scope in constraints:
        expected_ref = str(scope.get("expected_binding_ref") or "")
        key = expected_ref.removeprefix("driver_bindings.")
        if key not in driver_bindings:
            missing.append(key)
            continue
        if _path(event, str(scope.get("actual_path"))) != driver_bindings[key]:
            return False, missing
    return not missing, missing


def _scalar_leaves(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [leaf for child in value.values() for leaf in _scalar_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _scalar_leaves(child)]
    if value is None:
        return []
    return [value]


def _surface(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return " ".join(_TOKEN.findall(str(value).casefold().replace("_", " ")))


def _argument_disclosure_match(
    left: Mapping[str, Any], right: Mapping[str, Any], matcher: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    content = _surface(left.get("content"))
    required = []
    missing_paths = []
    for path in matcher.get("required_argument_paths") or []:
        value = _path(right, path)
        if value is None:
            missing_paths.append(path)
            continue
        required.extend(
            {"path": path, "value": leaf, "surface": _surface(leaf)}
            for leaf in _scalar_leaves(value)
        )
    missing_values = [
        item for item in required if item["surface"] and item["surface"] not in content
    ]
    passed = not missing_paths and not missing_values and bool(required)
    return passed, {
        "required_value_count": len(required),
        "missing_argument_paths": missing_paths,
        "missing_values": missing_values,
    }


def _extract_temporal_observations(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if runtime.get("relation") != "precedes":
        return {"status": "unavailable", "reason": "unsupported relation", "matches": []}
    left_contract = _mapping(runtime.get("left_event"), "$.left_event")
    right_contract = _mapping(runtime.get("right_event"), "$.right_event")
    if (
        left_contract.get("binding_kind") == "semantic_event_deferred"
        or right_contract.get("binding_status") != "bound"
    ):
        return {"status": "unavailable", "reason": "temporal side is unbound", "matches": []}
    right_filter = _mapping(right_contract.get("event_filter"), "$.right_event.event_filter")
    right_events = []
    missing_bindings = []
    for event in events:
        if not _event_matches_filter(event, right_filter):
            continue
        scope_ok, missing = _scope_matches(
            event, right_contract.get("scope_constraints") or [], driver_bindings
        )
        missing_bindings.extend(missing)
        if scope_ok:
            right_events.append(event)
    if missing_bindings:
        return {
            "status": "missing_driver_bindings",
            "missing_driver_bindings": sorted(set(missing_bindings)),
            "matches": [],
        }
    left_filter = _mapping(left_contract.get("event_filter"), "$.left_event.event_filter")
    matches = []
    unmatched_targets = []
    for right in right_events:
        candidates = [
            event
            for event in events
            if event.get("event_index", -1) < right.get("event_index", -1)
            and _event_matches_filter(event, left_filter)
        ]
        witness = None
        witness_evidence = None
        matcher = left_contract.get("matcher") or {}
        for left in reversed(candidates):
            if matcher.get("kind") == "contains_right_call_argument_values":
                passed, evidence = _argument_disclosure_match(left, right, matcher)
                if not passed:
                    continue
                witness_evidence = evidence
            witness = left
            break
        if witness is None:
            unmatched_targets.append(right.get("event_index"))
            continue
        matches.append(
            {
                "event_index": right.get("event_index"),
                "source_message_index": right.get("source_message_index"),
                "value": {
                    "relation": "precedes",
                    "left_event": deepcopy(dict(witness)),
                    "right_event": deepcopy(dict(right)),
                    "matcher_evidence": witness_evidence,
                },
            }
        )
    return {
        "status": "observed",
        "target_event_count": len(right_events),
        "matched_target_count": len(matches),
        "unmatched_target_event_indices": unmatched_targets,
        "matches": matches,
    }


def extract_bound_observations(
    binding: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute only fully bound event-filter extraction, without judging verdicts."""

    if binding.get("binding_status") != "bound":
        return {
            "status": "unavailable",
            "reason": f"binding_status={binding.get('binding_status')}",
            "matches": [],
        }
    runtime = _mapping(binding.get("runtime_binding"), "$.runtime_binding")
    if runtime.get("extractor_kind") == "temporal_relation":
        return _extract_temporal_observations(runtime, events, driver_bindings)
    if runtime.get("extractor_kind") != "event_filter":
        return {"status": "unavailable", "reason": "unsupported extractor", "matches": []}
    event_filter = _mapping(runtime.get("event_filter"), "$.event_filter")
    missing_bindings = []
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        scope_ok, missing = _scope_matches(
            event, runtime.get("scope_constraints") or [], driver_bindings
        )
        missing_bindings.extend(missing)
        if not scope_ok:
            continue
        projection = runtime.get("projection") or {}
        if projection.get("kind") == "event":
            value = deepcopy(dict(event))
        elif projection.get("kind") == "field":
            value = _path(event, projection["path"])
        elif projection.get("kind") == "field_tuple":
            value = {
                path: _path(event, path) for path in projection.get("paths") or []
            }
        else:
            return {"status": "unavailable", "reason": "unsupported projection", "matches": []}
        matcher = runtime.get("matcher") or {}
        if matcher.get("kind") == "contains_literal":
            literal = matcher.get("literal")
            if not isinstance(value, str) or not isinstance(literal, str):
                continue
            if matcher.get("case_sensitive", True):
                if literal not in value:
                    continue
            elif literal.casefold() not in value.casefold():
                continue
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": value,
            }
        )
    if missing_bindings:
        return {
            "status": "missing_driver_bindings",
            "missing_driver_bindings": sorted(set(missing_bindings)),
            "matches": [],
        }
    return {"status": "observed", "matches": matches}


def compile_runtime_observation_bindings_file(
    *,
    accepted_requirements_path: str | Path,
    expectations_path: str | Path,
    runtime_observation_profile_path: str | Path,
    execution_matching_contracts_path: str | Path,
    tool_catalog_path: str | Path | None = None,
    semantic_unit_model_path: str | Path | None = None,
    output_path: str | Path,
) -> dict[str, Any]:
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    result = compile_runtime_observation_bindings(
        load(accepted_requirements_path),
        load(expectations_path),
        load(runtime_observation_profile_path),
        load(execution_matching_contracts_path),
        load(tool_catalog_path) if tool_catalog_path is not None else None,
        load(semantic_unit_model_path) if semantic_unit_model_path is not None else None,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
