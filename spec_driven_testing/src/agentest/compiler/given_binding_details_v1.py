"""Prepare the second, detail-level decisions for partial Given bindings.

This layer does not ask a model to invent a Driver contract from prose.  It
mechanically exposes accepted policy lines, typed observable endpoints, and
closed candidate relations/scenarios.  A later decision selects only among
those supplied candidates.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import (
    CAP_AGENT_TOOL_SCHEMAS,
    CAP_TOOL_OBSERVABLE_ENDPOINTS,
    content_sha256,
    validate_accepted_artifact_catalog,
)
from .given_evidence_binding_v1 import validate_given_evidence_binding_set
from .then_atomization import ThenAtomizationError


TASK_NAME = "given_binding_detail"
PACKET_SET_VERSION = "agentspectesting.given-binding-detail-packet-set/v0.1"
PACKET_VERSION = "agentspectesting.given-binding-detail-packet/v0.1"
RESULT_SET_VERSION = "agentspectesting.given-binding-detail-result-set/v0.1"
DETAIL_TYPES = frozenset(
    {
        "fixture_relation_detail",
        "binary_operand_detail",
        "collection_operand_detail",
        "membership_operand_detail",
        "policy_evaluator_detail",
        "status_semantics_detail",
        "capability_scenario_detail",
        "component_requirement_detail",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "with", "user", "agent",
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _document(catalog: Mapping[str, Any], capability: str) -> Mapping[str, Any]:
    providers = catalog["capability_index"].get(capability) or []
    if len(providers) != 1:
        raise ThenAtomizationError(
            f"accepted artifact capability {capability!r} must have one provider"
        )
    return _mapping(catalog["artifacts"][providers[0]]["document"], capability)


def _tokens(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    result = set()
    for token in _TOKEN_RE.findall(value.lower().replace("_", " ")):
        variants = {token}
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            variants.add(token[:-1])
        if len(token) > 5 and token.endswith("ed"):
            variants.update({token[:-2], token[:-1]})
        if len(token) > 6 and token.endswith("ing"):
            variants.update({token[:-3], token[:-3] + "e"})
        result.update(item for item in variants if item not in _STOP)
    return result


def _system_policy_lines(agent_spec: Mapping[str, Any]) -> list[str]:
    prompt = agent_spec.get("system_prompt")
    if isinstance(prompt, Mapping):
        prompt = prompt.get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ThenAtomizationError("accepted system prompt is missing")
    return [line.strip(" #-\t") for line in prompt.splitlines() if line.strip(" #-\t")]


def _tool_inventory(tool_catalog: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tools = []
    endpoints = []
    for tool in tool_catalog.get("tools") or []:
        source_text = str((tool.get("implementation") or {}).get("source_text") or "")
        match = re.search(r"ToolType\.(READ|WRITE)", source_text)
        access_mode = match.group(1).lower() if match else "unknown"
        tool_name = tool.get("tool_name")
        tools.append(
            {
                "tool_name": tool_name,
                "description": tool.get("description") or "",
                "access_mode": access_mode,
            }
        )
        for endpoint in tool.get("observable_endpoints") or []:
            item = {
                "candidate_id": endpoint.get("endpoint_id"),
                "endpoint_id": endpoint.get("endpoint_id"),
                "source_kind": endpoint.get("source_kind"),
                "tool_name": endpoint.get("tool_name") or tool_name,
                "source_path": endpoint.get("source_path"),
                "value_type": endpoint.get("type"),
                "description": endpoint.get("description") or "",
                "allowed_values": deepcopy(endpoint.get("allowed_values") or []),
                "access_mode": access_mode,
            }
            if isinstance(item["candidate_id"], str) and item["candidate_id"]:
                endpoints.append(item)
    return tools, endpoints


def _relevant_endpoints(
    endpoints: list[dict[str, Any]], query_text: str, *, limit: int = 30
) -> list[dict[str, Any]]:
    query = _tokens(query_text)
    scored = []
    for endpoint in endpoints:
        text = " ".join(
            str(endpoint.get(field) or "")
            for field in ("candidate_id", "tool_name", "source_path", "description")
        )
        text += " " + " ".join(str(item) for item in endpoint["allowed_values"])
        overlap = query & _tokens(text)
        if overlap:
            read_bonus = 1 if endpoint["access_mode"] == "read" else 0
            scored.append((-len(overlap), -read_bonus, endpoint["candidate_id"], endpoint))
    return [deepcopy(item) for _, _, _, item in sorted(scored)[:limit]]


def _relation_operator(condition: str, shape: str) -> str | None:
    text = condition.lower()
    if shape == "binary_value_comparison":
        for phrases, operator in (
            (("higher than", "greater than", "more than"), "gt"),
            (("lower than", "less than"), "lt"),
            (("at least",), "ge"),
            (("at most",), "le"),
            (("equal to", "equals"), "eq"),
            (("not equal", "differs"), "ne"),
        ):
            if any(phrase in text for phrase in phrases):
                return operator
    if shape == "collection_count_comparison":
        if any(phrase in text for phrase in ("changes", "change", "different")):
            return "count_ne"
        if any(phrase in text for phrase in ("unchanged", "remains the same", "same number")):
            return "count_eq"
    if shape == "set_membership":
        if "at least one" in text and ("not in" in text or "not in the" in text):
            return "any_not_in"
        if "not in" in text:
            return "not_in"
        if " in " in f" {text} ":
            return "in"
    return None


def _operand_candidates(
    endpoints: list[dict[str, Any]], query_text: str, shape: str, *, focus_text: str
) -> list[dict[str, Any]]:
    query_tokens = _tokens(query_text)
    focus_tokens = _tokens(focus_text)
    eligible = []
    for endpoint in endpoints:
        path = str(endpoint.get("source_path") or "").lower()
        description_tokens = _tokens(endpoint.get("description"))
        is_post_write_return = (
            endpoint.get("access_mode") == "write"
            and endpoint.get("source_kind") == "tool_return"
        )
        if is_post_write_return:
            continue
        if shape == "binary_value_comparison":
            target_value_words = {
                word for word in ("price", "amount") if word in focus_tokens
            } or {"price", "amount"}
            if not any(word in path for word in target_value_words):
                continue
        elif shape == "collection_count_comparison":
            if endpoint.get("value_type") != "array":
                continue
            if "passenger" in query_tokens and "passenger" not in (
                _tokens(path) | description_tokens
            ):
                continue
        elif shape == "set_membership":
            if endpoint.get("source_kind") == "tool_argument":
                if not any(word in path for word in ("origin", "destination", "airport")):
                    continue
            elif endpoint.get("source_kind") == "tool_return":
                if not any(word in path for word in ("iata", "airport", "returns")):
                    continue
            else:
                continue
        eligible.append(endpoint)
    relevant = _relevant_endpoints(eligible, query_text, limit=32)
    if shape == "set_membership":
        value_items = [item for item in relevant if item["source_kind"] == "tool_argument"]
        if value_items:
            best_tool_overlap = max(
                len(_tokens(item["tool_name"]) & query_tokens) for item in value_items
            )
            value_items = [
                item
                for item in value_items
                if len(_tokens(item["tool_name"]) & query_tokens) == best_tool_overlap
            ]
        set_items = [
            item
            for item in eligible
            if item["source_kind"] == "tool_return"
            and "list" in str(item["tool_name"]).lower()
            and (
                "iata" in str(item["source_path"]).lower()
                or "airport" in (str(item["tool_name"]) + " " + item["description"]).lower()
            )
        ]
        scalar_set_items = [
            item for item in set_items if item.get("value_type") in {"string", "integer", "number"}
        ]
        if scalar_set_items:
            best_field_overlap = max(
                len(
                    (_tokens(item.get("source_path")) | _tokens(item.get("description")))
                    & focus_tokens
                )
                for item in scalar_set_items
            )
            set_items = [
                item
                for item in scalar_set_items
                if len(
                    (_tokens(item.get("source_path")) | _tokens(item.get("description")))
                    & focus_tokens
                )
                == best_field_overlap
            ]
        set_items.sort(
            key=lambda item: (
                0 if "iata" in str(item["source_path"]).lower() else 1,
                item["endpoint_id"],
            )
        )
        relevant = value_items + set_items
    result = []
    for endpoint in relevant:
        path = str(endpoint.get("source_path") or "")
        value_type = endpoint.get("value_type")
        reducers = ["identity"]
        if shape == "binary_value_comparison" and "[]" in path:
            reducers = ["sum_selected_values" if ".prices.*" in path else "sum"]
        elif value_type == "array" or "[]" in path:
            reducers.append("count")
        if (
            shape != "binary_value_comparison"
            and value_type in {"integer", "number"}
            and "[]" in path
        ):
            reducers.append("sum")
        for reducer in reducers:
            if shape == "collection_count_comparison" and reducer != "count":
                continue
            if shape == "set_membership" and reducer not in {"identity"}:
                continue
            candidate = {
                "operand_id": f"{endpoint['candidate_id']}::{reducer}",
                "endpoint_id": endpoint["endpoint_id"],
                "source_kind": endpoint["source_kind"],
                "source_path": endpoint["source_path"],
                "tool_name": endpoint["tool_name"],
                "access_mode": endpoint["access_mode"],
                "reducer": reducer,
                "description": endpoint["description"],
            }
            if shape == "set_membership":
                candidate["operand_role"] = (
                    "membership_value"
                    if endpoint["source_kind"] == "tool_argument"
                    else "authoritative_set"
                )
            result.append(candidate)
    if shape == "binary_value_comparison":
        price_sources = []
        seen_price_endpoints = set()
        for item in result:
            if item["endpoint_id"] in seen_price_endpoints:
                continue
            seen_price_endpoints.add(item["endpoint_id"])
            if any(word in item["source_path"].lower() for word in ("price", "amount")):
                price_sources.append(item)
        passenger_counts = []
        for endpoint in endpoints:
            path = str(endpoint.get("source_path") or "").lower()
            if (
                "passenger" in path
                and endpoint.get("value_type") == "array"
                and (path.endswith("passengers") or path.endswith("saved_passengers"))
                and endpoint.get("access_mode") == "read"
                and endpoint.get("source_kind") == "tool_return"
                and not (
                    endpoint.get("access_mode") == "write"
                    and endpoint.get("source_kind") == "tool_return"
                )
            ):
                passenger_counts.append(endpoint)
        if passenger_counts:
            best_overlap = max(
                len(_tokens(item["tool_name"]) & query_tokens)
                for item in passenger_counts
            )
            passenger_counts = [
                item
                for item in passenger_counts
                if len(_tokens(item["tool_name"]) & query_tokens) == best_overlap
            ]
        composites = []
        for price in price_sources:
            for passengers in passenger_counts:
                composites.append(
                    {
                        "operand_id": (
                            f"total::{price['endpoint_id']}::{passengers['endpoint_id']}"
                        ),
                        "expression": {
                            "op": "multiply",
                            "args": [
                                {
                                    "op": (
                                        "sum_selected_values"
                                        if ".prices.*" in price["source_path"]
                                        else "sum"
                                    ),
                                    "endpoint_id": price["endpoint_id"],
                                },
                                {
                                    "op": "count",
                                    "endpoint_id": passengers["endpoint_id"],
                                },
                            ],
                        },
                        "description": "total itinerary price for all passengers",
                    }
                )
        # Complete price formulas are more useful than their partial factors.
        unique = []
        seen = set()
        for item in composites + result:
            if item["operand_id"] not in seen:
                seen.add(item["operand_id"])
                unique.append(item)
        result = unique
    return result[:32]


def _fixture_relation_options(locator_id: str, endpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    options = []
    if locator_id.startswith("fixture_root::"):
        root = locator_id.split("::", 1)[1]
        for relation in ("key_absent", "key_present"):
            options.append(
                {
                    "candidate_id": f"{locator_id}::{relation}",
                    "contract": {
                        "kind": "fixture_collection_relation",
                        "collection_root": root,
                        "relation": relation,
                        "key_source": "request_entity_identifier",
                    },
                }
            )
    else:
        endpoint = next(
            (item for item in endpoints if item["endpoint_id"] == locator_id), None
        )
        if endpoint is None:
            raise ThenAtomizationError(f"selected locator is not in accepted catalog: {locator_id}")
        values = endpoint["allowed_values"]
        for value in values:
            for relation in ("eq", "ne"):
                options.append(
                    {
                        "candidate_id": f"{locator_id}::{relation}::{value}",
                        "contract": {
                            "kind": "observable_value_relation",
                            "endpoint_id": locator_id,
                            "relation": relation,
                            "value": value,
                        },
                    }
                )
    options.append(
        {
            "candidate_id": "insufficient",
            "contract": None,
            "meaning": "none of the supplied relation contracts makes the exact condition true",
        }
    )
    return options


def _policy_evidence_candidates(
    endpoints: list[dict[str, Any]], query_text: str
) -> list[dict[str, Any]]:
    query = _tokens(query_text)
    selected = _relevant_endpoints(endpoints, query_text, limit=16)
    seen = {item["endpoint_id"] for item in selected}
    # Lexical field ranking can crowd out a semantically relevant read tool whose
    # return is intentionally generic (for example returns:string).  Preserve one
    # root return from every query-mentioned read tool as a retrieval backstop.
    backstops = []
    for endpoint in endpoints:
        if (
            endpoint["access_mode"] != "read"
            or endpoint["source_kind"] != "tool_return"
            or str(endpoint["source_path"]) != "returns"
            or not (_tokens(endpoint["tool_name"]) & query)
            or endpoint["endpoint_id"] in seen
        ):
            continue
        backstops.append(deepcopy(endpoint))
        seen.add(endpoint["endpoint_id"])
    return (selected + sorted(backstops, key=lambda item: item["endpoint_id"]))[:24]


def _capability_scenarios(
    polarity: str,
    tools: list[dict[str, Any]],
    policy_lines: list[str],
) -> list[dict[str, Any]]:
    if polarity == "request_in_scope":
        options = [
            {
                "candidate_id": f"tool_action::{tool['tool_name']}",
                "scenario_kind": "declared_tool_action_request",
                "tool_name": tool["tool_name"],
                "description": tool["description"],
            }
            for tool in tools
            if tool["tool_name"] not in {"calculate", "transfer_to_human_agents"}
        ]
    else:
        options = []
        for ordinal, line in enumerate(policy_lines, 1):
            lower = line.lower()
            if (
                "transfer" in lower
                and ("cannot help" in lower or "transfer is needed" in lower)
                and "if and only if" not in lower
            ):
                section = None
                for prior in reversed(policy_lines[max(0, ordinal - 12) : ordinal - 1]):
                    if (
                        1 <= len(prior.split()) <= 5
                        and not prior.rstrip().endswith((".", "!", "?"))
                    ):
                        section = prior.rstrip(":")
                        break
                options.append(
                    {
                        "candidate_id": f"policy_transfer_scenario::{ordinal:03d}",
                        "scenario_kind": "policy_grounded_transfer_request",
                        "request_action": section,
                        "required_precondition": line,
                        "policy_line": line,
                    }
                )
    options.append(
        {
            "candidate_id": "insufficient",
            "scenario_kind": "non_binding_outcome",
            "meaning": "accepted policy does not supply a concrete witness scenario",
        }
    )
    return options


def _user_fact_candidates(policy_lines: list[str]) -> list[dict[str, Any]]:
    values = []
    for line in policy_lines:
        fragments = []
        fragments.extend(
            match.group(1)
            for match in re.finditer(r"given\s+([^.;]+?)\s+reasons?", line, re.I)
        )
        fragments.extend(re.findall(r"reason[^()]*(?:\(([^)]+)\))", line, re.I))
        for fragment in fragments:
            cleaned = re.sub(r"\b(?:and|or)\b", ",", fragment, flags=re.I)
            for value in cleaned.split(","):
                value = value.strip(" .;:-")
                if value and value.lower() not in {item.lower() for item in values}:
                    values.append(value)
    return [
        {
            "user_fact_id": f"USER-FACT::{index:02d}",
            "fact": f"The user's cancellation reason is {value}.",
        }
        for index, value in enumerate(values, 1)
    ]


def _component_contract_options(
    condition: str, policy_lines: list[str]
) -> list[dict[str, Any]]:
    facts = _user_fact_candidates(policy_lines)
    covered_values = []
    defining_line_ids = []
    for index, line in enumerate(policy_lines, 1):
        matches = list(re.finditer(r"given\s+([^.;]+?)\s+reasons?", line, re.I))
        if not matches:
            continue
        defining_line_ids.append(f"POLICY::{index:02d}")
        for match in matches:
            cleaned = re.sub(r"\b(?:and|or)\b", ",", match.group(1), flags=re.I)
            covered_values.extend(
                item.strip(" .;:-").lower()
                for item in cleaned.split(",")
                if item.strip(" .;:-")
            )
    covered = set(covered_values)
    negative = bool(re.search(r"\bnot\b", condition, re.I))
    options = []
    for fact in facts:
        value = fact["fact"].removeprefix("The user's cancellation reason is ").rstrip(".").lower()
        satisfies = value not in covered if negative else value in covered
        if not satisfies or not covered or not defining_line_ids:
            continue
        relation = "not_in" if negative else "in"
        options.append(
            {
                "candidate_id": f"COMPONENT::{fact['user_fact_id']}::{relation}",
                "contract": {
                    "user_fact": deepcopy(fact),
                    "policy_line_ids": defining_line_ids,
                    "derived_evaluator": {
                        "op": relation,
                        "value_ref": "user_fact.cancellation_reason",
                        "set": sorted(covered),
                    },
                },
            }
        )
    options.append(
        {
            "candidate_id": "insufficient",
            "contract": None,
            "meaning": "no supplied complete component contract makes the exact condition true",
        }
    )
    return options


def build_given_binding_detail_packets(
    binding_set: Mapping[str, Any], accepted_artifact_catalog: Mapping[str, Any]
) -> dict[str, Any]:
    bindings = validate_given_evidence_binding_set(binding_set)
    catalog = validate_accepted_artifact_catalog(accepted_artifact_catalog)
    tool_catalog = _document(catalog, CAP_TOOL_OBSERVABLE_ENDPOINTS)
    agent_spec = _document(catalog, CAP_AGENT_TOOL_SCHEMAS)
    tools, endpoints = _tool_inventory(tool_catalog)
    policy_lines = _system_policy_lines(agent_spec)
    packets = []
    for binding in bindings["bindings"]:
        if binding["binding_status"] != "partially_bound":
            continue
        requirement = binding["base_requirement"]
        condition = requirement["condition_statement"]
        prior_answer = binding["binding_decision"]["answer"]
        common = {
            "condition": condition,
            "when": binding["when"],
            "selected_narrow_decision": deepcopy(prior_answer),
            "temporal_rule": "all evidence must exist before the When starts",
        }
        task = binding["next_binding_task"]
        if task == "fixture_relation_binding":
            detail_type = "fixture_relation_detail"
            locator_id = prior_answer["selected_candidate_id"]
            model_input = {
                **common,
                "question": "Which supplied relation contract makes the exact condition true?",
                "candidate_contracts": _fixture_relation_options(locator_id, endpoints),
                "answer_contract": {
                    "selected_candidate_id": "exactly one candidate_id",
                    "reason": "one concise sentence",
                },
            }
        elif task == "evaluator_operand_binding":
            shape = prior_answer["decision"]
            accepted_lines = deepcopy(requirement.get("candidate_policy_lines") or [])
            query = " ".join([condition, binding["when"], *accepted_lines])
            operand_candidates = _operand_candidates(
                endpoints, query, shape, focus_text=condition
            )
            operator = _relation_operator(condition, shape)
            if shape == "binary_value_comparison":
                detail_type = "binary_operand_detail"
                question = "Which supplied operands are the left and right values in the comparison?"
                answer_contract = {
                    "left_operand_id": "one supplied operand_id",
                    "right_operand_id": "one different supplied operand_id",
                    "reason": "one concise sentence",
                }
            elif shape == "collection_count_comparison":
                detail_type = "collection_operand_detail"
                question = "Which supplied collections are the before and proposed counts?"
                answer_contract = {
                    "left_operand_id": "one supplied count operand_id",
                    "right_operand_id": "one different supplied count operand_id",
                    "reason": "one concise sentence",
                }
            elif shape == "set_membership":
                detail_type = "membership_operand_detail"
                question = "Which supplied values and authoritative set decide membership?"
                answer_contract = {
                    "value_operand_ids": "one or more supplied operand_id values",
                    "set_operand_id": "one different supplied operand_id",
                    "reason": "one concise sentence",
                }
            elif shape == "policy_eligibility":
                status_endpoint = next(
                    (
                        item
                        for item in endpoints
                        if item["access_mode"] == "read"
                        and item["source_kind"] == "tool_return"
                        and "status" in str(item["endpoint_id"]).lower()
                        and item["allowed_values"]
                        and "flight" in str(item["tool_name"]).lower()
                    ),
                    None,
                )
                flown_line = next(
                    (
                        (index, line)
                        for index, line in enumerate(accepted_lines, 1)
                        if "flown" in line.lower()
                    ),
                    None,
                )
                if status_endpoint is not None and flown_line is not None:
                    detail_type = "status_semantics_detail"
                    question = (
                        "Which supplied flight-status values mean the flight has already "
                        "started or completed flying?"
                    )
                    answer_contract = {
                        "blocking_status_values": "one or more supplied allowed_status_values",
                        "reason": "one concise sentence",
                    }
                else:
                    detail_type = "policy_evaluator_detail"
                    question = "Which supplied policy lines and evidence endpoints define this eligibility test?"
                    answer_contract = {
                        "decision": "bound or insufficient",
                        "policy_line_ids": "one or more supplied IDs when bound; otherwise []",
                        "evidence_ids": "supplied endpoint IDs needed by the rule; may be []",
                        "reason": "one concise sentence",
                    }
            else:
                raise ThenAtomizationError(f"unsupported evaluator shape: {shape}")
            model_input = {
                **common,
                "question": question,
                "fixed_operator": operator,
                "exactness_rule": (
                    "A bound answer must define the exact condition, not merely list "
                    "unrelated examples or alternative ways a similar outcome can occur."
                ),
                "policy_lines": [
                    {"policy_line_id": f"POLICY::{index:02d}", "text": line}
                    for index, line in enumerate(accepted_lines, 1)
                ],
                "answer_contract": answer_contract,
            }
            if shape == "policy_eligibility":
                if detail_type == "status_semantics_detail":
                    model_input["blocking_policy_line"] = {
                        "policy_line_id": f"POLICY::{flown_line[0]:02d}",
                        "text": flown_line[1],
                    }
                    model_input["status_endpoint"] = deepcopy(status_endpoint)
                    model_input["allowed_status_values"] = deepcopy(
                        status_endpoint["allowed_values"]
                    )
                    model_input["resulting_evaluator"] = {
                        "op": "none_in",
                        "collection_source": "reservation.flights",
                        "value_source_endpoint": status_endpoint["endpoint_id"],
                        "blocking_values_ref": "answer.blocking_status_values",
                    }
                else:
                    pre_when_endpoints = [
                        item
                        for item in endpoints
                        if not (
                            item["access_mode"] == "write"
                            and item["source_kind"] == "tool_return"
                        )
                    ]
                    model_input["evidence_candidates"] = _policy_evidence_candidates(
                        pre_when_endpoints, query
                    )
            else:
                model_input["operand_candidates"] = operand_candidates
        elif task == "capability_scenario_realization":
            detail_type = "capability_scenario_detail"
            polarity = prior_answer["decision"]
            model_input = {
                **common,
                "question": "Which supplied concrete request scenario witnesses the required scope polarity?",
                "required_polarity": polarity,
                "candidate_scenarios": _capability_scenarios(polarity, tools, policy_lines),
                "answer_contract": {
                    "selected_candidate_id": "exactly one candidate_id",
                    "reason": "one concise sentence",
                },
            }
        elif task == "component_requirement_binding":
            detail_type = "component_requirement_detail"
            accepted_lines = deepcopy(requirement.get("candidate_policy_lines") or [])
            query = " ".join([condition, *accepted_lines])
            selected_source_kinds = set(prior_answer["source_kinds"])
            evidence = (
                _relevant_endpoints(endpoints, query, limit=12)
                if selected_source_kinds & {"fixture_state", "prior_runtime_event"}
                else []
            )
            model_input = {
                **common,
                "question": "Which supplied complete component contract makes the exact condition true?",
                "required_source_kinds": deepcopy(prior_answer["source_kinds"]),
                "policy_lines": [
                    {"policy_line_id": f"POLICY::{index:02d}", "text": line}
                    for index, line in enumerate(accepted_lines, 1)
                ],
                "candidate_contracts": _component_contract_options(
                    condition, accepted_lines
                ),
                "answer_contract": {
                    "selected_candidate_id": "exactly one supplied candidate_id",
                    "reason": "one concise sentence",
                },
            }
        else:
            raise ThenAtomizationError(f"unsupported detail binding task: {task}")
        packet = {
            "schema_version": PACKET_VERSION,
            "packet_id": f"GIVEN-BIND-DETAIL::{content_sha256(model_input)[:16]}::D02",
            "task_name": TASK_NAME,
            "detail_type": detail_type,
            "model_input": model_input,
            "member": {
                "binding_id": binding["binding_id"],
                "requirement_id": binding["requirement_id"],
                "branch_id": binding["branch_id"],
                "condition_id": binding["condition_id"],
                "source_binding_fingerprint": binding["binding_fingerprint"],
            },
        }
        packet["packet_fingerprint"] = content_sha256(packet)
        packets.append(packet)
    packets.sort(key=lambda item: item["packet_id"])
    counts = Counter(item["detail_type"] for item in packets)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": packets,
        "summary": {
            "packet_count": len(packets),
            "detail_type_counts": dict(sorted(counts.items())),
            "expected_model_calls": len(packets),
        },
        "source_binding_set_fingerprint": bindings["binding_set_fingerprint"],
        "source_artifact_catalog_fingerprint": catalog["catalog_fingerprint"],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_binding_detail_packet_set(result)


def validate_given_binding_detail_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_binding_detail_packet_set")))
    supplied = result.pop("packet_set_fingerprint", None)
    if (
        result.get("schema_version") != PACKET_SET_VERSION
        or result.get("task_name") != TASK_NAME
        or supplied != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given binding detail packet set")
    packet_ids = []
    binding_ids = []
    counts = Counter()
    for raw in result.get("packets") or []:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        fingerprint = packet.pop("packet_fingerprint", None)
        if (
            packet.get("schema_version") != PACKET_VERSION
            or packet.get("task_name") != TASK_NAME
            or packet.get("detail_type") not in DETAIL_TYPES
            or fingerprint != content_sha256(packet)
        ):
            raise ThenAtomizationError("invalid Given binding detail packet")
        packet_ids.append(packet.get("packet_id"))
        binding_ids.append(_mapping(packet.get("member"), "$.member").get("binding_id"))
        counts[packet["detail_type"]] += 1
    if len(packet_ids) != len(set(packet_ids)) or len(binding_ids) != len(set(binding_ids)):
        raise ThenAtomizationError("Given binding detail identities overlap")
    expected = {
        "packet_count": len(packet_ids),
        "detail_type_counts": dict(sorted(counts.items())),
        "expected_model_calls": len(packet_ids),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given binding detail summary mismatch")
    result["packet_set_fingerprint"] = supplied
    return result


def select_given_binding_detail_packets(
    packet_set: Mapping[str, Any], condition_ids: list[str]
) -> dict[str, Any]:
    packets = validate_given_binding_detail_packet_set(packet_set)
    if (
        not isinstance(condition_ids, list)
        or not condition_ids
        or len(condition_ids) != len(set(condition_ids))
        or any(not isinstance(item, str) or not item for item in condition_ids)
    ):
        raise ThenAtomizationError("Given binding detail selection IDs are invalid")
    index = {item["member"]["condition_id"]: item for item in packets["packets"]}
    missing = sorted(set(condition_ids) - set(index))
    if missing:
        raise ThenAtomizationError(f"unknown Given binding detail conditions: {missing}")
    selected = [deepcopy(index[item]) for item in condition_ids]
    counts = Counter(item["detail_type"] for item in selected)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": selected,
        "summary": {
            "packet_count": len(selected),
            "detail_type_counts": dict(sorted(counts.items())),
            "expected_model_calls": len(selected),
        },
        "source_binding_set_fingerprint": packets["source_binding_set_fingerprint"],
        "source_artifact_catalog_fingerprint": packets[
            "source_artifact_catalog_fingerprint"
        ],
        "source_full_packet_set_fingerprint": packets["packet_set_fingerprint"],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_binding_detail_packet_set(result)


def render_given_binding_detail_prompt(packet: Mapping[str, Any], template: str) -> str:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    if packet.get("task_name") != TASK_NAME or f"TASK: {TASK_NAME}" not in template:
        raise ThenAtomizationError("Given binding detail prompt/task mismatch")
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        packet["model_input"], ensure_ascii=False, indent=2
    )


def resolve_mechanical_component_detail_responses(
    packet_set: Mapping[str, Any]
) -> dict[str, Any]:
    """Choose a stable witness when the compiler already enumerated exact contracts."""

    packets = validate_given_binding_detail_packet_set(packet_set)
    responses = {}
    for packet in packets["packets"]:
        if packet["detail_type"] != "component_requirement_detail":
            continue
        candidates = [
            item
            for item in packet["model_input"].get("candidate_contracts") or []
            if item.get("contract") is not None
        ]
        selected = candidates[0]["candidate_id"] if candidates else "insufficient"
        response = {
            "selected_candidate_id": selected,
            "reason": (
                "Selected the first stable complete contract generated from accepted policy values."
                if candidates
                else "No complete component contract was generated from accepted evidence."
            ),
        }
        responses[packet["packet_id"]] = validate_given_binding_detail_response(
            packet, response
        )
    return responses


def _reason(response: Mapping[str, Any]) -> str:
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 400:
        raise ThenAtomizationError("Given binding detail reason is invalid")
    return reason.strip()


def validate_given_binding_detail_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    packet = _mapping(packet, "$packet")
    model_input = _mapping(packet.get("model_input"), "$.model_input")
    response = deepcopy(dict(_mapping(response, "$response")))
    reason = _reason(response)
    detail_type = packet.get("detail_type")
    if detail_type in {"fixture_relation_detail", "capability_scenario_detail"}:
        if set(response) != {"selected_candidate_id", "reason"}:
            raise ThenAtomizationError("Given detail choice response fields are invalid")
        field = (
            "candidate_contracts"
            if detail_type == "fixture_relation_detail"
            else "candidate_scenarios"
        )
        allowed = {item.get("candidate_id") for item in model_input.get(field) or []}
        if response.get("selected_candidate_id") not in allowed:
            raise ThenAtomizationError("Given detail candidate is not allowed")
        return {
            "selected_candidate_id": response["selected_candidate_id"],
            "reason": reason,
        }
    if detail_type in {"binary_operand_detail", "collection_operand_detail"}:
        if set(response) != {"left_operand_id", "right_operand_id", "reason"}:
            raise ThenAtomizationError("Given operand response fields are invalid")
        allowed = {
            item.get("operand_id") for item in model_input.get("operand_candidates") or []
        }
        left = response.get("left_operand_id")
        right = response.get("right_operand_id")
        if left not in allowed or right not in allowed or left == right:
            raise ThenAtomizationError("Given operand selection is invalid")
        return {"left_operand_id": left, "right_operand_id": right, "reason": reason}
    if detail_type == "membership_operand_detail":
        if set(response) != {"value_operand_ids", "set_operand_id", "reason"}:
            raise ThenAtomizationError("Given membership response fields are invalid")
        allowed = {
            item.get("operand_id") for item in model_input.get("operand_candidates") or []
        }
        roles = {
            item.get("operand_id"): item.get("operand_role")
            for item in model_input.get("operand_candidates") or []
        }
        values = response.get("value_operand_ids")
        set_operand = response.get("set_operand_id")
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(item not in allowed for item in values)
            or set_operand not in allowed
            or set_operand in values
            or any(roles.get(item) != "membership_value" for item in values)
            or roles.get(set_operand) != "authoritative_set"
        ):
            raise ThenAtomizationError("Given membership operand selection is invalid")
        return {
            "value_operand_ids": values,
            "set_operand_id": set_operand,
            "reason": reason,
        }
    if detail_type == "policy_evaluator_detail":
        if set(response) != {"decision", "policy_line_ids", "evidence_ids", "reason"}:
            raise ThenAtomizationError("Given policy evaluator response fields are invalid")
        decision = response.get("decision")
        policy_ids = response.get("policy_line_ids")
        evidence_ids = response.get("evidence_ids")
        allowed_policy = {
            item.get("policy_line_id") for item in model_input.get("policy_lines") or []
        }
        allowed_evidence = {
            item.get("endpoint_id") for item in model_input.get("evidence_candidates") or []
        }
        if decision not in {"bound", "insufficient"}:
            raise ThenAtomizationError("Given policy evaluator decision is invalid")
        if not isinstance(policy_ids, list) or len(policy_ids) != len(set(policy_ids)):
            raise ThenAtomizationError("Given policy line selection is invalid")
        if not isinstance(evidence_ids, list) or len(evidence_ids) != len(set(evidence_ids)):
            raise ThenAtomizationError("Given policy evidence selection is invalid")
        if any(item not in allowed_policy for item in policy_ids) or any(
            item not in allowed_evidence for item in evidence_ids
        ):
            raise ThenAtomizationError("Given policy evaluator selected unknown evidence")
        if (decision == "bound" and not policy_ids) or (
            decision == "insufficient" and (policy_ids or evidence_ids)
        ):
            raise ThenAtomizationError("Given policy evaluator answer is inconsistent")
        return {
            "decision": decision,
            "policy_line_ids": policy_ids,
            "evidence_ids": evidence_ids,
            "reason": reason,
        }
    if detail_type == "status_semantics_detail":
        if set(response) != {"blocking_status_values", "reason"}:
            raise ThenAtomizationError("Given status semantics response fields are invalid")
        values = response.get("blocking_status_values")
        allowed = set(model_input.get("allowed_status_values") or [])
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(item not in allowed for item in values)
        ):
            raise ThenAtomizationError("Given blocking status selection is invalid")
        return {"blocking_status_values": values, "reason": reason}
    if detail_type == "component_requirement_detail":
        if set(response) != {"selected_candidate_id", "reason"}:
            raise ThenAtomizationError("Given component response fields are invalid")
        allowed = {
            item.get("candidate_id")
            for item in model_input.get("candidate_contracts") or []
        }
        selected = response.get("selected_candidate_id")
        if selected not in allowed:
            raise ThenAtomizationError("Given component candidate is not allowed")
        return {"selected_candidate_id": selected, "reason": reason}
    raise ThenAtomizationError("unsupported Given binding detail response type")


def materialize_given_binding_detail_results(
    packet_set: Mapping[str, Any], responses: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_given_binding_detail_packet_set(packet_set)
    response_map = deepcopy(dict(_mapping(responses, "$responses")))
    if set(response_map) != {item["packet_id"] for item in packets["packets"]}:
        raise ThenAtomizationError("Given binding detail responses must cover the closed batch")
    records = []
    for packet in packets["packets"]:
        records.append(
            {
                **deepcopy(packet["member"]),
                "detail_type": packet["detail_type"],
                "answer": validate_given_binding_detail_response(
                    packet, response_map[packet["packet_id"]]
                ),
                "source_packet_fingerprint": packet["packet_fingerprint"],
            }
        )
    counts = Counter(item["detail_type"] for item in records)
    result = {
        "schema_version": RESULT_SET_VERSION,
        "results": records,
        "summary": {
            "result_count": len(records),
            "detail_type_counts": dict(sorted(counts.items())),
        },
        "source_packet_set_fingerprint": packets["packet_set_fingerprint"],
        "source_binding_set_fingerprint": packets["source_binding_set_fingerprint"],
    }
    result["result_set_fingerprint"] = content_sha256(result)
    return validate_given_binding_detail_result_set(result)


def validate_given_binding_detail_result_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_binding_detail_result_set")))
    supplied = result.pop("result_set_fingerprint", None)
    if result.get("schema_version") != RESULT_SET_VERSION or supplied != content_sha256(result):
        raise ThenAtomizationError("invalid Given binding detail result set")
    records = result.get("results")
    if not isinstance(records, list):
        raise ThenAtomizationError("Given binding detail results must be an array")
    binding_ids = []
    counts = Counter()
    for raw in records:
        item = _mapping(raw, "$.results[]")
        if item.get("detail_type") not in DETAIL_TYPES:
            raise ThenAtomizationError("Given binding detail result type is invalid")
        if not isinstance(item.get("answer"), Mapping):
            raise ThenAtomizationError("Given binding detail answer is missing")
        binding_ids.append(item.get("binding_id"))
        counts[item["detail_type"]] += 1
    if len(binding_ids) != len(set(binding_ids)):
        raise ThenAtomizationError("Given binding detail result identities overlap")
    expected = {
        "result_count": len(records),
        "detail_type_counts": dict(sorted(counts.items())),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given binding detail result summary mismatch")
    result["result_set_fingerprint"] = supplied
    return result
