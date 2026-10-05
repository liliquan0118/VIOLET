"""Prepare narrow decisions needed to bind natural-language Given requirements."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import (
    CAP_TOOL_OBSERVABLE_ENDPOINTS,
    content_sha256,
    validate_accepted_artifact_catalog,
)
from .driver_evidence_capabilities_v1 import (
    SOURCE_KINDS,
    validate_driver_evidence_capability_profile,
)
from .given_evidence_requirements_v1 import (
    validate_given_evidence_requirement_set,
)
from .then_atomization import ThenAtomizationError


TASK_NAME = "given_binding_decision"
PACKET_SET_VERSION = "agentspectesting.given-binding-decision-packet-set/v0.1"
PACKET_VERSION = "agentspectesting.given-binding-decision-packet/v0.1"
RESULT_SET_VERSION = "agentspectesting.given-binding-decision-result-set/v0.1"
DECISION_TYPES = frozenset(
    {
        "fixture_locator_choice",
        "evaluator_shape_choice",
        "capability_polarity_choice",
        "source_component_choice",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "in", "is", "it", "of", "on", "or", "otherwise",
    "that", "the", "this", "to", "was", "were", "with", "user",
}
_EVALUATOR_OPTIONS = [
    {
        "decision": "binary_value_comparison",
        "meaning": "compare one before/current value with one proposed/after value",
    },
    {
        "decision": "collection_count_comparison",
        "meaning": "compare counts or cardinalities of collections",
    },
    {
        "decision": "set_membership",
        "meaning": "test whether one or more values belong to an authoritative set",
    },
    {
        "decision": "policy_eligibility",
        "meaning": "evaluate a named policy predicate from one or more facts",
    },
    {
        "decision": "insufficient",
        "meaning": "the accepted context does not determine an evaluator shape",
    },
]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _tokens(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    result = set()
    for item in _TOKEN_RE.findall(value.lower().replace("_", " ")):
        variants = {item}
        if len(item) > 4 and item.endswith("s") and not item.endswith("ss"):
            variants.add(item[:-1])
        if len(item) > 5 and item.endswith("ed"):
            variants.update({item[:-2], item[:-1]})
        if len(item) > 6 and item.endswith("ing"):
            variants.update({item[:-3], item[:-3] + "e"})
        result.update(value for value in variants if value not in _STOP)
    return result


def _endpoint_candidate(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "candidate_id": endpoint.get("candidate_id") or endpoint.get("endpoint_id"),
        "source_kind": endpoint.get("source_kind"),
        "tool_name": endpoint.get("tool_name"),
        "source_path": endpoint.get("source_path"),
        "description": endpoint.get("description") or "",
        "allowed_values": deepcopy(endpoint.get("allowed_values") or []),
    }
    if endpoint.get("access_mode") is not None:
        result["access_mode"] = endpoint.get("access_mode")
    return result


def _merge_endpoint_candidates(
    accepted: list[Mapping[str, Any]], retrieved: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for endpoint in [*accepted, *retrieved]:
        candidate = _endpoint_candidate(endpoint)
        candidate_id = candidate["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        result.append(candidate)
    return result[:16]


def _tool_catalog(catalog: Mapping[str, Any]) -> Mapping[str, Any]:
    providers = catalog["capability_index"].get(CAP_TOOL_OBSERVABLE_ENDPOINTS) or []
    if len(providers) != 1:
        raise ThenAtomizationError("tool observable catalog must have one provider")
    return _mapping(
        catalog["artifacts"][providers[0]]["document"],
        "$.tool_observable_catalog",
    )


def _relevant_endpoints(
    tool_catalog: Mapping[str, Any], condition: str, *, returns_only: bool
) -> list[dict[str, Any]]:
    query = _tokens(condition)
    scored = []
    for tool in tool_catalog.get("tools") or []:
        source_text = str((tool.get("implementation") or {}).get("source_text") or "")
        mode_match = re.search(r"ToolType\.(READ|WRITE)", source_text)
        access_mode = mode_match.group(1).lower() if mode_match else "unknown"
        for endpoint in tool.get("observable_endpoints") or []:
            if returns_only and endpoint.get("source_kind") != "tool_return":
                continue
            if returns_only and access_mode == "write":
                continue
            text = " ".join(
                str(endpoint.get(field) or "")
                for field in (
                    "endpoint_id", "tool_name", "source_path", "description"
                )
            )
            text += " " + " ".join(str(item) for item in endpoint.get("allowed_values") or [])
            overlap = query & _tokens(text)
            if overlap:
                enriched = deepcopy(dict(endpoint))
                enriched["access_mode"] = access_mode
                scored.append(
                    (-len(overlap), str(endpoint.get("endpoint_id")), enriched)
                )
    result = []
    for _, _, endpoint in sorted(scored)[:16]:
        result.append(_endpoint_candidate(endpoint))
    return result


def _fixture_support_index(value: Any) -> tuple[dict[str, dict[str, Any]], str]:
    if not isinstance(value, list):
        raise ThenAtomizationError("fixture support must be a branch record array")
    result = {}
    for ordinal, raw in enumerate(value):
        item = _mapping(raw, f"$fixture_support[{ordinal}]")
        gwt = _mapping(item.get("gwt"), f"$fixture_support[{ordinal}].gwt")
        branch_id = gwt.get("branch_id")
        if not isinstance(branch_id, str) or not branch_id or branch_id in result:
            raise ThenAtomizationError("fixture support branch IDs are invalid")
        matches = item.get("matches") or []
        if not isinstance(matches, list):
            raise ThenAtomizationError("fixture support matches must be an array")
        root_ids = [
            match.get("root_id")
            for match in matches
            if isinstance(match, Mapping)
        ]
        if root_ids and all(root_id is None for root_id in root_ids):
            root_id_presence = "all_missing"
        elif root_ids and all(root_id is not None for root_id in root_ids):
            root_id_presence = "all_present"
        elif root_ids:
            root_id_presence = "mixed"
        else:
            root_id_presence = "no_samples"
        result[branch_id] = {
            "root": item.get("root"),
            "structured_conditions": deepcopy(item.get("conditions") or []),
            "dropped_clauses": deepcopy(item.get("dropped") or []),
            "lookup_status": item.get("lookup_status"),
            "n_matches": item.get("n_matches", len(matches)),
            "root_id_presence": root_id_presence,
            "identity_samples": [
                {
                    "user_id": match.get("user_id"),
                    "root_id": match.get("root_id"),
                }
                for match in matches[:3]
                if isinstance(match, Mapping)
            ],
        }
    return result, content_sha256(value)


def _fixture_options(
    support: Mapping[str, Any], endpoints: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    options = []
    root = support.get("root")
    if isinstance(root, str) and root:
        options.append(
            {
                "candidate_id": f"fixture_root::{root}",
                "kind": "fixture_collection",
                "root": root,
                "root_id_presence": support.get("root_id_presence"),
                "supports": (
                    "establish entity existence or absence by checking whether a "
                    "chosen root identifier is present in this fixture collection"
                ),
            }
        )
    options.extend(endpoints)
    options.append(
        {
            "candidate_id": "insufficient",
            "kind": "non_binding_outcome",
            "meaning": "none of the supplied candidates directly establishes the condition",
        }
    )
    return options


def build_given_binding_decision_packets(
    requirement_set: Mapping[str, Any],
    fixture_support: Any,
    accepted_artifact_catalog: Mapping[str, Any],
    driver_capability_profile: Mapping[str, Any],
) -> dict[str, Any]:
    requirements = validate_given_evidence_requirement_set(requirement_set)
    support_index, support_fingerprint = _fixture_support_index(fixture_support)
    catalog = validate_accepted_artifact_catalog(accepted_artifact_catalog)
    profile = validate_driver_evidence_capability_profile(driver_capability_profile)
    if requirements["source_driver_profile_fingerprint"] != profile[
        "profile_fingerprint"
    ]:
        raise ThenAtomizationError("Given binding Driver profile lineage mismatch")
    branch_ids = {item["branch_id"] for item in requirements["branches"]}
    if set(support_index) != branch_ids:
        raise ThenAtomizationError("fixture support does not cover the Given branches")
    tool_catalog = _tool_catalog(catalog)
    source_descriptions = {
        item["source_kind"]: item["description"]
        for item in profile["evidence_sources"]
    }
    packets = []
    for requirement in requirements["requirements"]:
        if requirement["resolution_status"] != "needs_binding":
            continue
        task = requirement["binding_task"]
        payload = requirement["requirement"]
        condition = payload["condition_statement"]
        common = {
            "condition": condition,
            "when": requirement["when"],
            "temporal_rule": "use only evidence that exists before the When starts",
            "decision_scope": (
                "classify only the exact condition; do not import a sibling Given "
                "condition, and do not reject a condition merely because policy forbids it"
            ),
        }
        if task == "fixture_locator_binding":
            decision_type = "fixture_locator_choice"
            support = support_index[requirement["branch_id"]]
            endpoints = _relevant_endpoints(
                tool_catalog, condition, returns_only=True
            )
            model_input = {
                **common,
                "question": "Which supplied candidate can directly establish this condition?",
                "branch_fixture_support": deepcopy(support),
                "candidate_locators": _fixture_options(support, endpoints),
                "answer_contract": {
                    "selected_candidate_id": "exactly one candidate_id",
                    "reason": "one concise sentence",
                },
            }
        elif task == "evaluator_contract_binding":
            decision_type = "evaluator_shape_choice"
            model_input = {
                **common,
                "question": "Which one evaluator shape describes how to decide this condition?",
                "accepted_policy_lines": deepcopy(
                    payload.get("candidate_policy_lines") or []
                ),
                "candidate_observables": _merge_endpoint_candidates(
                    payload.get("candidate_observables") or [],
                    _relevant_endpoints(
                        tool_catalog, condition, returns_only=False
                    ),
                ),
                "decision_options": deepcopy(_EVALUATOR_OPTIONS),
                "answer_contract": {
                    "decision": "exactly one decision option",
                    "reason": "one concise sentence",
                },
            }
        elif task == "capability_scenario_binding":
            decision_type = "capability_polarity_choice"
            model_input = {
                **common,
                "question": (
                    "Must the Driver choose a request inside or outside the Agent's "
                    "declared action scope?"
                ),
                "accepted_policy_lines": deepcopy(
                    payload.get("candidate_policy_lines") or []
                ),
                "available_agent_action_names": deepcopy(
                    payload.get("available_agent_action_names") or []
                ),
                "decision_options": ["request_in_scope", "request_out_of_scope"],
                "answer_contract": {
                    "decision": "exactly one decision option",
                    "reason": "one concise sentence",
                },
            }
        elif task == "source_component_decomposition":
            decision_type = "source_component_choice"
            allowed = sorted(SOURCE_KINDS - {"multiple_sources"})
            model_input = {
                **common,
                "question": "Which evidence-source families are jointly required?",
                "accepted_policy_lines": deepcopy(
                    payload.get("candidate_policy_lines") or []
                ),
                "allowed_source_kinds": [
                    {
                        "source_kind": item,
                        "meaning": source_descriptions[item],
                    }
                    for item in allowed
                ],
                "answer_contract": {
                    "source_kinds": "two or more distinct allowed source_kind values",
                    "reason": "one concise sentence",
                },
            }
        else:
            raise ThenAtomizationError(f"unsupported Given binding task: {task}")
        packet = {
            "schema_version": PACKET_VERSION,
            "packet_id": f"GIVEN-BIND::{content_sha256(model_input)[:16]}::D01",
            "task_name": TASK_NAME,
            "decision_type": decision_type,
            "model_input": model_input,
            "members": [
                {
                    "requirement_id": requirement["requirement_id"],
                    "branch_id": requirement["branch_id"],
                    "condition_id": requirement["condition_id"],
                    "source_requirement_fingerprint": requirement[
                        "requirement_fingerprint"
                    ],
                }
            ],
        }
        packet["packet_fingerprint"] = content_sha256(packet)
        packets.append(packet)
    packets.sort(key=lambda item: item["packet_id"])
    counts = Counter(item["decision_type"] for item in packets)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": packets,
        "summary": {
            "packet_count": len(packets),
            "requirement_count": sum(len(item["members"]) for item in packets),
            "decision_type_counts": dict(sorted(counts.items())),
            "expected_model_calls": len(packets),
        },
        "source_requirement_set_fingerprint": requirements[
            "requirement_set_fingerprint"
        ],
        "source_fixture_support_fingerprint": support_fingerprint,
        "source_artifact_catalog_fingerprint": catalog["catalog_fingerprint"],
        "source_driver_profile_fingerprint": profile["profile_fingerprint"],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_binding_decision_packet_set(result)


def validate_given_binding_decision_packet_set(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_binding_decision_packet_set")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if (
        result.get("schema_version") != PACKET_SET_VERSION
        or result.get("task_name") != TASK_NAME
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given binding decision packet set")
    packet_ids = []
    requirement_ids = []
    counts = Counter()
    for raw in result.get("packets") or []:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        supplied = packet.pop("packet_fingerprint", None)
        if (
            packet.get("schema_version") != PACKET_VERSION
            or packet.get("task_name") != TASK_NAME
            or packet.get("decision_type") not in DECISION_TYPES
            or supplied != content_sha256(packet)
        ):
            raise ThenAtomizationError("invalid Given binding decision packet")
        packet_ids.append(packet.get("packet_id"))
        requirement_ids.extend(
            item.get("requirement_id") for item in packet.get("members") or []
        )
        counts[packet["decision_type"]] += 1
    if (
        len(packet_ids) != len(set(packet_ids))
        or len(requirement_ids) != len(set(requirement_ids))
    ):
        raise ThenAtomizationError("Given binding decision identities overlap")
    expected = {
        "packet_count": len(packet_ids),
        "requirement_count": len(requirement_ids),
        "decision_type_counts": dict(sorted(counts.items())),
        "expected_model_calls": len(packet_ids),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given binding decision summary mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def select_given_binding_decision_packets(
    packet_set: Mapping[str, Any], requirement_ids: list[str]
) -> dict[str, Any]:
    packets = validate_given_binding_decision_packet_set(packet_set)
    if (
        not isinstance(requirement_ids, list)
        or not requirement_ids
        or len(requirement_ids) != len(set(requirement_ids))
        or any(not isinstance(item, str) or not item for item in requirement_ids)
    ):
        raise ThenAtomizationError("Given binding selection IDs are invalid")
    index = {
        member["requirement_id"]: packet
        for packet in packets["packets"]
        for member in packet["members"]
    }
    missing = sorted(set(requirement_ids) - set(index))
    if missing:
        raise ThenAtomizationError(f"unknown Given binding requirements: {missing}")
    selected = []
    seen_packets = set()
    for requirement_id in requirement_ids:
        packet = index[requirement_id]
        if packet["packet_id"] not in seen_packets:
            seen_packets.add(packet["packet_id"])
            selected.append(deepcopy(packet))
    counts = Counter(item["decision_type"] for item in selected)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": selected,
        "summary": {
            "packet_count": len(selected),
            "requirement_count": sum(len(item["members"]) for item in selected),
            "decision_type_counts": dict(sorted(counts.items())),
            "expected_model_calls": len(selected),
        },
        "source_requirement_set_fingerprint": packets[
            "source_requirement_set_fingerprint"
        ],
        "source_fixture_support_fingerprint": packets[
            "source_fixture_support_fingerprint"
        ],
        "source_artifact_catalog_fingerprint": packets[
            "source_artifact_catalog_fingerprint"
        ],
        "source_driver_profile_fingerprint": packets[
            "source_driver_profile_fingerprint"
        ],
        "source_full_packet_set_fingerprint": packets["packet_set_fingerprint"],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_binding_decision_packet_set(result)


def validate_given_binding_decision_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    packet = _mapping(packet, "$packet")
    response = deepcopy(dict(_mapping(response, "$response")))
    decision_type = packet.get("decision_type")
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 400:
        raise ThenAtomizationError("Given binding decision reason is invalid")
    model_input = _mapping(packet.get("model_input"), "$packet.model_input")
    if decision_type == "fixture_locator_choice":
        if set(response) != {"selected_candidate_id", "reason"}:
            raise ThenAtomizationError("fixture locator response fields are invalid")
        allowed = {
            item.get("candidate_id")
            for item in model_input.get("candidate_locators") or []
        }
        if response.get("selected_candidate_id") not in allowed:
            raise ThenAtomizationError("fixture locator candidate is not allowed")
        return {
            "selected_candidate_id": response["selected_candidate_id"],
            "reason": reason.strip(),
        }
    if decision_type in {"evaluator_shape_choice", "capability_polarity_choice"}:
        if set(response) != {"decision", "reason"}:
            raise ThenAtomizationError("Given binding choice response fields are invalid")
        options = model_input.get("decision_options") or []
        allowed = {
            item.get("decision") if isinstance(item, Mapping) else item
            for item in options
        }
        if response.get("decision") not in allowed:
            raise ThenAtomizationError("Given binding decision is not allowed")
        return {"decision": response["decision"], "reason": reason.strip()}
    if decision_type == "source_component_choice":
        if set(response) != {"source_kinds", "reason"}:
            raise ThenAtomizationError("source component response fields are invalid")
        source_kinds = response.get("source_kinds")
        allowed = {
            item.get("source_kind")
            for item in model_input.get("allowed_source_kinds") or []
        }
        if (
            not isinstance(source_kinds, list)
            or len(source_kinds) < 2
            or len(source_kinds) != len(set(source_kinds))
            or any(item not in allowed for item in source_kinds)
        ):
            raise ThenAtomizationError("source component selection is invalid")
        return {"source_kinds": sorted(source_kinds), "reason": reason.strip()}
    raise ThenAtomizationError("unsupported Given binding decision type")


def render_given_binding_decision_prompt(
    packet: Mapping[str, Any], template: str
) -> str:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    if packet.get("task_name") != TASK_NAME or f"TASK: {TASK_NAME}" not in template:
        raise ThenAtomizationError("Given binding prompt/task mismatch")
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        packet["model_input"], ensure_ascii=False, indent=2
    )


def materialize_given_binding_decision_results(
    packet_set: Mapping[str, Any], responses: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_given_binding_decision_packet_set(packet_set)
    response_map = deepcopy(dict(_mapping(responses, "$responses")))
    if set(response_map) != {item["packet_id"] for item in packets["packets"]}:
        raise ThenAtomizationError("Given binding responses must cover the closed batch")
    records = []
    for packet in packets["packets"]:
        answer = validate_given_binding_decision_response(
            packet, _mapping(response_map[packet["packet_id"]], "$.response")
        )
        for member in packet["members"]:
            records.append(
                {
                    "requirement_id": member["requirement_id"],
                    "branch_id": member["branch_id"],
                    "condition_id": member["condition_id"],
                    "decision_type": packet["decision_type"],
                    "answer": answer,
                    "resolution_basis": "parsed_initial_response",
                    "source_packet_fingerprint": packet["packet_fingerprint"],
                    "source_requirement_fingerprint": member[
                        "source_requirement_fingerprint"
                    ],
                }
            )
    return _build_result_set(
        records,
        source_requirement_set_fingerprint=packets[
            "source_requirement_set_fingerprint"
        ],
        source_initial_packet_set_fingerprint=packets["packet_set_fingerprint"],
        source_adjudication_packet_set_fingerprint=None,
    )


def reconcile_given_binding_decision_results(
    initial_packet_set: Mapping[str, Any],
    initial_responses: Mapping[str, Any],
    adjudication_packet_set: Mapping[str, Any],
    adjudication_responses: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace an explicit requirement subset with improved-input adjudications."""

    initial = validate_given_binding_decision_packet_set(initial_packet_set)
    adjudication = validate_given_binding_decision_packet_set(
        adjudication_packet_set
    )
    if initial["source_requirement_set_fingerprint"] != adjudication[
        "source_requirement_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given binding adjudication requirement lineage mismatch")
    initial_map = deepcopy(dict(_mapping(initial_responses, "$initial_responses")))
    adjudication_map = deepcopy(
        dict(_mapping(adjudication_responses, "$adjudication_responses"))
    )
    if set(initial_map) != {item["packet_id"] for item in initial["packets"]}:
        raise ThenAtomizationError("initial Given binding response batch is incomplete")
    if set(adjudication_map) != {
        item["packet_id"] for item in adjudication["packets"]
    }:
        raise ThenAtomizationError("Given binding adjudication response batch is incomplete")
    initial_by_requirement = {
        member["requirement_id"]: (packet, member)
        for packet in initial["packets"]
        for member in packet["members"]
    }
    adjudication_by_requirement = {}
    for packet in adjudication["packets"]:
        for member in packet["members"]:
            requirement_id = member["requirement_id"]
            original = initial_by_requirement.get(requirement_id)
            if (
                original is None
                or original[1]["source_requirement_fingerprint"]
                != member["source_requirement_fingerprint"]
            ):
                raise ThenAtomizationError(
                    "Given binding adjudication is not a requirement-identical subset"
                )
            adjudication_by_requirement[requirement_id] = (packet, member)
    records = []
    for initial_packet in initial["packets"]:
        for initial_member in initial_packet["members"]:
            requirement_id = initial_member["requirement_id"]
            if requirement_id in adjudication_by_requirement:
                packet, member = adjudication_by_requirement[requirement_id]
                response = adjudication_map[packet["packet_id"]]
                basis = "accepted_adjudication_response"
            else:
                packet, member = initial_packet, initial_member
                response = initial_map[packet["packet_id"]]
                basis = "accepted_initial_response"
            answer = validate_given_binding_decision_response(
                packet, _mapping(response, "$.response")
            )
            records.append(
                {
                    "requirement_id": requirement_id,
                    "branch_id": member["branch_id"],
                    "condition_id": member["condition_id"],
                    "decision_type": packet["decision_type"],
                    "answer": answer,
                    "resolution_basis": basis,
                    "source_packet_fingerprint": packet["packet_fingerprint"],
                    "source_requirement_fingerprint": member[
                        "source_requirement_fingerprint"
                    ],
                }
            )
    return _build_result_set(
        records,
        source_requirement_set_fingerprint=initial[
            "source_requirement_set_fingerprint"
        ],
        source_initial_packet_set_fingerprint=initial["packet_set_fingerprint"],
        source_adjudication_packet_set_fingerprint=adjudication[
            "packet_set_fingerprint"
        ],
    )


def _build_result_set(
    records: list[dict[str, Any]],
    *,
    source_requirement_set_fingerprint: str,
    source_initial_packet_set_fingerprint: str,
    source_adjudication_packet_set_fingerprint: str | None,
) -> dict[str, Any]:
    decision_counts = Counter(item["decision_type"] for item in records)
    basis_counts = Counter(item["resolution_basis"] for item in records)
    result = {
        "schema_version": RESULT_SET_VERSION,
        "results": records,
        "summary": {
            "requirement_count": len(records),
            "decision_type_counts": dict(sorted(decision_counts.items())),
            "resolution_basis_counts": dict(sorted(basis_counts.items())),
        },
        "source_requirement_set_fingerprint": source_requirement_set_fingerprint,
        "source_initial_packet_set_fingerprint": source_initial_packet_set_fingerprint,
        "source_adjudication_packet_set_fingerprint": (
            source_adjudication_packet_set_fingerprint
        ),
    }
    result["result_set_fingerprint"] = content_sha256(result)
    return validate_given_binding_decision_result_set(result)


def validate_given_binding_decision_result_set(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_binding_decision_result_set")))
    fingerprint = result.pop("result_set_fingerprint", None)
    if (
        result.get("schema_version") != RESULT_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given binding decision result set")
    records = result.get("results")
    if not isinstance(records, list):
        raise ThenAtomizationError("Given binding decision results must be an array")
    ids = [item.get("requirement_id") for item in records]
    if len(ids) != len(set(ids)):
        raise ThenAtomizationError("Given binding decision result identities overlap")
    if any(item.get("decision_type") not in DECISION_TYPES for item in records):
        raise ThenAtomizationError("Given binding decision result type is invalid")
    decision_counts = Counter(item["decision_type"] for item in records)
    basis_counts = Counter(item.get("resolution_basis") for item in records)
    expected = {
        "requirement_count": len(records),
        "decision_type_counts": dict(sorted(decision_counts.items())),
        "resolution_basis_counts": dict(sorted(basis_counts.items())),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given binding decision result summary mismatch")
    result["result_set_fingerprint"] = fingerprint
    return result
