"""Prepare one narrow evidence-source decision per unresolved Given atom."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .given_when_contract_v1 import validate_given_when_contract_set
from .then_atomization import ThenAtomizationError


TASK_NAME = "given_atom_evidence_source"
PACKET_SET_VERSION = "agentspectesting.given-atom-source-packet-set/v0.1"
PACKET_VERSION = "agentspectesting.given-atom-source-packet/v0.1"
RESOLUTION_SET_VERSION = "agentspectesting.given-atom-source-resolution-set/v0.1"
RESOLUTION_VERSION = "agentspectesting.given-atom-source-resolution/v0.1"
SOURCE_KINDS = frozenset(
    {
        "user_message",
        "fixture_or_tool_state",
        "prior_runtime_event",
        "agent_capability",
        "multiple_sources",
        "insufficient",
    }
)
_OPTION_DEFINITIONS = {
    "user_message": (
        "The user's utterance itself constitutes the condition, such as making "
        "a request or complaint; a bare claim about external state is not enough."
    ),
    "fixture_or_tool_state": (
        "Environment, database, or tool-observed state establishes it, including "
        "a deterministic calculation over those values."
    ),
    "prior_runtime_event": (
        "An earlier action, message, tool call, or tool outcome in this execution "
        "must already have occurred and can be checked in the runtime trace."
    ),
    "agent_capability": (
        "It is established by comparing the request with the agent's available "
        "actions or policy scope."
    ),
    "multiple_sources": (
        "At least two distinct source kinds above are inherently required to make "
        "the condition independently checkable."
    ),
    "insufficient": "The supplied context does not determine the source kind.",
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _tool_summary(agent_spec: Mapping[str, Any]) -> list[dict[str, str]]:
    result = []
    seen = set()
    for raw in agent_spec.get("tools") or []:
        tool = _mapping(raw, "$.agent_spec.tools[]")
        name = tool.get("name")
        description = tool.get("description")
        if not isinstance(name, str) or not name or name in seen:
            raise ThenAtomizationError("agent tool names must be unique non-empty strings")
        seen.add(name)
        result.append(
            {
                "tool_name": name,
                "description": description.strip()
                if isinstance(description, str) and description.strip()
                else "",
            }
        )
    return sorted(result, key=lambda item: item["tool_name"])


def build_given_atom_source_packets(
    given_when_contract_set: Mapping[str, Any], agent_spec: Mapping[str, Any]
) -> dict[str, Any]:
    contracts = validate_given_when_contract_set(given_when_contract_set)
    agent_spec = deepcopy(dict(_mapping(agent_spec, "$agent_spec")))
    actions = _tool_summary(agent_spec)
    packets = []
    for contract in contracts["contracts"]:
        for atom in contract["given_atoms"]:
            if atom.get("resolution_status") != "unresolved":
                continue
            payload = _mapping(atom.get("payload"), "$.given_atoms[].payload")
            note = payload.get("upstream_drop_reason") or payload.get("reason")
            packet = {
                "schema_version": PACKET_VERSION,
                "packet_id": f"{atom['given_atom_id']}::SOURCE01",
                "task_name": TASK_NAME,
                "branch_id": contract["branch_id"],
                "given_atom_id": atom["given_atom_id"],
                "model_input": {
                    "condition_clause": atom["source_text"],
                    "full_given": contract["gwt"]["given"],
                    "when_event": contract["gwt"]["when"],
                    "upstream_mapping_note": note,
                    "available_agent_actions": deepcopy(actions),
                    "source_kind_options": [
                        {"source_kind": key, "meaning": _OPTION_DEFINITIONS[key]}
                        for key in sorted(SOURCE_KINDS)
                    ],
                },
                "source_given_atom_fingerprint": atom["given_atom_fingerprint"],
                "source_given_when_contract_fingerprint": contract[
                    "given_when_contract_fingerprint"
                ],
                "source_agent_spec_fingerprint": content_sha256(agent_spec),
            }
            packet["packet_fingerprint"] = content_sha256(packet)
            packets.append(packet)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": packets,
        "summary": {
            "packet_count": len(packets),
            "expected_model_calls": len(packets),
            "source_branch_count": len({packet["branch_id"] for packet in packets}),
            "allowed_source_kinds": sorted(SOURCE_KINDS),
        },
        "source_given_when_contract_set_fingerprint": contracts[
            "given_when_contract_set_fingerprint"
        ],
        "source_agent_spec_fingerprint": content_sha256(agent_spec),
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_atom_source_packet_set(result)


def validate_given_atom_source_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_atom_source_packet_set")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != PACKET_SET_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid Given atom source packet set")
    packets = result.get("packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("Given atom source packets must be an array")
    ids = []
    branches = set()
    for raw in packets:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if packet.get("schema_version") != PACKET_VERSION or packet_fingerprint != content_sha256(packet):
            raise ThenAtomizationError("invalid Given atom source packet")
        if packet.get("task_name") != TASK_NAME:
            raise ThenAtomizationError("invalid Given atom source task")
        model_input = _mapping(packet.get("model_input"), "$.packet.model_input")
        if set(model_input) != {
            "condition_clause",
            "full_given",
            "when_event",
            "upstream_mapping_note",
            "available_agent_actions",
            "source_kind_options",
        }:
            raise ThenAtomizationError("Given atom model input fields are invalid")
        option_ids = [item.get("source_kind") for item in model_input["source_kind_options"]]
        if option_ids != sorted(SOURCE_KINDS):
            raise ThenAtomizationError("Given atom source options are invalid")
        ids.append(packet.get("packet_id"))
        branches.add(packet.get("branch_id"))
    if len(ids) != len(set(ids)) or any(not isinstance(item, str) or not item for item in ids):
        raise ThenAtomizationError("Given atom packet IDs must be unique")
    expected_summary = {
        "packet_count": len(packets),
        "expected_model_calls": len(packets),
        "source_branch_count": len(branches),
        "allowed_source_kinds": sorted(SOURCE_KINDS),
    }
    if result.get("summary") != expected_summary:
        raise ThenAtomizationError("Given atom packet summary mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def select_given_atom_source_packets(
    packet_set: Mapping[str, Any], packet_ids: list[str]
) -> dict[str, Any]:
    """Create a fingerprint-closed subset without changing any source packet."""

    packets = validate_given_atom_source_packet_set(packet_set)
    if not isinstance(packet_ids, list) or not packet_ids:
        raise ThenAtomizationError("Given atom packet selection must be non-empty")
    if any(not isinstance(item, str) or not item for item in packet_ids):
        raise ThenAtomizationError("Given atom packet selection IDs must be strings")
    if len(packet_ids) != len(set(packet_ids)):
        raise ThenAtomizationError("Given atom packet selection IDs must be unique")
    packet_index = {packet["packet_id"]: packet for packet in packets["packets"]}
    missing = [packet_id for packet_id in packet_ids if packet_id not in packet_index]
    if missing:
        raise ThenAtomizationError(
            f"unknown Given atom packet selection IDs: {missing}"
        )
    selected = [deepcopy(packet_index[packet_id]) for packet_id in packet_ids]
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": selected,
        "summary": {
            "packet_count": len(selected),
            "expected_model_calls": len(selected),
            "source_branch_count": len(
                {packet["branch_id"] for packet in selected}
            ),
            "allowed_source_kinds": sorted(SOURCE_KINDS),
        },
        "source_given_when_contract_set_fingerprint": packets[
            "source_given_when_contract_set_fingerprint"
        ],
        "source_agent_spec_fingerprint": packets["source_agent_spec_fingerprint"],
        "source_full_packet_set_fingerprint": packets["packet_set_fingerprint"],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_atom_source_packet_set(result)


def select_pending_given_atom_source_packets(
    packet_set: Mapping[str, Any], completed_responses: Mapping[str, Any]
) -> dict[str, Any]:
    """Select packets without an already valid response, preserving lineage."""

    packets = validate_given_atom_source_packet_set(packet_set)
    responses = deepcopy(dict(_mapping(completed_responses, "$completed_responses")))
    packet_index = {packet["packet_id"]: packet for packet in packets["packets"]}
    unexpected = sorted(set(responses) - set(packet_index))
    if unexpected:
        raise ThenAtomizationError(
            f"unexpected completed Given atom response IDs: {unexpected}"
        )
    for packet_id, response in responses.items():
        validate_given_atom_source_response(
            packet_index[packet_id],
            _mapping(response, f"$completed_responses[{packet_id!r}]"),
        )
    pending_ids = [
        packet["packet_id"]
        for packet in packets["packets"]
        if packet["packet_id"] not in responses
    ]
    if not pending_ids:
        raise ThenAtomizationError("no pending Given atom source packets")
    result = select_given_atom_source_packets(packets, pending_ids)
    result["completed_response_count"] = len(responses)
    result["source_completed_responses_fingerprint"] = content_sha256(responses)
    result.pop("packet_set_fingerprint")
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_atom_source_packet_set(result)


def merge_given_atom_source_responses(
    packet_set: Mapping[str, Any], response_maps: list[Mapping[str, Any]]
) -> dict[str, dict[str, str]]:
    """Merge disjoint, valid response maps against one complete packet set."""

    packets = validate_given_atom_source_packet_set(packet_set)
    if not isinstance(response_maps, list) or not response_maps:
        raise ThenAtomizationError("Given atom response maps must be non-empty")
    packet_index = {packet["packet_id"]: packet for packet in packets["packets"]}
    merged: dict[str, dict[str, str]] = {}
    for map_index, raw_map in enumerate(response_maps):
        response_map = _mapping(raw_map, f"$response_maps[{map_index}]")
        for packet_id, response in response_map.items():
            if packet_id not in packet_index:
                raise ThenAtomizationError(
                    f"unexpected Given atom source response ID: {packet_id}"
                )
            if packet_id in merged:
                raise ThenAtomizationError(
                    f"duplicate Given atom source response ID: {packet_id}"
                )
            merged[packet_id] = validate_given_atom_source_response(
                packet_index[packet_id],
                _mapping(response, f"$response_maps[{map_index}][{packet_id!r}]"),
            )
    return {
        packet["packet_id"]: merged[packet["packet_id"]]
        for packet in packets["packets"]
        if packet["packet_id"] in merged
    }


def validate_given_atom_source_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, str]:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    response = deepcopy(dict(_mapping(response, "$response")))
    if set(response) != {"source_kind", "reason"}:
        raise ThenAtomizationError("Given atom source response fields are invalid")
    source_kind = response.get("source_kind")
    reason = response.get("reason")
    if source_kind not in SOURCE_KINDS:
        raise ThenAtomizationError("Given atom source_kind is invalid")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 500:
        raise ThenAtomizationError("Given atom source reason must be concise non-empty text")
    return {"source_kind": source_kind, "reason": reason.strip()}


def render_given_atom_source_prompt(packet: Mapping[str, Any], template: str) -> str:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    if packet.get("task_name") != TASK_NAME:
        raise ThenAtomizationError("packet is not a Given atom source task")
    marker = "TASK: given_atom_evidence_source"
    if marker not in template:
        raise ThenAtomizationError("Given atom source prompt template/task mismatch")
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        packet["model_input"], ensure_ascii=False, indent=2
    )


def materialize_given_atom_source_resolutions(
    packet_set: Mapping[str, Any], responses: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_given_atom_source_packet_set(packet_set)
    response_map = deepcopy(dict(_mapping(responses, "$responses")))
    packet_index = {packet["packet_id"]: packet for packet in packets["packets"]}
    unexpected = sorted(set(response_map) - set(packet_index))
    if unexpected:
        raise ThenAtomizationError(f"unexpected Given atom source response IDs: {unexpected}")
    records = []
    counts: Counter[str] = Counter()
    for packet_id, packet in packet_index.items():
        raw = response_map.get(packet_id)
        if raw is None:
            source_kind = "insufficient"
            reason = "response_missing"
            response_status = "missing"
        else:
            try:
                checked = validate_given_atom_source_response(
                    packet, _mapping(raw, f"$.responses[{packet_id!r}]")
                )
                source_kind = checked["source_kind"]
                reason = checked["reason"]
                response_status = "valid"
            except ThenAtomizationError as exc:
                source_kind = "insufficient"
                reason = str(exc)
                response_status = "invalid"
        status = "unresolved" if source_kind == "insufficient" else "resolved"
        record = {
            "schema_version": RESOLUTION_VERSION,
            "resolution_id": f"{packet['given_atom_id']}::SOURCE-RESULT01",
            "packet_id": packet_id,
            "branch_id": packet["branch_id"],
            "given_atom_id": packet["given_atom_id"],
            "resolution_status": status,
            "source_kind": source_kind,
            "reason": reason,
            "response_status": response_status,
            "source_packet_fingerprint": packet["packet_fingerprint"],
            "source_given_atom_fingerprint": packet[
                "source_given_atom_fingerprint"
            ],
        }
        record["resolution_fingerprint"] = content_sha256(record)
        records.append(record)
        counts[status] += 1
    result = {
        "schema_version": RESOLUTION_SET_VERSION,
        "resolutions": records,
        "summary": {
            "resolution_count": len(records),
            "resolved_count": counts["resolved"],
            "unresolved_count": counts["unresolved"],
        },
        "source_packet_set_fingerprint": packets["packet_set_fingerprint"],
    }
    result["resolution_set_fingerprint"] = content_sha256(result)
    return validate_given_atom_source_resolution_set(result)


def validate_given_atom_source_resolution_set(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_atom_source_resolution_set")))
    fingerprint = result.pop("resolution_set_fingerprint", None)
    if result.get("schema_version") != RESOLUTION_SET_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid Given atom source resolution set")
    records = result.get("resolutions")
    if not isinstance(records, list):
        raise ThenAtomizationError("Given atom source resolutions must be an array")
    ids = []
    counts: Counter[str] = Counter()
    for raw in records:
        record = deepcopy(dict(_mapping(raw, "$.resolutions[]")))
        record_fingerprint = record.pop("resolution_fingerprint", None)
        if record.get("schema_version") != RESOLUTION_VERSION or record_fingerprint != content_sha256(record):
            raise ThenAtomizationError("invalid Given atom source resolution")
        source_kind = record.get("source_kind")
        status = record.get("resolution_status")
        if source_kind not in SOURCE_KINDS:
            raise ThenAtomizationError("Given atom source resolution kind is invalid")
        expected_status = "unresolved" if source_kind == "insufficient" else "resolved"
        if status != expected_status:
            raise ThenAtomizationError("Given atom source resolution status is invalid")
        ids.append(record.get("resolution_id"))
        counts[status] += 1
    if len(ids) != len(set(ids)) or any(not isinstance(item, str) or not item for item in ids):
        raise ThenAtomizationError("Given atom source resolution IDs must be unique")
    expected_summary = {
        "resolution_count": len(records),
        "resolved_count": counts["resolved"],
        "unresolved_count": counts["unresolved"],
    }
    if result.get("summary") != expected_summary:
        raise ThenAtomizationError("Given atom source resolution summary mismatch")
    result["resolution_set_fingerprint"] = fingerprint
    return result
