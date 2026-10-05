"""Prepare the first narrow resolver for unverified Given source clauses."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .given_when_contract_v2 import validate_given_when_contract_set_v2
from .then_atomization import ThenAtomizationError


TASK_NAME = "given_clause_atomicity"
PACKET_SET_VERSION = "agentspectesting.given-clause-atomicity-packet-set/v0.1"
PACKET_VERSION = "agentspectesting.given-clause-atomicity-packet/v0.1"
DECISIONS = frozenset({"single_condition", "multiple_conditions", "unclear"})
RESPONSE_SET_VERSION = "agentspectesting.given-clause-atomicity-response-set/v0.1"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def build_given_clause_atomicity_packets(
    contract_set: Mapping[str, Any],
) -> dict[str, Any]:
    contracts = validate_given_when_contract_set_v2(contract_set)
    packets = []
    for contract in contracts["contracts"]:
        for term in contract["given_logic"]["terms"]:
            if term["atomicity_status"] != "unverified_source_clause":
                continue
            packet = {
                "schema_version": PACKET_VERSION,
                "packet_id": f"{term['term_id']}::ATOMICITY01",
                "task_name": TASK_NAME,
                "branch_id": contract["branch_id"],
                "term_id": term["term_id"],
                "model_input": {"source_clause": term["source_text"]},
                "source_contract_fingerprint": contract[
                    "given_when_contract_fingerprint"
                ],
                "source_given_logic_fingerprint": contract["given_logic"][
                    "given_logic_fingerprint"
                ],
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
            "source_branch_count": len({item["branch_id"] for item in packets}),
            "allowed_decisions": sorted(DECISIONS),
        },
        "source_contract_set_fingerprint": contracts[
            "given_when_contract_set_fingerprint"
        ],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_clause_atomicity_packet_set(result)


def validate_given_clause_atomicity_packet_set(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_clause_atomicity_packet_set")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if (
        result.get("schema_version") != PACKET_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given clause atomicity packet set")
    packets = result.get("packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("Given clause atomicity packets must be an array")
    ids = []
    branches = set()
    for raw in packets:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if (
            packet.get("schema_version") != PACKET_VERSION
            or packet_fingerprint != content_sha256(packet)
            or packet.get("task_name") != TASK_NAME
        ):
            raise ThenAtomizationError("invalid Given clause atomicity packet")
        if set(_mapping(packet.get("model_input"), "$.model_input")) != {
            "source_clause"
        }:
            raise ThenAtomizationError("Given atomicity model input is not minimal")
        ids.append(packet.get("packet_id"))
        branches.add(packet.get("branch_id"))
    if len(ids) != len(set(ids)):
        raise ThenAtomizationError("Given atomicity packet IDs must be unique")
    expected_summary = {
        "packet_count": len(packets),
        "expected_model_calls": len(packets),
        "source_branch_count": len(branches),
        "allowed_decisions": sorted(DECISIONS),
    }
    if result.get("summary") != expected_summary:
        raise ThenAtomizationError("Given atomicity packet summary mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def select_given_clause_atomicity_packets(
    packet_set: Mapping[str, Any], packet_ids: list[str]
) -> dict[str, Any]:
    packets = validate_given_clause_atomicity_packet_set(packet_set)
    if not isinstance(packet_ids, list) or not packet_ids or len(packet_ids) != len(
        set(packet_ids)
    ):
        raise ThenAtomizationError("Given atomicity selection must be non-empty and unique")
    index = {item["packet_id"]: item for item in packets["packets"]}
    missing = [item for item in packet_ids if item not in index]
    if missing:
        raise ThenAtomizationError(f"unknown Given atomicity packet IDs: {missing}")
    selected = [deepcopy(index[item]) for item in packet_ids]
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": selected,
        "summary": {
            "packet_count": len(selected),
            "expected_model_calls": len(selected),
            "source_branch_count": len({item["branch_id"] for item in selected}),
            "allowed_decisions": sorted(DECISIONS),
        },
        "source_contract_set_fingerprint": packets[
            "source_contract_set_fingerprint"
        ],
        "source_full_packet_set_fingerprint": packets["packet_set_fingerprint"],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_clause_atomicity_packet_set(result)


def select_pending_given_clause_atomicity_packets(
    packet_set: Mapping[str, Any], completed_responses: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_given_clause_atomicity_packet_set(packet_set)
    responses = deepcopy(dict(_mapping(completed_responses, "$completed_responses")))
    index = {item["packet_id"]: item for item in packets["packets"]}
    unexpected = sorted(set(responses) - set(index))
    if unexpected:
        raise ThenAtomizationError(
            f"unexpected completed Given atomicity response IDs: {unexpected}"
        )
    for packet_id, response in responses.items():
        validate_given_clause_atomicity_response(
            index[packet_id],
            _mapping(response, f"$completed_responses[{packet_id!r}]"),
        )
    pending_ids = [
        item["packet_id"]
        for item in packets["packets"]
        if item["packet_id"] not in responses
    ]
    if not pending_ids:
        raise ThenAtomizationError("no pending Given atomicity packets")
    result = select_given_clause_atomicity_packets(packets, pending_ids)
    result["completed_response_count"] = len(responses)
    result["source_completed_responses_fingerprint"] = content_sha256(responses)
    result.pop("packet_set_fingerprint")
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_clause_atomicity_packet_set(result)


def validate_given_clause_atomicity_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, str]:
    packet = _mapping(packet, "$packet")
    if packet.get("task_name") != TASK_NAME:
        raise ThenAtomizationError("packet is not a Given atomicity task")
    response = deepcopy(dict(_mapping(response, "$response")))
    if set(response) != {"decision", "reason"}:
        raise ThenAtomizationError("Given atomicity response fields are invalid")
    if response.get("decision") not in DECISIONS:
        raise ThenAtomizationError("Given atomicity decision is invalid")
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 300:
        raise ThenAtomizationError("Given atomicity reason must be concise text")
    return {"decision": response["decision"], "reason": reason.strip()}


def render_given_clause_atomicity_prompt(
    packet: Mapping[str, Any], template: str
) -> str:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    if packet.get("task_name") != TASK_NAME or "TASK: given_clause_atomicity" not in template:
        raise ThenAtomizationError("Given atomicity prompt template/task mismatch")
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        packet["model_input"], ensure_ascii=False, indent=2
    )


def materialize_given_clause_atomicity_responses(
    packet_set: Mapping[str, Any], response_maps: list[Mapping[str, Any]]
) -> dict[str, Any]:
    packets = validate_given_clause_atomicity_packet_set(packet_set)
    if not isinstance(response_maps, list) or not response_maps:
        raise ThenAtomizationError("Given atomicity response maps must be non-empty")
    packet_index = {item["packet_id"]: item for item in packets["packets"]}
    merged: dict[str, dict[str, str]] = {}
    for map_index, raw_map in enumerate(response_maps):
        response_map = _mapping(raw_map, f"$response_maps[{map_index}]")
        for packet_id, response in response_map.items():
            if packet_id not in packet_index:
                raise ThenAtomizationError(
                    f"unexpected Given atomicity response ID: {packet_id}"
                )
            if packet_id in merged:
                raise ThenAtomizationError(
                    f"duplicate Given atomicity response ID: {packet_id}"
                )
            merged[packet_id] = validate_given_clause_atomicity_response(
                packet_index[packet_id],
                _mapping(response, f"$response_maps[{map_index}][{packet_id!r}]"),
            )
    records = [
        {
            "packet_id": packet["packet_id"],
            "term_id": packet["term_id"],
            "branch_id": packet["branch_id"],
            "source_clause": packet["model_input"]["source_clause"],
            "decision": merged[packet["packet_id"]]["decision"],
            "reason": merged[packet["packet_id"]]["reason"],
            "source_packet_fingerprint": packet["packet_fingerprint"],
        }
        for packet in packets["packets"]
        if packet["packet_id"] in merged
    ]
    counts = {
        decision: sum(item["decision"] == decision for item in records)
        for decision in sorted(DECISIONS)
    }
    result = {
        "schema_version": RESPONSE_SET_VERSION,
        "responses": records,
        "summary": {
            "response_count": len(records),
            "missing_count": len(packets["packets"]) - len(records),
            "decision_counts": counts,
        },
        "source_packet_set_fingerprint": packets["packet_set_fingerprint"],
    }
    result["response_set_fingerprint"] = content_sha256(result)
    return validate_given_clause_atomicity_response_set(result)


def validate_given_clause_atomicity_response_set(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_atomicity_response_set")))
    fingerprint = result.pop("response_set_fingerprint", None)
    if (
        result.get("schema_version") != RESPONSE_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given atomicity response set")
    records = result.get("responses")
    if not isinstance(records, list):
        raise ThenAtomizationError("Given atomicity responses must be an array")
    ids = [item.get("packet_id") for item in records]
    if len(ids) != len(set(ids)):
        raise ThenAtomizationError("Given atomicity response IDs must be unique")
    counts = {
        decision: sum(item.get("decision") == decision for item in records)
        for decision in sorted(DECISIONS)
    }
    summary = result.get("summary")
    if not isinstance(summary, Mapping) or summary.get("response_count") != len(
        records
    ) or summary.get("decision_counts") != counts:
        raise ThenAtomizationError("Given atomicity response summary mismatch")
    if any(item.get("decision") not in DECISIONS for item in records):
        raise ThenAtomizationError("Given atomicity response decision is invalid")
    result["response_set_fingerprint"] = fingerprint
    return result
