"""Deduplicate and prepare exact-span splitting for compound Given clauses."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .given_clause_atomicity_v1 import (
    validate_given_clause_atomicity_packet_set,
    validate_given_clause_atomicity_response_set,
)
from .then_atomization import ThenAtomizationError


TASK_NAME = "given_clause_exact_split"
PACKET_SET_VERSION = "agentspectesting.given-clause-split-packet-set/v0.1"
PACKET_VERSION = "agentspectesting.given-clause-split-packet/v0.1"
RESULT_SET_VERSION = "agentspectesting.given-clause-split-result-set/v0.1"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def build_given_clause_split_packets(
    atomicity_packets: Mapping[str, Any],
    atomicity_responses: Mapping[str, Any],
) -> dict[str, Any]:
    packets = validate_given_clause_atomicity_packet_set(atomicity_packets)
    responses = validate_given_clause_atomicity_response_set(atomicity_responses)
    if responses["source_packet_set_fingerprint"] != packets["packet_set_fingerprint"]:
        raise ThenAtomizationError("Given split atomicity lineage mismatch")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in responses["responses"]:
        if record["decision"] == "multiple_conditions":
            grouped[record["source_clause"]].append(record)
    split_packets = []
    for source_clause, members in grouped.items():
        clause_hash = content_sha256(source_clause)[:16]
        packet = {
            "schema_version": PACKET_VERSION,
            "packet_id": f"GIVEN-CLAUSE::{clause_hash}::SPLIT01",
            "task_name": TASK_NAME,
            "model_input": {"source_clause": source_clause},
            "member_term_ids": [item["term_id"] for item in members],
            "member_branch_ids": [item["branch_id"] for item in members],
            "source_atomicity_packet_fingerprints": [
                item["source_packet_fingerprint"] for item in members
            ],
        }
        packet["packet_fingerprint"] = content_sha256(packet)
        split_packets.append(packet)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": split_packets,
        "summary": {
            "source_multiple_term_count": sum(
                len(item["member_term_ids"]) for item in split_packets
            ),
            "unique_clause_count": len(split_packets),
            "expected_model_calls": len(split_packets),
        },
        "source_atomicity_packet_set_fingerprint": packets[
            "packet_set_fingerprint"
        ],
        "source_atomicity_response_set_fingerprint": responses[
            "response_set_fingerprint"
        ],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_clause_split_packet_set(result)


def validate_given_clause_split_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_clause_split_packet_set")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if (
        result.get("schema_version") != PACKET_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given clause split packet set")
    packets = result.get("packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("Given clause split packets must be an array")
    ids = []
    term_ids = []
    clauses = []
    for raw in packets:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if (
            packet.get("schema_version") != PACKET_VERSION
            or packet_fingerprint != content_sha256(packet)
            or packet.get("task_name") != TASK_NAME
        ):
            raise ThenAtomizationError("invalid Given clause split packet")
        model_input = _mapping(packet.get("model_input"), "$.model_input")
        if set(model_input) != {"source_clause"}:
            raise ThenAtomizationError("Given split model input is not minimal")
        ids.append(packet.get("packet_id"))
        clauses.append(model_input["source_clause"])
        term_ids.extend(packet.get("member_term_ids") or [])
    if len(ids) != len(set(ids)) or len(clauses) != len(set(clauses)):
        raise ThenAtomizationError("Given split packets are not uniquely deduplicated")
    if len(term_ids) != len(set(term_ids)):
        raise ThenAtomizationError("Given split term lineage overlaps")
    expected = {
        "source_multiple_term_count": len(term_ids),
        "unique_clause_count": len(packets),
        "expected_model_calls": len(packets),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given split packet summary mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def validate_given_clause_split_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, list[str]]:
    packet = _mapping(packet, "$packet")
    response = deepcopy(dict(_mapping(response, "$response")))
    if set(response) != {"conditions"}:
        raise ThenAtomizationError("Given split response fields are invalid")
    conditions = response.get("conditions")
    if (
        not isinstance(conditions, list)
        or not 2 <= len(conditions) <= 8
        or any(not isinstance(item, str) or not item for item in conditions)
        or len(conditions) != len(set(conditions))
    ):
        raise ThenAtomizationError("Given split conditions must be 2-8 unique strings")
    clause = packet["model_input"]["source_clause"]
    cursor = 0
    for condition in conditions:
        position = clause.find(condition, cursor)
        if position < 0:
            raise ThenAtomizationError(
                "Given split conditions must be ordered, non-overlapping exact spans"
            )
        cursor = position + len(condition)
    return {"conditions": conditions}


def render_given_clause_split_prompt(packet: Mapping[str, Any], template: str) -> str:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    if packet.get("task_name") != TASK_NAME or "TASK: given_clause_exact_split" not in template:
        raise ThenAtomizationError("Given split prompt template/task mismatch")
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        packet["model_input"], ensure_ascii=False, indent=2
    )


def materialize_given_clause_split_results(
    packet_set: Mapping[str, Any], responses: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_given_clause_split_packet_set(packet_set)
    response_map = deepcopy(dict(_mapping(responses, "$responses")))
    index = {item["packet_id"]: item for item in packets["packets"]}
    if set(response_map) != set(index):
        raise ThenAtomizationError("Given split responses must cover the closed batch")
    records = []
    for packet in packets["packets"]:
        checked = validate_given_clause_split_response(
            packet, _mapping(response_map[packet["packet_id"]], "$.response")
        )
        for term_id, branch_id, source_fingerprint in zip(
            packet["member_term_ids"],
            packet["member_branch_ids"],
            packet["source_atomicity_packet_fingerprints"],
            strict=True,
        ):
            records.append(
                {
                    "term_id": term_id,
                    "branch_id": branch_id,
                    "source_clause": packet["model_input"]["source_clause"],
                    "condition_spans": deepcopy(checked["conditions"]),
                    "source_split_packet_fingerprint": packet[
                        "packet_fingerprint"
                    ],
                    "source_atomicity_packet_fingerprint": source_fingerprint,
                }
            )
    result = {
        "schema_version": RESULT_SET_VERSION,
        "results": records,
        "summary": {
            "resolved_term_count": len(records),
            "unique_split_count": len(packets["packets"]),
        },
        "source_packet_set_fingerprint": packets["packet_set_fingerprint"],
    }
    result["result_set_fingerprint"] = content_sha256(result)
    return validate_given_clause_split_result_set(result)


def validate_given_clause_split_result_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_clause_split_result_set")))
    fingerprint = result.pop("result_set_fingerprint", None)
    if (
        result.get("schema_version") != RESULT_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given clause split result set")
    records = result.get("results")
    if not isinstance(records, list):
        raise ThenAtomizationError("Given clause split results must be an array")
    term_ids = [item.get("term_id") for item in records]
    if len(term_ids) != len(set(term_ids)):
        raise ThenAtomizationError("Given clause split result term IDs must be unique")
    for item in records:
        spans = item.get("condition_spans")
        if not isinstance(spans, list) or len(spans) < 2:
            raise ThenAtomizationError("Given clause split result spans are invalid")
    expected_summary = {
        "resolved_term_count": len(records),
        "unique_split_count": len(
            {item.get("source_split_packet_fingerprint") for item in records}
        ),
    }
    if result.get("summary") != expected_summary:
        raise ThenAtomizationError("Given clause split result summary mismatch")
    result["result_set_fingerprint"] = fingerprint
    return result
