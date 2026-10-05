"""Boundary-first semantic segmentation with binary model decisions.

The compiler proposes surface boundaries mechanically.  The model never
generates spans: one call decides whether one proposed boundary should split
the sentence.  Accepted boundaries are then materialized into exact spans by
the compiler.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .gwt_span_pipeline_v5 import validate_gwt_span_discovery_packet_set
from .then_atomization import ThenAtomizationError


BOUNDARY_PACKET_SET_VERSION = "agentspectesting.gwt-boundary-decision-packets/v0.6"
BOUNDARY_PACKET_VERSION = "agentspectesting.boundary-decision-packet/v0.6"
BOUNDARY_RESPONSE_SET_VERSION = "agentspectesting.gwt-boundary-decision-responses/v0.6"
MATERIALIZED_SPAN_SET_VERSION = "agentspectesting.gwt-materialized-spans/v0.6"
BOUNDARY_TASK = "binary_semantic_boundary_decision"
BOUNDARY_DECISIONS = frozenset({"yes", "no", "ambiguous"})

# These are language-level boundary cues, not airline/domain vocabulary.  The
# list is versioned as part of the generator so another language/parser can be
# substituted without changing the decision contract.
_CUE_PATTERNS = (
    r"\bif\s+and\s+only\s+if\b",
    r"\band\s+then\b",
    r"\bto(?=\s+either\b)",
    r"\b(if|unless|when|before|after|while|where|that|which|who|whose|with|without|within|and|or|not|in|for|by|per|to)\b",
    r"[,;:]",
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _fingerprinted(value: dict[str, Any]) -> dict[str, Any]:
    value["packet_fingerprint"] = content_sha256(value)
    return value


def _inside_quoted_text(text: str, position: int) -> bool:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if index >= position:
            break
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char not in {"'", '"'}:
            continue
        if char == "'" and 0 < index < len(text) - 1:
            if text[index - 1].isalnum() and text[index + 1].isalnum():
                continue
        if quote is None:
            quote = char
        elif quote == char:
            quote = None
    return quote is not None


def enumerate_surface_boundaries(primary_text: str) -> list[dict[str, Any]]:
    """Return non-overlapping possible split locations in textual order."""

    matches: list[tuple[int, int, str]] = []
    occupied: set[int] = set()
    for pattern in _CUE_PATTERNS:
        for match in re.finditer(pattern, primary_text, flags=re.IGNORECASE):
            start, end = match.span()
            if _inside_quoted_text(primary_text, start):
                continue
            if not primary_text[:start].strip(" \t\r\n,;:") or not primary_text[end:].strip(" \t\r\n,;:"):
                continue
            if any(index in occupied for index in range(start, end)):
                continue
            matches.append((start, end, primary_text[start:end]))
            occupied.update(range(start, end))
    matches.sort(key=lambda item: item[0])
    result = []
    for index, (start, end, cue) in enumerate(matches, start=1):
        result.append(
            {
                "boundary_id": f"B{index:02d}",
                "start_char": start,
                "end_char": end,
                "cue_span": cue,
                "marked_text": f"{primary_text[:start]}⟦{primary_text[start:end]}⟧{primary_text[end:]}",
            }
        )
    return result


def build_gwt_boundary_decision_packets(
    discovery_packet_set: Mapping[str, Any],
    *,
    selected_target_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = validate_gwt_span_discovery_packet_set(discovery_packet_set)
    available_ids = [item["target_id"] for item in source["discovery_packets"]]
    selected_ids = list(selected_target_ids) if selected_target_ids is not None else available_ids
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ThenAtomizationError("v0.6 selected target IDs must be non-empty and unique")
    missing = [target_id for target_id in selected_ids if target_id not in available_ids]
    if missing:
        raise ThenAtomizationError(f"v0.6 selected targets are missing: {missing}")
    selected_set = set(selected_ids)
    packets = []
    counts_by_target: dict[str, int] = {}
    for parent in source["discovery_packets"]:
        if parent["target_id"] not in selected_set:
            continue
        boundaries = enumerate_surface_boundaries(parent["primary_text"])
        counts_by_target[parent["target_id"]] = len(boundaries)
        for boundary in boundaries:
            task_input = {
                "target_id": parent["target_id"],
                "text_kind": parent["text_kind"],
                "primary_text": parent["primary_text"],
                "proposed_boundary": deepcopy(boundary),
            }
            packets.append(
                _fingerprinted(
                    {
                        "schema_version": BOUNDARY_PACKET_VERSION,
                        "task_name": BOUNDARY_TASK,
                        "target_id": parent["target_id"],
                        "spec_id": parent["spec_id"],
                        "branch_id": parent["branch_id"],
                        "text_kind": parent["text_kind"],
                        "boundary_id": boundary["boundary_id"],
                        "task_contract": {
                            "question": "Does the marked cue connect two claims that could have different truth values?",
                            "provided_inputs": ["primary_text", "proposed_boundary"],
                            "output_decision": "yes | no | ambiguous",
                            "deferred_decisions": [
                                "semantic role",
                                "modality",
                                "relation type",
                                "normalization",
                                "source-to-GWT mapping",
                            ],
                        },
                        "task_input": task_input,
                    }
                )
            )
    result = {
        "schema_version": BOUNDARY_PACKET_SET_VERSION,
        "source_discovery_packet_set_fingerprint": source["packet_set_fingerprint"],
        "source": deepcopy(source["source"]),
        "text_targets": [
            {
                "target_id": item["target_id"],
                "spec_id": item["spec_id"],
                "branch_id": item["branch_id"],
                "text_kind": item["text_kind"],
                "primary_text": item["primary_text"],
            }
            for item in source["discovery_packets"]
            if item["target_id"] in selected_set
        ],
        "boundary_packets": packets,
        "summary": {
            "text_target_count": len(selected_ids),
            "candidate_boundary_count": len(packets),
            "expected_boundary_model_calls": len(packets),
            "llm_generated_boundaries": 0,
            "llm_decisions_per_call": 1,
            "decisions_by_target": counts_by_target,
            "coverage_model_calls": 0,
        },
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return result


def validate_gwt_boundary_decision_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != BOUNDARY_PACKET_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.6 boundary packet-set schema")
    if fingerprint != content_sha256(result):
        raise ThenAtomizationError("v0.6 boundary packet-set fingerprint mismatch")
    packets = result.get("boundary_packets")
    targets = result.get("text_targets")
    if not isinstance(packets, list) or not isinstance(targets, list):
        raise ThenAtomizationError("v0.6 boundary packet arrays are missing")
    target_map = {item["target_id"]: item for item in targets}
    identities = []
    for raw in packets:
        packet = dict(_mapping(raw, "$.boundary_packets[]"))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if packet.get("schema_version") != BOUNDARY_PACKET_VERSION or packet.get("task_name") != BOUNDARY_TASK:
            raise ThenAtomizationError("v0.6 boundary packet schema/task mismatch")
        if packet_fingerprint != content_sha256(packet):
            raise ThenAtomizationError("v0.6 boundary packet fingerprint mismatch")
        task_input = _mapping(packet.get("task_input"), "$.boundary_packets[].task_input")
        if set(task_input) != {"target_id", "text_kind", "primary_text", "proposed_boundary"}:
            raise ThenAtomizationError("v0.6 model input is not the closed boundary contract")
        target = target_map.get(packet.get("target_id"))
        if target is None or task_input.get("primary_text") != target.get("primary_text"):
            raise ThenAtomizationError("v0.6 boundary target text mismatch")
        boundary = _mapping(task_input.get("proposed_boundary"), "$.proposed_boundary")
        if boundary.get("boundary_id") != packet.get("boundary_id"):
            raise ThenAtomizationError("v0.6 boundary identity mismatch")
        start, end = boundary.get("start_char"), boundary.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(target["primary_text"]):
            raise ThenAtomizationError("v0.6 boundary offsets are invalid")
        if target["primary_text"][start:end] != boundary.get("cue_span"):
            raise ThenAtomizationError("v0.6 boundary cue is not exact primary text")
        expected_marked = (
            f"{target['primary_text'][:start]}⟦{target['primary_text'][start:end]}⟧"
            f"{target['primary_text'][end:]}"
        )
        if boundary.get("marked_text") != expected_marked:
            raise ThenAtomizationError("v0.6 marked text does not identify the proposed boundary")
        identities.append((packet.get("target_id"), packet.get("boundary_id")))
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("v0.6 boundary packet identities must be unique")
    if result.get("summary", {}).get("expected_boundary_model_calls") != len(packets):
        raise ThenAtomizationError("v0.6 boundary call count mismatch")
    if result.get("summary", {}).get("llm_generated_boundaries") != 0:
        raise ThenAtomizationError("v0.6 boundaries must be compiler-generated")
    result["packet_set_fingerprint"] = fingerprint
    return result


def render_gwt_boundary_decision_prompt(packet: Mapping[str, Any], template: str) -> str:
    placeholder = "{boundary_decision_input}"
    if packet.get("task_name") != BOUNDARY_TASK or placeholder not in template:
        raise ThenAtomizationError("v0.6 boundary prompt template/task mismatch")
    import json

    return template.replace(placeholder, json.dumps(packet["task_input"], ensure_ascii=False, indent=2))


def validate_boundary_decision_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.boundary_response")))
    expected = {"target_id", "boundary_id", "decision", "reason"}
    if set(result) != expected:
        raise ThenAtomizationError("v0.6 boundary response fields are invalid")
    if result.get("target_id") != packet.get("target_id") or result.get("boundary_id") != packet.get("boundary_id"):
        raise ThenAtomizationError("v0.6 boundary response identity mismatch")
    if result.get("decision") not in BOUNDARY_DECISIONS:
        raise ThenAtomizationError("v0.6 boundary decision is invalid")
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        raise ThenAtomizationError("v0.6 boundary reason must be non-empty")
    return result


def validate_gwt_boundary_decision_response_set(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_gwt_boundary_decision_packet_set(packet_set)
    responses = deepcopy(dict(_mapping(response_set, "$.boundary_responses")))
    if set(responses) != {"schema_version", "responses"}:
        raise ThenAtomizationError("v0.6 boundary response-set fields are invalid")
    if responses.get("schema_version") != BOUNDARY_RESPONSE_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.6 boundary response-set schema")
    packet_map = {
        (item["target_id"], item["boundary_id"]): item for item in packets["boundary_packets"]
    }
    checked = {}
    if not isinstance(responses.get("responses"), list):
        raise ThenAtomizationError("v0.6 boundary responses must be an array")
    for raw in responses["responses"]:
        key = (raw.get("target_id"), raw.get("boundary_id")) if isinstance(raw, Mapping) else (None, None)
        if key not in packet_map or key in checked:
            raise ThenAtomizationError(f"unknown or duplicate v0.6 boundary response: {key}")
        checked[key] = validate_boundary_decision_response(packet_map[key], raw)
    if set(checked) != set(packet_map):
        raise ThenAtomizationError("v0.6 boundary responses must cover the closed packet set")
    return {
        "schema_version": BOUNDARY_RESPONSE_SET_VERSION,
        "responses": [
            checked[(item["target_id"], item["boundary_id"])]
            for item in packets["boundary_packets"]
        ],
    }


def _trim_fragment(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (text[start].isspace() or text[start] in ",;:"):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ",;:"):
        end -= 1
    return start, end


def materialize_gwt_spans(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_gwt_boundary_decision_packet_set(packet_set)
    responses = validate_gwt_boundary_decision_response_set(packets, response_set)
    response_map = {
        (item["target_id"], item["boundary_id"]): item for item in responses["responses"]
    }
    packets_by_target: dict[str, list[Mapping[str, Any]]] = {}
    for packet in packets["boundary_packets"]:
        packets_by_target.setdefault(packet["target_id"], []).append(packet)
    results = []
    total_units = 0
    for target in packets["text_targets"]:
        target_packets = packets_by_target.get(target["target_id"], [])
        ambiguous = [
            item["boundary_id"]
            for item in target_packets
            if response_map[(item["target_id"], item["boundary_id"])]["decision"] == "ambiguous"
        ]
        accepted = [
            item for item in target_packets
            if response_map[(item["target_id"], item["boundary_id"])]["decision"] == "yes"
        ]
        if ambiguous:
            results.append({**deepcopy(target), "status": "ambiguous", "ambiguous_boundary_ids": ambiguous, "accepted_boundaries": [], "semantic_units": []})
            continue
        text = target["primary_text"]
        cuts = sorted(
            (
                item["task_input"]["proposed_boundary"]["start_char"],
                item["task_input"]["proposed_boundary"]["end_char"],
                item,
            )
            for item in accepted
        )
        units = []
        cursor = 0
        for start, end, _ in [*cuts, (len(text), len(text), None)]:
            unit_start, unit_end = _trim_fragment(text, cursor, start)
            if unit_start < unit_end:
                units.append(
                    {
                        "unit_id": f"U{len(units) + 1:02d}",
                        "start_char": unit_start,
                        "end_char": unit_end,
                        "exact_span": text[unit_start:unit_end],
                    }
                )
            cursor = end
        total_units += len(units)
        results.append(
            {
                **deepcopy(target),
                "status": "resolved",
                "ambiguous_boundary_ids": [],
                "accepted_boundaries": [
                    {
                        "boundary_id": item["boundary_id"],
                        **deepcopy(item["task_input"]["proposed_boundary"]),
                    }
                    for item in accepted
                ],
                "semantic_units": units,
            }
        )
    result = {
        "schema_version": MATERIALIZED_SPAN_SET_VERSION,
        "boundary_packet_set_fingerprint": packets["packet_set_fingerprint"],
        "targets": results,
        "summary": {
            "target_count": len(results),
            "resolved_target_count": sum(item["status"] == "resolved" for item in results),
            "ambiguous_target_count": sum(item["status"] == "ambiguous" for item in results),
            "materialized_unit_count": total_units,
            "llm_generated_spans": 0,
        },
    }
    result["result_fingerprint"] = content_sha256(result)
    return result
