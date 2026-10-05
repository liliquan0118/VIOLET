"""Span-first staged semantic analysis for Source rules and GWT Then text."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .gwt_semantic_pipeline_v3 import build_gwt_semantic_stage1_packets
from .then_atomization import MODALITIES, RESOLUTION_STATUSES, ThenAtomizationError


DISCOVERY_PACKET_SET_VERSION = "agentspectesting.gwt-span-discovery-packets/v0.5"
DISCOVERY_PACKET_VERSION = "agentspectesting.proposition-span-discovery-packet/v0.5"
DISCOVERY_RESPONSE_SET_VERSION = "agentspectesting.gwt-span-discovery-responses/v0.5"
ROLE_PACKET_SET_VERSION = "agentspectesting.proposition-role-packets/v0.5"
ROLE_PACKET_VERSION = "agentspectesting.proposition-role-packet/v0.5"
ROLE_RESPONSE_SET_VERSION = "agentspectesting.proposition-role-responses/v0.5"
MODALITY_PACKET_SET_VERSION = "agentspectesting.proposition-modality-packets/v0.5"
MODALITY_PACKET_VERSION = "agentspectesting.proposition-modality-packet/v0.5"
MODALITY_RESPONSE_SET_VERSION = "agentspectesting.proposition-modality-responses/v0.5"

DISCOVERY_TASK = "proposition_span_discovery"
ROLE_TASK = "proposition_role_classification"
MODALITY_TASK = "proposition_modality_classification"

SOURCE_ROLES = frozenset({"governed_outcome", "activation_condition", "scope_reference"})
THEN_ROLES = frozenset(
    {"asserted_outcome", "condition_reference", "temporal_reference", "scope_reference"}
)
ROLE_LABELS = SOURCE_ROLES | THEN_ROLES | {"ambiguous"}
MODALITY_LABELS = MODALITIES | {"ambiguous"}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ThenAtomizationError(f"{path} must be a non-empty string")
    return value.strip()


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise ThenAtomizationError(
            f"{path} fields are invalid: expected {sorted(expected)}, got {sorted(value)}"
        )


def _fingerprinted(value: dict[str, Any]) -> dict[str, Any]:
    value["packet_fingerprint"] = content_sha256(value)
    return value


def build_gwt_span_discovery_packets(
    spec_document: Mapping[str, Any],
    *,
    selected_branch_ids: Sequence[str],
    source_path: str | None = None,
) -> dict[str, Any]:
    base = build_gwt_semantic_stage1_packets(
        spec_document,
        selected_branch_ids=selected_branch_ids,
        source_path=source_path,
    )
    discovery_packets = []
    for packet in base["source_packets"]:
        primary_text = packet["source_context"]["rule_text"]
        discovery_packets.append(
            _fingerprinted(
                {
                    "schema_version": DISCOVERY_PACKET_VERSION,
                    "task_name": DISCOVERY_TASK,
                    "text_kind": "source_rule",
                    "target_id": packet["spec_id"],
                    "spec_id": packet["spec_id"],
                    "branch_id": None,
                    "selected_for_test": None,
                    "task_contract": {
                        "question": "Which exact substrings of primary_text express independently truth-valued proposition candidates?",
                        "provided_inputs": ["primary_text"],
                        "why_input_is_sufficient": "The task only segments literal text; it does not interpret source evidence, GWT context, modality, or semantic roles.",
                        "deferred_decisions": [
                            "proposition role",
                            "modality and polarity",
                            "condition logic",
                            "normalization",
                            "GWT mapping",
                            "relations",
                        ],
                    },
                    "primary_text": primary_text,
                    "task_input": {
                        "target_id": packet["spec_id"],
                        "text_kind": "source_rule",
                        "primary_text": primary_text,
                    },
                }
            )
        )
    for packet in base["then_packets"]:
        primary_text = packet["gwt_context"]["then"]
        discovery_packets.append(
            _fingerprinted(
                {
                    "schema_version": DISCOVERY_PACKET_VERSION,
                    "task_name": DISCOVERY_TASK,
                    "text_kind": "gwt_then",
                    "target_id": packet["branch_id"],
                    "spec_id": packet["spec_id"],
                    "branch_id": packet["branch_id"],
                    "selected_for_test": packet["selected_for_test"],
                    "task_contract": {
                        "question": "Which exact substrings of primary_text express independently truth-valued proposition candidates?",
                        "provided_inputs": ["primary_text"],
                        "why_input_is_sufficient": "The task only segments literal Then text; referent resolution, role, modality, and source support are deferred.",
                        "deferred_decisions": [
                            "proposition role",
                            "modality and polarity",
                            "normalization",
                            "source support",
                            "relations",
                            "test observability",
                        ],
                    },
                    "primary_text": primary_text,
                    "gwt_context": deepcopy(packet["gwt_context"]),
                    "task_input": {
                        "target_id": packet["branch_id"],
                        "text_kind": "gwt_then",
                        "primary_text": primary_text,
                    },
                }
            )
        )
    payload = {
        "schema_version": DISCOVERY_PACKET_SET_VERSION,
        "source": deepcopy(base["source"]),
        "selected_branch_ids": deepcopy(base["selected_branch_ids"]),
        "included_sibling_branch_ids": deepcopy(base["included_sibling_branch_ids"]),
        "discovery_packets": discovery_packets,
        "summary": {
            "source_text_count": len(base["source_packets"]),
            "then_text_count": len(base["then_packets"]),
            "expected_discovery_model_calls": len(discovery_packets),
            "llm_generated_ids": 0,
            "llm_semantic_decisions": 0,
            "coverage_model_calls": 0,
        },
    }
    payload["packet_set_fingerprint"] = content_sha256(payload)
    return payload


def validate_gwt_span_discovery_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != DISCOVERY_PACKET_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.5 span-discovery packet-set schema")
    if fingerprint != content_sha256(result):
        raise ThenAtomizationError("v0.5 span-discovery packet-set fingerprint mismatch")
    packets = result.get("discovery_packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("v0.5 discovery_packets must be an array")
    target_ids = []
    for raw in packets:
        packet = dict(_mapping(raw, "$.discovery_packets[]"))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if packet.get("schema_version") != DISCOVERY_PACKET_VERSION or packet.get("task_name") != DISCOVERY_TASK:
            raise ThenAtomizationError("v0.5 discovery packet schema/task mismatch")
        if packet.get("text_kind") not in {"source_rule", "gwt_then"}:
            raise ThenAtomizationError("v0.5 text_kind is invalid")
        if packet_fingerprint != content_sha256(packet):
            raise ThenAtomizationError("v0.5 discovery packet fingerprint mismatch")
        if packet.get("task_input") != {
            "target_id": packet.get("target_id"),
            "text_kind": packet.get("text_kind"),
            "primary_text": packet.get("primary_text"),
        }:
            raise ThenAtomizationError("v0.5 discovery model input is not the minimal contract")
        target_ids.append(packet.get("target_id"))
    if len(target_ids) != len(set(target_ids)):
        raise ThenAtomizationError("v0.5 discovery target IDs must be unique")
    if result.get("summary", {}).get("expected_discovery_model_calls") != len(packets):
        raise ThenAtomizationError("v0.5 discovery call count mismatch")
    if result.get("summary", {}).get("llm_generated_ids") != 0:
        raise ThenAtomizationError("v0.5 discovery must not request IDs")
    if result.get("summary", {}).get("llm_semantic_decisions") != 0:
        raise ThenAtomizationError("v0.5 discovery must not request semantic decisions")
    result["packet_set_fingerprint"] = fingerprint
    return result


def render_gwt_span_discovery_prompt(packet: Mapping[str, Any], template: str) -> str:
    placeholder = "{proposition_span_discovery_input}"
    if packet.get("task_name") != DISCOVERY_TASK or placeholder not in template:
        raise ThenAtomizationError("v0.5 discovery prompt template/task mismatch")
    return template.replace(placeholder, json.dumps(packet["task_input"], ensure_ascii=False, indent=2))


def render_gwt_span_role_prompt(packet: Mapping[str, Any], template: str) -> str:
    placeholder = "{proposition_role_input}"
    if packet.get("task_name") != ROLE_TASK or placeholder not in template:
        raise ThenAtomizationError("v0.5 role prompt template/task mismatch")
    return template.replace(placeholder, json.dumps(packet["task_input"], ensure_ascii=False, indent=2))


def render_gwt_span_modality_prompt(packet: Mapping[str, Any], template: str) -> str:
    placeholder = "{proposition_modality_input}"
    if packet.get("task_name") != MODALITY_TASK or placeholder not in template:
        raise ThenAtomizationError("v0.5 modality prompt template/task mismatch")
    return template.replace(placeholder, json.dumps(packet["task_input"], ensure_ascii=False, indent=2))


def validate_proposition_span_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.span_discovery")))
    _exact_keys(result, {"target_id", "resolution_status", "proposition_spans", "reason"}, "$.span_discovery")
    if result.get("target_id") != packet.get("target_id"):
        raise ThenAtomizationError("span-discovery target_id mismatch")
    status = result.get("resolution_status")
    if status not in RESOLUTION_STATUSES:
        raise ThenAtomizationError("span-discovery resolution_status is invalid")
    _nonempty(result.get("reason"), "$.span_discovery.reason")
    spans = result.get("proposition_spans")
    if not isinstance(spans, list):
        raise ThenAtomizationError("proposition_spans must be an array")
    if status == "resolved" and not spans:
        raise ThenAtomizationError("resolved span discovery requires proposition spans")
    if status == "ambiguous" and spans:
        raise ThenAtomizationError("ambiguous span discovery must not select spans")
    checked_spans = []
    primary = packet["primary_text"]
    for index, raw in enumerate(spans):
        span = _nonempty(raw, f"$.span_discovery.proposition_spans[{index}]")
        if span not in primary:
            raise ThenAtomizationError(f"proposition span is not exact primary text: {span!r}")
        checked_spans.append(span)
    if len(checked_spans) != len(set(checked_spans)):
        raise ThenAtomizationError("proposition spans must not be duplicated")
    positions = [primary.find(span) for span in checked_spans]
    if positions != sorted(positions):
        raise ThenAtomizationError("proposition spans must follow primary-text order")
    result["proposition_spans"] = checked_spans
    return result


def validate_gwt_span_discovery_response_set(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_gwt_span_discovery_packet_set(packet_set)
    responses = deepcopy(dict(_mapping(response_set, "$.responses")))
    _exact_keys(responses, {"schema_version", "responses"}, "$.responses")
    if responses.get("schema_version") != DISCOVERY_RESPONSE_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.5 discovery response-set schema")
    packet_map = {item["target_id"]: item for item in packets["discovery_packets"]}
    checked = {}
    values = responses.get("responses")
    if not isinstance(values, list):
        raise ThenAtomizationError("v0.5 responses must be an array")
    for raw in values:
        target_id = raw.get("target_id") if isinstance(raw, Mapping) else None
        if target_id not in packet_map or target_id in checked:
            raise ThenAtomizationError(f"unknown or duplicate span response: {target_id}")
        checked[target_id] = validate_proposition_span_response(packet_map[target_id], raw)
    if set(checked) != set(packet_map):
        raise ThenAtomizationError("v0.5 span responses must cover the closed packet set")
    return {
        "schema_version": DISCOVERY_RESPONSE_SET_VERSION,
        "responses": [checked[item["target_id"]] for item in packets["discovery_packets"]],
    }


def build_gwt_span_role_packets(
    discovery_packet_set: Mapping[str, Any], discovery_response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_gwt_span_discovery_packet_set(discovery_packet_set)
    responses = validate_gwt_span_discovery_response_set(packets, discovery_response_set)
    packet_map = {item["target_id"]: item for item in packets["discovery_packets"]}
    role_packets = []
    for response in responses["responses"]:
        parent = packet_map[response["target_id"]]
        for index, span in enumerate(response["proposition_spans"], start=1):
            proposition = {"proposition_id": f"P{index:02d}", "exact_span": span}
            role_packets.append(
                _fingerprinted(
                    {
                        "schema_version": ROLE_PACKET_VERSION,
                        "task_name": ROLE_TASK,
                        "text_kind": parent["text_kind"],
                        "target_id": parent["target_id"],
                        "spec_id": parent["spec_id"],
                        "branch_id": parent["branch_id"],
                        "selected_for_test": parent["selected_for_test"],
                        "proposition_id": proposition["proposition_id"],
                        "task_contract": {
                            "question": "What single grammatical-policy role does this exact proposition span play in primary_text?",
                            "provided_inputs": ["text_kind", "primary_text", "proposition"],
                            "why_input_is_sufficient": "Role is decided from the proposition's relationship to its complete original sentence; no outside policy or GWT is needed.",
                            "deferred_decisions": ["modality", "normalization", "relations", "source-to-GWT mapping"],
                        },
                        "task_input": {
                            "target_id": parent["target_id"],
                            "text_kind": parent["text_kind"],
                            "primary_text": parent["primary_text"],
                            "proposition": proposition,
                        },
                    }
                )
            )
    result = {
        "schema_version": ROLE_PACKET_SET_VERSION,
        "discovery_packet_set_fingerprint": packets["packet_set_fingerprint"],
        "role_packets": role_packets,
        "summary": {
            "expected_role_model_calls": len(role_packets),
            "ids_assigned_mechanically": len(role_packets),
            "llm_decisions_per_call": 1,
            "coverage_model_calls": 0,
        },
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return result


def validate_gwt_span_role_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != ROLE_PACKET_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.5 role packet-set schema")
    if fingerprint != content_sha256(result):
        raise ThenAtomizationError("v0.5 role packet-set fingerprint mismatch")
    packets = result.get("role_packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("v0.5 role_packets must be an array")
    identities = []
    for raw in packets:
        packet = dict(_mapping(raw, "$.role_packets[]"))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if packet.get("schema_version") != ROLE_PACKET_VERSION or packet.get("task_name") != ROLE_TASK:
            raise ThenAtomizationError("v0.5 role packet schema/task mismatch")
        if packet_fingerprint != content_sha256(packet):
            raise ThenAtomizationError("v0.5 role packet fingerprint mismatch")
        task_input = _mapping(packet.get("task_input"), "$.role_packets[].task_input")
        proposition = _mapping(task_input.get("proposition"), "$.role_packets[].task_input.proposition")
        if set(task_input) != {"target_id", "text_kind", "primary_text", "proposition"}:
            raise ThenAtomizationError("v0.5 role model input is not the closed contract")
        if proposition.get("proposition_id") != packet.get("proposition_id"):
            raise ThenAtomizationError("v0.5 role proposition identity mismatch")
        if proposition.get("exact_span") not in task_input.get("primary_text", ""):
            raise ThenAtomizationError("v0.5 role proposition is not exact primary text")
        identities.append((packet.get("target_id"), packet.get("proposition_id")))
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("v0.5 role packet identities must be unique")
    if result.get("summary", {}).get("expected_role_model_calls") != len(packets):
        raise ThenAtomizationError("v0.5 role call count mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def validate_proposition_role_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.role_response")))
    _exact_keys(result, {"target_id", "proposition_id", "role", "reason"}, "$.role_response")
    if result.get("target_id") != packet.get("target_id") or result.get("proposition_id") != packet.get("proposition_id"):
        raise ThenAtomizationError("proposition-role identity mismatch")
    allowed = (SOURCE_ROLES if packet.get("text_kind") == "source_rule" else THEN_ROLES) | {"ambiguous"}
    if result.get("role") not in allowed:
        raise ThenAtomizationError("proposition role is invalid for this text kind")
    _nonempty(result.get("reason"), "$.role_response.reason")
    return result


def validate_gwt_span_role_response_set(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_gwt_span_role_packet_set(packet_set)
    responses = deepcopy(dict(_mapping(response_set, "$.role_responses")))
    _exact_keys(responses, {"schema_version", "responses"}, "$.role_responses")
    if responses.get("schema_version") != ROLE_RESPONSE_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.5 role response-set schema")
    packet_map = {
        (item["target_id"], item["proposition_id"]): item for item in packets["role_packets"]
    }
    checked = {}
    values = responses.get("responses")
    if not isinstance(values, list):
        raise ThenAtomizationError("v0.5 role responses must be an array")
    for raw in values:
        key = (raw.get("target_id"), raw.get("proposition_id")) if isinstance(raw, Mapping) else (None, None)
        if key not in packet_map or key in checked:
            raise ThenAtomizationError(f"unknown or duplicate role response: {key}")
        checked[key] = validate_proposition_role_response(packet_map[key], raw)
    if set(checked) != set(packet_map):
        raise ThenAtomizationError("v0.5 role responses must cover the closed packet set")
    return {
        "schema_version": ROLE_RESPONSE_SET_VERSION,
        "responses": [
            checked[(item["target_id"], item["proposition_id"])]
            for item in packets["role_packets"]
        ],
    }


def build_gwt_span_modality_packets(
    role_packet_set: Mapping[str, Any], role_response_set: Mapping[str, Any]
) -> dict[str, Any]:
    role_set = validate_gwt_span_role_packet_set(role_packet_set)
    responses = validate_gwt_span_role_response_set(role_set, role_response_set)
    fingerprint = role_set["packet_set_fingerprint"]
    packets = role_set["role_packets"]
    packet_map = {(item["target_id"], item["proposition_id"]): item for item in packets}
    checked = {
        (item["target_id"], item["proposition_id"]): item for item in responses["responses"]
    }
    modality_packets = []
    for key, packet in packet_map.items():
        role = checked[key]["role"]
        if role not in {"governed_outcome", "asserted_outcome"}:
            continue
        task_input = {
            "target_id": packet["target_id"],
            "text_kind": packet["text_kind"],
            "primary_text": packet["task_input"]["primary_text"],
            "proposition": deepcopy(packet["task_input"]["proposition"]),
            "confirmed_role": role,
        }
        modality_packets.append(
            _fingerprinted(
                {
                    "schema_version": MODALITY_PACKET_VERSION,
                    "task_name": MODALITY_TASK,
                    "text_kind": packet["text_kind"],
                    "target_id": packet["target_id"],
                    "proposition_id": packet["proposition_id"],
                    "task_contract": {
                        "question": "What single deontic modality applies to this confirmed governed/asserted proposition?",
                        "provided_inputs": ["text_kind", "primary_text", "proposition", "confirmed_role"],
                        "why_input_is_sufficient": "The complete original sentence supplies the modal cue and scope; role has already been fixed.",
                        "deferred_decisions": ["normalization", "relations", "source-to-GWT mapping"],
                    },
                    "task_input": task_input,
                }
            )
        )
    result = {
        "schema_version": MODALITY_PACKET_SET_VERSION,
        "role_packet_set_fingerprint": fingerprint,
        "modality_packets": modality_packets,
        "summary": {
            "expected_modality_model_calls": len(modality_packets),
            "llm_decisions_per_call": 1,
            "coverage_model_calls": 0,
        },
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return result


def validate_gwt_span_modality_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != MODALITY_PACKET_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.5 modality packet-set schema")
    if fingerprint != content_sha256(result):
        raise ThenAtomizationError("v0.5 modality packet-set fingerprint mismatch")
    packets = result.get("modality_packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("v0.5 modality_packets must be an array")
    identities = []
    for raw in packets:
        packet = dict(_mapping(raw, "$.modality_packets[]"))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if packet.get("schema_version") != MODALITY_PACKET_VERSION or packet.get("task_name") != MODALITY_TASK:
            raise ThenAtomizationError("v0.5 modality packet schema/task mismatch")
        if packet_fingerprint != content_sha256(packet):
            raise ThenAtomizationError("v0.5 modality packet fingerprint mismatch")
        if set(packet.get("task_input", {})) != {
            "target_id", "text_kind", "primary_text", "proposition", "confirmed_role"
        }:
            raise ThenAtomizationError("v0.5 modality model input is not the closed contract")
        identities.append((packet.get("target_id"), packet.get("proposition_id")))
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("v0.5 modality packet identities must be unique")
    if result.get("summary", {}).get("expected_modality_model_calls") != len(packets):
        raise ThenAtomizationError("v0.5 modality call count mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def validate_proposition_modality_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.modality_response")))
    _exact_keys(result, {"target_id", "proposition_id", "modality", "cue_spans", "reason"}, "$.modality_response")
    if result.get("target_id") != packet.get("target_id") or result.get("proposition_id") != packet.get("proposition_id"):
        raise ThenAtomizationError("proposition-modality identity mismatch")
    if result.get("modality") not in MODALITY_LABELS:
        raise ThenAtomizationError("proposition modality is invalid")
    primary = packet["task_input"]["primary_text"]
    cues = result.get("cue_spans")
    if not isinstance(cues, list):
        raise ThenAtomizationError("modality cue_spans must be an array")
    if result.get("modality") != "ambiguous" and not cues:
        raise ThenAtomizationError("resolved modality cue_spans must be non-empty")
    for index, cue in enumerate(cues):
        text = _nonempty(cue, f"$.modality_response.cue_spans[{index}]")
        if text not in primary:
            raise ThenAtomizationError("modality cue span is not exact primary text")
    _nonempty(result.get("reason"), "$.modality_response.reason")
    return result


def validate_gwt_span_modality_response_set(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_gwt_span_modality_packet_set(packet_set)
    responses = deepcopy(dict(_mapping(response_set, "$.modality_responses")))
    _exact_keys(responses, {"schema_version", "responses"}, "$.modality_responses")
    if responses.get("schema_version") != MODALITY_RESPONSE_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.5 modality response-set schema")
    packet_map = {
        (item["target_id"], item["proposition_id"]): item
        for item in packets["modality_packets"]
    }
    checked = {}
    values = responses.get("responses")
    if not isinstance(values, list):
        raise ThenAtomizationError("v0.5 modality responses must be an array")
    for raw in values:
        key = (raw.get("target_id"), raw.get("proposition_id")) if isinstance(raw, Mapping) else (None, None)
        if key not in packet_map or key in checked:
            raise ThenAtomizationError(f"unknown or duplicate modality response: {key}")
        checked[key] = validate_proposition_modality_response(packet_map[key], raw)
    if set(checked) != set(packet_map):
        raise ThenAtomizationError("v0.5 modality responses must cover the closed packet set")
    return {
        "schema_version": MODALITY_RESPONSE_SET_VERSION,
        "responses": [
            checked[(item["target_id"], item["proposition_id"])]
            for item in packets["modality_packets"]
        ],
    }


def build_gwt_span_discovery_packets_file(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    selected_branch_ids: Sequence[str],
) -> dict[str, Any]:
    source = Path(spec_path)
    result = build_gwt_span_discovery_packets(
        json.loads(source.read_text(encoding="utf-8")),
        selected_branch_ids=selected_branch_ids,
        source_path=str(source.resolve()),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
