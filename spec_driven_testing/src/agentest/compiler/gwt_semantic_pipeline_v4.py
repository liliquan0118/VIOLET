"""Candidate-first GWT semantic compilation.

Version 0.4 removes modality, condition, role, relation, and local-ID decisions
from the first model wave.  Models only discover positive behavior/event
candidates with exact evidence.  Stable IDs and every later decision are
separate compiler stages.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .gwt_semantic_pipeline_v3 import build_gwt_semantic_stage1_packets
from .then_atomization import MODALITIES, RESOLUTION_STATUSES, ThenAtomizationError


DISCOVERY_PACKET_SET_VERSION = "agentspectesting.gwt-semantic-discovery-packets/v0.4"
SOURCE_DISCOVERY_PACKET_VERSION = "agentspectesting.source-behavior-discovery-packet/v0.4"
THEN_DISCOVERY_PACKET_VERSION = "agentspectesting.then-entity-discovery-packet/v0.4"
DISCOVERY_RESPONSE_SET_VERSION = "agentspectesting.gwt-semantic-discovery-responses/v0.4"
DECISION_PACKET_SET_VERSION = "agentspectesting.gwt-semantic-decision-packets/v0.4"
SOURCE_CASE_PACKET_VERSION = "agentspectesting.source-behavior-cases-packet/v0.4"
THEN_ENTITY_PACKET_VERSION = "agentspectesting.then-entity-classification-packet/v0.4"
DECISION_RESPONSE_SET_VERSION = "agentspectesting.gwt-semantic-decision-responses/v0.4"

SOURCE_DISCOVERY_TASK = "source_behavior_discovery"
THEN_DISCOVERY_TASK = "then_entity_discovery"
SOURCE_CASE_TASK = "source_behavior_case_classification"
THEN_ENTITY_TASK = "then_entity_classification"

REFERENCE_ROLES = frozenset({"condition", "temporal_anchor", "scope_anchor"})
CASE_DERIVATIONS = frozenset({"explicit", "biconditional_complement"})


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


def _source_pool(context: Mapping[str, Any]) -> list[str]:
    values = [context.get("rule_text", "")]
    values.extend(context.get("clauses") or [])
    values.extend(
        item.get("quote", "")
        for item in context.get("evidence") or []
        if isinstance(item, Mapping)
    )
    return [value for value in values if isinstance(value, str) and value]


def _spans(value: Any, pool: Sequence[str], path: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise ThenAtomizationError(f"{path} must be {qualifier}")
    result = []
    for index, raw in enumerate(value):
        span = _nonempty(raw, f"{path}[{index}]")
        if not any(span in source for source in pool):
            raise ThenAtomizationError(f"{path}[{index}] is not an exact supplied substring")
        result.append(span)
    if len(result) != len(set(result)):
        raise ThenAtomizationError(f"{path} must not contain duplicates")
    return result


def _fingerprinted(value: dict[str, Any]) -> dict[str, Any]:
    value["packet_fingerprint"] = content_sha256(value)
    return value


def build_gwt_semantic_discovery_packets(
    spec_document: Mapping[str, Any],
    *,
    selected_branch_ids: Sequence[str],
    source_path: str | None = None,
) -> dict[str, Any]:
    """Prepare ID-free behavior/entity discovery tasks for all needed siblings."""

    base = build_gwt_semantic_stage1_packets(
        spec_document,
        selected_branch_ids=selected_branch_ids,
        source_path=source_path,
    )
    source_packets = []
    for packet in base["source_packets"]:
        task_input = {"spec_id": packet["spec_id"], "source_context": deepcopy(packet["source_context"])}
        source_packets.append(
            _fingerprinted(
                {
                    "schema_version": SOURCE_DISCOVERY_PACKET_VERSION,
                    "task_name": SOURCE_DISCOVERY_TASK,
                    "spec_id": packet["spec_id"],
                    "task_contract": {
                        "question": "Which independently assessable positive behaviors or states are governed by this source anchor?",
                        "provided_inputs": ["source_context"],
                        "deferred_decisions": [
                            "modality and polarity",
                            "activation conditions",
                            "biconditional expansion",
                            "GWT support",
                            "relations",
                        ],
                    },
                    "task_input": task_input,
                }
            )
        )
    then_packets = []
    for packet in base["then_packets"]:
        task_input = {"branch_id": packet["branch_id"], "gwt_context": deepcopy(packet["gwt_context"])}
        then_packets.append(
            _fingerprinted(
                {
                    "schema_version": THEN_DISCOVERY_PACKET_VERSION,
                    "task_name": THEN_DISCOVERY_TASK,
                    "spec_id": packet["spec_id"],
                    "branch_id": packet["branch_id"],
                    "selected_for_test": packet["selected_for_test"],
                    "task_contract": {
                        "question": "Which independently assessable positive event or state candidates are mentioned in this Then?",
                        "provided_inputs": ["gwt_context"],
                        "deferred_decisions": [
                            "asserted versus reference-only role",
                            "modality and polarity",
                            "source support",
                            "relations",
                            "test observability",
                        ],
                    },
                    "task_input": task_input,
                }
            )
        )
    payload = {
        "schema_version": DISCOVERY_PACKET_SET_VERSION,
        "source": deepcopy(base["source"]),
        "selected_branch_ids": deepcopy(base["selected_branch_ids"]),
        "included_sibling_branch_ids": deepcopy(base["included_sibling_branch_ids"]),
        "source_discovery_packets": source_packets,
        "then_discovery_packets": then_packets,
        "summary": {
            "selected_spec_count": len(source_packets),
            "selected_branch_count": len(base["selected_branch_ids"]),
            "included_sibling_branch_count": len(then_packets),
            "expected_discovery_model_calls": len(source_packets) + len(then_packets),
            "calls_by_task": {
                SOURCE_DISCOVERY_TASK: len(source_packets),
                THEN_DISCOVERY_TASK: len(then_packets),
            },
            "llm_generated_ids": 0,
            "coverage_model_calls": 0,
        },
    }
    payload["packet_set_fingerprint"] = content_sha256(payload)
    return payload


def validate_gwt_semantic_discovery_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != DISCOVERY_PACKET_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.4 discovery packet-set schema")
    if fingerprint != content_sha256(result):
        raise ThenAtomizationError("v0.4 discovery packet-set fingerprint mismatch")
    source_packets = result.get("source_discovery_packets")
    then_packets = result.get("then_discovery_packets")
    if not isinstance(source_packets, list) or not isinstance(then_packets, list):
        raise ThenAtomizationError("v0.4 discovery packet arrays are missing")
    for version, task, packets in (
        (SOURCE_DISCOVERY_PACKET_VERSION, SOURCE_DISCOVERY_TASK, source_packets),
        (THEN_DISCOVERY_PACKET_VERSION, THEN_DISCOVERY_TASK, then_packets),
    ):
        for raw in packets:
            packet = dict(_mapping(raw, "$.discovery_packets[]"))
            packet_fingerprint = packet.pop("packet_fingerprint", None)
            if packet.get("schema_version") != version or packet.get("task_name") != task:
                raise ThenAtomizationError("v0.4 discovery task schema/name mismatch")
            if packet_fingerprint != content_sha256(packet):
                raise ThenAtomizationError("v0.4 discovery task fingerprint mismatch")
    if result.get("summary", {}).get("expected_discovery_model_calls") != len(source_packets) + len(then_packets):
        raise ThenAtomizationError("v0.4 discovery call count mismatch")
    if result.get("summary", {}).get("llm_generated_ids") != 0:
        raise ThenAtomizationError("v0.4 discovery must not ask the model for IDs")
    result["packet_set_fingerprint"] = fingerprint
    return result


def render_gwt_semantic_discovery_prompt(packet: Mapping[str, Any], template: str) -> str:
    placeholder = {
        SOURCE_DISCOVERY_TASK: "{source_behavior_discovery_input}",
        THEN_DISCOVERY_TASK: "{then_entity_discovery_input}",
    }.get(packet.get("task_name"))
    if placeholder is None or placeholder not in template:
        raise ThenAtomizationError("v0.4 discovery prompt template/task mismatch")
    return template.replace(placeholder, json.dumps(packet["task_input"], ensure_ascii=False, indent=2))


def validate_source_behavior_discovery_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.source_behavior_discovery")))
    _exact_keys(result, {"spec_id", "resolution_status", "behaviors", "reason"}, "$.source_behavior_discovery")
    if result.get("spec_id") != packet.get("spec_id"):
        raise ThenAtomizationError("source behavior discovery spec_id mismatch")
    status = result.get("resolution_status")
    if status not in RESOLUTION_STATUSES:
        raise ThenAtomizationError("source behavior discovery status is invalid")
    _nonempty(result.get("reason"), "$.source_behavior_discovery.reason")
    behaviors = result.get("behaviors")
    if not isinstance(behaviors, list):
        raise ThenAtomizationError("source behaviors must be an array")
    if status == "resolved" and not behaviors:
        raise ThenAtomizationError("resolved source discovery requires behaviors")
    if status == "ambiguous" and behaviors:
        raise ThenAtomizationError("ambiguous source discovery must not select behaviors")
    pool = _source_pool(packet["task_input"]["source_context"])
    names = []
    for index, raw in enumerate(behaviors):
        path = f"$.source_behavior_discovery.behaviors[{index}]"
        item = _mapping(raw, path)
        _exact_keys(item, {"behavior", "evidence_spans"}, path)
        names.append(_nonempty(item.get("behavior"), f"{path}.behavior").casefold())
        item["evidence_spans"] = _spans(item.get("evidence_spans"), pool, f"{path}.evidence_spans")
    if len(names) != len(set(names)):
        raise ThenAtomizationError("source behaviors must not be duplicated")
    return result


def validate_then_entity_discovery_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.then_entity_discovery")))
    _exact_keys(result, {"branch_id", "resolution_status", "entities", "reason"}, "$.then_entity_discovery")
    if result.get("branch_id") != packet.get("branch_id"):
        raise ThenAtomizationError("Then entity discovery branch_id mismatch")
    status = result.get("resolution_status")
    if status not in RESOLUTION_STATUSES:
        raise ThenAtomizationError("Then entity discovery status is invalid")
    _nonempty(result.get("reason"), "$.then_entity_discovery.reason")
    entities = result.get("entities")
    if not isinstance(entities, list):
        raise ThenAtomizationError("Then entities must be an array")
    if status == "resolved" and not entities:
        raise ThenAtomizationError("resolved Then discovery requires entities")
    if status == "ambiguous" and entities:
        raise ThenAtomizationError("ambiguous Then discovery must not select entities")
    then = packet["task_input"]["gwt_context"]["then"]
    names = []
    for index, raw in enumerate(entities):
        path = f"$.then_entity_discovery.entities[{index}]"
        item = _mapping(raw, path)
        _exact_keys(item, {"event_or_state", "evidence_spans"}, path)
        names.append(_nonempty(item.get("event_or_state"), f"{path}.event_or_state").casefold())
        item["evidence_spans"] = _spans(item.get("evidence_spans"), [then], f"{path}.evidence_spans")
    if len(names) != len(set(names)):
        raise ThenAtomizationError("Then entities must not be duplicated")
    return result


def validate_gwt_semantic_discovery_response_set(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_gwt_semantic_discovery_packet_set(packet_set)
    responses = deepcopy(dict(_mapping(response_set, "$.discovery_responses")))
    _exact_keys(responses, {"schema_version", "source_responses", "then_responses"}, "$.discovery_responses")
    if responses.get("schema_version") != DISCOVERY_RESPONSE_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.4 discovery response-set schema")
    source_packet_map = {item["spec_id"]: item for item in packets["source_discovery_packets"]}
    then_packet_map = {item["branch_id"]: item for item in packets["then_discovery_packets"]}
    checked_sources = {}
    for raw in responses.get("source_responses") or []:
        spec_id = raw.get("spec_id") if isinstance(raw, Mapping) else None
        if spec_id not in source_packet_map or spec_id in checked_sources:
            raise ThenAtomizationError(f"unknown or duplicate source discovery response: {spec_id}")
        checked_sources[spec_id] = validate_source_behavior_discovery_response(source_packet_map[spec_id], raw)
    checked_thens = {}
    for raw in responses.get("then_responses") or []:
        branch_id = raw.get("branch_id") if isinstance(raw, Mapping) else None
        if branch_id not in then_packet_map or branch_id in checked_thens:
            raise ThenAtomizationError(f"unknown or duplicate Then discovery response: {branch_id}")
        checked_thens[branch_id] = validate_then_entity_discovery_response(then_packet_map[branch_id], raw)
    if set(checked_sources) != set(source_packet_map) or set(checked_thens) != set(then_packet_map):
        raise ThenAtomizationError("v0.4 discovery responses must cover the closed packet set")
    return {
        "schema_version": DISCOVERY_RESPONSE_SET_VERSION,
        "source_responses": [checked_sources[item["spec_id"]] for item in packets["source_discovery_packets"]],
        "then_responses": [checked_thens[item["branch_id"]] for item in packets["then_discovery_packets"]],
    }


def build_gwt_semantic_decision_packets(
    discovery_packet_set: Mapping[str, Any], discovery_response_set: Mapping[str, Any]
) -> dict[str, Any]:
    """Assign stable IDs mechanically and prepare one decision per candidate."""

    packets = validate_gwt_semantic_discovery_packet_set(discovery_packet_set)
    responses = validate_gwt_semantic_discovery_response_set(packets, discovery_response_set)
    source_packet_map = {item["spec_id"]: item for item in packets["source_discovery_packets"]}
    then_packet_map = {item["branch_id"]: item for item in packets["then_discovery_packets"]}
    source_case_packets = []
    for response in responses["source_responses"]:
        packet = source_packet_map[response["spec_id"]]
        for index, behavior in enumerate(response["behaviors"], start=1):
            behavior_with_id = {"behavior_id": f"B{index:02d}", **deepcopy(behavior)}
            source_case_packets.append(
                _fingerprinted(
                    {
                        "schema_version": SOURCE_CASE_PACKET_VERSION,
                        "task_name": SOURCE_CASE_TASK,
                        "spec_id": response["spec_id"],
                        "behavior_id": behavior_with_id["behavior_id"],
                        "task_contract": {
                            "question": "For this one positive behavior, which normative cases does the source explicitly state?",
                            "provided_inputs": ["source_context", "behavior"],
                            "deferred_decisions": ["GWT mapping", "relations", "test observability"],
                        },
                        "task_input": {
                            "spec_id": response["spec_id"],
                            "source_context": deepcopy(packet["task_input"]["source_context"]),
                            "behavior": behavior_with_id,
                        },
                    }
                )
            )
    then_entity_packets = []
    for response in responses["then_responses"]:
        packet = then_packet_map[response["branch_id"]]
        for index, entity in enumerate(response["entities"], start=1):
            entity_with_id = {"entity_id": f"E{index:02d}", **deepcopy(entity)}
            then_entity_packets.append(
                _fingerprinted(
                    {
                        "schema_version": THEN_ENTITY_PACKET_VERSION,
                        "task_name": THEN_ENTITY_TASK,
                        "spec_id": packet["spec_id"],
                        "branch_id": response["branch_id"],
                        "entity_id": entity_with_id["entity_id"],
                        "selected_for_test": packet["selected_for_test"],
                        "task_contract": {
                            "question": "For this one positive Then entity, is it asserted or reference-only, and what single modality/role applies?",
                            "provided_inputs": ["gwt_context", "entity"],
                            "deferred_decisions": ["source support", "relations", "test observability"],
                        },
                        "task_input": {
                            "branch_id": response["branch_id"],
                            "gwt_context": deepcopy(packet["task_input"]["gwt_context"]),
                            "entity": entity_with_id,
                        },
                    }
                )
            )
    result = {
        "schema_version": DECISION_PACKET_SET_VERSION,
        "discovery_packet_set_fingerprint": packets["packet_set_fingerprint"],
        "source_case_packets": source_case_packets,
        "then_entity_packets": then_entity_packets,
        "summary": {
            "expected_decision_model_calls": len(source_case_packets) + len(then_entity_packets),
            "calls_by_task": {
                SOURCE_CASE_TASK: len(source_case_packets),
                THEN_ENTITY_TASK: len(then_entity_packets),
            },
            "ids_assigned_mechanically": len(source_case_packets) + len(then_entity_packets),
            "coverage_model_calls": 0,
        },
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return result


def validate_source_behavior_case_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.source_behavior_cases")))
    _exact_keys(result, {"spec_id", "behavior_id", "resolution_status", "cases", "reason"}, "$.source_behavior_cases")
    if result.get("spec_id") != packet.get("spec_id") or result.get("behavior_id") != packet.get("behavior_id"):
        raise ThenAtomizationError("source behavior-case identity mismatch")
    if result.get("resolution_status") not in RESOLUTION_STATUSES:
        raise ThenAtomizationError("source behavior-case status is invalid")
    _nonempty(result.get("reason"), "$.source_behavior_cases.reason")
    cases = result.get("cases")
    if not isinstance(cases, list):
        raise ThenAtomizationError("source behavior cases must be an array")
    if result["resolution_status"] == "resolved" and not cases:
        raise ThenAtomizationError("resolved source behavior requires cases")
    if result["resolution_status"] == "ambiguous" and cases:
        raise ThenAtomizationError("ambiguous source behavior must not choose cases")
    pool = _source_pool(packet["task_input"]["source_context"])
    for index, raw in enumerate(cases):
        path = f"$.source_behavior_cases.cases[{index}]"
        item = _mapping(raw, path)
        _exact_keys(item, {"modality", "condition", "evidence_spans", "derivation"}, path)
        if item.get("modality") not in MODALITIES:
            raise ThenAtomizationError(f"{path}.modality is invalid")
        _nonempty(item.get("condition"), f"{path}.condition")
        if item.get("derivation") not in CASE_DERIVATIONS:
            raise ThenAtomizationError(f"{path}.derivation is invalid")
        item["evidence_spans"] = _spans(item.get("evidence_spans"), pool, f"{path}.evidence_spans")
    return result


def validate_then_entity_classification_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.then_entity_classification")))
    _exact_keys(
        result,
        {"branch_id", "entity_id", "role", "modality", "reference_role", "reason"},
        "$.then_entity_classification",
    )
    if result.get("branch_id") != packet.get("branch_id") or result.get("entity_id") != packet.get("entity_id"):
        raise ThenAtomizationError("Then entity classification identity mismatch")
    role = result.get("role")
    if role == "asserted":
        if result.get("modality") not in MODALITIES or result.get("reference_role") is not None:
            raise ThenAtomizationError("asserted entity requires one modality and no reference role")
    elif role == "reference":
        if result.get("modality") is not None or result.get("reference_role") not in REFERENCE_ROLES:
            raise ThenAtomizationError("reference entity requires one reference role and no modality")
    else:
        raise ThenAtomizationError("Then entity role is invalid")
    _nonempty(result.get("reason"), "$.then_entity_classification.reason")
    return result


def build_gwt_semantic_discovery_packets_file(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    selected_branch_ids: Sequence[str],
) -> dict[str, Any]:
    source = Path(spec_path)
    result = build_gwt_semantic_discovery_packets(
        json.loads(source.read_text(encoding="utf-8")),
        selected_branch_ids=selected_branch_ids,
        source_path=str(source.resolve()),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
