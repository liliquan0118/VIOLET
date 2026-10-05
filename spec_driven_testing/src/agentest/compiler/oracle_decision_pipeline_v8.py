"""Minimal model-facing decisions over rich v0.7 oracle candidates."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .oracle_requirement_pipeline_v7 import validate_oracle_requirement_packet_set
from .then_atomization import ThenAtomizationError


PACKET_SET_VERSION = "agentspectesting.oracle-decision-packets/v0.8"
PACKET_VERSION = "agentspectesting.oracle-decision-packet/v0.8"
RESPONSE_SET_VERSION = "agentspectesting.oracle-decision-responses/v0.8"
TASK_NAME = "oracle_requirement_membership"
DECISIONS = frozenset({"yes", "no", "ambiguous"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _candidate_observation(candidate: Mapping[str, Any]) -> str:
    kind = candidate.get("candidate_kind")
    contract = _mapping(candidate.get("observation_contract") or {}, "$.observation_contract")
    text = str(candidate.get("requirement_text") or "").strip().rstrip(".")
    tool_name = contract.get("tool_name")
    action = str(contract.get("action_description") or "").strip().rstrip(".")
    if not action and tool_name:
        action = str(tool_name).replace("_", " ")
    article = "an" if str(tool_name).casefold().startswith(tuple("aeiou")) else "a"
    if kind == "tool_call":
        return (
            f"Whether the agent performs the action represented by {article} "
            f"{tool_name} tool call ({action})."
        )
    if kind == "tool_argument":
        return (
            f"In the agent action represented by {article} {tool_name} tool call "
            f"({action}), the value of the {contract.get('parameter')} argument."
        )
    if kind == "tool_argument_constraint":
        # Sorted so the rendered question (and therefore this key) is
        # independent of which order the candidate generator happened to list
        # the parameters in -- e.g. a full-catalog-derived binding and an
        # explicit/legacy-derived binding for the same (tool, parameter set)
        # would otherwise render two differently-ordered, non-matching
        # questions for what is really the same judgment.
        parameters = ", ".join(sorted(contract.get("parameters") or []))
        constraint = str(contract.get("constraint_text") or text).strip().rstrip(".")
        return (
            f"Whether the agent performs the action represented by {article} {tool_name} "
            f"tool call ({action}) with values of {parameters} that satisfy or "
            f"violate this constraint: "
            f"{constraint}."
        )
    if kind == "assistant_literal":
        return f"Whether the assistant message contains exactly: {contract.get('literal')}"
    if kind == "temporal_relation":
        return text + "."
    unit_kind = (candidate.get("source_refs") or {}).get("semantic_unit_kind")
    if unit_kind == "action_commitment":
        return f"Whether the agent performs this behavior: {text}."
    if unit_kind == "scope_condition_relation":
        return f"Whether this condition holds: {text}."
    if kind == "unbound_branch_assertion":
        return f"Whether the agent satisfies this requirement: {text}."
    return f"Whether the oracle observes this behavior or condition: {text}."


def build_oracle_decision_packets(
    candidate_packet_set: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = validate_oracle_requirement_packet_set(candidate_packet_set)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for source in candidates["packets"]:
        task_input = _mapping(source.get("task_input"), "$.source.task_input")
        context = _mapping(task_input.get("branch_context"), "$.branch_context")
        gwt = _mapping(context.get("gwt"), "$.branch_context.gwt")
        candidate = _mapping(task_input.get("proposed_observation"), "$.proposed_observation")
        model_input = {
            "then_requirement": gwt.get("then"),
            "candidate_observation": _candidate_observation(candidate),
        }
        member = {
            "branch_id": source["branch_id"],
            "candidate_id": source["candidate_id"],
            "source_candidate_packet_fingerprint": source["packet_fingerprint"],
            "candidate_record": deepcopy(dict(candidate)),
        }
        key = (model_input["then_requirement"], model_input["candidate_observation"])
        if key in grouped:
            grouped[key]["members"].append(member)
            continue
        packet = {
            "schema_version": PACKET_VERSION,
            "task_name": TASK_NAME,
            # The first member remains at the top level for compatibility and
            # for stable filenames. Only model_input is rendered to the model.
            **deepcopy(member),
            "members": [member],
            "model_input": model_input,
        }
        grouped[key] = packet

    packets = list(grouped.values())
    for packet in packets:
        packet["packet_fingerprint"] = content_sha256(packet)
    candidate_count = len(candidates["packets"])
    result = {
        "schema_version": PACKET_SET_VERSION,
        "source_candidate_packet_set_fingerprint": candidates["packet_set_fingerprint"],
        "packets": packets,
        "summary": {
            "candidate_count": candidate_count,
            "unique_model_input_count": len(packets),
            "expected_model_calls": len(packets),
            "deduplicated_call_count": candidate_count - len(packets),
            "model_input_fields": ["then_requirement", "candidate_observation"],
            "model_output_fields": ["decision", "reason"],
        },
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return result


def decision_packet_members(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every local candidate identity represented by one model call.

    Packets produced before pre-call deduplication have no ``members`` field;
    treating their top-level envelope as one member keeps archived runs usable.
    """

    raw_members = packet.get("members")
    if raw_members is None:
        raw_members = [
            {
                "branch_id": packet.get("branch_id"),
                "candidate_id": packet.get("candidate_id"),
                "source_candidate_packet_fingerprint": packet.get(
                    "source_candidate_packet_fingerprint"
                ),
                "candidate_record": packet.get("candidate_record"),
            }
        ]
    if not isinstance(raw_members, list) or not raw_members:
        raise ThenAtomizationError("v0.8 packet members must be a non-empty array")
    members = []
    for raw in raw_members:
        member = deepcopy(dict(_mapping(raw, "$.packet.members[]")))
        if set(member) != {
            "branch_id",
            "candidate_id",
            "source_candidate_packet_fingerprint",
            "candidate_record",
        }:
            raise ThenAtomizationError("v0.8 packet member fields are invalid")
        if not isinstance(member["branch_id"], str) or not member["branch_id"]:
            raise ThenAtomizationError("v0.8 packet member branch_id is invalid")
        if not isinstance(member["candidate_id"], str) or not member["candidate_id"]:
            raise ThenAtomizationError("v0.8 packet member candidate_id is invalid")
        if not isinstance(member["source_candidate_packet_fingerprint"], str):
            raise ThenAtomizationError("v0.8 packet member lineage is invalid")
        _mapping(member["candidate_record"], "$.packet.member.candidate_record")
        members.append(member)
    return members


def validate_oracle_decision_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != PACKET_SET_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid v0.8 oracle decision packet set")
    packets = result.get("packets")
    if not isinstance(packets, list) or not packets:
        raise ThenAtomizationError("v0.8 packets must be a non-empty array")
    identities = []
    member_count = 0
    for raw in packets:
        packet = dict(_mapping(raw, "$.packets[]"))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if packet.get("schema_version") != PACKET_VERSION or packet.get("task_name") != TASK_NAME:
            raise ThenAtomizationError("v0.8 packet schema/task mismatch")
        if packet_fingerprint != content_sha256(packet):
            raise ThenAtomizationError("v0.8 packet fingerprint mismatch")
        model_input = _mapping(packet.get("model_input"), "$.packet.model_input")
        if set(model_input) != {"then_requirement", "candidate_observation"}:
            raise ThenAtomizationError("v0.8 model input is not minimal and closed")
        if not all(isinstance(model_input[key], str) and model_input[key].strip() for key in model_input):
            raise ThenAtomizationError("v0.8 model input fields must be non-empty strings")
        members = decision_packet_members(packet)
        representative = members[0]
        for field in (
            "branch_id",
            "candidate_id",
            "source_candidate_packet_fingerprint",
            "candidate_record",
        ):
            if packet.get(field) != representative[field]:
                raise ThenAtomizationError("v0.8 packet representative/member mismatch")
        identities.extend(
            (member["branch_id"], member["candidate_id"]) for member in members
        )
        member_count += len(members)
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("v0.8 packet identities must be unique")
    if result.get("summary", {}).get("expected_model_calls") != len(packets):
        raise ThenAtomizationError("v0.8 call count mismatch")
    summary = result.get("summary", {})
    if summary.get("candidate_count") != member_count:
        raise ThenAtomizationError("v0.8 candidate/member count mismatch")
    if "unique_model_input_count" in summary and summary["unique_model_input_count"] != len(packets):
        raise ThenAtomizationError("v0.8 unique model input count mismatch")
    if "deduplicated_call_count" in summary and summary["deduplicated_call_count"] != member_count - len(packets):
        raise ThenAtomizationError("v0.8 deduplicated call count mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def render_oracle_decision_prompt(packet: Mapping[str, Any], template: str) -> str:
    placeholder = "{oracle_decision_input}"
    if packet.get("task_name") != TASK_NAME or placeholder not in template:
        raise ThenAtomizationError("v0.8 prompt template/task mismatch")
    return template.replace(
        placeholder,
        json.dumps(packet["model_input"], ensure_ascii=False, indent=2),
    )


def validate_oracle_decision_response(response: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.oracle_decision_response")))
    if set(result) != {"decision", "reason"}:
        raise ThenAtomizationError("v0.8 response fields are invalid")
    if result.get("decision") not in DECISIONS:
        raise ThenAtomizationError("v0.8 decision is invalid")
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        raise ThenAtomizationError("v0.8 reason must be non-empty")
    return result


def attach_oracle_decision_identity(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    checked = validate_oracle_decision_response(response)
    return {
        "branch_id": packet["branch_id"],
        "candidate_id": packet["candidate_id"],
        **checked,
    }


def attach_oracle_decision_identities(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Propagate one model decision to every exactly equivalent source candidate."""

    checked = validate_oracle_decision_response(response)
    return [
        {
            "branch_id": member["branch_id"],
            "candidate_id": member["candidate_id"],
            **checked,
        }
        for member in decision_packet_members(packet)
    ]
