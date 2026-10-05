"""Build and validate model packets for deferred semantic Oracle judgments.

The functions in this module stop at the model boundary.  They do not perform
network calls.  Model-facing inputs are deliberately split into one small
decision per packet.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from ..compiler.artifacts import content_sha256
from ..compiler.runtime_observation_binding_v1 import normalize_tau_execution
from ..compiler.semantic_judge_contract_v1 import (
    validate_semantic_judge_contract_set,
)
from ..compiler.then_atomization import ThenAtomizationError


PACKET_SET_VERSION = "agentspectesting.semantic-judge-packet-set/v0.1"
PACKET_VERSION = "agentspectesting.semantic-judge-packet/v0.1"
CLAIM_RESULT_VERSION = "agentspectesting.semantic-claim-result/v0.1"
TASK_CLAIM_ATOMIZATION = "semantic_claim_atomization"
TASK_CLAIM_SUPPORT = "semantic_claim_support"
TASK_SUBJECTIVITY = "semantic_subjectivity"
TASKS = frozenset(
    {TASK_CLAIM_ATOMIZATION, TASK_CLAIM_SUPPORT, TASK_SUBJECTIVITY}
)
SUPPORT_VERDICTS = frozenset({"supported", "unsupported", "insufficient"})
SUBJECTIVITY_VERDICTS = frozenset(
    {"violation", "no_violation", "insufficient"}
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _contract(
    contract_set: Mapping[str, Any], branch_id: str
) -> Mapping[str, Any]:
    matches = [
        contract
        for contract in contract_set.get("contracts") or []
        if contract.get("branch_id") == branch_id
    ]
    if len(matches) != 1:
        raise ThenAtomizationError(
            f"branch_id must resolve to one semantic judge contract: {branch_id!r}"
        )
    if matches[0].get("contract_status") != "ready":
        raise ThenAtomizationError("semantic judge contract is not ready")
    return matches[0]


def _assistant_events(execution: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        deepcopy(event)
        for event in normalize_tau_execution(execution)
        if event.get("event_kind") == "assistant_message"
        and isinstance(event.get("content"), str)
        and event["content"].strip()
    ]


def _packet(
    *,
    packet_id: str,
    task_name: str,
    contract: Mapping[str, Any],
    execution_fingerprint: str,
    target_event: Mapping[str, Any],
    model_input: Mapping[str, Any],
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": PACKET_VERSION,
        "packet_id": packet_id,
        "task_name": task_name,
        "branch_id": contract["branch_id"],
        "requirement_id": contract["requirement_id"],
        "semantic_judge_contract_id": contract["semantic_judge_contract_id"],
        "target_event": {
            "event_index": target_event["event_index"],
            "source_message_index": target_event["source_message_index"],
        },
        "model_input": deepcopy(dict(model_input)),
        "source_execution_fingerprint": execution_fingerprint,
        "source_semantic_judge_contract_fingerprint": contract[
            "semantic_judge_contract_fingerprint"
        ],
        "lineage": deepcopy(dict(lineage or {})),
    }
    result["packet_fingerprint"] = content_sha256(result)
    return result


def build_initial_semantic_judge_packets(
    semantic_judge_contract_set: Mapping[str, Any],
    execution: Mapping[str, Any],
    branch_id: str,
) -> dict[str, Any]:
    """Build the first model stage: claims OR subjectivity, never both."""

    contracts = validate_semantic_judge_contract_set(semantic_judge_contract_set)
    execution = deepcopy(dict(_mapping(execution, "$execution")))
    contract = _contract(contracts, branch_id)
    execution_fingerprint = content_sha256(execution)
    events = _assistant_events(execution)
    judge_kind = contract["program"]["judge_kind"]
    packets = []
    for ordinal, event in enumerate(events, start=1):
        if judge_kind == "claim_source_support":
            task = TASK_CLAIM_ATOMIZATION
            model_input = {"assistant_message": event["content"]}
        elif judge_kind == "subjective_expression":
            task = TASK_SUBJECTIVITY
            model_input = {
                "policy_requirement": contract["policy_requirement"],
                "assistant_message": event["content"],
            }
        else:
            raise ThenAtomizationError("unsupported ready semantic judge program")
        packets.append(
            _packet(
                packet_id=f"{contract['semantic_judge_contract_id']}::{task}::{ordinal:03d}",
                task_name=task,
                contract=contract,
                execution_fingerprint=execution_fingerprint,
                target_event=event,
                model_input=model_input,
            )
        )
    result = {
        "schema_version": PACKET_SET_VERSION,
        "stage": "initial",
        "branch_id": branch_id,
        "semantic_judge_contract_id": contract["semantic_judge_contract_id"],
        "source_semantic_judge_contract_set_fingerprint": contracts[
            "semantic_judge_contract_set_fingerprint"
        ],
        "source_execution_fingerprint": execution_fingerprint,
        "packets": packets,
        "summary": {
            "target_assistant_message_count": len(events),
            "packet_count": len(packets),
            "expected_model_calls": len(packets),
            "task_name": packets[0]["task_name"] if packets else (
                TASK_CLAIM_ATOMIZATION
                if judge_kind == "claim_source_support"
                else TASK_SUBJECTIVITY
            ),
        },
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    validate_semantic_judge_packet_set(result)
    return result


def validate_semantic_judge_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$semantic_judge_packet_set")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != PACKET_SET_VERSION or fingerprint != content_sha256(
        result
    ):
        raise ThenAtomizationError("invalid semantic judge packet set")
    packets = result.get("packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("semantic judge packets must be an array")
    identities = []
    task_names = set()
    for raw in packets:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if packet.get("schema_version") != PACKET_VERSION or packet_fingerprint != content_sha256(
            packet
        ):
            raise ThenAtomizationError("invalid semantic judge packet")
        task = packet.get("task_name")
        if task not in TASKS:
            raise ThenAtomizationError("invalid semantic judge packet task")
        model_input = _mapping(packet.get("model_input"), "$.packet.model_input")
        expected_fields = {
            TASK_CLAIM_ATOMIZATION: {"assistant_message"},
            TASK_SUBJECTIVITY: {"policy_requirement", "assistant_message"},
            TASK_CLAIM_SUPPORT: {"claim", "sources"},
        }[task]
        if set(model_input) != expected_fields:
            raise ThenAtomizationError("semantic judge model input is not minimal and closed")
        if task == TASK_CLAIM_SUPPORT:
            if not isinstance(model_input.get("claim"), str) or not model_input["claim"].strip():
                raise ThenAtomizationError("claim support input requires one claim")
            sources = model_input.get("sources")
            if not isinstance(sources, list) or not sources:
                raise ThenAtomizationError("claim support input requires source evidence")
            source_ids = []
            for source in sources:
                source = _mapping(source, "$.packet.model_input.sources[]")
                if set(source) != {"source_id", "source_kind", "content"}:
                    raise ThenAtomizationError("claim source fields are invalid")
                if not all(isinstance(source.get(key), str) and source[key] for key in source):
                    raise ThenAtomizationError("claim sources require non-empty strings")
                source_ids.append(source["source_id"])
            if len(source_ids) != len(set(source_ids)):
                raise ThenAtomizationError("claim source IDs must be unique")
        elif any(
            not isinstance(model_input.get(field), str) or not model_input[field].strip()
            for field in expected_fields
        ):
            raise ThenAtomizationError("semantic judge model input strings must be non-empty")
        identities.append(packet.get("packet_id"))
        task_names.add(task)
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("semantic judge packet IDs must be unique")
    if len(task_names) > 1:
        raise ThenAtomizationError("one semantic judge packet set must contain one task")
    summary = _mapping(result.get("summary"), "$.summary")
    if summary.get("packet_count") != len(packets):
        raise ThenAtomizationError("semantic judge packet count mismatch")
    if summary.get("expected_model_calls") != len(packets):
        raise ThenAtomizationError("semantic judge expected call count mismatch")
    if packets and summary.get("task_name") != packets[0].get("task_name"):
        raise ThenAtomizationError("semantic judge summary task mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def validate_claim_atomization_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    if packet.get("task_name") != TASK_CLAIM_ATOMIZATION:
        raise ThenAtomizationError("claim response attached to the wrong task")
    response = deepcopy(dict(_mapping(response, "$claim_atomization_response")))
    if set(response) != {"claims"} or not isinstance(response["claims"], list):
        raise ThenAtomizationError("claim response must contain only a claims array")
    message = packet["model_input"]["assistant_message"]
    texts = []
    for raw in response["claims"]:
        claim = _mapping(raw, "$.claims[]")
        if set(claim) != {"text"}:
            raise ThenAtomizationError("each claim must contain only text")
        text = claim.get("text")
        if not isinstance(text, str) or not text.strip() or text not in message:
            raise ThenAtomizationError("claim text must be an exact non-empty message span")
        texts.append(text)
    if len(texts) != len(set(texts)):
        raise ThenAtomizationError("claim spans must be unique")
    return response


def attach_claim_atomization_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    response = validate_claim_atomization_response(packet, response)
    result = {
        "schema_version": CLAIM_RESULT_VERSION,
        "source_packet_id": packet["packet_id"],
        "source_packet_fingerprint": packet["packet_fingerprint"],
        "semantic_judge_contract_id": packet["semantic_judge_contract_id"],
        "branch_id": packet["branch_id"],
        "target_event": deepcopy(packet["target_event"]),
        "claims": [
            {"claim_id": f"{packet['packet_id']}::C{index:02d}", "text": claim["text"]}
            for index, claim in enumerate(response["claims"], start=1)
        ],
    }
    result["claim_result_fingerprint"] = content_sha256(result)
    return result


def _agent_spec_sources(
    agent_spec: Mapping[str, Any], channels: set[str]
) -> list[dict[str, str]]:
    agent_spec = _mapping(agent_spec, "$agent_spec")
    sources = []
    if "system_policy" in channels:
        prompt = agent_spec.get("system_prompt")
        if isinstance(prompt, Mapping):
            prompt = prompt.get("text")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ThenAtomizationError("agent spec must provide non-empty system_prompt text")
        sources.append(
            {
                "source_id": "system_policy",
                "source_kind": "system_policy",
                "content": prompt,
            }
        )
    if "agent_tool_schema" in channels:
        tools = agent_spec.get("tools")
        if not isinstance(tools, list):
            raise ThenAtomizationError("agent spec tools must be an array")
        names = []
        for raw in tools:
            tool = _mapping(raw, "$.agent_spec.tools[]")
            name = tool.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise ThenAtomizationError("agent spec tool names must be unique strings")
            names.append(name)
            view = {
                key: deepcopy(tool[key])
                for key in (
                    "name",
                    "description",
                    "parameters",
                    "required_parameters",
                )
                if key in tool
            }
            sources.append(
                {
                    "source_id": f"tool_schema:{name}",
                    "source_kind": "agent_tool_schema",
                    "content": json.dumps(
                        view, ensure_ascii=False, sort_keys=True, indent=2
                    ),
                }
            )
    return sources


def _prior_runtime_sources(
    events: Sequence[Mapping[str, Any]], target_event_index: int, channels: set[str]
) -> list[dict[str, str]]:
    sources = []
    for event in events:
        if event.get("event_index", -1) >= target_event_index:
            continue
        kind = event.get("event_kind")
        if kind == "user_message" and "prior_user_message" in channels:
            source_kind = "prior_user_message"
        elif kind == "tool_result" and "prior_tool_result" in channels:
            source_kind = "prior_tool_result"
        else:
            continue
        content = _text(event.get("content"))
        if not content.strip():
            continue
        sources.append(
            {
                "source_id": f"{source_kind}@event:{event['event_index']}",
                "source_kind": source_kind,
                "content": content,
            }
        )
    return sources


def build_claim_support_packets(
    semantic_judge_contract_set: Mapping[str, Any],
    execution: Mapping[str, Any],
    branch_id: str,
    claim_results: Sequence[Mapping[str, Any]],
    agent_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one support decision per extracted claim using closed prior sources."""

    contracts = validate_semantic_judge_contract_set(semantic_judge_contract_set)
    execution = deepcopy(dict(_mapping(execution, "$execution")))
    contract = _contract(contracts, branch_id)
    if contract["program"]["judge_kind"] != "claim_source_support":
        raise ThenAtomizationError("claim support packets require a claim support contract")
    execution_fingerprint = content_sha256(execution)
    events = normalize_tau_execution(execution)
    event_index = {event["event_index"]: event for event in events}
    channels = set(contract["program"]["source_policy"]["channels"])
    packets = []
    seen_claim_ids = set()
    for raw_result in claim_results:
        result = deepcopy(dict(_mapping(raw_result, "$claim_results[]")))
        fingerprint = result.pop("claim_result_fingerprint", None)
        if result.get("schema_version") != CLAIM_RESULT_VERSION or fingerprint != content_sha256(
            result
        ):
            raise ThenAtomizationError("invalid semantic claim result")
        if result.get("branch_id") != branch_id or result.get(
            "semantic_judge_contract_id"
        ) != contract["semantic_judge_contract_id"]:
            raise ThenAtomizationError("claim result belongs to a different contract")
        target = _mapping(result.get("target_event"), "$.claim_result.target_event")
        target_index = target.get("event_index")
        target_event = event_index.get(target_index)
        if (
            target_event is None
            or target_event.get("event_kind") != "assistant_message"
            or target_event.get("source_message_index") != target.get("source_message_index")
        ):
            raise ThenAtomizationError("claim target event does not exist in execution")
        sources = _prior_runtime_sources(events, target_index, channels)
        if {"system_policy", "agent_tool_schema"} & channels:
            sources.extend(_agent_spec_sources(agent_spec, channels))
        if not sources:
            raise ThenAtomizationError("claim support source set is empty")
        for claim in result.get("claims") or []:
            claim = _mapping(claim, "$.claim_result.claims[]")
            claim_id = claim.get("claim_id")
            claim_text = claim.get("text")
            if (
                not isinstance(claim_id, str)
                or not claim_id
                or claim_id in seen_claim_ids
                or not isinstance(claim_text, str)
                or not claim_text.strip()
                or claim_text not in target_event["content"]
            ):
                raise ThenAtomizationError("semantic claim identity or text is invalid")
            seen_claim_ids.add(claim_id)
            packets.append(
                _packet(
                    packet_id=f"{claim_id}::{TASK_CLAIM_SUPPORT}",
                    task_name=TASK_CLAIM_SUPPORT,
                    contract=contract,
                    execution_fingerprint=execution_fingerprint,
                    target_event=target_event,
                    model_input={"claim": claim_text, "sources": sources},
                    lineage={
                        "claim_id": claim_id,
                        "source_claim_result_fingerprint": fingerprint,
                    },
                )
            )
    result = {
        "schema_version": PACKET_SET_VERSION,
        "stage": "claim_support",
        "branch_id": branch_id,
        "semantic_judge_contract_id": contract["semantic_judge_contract_id"],
        "source_semantic_judge_contract_set_fingerprint": contracts[
            "semantic_judge_contract_set_fingerprint"
        ],
        "source_execution_fingerprint": execution_fingerprint,
        "packets": packets,
        "summary": {
            "claim_result_count": len(claim_results),
            "packet_count": len(packets),
            "expected_model_calls": len(packets),
            "task_name": TASK_CLAIM_SUPPORT,
        },
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    validate_semantic_judge_packet_set(result)
    return result


def validate_claim_support_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    if packet.get("task_name") != TASK_CLAIM_SUPPORT:
        raise ThenAtomizationError("claim support response attached to the wrong task")
    response = deepcopy(dict(_mapping(response, "$claim_support_response")))
    if set(response) != {"verdict", "evidence", "reason"}:
        raise ThenAtomizationError("claim support response fields are invalid")
    verdict = response.get("verdict")
    if verdict not in SUPPORT_VERDICTS:
        raise ThenAtomizationError("claim support verdict is invalid")
    if not isinstance(response.get("reason"), str) or not response["reason"].strip():
        raise ThenAtomizationError("claim support reason must be non-empty")
    evidence = response.get("evidence")
    if verdict == "supported":
        evidence = _mapping(evidence, "$.evidence")
        if set(evidence) != {"source_id", "text"}:
            raise ThenAtomizationError("claim support evidence fields are invalid")
        sources = {
            source["source_id"]: source["content"]
            for source in packet["model_input"]["sources"]
        }
        if evidence.get("source_id") not in sources:
            raise ThenAtomizationError("claim support evidence source is unknown")
        span = evidence.get("text")
        if not isinstance(span, str) or not span.strip() or span not in sources[evidence["source_id"]]:
            raise ThenAtomizationError("claim support evidence must be an exact source span")
    elif evidence is not None:
        raise ThenAtomizationError("unsupported/insufficient claim must have null evidence")
    return response


def validate_subjectivity_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    if packet.get("task_name") != TASK_SUBJECTIVITY:
        raise ThenAtomizationError("subjectivity response attached to the wrong task")
    response = deepcopy(dict(_mapping(response, "$subjectivity_response")))
    if set(response) != {"verdict", "evidence_span", "reason"}:
        raise ThenAtomizationError("subjectivity response fields are invalid")
    verdict = response.get("verdict")
    if verdict not in SUBJECTIVITY_VERDICTS:
        raise ThenAtomizationError("subjectivity verdict is invalid")
    if not isinstance(response.get("reason"), str) or not response["reason"].strip():
        raise ThenAtomizationError("subjectivity reason must be non-empty")
    span = response.get("evidence_span")
    if verdict == "violation":
        message = packet["model_input"]["assistant_message"]
        if not isinstance(span, str) or not span.strip() or span not in message:
            raise ThenAtomizationError("subjectivity evidence must be an exact message span")
    elif span is not None:
        raise ThenAtomizationError("non-violation/insufficient must have null evidence span")
    return response


def render_semantic_judge_prompt(packet: Mapping[str, Any], template: str) -> str:
    placeholder = "{semantic_judge_input}"
    task_name = packet.get("task_name")
    task_marker = f"Task: {task_name}"
    if (
        task_name not in TASKS
        or placeholder not in template
        or task_marker not in template
    ):
        raise ThenAtomizationError("semantic judge prompt template/task mismatch")
    return template.replace(
        placeholder,
        json.dumps(packet["model_input"], ensure_ascii=False, indent=2),
    )
