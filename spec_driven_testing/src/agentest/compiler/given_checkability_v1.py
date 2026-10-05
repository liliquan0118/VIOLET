"""Gate natural-language Given conditions on operational checkability."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import (
    CAP_AGENT_TOOL_SCHEMAS,
    CAP_TOOL_OBSERVABLE_ENDPOINTS,
    content_sha256,
    validate_accepted_artifact_catalog,
)
from .given_evidence_source_v2 import validate_given_evidence_source_packet_set_v2
from .then_atomization import ThenAtomizationError


TASK_NAME = "given_condition_checkability"
PACKET_SET_VERSION = "agentspectesting.given-checkability-packet-set/v0.1"
PACKET_VERSION = "agentspectesting.given-checkability-packet/v0.1"
DECISIONS = frozenset({"checkable", "source_undefined", "insufficient_context"})
RESULT_SET_VERSION = "agentspectesting.given-checkability-result-set/v0.1"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _document_for_capability(catalog: Mapping[str, Any], capability: str) -> Mapping[str, Any]:
    providers = catalog["capability_index"].get(capability) or []
    if len(providers) != 1:
        raise ThenAtomizationError(f"accepted capability {capability!r} must have one provider")
    return _mapping(catalog["artifacts"][providers[0]]["document"], capability)


def _system_prompt_text(agent_spec: Mapping[str, Any]) -> str:
    prompt = agent_spec.get("system_prompt")
    if isinstance(prompt, Mapping):
        prompt = prompt.get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ThenAtomizationError("accepted system prompt text is missing")
    return prompt.strip()


def _return_field_vocabulary(tool_catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    unique = {}
    for tool in tool_catalog.get("tools") or []:
        for endpoint in tool.get("observable_endpoints") or []:
            if endpoint.get("source_kind") != "tool_return":
                continue
            key = (
                endpoint.get("source_path"),
                tuple(endpoint.get("allowed_values") or []),
            )
            unique[key] = {
                "path": endpoint.get("source_path"),
                "allowed_values": deepcopy(endpoint.get("allowed_values") or []),
            }
    return [unique[key] for key in sorted(unique, key=lambda item: (str(item[0]), item[1]))]


def build_given_checkability_packets(
    source_packet_set: Mapping[str, Any], accepted_artifact_catalog: Mapping[str, Any]
) -> dict[str, Any]:
    sources = validate_given_evidence_source_packet_set_v2(source_packet_set)
    catalog = validate_accepted_artifact_catalog(accepted_artifact_catalog)
    if sources["source_artifact_catalog_fingerprint"] != catalog["catalog_fingerprint"]:
        raise ThenAtomizationError("Given checkability artifact lineage mismatch")
    agent_spec = _document_for_capability(catalog, CAP_AGENT_TOOL_SCHEMAS)
    tool_catalog = _document_for_capability(catalog, CAP_TOOL_OBSERVABLE_ENDPOINTS)
    full_policy = _system_prompt_text(agent_spec)
    field_vocabulary = _return_field_vocabulary(tool_catalog)
    packets = []
    expanded_count = 0
    for source_packet in sources["packets"]:
        source_input = source_packet["model_input"]
        accepted = source_input["accepted_evidence"]
        no_match = not accepted["matching_system_policy_lines"] and not accepted[
            "matching_state_observables"
        ]
        definition_evidence = {
            "matching_system_policy_lines": deepcopy(
                accepted["matching_system_policy_lines"]
            ),
            "matching_state_observables": deepcopy(
                accepted["matching_state_observables"]
            ),
            "available_agent_action_names": deepcopy(
                accepted["available_agent_action_names"]
            ),
            "search_scope": deepcopy(accepted["search_scope"]),
            "complete_context_expansion": None,
        }
        needs_complete_context = (
            no_match or source_input["selected_spec_rule"].get("origin") != "prompt"
        )
        if needs_complete_context:
            definition_evidence["complete_context_expansion"] = {
                "reason": "no lexical policy or observable match; complete accepted context supplied to decide absence safely",
                "complete_system_policy": full_policy,
                "complete_tool_return_field_vocabulary": deepcopy(field_vocabulary),
            }
            expanded_count += 1
        model_input = {
            "condition": source_input["condition"],
            "when": source_input["when"],
            "selected_spec_rule": deepcopy(source_input["selected_spec_rule"]),
            "accepted_definition_evidence": definition_evidence,
        }
        packet = {
            "schema_version": PACKET_VERSION,
            "packet_id": f"GIVEN-CHECK::{content_sha256(model_input)[:16]}::C01",
            "task_name": TASK_NAME,
            "model_input": model_input,
            "members": deepcopy(source_packet["members"]),
            "source_evidence_packet_fingerprint": source_packet[
                "packet_fingerprint"
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
            "condition_count": sum(len(item["members"]) for item in packets),
            "complete_context_expansion_count": expanded_count,
            "expected_model_calls": len(packets),
        },
        "source_evidence_packet_set_fingerprint": sources["packet_set_fingerprint"],
        "source_artifact_catalog_fingerprint": catalog["catalog_fingerprint"],
        "source_driver_profile_fingerprint": sources[
            "source_driver_profile_fingerprint"
        ],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_checkability_packet_set(result)


def validate_given_checkability_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_checkability_packet_set")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if (
        result.get("schema_version") != PACKET_SET_VERSION
        or result.get("task_name") != TASK_NAME
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given checkability packet set")
    ids = []
    members = []
    expanded = 0
    for raw in result.get("packets") or []:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if (
            packet.get("schema_version") != PACKET_VERSION
            or packet.get("task_name") != TASK_NAME
            or packet_fingerprint != content_sha256(packet)
        ):
            raise ThenAtomizationError("invalid Given checkability packet")
        model_input = _mapping(packet.get("model_input"), "$.model_input")
        if set(model_input) != {
            "condition", "when", "selected_spec_rule", "accepted_definition_evidence"
        }:
            raise ThenAtomizationError("Given checkability input fields are invalid")
        evidence = _mapping(
            model_input["accepted_definition_evidence"], "$.accepted_definition_evidence"
        )
        if evidence.get("complete_context_expansion") is not None:
            expanded += 1
        ids.append(packet.get("packet_id"))
        members.extend(item.get("condition_id") for item in packet.get("members") or [])
    if len(ids) != len(set(ids)) or len(members) != len(set(members)):
        raise ThenAtomizationError("Given checkability identities are not unique")
    expected = {
        "packet_count": len(ids),
        "condition_count": len(members),
        "complete_context_expansion_count": expanded,
        "expected_model_calls": len(ids),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given checkability summary mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def validate_given_checkability_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, str]:
    _mapping(packet, "$packet")
    response = deepcopy(dict(_mapping(response, "$response")))
    if set(response) != {"checkability", "reason"}:
        raise ThenAtomizationError("Given checkability response fields are invalid")
    if response.get("checkability") not in DECISIONS:
        raise ThenAtomizationError("Given checkability decision is invalid")
    if response.get("checkability") == "source_undefined":
        expansion = packet["model_input"]["accepted_definition_evidence"].get(
            "complete_context_expansion"
        )
        if expansion is None:
            raise ThenAtomizationError(
                "source_undefined requires complete accepted context"
            )
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 400:
        raise ThenAtomizationError("Given checkability reason must be concise text")
    return {"checkability": response["checkability"], "reason": reason.strip()}


def select_given_checkability_packets(
    packet_set: Mapping[str, Any], packet_ids: list[str]
) -> dict[str, Any]:
    packets = validate_given_checkability_packet_set(packet_set)
    if (
        not isinstance(packet_ids, list)
        or not packet_ids
        or len(packet_ids) != len(set(packet_ids))
        or any(not isinstance(item, str) or not item for item in packet_ids)
    ):
        raise ThenAtomizationError("Given checkability selection IDs are invalid")
    index = {item["packet_id"]: item for item in packets["packets"]}
    missing = sorted(set(packet_ids) - set(index))
    if missing:
        raise ThenAtomizationError(f"unknown Given checkability packet IDs: {missing}")
    selected = [deepcopy(index[item]) for item in packet_ids]
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": selected,
        "summary": {
            "packet_count": len(selected),
            "condition_count": sum(len(item["members"]) for item in selected),
            "complete_context_expansion_count": sum(
                item["model_input"]["accepted_definition_evidence"][
                    "complete_context_expansion"
                ]
                is not None
                for item in selected
            ),
            "expected_model_calls": len(selected),
        },
        "source_evidence_packet_set_fingerprint": packets[
            "source_evidence_packet_set_fingerprint"
        ],
        "source_artifact_catalog_fingerprint": packets[
            "source_artifact_catalog_fingerprint"
        ],
        "source_driver_profile_fingerprint": packets[
            "source_driver_profile_fingerprint"
        ],
        "source_full_checkability_packet_set_fingerprint": packets[
            "packet_set_fingerprint"
        ],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_checkability_packet_set(result)


def materialize_given_checkability_results(
    initial_packet_set: Mapping[str, Any],
    initial_responses: Mapping[str, Any],
    adjudication_packet_set: Mapping[str, Any],
    adjudication_responses: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge valid initial answers with condition-identical adjudicated replacements."""

    initial = validate_given_checkability_packet_set(initial_packet_set)
    adjudication = validate_given_checkability_packet_set(adjudication_packet_set)
    if adjudication["source_evidence_packet_set_fingerprint"] != initial[
        "source_evidence_packet_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given checkability source lineage mismatch")
    initial_map = deepcopy(dict(_mapping(initial_responses, "$initial_responses")))
    adjudication_map = deepcopy(
        dict(_mapping(adjudication_responses, "$adjudication_responses"))
    )
    if set(initial_map) != {item["packet_id"] for item in initial["packets"]}:
        raise ThenAtomizationError("initial Given checkability batch is incomplete")
    if set(adjudication_map) != {
        item["packet_id"] for item in adjudication["packets"]
    }:
        raise ThenAtomizationError("adjudication Given checkability batch is incomplete")
    adjudication_by_conditions = {
        tuple(item["condition_id"] for item in packet["members"]): packet
        for packet in adjudication["packets"]
    }
    records = []
    initial_count = 0
    adjudicated_count = 0
    for packet in initial["packets"]:
        condition_key = tuple(
            member["condition_id"] for member in packet["members"]
        )
        try:
            checked = validate_given_checkability_response(
                packet, _mapping(initial_map[packet["packet_id"]], "$.initial_response")
            )
            evidence_packet = packet
            basis = "accepted_initial_response"
            initial_count += len(packet["members"])
        except ThenAtomizationError:
            evidence_packet = adjudication_by_conditions.get(condition_key)
            if evidence_packet is None:
                raise ThenAtomizationError(
                    f"inadmissible checkability response has no adjudication: {condition_key}"
                )
            checked = validate_given_checkability_response(
                evidence_packet,
                _mapping(
                    adjudication_map[evidence_packet["packet_id"]],
                    "$.adjudication_response",
                ),
            )
            basis = "accepted_adjudication_response"
            adjudicated_count += len(packet["members"])
        for member in packet["members"]:
            records.append(
                {
                    "branch_id": member["branch_id"],
                    "condition_id": member["condition_id"],
                    "checkability": checked["checkability"],
                    "reason": checked["reason"],
                    "resolution_basis": basis,
                    "source_checkability_packet_fingerprint": packet[
                        "packet_fingerprint"
                    ],
                    "evidence_packet_fingerprint": evidence_packet[
                        "packet_fingerprint"
                    ],
                }
            )
    counts = Counter(item["checkability"] for item in records)
    result = {
        "schema_version": RESULT_SET_VERSION,
        "results": records,
        "summary": {
            "condition_count": len(records),
            "checkability_counts": dict(sorted(counts.items())),
            "initial_response_count": initial_count,
            "adjudicated_response_count": adjudicated_count,
        },
        "source_initial_packet_set_fingerprint": initial["packet_set_fingerprint"],
        "source_adjudication_packet_set_fingerprint": adjudication[
            "packet_set_fingerprint"
        ],
        "source_evidence_packet_set_fingerprint": initial[
            "source_evidence_packet_set_fingerprint"
        ],
    }
    result["result_set_fingerprint"] = content_sha256(result)
    return validate_given_checkability_result_set(result)


def materialize_given_checkability_batch_results(
    packet_set: Mapping[str, Any], responses: Mapping[str, Any]
) -> dict[str, Any]:
    """Materialize one closed batch when every response is directly admissible."""

    packets = validate_given_checkability_packet_set(packet_set)
    response_map = deepcopy(dict(_mapping(responses, "$responses")))
    if set(response_map) != {item["packet_id"] for item in packets["packets"]}:
        raise ThenAtomizationError(
            "Given checkability responses must cover the closed batch"
        )
    records = []
    for packet in packets["packets"]:
        checked = validate_given_checkability_response(
            packet, _mapping(response_map[packet["packet_id"]], "$.response")
        )
        for member in packet["members"]:
            records.append(
                {
                    "branch_id": member["branch_id"],
                    "condition_id": member["condition_id"],
                    "checkability": checked["checkability"],
                    "reason": checked["reason"],
                    "resolution_basis": "accepted_initial_response",
                    "source_checkability_packet_fingerprint": packet[
                        "packet_fingerprint"
                    ],
                    "evidence_packet_fingerprint": packet["packet_fingerprint"],
                }
            )
    counts = Counter(item["checkability"] for item in records)
    result = {
        "schema_version": RESULT_SET_VERSION,
        "results": records,
        "summary": {
            "condition_count": len(records),
            "checkability_counts": dict(sorted(counts.items())),
            "initial_response_count": len(records),
            "adjudicated_response_count": 0,
        },
        "source_initial_packet_set_fingerprint": packets[
            "packet_set_fingerprint"
        ],
        "source_adjudication_packet_set_fingerprint": None,
        "source_evidence_packet_set_fingerprint": packets[
            "source_evidence_packet_set_fingerprint"
        ],
    }
    result["result_set_fingerprint"] = content_sha256(result)
    return validate_given_checkability_result_set(result)


def merge_given_checkability_result_sets(
    full_packet_set: Mapping[str, Any], component_result_sets: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Merge disjoint closed batches and prove complete coverage of the full batch."""

    full = validate_given_checkability_packet_set(full_packet_set)
    if not isinstance(component_result_sets, list) or not component_result_sets:
        raise ThenAtomizationError("Given checkability result components are required")
    components = [
        validate_given_checkability_result_set(item)
        for item in component_result_sets
    ]
    expected = {
        member["condition_id"]: {
            "branch_id": member["branch_id"],
            "packet_fingerprint": packet["packet_fingerprint"],
        }
        for packet in full["packets"]
        for member in packet["members"]
    }
    records_by_id = {}
    for component in components:
        for record in component["results"]:
            condition_id = record["condition_id"]
            if condition_id not in expected:
                raise ThenAtomizationError(
                    f"component condition is outside full checkability batch: {condition_id}"
                )
            if condition_id in records_by_id:
                raise ThenAtomizationError(
                    f"overlapping Given checkability result components: {condition_id}"
                )
            if record.get("branch_id") != expected[condition_id]["branch_id"]:
                raise ThenAtomizationError(
                    f"Given checkability branch mismatch: {condition_id}"
                )
            records_by_id[condition_id] = deepcopy(record)
    missing = sorted(set(expected) - set(records_by_id))
    if missing:
        raise ThenAtomizationError(
            f"merged Given checkability results are incomplete: {missing}"
        )
    records = [records_by_id[condition_id] for condition_id in expected]
    counts = Counter(item["checkability"] for item in records)
    result = {
        "schema_version": RESULT_SET_VERSION,
        "results": records,
        "summary": {
            "condition_count": len(records),
            "checkability_counts": dict(sorted(counts.items())),
            "initial_response_count": sum(
                item.get("resolution_basis") == "accepted_initial_response"
                for item in records
            ),
            "adjudicated_response_count": sum(
                item.get("resolution_basis") == "accepted_adjudication_response"
                for item in records
            ),
        },
        "source_initial_packet_set_fingerprint": None,
        "source_adjudication_packet_set_fingerprint": None,
        "source_evidence_packet_set_fingerprint": full[
            "source_evidence_packet_set_fingerprint"
        ],
        "source_full_packet_set_fingerprint": full["packet_set_fingerprint"],
        "source_component_result_set_fingerprints": [
            item["result_set_fingerprint"] for item in components
        ],
    }
    result["result_set_fingerprint"] = content_sha256(result)
    return validate_given_checkability_result_set(result)


def validate_given_checkability_result_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_checkability_result_set")))
    fingerprint = result.pop("result_set_fingerprint", None)
    if result.get("schema_version") != RESULT_SET_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid Given checkability result set")
    records = result.get("results")
    if not isinstance(records, list):
        raise ThenAtomizationError("Given checkability results must be an array")
    ids = [item.get("condition_id") for item in records]
    if len(ids) != len(set(ids)):
        raise ThenAtomizationError("Given checkability condition IDs are not unique")
    if any(item.get("checkability") not in DECISIONS for item in records):
        raise ThenAtomizationError("Given checkability result decision is invalid")
    counts = Counter(item["checkability"] for item in records)
    expected = {
        "condition_count": len(records),
        "checkability_counts": dict(sorted(counts.items())),
        "initial_response_count": sum(
            item.get("resolution_basis") == "accepted_initial_response"
            for item in records
        ),
        "adjudicated_response_count": sum(
            item.get("resolution_basis") == "accepted_adjudication_response"
            for item in records
        ),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given checkability result summary mismatch")
    result["result_set_fingerprint"] = fingerprint
    return result


def select_pending_given_checkability_packets(
    full_checkability_packet_set: Mapping[str, Any],
    completed_result_set: Mapping[str, Any],
    completed_source_packet_set: Mapping[str, Any],
    full_source_packet_set: Mapping[str, Any],
) -> dict[str, Any]:
    """Subtract completed calibration conditions with explicit subset-to-full lineage."""

    full_checks = validate_given_checkability_packet_set(full_checkability_packet_set)
    completed = validate_given_checkability_result_set(completed_result_set)
    completed_sources = validate_given_evidence_source_packet_set_v2(
        completed_source_packet_set
    )
    full_sources = validate_given_evidence_source_packet_set_v2(full_source_packet_set)
    if completed_sources.get("source_full_packet_set_fingerprint") != full_sources[
        "packet_set_fingerprint"
    ]:
        raise ThenAtomizationError("completed Given source subset is not linked to full set")
    if completed["source_evidence_packet_set_fingerprint"] != completed_sources[
        "packet_set_fingerprint"
    ]:
        raise ThenAtomizationError("completed Given checkability source lineage mismatch")
    if full_checks["source_evidence_packet_set_fingerprint"] != full_sources[
        "packet_set_fingerprint"
    ]:
        raise ThenAtomizationError("full Given checkability source lineage mismatch")
    completed_ids = {item["condition_id"] for item in completed["results"]}
    full_ids = {
        member["condition_id"]
        for packet in full_checks["packets"]
        for member in packet["members"]
    }
    if not completed_ids.issubset(full_ids):
        raise ThenAtomizationError("completed Given conditions are not a subset of full set")
    pending_packet_ids = []
    for packet in full_checks["packets"]:
        member_ids = {item["condition_id"] for item in packet["members"]}
        overlap = member_ids & completed_ids
        if overlap and overlap != member_ids:
            raise ThenAtomizationError("deduplicated checkability packet is partially completed")
        if not overlap:
            pending_packet_ids.append(packet["packet_id"])
    if not pending_packet_ids:
        raise ThenAtomizationError("no pending Given checkability packets")
    result = select_given_checkability_packets(full_checks, pending_packet_ids)
    result.pop("packet_set_fingerprint")
    result["completed_condition_count"] = len(completed_ids)
    result["source_completed_result_set_fingerprint"] = completed[
        "result_set_fingerprint"
    ]
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_checkability_packet_set(result)


def render_given_checkability_prompt(packet: Mapping[str, Any], template: str) -> str:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    if packet.get("task_name") != TASK_NAME or f"TASK: {TASK_NAME}" not in template:
        raise ThenAtomizationError("Given checkability prompt/task mismatch")
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        packet["model_input"], ensure_ascii=False, indent=2
    )
