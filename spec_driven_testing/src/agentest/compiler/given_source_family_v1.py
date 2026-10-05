"""Prepare one source-family classification for checkable Given conditions."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .driver_evidence_capabilities_v1 import SOURCE_KINDS
from .given_checkability_v1 import validate_given_checkability_result_set
from .given_evidence_source_v2 import validate_given_evidence_source_packet_set_v2
from .then_atomization import ThenAtomizationError


TASK_NAME = "given_source_family"
PACKET_SET_VERSION = "agentspectesting.given-source-family-packet-set/v0.1"
PACKET_VERSION = "agentspectesting.given-source-family-packet/v0.1"
RESULT_SET_VERSION = "agentspectesting.given-source-family-result-set/v0.1"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def build_given_source_family_packets(
    source_packet_set: Mapping[str, Any], checkability_result_set: Mapping[str, Any]
) -> dict[str, Any]:
    sources = validate_given_evidence_source_packet_set_v2(source_packet_set)
    checkability = validate_given_checkability_result_set(checkability_result_set)
    if checkability["source_evidence_packet_set_fingerprint"] != sources[
        "packet_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given source-family checkability lineage mismatch")
    check_index = {item["condition_id"]: item for item in checkability["results"]}
    packets = []
    skipped = []
    for source_packet in sources["packets"]:
        member_decisions = [check_index.get(item["condition_id"]) for item in source_packet["members"]]
        if any(item is None for item in member_decisions):
            continue
        decisions = {item["checkability"] for item in member_decisions}
        if len(decisions) != 1:
            raise ThenAtomizationError("deduplicated source packet has conflicting checkability")
        decision = next(iter(decisions))
        if decision != "checkable":
            skipped.extend(
                {
                    "branch_id": member["branch_id"],
                    "condition_id": member["condition_id"],
                    "checkability": decision,
                    "reason": checked["reason"],
                }
                for member, checked in zip(
                    source_packet["members"], member_decisions, strict=True
                )
            )
            continue
        source_input = source_packet["model_input"]
        model_input = {
            "condition": source_input["condition"],
            "when": source_input["when"],
            "selected_spec_rule": deepcopy(source_input["selected_spec_rule"]),
            "accepted_evidence": deepcopy(source_input["accepted_evidence"]),
            "allowed_source_kinds": sorted(SOURCE_KINDS),
        }
        packet = {
            "schema_version": PACKET_VERSION,
            "packet_id": f"GIVEN-FAMILY::{content_sha256(model_input)[:16]}::F01",
            "task_name": TASK_NAME,
            "model_input": model_input,
            "members": deepcopy(source_packet["members"]),
            "source_evidence_packet_fingerprint": source_packet[
                "packet_fingerprint"
            ],
            "source_checkability_evidence": [
                {
                    "condition_id": item["condition_id"],
                    "evidence_packet_fingerprint": item[
                        "evidence_packet_fingerprint"
                    ],
                }
                for item in member_decisions
            ],
        }
        packet["packet_fingerprint"] = content_sha256(packet)
        packets.append(packet)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": packets,
        "non_checkable_conditions": skipped,
        "summary": {
            "packet_count": len(packets),
            "checkable_condition_count": sum(len(item["members"]) for item in packets),
            "non_checkable_condition_count": len(skipped),
            "expected_model_calls": len(packets),
        },
        "source_evidence_packet_set_fingerprint": sources["packet_set_fingerprint"],
        "source_checkability_result_set_fingerprint": checkability[
            "result_set_fingerprint"
        ],
        "source_driver_profile_fingerprint": sources[
            "source_driver_profile_fingerprint"
        ],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_source_family_packet_set(result)


def validate_given_source_family_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_source_family_packet_set")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if (
        result.get("schema_version") != PACKET_SET_VERSION
        or result.get("task_name") != TASK_NAME
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given source-family packet set")
    ids = []
    condition_ids = []
    for raw in result.get("packets") or []:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if (
            packet.get("schema_version") != PACKET_VERSION
            or packet.get("task_name") != TASK_NAME
            or packet_fingerprint != content_sha256(packet)
        ):
            raise ThenAtomizationError("invalid Given source-family packet")
        model_input = _mapping(packet.get("model_input"), "$.model_input")
        if model_input.get("allowed_source_kinds") != sorted(SOURCE_KINDS):
            raise ThenAtomizationError("Given source-family options are invalid")
        ids.append(packet.get("packet_id"))
        condition_ids.extend(item.get("condition_id") for item in packet.get("members") or [])
    skipped = result.get("non_checkable_conditions") or []
    skipped_ids = [item.get("condition_id") for item in skipped]
    if len(ids) != len(set(ids)) or set(condition_ids) & set(skipped_ids):
        raise ThenAtomizationError("Given source-family identities overlap")
    expected = {
        "packet_count": len(ids),
        "checkable_condition_count": len(condition_ids),
        "non_checkable_condition_count": len(skipped_ids),
        "expected_model_calls": len(ids),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given source-family summary mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def validate_given_source_family_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, str]:
    _mapping(packet, "$packet")
    response = deepcopy(dict(_mapping(response, "$response")))
    if set(response) != {"source_kind", "reason"}:
        raise ThenAtomizationError("Given source-family response fields are invalid")
    if response.get("source_kind") not in SOURCE_KINDS:
        raise ThenAtomizationError("Given source-family response label is invalid")
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 400:
        raise ThenAtomizationError("Given source-family reason must be concise text")
    return {"source_kind": response["source_kind"], "reason": reason.strip()}


def select_given_source_family_packets(
    packet_set: Mapping[str, Any], packet_ids: list[str]
) -> dict[str, Any]:
    """Select a closed processing subset without repeating non-checkable records."""

    packets = validate_given_source_family_packet_set(packet_set)
    if (
        not isinstance(packet_ids, list)
        or not packet_ids
        or len(packet_ids) != len(set(packet_ids))
        or any(not isinstance(item, str) or not item for item in packet_ids)
    ):
        raise ThenAtomizationError("Given source-family selection IDs are invalid")
    index = {item["packet_id"]: item for item in packets["packets"]}
    missing = sorted(set(packet_ids) - set(index))
    if missing:
        raise ThenAtomizationError(
            f"unknown Given source-family packet IDs: {missing}"
        )
    selected = [deepcopy(index[item]) for item in packet_ids]
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": selected,
        "non_checkable_conditions": [],
        "summary": {
            "packet_count": len(selected),
            "checkable_condition_count": sum(
                len(item["members"]) for item in selected
            ),
            "non_checkable_condition_count": 0,
            "expected_model_calls": len(selected),
        },
        "source_evidence_packet_set_fingerprint": packets[
            "source_evidence_packet_set_fingerprint"
        ],
        "source_checkability_result_set_fingerprint": packets[
            "source_checkability_result_set_fingerprint"
        ],
        "source_driver_profile_fingerprint": packets[
            "source_driver_profile_fingerprint"
        ],
        "source_full_packet_set_fingerprint": packets["packet_set_fingerprint"],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_source_family_packet_set(result)


def render_given_source_family_prompt(packet: Mapping[str, Any], template: str) -> str:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    if packet.get("task_name") != TASK_NAME or f"TASK: {TASK_NAME}" not in template:
        raise ThenAtomizationError("Given source-family prompt/task mismatch")
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        packet["model_input"], ensure_ascii=False, indent=2
    )


def materialize_given_source_family_results(
    packet_set: Mapping[str, Any], responses: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_given_source_family_packet_set(packet_set)
    response_map = deepcopy(dict(_mapping(responses, "$responses")))
    if set(response_map) != {item["packet_id"] for item in packets["packets"]}:
        raise ThenAtomizationError("Given source-family responses must cover the closed batch")
    records = []
    for packet in packets["packets"]:
        checked = validate_given_source_family_response(
            packet, _mapping(response_map[packet["packet_id"]], "$.response")
        )
        for member in packet["members"]:
            records.append(
                {
                    "branch_id": member["branch_id"],
                    "condition_id": member["condition_id"],
                    "source_kind": checked["source_kind"],
                    "reason": checked["reason"],
                    "source_packet_fingerprint": packet["packet_fingerprint"],
                    "source_logical_form_fingerprint": member[
                        "source_logical_form_fingerprint"
                    ],
                }
            )
    counts = {}
    for item in records:
        counts[item["source_kind"]] = counts.get(item["source_kind"], 0) + 1
    result = {
        "schema_version": RESULT_SET_VERSION,
        "results": records,
        "non_checkable_conditions": deepcopy(packets["non_checkable_conditions"]),
        "summary": {
            "resolved_condition_count": len(records),
            "non_checkable_condition_count": len(packets["non_checkable_conditions"]),
            "source_kind_counts": dict(sorted(counts.items())),
        },
        "source_packet_set_fingerprint": packets["packet_set_fingerprint"],
        "source_checkability_result_set_fingerprint": packets[
            "source_checkability_result_set_fingerprint"
        ],
    }
    result["result_set_fingerprint"] = content_sha256(result)
    return validate_given_source_family_result_set(result)


def validate_given_source_family_result_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_source_family_result_set")))
    fingerprint = result.pop("result_set_fingerprint", None)
    if (
        result.get("schema_version") != RESULT_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given source-family result set")
    records = result.get("results")
    skipped = result.get("non_checkable_conditions")
    if not isinstance(records, list) or not isinstance(skipped, list):
        raise ThenAtomizationError("Given source-family result arrays are invalid")
    resolved_ids = [item.get("condition_id") for item in records]
    skipped_ids = [item.get("condition_id") for item in skipped]
    if (
        len(resolved_ids) != len(set(resolved_ids))
        or len(skipped_ids) != len(set(skipped_ids))
        or set(resolved_ids) & set(skipped_ids)
    ):
        raise ThenAtomizationError("Given source-family result identities overlap")
    if any(item.get("source_kind") not in SOURCE_KINDS for item in records):
        raise ThenAtomizationError("Given source-family result label is invalid")
    counts = Counter(item["source_kind"] for item in records)
    expected = {
        "resolved_condition_count": len(records),
        "non_checkable_condition_count": len(skipped),
        "source_kind_counts": dict(sorted(counts.items())),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given source-family result summary mismatch")
    result["result_set_fingerprint"] = fingerprint
    return result


def select_pending_given_source_family_packets(
    full_packet_set: Mapping[str, Any],
    completed_result_set: Mapping[str, Any],
    completed_packet_set: Mapping[str, Any],
) -> dict[str, Any]:
    """Subtract a completed calibration subset from the full source-family batch."""

    full = validate_given_source_family_packet_set(full_packet_set)
    completed = validate_given_source_family_result_set(completed_result_set)
    completed_packets = validate_given_source_family_packet_set(completed_packet_set)
    if completed["source_packet_set_fingerprint"] != completed_packets[
        "packet_set_fingerprint"
    ]:
        raise ThenAtomizationError("completed source-family result lineage mismatch")
    full_by_condition = {
        member["condition_id"]: packet
        for packet in full["packets"]
        for member in packet["members"]
    }
    completed_ids = {item["condition_id"] for item in completed["results"]}
    if not completed_ids.issubset(full_by_condition):
        raise ThenAtomizationError(
            "completed source-family conditions are outside the full batch"
        )
    for packet in completed_packets["packets"]:
        for member in packet["members"]:
            full_packet = full_by_condition.get(member["condition_id"])
            if full_packet is None or full_packet["packet_fingerprint"] != packet[
                "packet_fingerprint"
            ]:
                raise ThenAtomizationError(
                    "completed source-family packet is not identical to the full packet"
                )
    pending_ids = []
    for packet in full["packets"]:
        member_ids = {item["condition_id"] for item in packet["members"]}
        overlap = member_ids & completed_ids
        if overlap and overlap != member_ids:
            raise ThenAtomizationError(
                "deduplicated source-family packet is partially completed"
            )
        if not overlap:
            pending_ids.append(packet["packet_id"])
    if not pending_ids:
        raise ThenAtomizationError("no pending Given source-family packets")
    selected = select_given_source_family_packets(full, pending_ids)
    selected.pop("packet_set_fingerprint")
    selected["completed_condition_count"] = len(completed_ids)
    selected["source_completed_result_set_fingerprint"] = completed[
        "result_set_fingerprint"
    ]
    selected["packet_set_fingerprint"] = content_sha256(selected)
    return validate_given_source_family_packet_set(selected)


def reconcile_given_source_family_results(
    initial_packet_set: Mapping[str, Any],
    initial_responses: Mapping[str, Any],
    adjudication_packet_set: Mapping[str, Any],
    adjudication_responses: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace explicitly selected initial answers with adjudicated answers."""

    initial = validate_given_source_family_packet_set(initial_packet_set)
    adjudication = validate_given_source_family_packet_set(adjudication_packet_set)
    initial_map = deepcopy(dict(_mapping(initial_responses, "$initial_responses")))
    adjudication_map = deepcopy(
        dict(_mapping(adjudication_responses, "$adjudication_responses"))
    )
    if set(initial_map) != {item["packet_id"] for item in initial["packets"]}:
        raise ThenAtomizationError("initial source-family response batch is incomplete")
    if set(adjudication_map) != {
        item["packet_id"] for item in adjudication["packets"]
    }:
        raise ThenAtomizationError(
            "adjudication source-family response batch is incomplete"
        )
    initial_by_condition = {
        member["condition_id"]: packet
        for packet in initial["packets"]
        for member in packet["members"]
    }
    adjudication_by_condition = {}
    for packet in adjudication["packets"]:
        for member in packet["members"]:
            condition_id = member["condition_id"]
            initial_packet = initial_by_condition.get(condition_id)
            if initial_packet is None or initial_packet["packet_fingerprint"] != packet[
                "packet_fingerprint"
            ]:
                raise ThenAtomizationError(
                    "source-family adjudication packet is not an identical initial subset"
                )
            adjudication_by_condition[condition_id] = packet
    records = []
    initial_count = 0
    adjudicated_count = 0
    for packet in initial["packets"]:
        for member in packet["members"]:
            condition_id = member["condition_id"]
            evidence_packet = adjudication_by_condition.get(condition_id, packet)
            response_map = (
                adjudication_map
                if condition_id in adjudication_by_condition
                else initial_map
            )
            checked = validate_given_source_family_response(
                evidence_packet,
                _mapping(response_map[evidence_packet["packet_id"]], "$.response"),
            )
            basis = (
                "accepted_adjudication_response"
                if condition_id in adjudication_by_condition
                else "accepted_initial_response"
            )
            if basis == "accepted_adjudication_response":
                adjudicated_count += 1
            else:
                initial_count += 1
            records.append(
                {
                    "branch_id": member["branch_id"],
                    "condition_id": condition_id,
                    "source_kind": checked["source_kind"],
                    "reason": checked["reason"],
                    "resolution_basis": basis,
                    "source_packet_fingerprint": evidence_packet[
                        "packet_fingerprint"
                    ],
                    "source_logical_form_fingerprint": member[
                        "source_logical_form_fingerprint"
                    ],
                }
            )
    counts = Counter(item["source_kind"] for item in records)
    result = {
        "schema_version": RESULT_SET_VERSION,
        "results": records,
        "non_checkable_conditions": deepcopy(initial["non_checkable_conditions"]),
        "summary": {
            "resolved_condition_count": len(records),
            "non_checkable_condition_count": len(initial["non_checkable_conditions"]),
            "source_kind_counts": dict(sorted(counts.items())),
        },
        "resolution_summary": {
            "initial_response_count": initial_count,
            "adjudicated_response_count": adjudicated_count,
        },
        "source_packet_set_fingerprint": initial["packet_set_fingerprint"],
        "source_adjudication_packet_set_fingerprint": adjudication[
            "packet_set_fingerprint"
        ],
        "source_checkability_result_set_fingerprint": initial[
            "source_checkability_result_set_fingerprint"
        ],
    }
    result["result_set_fingerprint"] = content_sha256(result)
    return validate_given_source_family_result_set(result)


def merge_given_source_family_result_sets(
    full_packet_set: Mapping[str, Any], component_result_sets: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Merge disjoint source-family batches and prove full condition coverage."""

    full = validate_given_source_family_packet_set(full_packet_set)
    if not isinstance(component_result_sets, list) or not component_result_sets:
        raise ThenAtomizationError("source-family result components are required")
    components = [
        validate_given_source_family_result_set(item)
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
            target = expected.get(condition_id)
            if target is None:
                raise ThenAtomizationError(
                    f"component source-family condition is outside full batch: {condition_id}"
                )
            if condition_id in records_by_id:
                raise ThenAtomizationError(
                    f"overlapping source-family result components: {condition_id}"
                )
            if (
                record.get("branch_id") != target["branch_id"]
                or record.get("source_packet_fingerprint")
                != target["packet_fingerprint"]
            ):
                raise ThenAtomizationError(
                    f"source-family result lineage mismatch: {condition_id}"
                )
            records_by_id[condition_id] = deepcopy(record)
    missing = sorted(set(expected) - set(records_by_id))
    if missing:
        raise ThenAtomizationError(
            f"merged source-family results are incomplete: {missing}"
        )
    records = [records_by_id[condition_id] for condition_id in expected]
    counts = Counter(item["source_kind"] for item in records)
    result = {
        "schema_version": RESULT_SET_VERSION,
        "results": records,
        "non_checkable_conditions": deepcopy(full["non_checkable_conditions"]),
        "summary": {
            "resolved_condition_count": len(records),
            "non_checkable_condition_count": len(full["non_checkable_conditions"]),
            "source_kind_counts": dict(sorted(counts.items())),
        },
        "source_packet_set_fingerprint": full["packet_set_fingerprint"],
        "source_checkability_result_set_fingerprint": full[
            "source_checkability_result_set_fingerprint"
        ],
        "source_component_result_set_fingerprints": [
            item["result_set_fingerprint"] for item in components
        ],
    }
    result["result_set_fingerprint"] = content_sha256(result)
    return validate_given_source_family_result_set(result)
