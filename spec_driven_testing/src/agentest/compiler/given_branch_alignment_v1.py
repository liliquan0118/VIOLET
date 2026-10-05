"""Prepare the minimal branch-to-condition alignment task for Given variants."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .given_logical_form_v1 import validate_given_logical_form_set
from .then_atomization import ThenAtomizationError


TASK_NAME = "given_branch_condition_alignment"
PACKET_SET_VERSION = "agentspectesting.given-branch-alignment-packet-set/v0.1"
PACKET_VERSION = "agentspectesting.given-branch-alignment-packet/v0.1"
AUDIT_SET_VERSION = "agentspectesting.given-branch-alignment-audit-set/v0.1"
RESULT_SET_VERSION = "agentspectesting.given-branch-alignment-result-set/v0.1"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _condition_view(term: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    view: dict[str, Any] = {"condition_key": f"K{ordinal}"}
    if term["condition_kind"] == "structured_condition":
        view["structured_condition"] = deepcopy(term["condition"])
    elif term["condition_kind"] == "compiler_constant":
        view["constant"] = term["constant"]
    else:
        view["exact_span"] = term["exact_span"]
    return view


def build_given_branch_alignment_packets(
    logical_form_set: Mapping[str, Any],
) -> dict[str, Any]:
    logical_forms = validate_given_logical_form_set(logical_form_set)
    packets = []
    for form in logical_forms["forms"]:
        if form["branch_alignment"]["status"] != "requires_resolution":
            continue
        views = [
            _condition_view(term, ordinal)
            for ordinal, term in enumerate(form["condition_terms"], start=1)
        ]
        packet = {
            "schema_version": PACKET_VERSION,
            "packet_id": f"GIVEN-ALIGN::{content_sha256(form['branch_id'])[:16]}::A01",
            "task_name": TASK_NAME,
            "model_input": {
                "full_given": form["gwt"]["given"],
                "then_branch": form["gwt"]["then"],
                "conditions": views,
            },
            "branch_id": form["branch_id"],
            "condition_key_to_id": {
                f"K{ordinal}": term["condition_id"]
                for ordinal, term in enumerate(form["condition_terms"], start=1)
            },
            "source_logical_form_fingerprint": form["logical_form_fingerprint"],
        }
        packet["packet_fingerprint"] = content_sha256(packet)
        packets.append(packet)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": packets,
        "summary": {
            "requires_alignment_branch_count": len(packets),
            "expected_model_calls": len(packets),
        },
        "source_logical_form_set_fingerprint": logical_forms[
            "logical_form_set_fingerprint"
        ],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_branch_alignment_packet_set(result)


def validate_given_branch_alignment_packet_set(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_branch_alignment_packet_set")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if (
        result.get("schema_version") != PACKET_SET_VERSION
        or fingerprint != content_sha256(result)
        or result.get("task_name") != TASK_NAME
    ):
        raise ThenAtomizationError("invalid Given branch alignment packet set")
    packets = result.get("packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("Given branch alignment packets must be an array")
    packet_ids = []
    branch_ids = []
    for raw in packets:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if (
            packet.get("schema_version") != PACKET_VERSION
            or packet_fingerprint != content_sha256(packet)
            or packet.get("task_name") != TASK_NAME
        ):
            raise ThenAtomizationError("invalid Given branch alignment packet")
        model_input = _mapping(packet.get("model_input"), "$.model_input")
        if set(model_input) not in (
            {"full_given", "then_branch", "conditions"},
            {"then_branch", "conditions"},
        ):
            raise ThenAtomizationError("Given alignment input fields are invalid")
        keys = [item.get("condition_key") for item in model_input["conditions"]]
        expected_keys = [f"K{i}" for i in range(1, len(keys) + 1)]
        if len(keys) < 2 or keys != expected_keys:
            raise ThenAtomizationError("Given alignment condition keys are invalid")
        if packet.get("condition_key_to_id", {}).keys() != dict.fromkeys(keys).keys():
            raise ThenAtomizationError("Given alignment key mapping is invalid")
        packet_ids.append(packet.get("packet_id"))
        branch_ids.append(packet.get("branch_id"))
    if len(packet_ids) != len(set(packet_ids)) or len(branch_ids) != len(set(branch_ids)):
        raise ThenAtomizationError("Given alignment packet identities are not unique")
    expected_summary = {
        "requires_alignment_branch_count": len(packets),
        "expected_model_calls": len(packets),
    }
    if result.get("summary") != expected_summary:
        raise ThenAtomizationError("Given alignment packet summary mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def validate_given_branch_alignment_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    packet = _mapping(packet, "$packet")
    response = deepcopy(dict(_mapping(response, "$response")))
    if set(response) != {"condition_keys", "reason"}:
        raise ThenAtomizationError("Given alignment response fields are invalid")
    keys = response.get("condition_keys")
    allowed = set(packet["condition_key_to_id"])
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(item, str) for item in keys)
        or len(keys) != len(set(keys))
        or not set(keys).issubset(allowed)
    ):
        raise ThenAtomizationError("Given alignment condition keys are invalid")
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 300:
        raise ThenAtomizationError("Given alignment reason must be concise text")
    return {"condition_keys": keys, "reason": reason.strip()}


def render_given_branch_alignment_prompt(packet: Mapping[str, Any], template: str) -> str:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    if packet.get("task_name") != TASK_NAME or f"TASK: {TASK_NAME}" not in template:
        raise ThenAtomizationError("Given alignment prompt template/task mismatch")
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        packet["model_input"], ensure_ascii=False, indent=2
    )


def audit_given_branch_alignment_responses(
    packet_set: Mapping[str, Any], responses: Mapping[str, Any]
) -> dict[str, Any]:
    """Conservatively flag an OR-variant response that selects every alternative."""

    packets = validate_given_branch_alignment_packet_set(packet_set)
    response_map = deepcopy(dict(_mapping(responses, "$responses")))
    packet_index = {item["packet_id"]: item for item in packets["packets"]}
    if set(response_map) != set(packet_index):
        raise ThenAtomizationError("Given alignment responses must cover the closed batch")
    records = []
    for packet in packets["packets"]:
        checked = validate_given_branch_alignment_response(
            packet, _mapping(response_map[packet["packet_id"]], "$.response")
        )
        all_keys = list(packet["condition_key_to_id"])
        if set(checked["condition_keys"]) == set(all_keys):
            status = "needs_adjudication"
            diagnostic = "specific_variant_selected_every_given_alternative"
        else:
            status = "accepted"
            diagnostic = None
        records.append(
            {
                "packet_id": packet["packet_id"],
                "branch_id": packet["branch_id"],
                "status": status,
                "diagnostic": diagnostic,
                "response": checked,
                "source_packet_fingerprint": packet["packet_fingerprint"],
            }
        )
    result = {
        "schema_version": AUDIT_SET_VERSION,
        "records": records,
        "summary": {
            "response_count": len(records),
            "accepted_count": sum(item["status"] == "accepted" for item in records),
            "needs_adjudication_count": sum(
                item["status"] == "needs_adjudication" for item in records
            ),
        },
        "source_packet_set_fingerprint": packets["packet_set_fingerprint"],
    }
    result["audit_set_fingerprint"] = content_sha256(result)
    return result


def build_given_branch_alignment_adjudication_packets(
    packet_set: Mapping[str, Any], audit_set: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a smaller, clearer retry batch without the distracting full Given."""

    packets = validate_given_branch_alignment_packet_set(packet_set)
    audit = deepcopy(dict(_mapping(audit_set, "$audit_set")))
    fingerprint = audit.pop("audit_set_fingerprint", None)
    if (
        audit.get("schema_version") != AUDIT_SET_VERSION
        or fingerprint != content_sha256(audit)
        or audit.get("source_packet_set_fingerprint")
        != packets["packet_set_fingerprint"]
    ):
        raise ThenAtomizationError("invalid Given alignment audit set")
    source_index = {item["packet_id"]: item for item in packets["packets"]}
    retry_packets = []
    for record in audit.get("records") or []:
        if record.get("status") != "needs_adjudication":
            continue
        source = source_index[record["packet_id"]]
        packet = {
            "schema_version": PACKET_VERSION,
            "packet_id": source["packet_id"] + "::ADJ01",
            "task_name": TASK_NAME,
            "model_input": {
                "then_branch": source["model_input"]["then_branch"],
                "conditions": deepcopy(source["model_input"]["conditions"]),
            },
            "branch_id": source["branch_id"],
            "condition_key_to_id": deepcopy(source["condition_key_to_id"]),
            "source_logical_form_fingerprint": source[
                "source_logical_form_fingerprint"
            ],
            "source_alignment_packet_fingerprint": source["packet_fingerprint"],
            "source_audit_diagnostic": record["diagnostic"],
        }
        packet["packet_fingerprint"] = content_sha256(packet)
        retry_packets.append(packet)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": retry_packets,
        "summary": {
            "requires_alignment_branch_count": len(retry_packets),
            "expected_model_calls": len(retry_packets),
        },
        "source_logical_form_set_fingerprint": packets[
            "source_logical_form_set_fingerprint"
        ],
        "source_alignment_audit_set_fingerprint": fingerprint,
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return result


def materialize_final_given_branch_alignments(
    logical_form_set: Mapping[str, Any],
    packet_set: Mapping[str, Any],
    initial_responses: Mapping[str, Any],
    audit_set: Mapping[str, Any],
    adjudication_packet_set: Mapping[str, Any],
    adjudication_responses: Mapping[str, Any],
) -> dict[str, Any]:
    """Use accepted first-pass answers and adjudicated replacements to close alignment."""

    logical_forms = validate_given_logical_form_set(logical_form_set)
    packets = validate_given_branch_alignment_packet_set(packet_set)
    if packets["source_logical_form_set_fingerprint"] != logical_forms[
        "logical_form_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given alignment logical-form lineage mismatch")
    initial_map = deepcopy(dict(_mapping(initial_responses, "$initial_responses")))
    if set(initial_map) != {item["packet_id"] for item in packets["packets"]}:
        raise ThenAtomizationError("initial Given alignment responses are incomplete")

    audit = deepcopy(dict(_mapping(audit_set, "$audit_set")))
    audit_fingerprint = audit.pop("audit_set_fingerprint", None)
    if (
        audit.get("schema_version") != AUDIT_SET_VERSION
        or audit_fingerprint != content_sha256(audit)
        or audit.get("source_packet_set_fingerprint")
        != packets["packet_set_fingerprint"]
    ):
        raise ThenAtomizationError("invalid Given alignment audit lineage")
    audit_index = {item["packet_id"]: item for item in audit.get("records") or []}
    if set(audit_index) != set(initial_map):
        raise ThenAtomizationError("Given alignment audit does not cover initial batch")

    adjudication_packets = validate_given_branch_alignment_packet_set(
        adjudication_packet_set
    )
    if adjudication_packets.get("source_alignment_audit_set_fingerprint") != audit_fingerprint:
        raise ThenAtomizationError("Given alignment adjudication audit lineage mismatch")
    adjudication_map = deepcopy(
        dict(_mapping(adjudication_responses, "$adjudication_responses"))
    )
    adjudication_index = {
        item["source_alignment_packet_fingerprint"]: item
        for item in adjudication_packets["packets"]
    }
    if set(adjudication_map) != {
        item["packet_id"] for item in adjudication_packets["packets"]
    }:
        raise ThenAtomizationError("Given alignment adjudication responses are incomplete")

    records = []
    for packet in packets["packets"]:
        audit_record = audit_index[packet["packet_id"]]
        if audit_record["status"] == "accepted":
            checked = validate_given_branch_alignment_response(
                packet, _mapping(initial_map[packet["packet_id"]], "$.response")
            )
            basis = "accepted_initial_response"
            evidence_packet = packet
        elif audit_record["status"] == "needs_adjudication":
            evidence_packet = adjudication_index.get(packet["packet_fingerprint"])
            if evidence_packet is None:
                raise ThenAtomizationError(
                    f"missing Given alignment adjudication for {packet['branch_id']}"
                )
            checked = validate_given_branch_alignment_response(
                evidence_packet,
                _mapping(
                    adjudication_map[evidence_packet["packet_id"]],
                    "$.adjudication_response",
                ),
            )
            basis = "accepted_adjudication_response"
        else:
            raise ThenAtomizationError("unknown Given alignment audit status")
        active_ids = [
            packet["condition_key_to_id"][key] for key in checked["condition_keys"]
        ]
        records.append(
            {
                "branch_id": packet["branch_id"],
                "active_condition_ids": active_ids,
                "selected_condition_keys": checked["condition_keys"],
                "resolution_basis": basis,
                "reason": checked["reason"],
                "source_alignment_packet_fingerprint": packet[
                    "packet_fingerprint"
                ],
                "evidence_packet_fingerprint": evidence_packet[
                    "packet_fingerprint"
                ],
            }
        )
    result = {
        "schema_version": RESULT_SET_VERSION,
        "results": records,
        "summary": {
            "resolved_branch_count": len(records),
            "initial_response_count": sum(
                item["resolution_basis"] == "accepted_initial_response"
                for item in records
            ),
            "adjudicated_response_count": sum(
                item["resolution_basis"] == "accepted_adjudication_response"
                for item in records
            ),
        },
        "source_logical_form_set_fingerprint": logical_forms[
            "logical_form_set_fingerprint"
        ],
        "source_packet_set_fingerprint": packets["packet_set_fingerprint"],
        "source_audit_set_fingerprint": audit_fingerprint,
        "source_adjudication_packet_set_fingerprint": adjudication_packets[
            "packet_set_fingerprint"
        ],
    }
    result["result_set_fingerprint"] = content_sha256(result)
    return result


def apply_final_given_branch_alignments(
    logical_form_set: Mapping[str, Any], alignment_result_set: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a fingerprint-closed logical-form set with every active condition known."""

    logical_forms = validate_given_logical_form_set(logical_form_set)
    results = deepcopy(dict(_mapping(alignment_result_set, "$alignment_result_set")))
    result_fingerprint = results.pop("result_set_fingerprint", None)
    if (
        results.get("schema_version") != RESULT_SET_VERSION
        or result_fingerprint != content_sha256(results)
        or results.get("source_logical_form_set_fingerprint")
        != logical_forms["logical_form_set_fingerprint"]
    ):
        raise ThenAtomizationError("invalid Given alignment result set")
    result_index = {item["branch_id"]: item for item in results.get("results") or []}
    output = deepcopy(logical_forms)
    output.pop("logical_form_set_fingerprint", None)
    alignment_counts: dict[str, int] = {}
    for form in output["forms"]:
        form.pop("logical_form_fingerprint", None)
        if form["branch_alignment"]["status"] == "requires_resolution":
            resolution = result_index.pop(form["branch_id"], None)
            if resolution is None:
                raise ThenAtomizationError(
                    f"missing final Given alignment for {form['branch_id']}"
                )
            if not set(resolution["active_condition_ids"]).issubset(
                set(form["condition_ids"])
            ):
                raise ThenAtomizationError("final Given alignment selected unknown condition")
            form["branch_alignment"] = {
                "status": "reviewed_model_alignment",
                "active_condition_ids": deepcopy(resolution["active_condition_ids"]),
                "resolution_basis": resolution["resolution_basis"],
                "reason": resolution["reason"],
                "source_alignment_packet_fingerprint": resolution[
                    "source_alignment_packet_fingerprint"
                ],
                "evidence_packet_fingerprint": resolution[
                    "evidence_packet_fingerprint"
                ],
            }
        status = form["branch_alignment"]["status"]
        alignment_counts[status] = alignment_counts.get(status, 0) + 1
        form["logical_form_fingerprint"] = content_sha256(form)
    if result_index:
        raise ThenAtomizationError("final Given alignment contains unknown branches")
    output["summary"]["branch_alignment_status_counts"] = dict(
        sorted(alignment_counts.items())
    )
    output["source_pre_alignment_logical_form_set_fingerprint"] = logical_forms[
        "logical_form_set_fingerprint"
    ]
    output["source_alignment_result_set_fingerprint"] = result_fingerprint
    output["logical_form_set_fingerprint"] = content_sha256(output)
    return validate_given_logical_form_set(output)
