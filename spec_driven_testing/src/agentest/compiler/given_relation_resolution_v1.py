"""Prepare deduplicated all/any decisions for materialized Given conditions."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .given_condition_terms_v1 import validate_given_condition_term_set
from .then_atomization import ThenAtomizationError


TASK_NAME = "given_condition_relation"
PACKET_SET_VERSION = "agentspectesting.given-relation-packet-set/v0.1"
PACKET_VERSION = "agentspectesting.given-relation-packet/v0.1"
DECISIONS = frozenset({"all_required", "any_sufficient", "unclear"})


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
        view["source_clause"] = term["source_clause"]
    return view


def build_given_relation_packets(term_set: Mapping[str, Any]) -> dict[str, Any]:
    terms = validate_given_condition_term_set(term_set)
    candidates = []
    mechanical_single = []
    for branch in terms["branches"]:
        if branch["source_given_logic_status"] != "requires_clause_resolution":
            continue
        if branch["materialization_status"] != "complete":
            continue
        if len(branch["condition_terms"]) == 1:
            mechanical_single.append(branch["branch_id"])
            continue
        views = [
            _condition_view(term, ordinal)
            for ordinal, term in enumerate(branch["condition_terms"], start=1)
        ]
        candidates.append((branch, views))
    grouped: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    for branch, views in candidates:
        key = content_sha256({"full_given": branch["gwt"]["given"], "conditions": views})
        grouped[key].append((branch, views))
    packets = []
    for key, members in grouped.items():
        representative, views = members[0]
        packet = {
            "schema_version": PACKET_VERSION,
            "packet_id": f"GIVEN-RELATION::{key[:16]}::R01",
            "task_name": TASK_NAME,
            "model_input": {
                "full_given": representative["gwt"]["given"],
                "conditions": views,
            },
            "members": [
                {
                    "branch_id": branch["branch_id"],
                    "condition_ids": [
                        term["condition_id"] for term in branch["condition_terms"]
                    ],
                    "source_condition_branch_fingerprint": branch[
                        "condition_branch_fingerprint"
                    ],
                }
                for branch, _ in members
            ],
        }
        packet["packet_fingerprint"] = content_sha256(packet)
        packets.append(packet)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": packets,
        "mechanical_single_branch_ids": mechanical_single,
        "summary": {
            "source_unresolved_logic_branch_count": len(candidates)
            + len(mechanical_single),
            "mechanical_single_branch_count": len(mechanical_single),
            "multi_condition_branch_count": len(candidates),
            "unique_relation_count": len(packets),
            "expected_model_calls": len(packets),
        },
        "source_condition_term_set_fingerprint": terms[
            "condition_term_set_fingerprint"
        ],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_relation_packet_set(result)


def validate_given_relation_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_relation_packet_set")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if (
        result.get("schema_version") != PACKET_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given relation packet set")
    packets = result.get("packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("Given relation packets must be an array")
    ids = []
    branches = []
    for raw in packets:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if (
            packet.get("schema_version") != PACKET_VERSION
            or packet_fingerprint != content_sha256(packet)
            or packet.get("task_name") != TASK_NAME
        ):
            raise ThenAtomizationError("invalid Given relation packet")
        model_input = _mapping(packet.get("model_input"), "$.model_input")
        if set(model_input) != {"full_given", "conditions"}:
            raise ThenAtomizationError("Given relation input fields are invalid")
        keys = [item.get("condition_key") for item in model_input["conditions"]]
        if keys != [f"K{i}" for i in range(1, len(keys) + 1)] or len(keys) < 2:
            raise ThenAtomizationError("Given relation condition keys are invalid")
        ids.append(packet.get("packet_id"))
        branches.extend(member.get("branch_id") for member in packet.get("members") or [])
    if len(ids) != len(set(ids)) or len(branches) != len(set(branches)):
        raise ThenAtomizationError("Given relation packet identities overlap")
    singles = result.get("mechanical_single_branch_ids")
    if not isinstance(singles, list) or set(singles) & set(branches):
        raise ThenAtomizationError("Given relation mechanical branches overlap")
    expected = {
        "source_unresolved_logic_branch_count": len(branches) + len(singles),
        "mechanical_single_branch_count": len(singles),
        "multi_condition_branch_count": len(branches),
        "unique_relation_count": len(packets),
        "expected_model_calls": len(packets),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given relation packet summary mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def validate_given_relation_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, str]:
    packet = _mapping(packet, "$packet")
    response = deepcopy(dict(_mapping(response, "$response")))
    if set(response) != {"decision", "reason"}:
        raise ThenAtomizationError("Given relation response fields are invalid")
    if response.get("decision") not in DECISIONS:
        raise ThenAtomizationError("Given relation decision is invalid")
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 300:
        raise ThenAtomizationError("Given relation reason must be concise text")
    return {"decision": response["decision"], "reason": reason.strip()}


def render_given_relation_prompt(packet: Mapping[str, Any], template: str) -> str:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    if packet.get("task_name") != TASK_NAME or "TASK: given_condition_relation" not in template:
        raise ThenAtomizationError("Given relation prompt template/task mismatch")
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        packet["model_input"], ensure_ascii=False, indent=2
    )
