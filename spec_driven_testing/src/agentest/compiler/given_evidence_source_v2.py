"""Prepare evidence-source decisions for final, branch-aligned Given conditions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import (
    CAP_AGENT_TOOL_SCHEMAS,
    CAP_TOOL_OBSERVABLE_ENDPOINTS,
    content_sha256,
    validate_accepted_artifact_catalog,
)
from .driver_evidence_capabilities_v1 import (
    NON_SOURCE_OUTCOMES,
    SOURCE_KINDS,
    validate_driver_evidence_capability_profile,
)
from .given_logical_form_v1 import validate_given_logical_form_set
from .then_atomization import ThenAtomizationError
from .tool_endpoint_lexical_retrieval_v1 import (
    endpoint_view as _endpoint_view_impl,
    relevant_endpoints as _relevant_endpoints_impl,
    tokens as _tokens,
)


TASK_NAME = "given_evidence_source_v2"
PACKET_SET_VERSION = "agentspectesting.given-evidence-source-packet-set/v0.2"
PACKET_VERSION = "agentspectesting.given-evidence-source-packet/v0.2"
MECHANICAL_VERSION = "agentspectesting.given-evidence-mechanical-resolution/v0.2"
MODEL_SOURCE_KINDS = SOURCE_KINDS | NON_SOURCE_OUTCOMES

_DERIVED_OPS = {
    "count_eq", "count_ne", "count_gt", "count_ge", "count_lt", "count_le",
    "gt", "ge", "lt", "le",
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _document_for_capability(catalog: Mapping[str, Any], capability: str) -> Mapping[str, Any]:
    providers = catalog["capability_index"].get(capability) or []
    if len(providers) != 1:
        raise ThenAtomizationError(
            f"accepted artifact capability {capability!r} must have one provider"
        )
    return _mapping(catalog["artifacts"][providers[0]]["document"], capability)


def _system_prompt_lines(agent_spec: Mapping[str, Any]) -> list[str]:
    prompt = agent_spec.get("system_prompt")
    if isinstance(prompt, Mapping):
        prompt = prompt.get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ThenAtomizationError("accepted agent spec system prompt text is missing")
    return [line.strip(" #-\t") for line in prompt.splitlines() if line.strip(" #-\t")]


def _relevant_policy_lines(lines: list[str], condition_text: str, limit: int = 5) -> list[str]:
    query = _tokens(condition_text)
    scored = []
    for ordinal, line in enumerate(lines):
        overlap = query & _tokens(line)
        if overlap:
            scored.append((-len(overlap), ordinal, line))
    return [line for _, _, line in sorted(scored)[:limit]]


_endpoint_view = _endpoint_view_impl
_relevant_endpoints = _relevant_endpoints_impl


def _has_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "ref" in value or any(_has_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_reference(item) for item in value)
    return False


def _mechanical_source(condition: Mapping[str, Any]) -> tuple[str, str]:
    op = condition.get("op")
    if op in _DERIVED_OPS or condition.get("quant") is not None or _has_reference(
        condition.get("value")
    ):
        return "derived_state", "structured predicate requires deterministic evaluation"
    return "fixture_state", "structured predicate directly addresses fixture state"


def _condition_text(term: Mapping[str, Any]) -> str:
    text = term.get("exact_span")
    if not isinstance(text, str) or not text.strip():
        raise ThenAtomizationError("natural-language Given condition text is missing")
    return text.strip()


def build_given_evidence_source_packets_v2(
    logical_form_set: Mapping[str, Any],
    capability_profile: Mapping[str, Any],
    accepted_artifact_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    forms = validate_given_logical_form_set(logical_form_set)
    profile = validate_driver_evidence_capability_profile(capability_profile)
    catalog = validate_accepted_artifact_catalog(accepted_artifact_catalog)
    if profile["domain"] != catalog["domain"]:
        raise ThenAtomizationError("Driver profile and artifact catalog domains disagree")
    agent_spec = _document_for_capability(catalog, CAP_AGENT_TOOL_SCHEMAS)
    tool_catalog = _document_for_capability(catalog, CAP_TOOL_OBSERVABLE_ENDPOINTS)
    policy_lines = _system_prompt_lines(agent_spec)
    allowed_source_kinds = sorted(MODEL_SOURCE_KINDS)

    mechanical = []
    model_candidates = []
    active_count = 0
    for form in forms["forms"]:
        active_ids = form["branch_alignment"].get("active_condition_ids")
        if not isinstance(active_ids, list):
            raise ThenAtomizationError(
                f"Given branch alignment is not closed for {form['branch_id']}"
            )
        term_index = {item["condition_id"]: item for item in form["condition_terms"]}
        for condition_id in active_ids:
            active_count += 1
            term = term_index[condition_id]
            kind = term["condition_kind"]
            if kind == "compiler_constant":
                source_kind = "compiler_constant"
                reason = "the branch is unconditionally active"
            elif kind == "structured_condition":
                source_kind, reason = _mechanical_source(term["condition"])
            else:
                text = _condition_text(term)
                endpoints, endpoint_count, action_names = _relevant_endpoints(
                    tool_catalog, text
                )
                model_input = {
                    "condition": text,
                    "when": form["gwt"]["when"],
                    "selected_spec_rule": deepcopy(form["source_rule_context"]),
                    "accepted_evidence": {
                        "matching_system_policy_lines": _relevant_policy_lines(
                            policy_lines, text
                        ),
                        "matching_state_observables": endpoints,
                        "available_agent_action_names": action_names,
                        "search_scope": {
                            "system_policy_line_count": len(policy_lines),
                            "observable_endpoint_count": endpoint_count,
                            "matching_endpoint_limit": 8,
                        },
                    },
                    "allowed_source_kinds": allowed_source_kinds,
                }
                model_candidates.append((form, term, model_input))
                continue
            record = {
                "schema_version": MECHANICAL_VERSION,
                "branch_id": form["branch_id"],
                "condition_id": condition_id,
                "condition_kind": kind,
                "source_kind": source_kind,
                "resolution_basis": "mechanical",
                "reason": reason,
                "source_logical_form_fingerprint": form[
                    "logical_form_fingerprint"
                ],
            }
            record["resolution_fingerprint"] = content_sha256(record)
            mechanical.append(record)

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for form, term, model_input in model_candidates:
        grouped[content_sha256(model_input)].append((form, term, model_input))
    packets = []
    for key, members in grouped.items():
        representative_form, _, model_input = members[0]
        packet = {
            "schema_version": PACKET_VERSION,
            "packet_id": f"GIVEN-SOURCE-V2::{key[:16]}::S01",
            "task_name": TASK_NAME,
            "model_input": deepcopy(model_input),
            "members": [
                {
                    "branch_id": form["branch_id"],
                    "condition_id": term["condition_id"],
                    "source_logical_form_fingerprint": form[
                        "logical_form_fingerprint"
                    ],
                }
                for form, term, _ in members
            ],
        }
        packet["packet_fingerprint"] = content_sha256(packet)
        packets.append(packet)
    packets.sort(key=lambda item: item["packet_id"])
    counts = Counter(item["source_kind"] for item in mechanical)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": packets,
        "mechanical_resolutions": mechanical,
        "summary": {
            "active_condition_count": active_count,
            "mechanical_resolution_count": len(mechanical),
            "mechanical_source_kind_counts": dict(sorted(counts.items())),
            "model_condition_count": len(model_candidates),
            "unique_model_packet_count": len(packets),
            "expected_model_calls": len(packets),
        },
        "source_logical_form_set_fingerprint": forms[
            "logical_form_set_fingerprint"
        ],
        "source_driver_profile_fingerprint": profile["profile_fingerprint"],
        "source_artifact_catalog_fingerprint": catalog["catalog_fingerprint"],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_evidence_source_packet_set_v2(result)


def validate_given_evidence_source_packet_set_v2(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_evidence_source_packet_set")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if (
        result.get("schema_version") != PACKET_SET_VERSION
        or result.get("task_name") != TASK_NAME
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given evidence-source v2 packet set")
    packet_ids = []
    model_conditions = []
    for raw in result.get("packets") or []:
        packet = deepcopy(dict(_mapping(raw, "$.packets[]")))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if (
            packet.get("schema_version") != PACKET_VERSION
            or packet.get("task_name") != TASK_NAME
            or packet_fingerprint != content_sha256(packet)
        ):
            raise ThenAtomizationError("invalid Given evidence-source v2 packet")
        model_input = _mapping(packet.get("model_input"), "$.model_input")
        if set(model_input) != {
            "condition", "when", "selected_spec_rule", "accepted_evidence",
            "allowed_source_kinds",
        }:
            raise ThenAtomizationError("Given evidence-source v2 input is not minimal")
        if model_input["allowed_source_kinds"] != sorted(MODEL_SOURCE_KINDS):
            raise ThenAtomizationError("Given evidence-source v2 options are incomplete")
        packet_ids.append(packet.get("packet_id"))
        model_conditions.extend(item.get("condition_id") for item in packet.get("members") or [])
    mechanical = result.get("mechanical_resolutions") or []
    mechanical_conditions = [item.get("condition_id") for item in mechanical]
    if len(packet_ids) != len(set(packet_ids)):
        raise ThenAtomizationError("Given evidence-source packet IDs are not unique")
    if set(model_conditions) & set(mechanical_conditions):
        raise ThenAtomizationError("model and mechanical Given conditions overlap")
    if len(model_conditions + mechanical_conditions) != len(
        set(model_conditions + mechanical_conditions)
    ):
        raise ThenAtomizationError("Given evidence condition identities overlap")
    counts = Counter(item.get("source_kind") for item in mechanical)
    expected = {
        "active_condition_count": len(model_conditions) + len(mechanical_conditions),
        "mechanical_resolution_count": len(mechanical),
        "mechanical_source_kind_counts": dict(sorted(counts.items())),
        "model_condition_count": len(model_conditions),
        "unique_model_packet_count": len(packet_ids),
        "expected_model_calls": len(packet_ids),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given evidence-source v2 summary mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def validate_given_evidence_source_response_v2(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, str]:
    _mapping(packet, "$packet")
    response = deepcopy(dict(_mapping(response, "$response")))
    if set(response) != {"source_kind", "reason"}:
        raise ThenAtomizationError("Given evidence-source response fields are invalid")
    if response.get("source_kind") not in MODEL_SOURCE_KINDS:
        raise ThenAtomizationError("Given evidence-source response label is invalid")
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 400:
        raise ThenAtomizationError("Given evidence-source reason must be concise text")
    return {"source_kind": response["source_kind"], "reason": reason.strip()}


def render_given_evidence_source_prompt_v2(packet: Mapping[str, Any], template: str) -> str:
    packet = deepcopy(dict(_mapping(packet, "$packet")))
    if packet.get("task_name") != TASK_NAME or f"TASK: {TASK_NAME}" not in template:
        raise ThenAtomizationError("Given evidence-source v2 prompt/task mismatch")
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        packet["model_input"], ensure_ascii=False, indent=2
    )


def select_given_evidence_source_packets_v2(
    packet_set: Mapping[str, Any], packet_ids: list[str]
) -> dict[str, Any]:
    """Select a fingerprint-closed calibration subset without supplying answers."""

    packets = validate_given_evidence_source_packet_set_v2(packet_set)
    if (
        not isinstance(packet_ids, list)
        or not packet_ids
        or len(packet_ids) != len(set(packet_ids))
        or any(not isinstance(item, str) or not item for item in packet_ids)
    ):
        raise ThenAtomizationError("Given evidence-source selection IDs are invalid")
    packet_index = {item["packet_id"]: item for item in packets["packets"]}
    missing = sorted(set(packet_ids) - set(packet_index))
    if missing:
        raise ThenAtomizationError(f"unknown Given evidence-source packet IDs: {missing}")
    selected = [deepcopy(packet_index[item]) for item in packet_ids]
    model_condition_count = sum(len(item["members"]) for item in selected)
    result = {
        "schema_version": PACKET_SET_VERSION,
        "task_name": TASK_NAME,
        "packets": selected,
        "mechanical_resolutions": [],
        "summary": {
            "active_condition_count": model_condition_count,
            "mechanical_resolution_count": 0,
            "mechanical_source_kind_counts": {},
            "model_condition_count": model_condition_count,
            "unique_model_packet_count": len(selected),
            "expected_model_calls": len(selected),
        },
        "source_logical_form_set_fingerprint": packets[
            "source_logical_form_set_fingerprint"
        ],
        "source_driver_profile_fingerprint": packets[
            "source_driver_profile_fingerprint"
        ],
        "source_artifact_catalog_fingerprint": packets[
            "source_artifact_catalog_fingerprint"
        ],
        "source_full_packet_set_fingerprint": packets["packet_set_fingerprint"],
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return validate_given_evidence_source_packet_set_v2(result)
