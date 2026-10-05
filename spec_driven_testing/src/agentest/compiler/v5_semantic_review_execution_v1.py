"""Closed-batch validation and one-attempt text review, independent of SDK setup."""

import json
from copy import deepcopy

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import verify_fingerprint
from .v5_semantic_review_packets_v1 import DECISIONS, validate_text_review_response
from .v5_when_split_review_v1 import (
    DECISIONS as SPLIT_DECISIONS, KIND, RESTRICTIONS, validate_split_response,
)
from .v5_provider_error_diagnostics_v1 import diagnose_provider_error
from .v5_json_mode_preflight_v1 import validate_json_object_messages
from .v5_restriction_batch_v1 import PILOT_SCHEMA, PREFLIGHT_SCHEMA, validate_restriction_sources
from .v5_when_explicit_information_v1 import (
    EXTRACT, EXTRACT_V2, ALIGN, PILOT_SCHEMA as DETAIL_PILOT_SCHEMA, PREFLIGHT_SCHEMA as DETAIL_PREFLIGHT_SCHEMA,
    PILOT_SCHEMA_V2 as DETAIL_PILOT_SCHEMA_V2, PREFLIGHT_SCHEMA_V2 as DETAIL_PREFLIGHT_SCHEMA_V2,
    validate_explicit_response, validate_detail_sources,
)
from .v5_when_difference_review_v1 import (
    DISCOVER, PILOT_SCHEMA as DIFFERENCE_PILOT_SCHEMA,
    PREFLIGHT_SCHEMA as DIFFERENCE_PREFLIGHT_SCHEMA,
    validate_difference_response, validate_difference_sources,
)
from .v5_when_context_compatibility_v1 import (
    TASK as CONTEXT_TASK, PILOT_SCHEMA as CONTEXT_PILOT_SCHEMA,
    PREFLIGHT_SCHEMA as CONTEXT_PREFLIGHT_SCHEMA, validate_context_response, validate_context_sources,
)
from .v5_when_kind_expansion_v1 import (
    PILOT_SCHEMA as EXPANSION_PILOT_SCHEMA, PREFLIGHT_SCHEMA as EXPANSION_PREFLIGHT_SCHEMA,
    validate_expansion_sources,
)


# A batch's schema fixes its allowed questions; approval for question one must
# never silently expand into question two after receiving a model answer.
BATCH_PROFILES = {
    "agentspectesting.v5-semantic-pilot/v0.1": (
        "agentspectesting.v5-2b1-preflight/v0.1", set(DECISIONS)),
    "agentspectesting.v5-when-split-pilot/v0.1": (
        "agentspectesting.v5-when-split-preflight/v0.1", {KIND}),
    PILOT_SCHEMA: (PREFLIGHT_SCHEMA, {RESTRICTIONS}),
    DETAIL_PILOT_SCHEMA: (DETAIL_PREFLIGHT_SCHEMA, {EXTRACT}),
    DETAIL_PILOT_SCHEMA_V2: (DETAIL_PREFLIGHT_SCHEMA_V2, {EXTRACT_V2}),
    DIFFERENCE_PILOT_SCHEMA: (DIFFERENCE_PREFLIGHT_SCHEMA, {DISCOVER}),
    CONTEXT_PILOT_SCHEMA: (CONTEXT_PREFLIGHT_SCHEMA, {CONTEXT_TASK}),
    EXPANSION_PILOT_SCHEMA: (EXPANSION_PREFLIGHT_SCHEMA, {KIND}),
}


def _response_validator(packet):
    verify_fingerprint(packet, "packet_fingerprint")
    if packet["task"] == CONTEXT_TASK:
        return validate_context_response
    if packet["task"] == DISCOVER:
        return validate_difference_response
    if packet["task"] in {EXTRACT, EXTRACT_V2, ALIGN}:
        return validate_explicit_response
    if packet["task"] in SPLIT_DECISIONS:
        return validate_split_response
    if packet["task"] in DECISIONS:
        return validate_text_review_response
    raise ValueError("unsupported review task")


def validate_approved_pilot(pilot, preflight, *, approved_fingerprint, approved_count):
    verify_fingerprint(pilot, "pilot_fingerprint")
    verify_fingerprint(preflight, "preflight_fingerprint")
    profile = BATCH_PROFILES.get(pilot.get("schema_version"))
    if profile is None or preflight.get("schema_version") != profile[0]:
        raise ValueError("unsupported or mismatched pilot/preflight schemas")
    if pilot["pilot_fingerprint"] != approved_fingerprint or preflight["pilot_fingerprint"] != approved_fingerprint:
        raise ValueError("pilot differs from approved fingerprint")
    if type(approved_count) is not int or approved_count != len(pilot["packets"]) or approved_count <= 0:
        raise ValueError("approval count must exactly match closed pilot")
    if pilot["source_packet_set_fingerprint"] != preflight["source_packet_set_fingerprint"]:
        raise ValueError("pilot and preflight pool differ")
    if pilot["automatic_retries"] != 0 or pilot["max_calls_per_packet"] != 1 or pilot["target_agent_calls"] != 0:
        raise ValueError("pilot is not a one-attempt text-only batch")
    if pilot["proposed_model"] != "deepseek-v4-flash":
        raise ValueError("unexpected model")
    ids = []
    for packet in pilot["packets"]:
        verify_fingerprint(packet, "packet_fingerprint")
        if packet["task"] not in profile[1]:
            raise ValueError("review task is outside this approved batch type")
        if [m["role"] for m in packet["messages"]] != ["system", "user"]:
            raise ValueError("unexpected outbound messages")
        validate_json_object_messages(packet["messages"])
        ids.append(packet["packet_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pilot packet IDs")
    if pilot["schema_version"] == PILOT_SCHEMA:
        validate_restriction_sources(pilot, preflight)
    if pilot["schema_version"] in {DETAIL_PILOT_SCHEMA, DETAIL_PILOT_SCHEMA_V2}:
        validate_detail_sources(pilot, preflight)
    if pilot["schema_version"] == DIFFERENCE_PILOT_SCHEMA:
        validate_difference_sources(pilot, preflight)
    if pilot["schema_version"] == CONTEXT_PILOT_SCHEMA:
        validate_context_sources(pilot, preflight)
    if pilot["schema_version"] == EXPANSION_PILOT_SCHEMA:
        validate_expansion_sources(pilot, preflight)
    return deepcopy(pilot["packets"])


def call_once(packet, create_completion, *, max_tokens):
    """No loop, retry, model fallback, or repair call exists in this function."""
    # Fail before contacting a provider, not after spending a call on an unknown
    # task or an altered packet. Batch authorization is checked by the caller.
    validator = _response_validator(packet)
    validate_json_object_messages(packet["messages"])
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    record = {"packet_id": packet["packet_id"], "task": packet["task"],
              "source_packet_fingerprint": packet["packet_fingerprint"],
              "messages_fingerprint": content_sha256(packet["messages"]),
              "attempt_count": 1, "status": "transport_error", "error_type": None}
    try:
        result = create_completion(
            model="deepseek-v4-flash", messages=deepcopy(packet["messages"]),
            max_tokens=max_tokens, temperature=0.0, response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception as exc:
        # Keep the legacy broad status for readers of older manifests. The
        # diagnostic category distinguishes HTTP rejection from network errors.
        diagnostics = diagnose_provider_error(exc)
        record["error_type"] = diagnostics["exception_type"]
        record["error_diagnostics"] = diagnostics
        return record
    record["status"] = "invalid_response"
    try:
        record["raw_response"] = result.model_dump(mode="json")
        choice = result.choices[0]
        record["finish_reason"] = choice.finish_reason
        record["raw_text"] = choice.message.content or ""
        usage = getattr(result, "usage", None)
        record["usage"] = usage.model_dump(mode="json") if usage is not None else {}
        if choice.finish_reason != "stop":
            record["error_type"] = "NonStopFinishReason"
            return record
        checked = validator(packet, json.loads(record["raw_text"]))
        record["validated_response"] = checked
        record["status"] = "format_valid"
    except Exception as exc:
        record["error_type"] = type(exc).__name__
    return record
