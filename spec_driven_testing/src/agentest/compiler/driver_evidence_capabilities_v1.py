"""Validate explicit Driver evidence capabilities used by Given compilation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .then_atomization import ThenAtomizationError


PROFILE_VERSION = "agentspectesting.driver-evidence-capability-profile/v0.1"
SOURCE_KINDS = frozenset(
    {
        "user_speech_act",
        "user_supplied_fact",
        "fixture_state",
        "derived_state",
        "prior_runtime_event",
        "agent_capability",
        "multiple_sources",
    }
)
NON_SOURCE_OUTCOMES = frozenset({"source_undefined", "insufficient_context"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def validate_driver_evidence_capability_profile(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$driver_evidence_capability_profile")))
    supplied_fingerprint = result.pop("profile_fingerprint", None)
    if result.get("schema_version") != PROFILE_VERSION:
        raise ThenAtomizationError("invalid Driver evidence capability profile version")
    for field in ("profile_id", "domain"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ThenAtomizationError(f"Driver evidence profile {field} is required")
    sources = result.get("evidence_sources")
    if not isinstance(sources, list):
        raise ThenAtomizationError("Driver evidence sources must be an array")
    source_ids = []
    for raw in sources:
        item = _mapping(raw, "$.evidence_sources[]")
        source_ids.append(item.get("source_kind"))
        if not isinstance(item.get("description"), str) or not item[
            "description"
        ].strip():
            raise ThenAtomizationError("Driver evidence source description is required")
        if not isinstance(item.get("supports"), list) or not isinstance(
            item.get("limitations"), list
        ):
            raise ThenAtomizationError("Driver evidence supports/limitations are required")
    if set(source_ids) != SOURCE_KINDS or len(source_ids) != len(SOURCE_KINDS):
        raise ThenAtomizationError("Driver evidence source kinds are incomplete")
    outcomes = result.get("non_source_outcomes")
    if not isinstance(outcomes, list):
        raise ThenAtomizationError("Driver evidence non-source outcomes must be an array")
    outcome_ids = [item.get("outcome") for item in outcomes]
    if set(outcome_ids) != NON_SOURCE_OUTCOMES or len(outcome_ids) != len(
        NON_SOURCE_OUTCOMES
    ):
        raise ThenAtomizationError("Driver evidence non-source outcomes are incomplete")
    computed_fingerprint = content_sha256(result)
    if supplied_fingerprint is not None and supplied_fingerprint != computed_fingerprint:
        raise ThenAtomizationError("Driver evidence capability fingerprint mismatch")
    result["profile_fingerprint"] = computed_fingerprint
    return result
