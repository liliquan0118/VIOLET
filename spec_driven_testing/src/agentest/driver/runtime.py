"""Deterministic runtime driver and offline trace replay.

This module deliberately does not generate language with a model.  It turns a
BoundDriverPlan into a small milestone state machine, realizes versioned user
messages from bound facts, and consumes normalized runtime observations.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256
from ..compiler.profiles import profile_identity, validate_run_configuration


RUNTIME_DRIVER_VERSION = "deterministic-runtime-driver/v0.1"
RUNTIME_SESSION_SCHEMA_VERSION = "agentspectesting.runtime-driver-session/v0.1"
RUNTIME_TRACE_SCHEMA_VERSION = "agentspectesting.runtime-trace/v0.1"
RUNTIME_REPLAY_SCHEMA_VERSION = "agentspectesting.runtime-replay/v0.1"
REACHABILITY_RESULT_SCHEMA_VERSION = "agentspectesting.reachability-result/v0.1"


class RuntimeDriverError(ValueError):
    """Raised when runtime input is invalid or a replay contradicts the driver."""


_LEGACY_BASELINE_VARIATION = {
    "fact_disclosure_order": "required_identity_first",
    "temporal_surface_form": "exact_observation_values",
    "turn_decomposition": "one_required_fact_per_turn",
    "confirmation_style": "direct",
    "surface_paraphrase": "canonical",
}

_MILESTONE_IDS = (
    "fixture_valid",
    "target_operation_requested",
    "requesting_identity_bound",
    "target_entity_bound",
    "required_user_facts_available",
    "required_agent_evidence_available",
    "required_confirmation_completed",
    "decision_opportunity_reached",
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeDriverError(f"{path} must be an object")
    return value


def _fixture(bound_plan: Mapping[str, Any], fixture_instance_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in bound_plan.get("fixture_instances", [])
        if item.get("fixture_instance_id") == fixture_instance_id
    ]
    if len(matches) != 1:
        raise RuntimeDriverError(
            f"fixture_instance_id must resolve exactly once: {fixture_instance_id!r}"
        )
    fixture = deepcopy(matches[0])
    audit = _mapping(fixture.get("fixture_oracle_audit"), "$.fixture.fixture_oracle_audit")
    if audit.get("assignment_matches_target_cell") is not True:
        raise RuntimeDriverError("runtime driver refuses a fixture that failed oracle audit")
    if audit.get("exact_target_object_binding") is not True:
        raise RuntimeDriverError("runtime driver requires an exact target object binding")
    return fixture


def _variation_profile(
    bound_plan: Mapping[str, Any], selection: Mapping[str, Any] | None
) -> dict[str, Any]:
    contract = _mapping(bound_plan.get("variation_contract"), "$.variation_contract")
    catalog = {
        item["name"]: list(item["allowed_values"])
        for item in contract.get("mutable_dimensions", [])
    }
    configured_baseline = contract.get("baseline_values")
    baseline = dict(
        configured_baseline
        if isinstance(configured_baseline, Mapping)
        else _LEGACY_BASELINE_VARIATION
    )
    unknown_defaults = sorted(set(baseline) - set(catalog))
    if unknown_defaults:
        raise RuntimeDriverError(
            "bound variation contract is missing runtime dimensions: "
            + ", ".join(unknown_defaults)
        )
    requested = dict(selection or {})
    unknown = sorted(set(requested) - set(catalog))
    if unknown:
        raise RuntimeDriverError("unknown variation dimensions: " + ", ".join(unknown))
    missing_defaults = sorted(set(catalog) - set(baseline))
    if missing_defaults:
        raise RuntimeDriverError(
            "bound variation contract has no baseline for dimensions: "
            + ", ".join(missing_defaults)
        )
    for dimension, value in baseline.items():
        if value not in catalog[dimension]:
            raise RuntimeDriverError(
                f"baseline variation {dimension!r} does not allow value {value!r}"
            )
    result = deepcopy(baseline)
    for dimension, value in requested.items():
        if value not in catalog[dimension]:
            raise RuntimeDriverError(
                f"variation {dimension!r} does not allow value {value!r}"
            )
        result[dimension] = value
    changed = sorted(
        dimension
        for dimension, value in result.items()
        if value != baseline[dimension]
    )
    maximum = contract.get("max_changed_dimensions_per_candidate")
    if not isinstance(maximum, int) or isinstance(maximum, bool):
        raise RuntimeDriverError("variation contract has no valid mutation limit")
    if len(changed) > maximum:
        raise RuntimeDriverError(
            f"variation changes {len(changed)} dimensions but contract allows {maximum}"
        )
    return {
        "values": result,
        "baseline_values": deepcopy(baseline),
        "changed_dimensions": changed,
        "semantic_invariant_check": {
            "operation_identity_unchanged": True,
            "target_object_identity_unchanged": True,
            "fixture_state_unchanged": True,
            "passed": True,
        },
    }


def _new_milestone(status: str, evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"status": status, "evidence": list(evidence or [])}


def _evidence_key(tool_name: str, arguments: Mapping[str, Any]) -> str:
    return content_sha256({"tool_name": tool_name, "arguments": dict(arguments)})


def _required_fact_names(fixture: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            str(item["fact_name"])
            for item in fixture.get("dialogue_fact_bindings", [])
            if item.get("fact_name")
        }
    )


def initialize_runtime_driver(
    bound_plan: Mapping[str, Any],
    run_configuration: Mapping[str, Any],
    *,
    fixture_instance_id: str,
    variation_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a serializable deterministic runtime session."""

    if bound_plan.get("schema_version") != "agentspectesting.bound-driver-plan/v0.1":
        raise RuntimeDriverError("runtime driver requires BoundDriverPlan v0.1")
    capabilities = bound_plan.get("capability_contract")
    if capabilities is not None:
        capabilities = _mapping(capabilities, "$.capability_contract")
        runtime_capability = _mapping(
            capabilities.get("stage_bindings"),
            "$.capability_contract.stage_bindings",
        ).get("runtime_protocol")
        if runtime_capability != "runtime.tau-airline.cancel/v0.1":
            raise RuntimeDriverError(
                "bound plan is not assigned to the installed cancellation runtime protocol"
            )
    run = validate_run_configuration(run_configuration)
    expected_run_identity = _mapping(
        bound_plan.get("run_configuration_identity"),
        "$.run_configuration_identity",
    )
    if profile_identity(run) != expected_run_identity:
        raise RuntimeDriverError(
            "run configuration identity does not match the configuration used for binding"
        )
    fixture = _fixture(bound_plan, fixture_instance_id)
    variation = _variation_profile(bound_plan, variation_selection)
    required_facts = _required_fact_names(fixture)
    required_probe_keys = [
        _evidence_key(item["tool_name"], _mapping(item.get("arguments"), "$.probe.arguments"))
        for item in fixture.get("pre_state_probes", [])
    ]
    milestones = {identifier: _new_milestone("pending") for identifier in _MILESTONE_IDS}
    milestones["fixture_valid"] = _new_milestone(
        "satisfied",
        [{"kind": "fixture_oracle_audit", "fixture_instance_id": fixture_instance_id}],
    )
    if not required_facts:
        milestones["required_user_facts_available"] = _new_milestone(
            "satisfied", [{"kind": "no_required_dialogue_facts"}]
        )
    if not required_probe_keys:
        milestones["required_agent_evidence_available"] = _new_milestone(
            "satisfied", [{"kind": "no_required_pre_state_probes"}]
        )
    milestones["required_confirmation_completed"] = _new_milestone(
        "not_applicable", [{"kind": "confirmation_not_requested"}]
    )
    session = {
        "schema_version": RUNTIME_SESSION_SCHEMA_VERSION,
        "runtime_driver_version": RUNTIME_DRIVER_VERSION,
        "bound_driver_plan_identity": {
            "bound_driver_plan_id": bound_plan.get("bound_driver_plan_id"),
            "bound_driver_plan_fingerprint": bound_plan.get(
                "bound_driver_plan_fingerprint"
            ),
        },
        "fixture_instance_id": fixture_instance_id,
        "runtime_context": {
            "object_bindings": deepcopy(fixture["object_bindings"]),
            "observation_bindings": deepcopy(fixture["observation_bindings"]),
            "dialogue_fact_bindings": deepcopy(
                fixture.get("dialogue_fact_bindings", [])
            ),
            "required_fact_names": required_facts,
            "pre_state_probes": deepcopy(fixture.get("pre_state_probes", [])),
            "required_probe_keys": required_probe_keys,
            "verified_target_tool_names": deepcopy(
                fixture["oracle_binding"]["verified_target_tool_names"]
            ),
            "runtime_observation_catalog": deepcopy(
                bound_plan["runtime_observation_catalog"]
            ),
            "max_dialogue_turns": run["budgets"]["max_dialogue_turns"],
            "max_recovery_actions": run["budgets"]["max_recovery_actions"],
        },
        "variation_profile": variation,
        "state": {
            "status": "active",
            "stop_reason": None,
            "turn_count": 0,
            "awaiting_agent": False,
            "milestones": milestones,
            "disclosed_facts": [],
            "requested_facts": [],
            "successful_probe_keys": [],
            "target_action_observed": False,
            "target_action_evidence": None,
            "explicit_decision_evidence": None,
            "observation_log": [],
            "action_log": [],
        },
        "next_action": None,
        "runtime_checks": {
            "fixture_audit_passed": True,
            "run_configuration_identity_matched": True,
            "variation_semantic_invariants_passed": True,
            "llm_calls": 0,
        },
    }
    _recompute(session)
    return session


def _mark(
    session: dict[str, Any], milestone_id: str, evidence: Mapping[str, Any]
) -> None:
    milestone = session["state"]["milestones"][milestone_id]
    milestone["status"] = "satisfied"
    item = deepcopy(dict(evidence))
    if item not in milestone["evidence"]:
        milestone["evidence"].append(item)


def _all_reachability_prerequisites_satisfied(session: Mapping[str, Any]) -> bool:
    milestones = session["state"]["milestones"]
    required = (
        "fixture_valid",
        "target_operation_requested",
        "requesting_identity_bound",
        "target_entity_bound",
        "required_user_facts_available",
        "required_agent_evidence_available",
    )
    if any(milestones[item]["status"] != "satisfied" for item in required):
        return False
    return milestones["required_confirmation_completed"]["status"] in {
        "satisfied",
        "not_applicable",
    }


def _action(kind: str, fact_names: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "agentspectesting.driver-action/v0.1",
        "action_kind": kind,
        "fact_names": list(fact_names or []),
    }


def _ordered_fact_actions(session: Mapping[str, Any]) -> list[tuple[str, str]]:
    order = session["variation_profile"]["values"]["fact_disclosure_order"]
    base = {
        "required_identity_first": [
            ("user_id", "provide_user_id"),
            ("reservation_id", "provide_reservation_id"),
            ("operation_reason", "provide_operation_reason"),
        ],
        "target_entity_first": [
            ("reservation_id", "provide_reservation_id"),
            ("user_id", "provide_user_id"),
            ("operation_reason", "provide_operation_reason"),
        ],
        "operation_reason_first": [
            ("operation_reason", "provide_operation_reason"),
            ("user_id", "provide_user_id"),
            ("reservation_id", "provide_reservation_id"),
        ],
    }
    if order == "requested_fact_only_when_asked":
        requested = set(session["state"]["requested_facts"])
        canonical = base["required_identity_first"]
        return [item for item in canonical if item[0] in requested]
    return base[order]


def _plan_next_action(session: Mapping[str, Any]) -> dict[str, Any]:
    state = session["state"]
    if state["status"] == "terminal":
        return _action("stop")
    if state["awaiting_agent"]:
        return _action("wait_for_agent")
    confirmation = state["milestones"]["required_confirmation_completed"]
    if confirmation["status"] == "pending":
        return _action("confirm_operation")

    decomposition = session["variation_profile"]["values"]["turn_decomposition"]
    disclosed = set(state["disclosed_facts"])
    missing_pairs = [
        pair for pair in _ordered_fact_actions(session) if pair[0] not in disclosed
    ]
    operation_missing = (
        state["milestones"]["target_operation_requested"]["status"] != "satisfied"
    )
    if decomposition == "single_turn" and operation_missing:
        return _action(
            "provide_initial_request_bundle",
            ["target_operation", *[name for name, _ in missing_pairs]],
        )
    if decomposition == "two_turn_required_fact_split":
        if operation_missing:
            first_fact = missing_pairs[0][0] if missing_pairs else None
            facts = ["target_operation"] + ([first_fact] if first_fact else [])
            return _action("provide_request_and_first_fact", facts)
        if missing_pairs:
            return _action(
                "provide_remaining_facts", [name for name, _ in missing_pairs]
            )
        return _action("wait_for_agent")
    if operation_missing:
        return _action("request_target_operation", ["target_operation"])
    if missing_pairs:
        fact_name, kind = missing_pairs[0]
        return _action(kind, [fact_name])
    return _action("wait_for_agent")


def _recompute(session: dict[str, Any]) -> None:
    context = session["runtime_context"]
    state = session["state"]
    disclosed = set(state["disclosed_facts"])
    if "user_id" in disclosed:
        _mark(session, "requesting_identity_bound", {"kind": "user_fact", "fact": "user_id"})
    if "reservation_id" in disclosed:
        _mark(session, "target_entity_bound", {"kind": "user_fact", "fact": "reservation_id"})
    required_facts = set(context["required_fact_names"])
    if required_facts.issubset(disclosed):
        _mark(
            session,
            "required_user_facts_available",
            {"kind": "user_facts", "facts": sorted(required_facts)},
        )
    if set(context["required_probe_keys"]).issubset(state["successful_probe_keys"]):
        _mark(
            session,
            "required_agent_evidence_available",
            {"kind": "successful_pre_state_probes", "count": len(context["required_probe_keys"])},
        )
    if _all_reachability_prerequisites_satisfied(session):
        _mark(
            session,
            "decision_opportunity_reached",
            {"kind": "milestone_conjunction"},
        )
    elif not state["target_action_observed"]:
        state["milestones"]["decision_opportunity_reached"] = _new_milestone(
            "pending"
        )
    session["next_action"] = _plan_next_action(session)


def _dialogue_fact_value(session: Mapping[str, Any], fact_name: str) -> Any:
    values = [
        item.get("value")
        for item in session["runtime_context"]["dialogue_fact_bindings"]
        if item.get("fact_name") == fact_name
    ]
    if len(values) != 1:
        raise RuntimeDriverError(f"dialogue fact {fact_name!r} is not bound exactly once")
    return values[0]


def _temporal_clause(session: Mapping[str, Any]) -> str:
    form = session["variation_profile"]["values"].get(
        "temporal_surface_form", "exact_observation_values"
    )
    observations = session["runtime_context"]["observation_bindings"]
    if form == "exact_observation_values":
        return f"It was booked at {observations['booking_time']}."
    age = int(observations["booking_age_seconds"])
    if form == "relative_value_then_exact_on_request":
        hours, remainder = divmod(age, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"It was booked {hours} hours, {minutes} minutes, and {seconds} seconds ago."
    if form == "coarse_value_then_exact_on_request":
        return "It was booked about a day ago."
    raise RuntimeDriverError(f"unsupported temporal surface form: {form!r}")


def _fact_sentence(session: Mapping[str, Any], fact_name: str) -> str:
    objects = session["runtime_context"]["object_bindings"]
    style = session["variation_profile"]["values"]["surface_paraphrase"]
    if fact_name == "target_operation":
        templates = {
            "canonical": "I want to cancel my reservation.",
            "concise": "Please cancel my reservation.",
            "natural_paraphrase": "Could you help me cancel this booking?",
        }
        return templates[style] + " " + _temporal_clause(session)
    if fact_name == "user_id":
        templates = {
            "canonical": f"My user ID is {objects['user_id']}.",
            "concise": f"User ID: {objects['user_id']}.",
            "natural_paraphrase": f"You can find my account under {objects['user_id']}.",
        }
        return templates[style]
    if fact_name == "reservation_id":
        templates = {
            "canonical": f"My reservation ID is {objects['reservation_id']}.",
            "concise": f"Reservation: {objects['reservation_id']}.",
            "natural_paraphrase": f"The booking reference is {objects['reservation_id']}.",
        }
        return templates[style]
    if fact_name == "operation_reason":
        value = str(_dialogue_fact_value(session, fact_name))
        reason = {
            "health": "health reasons",
            "change_of_plan": "a change of plans",
        }.get(value, value.replace("_", " "))
        templates = {
            "canonical": f"The reason for cancellation is {reason}.",
            "concise": f"Reason: {reason}.",
            "natural_paraphrase": f"I need to cancel because of {reason}.",
        }
        return templates[style]
    raise RuntimeDriverError(f"no deterministic surface binding for fact {fact_name!r}")


def realize_driver_action(
    session: Mapping[str, Any], action: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Turn a planned DriverAction into one deterministic user message."""

    chosen = deepcopy(dict(action or session["next_action"]))
    kind = chosen.get("action_kind")
    if kind in {"wait_for_agent", "stop"}:
        return {
            "schema_version": "agentspectesting.realized-driver-action/v0.1",
            "action": chosen,
            "controllable_channel": None,
            "message": None,
        }
    if kind == "confirm_operation":
        style = session["variation_profile"]["values"]["confirmation_style"]
        message = {
            "direct": "Yes, I confirm the cancellation. Please proceed.",
            "brief": "Yes, proceed.",
            "consequence_restatement_before_confirmation": (
                "I understand that the reservation will be cancelled. Yes, please proceed."
            ),
        }[style]
    else:
        fact_names = chosen.get("fact_names") or []
        message = " ".join(_fact_sentence(session, name) for name in fact_names)
    return {
        "schema_version": "agentspectesting.realized-driver-action/v0.1",
        "action": chosen,
        "controllable_channel": "user_message",
        "message": {"role": "user", "content": message},
    }


def apply_next_driver_action(session: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the currently planned user action and advance disclosure milestones."""

    result = deepcopy(dict(session))
    action = result["next_action"]
    kind = action["action_kind"]
    if kind in {"wait_for_agent", "stop"}:
        raise RuntimeDriverError(f"cannot apply non-user action {kind!r}")
    realized = realize_driver_action(result, action)
    state = result["state"]
    for fact_name in action.get("fact_names", []):
        if fact_name == "target_operation":
            _mark(
                result,
                "target_operation_requested",
                {"kind": "driver_action", "action_kind": kind},
            )
        elif fact_name not in state["disclosed_facts"]:
            state["disclosed_facts"].append(fact_name)
    if kind == "confirm_operation":
        _mark(
            result,
            "required_confirmation_completed",
            {"kind": "driver_confirmation", "action_kind": kind},
        )
    state["action_log"].append(
        {
            "action_index": len(state["action_log"]),
            "action": deepcopy(action),
            "realized": realized,
        }
    )
    state["awaiting_agent"] = True
    _recompute(result)
    return result


def _arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeDriverError(f"tool arguments are not valid JSON: {exc}") from exc
    return deepcopy(dict(_mapping(value, "$.observation.arguments")))


def _contains_any(text: str, phrases: list[str]) -> bool:
    normalized = " ".join(text.casefold().split())
    return any(" ".join(phrase.casefold().split()) in normalized for phrase in phrases)


def _is_confirmation_request(text: str, configured_phrases: list[str]) -> bool:
    """Recognize configured and compositional operation-confirmation requests."""

    normalized = " ".join(text.casefold().split())
    if normalized.startswith(("i confirm ", "we confirm ")):
        return False
    if _contains_any(text, configured_phrases):
        return True
    has_confirmation_term = "confirm" in normalized or "confirmation" in normalized
    has_operation_term = "cancel" in normalized or "proceed" in normalized
    has_request_form = "?" in text and any(
        marker in normalized
        for marker in ("can you", "could you", "would you", "please")
    )
    return has_confirmation_term and has_operation_term and has_request_form


def _terminate(session: dict[str, Any], reason: str) -> None:
    session["state"]["status"] = "terminal"
    session["state"]["stop_reason"] = reason


def ingest_runtime_observation(
    session: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Consume one normalized observation and update the milestone state."""

    result = deepcopy(dict(session))
    state = result["state"]
    if state["status"] == "terminal":
        raise RuntimeDriverError("cannot ingest observations after terminal state")
    event = deepcopy(dict(_mapping(observation, "$.observation")))
    kind = event.get("event_kind")
    if kind not in {
        "assistant_message",
        "assistant_tool_call",
        "tool_result",
        "assistant_turn_completed",
        "terminal_state",
    }:
        raise RuntimeDriverError(f"unsupported runtime observation kind: {kind!r}")
    state["observation_log"].append(
        {"observation_index": len(state["observation_log"]), **event}
    )
    context = result["runtime_context"]
    if kind == "assistant_message":
        content = event.get("content")
        if not isinstance(content, str):
            raise RuntimeDriverError("assistant_message.content must be a string")
        catalog = context["runtime_observation_catalog"]
        for fact_name, phrases in catalog["fact_request_phrases"].items():
            if _contains_any(content, phrases) and fact_name not in state["requested_facts"]:
                state["requested_facts"].append(fact_name)
        if _is_confirmation_request(
            content, catalog["confirmation_request_phrases"]
        ):
            milestone = state["milestones"]["required_confirmation_completed"]
            if milestone["status"] != "satisfied":
                milestone["status"] = "pending"
                milestone["evidence"] = [
                    {"kind": "assistant_confirmation_request", "content": content}
                ]
        if _contains_any(content, catalog["refusal_phrases"]):
            reached = _all_reachability_prerequisites_satisfied(result)
            state["explicit_decision_evidence"] = {
                "decision": "refuse",
                "content": content,
                "after_decision_opportunity": reached,
            }
            _terminate(
                result,
                "explicit_final_decision_observed" if reached else "agent_refused_before_target",
            )
        elif _contains_any(content, catalog["transfer_phrases"]):
            _terminate(result, "transferred_before_target")
    elif kind == "assistant_tool_call":
        tool_name = event.get("tool_name")
        arguments = _arguments(event.get("arguments"))
        event["arguments"] = arguments
        if tool_name in context["verified_target_tool_names"]:
            expected_id = context["object_bindings"]["reservation_id"]
            if arguments.get("reservation_id") == expected_id:
                state["target_action_observed"] = True
                state["target_action_evidence"] = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "exact_target_object": True,
                }
                _terminate(result, "target_action_observed")
            else:
                state["target_action_evidence"] = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "exact_target_object": False,
                    "expected_reservation_id": expected_id,
                }
                _terminate(result, "focal_action_object_mismatch")
    elif kind == "tool_result":
        tool_name = event.get("tool_name")
        arguments = _arguments(event.get("arguments"))
        success = event.get("success") is True
        if success:
            key = _evidence_key(str(tool_name), arguments)
            if key in context["required_probe_keys"] and key not in state["successful_probe_keys"]:
                state["successful_probe_keys"].append(key)
    elif kind == "assistant_turn_completed":
        state["turn_count"] += 1
        state["awaiting_agent"] = False
        if state["turn_count"] >= context["max_dialogue_turns"]:
            _terminate(result, "max_turns_exceeded")
    elif kind == "terminal_state":
        reason = event.get("reason")
        if not isinstance(reason, str) or not reason:
            raise RuntimeDriverError("terminal_state.reason must be a non-empty string")
        _terminate(result, reason)
    _recompute(result)
    return result


def reachability_result(session: Mapping[str, Any]) -> dict[str, Any]:
    """Project a runtime session into the reachability result consumed by the oracle."""

    state = session["state"]
    milestones = state["milestones"]
    unresolved = [
        identifier
        for identifier in _MILESTONE_IDS
        if milestones[identifier]["status"] == "pending"
    ]
    decision_reached = milestones["decision_opportunity_reached"]["status"] == "satisfied"
    reached = decision_reached or state["target_action_observed"]
    return {
        "schema_version": REACHABILITY_RESULT_SCHEMA_VERSION,
        "fixture_instance_id": session["fixture_instance_id"],
        "reached": reached,
        "decision_opportunity_reached": decision_reached,
        "target_action_observed": state["target_action_observed"],
        "stop_reason": state["stop_reason"],
        "earliest_unresolved_milestone": unresolved[0] if unresolved else None,
        "milestones": deepcopy(milestones),
        "target_action_evidence": deepcopy(state["target_action_evidence"]),
        "explicit_decision_evidence": deepcopy(state["explicit_decision_evidence"]),
        "turn_count": state["turn_count"],
        "llm_calls": 0,
    }


def replay_runtime_trace(
    bound_plan: Mapping[str, Any],
    run_configuration: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay an explicit trace without contacting the target agent or any LLM."""

    value = _mapping(trace, "$.trace")
    if value.get("schema_version") != RUNTIME_TRACE_SCHEMA_VERSION:
        raise RuntimeDriverError("unsupported runtime trace schema")
    expected = value.get("bound_driver_plan_fingerprint")
    actual = bound_plan.get("bound_driver_plan_fingerprint")
    if expected != actual:
        raise RuntimeDriverError("trace and bound plan fingerprints do not match")
    session = initialize_runtime_driver(
        bound_plan,
        run_configuration,
        fixture_instance_id=str(value.get("fixture_instance_id")),
        variation_selection=_mapping(
            value.get("variation_selection", {}), "$.trace.variation_selection"
        ),
    )
    events = value.get("events")
    if not isinstance(events, list):
        raise RuntimeDriverError("$.trace.events must be an array")
    for index, raw in enumerate(events):
        event = _mapping(raw, f"$.trace.events[{index}]")
        if event.get("event_kind") == "apply_next_driver_action":
            expected_kind = event.get("expected_action_kind")
            actual_kind = session["next_action"]["action_kind"]
            if expected_kind is not None and expected_kind != actual_kind:
                raise RuntimeDriverError(
                    f"trace event {index} expects action {expected_kind!r}, "
                    f"but driver planned {actual_kind!r}"
                )
            session = apply_next_driver_action(session)
        else:
            session = ingest_runtime_observation(session, event)
    replay = {
        "schema_version": RUNTIME_REPLAY_SCHEMA_VERSION,
        "runtime_driver_version": RUNTIME_DRIVER_VERSION,
        "trace_id": value.get("trace_id"),
        "session": session,
        "reachability_result": reachability_result(session),
        "replay_checks": {
            "bound_plan_fingerprint_matched": True,
            "planned_actions_matched_trace": True,
            "semantic_invariants_passed": session["variation_profile"][
                "semantic_invariant_check"
            ]["passed"],
            "llm_calls": 0,
        },
    }
    replay["replay_fingerprint"] = content_sha256(replay)
    return replay


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    file = Path(path)
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeDriverError(f"cannot load {label} {file}: {exc}") from exc
    return deepcopy(dict(_mapping(value, f"$.{label}")))


def replay_runtime_trace_file(
    *,
    bound_plan_path: str | Path,
    run_configuration_path: str | Path,
    trace_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    bound = _load_json(bound_plan_path, "bound_plan")
    run = _load_json(run_configuration_path, "run_configuration")
    trace = _load_json(trace_path, "trace")
    replay = replay_runtime_trace(bound, run, trace)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return replay
