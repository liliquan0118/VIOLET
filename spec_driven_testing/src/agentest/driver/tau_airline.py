"""Bind an unbound v0.3 plan to deterministic tau-bench airline fixtures."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256
from ..compiler.experiment_planner import (
    ExperimentPlanningError,
    validate_probe_contract,
)
from ..compiler.profiles import (
    load_configuration_file,
    profile_identity,
    validate_run_configuration,
)
from ..fixture_materializer import load_tau_airline_database


BOUND_DRIVER_PLAN_SCHEMA_VERSION = "agentspectesting.bound-driver-plan/v0.1"
DRIVER_BINDER_VERSION = "tau-airline-predicate-binding/v0.1"
FOCUS_CONTROL_BINDER_VERSION = "tau-airline-focus-control-binding/v0.1"

_DURATION_SECONDS = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
}


class DriverBindingError(ValueError):
    """Raised when a v0.3 target cannot be bound without semantic guessing."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DriverBindingError(f"{path} must be an object")
    return value


def _parse_reference_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        raise DriverBindingError(
            "Driver Binding requires an observed policy reference time"
        )
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DriverBindingError(
            f"invalid policy reference time: {value!r}"
        ) from exc


def _canonical_datetime(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _tau_local_datetime(value: datetime) -> str:
    return value.replace(tzinfo=None).isoformat(timespec="seconds")


def _flight_instance(
    database: Mapping[str, Any], flight_number: str, date: str
) -> Mapping[str, Any] | None:
    flight = (database.get("flights") or {}).get(flight_number)
    if not isinstance(flight, Mapping):
        return None
    instance = (flight.get("dates") or {}).get(date)
    return instance if isinstance(instance, Mapping) else None


def _candidate_from_reservation(
    database: Mapping[str, Any],
    reservation_id: str,
    reservation: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    users = database.get("users") or {}
    user_id = reservation.get("user_id")
    user = users.get(user_id) if isinstance(users, Mapping) else None
    if reservation.get("status") in {"cancelled", "refunded", "voided", "expired"}:
        reasons.append("focal operation is unreachable from terminal reservation state")
    if not isinstance(user, Mapping):
        reasons.append("requesting user does not exist")
    elif reservation_id not in (user.get("reservations") or []):
        reasons.append("reservation is not listed under its requesting user")

    raw_segments = reservation.get("flights") or []
    if not isinstance(raw_segments, list) or not raw_segments:
        reasons.append("reservation has no observable flight segments")
    flights: list[dict[str, Any]] = []
    for index, segment in enumerate(raw_segments if isinstance(raw_segments, list) else []):
        if not isinstance(segment, Mapping):
            reasons.append(f"flight segment {index} is malformed")
            continue
        flight_number = segment.get("flight_number")
        date = segment.get("date")
        if not isinstance(flight_number, str) or not isinstance(date, str):
            reasons.append(f"flight segment {index} lacks flight_number/date")
            continue
        instance = _flight_instance(database, flight_number, date)
        if instance is None:
            reasons.append(f"flight instance not found: {flight_number}/{date}")
            continue
        flights.append(
            {
                "flight_number": flight_number,
                "date": date,
                "origin": segment.get("origin"),
                "destination": segment.get("destination"),
                "status": instance.get("status"),
            }
        )
    if reasons:
        return None, reasons

    assert isinstance(user, Mapping)
    name = user.get("name") if isinstance(user.get("name"), Mapping) else {}
    return {
        "reservation_id": reservation_id,
        "user_id": user_id,
        "user": {
            "first_name": name.get("first_name"),
            "last_name": name.get("last_name"),
            "email": user.get("email"),
            "date_of_birth": user.get("dob"),
        },
        "reservation": {
            "original_created_at": reservation.get("created_at"),
            "status": reservation.get("status"),
            "origin": reservation.get("origin"),
            "destination": reservation.get("destination"),
            "flight_type": reservation.get("flight_type"),
            "cabin": reservation.get("cabin"),
            "insurance": reservation.get("insurance"),
        },
        "flights": flights,
    }, []


def _compare(operator: str, left: Any, right: Any) -> bool:
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "in":
        return left in right
    raise DriverBindingError(
        f"tau airline fixture adapter cannot evaluate operator {operator!r}"
    )


def _numeric_compare(operator: str, left: float, right: float) -> bool:
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    raise DriverBindingError(f"unsupported numeric comparison operator: {operator!r}")


def _condition_operand_id(condition: Mapping[str, Any]) -> str:
    identifiers = (condition.get("runtime_binding") or {}).get(
        "condition_operand_ids"
    ) or []
    if len(identifiers) != 1 or not isinstance(identifiers[0], str):
        raise DriverBindingError(
            f"predicate {condition.get('predicate_id')!r} must have one condition operand"
        )
    return identifiers[0]


def _semantic_evaluator(
    condition: Mapping[str, Any], catalog: Mapping[str, Any]
) -> Mapping[str, Any]:
    operand_id = _condition_operand_id(condition)
    matches = [
        binding
        for binding in catalog.get("semantic_predicate_bindings") or []
        if isinstance(binding, Mapping)
        and binding.get("condition_operand_id") == operand_id
    ]
    if len(matches) != 1:
        raise DriverBindingError(
            f"SUT adapter must provide exactly one semantic binding for {operand_id!r}"
        )
    return _mapping(matches[0].get("evaluator"), "$.semantic_binding.evaluator")


def _semantic_binding(
    condition_operand_id: str, catalog: Mapping[str, Any]
) -> Mapping[str, Any]:
    matches = [
        binding
        for binding in catalog.get("semantic_predicate_bindings") or []
        if isinstance(binding, Mapping)
        and binding.get("condition_operand_id") == condition_operand_id
    ]
    if len(matches) != 1:
        raise DriverBindingError(
            f"SUT adapter must provide exactly one semantic binding for {condition_operand_id!r}"
        )
    return matches[0]


def _observation_source_path(
    observation_value_id: str, catalog: Mapping[str, Any]
) -> str:
    matches = [
        binding
        for binding in catalog.get("observation_bindings") or []
        if isinstance(binding, Mapping)
        and binding.get("observation_value_id") == observation_value_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("source_path"), str):
        raise DriverBindingError(
            "SUT adapter must provide exactly one source path for observation "
            f"{observation_value_id!r}"
        )
    return str(matches[0]["source_path"])


def _observation_binding(
    observation_value_id: str, catalog: Mapping[str, Any]
) -> Mapping[str, Any]:
    matches = [
        binding
        for binding in catalog.get("observation_bindings") or []
        if isinstance(binding, Mapping)
        and binding.get("observation_value_id") == observation_value_id
    ]
    if len(matches) != 1:
        raise DriverBindingError(
            "SUT adapter must provide exactly one observation binding for "
            f"{observation_value_id!r}"
        )
    return matches[0]


def _path_value(root: Mapping[str, Any], path: str) -> Any:
    current: Any = root
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise DriverBindingError(f"fixture source path is unavailable: {path!r}")
        current = current[component]
    return current


def _dialogue_fact_for_condition(
    condition: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    evaluator = _semantic_evaluator(condition, catalog)
    if evaluator.get("kind") != "dialogue_value_by_required_truth":
        raise DriverBindingError(
            "dialogue-controlled predicate is not bound to a dialogue evaluator: "
            f"{condition.get('predicate_id')!r}"
        )
    required = condition.get("required_truth_value")
    value = evaluator.get("true_value") if required is True else evaluator.get("false_value")
    if value is None:
        raise DriverBindingError("dialogue evaluator lacks a value for required truth")
    return {
        "fact_name": evaluator.get("fact_name"),
        "value": value,
        "predicate_id": condition.get("predicate_id"),
        "condition_operand_id": _condition_operand_id(condition),
        "evaluated_truth": required,
        "evidence_channel": evaluator.get("evidence_channel"),
    }


def _evaluate_condition(
    condition: Mapping[str, Any],
    candidate: Mapping[str, Any],
    dialogue_facts: Mapping[str, Any],
    catalog: Mapping[str, Any],
    reference_time: datetime | None = None,
) -> tuple[bool, dict[str, Any]]:
    expression = _mapping(condition.get("typed_expression"), "$.condition.typed_expression")
    kind = expression.get("kind")

    if condition.get("control_role") == "dialogue_controlled":
        fact = dialogue_facts.get(str(condition.get("predicate_id")))
        if not isinstance(fact, Mapping):
            raise DriverBindingError(
                f"missing dialogue fact for {condition.get('predicate_id')!r}"
            )
        return bool(fact["evaluated_truth"]), {
            "evaluator": "dialogue_semantic_fact",
            "observed_value": fact["value"],
            "evidence_channel": fact["evidence_channel"],
        }

    if kind == "semantic_evidence_predicate":
        evaluator = _semantic_evaluator(condition, catalog)
        if evaluator.get("kind") == "any_collection_field_equals":
            collection = _path_value(candidate, str(evaluator.get("collection_path")))
            if not isinstance(collection, list):
                raise DriverBindingError("semantic evaluator collection is not an array")
            field = str(evaluator.get("field"))
            observed_values = [
                item.get(field) for item in collection if isinstance(item, Mapping)
            ]
            expected = evaluator.get("value")
            return any(value == expected for value in observed_values), {
                "evaluator": "sut_profile.any_collection_field_equals",
                "condition_operand_id": _condition_operand_id(condition),
                "source_path": f"{evaluator.get('collection_path')}[].{field}",
                "observed_values": observed_values,
                "comparison_value": expected,
            }

    if kind == "comparison":
        left = _mapping(expression.get("left"), "$.condition.typed_expression.left")
        right = _mapping(expression.get("right"), "$.condition.typed_expression.right")
        if left.get("kind") == "observation_value_ref":
            observation_id = left.get("observation_value_id")
            if not isinstance(observation_id, str):
                raise DriverBindingError("comparison observation ref lacks an ID")
            source_path = _observation_source_path(observation_id, catalog)
            observed = _path_value(candidate, source_path)
            expected = right.get("value")
            return _compare(str(expression.get("operator")), observed, expected), {
                "evaluator": "sut_profile.observation_source_path",
                "observation_value_id": observation_id,
                "source_path": source_path,
                "observed_value": observed,
                "comparison_value": expected,
            }
        if left.get("kind") == "function_call" and left.get("function") == "elapsed_time":
            if reference_time is None:
                raise DriverBindingError(
                    "elapsed-time condition requires the observed reference clock"
                )
            raw_booking = candidate["reservation"].get("original_created_at")
            booking_time = _parse_reference_time(raw_booking)
            if reference_time.tzinfo is not None and booking_time.tzinfo is None:
                booking_time = booking_time.replace(tzinfo=reference_time.tzinfo)
            if reference_time.tzinfo is None and booking_time.tzinfo is not None:
                reference_time = reference_time.replace(tzinfo=booking_time.tzinfo)
            observed = (reference_time - booking_time).total_seconds()
            raw_threshold = right.get("value")
            unit = str(right.get("unit"))
            if not isinstance(raw_threshold, (int, float)) or isinstance(raw_threshold, bool):
                raise DriverBindingError("duration comparison threshold must be numeric")
            try:
                threshold = float(raw_threshold) * _DURATION_SECONDS[unit]
            except KeyError as exc:
                raise DriverBindingError(
                    f"unsupported duration comparison unit: {unit!r}"
                ) from exc
            truth = _numeric_compare(
                str(expression.get("operator")), observed, threshold
            )
            return truth, {
                "evaluator": "elapsed_time_from_bound_observations",
                "reference_time": _canonical_datetime(reference_time),
                "booking_time": _canonical_datetime(booking_time),
                "observed_duration_seconds": observed,
                "accepted_operator": expression.get("operator"),
                "threshold_seconds": threshold,
            }

    if kind == "quantified_tool_observation":
        item = _mapping(expression.get("item_predicate"), "$.item_predicate")
        right = _mapping(item.get("right"), "$.item_predicate.right")
        values = right.get("values")
        if item.get("operator") == "in" and isinstance(values, list):
            statuses = [flight.get("status") for flight in candidate["flights"]]
            truth = any(status in values for status in statuses)
            return truth, {
                "evaluator": "tau_airline_quantified_flight_status",
                "quantifier": expression.get("quantifier"),
                "observed_values": statuses,
                "membership_values": deepcopy(values),
            }

    raise DriverBindingError(
        "tau airline adapter has no deterministic evaluator for predicate "
        f"{condition.get('predicate_id')!r}"
    )


def _target_map(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    contrasts = plan.get("contrast_targets") or []
    if len(contrasts) != 1 or not isinstance(contrasts[0], Mapping):
        raise DriverBindingError("current binder requires exactly one compiled contrast")
    return {"primary": plan["target"], "contrast": contrasts[0]}


def _candidate_predicate_audit(
    primary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    catalog: Mapping[str, Any],
    reference_time: datetime,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    focus = primary["focal_configuration"]["focus_predicate_ref"]
    dialogue_facts: dict[str, dict[str, Any]] = {}
    for condition in primary.get("conditions") or []:
        if condition.get("control_role") == "dialogue_controlled":
            fact = _dialogue_fact_for_condition(condition, catalog)
            dialogue_facts[str(condition["predicate_id"])] = fact

    witnesses: list[dict[str, Any]] = []
    failures: list[str] = []
    for condition in primary.get("conditions") or []:
        predicate_id = str(condition.get("predicate_id"))
        if predicate_id == focus:
            continue
        actual, evidence = _evaluate_condition(
            condition, candidate, dialogue_facts, catalog, reference_time
        )
        required = condition.get("required_truth_value")
        satisfied = actual is required
        witnesses.append(
            {
                "predicate_id": predicate_id,
                "required_truth_value": required,
                "evaluated_truth_value": actual,
                "satisfied": satisfied,
                "evidence": evidence,
            }
        )
        if not satisfied:
            failures.append(
                f"{predicate_id}: required {required!r}, observed {actual!r}"
            )
    return witnesses, dialogue_facts, failures


def _score_candidate(candidate: Mapping[str, Any], probes: list[Mapping[str, Any]]) -> tuple[Any, ...]:
    original = str(candidate["reservation"].get("original_created_at"))
    target_times = {
        str(probe["created_at_tau_local"])
        for probe in probes
        if probe.get("created_at_tau_local") is not None
    }
    statuses = [flight.get("status") for flight in candidate["flights"]]
    return (
        original not in target_times,
        any(status != "available" for status in statuses),
        len(candidate["flights"]),
        candidate["reservation_id"],
    )


def _duration_probe_bindings(
    plan: Mapping[str, Any], reference_time: datetime
) -> list[dict[str, Any]]:
    refinements = plan.get("boundary_refinements") or []
    if not refinements:
        raise DriverBindingError(
            "current tau airline binder requires compiled duration boundary probes"
        )
    probes = []
    for refinement in refinements:
        if not isinstance(refinement, Mapping):
            raise DriverBindingError("boundary_refinements must contain objects")
        observed = refinement.get("observed_duration_seconds")
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            raise DriverBindingError(
                "boundary refinement lacks observed_duration_seconds"
            )
        threshold = refinement.get("threshold_seconds")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise DriverBindingError("boundary refinement lacks threshold_seconds")
        operator = str(refinement.get("operator"))
        evaluated_truth = _numeric_compare(operator, float(observed), float(threshold))
        if evaluated_truth is not refinement.get("predicate_truth"):
            raise DriverBindingError(
                f"compiled probe {refinement.get('refinement_id')!r} has inconsistent truth"
            )
        booking_time = reference_time - timedelta(seconds=float(observed))
        probes.append(
            {
                "refinement_id": refinement["refinement_id"],
                "target_name": refinement["target_name"],
                "coverage_cell_id": refinement["target_coverage_cell_id"],
                "focus_predicate_id": refinement["focus_predicate_ref"],
                "reference_time": _canonical_datetime(reference_time),
                "created_at": _canonical_datetime(booking_time),
                "created_at_tau_local": _tau_local_datetime(booking_time),
                "observed_duration_seconds": float(observed),
                "threshold_seconds": float(threshold),
                "operator": operator,
                "predicate_truth": evaluated_truth,
                "expected_operation_decision": refinement[
                    "expected_operation_decision"
                ],
            }
        )
    return probes


def _focus_control_probe_bindings(
    plan: Mapping[str, Any], reference_time: datetime
) -> list[dict[str, Any]]:
    try:
        contract = validate_probe_contract(
            _mapping(plan.get("probe_contract"), "$.probe_contract")
        )
    except ExperimentPlanningError as exc:
        raise DriverBindingError(f"invalid probe contract: {exc}") from exc
    family = contract.get("probe_family")
    if family not in {"semantic_boolean_contrast", "enum_equality_contrast"}:
        raise DriverBindingError(
            f"focus-control binder does not support probe family {family!r}"
        )
    raw_instances = contract.get("probe_instances")
    if not isinstance(raw_instances, list) or len(raw_instances) != 2:
        raise DriverBindingError(
            "focus-control probe contract requires primary and contrast instances"
        )
    probes = []
    for raw in raw_instances:
        item = _mapping(raw, "$.probe_contract.probe_instances[]")
        truth = item.get("required_focus_truth")
        if not isinstance(truth, bool):
            raise DriverBindingError("focus-control probe truth must be boolean")
        probes.append(
            {
                "refinement_id": str(item["probe_id"]),
                "target_name": str(item["target_name"]),
                "coverage_cell_id": str(item["coverage_cell_id"]),
                "focus_predicate_id": str(item["focus_predicate_id"]),
                "predicate_truth": truth,
                "expected_operation_decision": item[
                    "expected_operation_decision"
                ],
                "control_contract": deepcopy(
                    dict(_mapping(item.get("control_contract"), "$.control_contract"))
                ),
                "reference_time": _canonical_datetime(reference_time),
            }
        )
    if {item["target_name"] for item in probes} != {"primary", "contrast"}:
        raise DriverBindingError("focus-control probes must target primary and contrast")
    if {item["predicate_truth"] for item in probes} != {True, False}:
        raise DriverBindingError("focus-control probes must form a truth contrast")
    return probes


def _probe_bindings(
    plan: Mapping[str, Any], reference_time: datetime, binder_capability: str
) -> list[dict[str, Any]]:
    if binder_capability == "binder.tau-airline.cancel-duration/v0.1":
        return _duration_probe_bindings(plan, reference_time)
    if binder_capability == "binder.tau-airline.cancel-focus-control/v0.1":
        return _focus_control_probe_bindings(plan, reference_time)
    raise DriverBindingError(
        f"compiled plan selects unsupported fixture binder {binder_capability!r}"
    )


def _focus_condition(target: Mapping[str, Any]) -> Mapping[str, Any]:
    focus = target["focal_configuration"]["focus_predicate_ref"]
    matches = [
        item
        for item in target.get("conditions") or []
        if isinstance(item, Mapping) and item.get("predicate_id") == focus
    ]
    if len(matches) != 1:
        raise DriverBindingError(
            f"target must contain exactly one focus condition for {focus!r}"
        )
    return matches[0]


def _booking_observations(
    selected: Mapping[str, Any], reference_time: datetime
) -> dict[str, Any]:
    booking = _parse_reference_time(
        selected["reservation"].get("original_created_at")
    )
    comparable_reference = reference_time
    if comparable_reference.tzinfo is not None and booking.tzinfo is None:
        booking = booking.replace(tzinfo=comparable_reference.tzinfo)
    if comparable_reference.tzinfo is None and booking.tzinfo is not None:
        comparable_reference = comparable_reference.replace(tzinfo=booking.tzinfo)
    return {
        "reference_time": _canonical_datetime(reference_time),
        "booking_time": _canonical_datetime(booking),
        "booking_time_tau_local": _tau_local_datetime(booking),
        "booking_age_seconds": (
            comparable_reference - booking
        ).total_seconds(),
    }


def _require_false_focus_base(
    *,
    primary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    dialogue_facts: Mapping[str, Any],
    catalog: Mapping[str, Any],
    reference_time: datetime,
) -> tuple[bool, dict[str, Any]]:
    actual, evidence = _evaluate_condition(
        _focus_condition(primary),
        candidate,
        dialogue_facts,
        catalog,
        reference_time,
    )
    return actual is False, evidence


def _materialize_focus_control(
    *,
    selected: Mapping[str, Any],
    probe: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Return minimal agent patch, focus evidence and materialized base state."""

    desired = bool(probe["predicate_truth"])
    control = _mapping(probe.get("control_contract"), "$.probe.control_contract")
    reservation_id = str(selected["reservation_id"])
    base_state = {
        "reservation": deepcopy(selected["reservation"]),
        "flights": deepcopy(selected["flights"]),
    }
    kind = control.get("kind")

    if kind == "semantic_predicate_truth":
        operand_id = str(control.get("condition_operand_id"))
        binding = _semantic_binding(operand_id, catalog)
        evaluator = _mapping(binding.get("evaluator"), "$.semantic_binding.evaluator")
        materializer = _mapping(
            binding.get("materializer"), "$.semantic_binding.materializer"
        )
        if (
            evaluator.get("kind") != "any_collection_field_equals"
            or materializer.get("kind")
            != "set_first_bound_flight_instance_field"
        ):
            raise DriverBindingError(
                f"semantic focus {operand_id!r} lacks an installed materializer"
            )
        if not selected["flights"]:
            raise DriverBindingError("semantic flight-state focus requires a bound flight")
        flight = selected["flights"][0]
        field = str(materializer.get("field"))
        satisfying_value = evaluator.get("value")
        observed_before = [item.get(field) for item in selected["flights"]]
        if desired:
            patch = {
                "flights": {
                    flight["flight_number"]: {
                        "dates": {
                            flight["date"]: {field: satisfying_value}
                        }
                    }
                }
            }
            base_state["flights"][0][field] = satisfying_value
            observed_after = [item.get(field) for item in base_state["flights"]]
        else:
            if any(value == satisfying_value for value in observed_before):
                raise DriverBindingError(
                    "counterexample probe requires a nonmatching base fixture"
                )
            patch = None
            observed_after = observed_before
        evidence = {
            "evaluator": "sut_profile.any_collection_field_equals",
            "condition_operand_id": operand_id,
            "observed_values_before": observed_before,
            "observed_values_after": observed_after,
            "comparison_value": satisfying_value,
            "materializer_kind": materializer["kind"],
        }
        return patch, evidence, base_state

    if kind == "observation_comparison":
        observation_id = str(control.get("observation_value_id"))
        binding = _observation_binding(observation_id, catalog)
        materializer = _mapping(
            binding.get("materializer"), "$.observation_binding.materializer"
        )
        if materializer.get("kind") != "set_bound_reservation_field":
            raise DriverBindingError(
                f"observation focus {observation_id!r} lacks an installed materializer"
            )
        field = str(materializer.get("field"))
        literal = _mapping(control.get("literal"), "$.control_contract.literal")
        comparison_value = literal.get("value")
        operator = str(control.get("operator"))
        observed_before = selected["reservation"].get(field)
        actual_before = _compare(operator, observed_before, comparison_value)
        if actual_before is desired:
            patch = None
            observed_after = observed_before
        elif operator == "==" and desired:
            observed_after = comparison_value
            patch = {
                "reservations": {
                    reservation_id: {field: observed_after}
                }
            }
            base_state["reservation"][field] = observed_after
        elif operator == "!=" and not desired:
            observed_after = comparison_value
            patch = {
                "reservations": {
                    reservation_id: {field: observed_after}
                }
            }
            base_state["reservation"][field] = observed_after
        else:
            raise DriverBindingError(
                "counterexample materialization requires a nonmatching base fixture"
            )
        if _compare(operator, observed_after, comparison_value) is not desired:
            raise DriverBindingError("materialized observation does not satisfy probe truth")
        evidence = {
            "evaluator": "sut_profile.observation_source_path",
            "observation_value_id": observation_id,
            "source_path": binding["source_path"],
            "observed_value_before": observed_before,
            "observed_value_after": observed_after,
            "operator": operator,
            "comparison_value": comparison_value,
            "materializer_kind": materializer["kind"],
        }
        return patch, evidence, base_state

    raise DriverBindingError(f"unsupported focus control kind: {kind!r}")


def _duration_fixture_instance(
    *,
    plan: Mapping[str, Any],
    target: Mapping[str, Any],
    selected: Mapping[str, Any],
    base_witnesses: list[Mapping[str, Any]],
    dialogue_facts: Mapping[str, Mapping[str, Any]],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    reservation_id = selected["reservation_id"]
    local_created_at = probe["created_at_tau_local"]
    original_created_at = selected["reservation"].get("original_created_at")
    patch_required = original_created_at != local_created_at
    focus_witness = {
        "predicate_id": probe["focus_predicate_id"],
        "required_truth_value": probe["predicate_truth"],
        "evaluated_truth_value": probe["predicate_truth"],
        "satisfied": True,
        "evidence": {
            "evaluator": "elapsed_time_from_bound_observations",
            "reference_time": probe["reference_time"],
            "booking_time": probe["created_at"],
            "observed_duration_seconds": probe["observed_duration_seconds"],
            "accepted_operator": probe["operator"],
            "threshold_seconds": probe["threshold_seconds"],
        },
    }
    witnesses = [focus_witness, *deepcopy(base_witnesses)]
    expected_assignment = {
        item["predicate_id"]: item["expected_truth_value"]
        for item in target["focal_configuration"]["required_factor_values"]
    }
    actual_assignment = {
        item["predicate_id"]: item["evaluated_truth_value"] for item in witnesses
    }
    assignment_matches = expected_assignment == actual_assignment
    if not assignment_matches:
        raise DriverBindingError(
            f"bound fixture does not match target cell {probe['coverage_cell_id']!r}"
        )
    return {
        "fixture_instance_id": (
            f"tau-airline::{reservation_id}::{probe['refinement_id']}"
        ),
        "refinement_id": probe["refinement_id"],
        "target_name": probe["target_name"],
        "coverage_cell_id": probe["coverage_cell_id"],
        "expected_operation_decision": probe["expected_operation_decision"],
        "object_bindings": {
            "user_id": selected["user_id"],
            "reservation_id": reservation_id,
            "user_first_name": selected["user"]["first_name"],
            "user_last_name": selected["user"]["last_name"],
            "user_email": selected["user"]["email"],
            "user_date_of_birth": selected["user"]["date_of_birth"],
        },
        "observation_bindings": {
            "reference_time": probe["reference_time"],
            "booking_time": probe["created_at"],
            "booking_time_tau_local": local_created_at,
            "booking_age_seconds": probe["observed_duration_seconds"],
        },
        "dialogue_fact_bindings": [
            deepcopy(value) for _, value in sorted(dialogue_facts.items())
        ],
        "base_state": {
            "reservation": deepcopy(selected["reservation"]),
            "flights": deepcopy(selected["flights"]),
        },
        "initial_state_patch": {
            "agent_data": (
                {"reservations": {reservation_id: {"created_at": local_created_at}}}
                if patch_required
                else None
            ),
            "user_data": None,
        },
        "predicate_witnesses": witnesses,
        "fixture_oracle_audit": {
            "expected_assignment": expected_assignment,
            "evaluated_assignment": actual_assignment,
            "assignment_matches_target_cell": assignment_matches,
            "exact_target_object_binding": True,
        },
        "pre_state_probes": [
            {
                "tool_name": "get_reservation_details",
                "arguments": {"reservation_id": reservation_id},
            },
            *[
                {
                    "tool_name": "get_flight_status",
                    "arguments": {
                        "flight_number": flight["flight_number"],
                        "date": flight["date"],
                    },
                }
                for flight in selected["flights"]
            ],
        ],
        "oracle_binding": {
            "coverage_cell_id": probe["coverage_cell_id"],
            "correctness_oracle_status": target["generation_contract"].get(
                "correctness_oracle_status"
            ),
            "expected_operation_decision": probe["expected_operation_decision"],
            "verified_target_tool_names": deepcopy(
                target["subject"]["verified_tool_names"]
            ),
        },
    }


def _focus_control_fixture_instance(
    *,
    plan: Mapping[str, Any],
    target: Mapping[str, Any],
    selected: Mapping[str, Any],
    base_witnesses: list[Mapping[str, Any]],
    dialogue_facts: Mapping[str, Mapping[str, Any]],
    probe: Mapping[str, Any],
    catalog: Mapping[str, Any],
    reference_time: datetime,
) -> dict[str, Any]:
    reservation_id = str(selected["reservation_id"])
    patch, focus_evidence, materialized_base = _materialize_focus_control(
        selected=selected,
        probe=probe,
        catalog=catalog,
    )
    focus_witness = {
        "predicate_id": probe["focus_predicate_id"],
        "required_truth_value": probe["predicate_truth"],
        "evaluated_truth_value": probe["predicate_truth"],
        "satisfied": True,
        "evidence": focus_evidence,
    }
    witnesses = [focus_witness, *deepcopy(base_witnesses)]
    expected_assignment = {
        item["predicate_id"]: item["expected_truth_value"]
        for item in target["focal_configuration"]["required_factor_values"]
    }
    actual_assignment = {
        item["predicate_id"]: item["evaluated_truth_value"] for item in witnesses
    }
    assignment_matches = expected_assignment == actual_assignment
    if not assignment_matches:
        raise DriverBindingError(
            f"bound fixture does not match target cell {probe['coverage_cell_id']!r}"
        )
    observations = _booking_observations(selected, reference_time)
    observations["focus_control"] = {
        "kind": probe["control_contract"]["kind"],
        "required_truth_value": probe["predicate_truth"],
        "evidence": deepcopy(focus_evidence),
    }
    return {
        "fixture_instance_id": (
            f"tau-airline::{reservation_id}::{probe['refinement_id']}"
        ),
        "refinement_id": probe["refinement_id"],
        "target_name": probe["target_name"],
        "coverage_cell_id": probe["coverage_cell_id"],
        "expected_operation_decision": probe["expected_operation_decision"],
        "object_bindings": {
            "user_id": selected["user_id"],
            "reservation_id": reservation_id,
            "user_first_name": selected["user"]["first_name"],
            "user_last_name": selected["user"]["last_name"],
            "user_email": selected["user"]["email"],
            "user_date_of_birth": selected["user"]["date_of_birth"],
        },
        "observation_bindings": observations,
        "dialogue_fact_bindings": [
            deepcopy(value) for _, value in sorted(dialogue_facts.items())
        ],
        "base_state": materialized_base,
        "initial_state_patch": {
            "agent_data": patch,
            "user_data": None,
        },
        "predicate_witnesses": witnesses,
        "fixture_oracle_audit": {
            "expected_assignment": expected_assignment,
            "evaluated_assignment": actual_assignment,
            "assignment_matches_target_cell": assignment_matches,
            "exact_target_object_binding": True,
        },
        "pre_state_probes": [
            {
                "tool_name": "get_reservation_details",
                "arguments": {"reservation_id": reservation_id},
            },
            *[
                {
                    "tool_name": "get_flight_status",
                    "arguments": {
                        "flight_number": flight["flight_number"],
                        "date": flight["date"],
                    },
                }
                for flight in selected["flights"]
            ],
        ],
        "oracle_binding": {
            "coverage_cell_id": probe["coverage_cell_id"],
            "correctness_oracle_status": target["generation_contract"].get(
                "correctness_oracle_status"
            ),
            "expected_operation_decision": probe["expected_operation_decision"],
            "verified_target_tool_names": deepcopy(
                target["subject"]["verified_tool_names"]
            ),
        },
    }


def _fixture_instance(
    *,
    plan: Mapping[str, Any],
    target: Mapping[str, Any],
    selected: Mapping[str, Any],
    base_witnesses: list[Mapping[str, Any]],
    dialogue_facts: Mapping[str, Mapping[str, Any]],
    probe: Mapping[str, Any],
    catalog: Mapping[str, Any],
    reference_time: datetime,
) -> dict[str, Any]:
    if probe.get("control_contract") is None:
        return _duration_fixture_instance(
            plan=plan,
            target=target,
            selected=selected,
            base_witnesses=base_witnesses,
            dialogue_facts=dialogue_facts,
            probe=probe,
        )
    return _focus_control_fixture_instance(
        plan=plan,
        target=target,
        selected=selected,
        base_witnesses=base_witnesses,
        dialogue_facts=dialogue_facts,
        probe=probe,
        catalog=catalog,
        reference_time=reference_time,
    )


def _fingerprint_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(plan))
    payload.pop("bound_driver_plan_fingerprint", None)
    source = payload.get("database_source") or {}
    source.pop("database_path", None)
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return _canonical_datetime(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def bind_tau_airline_driver(
    compiled_plan: Mapping[str, Any],
    database: Mapping[str, Any],
    run_configuration: Mapping[str, Any],
    *,
    reference_time: str | datetime,
    database_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one entity and bind every compiled cancellation probe."""

    if compiled_plan.get("schema_version") != "agentspectesting.compiled-test-plan/v0.3":
        raise DriverBindingError(
            "Driver Binding requires agentspectesting.compiled-test-plan/v0.3"
        )
    environment = _mapping(
        compiled_plan.get("environment_requirements"), "$.environment_requirements"
    )
    if environment.get("sut") != "tau_bench" or environment.get("domain") != "airline":
        raise DriverBindingError("this adapter supports only tau_bench airline plans")
    tools = set(compiled_plan["target"]["subject"].get("verified_tool_names") or [])
    if tools != {"cancel_reservation"}:
        raise DriverBindingError(
            "current tau airline binder supports the cancel_reservation operation contract"
        )
    capability_contract = _mapping(
        compiled_plan.get("capability_contract"), "$.capability_contract"
    )
    binder_capability = _mapping(
        capability_contract.get("stage_bindings"),
        "$.capability_contract.stage_bindings",
    ).get("fixture_binder")
    supported_binders = {
        "binder.tau-airline.cancel-duration/v0.1",
        "binder.tau-airline.cancel-focus-control/v0.1",
    }
    if binder_capability not in supported_binders:
        raise DriverBindingError(
            "compiled plan is not bound to an installed tau-airline cancellation binder"
        )
    run = validate_run_configuration(run_configuration)
    clock = _parse_reference_time(reference_time)
    probes = _probe_bindings(compiled_plan, clock, str(binder_capability))
    targets = _target_map(compiled_plan)
    primary = targets["primary"]
    binding_catalog = _mapping(
        environment.get("fixture_binding_catalog"),
        "$.environment_requirements.fixture_binding_catalog",
    )

    reservations = database.get("reservations")
    if not isinstance(reservations, Mapping) or not reservations:
        raise DriverBindingError("airline database has no reservations")
    candidates: list[dict[str, Any]] = []
    rejected_count = 0
    rejection_reasons: dict[str, int] = {}
    max_candidates = run["budgets"]["max_fixture_candidates"]
    for reservation_id, raw in sorted(reservations.items()):
        if not isinstance(raw, Mapping):
            rejected_count += 1
            rejection_reasons["malformed reservation"] = (
                rejection_reasons.get("malformed reservation", 0) + 1
            )
            continue
        candidate, reasons = _candidate_from_reservation(
            database, str(reservation_id), raw
        )
        if candidate is not None:
            witnesses, dialogue_facts, failures = _candidate_predicate_audit(
                primary, candidate, binding_catalog, clock
            )
            if not failures and binder_capability == "binder.tau-airline.cancel-focus-control/v0.1":
                false_base, focus_evidence = _require_false_focus_base(
                    primary=primary,
                    candidate=candidate,
                    dialogue_facts=dialogue_facts,
                    catalog=binding_catalog,
                    reference_time=clock,
                )
                if not false_base:
                    failures = [
                        "focus predicate: base fixture must satisfy the false contrast"
                    ]
                else:
                    candidate["focus_base_evidence"] = focus_evidence
            if failures:
                reasons = failures
            else:
                candidate["base_predicate_witnesses"] = witnesses
                candidate["dialogue_facts"] = dialogue_facts
                candidates.append(candidate)
                if max_candidates and len(candidates) >= max_candidates:
                    break
                continue
        rejected_count += 1
        for reason in set(reasons):
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    if not candidates:
        raise DriverBindingError(
            "no reservation satisfies the compiled non-focus predicate assignment"
        )
    selected = sorted(candidates, key=lambda value: _score_candidate(value, probes))[0]
    fixtures = [
        _fixture_instance(
            plan=compiled_plan,
            target=targets[probe["target_name"]],
            selected=selected,
            base_witnesses=selected["base_predicate_witnesses"],
            dialogue_facts=selected["dialogue_facts"],
            probe=probe,
            catalog=binding_catalog,
            reference_time=clock,
        )
        for probe in probes
    ]
    patch_count = sum(
        fixture["initial_state_patch"]["agent_data"] is not None
        for fixture in fixtures
    )
    if patch_count > run["budgets"]["max_state_patches"]:
        raise DriverBindingError(
            f"binding requires {patch_count} state patches but run configuration allows "
            f"{run['budgets']['max_state_patches']}"
        )
    database_fingerprint = content_sha256(database)
    bound = {
        "schema_version": BOUND_DRIVER_PLAN_SCHEMA_VERSION,
        "driver_binder_version": (
            DRIVER_BINDER_VERSION
            if binder_capability == "binder.tau-airline.cancel-duration/v0.1"
            else FOCUS_CONTROL_BINDER_VERSION
        ),
        "bound_driver_plan_id": f"{compiled_plan['compiled_plan_id']}.tau-airline",
        "compiled_plan_identity": {
            "compiled_plan_id": compiled_plan["compiled_plan_id"],
            "compiled_plan_fingerprint": compiled_plan[
                "compiled_plan_fingerprint"
            ],
        },
        "run_configuration_identity": profile_identity(run),
        "sut_adapter_profile_identity": deepcopy(
            compiled_plan["configuration"]["sut_adapter_profile"]
        ),
        "capability_contract": deepcopy(compiled_plan["capability_contract"]),
        "database_source": {
            **_json_safe(deepcopy(dict(database_metadata or {}))),
            "database_content_fingerprint": database_fingerprint,
        },
        "reference_clock": {
            "source": "observed_environment_policy_clock",
            "value": _canonical_datetime(clock),
        },
        "selection": {
            "candidate_count": len(candidates),
            "rejected_count": rejected_count,
            "ranking_policy": [
                (
                    "prefer an existing booking_time matching any compiled probe"
                    if binder_capability == "binder.tau-airline.cancel-duration/v0.1"
                    else "require a false-focus base shared by the truth contrast"
                ),
                "prefer all flight statuses available",
                "prefer fewer flight segments",
                "deterministic reservation_id tie-break",
            ],
            "selected_reservation_id": selected["reservation_id"],
            "selected_user_id": selected["user_id"],
            "selected_score": list(_score_candidate(selected, probes)),
            "top_rejection_reasons": [
                {"reason": reason, "count": count}
                for reason, count in sorted(
                    rejection_reasons.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ],
        },
        "fixture_instances": fixtures,
        "interaction_plan": deepcopy(compiled_plan["interaction_requirements"]),
        "reachability_contract": deepcopy(compiled_plan["reachability_contract"]),
        "runtime_observation_catalog": deepcopy(
            compiled_plan["environment_requirements"][
                "runtime_observation_catalog"
            ]
        ),
        "variation_contract": deepcopy(compiled_plan["variation_contract"]),
        "guidance_contract": deepcopy(compiled_plan["guidance_contract"]),
        "search_design": {
            "boundary_refinements": deepcopy(
                compiled_plan.get("boundary_refinements") or []
            ),
            "failure_hypotheses": deepcopy(
                (compiled_plan.get("derived_test_design") or {}).get(
                    "failure_hypotheses"
                )
                or []
            ),
            "probe_contract": deepcopy(
                compiled_plan.get("probe_contract") or {}
            ),
        },
        "binding_checks": {
            "compiled_plan_fingerprint_present": True,
            "database_content_fingerprinted": True,
            "reference_clock_observed": True,
            "requesting_identity_bound": True,
            "target_entity_bound": True,
            "all_nonfocus_predicates_evaluated": True,
            "all_fixture_assignments_match_target_cells": all(
                fixture["fixture_oracle_audit"]["assignment_matches_target_cell"]
                for fixture in fixtures
            ),
            "one_base_entity_shared_across_probes": True,
            "only_compiled_probe_observation_is_patched": True,
            "run_budget_respected": True,
            "llm_calls": 0,
        },
    }
    bound["bound_driver_plan_fingerprint"] = content_sha256(
        _fingerprint_payload(bound)
    )
    return bound


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    file = Path(path)
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriverBindingError(f"cannot load {label} {file}: {exc}") from exc
    if not isinstance(value, dict):
        raise DriverBindingError(f"{label} must contain one JSON object")
    return value


def bind_tau_airline_driver_file(
    *,
    compiled_plan_path: str | Path,
    run_configuration_path: str | Path,
    output_path: str | Path,
    database_path: str | Path | None = None,
    reference_time: str | None = None,
) -> dict[str, Any]:
    plan = _load_json_object(compiled_plan_path, "compiled plan")
    run = load_configuration_file(
        run_configuration_path, kind="run_configuration"
    )
    database, metadata = load_tau_airline_database(
        Path(database_path) if database_path is not None else None
    )
    observed_clock = reference_time or metadata.get("policy_clock")
    bound = bind_tau_airline_driver(
        plan,
        database,
        run,
        reference_time=observed_clock,
        database_metadata=metadata,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(bound, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return bound
