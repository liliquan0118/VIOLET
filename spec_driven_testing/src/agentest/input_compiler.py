"""Validate a spec-test input and compile it into a deterministic generation plan.

This module is deliberately independent from both an LLM and tau-bench.  It is the
boundary between a human/audited test intent and later fixture/dialogue generation.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


class InputContractError(ValueError):
    """Raised when a test-generation input is structurally or semantically invalid."""


_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)
_CLOCK_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2}:\d{2})(?:\s*(?P<zone>[A-Za-z]+)|(?P<offset>[+-]\d{2}:\d{2}))?$"
)
_FIXED_ZONE_OFFSETS = {
    "UTC": timezone.utc,
    "GMT": timezone.utc,
    "EST": timezone(timedelta(hours=-5), name="EST"),
    "EDT": timezone(timedelta(hours=-4), name="EDT"),
}

_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "input_id",
    "domain",
    "source_spec",
    "decision_contract",
    "test_objective",
    "fixture_contract",
    "dialogue_generation_contract",
    "search_guidance",
    "oracle_contract",
    "output_contract",
}


def _require(mapping: Mapping[str, Any], keys: set[str], path: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise InputContractError(f"{path}: missing required fields: {', '.join(missing)}")


def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputContractError(f"{path}: expected object")
    return value


def _as_nonempty_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise InputContractError(f"{path}: expected a non-empty array")
    return value


def parse_iso_duration(value: str) -> timedelta:
    """Parse the day/time subset of ISO-8601 durations used by test inputs."""

    match = _DURATION_RE.fullmatch(value)
    if not match or not any(match.groupdict().values()):
        raise InputContractError(f"invalid ISO-8601 duration: {value!r}")
    parts = {name: int(raw or 0) for name, raw in match.groupdict().items()}
    duration = timedelta(**parts)
    if duration <= timedelta(0):
        raise InputContractError(f"duration must be positive: {value!r}")
    return duration


def parse_reference_clock(value: str) -> datetime:
    """Parse the explicit policy clock without consulting the runtime wall clock."""

    match = _CLOCK_RE.fullmatch(value.strip())
    if not match:
        raise InputContractError(f"unsupported reference clock: {value!r}")
    zone_name = match.group("zone")
    offset = match.group("offset")
    if zone_name:
        tz = _FIXED_ZONE_OFFSETS.get(zone_name.upper())
        if tz is None:
            raise InputContractError(
                f"unsupported timezone abbreviation {zone_name!r}; use UTC, GMT, EST, EDT, or an explicit offset"
            )
    elif offset:
        sign = 1 if offset[0] == "+" else -1
        hours, minutes = map(int, offset[1:].split(":"))
        tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
    else:
        raise InputContractError("reference clock must declare a timezone")
    return datetime.fromisoformat(f"{match.group('date')}T{match.group('time')}").replace(tzinfo=tz)


def _iso_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _evaluate_strict_threshold(age: timedelta, threshold: timedelta) -> bool:
    return age < threshold


def _normalize_expected_outcome(value: str) -> str:
    if value == "allow_cancel":
        return "allow_cancel"
    if value == "refuse_cancel":
        return "refuse_cancel"
    if value == "refuse_cancel_under_strict_less_than_assumption":
        return "refuse_cancel"
    raise InputContractError(f"unsupported boundary expected_outcome: {value!r}")


def _constraint_index(constraints: list[Any], path: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(constraints):
        item = _as_mapping(raw, f"{path}[{index}]")
        _require(item, {"field", "operator", "value", "reason", "observable_via"}, f"{path}[{index}]")
        field = item["field"]
        if not isinstance(field, str) or not field:
            raise InputContractError(f"{path}[{index}].field: expected non-empty string")
        if field in indexed:
            raise InputContractError(f"{path}: duplicate field constraint {field!r}")
        indexed[field] = item
    return indexed


def _validate_branch_isolation(
    decision: Mapping[str, Any], isolation: dict[str, Mapping[str, Any]]
) -> None:
    alternatives = set(decision.get("alternative_allow_conditions", []))
    if alternatives != {"any_flight_cancelled", "cabin_is_business", "insurance_covers_reason"}:
        raise InputContractError(
            "decision_contract.alternative_allow_conditions: the current cancellation compiler expects "
            "any_flight_cancelled, cabin_is_business, and insurance_covers_reason"
        )

    expected_isolation = {
        "itinerary.any_flight_flown": ("==", False),
        "itinerary.any_flight_cancelled": ("==", False),
        "reservation.insurance": ("==", "no"),
    }
    for field, expected in expected_isolation.items():
        constraint = isolation.get(field)
        if constraint is None:
            raise InputContractError(f"fixture_contract.isolation_constraints: missing {field!r}")
        actual = (constraint["operator"], constraint["value"])
        if actual != expected:
            raise InputContractError(
                f"fixture_contract.isolation_constraints[{field!r}]: expected {expected!r}, got {actual!r}"
            )

    cabin = isolation.get("reservation.cabin")
    if cabin is None or cabin["operator"] != "in":
        raise InputContractError("fixture_contract.isolation_constraints: reservation.cabin must use 'in'")
    cabin_values = set(cabin["value"]) if isinstance(cabin["value"], list) else set()
    if "business" in cabin_values or not cabin_values:
        raise InputContractError("reservation.cabin isolation must contain only non-business cabin values")


def _compile_variants(
    fixture: Mapping[str, Any], predicate: Mapping[str, Any]
) -> tuple[datetime, timedelta, list[dict[str, Any]]]:
    clock = _as_mapping(fixture["reference_clock"], "$.fixture_contract.reference_clock")
    _require(clock, {"source", "value", "timezone_handling"}, "$.fixture_contract.reference_clock")
    reference_time = parse_reference_clock(str(clock["value"]))
    threshold = parse_iso_duration(str(predicate["threshold"]))

    raw_variants = _as_nonempty_list(
        fixture["boundary_variants"], "$.fixture_contract.boundary_variants"
    )
    compiled: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    primary_count = 0
    for index, raw in enumerate(raw_variants):
        path = f"$.fixture_contract.boundary_variants[{index}]"
        variant = _as_mapping(raw, path)
        _require(variant, {"variant_id", "age", "relation", "expected_outcome", "role"}, path)
        variant_id = str(variant["variant_id"])
        if variant_id in seen_ids:
            raise InputContractError(f"{path}.variant_id: duplicate {variant_id!r}")
        seen_ids.add(variant_id)
        age = parse_iso_duration(str(variant["age"]))
        predicate_truth = _evaluate_strict_threshold(age, threshold)
        computed_outcome = "allow_cancel" if predicate_truth else "refuse_cancel"
        declared_outcome = _normalize_expected_outcome(str(variant["expected_outcome"]))
        if declared_outcome != computed_outcome:
            raise InputContractError(
                f"{path}.expected_outcome: declared {declared_outcome!r}, but strict threshold computes {computed_outcome!r}"
            )
        if variant["role"] == "primary":
            primary_count += 1
        created_at = reference_time - age
        compiled.append(
            {
                "variant_id": variant_id,
                "role": variant["role"],
                "reference_time": _iso_timestamp(reference_time),
                "created_at": _iso_timestamp(created_at),
                "booking_age_iso8601": variant["age"],
                "booking_age_seconds": int(age.total_seconds()),
                "threshold_iso8601": predicate["threshold"],
                "threshold_seconds": int(threshold.total_seconds()),
                "predicate_truth": predicate_truth,
                "expected_policy_outcome": computed_outcome,
                "declared_relation": variant["relation"],
            }
        )
    if primary_count != 1:
        raise InputContractError(
            f"fixture_contract.boundary_variants: expected exactly one primary variant, found {primary_count}"
        )
    return reference_time, threshold, compiled


def _validate_primary_created_at(
    required: dict[str, Mapping[str, Any]], variants: list[dict[str, Any]]
) -> None:
    constraint = required.get("reservation.created_at")
    if constraint is None:
        raise InputContractError("fixture_contract.required_state: missing reservation.created_at")
    if constraint["operator"] != "==":
        raise InputContractError("reservation.created_at must use equality")
    primary = next(item for item in variants if item["role"] == "primary")
    raw = str(constraint["value"])
    try:
        declared = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise InputContractError(f"reservation.created_at is not ISO-8601: {raw!r}") from exc
    expected = datetime.fromisoformat(primary["created_at"])
    if declared.tzinfo is None:
        declared = declared.replace(tzinfo=expected.tzinfo)
    if declared != expected:
        raise InputContractError(
            "fixture_contract.required_state reservation.created_at does not match the primary boundary variant: "
            f"declared {_iso_timestamp(declared)}, computed {_iso_timestamp(expected)}"
        )


def compile_test_input(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and compile one input document into a deterministic generation plan."""

    root = _as_mapping(document, "$")
    _require(root, _REQUIRED_TOP_LEVEL, "$")
    if root["schema_version"] != "agent-spec-test-input/v0.1":
        raise InputContractError(f"unsupported schema_version: {root['schema_version']!r}")

    source = _as_mapping(root["source_spec"], "$.source_spec")
    _require(
        source,
        {"workbook", "sheet", "row", "spec_id", "branch_id", "kind", "origin", "deontic", "rule_text", "given", "when", "then", "evidence_quote"},
        "$.source_spec",
    )
    objective = _as_mapping(root["test_objective"], "$.test_objective")
    decision = _as_mapping(root["decision_contract"], "$.decision_contract")
    fixture = _as_mapping(root["fixture_contract"], "$.fixture_contract")
    dialogue = _as_mapping(root["dialogue_generation_contract"], "$.dialogue_generation_contract")
    guidance = _as_mapping(root["search_guidance"], "$.search_guidance")
    oracle = _as_mapping(root["oracle_contract"], "$.oracle_contract")
    output = _as_mapping(root["output_contract"], "$.output_contract")

    if objective.get("target_branch") != source["branch_id"]:
        raise InputContractError("test_objective.target_branch must equal source_spec.branch_id")
    if objective.get("expected_policy_outcome") != "allow_cancel":
        raise InputContractError("this compiler currently supports the within-24h allow branch")
    if decision.get("action_under_test") != "cancel_reservation":
        raise InputContractError("this compiler currently supports cancel_reservation")

    predicate = _as_mapping(decision.get("target_predicate"), "$.decision_contract.target_predicate")
    _require(
        predicate,
        {"name", "source_field", "source_path", "type", "reference", "operator", "threshold", "expression"},
        "$.decision_contract.target_predicate",
    )
    if predicate["name"] != "booking_within_24h" or predicate["operator"] != "<":
        raise InputContractError("this compiler requires booking_within_24h with strict '<' semantics")
    if predicate["source_field"] != "created_at":
        raise InputContractError("booking_within_24h must be grounded in created_at")

    required_state = _as_nonempty_list(fixture.get("required_state"), "$.fixture_contract.required_state")
    isolation_constraints = _as_nonempty_list(
        fixture.get("isolation_constraints"), "$.fixture_contract.isolation_constraints"
    )
    required_index = _constraint_index(required_state, "$.fixture_contract.required_state")
    isolation_index = _constraint_index(
        isolation_constraints, "$.fixture_contract.isolation_constraints"
    )
    _validate_branch_isolation(decision, isolation_index)
    reference_time, threshold, variants = _compile_variants(fixture, predicate)
    _validate_primary_created_at(required_index, variants)

    mutable_dimensions = _as_nonempty_list(
        dialogue.get("mutable_dimensions"), "$.dialogue_generation_contract.mutable_dimensions"
    )
    mutation_catalog: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(mutable_dimensions):
        item = _as_mapping(raw, f"$.dialogue_generation_contract.mutable_dimensions[{index}]")
        _require(item, {"name", "allowed_values", "purpose"}, f"$.dialogue_generation_contract.mutable_dimensions[{index}]")
        values = _as_nonempty_list(
            item["allowed_values"],
            f"$.dialogue_generation_contract.mutable_dimensions[{index}].allowed_values",
        )
        name = str(item["name"])
        if name in mutation_catalog:
            raise InputContractError(f"duplicate mutable dimension: {name!r}")
        mutation_catalog[name] = {"allowed_values": deepcopy(values), "purpose": item["purpose"]}

    budget = _as_mapping(dialogue.get("candidate_budget"), "$.dialogue_generation_contract.candidate_budget")
    _require(budget, {"baseline", "adversarial", "max_turns"}, "$.dialogue_generation_contract.candidate_budget")
    if any(not isinstance(budget[key], int) or budget[key] < 0 for key in budget):
        raise InputContractError("candidate_budget values must be non-negative integers")
    if budget["baseline"] < 1 or budget["max_turns"] < 1:
        raise InputContractError("candidate_budget requires at least one baseline and one max turn")

    plan = {
        "plan_schema_version": "generation-plan/v0.1",
        "input_id": root["input_id"],
        "domain": root["domain"],
        "source": {
            "workbook": source["workbook"],
            "sheet": source["sheet"],
            "row": source["row"],
            "spec_id": source["spec_id"],
            "branch_id": source["branch_id"],
            "evidence_quote": source["evidence_quote"],
        },
        "decision": {
            "action_under_test": decision["action_under_test"],
            "target_predicate": deepcopy(predicate),
            "expected_policy_outcome": objective["expected_policy_outcome"],
            "precedence": deepcopy(decision["precedence"]),
            "alternative_allow_conditions": deepcopy(decision["alternative_allow_conditions"]),
        },
        "clock": {
            "reference_time": _iso_timestamp(reference_time),
            "source": fixture["reference_clock"]["source"],
            "threshold_iso8601": predicate["threshold"],
            "threshold_seconds": int(threshold.total_seconds()),
        },
        "fixture": {
            "entity_selection": deepcopy(fixture["entity_selection"]),
            "required_state": deepcopy(required_state),
            "isolation_constraints": deepcopy(isolation_constraints),
            "variants": variants,
        },
        "dialogue_search": {
            "base_user_goal": dialogue["base_user_goal"],
            "user_known_information": deepcopy(dialogue["user_known_information"]),
            "disclosure_policy": dialogue["disclosure_policy"],
            "semantic_invariants": deepcopy(dialogue["semantic_invariants"]),
            "mutation_catalog": mutation_catalog,
            "forbidden_mutations": deepcopy(dialogue["forbidden_mutations"]),
            "candidate_budget": deepcopy(budget),
            "feedback_state_fields": deepcopy(guidance["search_state"]),
            "next_step_rules": deepcopy(guidance["next_step_rules"]),
            "stop_conditions": deepcopy(guidance["stop_conditions"]),
        },
        "oracle": deepcopy(oracle),
        "output_contract": deepcopy(output),
        "compiler_checks": {
            "source_branch_matches_objective": True,
            "strict_temporal_threshold_is_computable": True,
            "primary_fixture_matches_boundary_variant": True,
            "alternative_allow_conditions_are_isolated": True,
            "higher_priority_transfer_condition_is_false": True,
            "all_declared_variant_outcomes_match_rule": True,
        },
    }
    return plan


def compile_input_file(input_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Load, compile, and optionally persist a generation plan."""

    with input_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    plan = compile_test_input(document)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return plan
