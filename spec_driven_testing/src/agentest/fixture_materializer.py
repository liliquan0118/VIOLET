"""Select and bind a tau-bench airline fixture for a compiled generation plan."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .input_compiler import InputContractError


class FixtureMaterializationError(ValueError):
    """Raised when no database entity can satisfy a compiled fixture contract."""


_TERMINAL_RESERVATION_STATUSES = {"cancelled", "refunded", "voided", "expired"}
_DISALLOWED_FLIGHT_STATUSES = {"cancelled", "flying", "landed"}


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise FixtureMaterializationError(f"{path}: expected one JSON object")
    return value


def load_tau_airline_database(database_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a JSON snapshot or lazily read a fresh database from tau2.

    The lazy import keeps the core package usable in environments without tau2.
    It performs no tool calls and does not mutate the source database.
    """

    if database_path is not None:
        database = _load_json_object(database_path)
        return database, {"loader": "json_file", "database_path": str(database_path.resolve())}

    try:
        from tau2.domains.airline.environment import get_environment
    except ImportError as exc:
        raise FixtureMaterializationError(
            "tau2 is unavailable; run with the tau-bench Python environment or pass --database"
        ) from exc
    environment = get_environment()
    database = environment.tools.db.model_dump(mode="json")
    return database, {
        "loader": "tau2.domains.airline.environment.get_environment",
        "database_model": type(environment.tools.db).__name__,
        "policy_clock": environment.tools._get_datetime(),
    }


def _flight_instance(
    database: Mapping[str, Any], flight_number: str, date: str
) -> Mapping[str, Any] | None:
    flight = (database.get("flights") or {}).get(flight_number)
    if not isinstance(flight, Mapping):
        return None
    instance = (flight.get("dates") or {}).get(date)
    return instance if isinstance(instance, Mapping) else None


def _candidate_from_reservation(
    database: Mapping[str, Any], reservation_id: str, reservation: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    users = database.get("users") or {}
    user_id = reservation.get("user_id")
    user = users.get(user_id) if isinstance(users, Mapping) else None

    status = reservation.get("status")
    if status in _TERMINAL_RESERVATION_STATUSES:
        reasons.append(f"terminal reservation status: {status}")
    if reservation.get("cabin") not in {"economy", "basic_economy"}:
        reasons.append("cabin is business or unsupported")
    if reservation.get("insurance") != "no":
        reasons.append("insurance is not 'no'")
    if not isinstance(user, Mapping):
        reasons.append("requesting user does not exist")
    elif reservation_id not in (user.get("reservations") or []):
        reasons.append("reservation is not listed under its user")

    raw_segments = reservation.get("flights") or []
    if not isinstance(raw_segments, list) or not raw_segments:
        reasons.append("reservation has no flights")

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
        flight_status = instance.get("status")
        if flight_status in _DISALLOWED_FLIGHT_STATUSES:
            reasons.append(f"disallowed flight status: {flight_number}/{date}={flight_status}")
        flights.append(
            {
                "flight_number": flight_number,
                "date": date,
                "origin": segment.get("origin"),
                "destination": segment.get("destination"),
                "status": flight_status,
            }
        )

    if reasons:
        return None, reasons

    assert isinstance(user, Mapping)
    name = user.get("name") if isinstance(user.get("name"), Mapping) else {}
    candidate = {
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
            "status": status,
            "origin": reservation.get("origin"),
            "destination": reservation.get("destination"),
            "flight_type": reservation.get("flight_type"),
            "cabin": reservation.get("cabin"),
            "insurance": reservation.get("insurance"),
        },
        "flights": flights,
    }
    return candidate, []


def _primary_variant(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    variants = ((plan.get("fixture") or {}).get("variants") or [])
    primary = [item for item in variants if isinstance(item, Mapping) and item.get("role") == "primary"]
    if len(primary) != 1:
        raise FixtureMaterializationError(
            f"compiled plan must contain exactly one primary fixture variant, found {len(primary)}"
        )
    return primary[0]


def _score_candidate(candidate: Mapping[str, Any], primary: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prefer no time patch, then a small/simple, semantically neutral fixture."""

    target_local = str(primary["created_at"]).rsplit("-", 1)[0]
    original = str(candidate["reservation"].get("original_created_at"))
    return (
        original != target_local,
        len(candidate["flights"]),
        candidate["reservation"].get("cabin") != "economy",
        candidate["reservation_id"],
    )


def _local_tau_timestamp(canonical_timestamp: str) -> str:
    """Convert a canonical offset timestamp to tau airline's local-naive DB format."""

    # Compiled timestamps always include seconds and a numeric offset.  Tau's
    # airline DB stores created_at as a local-naive ISO string under a fixed
    # policy clock, so the offset is retained separately in oracle bindings.
    if canonical_timestamp.endswith("Z"):
        return canonical_timestamp[:-1]
    plus = canonical_timestamp.rfind("+")
    minus = canonical_timestamp.rfind("-")
    offset_index = max(plus, minus if minus > 9 else -1)
    return canonical_timestamp[:offset_index] if offset_index > 9 else canonical_timestamp


def _variant_fixture(
    plan: Mapping[str, Any], selected: Mapping[str, Any], variant: Mapping[str, Any]
) -> dict[str, Any]:
    reservation_id = selected["reservation_id"]
    local_created_at = _local_tau_timestamp(str(variant["created_at"]))
    original_created_at = selected["reservation"].get("original_created_at")
    patch_required = original_created_at != local_created_at
    agent_data = (
        {"reservations": {reservation_id: {"created_at": local_created_at}}}
        if patch_required
        else None
    )
    return {
        "fixture_id": f"tau2.airline::{reservation_id}::{variant['variant_id']}",
        "variant_id": variant["variant_id"],
        "role": variant["role"],
        "bindings": {
            "user_id": selected["user_id"],
            "reservation_id": reservation_id,
            "user_first_name": selected["user"]["first_name"],
            "user_last_name": selected["user"]["last_name"],
            "user_email": selected["user"]["email"],
            "user_date_of_birth": selected["user"]["date_of_birth"],
            "cancellation_reason": "change_of_plan",
            "reference_time": variant["reference_time"],
            "created_at": variant["created_at"],
            "created_at_tau_local": local_created_at,
            "booking_age_iso8601": variant["booking_age_iso8601"],
            "booking_age_seconds": variant["booking_age_seconds"],
        },
        "base_state": {
            "reservation": deepcopy(selected["reservation"]),
            "flights": deepcopy(selected["flights"]),
        },
        "initial_state_patch": {
            "agent_data": agent_data,
            "user_data": None,
        },
        "predicate_audit": {
            "booking_within_24h": variant["predicate_truth"],
            "any_flight_flown": False,
            "any_flight_cancelled": False,
            "cabin_is_business": False,
            "insurance_covers_reason": False,
            "expected_policy_outcome": variant["expected_policy_outcome"],
        },
        "pre_state_probes": [
            {
                "probe_id": f"pre::{variant['variant_id']}::reservation",
                "tool_name": "get_reservation_details",
                "arguments": {"reservation_id": reservation_id},
            },
            *[
                {
                    "probe_id": f"pre::{variant['variant_id']}::flight::{index}",
                    "tool_name": "get_flight_status",
                    "arguments": {
                        "flight_number": flight["flight_number"],
                        "date": flight["date"],
                    },
                }
                for index, flight in enumerate(selected["flights"])
            ],
        ],
        "expected_terminal": {
            "policy_outcome": variant["expected_policy_outcome"],
            "reservation_status": (
                "cancelled" if variant["expected_policy_outcome"] == "allow_cancel" else "unchanged"
            ),
            "action": (
                "cancel_reservation_called_once"
                if variant["expected_policy_outcome"] == "allow_cancel"
                else "cancel_reservation_not_called"
            ),
        },
    }


def materialize_fixture_bundle(
    plan: Mapping[str, Any], database: Mapping[str, Any], database_metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Select one base reservation and produce a fixture for every boundary variant."""

    if plan.get("plan_schema_version") != "generation-plan/v0.1":
        raise InputContractError(f"unsupported generation plan: {plan.get('plan_schema_version')!r}")
    if plan.get("domain") != "airline":
        raise FixtureMaterializationError("the current materializer supports only the airline domain")
    if (plan.get("decision") or {}).get("action_under_test") != "cancel_reservation":
        raise FixtureMaterializationError("the current materializer supports cancel_reservation")

    reservations = database.get("reservations")
    if not isinstance(reservations, Mapping) or not reservations:
        raise FixtureMaterializationError("airline database has no reservations")

    primary = _primary_variant(plan)
    candidates: list[dict[str, Any]] = []
    rejected_count = 0
    rejection_reason_counts: dict[str, int] = {}
    for reservation_id, raw in sorted(reservations.items()):
        if not isinstance(raw, Mapping):
            rejected_count += 1
            rejection_reason_counts["malformed reservation"] = (
                rejection_reason_counts.get("malformed reservation", 0) + 1
            )
            continue
        candidate, reasons = _candidate_from_reservation(database, str(reservation_id), raw)
        if candidate is not None:
            candidates.append(candidate)
        else:
            rejected_count += 1
            for reason in set(reasons):
                rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
    if not candidates:
        raise FixtureMaterializationError(
            "no reservation satisfies ownership, active status, non-business cabin, no insurance, and non-flown/non-cancelled flights"
        )

    ranked = sorted(candidates, key=lambda item: _score_candidate(item, primary))
    selected = ranked[0]
    variants = [
        _variant_fixture(plan, selected, variant)
        for variant in plan["fixture"]["variants"]
    ]
    bundle = {
        "fixture_schema_version": "fixture-bundle/v0.1",
        "input_id": plan["input_id"],
        "domain": plan["domain"],
        "source_branch": plan["source"]["branch_id"],
        "database_source": deepcopy(dict(database_metadata or {})),
        "selection": {
            "candidate_count": len(candidates),
            "rejected_count": rejected_count,
            "ranking_policy": [
                "prefer a reservation whose existing created_at matches the primary variant",
                "prefer fewer flight segments",
                "prefer economy over basic_economy",
                "use reservation_id as deterministic tie-breaker",
            ],
            "selected_rank": 1,
            "selected_reservation_id": selected["reservation_id"],
            "selected_user_id": selected["user_id"],
            "selected_score": list(_score_candidate(selected, primary)),
            "top_rejection_reasons": [
                {"reason": reason, "count": count}
                for reason, count in sorted(
                    rejection_reason_counts.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ],
        },
        "selected_entity": deepcopy(selected),
        "variants": variants,
        "materialization_checks": {
            "reservation_exists": True,
            "ownership_valid": True,
            "reservation_active": True,
            "cabin_non_business": True,
            "insurance_disabled": True,
            "all_flights_observable": True,
            "no_flight_flown": True,
            "no_flight_cancelled": True,
            "one_base_entity_shared_across_variants": True,
            "only_created_at_is_patched": True,
        },
    }
    return bundle


def materialize_fixture_file(
    plan_path: Path,
    output_path: Path,
    database_path: Path | None = None,
) -> dict[str, Any]:
    plan = _load_json_object(plan_path)
    database, metadata = load_tau_airline_database(database_path)
    bundle = materialize_fixture_bundle(plan, database, metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return bundle
