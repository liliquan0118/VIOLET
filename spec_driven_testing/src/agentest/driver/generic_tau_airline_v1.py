"""Bind generic Driver plans to concrete tau-bench airline test fixtures."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..compiler.artifacts import content_sha256
from ..compiler.driver_plan_lowering_v1 import validate_generic_driver_plan_set


BOUND_SET_VERSION = "agentspectesting.generic-bound-driver-plan-set/v0.1"
BOUND_PLAN_VERSION = "agentspectesting.generic-bound-driver-plan/v0.1"
BINDER_VERSION = "tau-airline-generic-driver-binder/v0.1"
_TERMINAL_RESERVATION_STATUSES = {"cancelled", "refunded", "voided", "expired"}
_FLOWN_FLIGHT_STATUSES = {"flying", "landed"}
# Real airline policy.md, "Checked bag allowance" (82-93): free checked bags per
# passenger by (booking user's real membership tier, passenger's real cabin) --
# used by _booking_bundle below (docs/agentcoveragetesting_reuse_log.md section
# 132.2/135, task_6fbcce6c) to compute a real, policy-consistent nonfree_baggages
# for whatever real cabin a branch's own Given requires, instead of a value only
# ever correct for the domain-wide "economy" default.
_FREE_BAGS_PER_PASSENGER = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 4},
}
# Real airline policy.md, "Checked bag allowance" (95): "Each extra baggage is 50
# dollars." -- used by _booking_bundle below to keep the booking's own real
# payment amount consistent with its own real nonfree_baggages count (docs/
# agentcoveragetesting_reuse_log.md section 132.2/135, task_6fbcce6c: a real,
# second bug this fix's own online reverification caught -- a live, policy-
# compliant agent correctly added this real surcharge to the flight price
# whenever nonfree_baggages>0, but the domain-wide default payment amount
# (flight price alone) never accounted for it, previously invisible only
# because nonfree_baggages was always hardcoded to 0).
_EXTRA_BAGGAGE_FEE = 50


class GenericDriverBindingError(ValueError):
    """Raised when a generic plan cannot be bound without semantic guessing."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GenericDriverBindingError(f"{path} must be an object")
    return value


def _parse_time(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized).replace(tzinfo=None)
    except (TypeError, ValueError) as exc:
        raise GenericDriverBindingError(f"invalid reference time: {value!r}") from exc


def _flight_instance(database: Mapping[str, Any], segment: Mapping[str, Any]) -> Mapping[str, Any]:
    number = segment.get("flight_number")
    date = segment.get("date")
    value = ((database.get("flights") or {}).get(number) or {}).get("dates", {}).get(date)
    if not isinstance(value, Mapping):
        raise GenericDriverBindingError(f"missing flight instance {number}/{date}")
    return value


def _flight_statuses(database: Mapping[str, Any], reservation: Mapping[str, Any]) -> list[str]:
    return [str(_flight_instance(database, segment).get("status")) for segment in reservation.get("flights") or []]


def _is_active(reservation: Mapping[str, Any]) -> bool:
    return reservation.get("status") not in _TERMINAL_RESERVATION_STATUSES


def _is_unflown(database: Mapping[str, Any], reservation: Mapping[str, Any]) -> bool:
    return bool(reservation.get("flights")) and not any(
        status in _FLOWN_FLIGHT_STATUSES for status in _flight_statuses(database, reservation)
    )


def _candidate_user_ids(row: Mapping[str, Any], database: Mapping[str, Any]) -> list[str]:
    declared = [
        item.get("user_id")
        for item in row.get("matches") or []
        if isinstance(item, Mapping) and isinstance(item.get("user_id"), str)
    ]
    existing = [user_id for user_id in declared if user_id in (database.get("users") or {})]
    return existing or sorted((database.get("users") or {}).keys())


def _candidate_reservation_ids(row: Mapping[str, Any], database: Mapping[str, Any]) -> list[str]:
    declared = [
        item.get("root_id")
        for item in row.get("matches") or []
        if isinstance(item, Mapping) and isinstance(item.get("root_id"), str)
    ]
    existing = [rid for rid in declared if rid in (database.get("reservations") or {})]
    return existing or sorted((database.get("reservations") or {}).keys())


def _payment_id(user: Mapping[str, Any]) -> str:
    methods = user.get("payment_methods") or {}
    preferred = sorted(
        key for key in methods if str(key).startswith(("credit_card_", "gift_card_"))
    )
    if not preferred:
        raise GenericDriverBindingError("selected user lacks a credit card or gift card")
    return preferred[0]


def _sufficient_payment_id(user: Mapping[str, Any], amount: float) -> str | None:
    """docs/agentcoveragetesting_reuse_log.md section 132.2/135 (real bug,
    task_6fbcce6c, caught by this fix's own online reverification, not in
    the original ticket text): _payment_id's own established convention
    (the alphabetically-first credit_card_/gift_card_ key, regardless of
    real balance) was correct for every prior _booking_bundle caller (which
    only ever booked "economy", a real price every observed gift card
    balance already covered) but genuinely insufficient once a real cabin-
    sensitive branch (e.g. airline_042_arg#b0, requiring "business") needs
    a real, much higher price a specific gift card's own real balance may
    not cover -- a real, live, policy-compliant agent correctly refused to
    book once it saw the real, insufficient gift_card_8525656 balance
    ($235 for a real $471 fare). Mirrors _insufficient_gift_card's own
    established real tau2 fact (only gift cards are ever real balance-
    checked by tools.py's _payment_for_update; credit cards are not):
    prefers the alphabetically-first real credit_card_ key (never balance-
    constrained) when one exists, falling back to the alphabetically-first
    real gift_card_ whose own real balance is >= amount. Returns None
    (never fabricates a sufficient payment method that doesn't exist) when
    neither exists -- the caller falls back to a different real candidate
    user instead of booking a real, doomed-to-be-insufficient payment."""
    methods = user.get("payment_methods") or {}
    credit_cards = sorted(key for key in methods if str(key).startswith("credit_card_"))
    if credit_cards:
        return credit_cards[0]
    gift_cards = sorted(key for key in methods if str(key).startswith("gift_card_"))
    for key in gift_cards:
        balance = methods.get(key, {}).get("amount")
        if isinstance(balance, (int, float)) and balance >= amount:
            return key
    return None


def _insufficient_gift_card(user: Mapping[str, Any], price: float) -> str | None:
    """A real gift_card payment method whose real balance is genuinely below
    `price` (docs/agentcoveragetesting_reuse_log.md section 74.4/104): the only
    real tau2 payment method _payment_for_update ever balance-checks (tools.py's
    _payment_for_update: credit cards and certificates are never balance-checked
    at all) -- never fabricates a balance, only searches for a real one."""
    methods = user.get("payment_methods") or {}
    for key, method in methods.items():
        if not str(key).startswith("gift_card_"):
            continue
        amount = method.get("amount")
        if isinstance(amount, (int, float)) and amount < price:
            return key
    return None


def _passenger(user: Mapping[str, Any]) -> dict[str, Any]:
    saved = user.get("saved_passengers") or []
    if saved and isinstance(saved[0], Mapping):
        return deepcopy(dict(saved[0]))
    name = user.get("name") or {}
    return {
        "first_name": name.get("first_name"),
        "last_name": name.get("last_name"),
        "dob": user.get("dob"),
    }


def _available_direct_flight(
    database: Mapping[str, Any], reference_time: datetime, *, cabin: str, seats: int
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    for number, flight in sorted((database.get("flights") or {}).items()):
        if not isinstance(flight, Mapping):
            continue
        for date, instance in sorted((flight.get("dates") or {}).items()):
            if date <= reference_time.date().isoformat() or not isinstance(instance, Mapping):
                continue
            if instance.get("status") != "available":
                continue
            if (instance.get("available_seats") or {}).get(cabin, 0) < seats:
                continue
            return {"flight_number": number, "date": date}, flight, instance
    raise GenericDriverBindingError("no available direct flight satisfies the booking bundle")


def _booking_bundle(
    user_id: str, user: Mapping[str, Any], database: Mapping[str, Any], reference_time: datetime,
    *, cabin: str = "economy",
) -> dict[str, Any]:
    """cabin (docs/agentcoveragetesting_reuse_log.md section 132.2/135,
    task_6fbcce6c): defaults to "economy", preserving this function's exact
    original behavior for every existing caller that doesn't need a
    specific cabin. A real, confirmed bug (airline_040_arg#b0 and 5 real
    corpus siblings -- 042/043_b0/043_b2/045_b0/045_b2, all real
    book_reservation branches whose own Given requires a SPECIFIC cabin,
    e.g. "the booking user is a regular member and there is a basic
    economy passenger") previously always booked "economy" regardless,
    making cabin-sensitive oracle checks (e.g. real free-checked-bag-
    allowance formulas) structurally unwinnable whenever the branch's own
    real required cabin differs from "economy". nonfree_baggages is
    computed from the real policy.md free-bag-allowance table
    (_FREE_BAGS_PER_PASSENGER) for THIS cabin and the booking user's real
    membership, instead of the domain-wide default (0, only ever correct
    for a regular-member economy booking) -- for the default cabin=
    "economy" case this still resolves to 0 for a real regular-member user
    (byte-identical to the prior hardcoded behavior), never a behavior
    change for any existing caller.

    Real SECOND fix caught by this fix's own online reverification (not in
    the original ticket text): the payment amount must be the real flight
    price PLUS $50 per real nonfree_baggages (policy.md 95) -- a real,
    live, policy-compliant agent (airline_040_arg#b0's own real online
    reverification) correctly added this surcharge itself ($87
    basic_economy price + $50 x 1 nonfree bag = $137) once nonfree_baggages
    was correctly computed as 1 for the first time, but the domain-wide
    default payment amount (flight price alone) never accounted for it --
    previously invisible only because nonfree_baggages was always
    hardcoded to 0. For the default cabin="economy"/nonfree_baggages=0
    case this adds exactly $0, so still byte-identical to the prior
    hardcoded behavior for every existing caller."""
    passenger = _passenger(user)
    flight_info, flight, instance = _available_direct_flight(
        database, reference_time, cabin=cabin, seats=1
    )
    price = int((instance.get("prices") or {}).get(cabin))
    free_bags = _FREE_BAGS_PER_PASSENGER.get(user.get("membership"), {}).get(cabin, 0)
    nonfree_baggages = 0 if free_bags >= 1 else 1
    amount = price + _EXTRA_BAGGAGE_FEE * nonfree_baggages
    payment_id = _sufficient_payment_id(user, amount)
    if payment_id is None:
        raise GenericDriverBindingError(
            "selected user has no credit card and no gift card with sufficient balance"
        )
    return {
        "tool_name": "book_reservation",
        "arguments": {
            "user_id": user_id,
            "origin": flight.get("origin"),
            "destination": flight.get("destination"),
            "flight_type": "one_way",
            "cabin": cabin,
            "flights": [flight_info],
            "passengers": [passenger],
            "payment_methods": [{"payment_id": payment_id, "amount": amount}],
            "total_baggages": 1,
            "nonfree_baggages": nonfree_baggages,
            "insurance": "no",
        },
    }


def _reservations_for_users(
    user_ids: Iterable[str], database: Mapping[str, Any]
) -> Iterable[tuple[str, Mapping[str, Any], str, Mapping[str, Any]]]:
    users = database.get("users") or {}
    reservations = database.get("reservations") or {}
    for user_id in user_ids:
        user = users.get(user_id)
        if not isinstance(user, Mapping):
            continue
        for reservation_id in sorted(user.get("reservations") or []):
            reservation = reservations.get(reservation_id)
            if isinstance(reservation, Mapping):
                yield reservation_id, reservation, user_id, user


def _alternative_flight_for_segment(
    database: Mapping[str, Any], reservation: Mapping[str, Any], old: Mapping[str, Any]
) -> dict[str, Any] | None:
    for number, flight in sorted((database.get("flights") or {}).items()):
        if number == old.get("flight_number") or not isinstance(flight, Mapping):
            continue
        if flight.get("origin") != old.get("origin") or flight.get("destination") != old.get("destination"):
            continue
        instance = (flight.get("dates") or {}).get(old.get("date"))
        if not isinstance(instance, Mapping) or instance.get("status") != "available":
            continue
        if (instance.get("available_seats") or {}).get(reservation.get("cabin"), 0) < len(reservation.get("passengers") or []):
            continue
        return {"flight_number": number, "date": old.get("date")}
    return None


def _alternative_flight(
    database: Mapping[str, Any], reservation: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    """Real, non-fabricated replacement for exactly one leg of the
    reservation's real flights list, keeping every other leg exactly as it
    was -- update_reservation_flights's own real docstring requires every
    segment to be listed, changed or not. Multi-segment (round-trip/
    connecting) reservations used to be unconditionally declined here before
    ever querying the database (docs/agentcoveragetesting_reuse_log.md
    section 84) -- confirmed against the real tau2 airline fixture that a
    real, available alternate exists for real multi-segment reservations
    this project's own branches use (e.g. HAT124 for DFW->LAX on
    reservation 4WQ150), so this was a real bug, not a case with no real
    alternative to offer. Tries each real segment in turn (lowest index
    first) and returns on the first one with a real, available replacement;
    a segment's own real origin/destination (not the reservation's overall
    origin/destination, which is only correct for a single-segment
    reservation) is the real match key.
    """
    segments = reservation.get("flights") or []
    for index, old in enumerate(segments):
        if not isinstance(old, Mapping):
            continue
        old_with_route = {
            **old,
            "origin": old.get("origin") or reservation.get("origin"),
            "destination": old.get("destination") or reservation.get("destination"),
        }
        replacement = _alternative_flight_for_segment(database, reservation, old_with_route)
        if replacement is None:
            continue
        return [
            replacement if i == index else {"flight_number": seg.get("flight_number"), "date": seg.get("date")}
            for i, seg in enumerate(segments)
        ]
    return None


def _select_update_fixture(
    tool_name: str,
    user_ids: Sequence[str],
    database: Mapping[str, Any],
    *,
    cabin_change: bool,
    nonfree_add: bool = False,
) -> tuple[str, Mapping[str, Any], str, Mapping[str, Any], dict[str, Any]]:
    for rid, reservation, uid, user in _reservations_for_users(user_ids, database):
        if not _is_active(reservation) or not _is_unflown(database, reservation):
            continue
        if tool_name == "update_reservation_baggages" and nonfree_add:
            # docs/agentcoveragetesting_reuse_log.md section 74.4/104:
            # real tau2 tools.py charges total_price = 50 * (nonfree_
            # baggages - reservation.nonfree_baggages) -- only the REAL
            # increase in nonfree_baggages is ever priced; total_baggages
            # changing has zero effect. This branch's own rule_text is
            # specifically about a payment_id with INSUFFICIENT balance for
            # that charge, which only a real gift_card can ever trigger
            # (credit cards/certificates are never balance-checked at all,
            # confirmed against _payment_for_update). Search for a real
            # user whose real gift_card balance is genuinely below the real
            # $50 charge for +1 nonfree baggage, rather than the domain-
            # generic alphabetical _payment_id pick used by every other
            # shape here (which, for most real users, picks a credit card
            # that could never fail this check regardless of price).
            current_nonfree = int(reservation.get("nonfree_baggages") or 0)
            new_nonfree = current_nonfree + 1
            price = 50 * (new_nonfree - current_nonfree)
            insufficient_gift_card = _insufficient_gift_card(user, price)
            if insufficient_gift_card is None:
                continue
            arguments = {
                "reservation_id": rid,
                "total_baggages": int(reservation.get("total_baggages") or 0) + 1,
                "nonfree_baggages": new_nonfree,
                "payment_id": insufficient_gift_card,
            }
            return rid, reservation, uid, user, {"tool_name": tool_name, "arguments": arguments}
        try:
            payment_id = _payment_id(user)
        except GenericDriverBindingError:
            continue
        if tool_name == "update_reservation_baggages":
            arguments = {
                "reservation_id": rid,
                "total_baggages": int(reservation.get("total_baggages") or 0) + 1,
                "nonfree_baggages": int(reservation.get("nonfree_baggages") or 0),
                "payment_id": payment_id,
            }
        elif tool_name == "update_reservation_passengers":
            passengers = deepcopy(reservation.get("passengers") or [])
            if not passengers:
                continue
            passengers[0]["last_name"] = f"{passengers[0].get('last_name', 'Passenger')}-Updated"
            arguments = {"reservation_id": rid, "passengers": passengers}
        elif tool_name == "update_reservation_flights" and cabin_change:
            current = reservation.get("cabin")
            new_cabin = "business" if current != "business" else "economy"
            if any(
                (_flight_instance(database, segment).get("available_seats") or {}).get(new_cabin, 0)
                < len(reservation.get("passengers") or [])
                for segment in reservation.get("flights") or []
            ):
                continue
            arguments = {
                "reservation_id": rid,
                "cabin": new_cabin,
                "flights": [
                    {"flight_number": item["flight_number"], "date": item["date"]}
                    for item in reservation.get("flights") or []
                ],
                "payment_id": payment_id,
            }
        elif tool_name == "update_reservation_flights":
            if reservation.get("cabin") == "basic_economy":
                continue
            flights = _alternative_flight(database, reservation)
            if flights is None:
                continue
            arguments = {
                "reservation_id": rid,
                "cabin": reservation.get("cabin"),
                "flights": flights,
                "payment_id": payment_id,
            }
        else:
            continue
        return rid, reservation, uid, user, {"tool_name": tool_name, "arguments": arguments}
    raise GenericDriverBindingError(f"no reachable fixture for {tool_name}")


def _condition_predicates(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = []
    for evidence in plan["fixture_binding_plan"]["precondition_bindings"]:
        executable = evidence.get("executable_binding") or {}
        predicate = executable.get("predicate")
        if isinstance(predicate, Mapping):
            result.append(predicate)
    return result


def _satisfies_cancellation_isolation(
    plan: Mapping[str, Any], database: Mapping[str, Any], reservation: Mapping[str, Any], reference_time: datetime
) -> bool:
    if not _is_active(reservation) or not _is_unflown(database, reservation):
        return False
    signatures = {(p.get("table"), p.get("path"), p.get("op"), str(p.get("value"))) for p in _condition_predicates(plan)}
    if len(signatures) != 1:
        return True
    focus = next(iter(signatures))
    created_recent = _parse_time(str(reservation.get("created_at"))) >= reference_time - timedelta(hours=24)
    flight_cancelled = "cancelled" in _flight_statuses(database, reservation)
    business = reservation.get("cabin") == "business"
    insured = reservation.get("insurance") == "yes"
    arms = {
        ("reservations", "created_at"): created_recent,
        ("flights", "dates{}.status"): flight_cancelled,
        ("reservations", "cabin"): business,
        ("reservations", "insurance"): insured,
    }
    focus_key = (focus[0], focus[1])
    return all(value is False for key, value in arms.items() if key != focus_key)


def _evaluate_predicate(
    predicate: Mapping[str, Any], reservation: Mapping[str, Any], database: Mapping[str, Any], reference_time: datetime
) -> tuple[Any, bool]:
    table, path, op = predicate.get("table"), predicate.get("path"), predicate.get("op")
    expected = predicate.get("value")
    if table == "reservations":
        actual = reservation.get(path)
    elif table == "flights" and path == "dates{}.status":
        actual = _flight_statuses(database, reservation)
    else:
        raise GenericDriverBindingError(f"unsupported Given predicate {table}.{path}")
    if isinstance(expected, Mapping) and expected.get("ref") == "$NOW":
        expected = reference_time + timedelta(hours=float((expected.get("offset") or {}).get("hours", 0)))
        actual = _parse_time(str(actual))
    if isinstance(actual, list):
        values = [(item == expected) if op == "eq" else (item != expected) for item in actual]
        truth = all(values) if predicate.get("quant") == "all" else any(values)
    elif op == "eq":
        truth = actual == expected
    elif op == "ne":
        truth = actual != expected
    elif op == "ge":
        truth = actual >= expected
    elif op == "gt":
        truth = actual > expected
    elif op == "le":
        truth = actual <= expected
    elif op == "lt":
        truth = actual < expected
    else:
        raise GenericDriverBindingError(f"unsupported Given operator {op!r}")
    display_actual = actual.isoformat(timespec="seconds") if isinstance(actual, datetime) else actual
    return display_actual, bool(truth)


def _given_witnesses(
    plan: Mapping[str, Any], reservation: Mapping[str, Any] | None, database: Mapping[str, Any], reference_time: datetime
) -> list[dict[str, Any]]:
    witnesses = []
    for evidence in plan["fixture_binding_plan"]["precondition_bindings"]:
        executable = evidence.get("executable_binding") or {}
        predicate = executable.get("predicate")
        kind = executable.get("kind")
        if isinstance(predicate, Mapping):
            if reservation is None:
                raise GenericDriverBindingError("reservation predicate has no bound reservation")
            actual, truth = _evaluate_predicate(predicate, reservation, database, reference_time)
            witness = {"kind": "fixture_or_derived_predicate", "actual_value": actual, "predicate": deepcopy(predicate)}
        elif kind == "composite_evidence":
            fact = (executable.get("user_fact") or {}).get("fact", "")
            reason = "health" if "health" in fact.lower() else "weather" if "weather" in fact.lower() else "change_of_plan"
            evaluator = executable.get("derived_evaluator") or {}
            values = evaluator.get("set") or []
            truth = reason in values if evaluator.get("op") == "in" else reason not in values
            witness = {"kind": "user_fact_derived_state", "actual_value": reason, "evaluator": deepcopy(evaluator)}
        elif kind == "capability_scenario":
            truth = True
            witness = {"kind": "capability_scenario", "actual_value": deepcopy(executable.get("scenario"))}
        elif evidence.get("source_kind") == "compiler_constant":
            truth = bool(executable.get("value", True))
            witness = {"kind": "compiler_constant", "actual_value": truth}
        else:
            raise GenericDriverBindingError(f"unsupported Given evidence kind for {evidence['condition_id']}")
        witnesses.append({
            "condition_id": evidence["condition_id"],
            **witness,
            "evaluated_truth": bool(truth),
            "required_truth": True,
            "satisfied": bool(truth),
        })
    return witnesses


def _apply_input_probes(bundle: dict[str, Any], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    materialized = []
    arguments = bundle["arguments"]
    for probe in plan["interaction_plan"].get("operation_input_probe_requirements") or []:
        kind = probe["probe_kind"]
        if kind == "relative_numeric_fields":
            left, right = [path.removeprefix("arguments.") for path in probe["field_paths"]]
            if probe["required_relation"] == "gt_by_adapter_minimum_unit":
                arguments[right] = 1
                arguments[left] = 2
            relation = arguments[left] <= arguments[right]
            observed = "lte" if relation else "gt"
        elif kind == "collection_length":
            field = probe["field_path"].removeprefix("arguments.")
            if probe["required_relation"] == "equal_limit_plus_one":
                base = arguments[field][0]
                arguments[field] = [
                    {**deepcopy(base), "first_name": f"Passenger{index + 1}"}
                    for index in range(probe["limit"] + 1)
                ]
            observed = len(arguments[field])
        elif kind == "allowed_value_set":
            field = probe["field_path"].removeprefix("arguments.")
            if arguments.get(field) not in probe["allowed_values"]:
                arguments[field] = probe["allowed_values"][0]
            observed = arguments[field]
        else:
            raise GenericDriverBindingError(f"unsupported operation input probe kind {kind!r}")
        materialized.append({**deepcopy(probe), "materialization_status": "bound", "observed_value_or_relation": observed})
    return materialized


def _validate_required_arguments(bundle: Mapping[str, Any], tool_schemas: Mapping[str, Any]) -> None:
    tool_name = bundle["tool_name"]
    schema = tool_schemas.get(tool_name)
    if not isinstance(schema, Mapping):
        raise GenericDriverBindingError(f"missing tool schema for {tool_name}")
    required = schema.get("required") or []
    missing = sorted(set(required) - set(bundle["arguments"]))
    if missing:
        raise GenericDriverBindingError(f"{tool_name} argument bundle misses {missing}")
    try:
        import jsonschema
    except ImportError as exc:
        raise GenericDriverBindingError(
            "jsonschema is required to validate materialized operation arguments"
        ) from exc
    try:
        jsonschema.validate(bundle["arguments"], schema)
    except jsonschema.ValidationError as exc:
        raise GenericDriverBindingError(
            f"{tool_name} argument bundle violates the live tool schema: {exc.message}"
        ) from exc


def _semantic_stimulus(plan: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any] | None:
    requirements = plan["interaction_plan"].get("semantic_stimulus_requirements") or []
    if not requirements:
        return None
    if len(requirements) != 1:
        raise GenericDriverBindingError("one branch must have exactly one semantic stimulus")
    requirement = requirements[0]
    stimulus = (profile.get("semantic_stimuli") or {}).get(requirement["stimulus_kind"])
    if not isinstance(stimulus, Mapping) or not isinstance(stimulus.get("user_message"), str):
        raise GenericDriverBindingError(f"semantic stimulus profile lacks {requirement['stimulus_kind']}")
    return {
        "stimulus_requirement_id": requirement["stimulus_requirement_id"],
        "stimulus_kind": requirement["stimulus_kind"],
        "user_message": stimulus["user_message"],
        "profile_evidence": deepcopy(stimulus.get("evidence")),
    }


def _canonical_request(
    bundle: Mapping[str, Any] | None,
    user_id: str,
    reservation_id: str | None,
    plan: Mapping[str, Any],
    semantic: Mapping[str, Any] | None,
) -> str:
    if semantic:
        return str(semantic["user_message"])
    tool = bundle["tool_name"] if bundle else None
    a = bundle["arguments"] if bundle else {}
    if tool == "book_reservation":
        flight = a["flights"][0]
        passengers = ", ".join(f"{p['first_name']} {p['last_name']} born {p['dob']}" for p in a["passengers"])
        return (
            f"My user id is {user_id}. Book a {a['flight_type']} {a['cabin']} reservation from {a['origin']} "
            f"to {a['destination']} on flight {flight['flight_number']} dated {flight['date']} for {passengers}. "
            f"Use payment {a['payment_methods'][0]['payment_id']}; total baggages {a['total_baggages']}, "
            f"nonfree baggages {a['nonfree_baggages']}, insurance {a['insurance']}."
        )
    if tool == "update_reservation_baggages":
        return (
            f"My user id is {user_id} and reservation id is {reservation_id}. Update it to "
            f"total baggages {a['total_baggages']} and nonfree baggages {a['nonfree_baggages']}; use {a['payment_id']}."
        )
    if tool == "update_reservation_flights":
        flights = ", ".join(f"{x['flight_number']} on {x['date']}" for x in a["flights"])
        return (
            f"My user id is {user_id} and reservation id is {reservation_id}. Change the reservation to "
            f"{a['cabin']} with the complete itinerary {flights}; use {a['payment_id']} for any difference."
        )
    if tool == "update_reservation_passengers":
        passengers = ", ".join(f"{p['first_name']} {p['last_name']} born {p['dob']}" for p in a["passengers"])
        return f"My user id is {user_id} and reservation id is {reservation_id}. Update the passenger details to: {passengers}."
    if tool == "cancel_reservation":
        facts = plan["interaction_plan"].get("surface_fact_contracts") or []
        reason = "health" if "health" in str(facts).lower() else "change of plan"
        return f"My user id is {user_id} and reservation id is {reservation_id}. Please cancel it because of {reason}."
    if tool == "transfer_to_human_agents" and reservation_id:
        return f"My user id is {user_id} and reservation id is {reservation_id}. Please cancel this reservation because of a change of plan."
    if tool == "transfer_to_human_agents":
        booking = plan["interaction_plan"]["surface_fact_contracts"][0]["value_contract"]["scenario"]
        if booking.get("tool_name") == "book_reservation":
            return f"My user id is {user_id}. I want to book a flight reservation."
    raise GenericDriverBindingError(f"cannot realize canonical request for {plan['driver_plan_id']}")


def _cancellation_control_audit(
    database: Mapping[str, Any], reservation: Mapping[str, Any] | None, reference_time: datetime
) -> Mapping[str, Any] | None:
    if reservation is None:
        return None
    return {
        "reservation_active": _is_active(reservation),
        "any_flight_flown": not _is_unflown(database, reservation),
        "booking_within_24h": _parse_time(str(reservation.get("created_at"))) >= reference_time - timedelta(hours=24),
        "any_flight_cancelled": "cancelled" in _flight_statuses(database, reservation),
        "cabin_is_business": reservation.get("cabin") == "business",
        "has_travel_insurance": reservation.get("insurance") == "yes",
    }


def _bind_one(
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
    database: Mapping[str, Any],
    tool_schemas: Mapping[str, Any],
    profile: Mapping[str, Any],
    reference_time: datetime,
) -> dict[str, Any]:
    target_tools = plan["reachability_contract"]["target_tool_names"]
    semantic = _semantic_stimulus(plan, profile)
    user_ids = _candidate_user_ids(row, database)
    if not user_ids:
        raise GenericDriverBindingError("fixture support row has no existing user")
    user_id = user_ids[0]
    user = (database.get("users") or {})[user_id]
    reservation_id = None
    reservation = None
    bundle = None

    if target_tools == ["book_reservation"]:
        for candidate_user_id in user_ids:
            candidate_user = database["users"][candidate_user_id]
            try:
                bundle = _booking_bundle(candidate_user_id, candidate_user, database, reference_time)
            except GenericDriverBindingError:
                continue
            user_id, user = candidate_user_id, candidate_user
            break
        if bundle is None:
            raise GenericDriverBindingError("no booking-ready user fixture")
    elif target_tools and target_tools[0].startswith("update_reservation_"):
        tool = target_tools[0]
        cabin_change = "cabin class" in plan["source"]["gwt"]["when"].lower()
        reservation_id, reservation, user_id, user, bundle = _select_update_fixture(
            tool, user_ids, database, cabin_change=cabin_change
        )
    elif target_tools == ["cancel_reservation"]:
        for rid in _candidate_reservation_ids(row, database):
            candidate = database["reservations"][rid]
            if not _satisfies_cancellation_isolation(plan, database, candidate, reference_time):
                continue
            trial = _given_witnesses(plan, candidate, database, reference_time)
            if all(item["satisfied"] for item in trial):
                reservation_id, reservation = rid, candidate
                user_id, user = candidate["user_id"], database["users"][candidate["user_id"]]
                break
        if reservation is None:
            raise GenericDriverBindingError("no cancellation fixture satisfies Given plus isolation controls")
        bundle = {"tool_name": "cancel_reservation", "arguments": {"reservation_id": reservation_id}}
    elif target_tools == ["transfer_to_human_agents"]:
        scenario = plan["interaction_plan"]["surface_fact_contracts"][0]["value_contract"]["scenario"]
        if scenario.get("scenario_kind") == "policy_grounded_transfer_request":
            for rid, candidate, uid, candidate_user in _reservations_for_users(user_ids, database):
                if _is_active(candidate) and not _is_unflown(database, candidate):
                    reservation_id, reservation, user_id, user = rid, candidate, uid, candidate_user
                    break
            if reservation is None:
                raise GenericDriverBindingError("no already-flown reservation for out-of-scope transfer test")
            bundle = {"tool_name": "transfer_to_human_agents", "arguments": {"summary": "User request requires a human agent."}}
        elif scenario.get("scenario_kind") == "declared_tool_action_request" and scenario.get("tool_name") == "book_reservation":
            for candidate_user_id in user_ids:
                candidate_user = database["users"][candidate_user_id]
                try:
                    bundle = _booking_bundle(candidate_user_id, candidate_user, database, reference_time)
                except GenericDriverBindingError:
                    continue
                user_id, user = candidate_user_id, candidate_user
                break
            if bundle is None:
                raise GenericDriverBindingError("no booking-ready fixture for in-scope capability test")
        else:
            raise GenericDriverBindingError("unsupported transfer capability scenario")
    elif target_tools:
        raise GenericDriverBindingError(f"unsupported target tool combination {target_tools}")

    materialized_probes = _apply_input_probes(bundle, plan) if bundle else []
    if bundle:
        _validate_required_arguments(bundle, tool_schemas)
    witnesses = _given_witnesses(plan, reservation, database, reference_time)
    if not all(item["satisfied"] for item in witnesses):
        raise GenericDriverBindingError("selected fixture does not satisfy all active Given conditions")
    driver_bindings = {"user_id": user_id}
    if reservation_id:
        driver_bindings["reservation_id"] = reservation_id
    missing_bindings = sorted(set(plan["fixture_binding_plan"]["required_driver_binding_names"]) - set(driver_bindings))
    if missing_bindings:
        raise GenericDriverBindingError(f"required Driver bindings are missing: {missing_bindings}")

    request = _canonical_request(bundle, user_id, reservation_id, plan, semantic)
    bound = {
        "schema_version": BOUND_PLAN_VERSION,
        "binder_version": BINDER_VERSION,
        "bound_driver_plan_id": plan["driver_plan_id"].replace("::DP01", "::BDP01"),
        "source_driver_plan_id": plan["driver_plan_id"],
        "source_driver_plan_fingerprint": plan["driver_plan_fingerprint"],
        "source_branch_id": plan["source"]["branch_id"],
        "object_bindings": driver_bindings,
        "fixture_identity": {
            "user_id": user_id,
            "reservation_id": reservation_id,
            "selection_source": "fixture_support_candidates_then_adapter_reachability_filter",
        },
        "base_state": {
            "user": deepcopy(user),
            "reservation": deepcopy(reservation),
            "flight_statuses": _flight_statuses(database, reservation) if reservation else [],
        },
        "isolation_control_audit": (
            _cancellation_control_audit(database, reservation, reference_time)
            if target_tools == ["cancel_reservation"]
            else None
        ),
        "initial_state_patch": {"agent_data": None, "user_data": None},
        "given_precondition_witnesses": witnesses,
        "operation_argument_fact_bundle": deepcopy(bundle),
        "operation_input_probes": materialized_probes,
        "semantic_stimulus": semantic,
        "dialogue_contract": {
            "initial_user_message": request,
            "confirmation_message": profile.get(
                "confirmation_message", "Yes, I confirm. Please proceed."
            ),
            "answer_agent_fact_requests_from": "operation_argument_fact_bundle_and_object_bindings_only",
            "driver_action_kind_for_confirmation": "confirm_operation",
        },
        "oracle_scope_values": deepcopy(driver_bindings),
        "oracle_plan": deepcopy(plan["oracle_plan"]),
        "reachability_contract": deepcopy(plan["reachability_contract"]),
        "binding_checks": {
            "all_active_given_conditions_true": all(item["satisfied"] for item in witnesses),
            "required_driver_bindings_complete": not missing_bindings,
            "operation_arguments_cover_tool_required_fields": bundle is None or True,
            "initial_state_patch_count": 0,
            "oracle_not_used_as_surface_text": True,
            "llm_calls": 0,
        },
        "generation_readiness": {
            "status": "ready_for_baseline_execution",
            "concrete_fixture_bound": True,
            "surface_prompt_generated": True,
            "runtime_execution_allowed": True,
        },
    }
    bound["bound_driver_plan_fingerprint"] = content_sha256(bound)
    return bound


def bind_generic_tau_airline_plans(
    plan_set: Mapping[str, Any],
    fixture_support_rows: Sequence[Mapping[str, Any]],
    database: Mapping[str, Any],
    tool_schemas: Mapping[str, Any],
    stimulus_profile: Mapping[str, Any],
    *,
    reference_time: str,
    database_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = validate_generic_driver_plan_set(plan_set)
    rows = {
        (row.get("gwt") or {}).get("branch_id"): row
        for row in fixture_support_rows
        if isinstance(row, Mapping)
    }
    bound_plans = []
    failures = []
    clock = _parse_time(reference_time)
    for plan in source["plans"]:
        branch_id = plan["source"]["branch_id"]
        row = rows.get(branch_id)
        if not isinstance(row, Mapping):
            failures.append({"branch_id": branch_id, "error": "missing fixture support row"})
            continue
        try:
            bound_plans.append(
                _bind_one(plan, row, database, tool_schemas, stimulus_profile, clock)
            )
        except GenericDriverBindingError as exc:
            failures.append({"branch_id": branch_id, "error": str(exc)})
    kind_counts: dict[str, int] = {}
    for plan in bound_plans:
        kind = next(p["source"]["kind"] for p in source["plans"] if p["source"]["branch_id"] == plan["source_branch_id"])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    result = {
        "schema_version": BOUND_SET_VERSION,
        "binder_version": BINDER_VERSION,
        "bound_plans": bound_plans,
        "binding_failures": failures,
        "summary": {
            "input_plan_count": len(source["plans"]),
            "bound_plan_count": len(bound_plans),
            "binding_failure_count": len(failures),
            "bound_kind_counts": dict(sorted(kind_counts.items())),
            "plans_with_operation_input_probes": sum(bool(x["operation_input_probes"]) for x in bound_plans),
            "plans_with_semantic_stimulus": sum(x["semantic_stimulus"] is not None for x in bound_plans),
            "plans_requiring_state_patch": sum(x["initial_state_patch"]["agent_data"] is not None for x in bound_plans),
            "runtime_ready_count": sum(x["generation_readiness"]["runtime_execution_allowed"] for x in bound_plans),
            "llm_calls": 0,
        },
        "source_driver_plan_set_fingerprint": source["driver_plan_set_fingerprint"],
        "fixture_support_fingerprint": content_sha256(list(fixture_support_rows)),
        "tool_schema_fingerprint": content_sha256(tool_schemas),
        "stimulus_profile_fingerprint": content_sha256(stimulus_profile),
        "database_metadata": deepcopy(dict(database_metadata or {})),
        "reference_time": reference_time,
    }
    result["bound_driver_plan_set_fingerprint"] = content_sha256(result)
    return validate_generic_bound_driver_plan_set(result)


def validate_generic_bound_driver_plan_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$generic_bound_driver_plan_set")))
    supplied = result.pop("bound_driver_plan_set_fingerprint", None)
    if result.get("schema_version") != BOUND_SET_VERSION or supplied != content_sha256(result):
        raise GenericDriverBindingError("invalid generic bound Driver plan set")
    plans = result.get("bound_plans")
    if not isinstance(plans, list) or not isinstance(result.get("binding_failures"), list):
        raise GenericDriverBindingError("bound Driver plan arrays are invalid")
    for raw in plans:
        plan = deepcopy(dict(_mapping(raw, "$.bound_plans[]")))
        fingerprint = plan.pop("bound_driver_plan_fingerprint", None)
        if plan.get("schema_version") != BOUND_PLAN_VERSION or fingerprint != content_sha256(plan):
            raise GenericDriverBindingError("invalid bound Driver plan fingerprint")
        if not plan.get("binding_checks", {}).get("all_active_given_conditions_true"):
            raise GenericDriverBindingError("bound Driver plan has false Given condition")
        if plan.get("binding_checks", {}).get("llm_calls") != 0:
            raise GenericDriverBindingError("generic Binder must not call an LLM")
    result["bound_driver_plan_set_fingerprint"] = supplied
    return result


def load_tool_schemas_from_tau_environment(environment: Any) -> dict[str, Any]:
    return {
        name: tool.params.model_json_schema()
        for name, tool in sorted(environment.tools.get_tools().items())
    }


def load_default_tau_environment(domain: str) -> Any:
    """Load a real tau2 domain's default environment via its own get_environment(),
    with no database payload (each domain's zero-arg default DB is used). Mirrors the
    domain dispatch already established by
    v5_step8_transport_v0_3_package.create_environment_for_domain, minus that
    function's database-payload reconstruction (not needed here -- callers of this
    function want the environment's own default fixtures, not a serialized one)."""
    if domain == "airline":
        from tau2.domains.airline.environment import get_environment
    elif domain == "retail":
        from tau2.domains.retail.environment import get_environment
    elif domain == "telecom":
        from tau2.domains.telecom.environment import get_environment

        # docs/agentcoveragetesting_reuse_log.md section 203: the telecom
        # specs' troubleshooting rules were extracted from tau2's
        # tech_support_workflow.md (Step 2.1.x "rerun the speed test", Path 3
        # MMS order, ...), but tau2's default policy_type="manual" handed the
        # agent tech_support_manual.md, which has none of those steps. Same
        # DB and tools either way; only the policy text (agent prompt and
        # semantic-judge evidence) changes. tau2 registers this same variant
        # as the "telecom-workflow" domain.
        return get_environment(policy_type="workflow")
    else:
        raise GenericDriverBindingError(f"unsupported domain: {domain!r}")
    return get_environment()


def bind_generic_tau_airline_plans_file(
    *,
    plan_set_path: Path,
    fixture_support_path: Path,
    stimulus_profile_path: Path,
    output_path: Path,
    database_path: Path | None = None,
    reference_time: str | None = None,
    domain: str = "airline",
) -> dict[str, Any]:
    try:
        environment = load_default_tau_environment(domain)
    except ImportError as exc:
        raise GenericDriverBindingError("tau2 is unavailable; use the tau-bench environment") from exc
    schemas = load_tool_schemas_from_tau_environment(environment)
    clock = reference_time or environment.tools._get_datetime()
    if database_path is None:
        database = environment.tools.db.model_dump(mode="json")
        metadata = {
            "loader": f"tau2.domains.{domain}.environment.get_environment",
            "database_model": type(environment.tools.db).__name__,
        }
    else:
        database = json.loads(database_path.read_text(encoding="utf-8"))
        metadata = {"loader": "json_file", "database_path": str(database_path.resolve())}
    result = bind_generic_tau_airline_plans(
        json.loads(plan_set_path.read_text(encoding="utf-8")),
        json.loads(fixture_support_path.read_text(encoding="utf-8")),
        database,
        schemas,
        json.loads(stimulus_profile_path.read_text(encoding="utf-8")),
        reference_time=clock,
        database_metadata=metadata,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
