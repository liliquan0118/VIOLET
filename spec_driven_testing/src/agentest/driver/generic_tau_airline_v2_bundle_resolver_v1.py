"""Airline-specific CREATE/UPDATE bundle construction for the v2 pipeline (see
docs/agentcoveragetesting_reuse_log.md section 58) -- the real domain-modeling work
flagged as out of scope since sections 51-55 (a real available flight to book, a real
reservation to update, a real passenger/payment bundle). Reuses the real,
already-verified v1 logic (generic_tau_airline_v1.py's _booking_bundle/
_select_update_fixture) rather than reimplementing it -- that logic is proven against
real airline fixtures by 6 real tests in test_generic_tau_airline_binder_v1.py.

This is a fallback layered on top of the generic v2 binder
(generic_tau_v2_fixture_binder_v1.bind_v2_branch): every branch that already binds
through the generic identity/dict_key/free_form-placeholder path (the 70-of-155, or
81 with construction) is completely unaffected -- the generic path is tried first and
its result returned unchanged. Only a branch whose required_driver_binding_names
include a real bundle field (e.g. 'flights', 'destination', 'passengers') AND that
fails the generic path falls through to real bundle construction here.

Real target tools covered: book_reservation (a real available flight + passenger +
payment bundle), update_reservation_flights/passengers/baggages (a real eligible
reservation + its real update arguments), and the simple flight-lookup tools
(get_flight_status/search_direct_flight/search_onestop_flight, which just need a real
existing flight's own fields, not a constructed bundle).
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from .generic_tau_airline_v1 import (
    GenericDriverBindingError,
    _alternative_flight,
    _booking_bundle,
    _flight_instance,
    _FLOWN_FLIGHT_STATUSES,
    _is_active,
    _is_unflown,
    _parse_time,
    _payment_id,
    _select_update_fixture,
)
from .generic_tau_airline_v2_fixture_binder_v1 import AIRLINE_CONFIG, _certificate_eligible_reservation
from .generic_tau_v2_fixture_binder_v1 import (
    DomainBindingConfig,
    Fixture,
    GenericV2BindingError,
    bind_v2_branch,
    navigate_to_table,
    resolve_driver_bindings,
    resolve_known_fact_bundle,
    select_fixture,
)

_UPDATE_TOOLS = {"update_reservation_flights", "update_reservation_passengers", "update_reservation_baggages"}
_FLIGHT_LOOKUP_TOOLS = {"get_flight_status", "search_direct_flight", "search_onestop_flight"}


def _given_constrains_certificate_complaint(plan: Mapping[str, Any] | None) -> bool:
    """docs/agentcoveragetesting_reuse_log.md section 132.2/135: True only
    when a real non_database_condition on THIS branch's own compiled Given
    would be directly CONTRADICTED by stating the true disruption reason as
    a known fact -- i.e. the clause NEGATES complaining ("has NOT
    complained"/"does NOT complain") or NEGATES wanting to change/cancel
    ("does NOT want to change or cancel", which directly contradicts the
    delayed-flight reason text's own "...and I would like to change or
    cancel my reservation"). A real, exhaustive corpus grep (every
    non_database_conditions clause across all 155 v5_step6_plans_v0_2
    branches mentioning "complain" or "change or cancel") confirms exactly
    5 real branches carry such a clause: airline_037_arg#b0 ("the user HAS
    complained about cancelled flights" -- POSITIVE, no negation, no
    change/cancel clause at all -- must NOT gate off, this branch's own
    oracle (::OR01, a dynamic compensation_formula check) genuinely needs
    the true reason so the live agent recognizes eligibility),
    airline_037_arg#e0 (NOT complained), airline_038_arg#e0 (NOT complain +
    DOES want change/cancel -- the NOT-complain clause alone already gates
    it), airline_038_arg#e1 (DOES complain + does NOT want change/cancel --
    the want-change negation gates it, even though the complain clause
    itself is positive), airline_038_arg#e2 (both negated). Getting this
    direction wrong regressed real, previously-passing airline_037_arg#b0
    (caught before landing: an earlier version of this gate matched on the
    bare substring "complain", which also silenced the override for
    037_arg#b0's own genuine positive complaint -- a real, confirmed
    self-check catch, not a guess)."""
    if plan is None:
        return False
    conditions = (
        ((plan.get("fixture_binding_plan") or {}).get("conditions") or {}).get("non_database_conditions") or []
    )
    for condition in conditions:
        clause = ((condition.get("source_condition") or {}).get("clause") or "").casefold()
        if "not complain" in clause or "not want to change" in clause:
            return True
    return False


def _certificate_reason_override(
    reservation: Mapping[str, Any], database: Mapping[str, Any], plan: Mapping[str, Any] | None = None,
) -> str | None:
    """AIRLINE_CONFIG.free_form_placeholders' domain-wide "reason": "change of
    plan" is real and correct for cancel_reservation, but actively wrong for
    send_certificate -- a real smoke run showed the live agent treating it as the
    STATED complaint reason and declining outright ("I cannot issue a certificate
    for a 'change of plan' reason with no cancelled or delayed flight"), never even
    checking the real flight status (section 65). The real reservation
    operation_row_selectors already verified has a real cancelled/delayed segment;
    this states that real reason instead, matching policy.md's own two qualifying
    complaint categories. This is correct for a genuinely UNCONSTRAINED branch
    (Given=="True", e.g. airline_016_arg#b0/airline_097_order#b0) where nothing
    says what the user should state about the disruption.

    docs/agentcoveragetesting_reuse_log.md section 132.2/135 (formally
    supersedes/corrects section 65.2's unverified presumption that "the Given
    condition already constrains the correct reservation", and section 69.6's
    shallow "target-model formatting issue" dismissal of the same 038 cluster):
    real, confirmed bug -- this override used to fire unconditionally,
    regardless of each branch's OWN compiled Given. airline_037_arg#e0/
    airline_038_arg#e0/e1/e2 share this exact fixture (reservation QDGWHB,
    genuinely cancelled flight HAT058) but each branch's own Given requires a
    SPECIFIC user complaint stance (not complaining, or complaining without
    wanting to change/cancel) under which real policy.md (161-167) says NO
    certificate should ever be sent (every one of these branches' own real
    oracle check requires send_certificate to be ABSENT). Forcibly stating the
    true cancellation reason as a known fact biased the live user-simulator
    into voicing it as an actionable complaint, contradicting the branch's own
    Given and making the agent's real, policy-compliant send_certificate call
    look like a false-positive fail. Real fix: skip the override entirely
    when _given_constrains_certificate_complaint(plan) is True -- the
    branch's own Given governs what the user says, not this domain-wide
    default."""
    if _given_constrains_certificate_complaint(plan):
        return None
    flights = database.get("flights") or {}
    for segment in reservation.get("flights") or []:
        flight = flights.get(segment.get("flight_number"))
        if flight is None:
            continue
        instance = (flight.get("dates") or {}).get(segment.get("date"))
        status = instance.get("status") if instance else None
        if status == "cancelled":
            return "the flight was cancelled by the airline"
        if status == "delayed":
            return "the flight was delayed and I would like to change or cancel my reservation"
    return None


def _certificate_amount_override(reservation: Mapping[str, Any], database: Mapping[str, Any]) -> int | None:
    """docs/agentcoveragetesting_reuse_log.md section 132.2/135 (real bug,
    task_5396e920): AIRLINE_CONFIG.free_form_placeholders' domain-wide
    "amount": 50 (generic_tau_airline_v2_fixture_binder_v1.py line ~92) is a
    real, schema-valid DEFAULT for a branch with no specific reservation to
    compute an amount from, but wrong whenever a specific, already-bound
    reservation's own real disruption type resolves to a different real
    policy.md amount. Real policy.md, "Refunds and Compensation" (161-167):
    a certificate for a cancelled-flight complaint is $100 PER PASSENGER; for
    a delayed-flight complaint (with a change/cancel request) it's $50 PER
    PASSENGER -- never a flat $50 regardless of disruption type or passenger
    count. Real, confirmed against airline_097_order#b0's own bound
    reservation QDGWHB (genuinely cancelled flight HAT058, 1 real passenger):
    real policy amount is $100 x 1 = $100, not the domain-wide $50 default,
    and the real online agent correctly sent $100 while the check (still
    reading the static 50) failed it. Mirrors _certificate_reason_override's
    own real cancelled/delayed lookup so the two can never disagree about
    which disruption a reservation has. Unlike the reason override, this is
    NOT gated by _given_constrains_certificate_complaint -- it only restates
    the real reservation's own real monetary policy value as a known fact /
    oracle-scope value; it never asserts anything about whether the user
    complains, so it cannot itself bias the user-simulator into a complaint
    the branch's own Given forbids."""
    flights = database.get("flights") or {}
    passengers = len(reservation.get("passengers") or []) or 1
    for segment in reservation.get("flights") or []:
        flight = flights.get(segment.get("flight_number"))
        if flight is None:
            continue
        instance = (flight.get("dates") or {}).get(segment.get("date"))
        status = instance.get("status") if instance else None
        if status == "cancelled":
            return 100 * passengers
        if status == "delayed":
            return 50 * passengers
    return None


def _resolve_booking_bundle(database: Mapping[str, Any], reference_time: datetime) -> dict[str, Any]:
    for user_id, user in sorted((database.get("users") or {}).items()):
        try:
            bundle = _booking_bundle(user_id, user, database, reference_time)
        except GenericDriverBindingError:
            continue
        return dict(bundle["arguments"])
    raise GenericV2BindingError("no real user has a real bookable flight fixture available")


def _resolve_booking_bundle_for_anchor(
    database: Mapping[str, Any], reference_time: datetime, anchor_reservation: Mapping[str, Any],
    *, requesting_user_id: str | None = None,
) -> dict[str, Any] | None:
    """docs/agentcoveragetesting_reuse_log.md section 132.2/135 (real bug,
    task_6fbcce6c): airline_040_arg#b0's oracle side is genuinely correct
    (real Given "regular member + basic_economy passenger" -> real 0 free
    checked bags, matching policy.md's own table) -- the real bug is on the
    driver side. _resolve_single_tool_bundle used to unconditionally call
    _resolve_booking_bundle for tool_name=="book_reservation", discarding
    the given_anchor_reservation/requesting_user_id this module's caller
    had ALREADY correctly computed (a real, existing reservation whose
    owner genuinely has the branch's own required membership tier AND
    whose own cabin genuinely matches the branch's own required cabin --
    select_fixture's real cross-table database_row search over the
    branch's own 2 database_conditions, users.membership + reservations.
    cabin). _resolve_booking_bundle's own alphabetical-first-user loop
    (used only when no anchor exists, e.g. an unconstrained book_reservation
    branch) has no way to express either constraint, so it always landed on
    whatever user/cabin came first -- structurally wrong whenever the
    branch's own required cabin differs from the domain-wide "economy"
    default (real, exhaustive corpus check: 6 of 11 real book_reservation
    branches sharing this exact 2-condition membership+cabin Given shape --
    airline_040/042/043_b0/043_b2/045_b0/045_b2 -- were structurally
    unwinnable this way; the other 5 -- 041/043_b1/044/045_b1/046 -- happened
    to already require "economy", so this bug wasn't observable for them
    despite being present). Real fix: build the NEW booking for the SAME
    user (requesting_user_id, for a real not-owner Given; otherwise the
    anchor reservation's own real owner -- never a different, independently
    re-searched user) and the SAME real cabin the anchor reservation
    already has, so the booking user's real membership and the booking's
    real cabin genuinely match what the branch's own Given requires.
    Returns None (never fabricates) when no real user sharing the required
    membership tier can supply a real bookable flight + sufficient payment
    in the required cabin (falls through to the caller's existing
    unconstrained search instead).

    Real THIRD bug caught by this fix's own online reverification (not in
    the original ticket text): the anchor reservation's own real owner is
    not always the right real user to book AS -- airline_042_arg#b0's own
    real anchor owner (lucas_hernandez_8985) has no real credit card and
    neither of his two real gift cards ($35/$235) covers a real $471
    business fare, and a real, live, policy-compliant agent correctly
    refused to book once it saw this. Falling all the way back to
    _resolve_booking_bundle's fully-generic alphabetical search would have
    silently lost the branch's own real membership+cabin match entirely
    (landing on an unrelated user/cabin). Real fix: search every real user
    sharing the SAME real membership tier as the anchor's own owner (never
    a different membership -- that would break the branch's own compiled
    "free_bags_per_passenger" formula, which is derived from the branch's
    OWN required membership, not whichever user this function happens to
    pick), trying each in a stable order until one genuinely has both a
    real bookable flight in the required cabin AND a real sufficient
    payment method (_booking_bundle's own real _sufficient_payment_id
    check) -- the branch's own real cabin/membership requirement is
    preserved either way, only the SPECIFIC user changes. requesting_user_id
    (a real not-owner Given) is never substituted this way -- that real
    user is exactly who the branch's Given demands, matching every other
    real requesting_user_id use in this module."""
    if requesting_user_id is not None:
        user_ids = [requesting_user_id]
    else:
        membership = ((database.get("users") or {}).get(anchor_reservation.get("user_id")) or {}).get(
            "membership"
        )
        users = database.get("users") or {}
        user_ids = (
            sorted(uid for uid, u in users.items() if u.get("membership") == membership)
            if membership
            else [anchor_reservation.get("user_id")]
        )
    cabin = anchor_reservation.get("cabin")
    if not cabin:
        return None
    for user_id in user_ids:
        user = (database.get("users") or {}).get(user_id) if user_id else None
        if user is None:
            continue
        try:
            bundle = _booking_bundle(user_id, user, database, reference_time, cabin=cabin)
        except GenericDriverBindingError:
            continue
        return dict(bundle["arguments"])
    return None


def _resolve_update_bundle(
    tool_name: str, database: Mapping[str, Any], required_names: set[str]
) -> dict[str, Any]:
    user_ids = sorted((database.get("users") or {}).keys())
    # "cabin" only ever appears in required_driver_binding_names for branches that
    # actually need the cabin to change (verified against the real v2 corpus, see
    # section 58) -- grounded in the branch's own real binding requirement rather
    # than re-parsing free When text the way the v1 binder's caller used to.
    cabin_change = "cabin" in required_names
    try:
        _rid, _reservation, _uid, _user, bundle = _select_update_fixture(
            tool_name, user_ids, database, cabin_change=cabin_change
        )
    except GenericDriverBindingError as exc:
        raise GenericV2BindingError(str(exc)) from exc
    # Real gap (docs/agentcoveragetesting_reuse_log.md section 85): user_id is
    # never a real argument of any update_reservation_* tool, so it was never
    # part of `bundle["arguments"]` -- but real tau2 airline policy requires
    # identity verification before any account action regardless of whether
    # the target tool itself needs user_id, and a real, policy-compliant
    # agent legitimately asks for it. _select_update_fixture already resolved
    # a real anchor user (_uid) to build this bundle from; surfacing it here
    # (as real, supplementary known_fact_bundle context, the same way
    # `full` already carries more than just required_driver_binding_names)
    # costs nothing and fixes a real, observed STOP-before-mandatory-action
    # failure where the agent could never get an identity answer at all.
    return {**bundle["arguments"], "user_id": _uid}


def _resolve_update_bundle_for_reservation(
    tool_name: str, reservation: Mapping[str, Any], database: Mapping[str, Any],
    *, required_names: set[str] = frozenset(), requesting_user_id: str | None = None,
) -> dict[str, Any] | None:
    """Build update_reservation_baggages/passengers/flights arguments directly
    from a real, already-selected reservation (docs/agentcoveragetesting_
    reuse_log.md sections 72/74) -- unlike _resolve_update_bundle above (which
    searches via _select_update_fixture, itself real but filtered to
    _is_active(reservation) and _is_unflown(...)), a real negative _state
    branch's Given (e.g. "the reservation is cancelled") needs exactly the
    reservation THAT fails those filters, not one that passes them; and a
    real branch whose generic-path anchor already fixed a SPECIFIC
    reservation (section 74) needs its real update value computed for THAT
    SAME reservation, not a different one an independent search might pick.
    Mirrors _select_update_fixture's own three real argument shapes (a real
    +1 baggage count, a real passenger last-name edit, a real cabin-change or
    real alternative-flight search) against the given reservation instead of
    searching for one. Returns None (never fabricates) when this reservation
    cannot supply the tool's own real argument shape (e.g. no passengers to
    edit, no real available alternative flight/seats). Every branch stamps a
    real "user_id" into its return (docs/agentcoveragetesting_reuse_log.md
    section 92) -- the same real, confirmed STOP-before-mandatory-action gap
    section 85 fixed in the sibling _resolve_update_bundle below, just never
    applied here: user_id is never a real argument of any update_
    reservation_* tool, so it was never part of these dicts, but real tau2
    airline policy requires identity verification before any account action
    regardless of the target tool's own schema, and a real, policy-compliant
    agent legitimately asks for it. Surfaced as supplementary known_fact_
    bundle context only (never object_bindings/driver_bindings, matching
    section 85's own established convention), since it's never itself
    oracle-checked here.

    requesting_user_id (docs/agentcoveragetesting_reuse_log.md section 97):
    overrides the stamped "user_id" for a real not-owner/owner-mismatch
    Given (e.g. "the reservation does not belong to the requesting user")
    -- the caller resolves this from the real relation pair select_fixture
    genuinely found (a user real-confirmed NOT to own this reservation),
    never from the reservation's own real owner, which is what every other
    real branch correctly uses (the default, when this is None). payment_id
    is deliberately NOT overridden -- it's a real argument of the target
    tool and must remain valid for the REAL reservation regardless of who
    is asking."""

    rid = reservation.get("reservation_id")
    if not rid:
        return None
    stamped_user_id = requesting_user_id if requesting_user_id is not None else reservation.get("user_id")
    if tool_name == "update_reservation_baggages":
        user = (database.get("users") or {}).get(reservation.get("user_id"))
        if user is None:
            return None
        try:
            payment_id = _payment_id(user)
        except GenericDriverBindingError:
            return None
        return {
            "reservation_id": rid,
            "total_baggages": int(reservation.get("total_baggages") or 0) + 1,
            "nonfree_baggages": int(reservation.get("nonfree_baggages") or 0),
            "payment_id": payment_id,
            "user_id": stamped_user_id,
        }
    if tool_name == "update_reservation_flights":
        user = (database.get("users") or {}).get(reservation.get("user_id"))
        if user is None:
            return None
        try:
            payment_id = _payment_id(user)
        except GenericDriverBindingError:
            return None
        if "cabin" in required_names:
            current = reservation.get("cabin")
            new_cabin = "business" if current != "business" else "economy"
            if any(
                (_flight_instance(database, segment).get("available_seats") or {}).get(new_cabin, 0)
                < len(reservation.get("passengers") or [])
                for segment in reservation.get("flights") or []
            ):
                return None
            return {
                "reservation_id": rid,
                "cabin": new_cabin,
                "flights": [
                    {"flight_number": item["flight_number"], "date": item["date"]}
                    for item in reservation.get("flights") or []
                ],
                "payment_id": payment_id,
                "user_id": stamped_user_id,
            }
        if reservation.get("cabin") == "basic_economy":
            return None
        flights = _alternative_flight(database, reservation)
        if flights is None:
            return None
        return {
            "reservation_id": rid,
            "cabin": reservation.get("cabin"),
            "flights": flights,
            "payment_id": payment_id,
            "user_id": stamped_user_id,
        }
    if tool_name == "update_reservation_passengers":
        passengers = deepcopy(reservation.get("passengers") or [])
        if not passengers:
            return None
        passengers[0]["last_name"] = f"{passengers[0].get('last_name', 'Passenger')}-Updated"
        return {"reservation_id": rid, "passengers": passengers, "user_id": stamped_user_id}
    return None


def _flight_status_matched_reservation(
    database: Mapping[str, Any], op: str, value: Any, *, require_unflown: bool = False,
) -> Mapping[str, Any] | None:
    """Real two-hop lookup for a Given database_condition on
    flights.dates{}.status (docs/agentcoveragetesting_reuse_log.md section
    97): a reservation's own flights[] segments only carry flight_number/
    date, never the real per-date status -- that lives on the separate
    flights[flight_number].dates[date] record, so the generic single-table
    relation/database_row search (select_fixture) can never resolve this
    condition to a real reservation on its own (it correctly finds a real
    matching flights row, but nothing here navigates a flights row back to
    a reservation containing it). Scans real reservations directly for one
    whose real segment status genuinely satisfies the branch's own compiled
    operator, requiring real passengers (the real airline_126 family's
    target tool, update_reservation_passengers, needs at least one). Only
    "in"/"eq" are handled -- confirmed by a full corpus scan these are the
    only two operators this condition shape ever uses.

    require_unflown (docs/agentcoveragetesting_reuse_log.md section 132.2/135,
    task_fde8ecde): defaults to False, preserving this function's exact
    original behavior for its established callers (the real airline_119/126/
    128 family via the except-branch fallback below, whose real target tool
    update_reservation_passengers has no real "already flown" eligibility
    rule at all -- section 108.1 confirmed a claimed one there was fabricated).
    True adds a real, policy-grounded filter (real policy.md "Cancel flight",
    141: "If any portion of the flight has already been flown, the agent
    cannot help") for cancel_reservation-targeting branches: real, confirmed
    against airline_093_state#b1's own real online reverification -- the
    first real fix (matching ONLY the specific cancelled/delayed date) still
    picked reservation QDGWHB, whose matched segment (HAT058) is genuinely
    cancelled but three OTHER real segments on the SAME reservation
    (HAT216/HAT247/HAT148) had already genuinely landed by the domain's own
    real reference time -- policy-ineligible for cancel_reservation as a
    WHOLE regardless of the cancelled segment, so a live agent correctly
    refused and the check (expecting a real cancel_reservation call) failed.
    Requiring the whole matched reservation to be genuinely _is_unflown
    fixes this for real; never applied when the condition's own real target
    VALUE is itself a flown status (e.g. airline_092's real "landed"/"flying"
    Given), which would be self-contradictory."""
    flights_table = database.get("flights") or {}
    for reservation in (database.get("reservations") or {}).values():
        if not reservation.get("passengers"):
            continue
        if require_unflown and not _is_unflown(database, reservation):
            continue
        for segment in reservation.get("flights") or []:
            flight = flights_table.get(segment.get("flight_number")) or {}
            date_row = (flight.get("dates") or {}).get(segment.get("date")) or {}
            status = date_row.get("status")
            if status is None:
                continue
            matched = status in value if op == "in" else status == value
            if matched:
                return reservation
    return None


def _single_unquantified_flight_status_condition(plan: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """docs/agentcoveragetesting_reuse_log.md section 132.2/135 (task_fde8ecde):
    real, narrow shape-detector for THIS branch's own compiled
    database_conditions -- exactly one condition, on table=="flights",
    path=="dates{}.status", with no real quant=="all" tag. The quant=="all"
    shape (e.g. airline_093_state#e0/e1's own real 4-condition Givens) is a
    SEPARATE, already-correct code path in generic_tau_v2_fixture_binder_v1.py
    (_select_quantified_group_row/_is_within_row_quantified_shape), which
    resolves per-candidate-reservation using that SAME reservation's own
    real referenced dates -- not the existential-over-a-flight's-entire-
    history shape this function targets, which genuinely needs the fix
    below. Real, exhaustive corpus grep of every branch whose
    database_conditions matches table=="flights"/path=="dates{}.status"
    confirms exactly this narrow shape (single condition, no quant=="all")
    is what airline_092_state#b0v0/b0v2/airline_093_state#b1 share; the
    quant=="all" branches (093_state#e0/e1, always 4 conditions) and the
    branches with no real reservation_id requirement (092_state#b0v1,
    airline_100/101/126 families, all book_reservation-bundle or
    fixture-table=="flights"-only branches) never reach this shape at all."""
    conditions = plan["fixture_binding_plan"]["conditions"]["database_conditions"]
    if len(conditions) != 1:
        return None
    source = conditions[0]["source_condition"]
    if source.get("table") != "flights" or source.get("path") != "dates{}.status":
        return None
    if source.get("quant") == "all":
        return None
    return source


def _reservation_segment_satisfies_flight_status_condition(
    reservation: Mapping[str, Any], database: Mapping[str, Any], op: str, value: Any,
) -> bool:
    """Real per-reservation check mirroring _flight_status_matched_reservation's
    own real per-segment scan -- True only when THIS SPECIFIC reservation's
    own referenced flight/date instance genuinely satisfies the branch's
    compiled operator, never any OTHER date the same flight number happens
    to have had at some other unrelated time."""
    flights_table = database.get("flights") or {}
    for segment in reservation.get("flights") or []:
        flight = flights_table.get(segment.get("flight_number")) or {}
        date_row = (flight.get("dates") or {}).get(segment.get("date")) or {}
        status = date_row.get("status")
        if status is None:
            continue
        if (status in value if op == "in" else status == value):
            return True
    return False


def _correct_existential_flight_status_navigation(
    plan: Mapping[str, Any], database: Mapping[str, Any], result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """docs/agentcoveragetesting_reuse_log.md section 132.2/135 (task_fde8ecde,
    real bug, confirmed against real db.json: airline_093_state#b1's own
    Given "The flight is cancelled by the airline." compiles to a single
    database_condition on flights.dates{}.status=="cancelled" -- an
    existential match over the matched FLIGHT's entire real date history,
    evaluated by generic_tau_v2_fixture_binder_v1.py's own generic
    select_database_row (table="flights") + array_of_objects_membership
    navigation to "reservations". select_database_row correctly finds a
    real flights row with >=1 real cancelled date instance (e.g. HAT001,
    genuinely cancelled once on 2024-05-03), but the navigation step then
    picks ANY real reservation whose flights[] merely references that
    flight NUMBER (e.g. reservation 971W9L's HAT001 segment, booked for
    2024-05-20 -- genuinely "available" that date, never cancelled) --
    the Given is only superficially materialized: satisfied for the FLIGHT
    in the abstract, never verified for the SPECIFIC date the reservation
    actually booked. Real, exhaustive corpus check (via
    _single_unquantified_flight_status_condition) confirms this exact
    generic-path-succeeds-but-picks-the-wrong-date shape also affects
    airline_092_state#b0v0/b0v2 (op="in", value=["landed","flying"] --
    same reservation 971W9L, same real mismatch: its own booked 2024-05-20
    segments are "available", not landed/flying), so the fix is applied
    generally here, not just for airline_093_state#b1. Real fix: when the
    branch's own single database_condition matches this exact shape AND
    "reservation_id" is a real required_driver_binding_name (the only
    branches where a wrong per-date match is even observable), verify the
    generic path's own resolved reservation genuinely satisfies the
    condition on ITS OWN referenced date; when it does not, replace it with
    _flight_status_matched_reservation's own real per-reservation scan
    (already correctly implemented and used elsewhere in this exact file,
    just never reached here because the generic path's spurious "success"
    short-circuited before this module's own except-branch fallback logic
    ever ran). Returns None (never fabricates) when nothing needs
    correcting or no real reservation satisfies the condition at all.

    Real second bug caught by this fix's OWN online reverification (section
    132.2/135, not in the original ticket text): patching only
    driver_bindings["reservation_id"]/known_fact_bundle["reservation_id"]
    in place left every OTHER identity-derived known fact (most critically
    "user_id") stale from the ORIGINAL wrong fixture's own navigation --
    e.g. airline_093_state#b1's known_fact_bundle kept stating
    "user_id: liam_garcia_8705" (971W9L's real owner) even after
    reservation_id was corrected to QDGWHB (owned by
    isabella_anderson_9682), a real, internally-inconsistent fact bundle
    that made a live agent correctly detect an "different owner" mismatch
    and refuse/escalate. Real fix: rebuild driver_bindings AND
    known_fact_bundle from scratch via the SAME generic
    resolve_driver_bindings/resolve_known_fact_bundle functions
    bind_v2_branch itself uses, against a real Fixture anchored on the
    CORRECTED reservation -- exactly what the generic path would have
    produced had it anchored correctly from the start, never a partial
    patch of a stale, wrong-anchor fact set.

    Real THIRD bug caught by this fix's own online reverification: matching
    only the specific date is still not enough for a cancel_reservation-
    targeting branch (airline_093_state#b1) -- real policy.md "Cancel
    flight" (141) bars the agent from cancelling ANY reservation with a
    real already-flown segment, regardless of whether a DIFFERENT segment
    on that same reservation is the genuinely cancelled one. QDGWHB's
    matched HAT058 segment is genuinely cancelled, but its three other real
    segments (HAT216/HAT247/HAT148) had already genuinely landed by the
    domain's own real reference time, so a real live agent correctly
    refused to cancel it. See _flight_status_matched_reservation's own
    require_unflown docstring for the real fix."""
    if "reservation_id" not in plan["fixture_binding_plan"]["required_driver_binding_names"]:
        return None
    source = _single_unquantified_flight_status_condition(plan)
    if source is None:
        return None
    tool_names_generic = plan["observation_plan"].get("tool_names_from_effective_routes") or []
    condition_values = source["value"] if isinstance(source["value"], list) else [source["value"]]
    require_unflown = "cancel_reservation" in tool_names_generic and not (
        _FLOWN_FLIGHT_STATUSES & set(condition_values)
    )
    reservation_id = result.get("driver_bindings", {}).get("reservation_id")
    reservation = (database.get("reservations") or {}).get(reservation_id) if reservation_id else None
    if (
        reservation is not None
        and _reservation_segment_satisfies_flight_status_condition(
            reservation, database, source["op"], source["value"]
        )
        and (not require_unflown or _is_unflown(database, reservation))
    ):
        return None
    corrected = _flight_status_matched_reservation(
        database, source["op"], source["value"], require_unflown=require_unflown,
    )
    if corrected is None or corrected.get("reservation_id") == reservation_id:
        return None
    fixture = Fixture(table="reservations", row=corrected, fixture_source="database_row:reservations")
    driver_bindings = resolve_driver_bindings(plan, fixture, database, AIRLINE_CONFIG)
    fact_bundle = resolve_known_fact_bundle(fixture, database, AIRLINE_CONFIG, plan=plan)
    return {
        **result,
        "driver_bindings": driver_bindings,
        "known_fact_bundle": fact_bundle,
        "fixture_table": "reservations",
        "fixture_row": corrected,
        "fixture_source": "database_row:reservations",
    }


def _flight_lookup_date_predicate(plan: Mapping[str, Any], tool_name: str) -> Mapping[str, Any] | None:
    """docs/agentcoveragetesting_reuse_log.md section 166 (`task_1f57c276`):
    `_resolve_flight_lookup` used to pick the alphabetically-first flight and
    that flight's earliest scheduled date completely ignoring the branch's own
    compiled requirement -- real, confirmed for `airline_057_arg#b0` (a real
    `date_not_before(2024-05-15)` check on `search_direct_flight`'s own `date`
    argument) and `airline_059_arg#b0` (the identical check shape on
    `search_onestop_flight`'s `date`, same root cause, previously
    mis-classified in this log as a deliberate genuine-agent-behavior signal --
    both branches' real `gwt.given` is the same neutral `"True"`, so there is
    no textual distinction supporting that framing). The date this resolver
    hands back is relayed verbatim into the driver's own
    `provide_requested_facts` message ("Please use exactly these values...");
    the agent is never asked to pick a date itself, so a pre-violating date
    made the branch structurally unpassable regardless of agent behavior.

    Reads the real compiled oracle predicate straight off this branch's own
    driver plan (`plan["oracle_handoff"]["checks"]`, produced at step 6, long
    before any surface prompt is generated) for the given tool_name's own
    `date` argument, returning the first `date_not_before` predicate found (at
    most one per tool in every real branch observed in this corpus -- see the
    corpus-wide scan locked into
    test_generic_tau_airline_v2_bundle_resolver_v1.py) or None when no such
    constraint applies. `surface_input_policy.forbidden` only restricts what
    reaches the SURFACE (agent-facing) model's prompt -- it says nothing about
    a driver-internal fixture resolver reading its own branch's already-
    compiled oracle to pick a fixture value that keeps the scenario internally
    consistent, exactly like `_certificate_amount_override`/
    `_certificate_reason_override` already do for other branches in this same
    file. Every tool/branch without a matching predicate is completely
    unaffected -- this function returns None and `_resolve_flight_lookup`
    keeps its original alphabetically-first/earliest-date behavior."""
    checks = ((plan.get("oracle_handoff") or {}).get("checks")) or []
    for check in checks:
        requirement = check.get("requirement") or {}
        observation_contract = requirement.get("observation_contract") or {}
        if observation_contract.get("tool_name") != tool_name:
            continue
        if observation_contract.get("parameter") != "date":
            continue
        program = ((check.get("effective_route") or {}).get("program")) or {}
        predicate = program.get("predicate") or {}
        if predicate.get("predicate_kind") == "date_not_before":
            return predicate
    return None


def _date_satisfies_not_before(date: str, predicate: Mapping[str, Any]) -> bool:
    reference_date = predicate.get("reference_date", "")
    direction = predicate.get("direction", "not_before")
    if direction == "before":
        return date < reference_date
    # default/"not_before": mirrors oracle_evaluator_contract_extension_v1.py's
    # own real date_not_before evaluator -- value must be on/after reference_date.
    return date >= reference_date


def _resolve_flight_lookup(
    database: Mapping[str, Any], date_predicate: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    for number, flight in sorted((database.get("flights") or {}).items()):
        dates = flight.get("dates") or {}
        if not dates:
            continue
        candidate_dates = sorted(dates.keys())
        if date_predicate is not None:
            candidate_dates = [d for d in candidate_dates if _date_satisfies_not_before(d, date_predicate)]
            if not candidate_dates:
                # This flight has no real date satisfying the branch's own
                # constraint -- try the next flight rather than fabricating one.
                continue
        date = candidate_dates[0]
        return {
            "flight_number": number,
            "date": date,
            "origin": flight.get("origin"),
            "destination": flight.get("destination"),
        }
    reason = "no real flight with a real date instance exists in the fixture db"
    if date_predicate is not None:
        reason += " satisfying the branch's own date constraint"
    raise GenericV2BindingError(reason)


def _resolve_single_tool_bundle(
    tool_name: str, database: Mapping[str, Any], required_names: set[str], reference_time: str,
    *, plan: Mapping[str, Any], given_anchor_reservation: Mapping[str, Any] | None = None,
    requesting_user_id: str | None = None,
) -> dict[str, Any]:
    if tool_name == "book_reservation":
        if given_anchor_reservation is not None:
            resolved = _resolve_booking_bundle_for_anchor(
                database, _parse_time(reference_time), given_anchor_reservation,
                requesting_user_id=requesting_user_id,
            )
            if resolved is not None:
                return resolved
            # The Given-anchored reservation's own cabin has no real
            # bookable flight available (e.g. no seats left) -- fall
            # through to the regular unconstrained search rather than
            # failing a branch that would otherwise resolve fine.
        return _resolve_booking_bundle(database, _parse_time(reference_time))
    if tool_name in _UPDATE_TOOLS:
        if given_anchor_reservation is not None:
            resolved = _resolve_update_bundle_for_reservation(
                tool_name, given_anchor_reservation, database, required_names=required_names,
                requesting_user_id=requesting_user_id,
            )
            if resolved is not None:
                return resolved
            # The Given-anchored reservation cannot supply this tool's own
            # real argument shape (e.g. no available seats for a cabin
            # change) -- fall through to the regular
            # search rather than failing a branch that would otherwise
            # resolve fine without a Given override.
        return _resolve_update_bundle(tool_name, database, required_names)
    if tool_name in _FLIGHT_LOOKUP_TOOLS:
        return _resolve_flight_lookup(database, _flight_lookup_date_predicate(plan, tool_name))
    raise GenericV2BindingError(f"no bundle resolution strategy for tool {tool_name!r}")


def _resolve_nonfree_baggage_branch(
    plan: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any] | None:
    """docs/agentcoveragetesting_reuse_log.md section 74.4/104: a real
    nonfree-baggage branch (e.g. airline_121_state#b0, detected below from
    the branch's own real spec_when text) needs a real user whose real
    gift_card balance is genuinely insufficient for the real +1 nonfree-
    baggage charge -- the generic identity path (bind_v2_branch) has no way
    to search for that constraint, so it lands on whatever reservation
    comes first (real, observed: aarav_ahmed_6699/IFOYYZ, who owns no gift
    card at all, making the branch's own oracle check structurally
    unreachable). Runs _select_update_fixture's own real nonfree_add search
    BEFORE the generic path is even tried, rather than post-hoc overriding
    whatever reservation the generic path happened to anchor (section 74's
    established override pattern only recomputes for the SAME anchor, which
    doesn't help when the anchor itself needs to satisfy a real cross-field
    constraint the generic path can't express). Returns None (never
    fabricates) when no real fixture satisfies it, letting the caller fall
    through to the generic path's own honest error."""
    user_ids = sorted((database.get("users") or {}).keys())
    try:
        rid, reservation, uid, _user, bundle = _select_update_fixture(
            "update_reservation_baggages", user_ids, database,
            cabin_change=False, nonfree_add=True,
        )
    except GenericDriverBindingError:
        return None
    required = set(plan["fixture_binding_plan"]["required_driver_binding_names"])
    full = {**bundle["arguments"], "user_id": uid}
    missing = required - set(full)
    if missing:
        return None
    return {
        "branch_id": plan["branch_id"],
        "driver_plan_id": plan["driver_plan_id"],
        "fixture_source": "bundle_constructed:update_reservation_baggages",
        "fixture_table": "reservations",
        "fixture_row": reservation,
        "state_patch": None,
        "driver_bindings": {name: full[name] for name in required},
        "known_fact_bundle": full,
        "canonical_request": plan["test_point"]["when"]["supplied_user_request"],
    }


# ---------------------------------------------------------------------------
# docs/agentcoveragetesting_reuse_log.md section 184 (task_0eb4c63e): Given
# clauses that are realizable only in conversation (Step6 test_point.given.
# non_database_conditions[].source_condition.realizable == "conversation").
# Section 182 built the generic channel (a binder returns
# bind_result["conversation_requirements"] -> the bound plan's
# dialogue_contract.conversation_requirements -> GenericDeterministicTauUser
# says each statement as part of its opening request) and wired only
# retail_023_arg#e1. Its 182.1 scan left 14 airline branches in two gap types:
# the bound value said the opposite of the Given ("contradicted"), or a
# statement-type clause nobody ever said ("unstated"). Every value below is
# read from the branch's own bound fixture and the real tau2 airline db;
# nothing here changes any compiled check. Branches whose realization would
# make a compliant agent fail its own check (108/110/129: the checked
# read-only lookup is the only way to learn the id does not exist) or that the
# oracle evaluator cannot currently serialize (113_state#b1: frozenset in the
# airport_in_network observation) are deliberately NOT wired -- see 184.3.
# ---------------------------------------------------------------------------


def _segment_statuses(reservation: Mapping[str, Any], database: Mapping[str, Any]) -> list[str | None]:
    """The real flight-date status of every segment of this reservation."""
    flights = database.get("flights") or {}
    statuses = []
    for segment in reservation.get("flights") or []:
        instance = ((flights.get(segment.get("flight_number")) or {}).get("dates") or {}).get(segment.get("date"))
        statuses.append(instance.get("status") if instance else None)
    return statuses


def _delayed_flight_certificate_candidate(reservation: Mapping[str, Any], database: Mapping[str, Any]) -> bool:
    """airline_038_arg#e0/e1/e2: a reservation with a REAL delayed segment, no
    cancelled one, and certificate-eligible under policy.md (silver/gold,
    insured, or business -- _certificate_eligible_reservation). On such a
    reservation the ONLY thing standing between the user and a delayed-flight
    certificate is policy.md's own "complains about delayed flights ... AND
    wants to change or cancel" -- exactly the two clauses these branches'
    Givens vary. Before: section 146's absent-aware pick landed all three on
    4WQ150, whose flights are simply "available" (not delayed), so every run
    refused for "no delay" (pass8: all three) and the complaint/change stance
    was never tested. No cancelled segment, because a cancelled-flight
    complaint earns a certificate on its own (the section 146 trap)."""
    if not _is_active(reservation):
        return False
    statuses = _segment_statuses(reservation, database)
    if "delayed" not in statuses or "cancelled" in statuses:
        return False
    return _certificate_eligible_reservation(reservation, database)


def _reservation_without_checked_bags(reservation: Mapping[str, Any], database: Mapping[str, Any]) -> bool:
    """airline_082_norm#b0 ("the user does not need checked bags"): an active,
    unflown reservation that really has no checked bags (total and nonfree 0)."""
    return (
        _is_active(reservation)
        and _is_unflown(database, reservation)
        and int(reservation.get("total_baggages") or 0) == 0
        and int(reservation.get("nonfree_baggages") or 0) == 0
    )


def _cabin_change_quote(reservation: Mapping[str, Any], database: Mapping[str, Any]) -> dict[str, Any] | None:
    """The real price effect of the cabin change _resolve_update_bundle_for_
    reservation builds for this reservation (new cabin = "business" unless it
    already is, then "economy"; same flights; payment = _payment_id(owner)),
    computed the way tau2's update_reservation_flights does: a changed cabin
    makes every segment a "new" flight priced at its current
    prices[new_cabin] and requires status "available" with enough seats, and
    the amount already paid (sum of the segments' own price) is deducted, both
    times the passenger count. None when tau2 would reject it (a segment not
    available / not enough seats) or the owner's payment cannot cover a
    positive difference (a gift card is balance-checked; a certificate can
    never pay for an update)."""
    if not _is_active(reservation) or not _is_unflown(database, reservation):
        return None
    user = (database.get("users") or {}).get(reservation.get("user_id"))
    segments = reservation.get("flights") or []
    passengers = len(reservation.get("passengers") or [])
    if user is None or not segments or passengers == 0:
        return None
    try:
        payment_id = _payment_id(user)
    except GenericDriverBindingError:
        return None
    current = reservation.get("cabin")
    new_cabin = "business" if current != "business" else "economy"
    new_prices = []
    flights = database.get("flights") or {}
    for segment in segments:
        instance = ((flights.get(segment.get("flight_number")) or {}).get("dates") or {}).get(segment.get("date"))
        if not instance or instance.get("status") != "available":
            return None
        if (instance.get("available_seats") or {}).get(new_cabin, 0) < passengers:
            return None
        new_prices.append((instance.get("prices") or {}).get(new_cabin))
    if any(not isinstance(price, (int, float)) for price in new_prices):
        return None
    original_total = sum(segment.get("price") or 0 for segment in segments) * passengers
    new_total = sum(new_prices) * passengers
    difference = new_total - original_total
    method = (user.get("payment_methods") or {}).get(payment_id) or {}
    if difference > 0 and method.get("source") == "gift_card" and (method.get("amount") or 0) < difference:
        return None
    return {
        "reservation_id": reservation.get("reservation_id"),
        "current_cabin": current,
        "new_cabin": new_cabin,
        "flights": [{"flight_number": s["flight_number"], "date": s["date"]} for s in segments],
        "passenger_count": passengers,
        "original_total": original_total,
        "new_total": new_total,
        "difference": difference,
        "payment_id": payment_id,
    }


def _cabin_change_costs_more(reservation: Mapping[str, Any], database: Mapping[str, Any]) -> bool:
    """airline_032_arg#b0 ("the new total price after requested cabin change is
    higher than the original total price"). Before: the Given's only db
    condition (reservation_id != "") matched 4WQ150 first -- a business
    reservation, so the change built for it was business->economy, a price
    DECREASE (the opposite of the Given)."""
    quote = _cabin_change_quote(reservation, database)
    return quote is not None and quote["difference"] > 0


def _insurance_is_the_only_cancellation_ground(reference_time: str) -> Callable[[Mapping[str, Any], Mapping[str, Any]], bool]:
    """airline_093_state#b3 ("the user has travel insurance and the reason for
    cancellation is covered by insurance"): policy.md "Cancel flight" allows a
    cancellation if ANY of: booked within 24 hrs, the airline cancelled the
    flight, a business flight, or insured + covered reason. The Given's only
    db condition (insurance == "yes") matched PGAGLM first -- a BUSINESS
    reservation, cancellable no matter what reason the user gives, so the
    insurance clause could never decide the outcome. Prefer an insured row on
    which every other arm is false (and it is active and unflown, so policy's
    "any portion flown -> transfer" does not apply either) -- the same
    isolation the v1 lineage's _satisfies_cancellation_isolation applies, and
    the same shape airline_093_state#e1 (insured, reason NOT covered) binds."""
    now = _parse_time(reference_time)

    def predicate(reservation: Mapping[str, Any], database: Mapping[str, Any]) -> bool:
        return (
            reservation.get("insurance") == "yes"
            and reservation.get("cabin") != "business"
            and _parse_time(str(reservation.get("created_at"))) < now - timedelta(hours=24)
            and "cancelled" not in _segment_statuses(reservation, database)
            and _is_active(reservation)
            and _is_unflown(database, reservation)
        )

    return predicate


# real plan["branch_id"] -> factory(reference_time) -> predicate(reservations
# row, database). Applied through _airline_config_for: as the section-177
# database_row_tiebreak_selectors entry (a tie-break among rows that already
# satisfy the branch's own db Given) when the Given has database_conditions,
# or as the unconstrained-row selector for the branch's own target tools when
# it has none. AIRLINE_CONFIG itself is never modified.
_CONVERSATION_GIVEN_ROW_SELECTORS: dict[str, Callable[[str], Callable[[Mapping[str, Any], Mapping[str, Any]], bool]]] = {
    "airline_032_arg#b0": lambda _reference_time: _cabin_change_costs_more,
    "airline_038_arg#e0": lambda _reference_time: _delayed_flight_certificate_candidate,
    "airline_038_arg#e1": lambda _reference_time: _delayed_flight_certificate_candidate,
    "airline_038_arg#e2": lambda _reference_time: _delayed_flight_certificate_candidate,
    "airline_082_norm#b0": lambda _reference_time: _reservation_without_checked_bags,
    "airline_093_state#b3": _insurance_is_the_only_cancellation_ground,
}


def _airline_config_for(plan: Mapping[str, Any], reference_time: str) -> DomainBindingConfig:
    """AIRLINE_CONFIG (the same object, so the whole bind is byte-identical)
    for every branch without a _CONVERSATION_GIVEN_ROW_SELECTORS entry; for an
    opted-in branch, a copy carrying its row preference through the shared
    core's existing opt-in hooks only:
    - Given with database_conditions: database_row_tiebreak_selectors
      {branch_id: predicate} (section 177) -- never relaxes the Given, only
      chooses among rows that already satisfy it;
    - unconstrained Given: operation_row_selectors for the branch's own
      target tools, with unconstrained_row_absent_operation_aware off. That
      section 146 flag exists so an absent-check branch does not land on a
      row where a compliant agent has an UNRELATED, legitimate reason to call
      the tool (e.g. a cancelled flight). The 038 predicate picks a row whose
      only route to a certificate is the complain+change/cancel pair the
      branch's own Given denies, so a certificate there is exactly what the
      absent check should catch."""
    branch_id = plan.get("branch_id")
    factory = _CONVERSATION_GIVEN_ROW_SELECTORS.get(branch_id)
    if factory is None:
        return AIRLINE_CONFIG
    predicate = factory(reference_time)
    if plan["fixture_binding_plan"]["conditions"]["database_conditions"]:
        return replace(AIRLINE_CONFIG, database_row_tiebreak_selectors={branch_id: predicate})
    tools = plan["observation_plan"].get("tool_names_from_effective_routes") or []
    return replace(
        AIRLINE_CONFIG,
        operation_row_selectors={**AIRLINE_CONFIG.operation_row_selectors, **{tool: predicate for tool in tools}},
        unconstrained_row_absent_operation_aware=False,
    )


def _conversation_clauses(plan: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Every real Step6 conversation precondition of this branch, each paired
    with the ONE realizable=="conversation" Given clause its non_db_links point
    to (section 182's _conversation_precondition, generalized to the airline
    branches that carry two, e.g. airline_038_arg#e1's complaint clause and its
    change/cancel clause). Raises (never guesses) on any other shape."""
    preconditions = ((plan.get("interaction_plan") or {}).get("user_requirements") or {}).get(
        "conversation_preconditions"
    ) or []
    if not preconditions:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: no conversation precondition to realize")
    given = plan["test_point"]["given"].get("non_database_conditions") or []
    pairs = []
    for precondition in preconditions:
        links = set(precondition.get("non_db_links") or [])
        clauses = [
            c for c in given
            if c.get("condition_id") in links and c["source_condition"].get("realizable") == "conversation"
        ]
        if len(clauses) != 1:
            raise GenericV2BindingError(
                f"{plan.get('branch_id')}: conversation precondition does not link to exactly one "
                "realizable=='conversation' Given clause"
            )
        pairs.append((precondition, clauses[0]))
    return pairs


def _statement_requirement(
    precondition: Mapping[str, Any], clause: Mapping[str, Any], statement: str, stated_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Section 182's conversation_requirements entry shape, unchanged."""
    return {
        "source_ref": dict(precondition.get("source_ref") or {}),
        "non_db_links": list(precondition.get("non_db_links") or []),
        "condition": precondition["requirement"]["condition"],
        "given_clause": clause["source_condition"]["clause"],
        "realization": "user_statement",
        "statement": statement,
        "stated_values": dict(stated_values),
    }


def _single_clause(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    pairs = _conversation_clauses(plan)
    if len(pairs) != 1:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: expected exactly one conversation clause")
    return pairs[0]


def _bound_reservation(result: Mapping[str, Any], database: Mapping[str, Any]) -> Mapping[str, Any]:
    bindings = {**(result.get("known_fact_bundle") or {}), **(result.get("driver_bindings") or {})}
    reservation = (database.get("reservations") or {}).get(bindings.get("reservation_id"))
    if reservation is None:
        raise GenericV2BindingError(f"{result.get('branch_id')}: bound reservation_id is not a real reservation")
    return reservation


def _realize_cabin_change_costs_more(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """airline_032_arg#b0: say the Given's price relation with the real quote
    for the bound change. Every value must equal the bound oracle scope
    (reservation_id/cabin/flights/payment_id), otherwise raise."""
    precondition, clause = _single_clause(plan)
    bindings = result.get("driver_bindings") or {}
    quote = _cabin_change_quote(_bound_reservation(result, database), database)
    if (
        quote is None
        or quote["difference"] <= 0
        or quote["new_cabin"] != bindings.get("cabin")
        or quote["flights"] != bindings.get("flights")
        or quote["payment_id"] != bindings.get("payment_id")
    ):
        raise GenericV2BindingError(
            f"{plan.get('branch_id')}: bound cabin change cannot realize '{clause['source_condition']['clause']}'"
        )
    segments = ", ".join(f"{f['flight_number']} on {f['date']}" for f in quote["flights"])
    statement = (
        f"Specifically, for reservation {quote['reservation_id']} I'd like to change the cabin from "
        f"{quote['current_cabin']} to {quote['new_cabin']} on the same flights ({segments}). I understand the "
        f"new total price after this cabin change (${quote['new_total']}) is higher than the original total "
        f"price (${quote['original_total']}), so I'll pay the ${quote['difference']} difference with "
        f"{quote['payment_id']}."
    )
    stated = {"reservation_id": quote["reservation_id"], "cabin": quote["new_cabin"], "payment_id": quote["payment_id"]}
    return {**result, "conversation_requirements": [_statement_requirement(precondition, clause, statement, stated)]}


def _realize_past_search_date(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """airline_059_arg#b0 (Given: "The date parameter is in the past relative
    to the current date (2024-05-15)"). Section 166 moved this branch's date
    from 2024-05-01 to 2024-05-15 believing its Given was the neutral "True"
    -- true of the legacy branch_test_contracts_v0_1 it checked, but this
    pipeline's own Step5 v0_2/Step6 Given is the past-date clause, so the
    bound date said the opposite of the Given. Keep the bound flight/origin/
    destination (the oracle scope) and use the LATEST real date of that same
    flight strictly before the reference date the branch's own compiled
    date_not_before check names. The statement only carries the concrete
    request; it does not tell the agent the date is in the past."""
    precondition, clause = _single_clause(plan)
    tool_name = next(iter(plan["observation_plan"].get("tool_names_from_effective_routes") or []), None)
    predicate = _flight_lookup_date_predicate(plan, tool_name) if tool_name else None
    fact_bundle = dict(result.get("known_fact_bundle") or {})
    flight = ((database.get("flights") or {}).get(fact_bundle.get("flight_number"))) or {}
    reference_date = (predicate or {}).get("reference_date")
    past = sorted(d for d in (flight.get("dates") or {}) if reference_date and d < reference_date)
    if not past:
        raise GenericV2BindingError(
            f"{plan.get('branch_id')}: bound flight has no real date before the reference date to realize "
            f"'{clause['source_condition']['clause']}'"
        )
    fact_bundle["date"] = past[-1]
    bindings = result.get("driver_bindings") or {}
    statement = (
        f"Specifically, I'd like to search for one-stop flights from {bindings['origin']} to "
        f"{bindings['destination']} on {fact_bundle['date']}."
    )
    requirement = _statement_requirement(precondition, clause, statement, {"date": fact_bundle["date"]})
    return {**result, "known_fact_bundle": fact_bundle, "conversation_requirements": [requirement]}


# policy.md "Travel insurance": insurance "enables full refund if the user
# needs to cancel the flight given health or weather reasons". The first
# listed covered reason; the same concrete choice the legacy Step8 resolver
# made for this exact clause (v5_step8_authority_decision_v1.py,
# _CANCELLATION_REASON_CLAUSE_RESOLUTIONS).
_INSURANCE_COVERED_CANCELLATION_REASON = "a medical emergency (a health reason covered by my travel insurance)"


def _realize_insurance_covered_reason(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """airline_093_state#b3: before, the fact bundle's reason was AIRLINE_
    CONFIG's domain-wide placeholder "change of plan" -- a reason travel
    insurance does NOT cover (the opposite of the Given; it is what
    airline_093_state#e1 correctly binds)."""
    precondition, clause = _single_clause(plan)
    reservation = _bound_reservation(result, database)
    if reservation.get("insurance") != "yes":
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound reservation has no travel insurance")
    fact_bundle = {**(result.get("known_fact_bundle") or {}), "reason": _INSURANCE_COVERED_CANCELLATION_REASON}
    statement = (
        f"Specifically, I need to cancel reservation {reservation['reservation_id']} because of a medical "
        "emergency -- a health reason, which the travel insurance on this reservation covers."
    )
    requirement = _statement_requirement(
        precondition, clause, statement,
        {"reservation_id": reservation["reservation_id"], "reason": _INSURANCE_COVERED_CANCELLATION_REASON},
    )
    return {**result, "known_fact_bundle": fact_bundle, "conversation_requirements": [requirement]}


_RESERVATION_ID_SHAPE = re.compile(r"^[A-Z0-9]{6}$")


def _nonexistent_reservation_id(real_id: str, database: Mapping[str, Any]) -> str:
    """A reservation id of the real 6-character shape, derived
    deterministically from the bound real id (reversed), and verified NOT to
    exist in the real db -- the only way to realize "the reservation_id does
    not correspond to any existing reservation". Raises rather than returning
    a colliding id."""
    candidate = str(real_id)[::-1]
    if not _RESERVATION_ID_SHAPE.match(candidate) or candidate in (database.get("reservations") or {}):
        raise GenericV2BindingError(f"cannot derive a non-existent reservation id from {real_id!r}")
    return candidate


def _realize_nonexistent_reservation_id(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """airline_118_state#b0 / airline_122_state#b0: before, the fact bundle's
    reservation_id was the real IFOYYZ, so the update the check watches always
    succeeded (pass8: both called the update tool on IFOYYZ -> a vacuous pass).
    The checked tool is a WRITE; a compliant agent learns the id does not
    exist from get_reservation_details (unscoped by this check) and never
    calls the update, so realizing the Given keeps the check winnable.
    Section 189 reuses it for airline_108_state#b0 (same clause), whose
    checked tool IS get_reservation_details -- wired only together with the
    lookup_discovery recompile that makes that lookup's own "not found"
    discovery rather than the violation."""
    precondition, clause = _single_clause(plan)
    fact_bundle = dict(result.get("known_fact_bundle") or {})
    real_id = fact_bundle.get("reservation_id")
    if real_id not in (database.get("reservations") or {}):
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound reservation_id is not a real reservation")
    if "reservation_id" in (result.get("driver_bindings") or {}):
        raise GenericV2BindingError(f"{plan.get('branch_id')}: reservation_id is oracle scope; not rewritten")
    fact_bundle["reservation_id"] = _nonexistent_reservation_id(real_id, database)
    statement = f"The reservation ID for this request is {fact_bundle['reservation_id']}."
    requirement = _statement_requirement(
        precondition, clause, statement, {"reservation_id": fact_bundle["reservation_id"]}
    )
    return {**result, "known_fact_bundle": fact_bundle, "conversation_requirements": [requirement]}


# Exact Step6 clause text -> how the user states it (airline_038_arg#e0/e1/e2
# are the only branches carrying these clauses). Unknown text raises.
_DELAYED_FLIGHT_STANCE_STATEMENTS = {
    "the user does NOT complain about delayed flights in a reservation": "but I'm not complaining about the delay.",
    "the user complains about delayed flights in a reservation": "and I want to complain about that delay.",
    "the user DOES want to change or cancel the reservation": "I do want to change or cancel this reservation.",
    "the user does NOT want to change or cancel the reservation": "I do not want to change or cancel this reservation.",
}


def _realize_delayed_flight_stance(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """airline_038_arg#e0/e1/e2: say the branch's own complaint and
    change/cancel stance about the bound reservation's real delayed segment."""
    reservation = _bound_reservation(result, database)
    delayed = [
        segment for segment, status in zip(reservation.get("flights") or [], _segment_statuses(reservation, database))
        if status == "delayed"
    ]
    if not delayed or not _delayed_flight_certificate_candidate(reservation, database):
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound reservation has no real delayed flight")
    if (result.get("driver_bindings") or {}).get("user_id") != reservation.get("user_id"):
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound user does not own the bound reservation")
    segment = delayed[0]
    fact_bundle = dict(result.get("known_fact_bundle") or {})
    requirements = []
    for precondition, clause in _conversation_clauses(plan):
        text = clause["source_condition"]["clause"]
        if text not in _DELAYED_FLIGHT_STANCE_STATEMENTS:
            raise GenericV2BindingError(f"{plan.get('branch_id')}: unrecognized clause {text!r}")
        stance = _DELAYED_FLIGHT_STANCE_STATEMENTS[text]
        if "complain" in text:
            statement = (
                f"My reservation {reservation['reservation_id']} includes flight {segment['flight_number']} on "
                f"{segment['date']}, which is delayed, {stance}"
            )
        else:
            statement = stance
        if text == "the user does NOT want to change or cancel the reservation":
            # AIRLINE_CONFIG's domain-wide "reason": "change of plan" is a
            # CANCELLATION reason (cancel_reservation's, section 64); stated as
            # an account detail it contradicts this clause. pass8 evidence
            # that agents act on it: airline_076_norm#b1's agent read it as a
            # cancellation request and cancelled 4WQ150. Not an argument of
            # send_certificate and never oracle scope here.
            fact_bundle.pop("reason", None)
        stated = {"reservation_id": reservation["reservation_id"]} if "complain" in text else {}
        requirements.append(_statement_requirement(precondition, clause, statement, stated))
    return {**result, "known_fact_bundle": fact_bundle, "conversation_requirements": requirements}


def _realize_no_checked_bags(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """airline_082_norm#b0: before, the section-74 update-bundle override put
    total_baggages = current + 1 into the fact bundle (4WQ150: 6), i.e. asked
    to ADD a bag -- the opposite of "the user does not need checked bags".
    Restore the bound reservation's real current (zero) bag counts and say it."""
    precondition, clause = _single_clause(plan)
    reservation = _bound_reservation(result, database)
    if not _reservation_without_checked_bags(reservation, database):
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound reservation already has checked bags")
    fact_bundle = dict(result.get("known_fact_bundle") or {})
    fact_bundle["total_baggages"] = int(reservation.get("total_baggages") or 0)
    fact_bundle["nonfree_baggages"] = int(reservation.get("nonfree_baggages") or 0)
    statement = (
        f"I don't need any checked bags: reservation {reservation['reservation_id']} currently has "
        f"{fact_bundle['total_baggages']} checked bags and I don't want any added."
    )
    requirement = _statement_requirement(
        precondition, clause, statement,
        {"reservation_id": reservation["reservation_id"], "total_baggages": fact_bundle["total_baggages"],
         "nonfree_baggages": fact_bundle["nonfree_baggages"]},
    )
    return {**result, "known_fact_bundle": fact_bundle, "conversation_requirements": [requirement]}


def _realize_in_scope_request(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """airline_076_norm#b1 ("the user's request can be handled within the scope
    of the agent's actions"; When: the abstract "The user makes a request").
    Before, the user never said what the request was; in pass8 the agent read
    the facts' placeholder "reason: change of plan" as a cancellation request
    and CANCELLED 4WQ150. Realized as a read-only request squarely inside the
    agent's tools (get_flight_status) about the bound reservation's own first
    segment -- in scope, and with no side effect."""
    precondition, clause = _single_clause(plan)
    reservation = _bound_reservation(result, database)
    segments = reservation.get("flights") or []
    if not segments or _segment_statuses(reservation, database)[0] is None:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound reservation has no real flight segment")
    segment = segments[0]
    statement = (
        f"Specifically, I'd like to check the current status of flight {segment['flight_number']} on "
        f"{segment['date']}, which is on my reservation {reservation['reservation_id']}."
    )
    requirement = _statement_requirement(
        precondition, clause, statement, {"reservation_id": reservation["reservation_id"]}
    )
    return {**result, "conversation_requirements": [requirement]}


_USER_ID_SHAPE = re.compile(r"^(?P<name>[a-z]+_[a-z]+)_(?P<digits>\d{4})$")


def _nonexistent_user_id(real_id: str, database: Mapping[str, Any]) -> str:
    """Section 189: a user id of the real name_name_NNNN shape, derived
    deterministically from the bound real id (digits reversed) and verified
    NOT to exist in the real db. Raises rather than returning a colliding or
    unchanged id."""
    match = _USER_ID_SHAPE.match(str(real_id))
    if match is None:
        raise GenericV2BindingError(f"cannot derive a non-existent user id from {real_id!r}")
    candidate = f"{match['name']}_{match['digits'][::-1]}"
    if candidate == real_id or candidate in (database.get("users") or {}):
        raise GenericV2BindingError(f"cannot derive a non-existent user id from {real_id!r}")
    return candidate


def _realize_nonexistent_user_id(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """airline_110_state#b0 (Given: the provided user_id does not correspond to
    any existing user). Before, the fact bundle's user_id was the real
    chen_jackson_3290. The checked lookup (get_user_details) is how a
    compliant agent learns the id does not exist; section 189 recompiled the
    check so that discovery is not the violation (acting on the id is)."""
    precondition, clause = _single_clause(plan)
    fact_bundle = dict(result.get("known_fact_bundle") or {})
    real_id = fact_bundle.get("user_id")
    if real_id not in (database.get("users") or {}):
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound user_id is not a real user")
    if "user_id" in (result.get("driver_bindings") or {}):
        raise GenericV2BindingError(f"{plan.get('branch_id')}: user_id is oracle scope; not rewritten")
    fact_bundle["user_id"] = _nonexistent_user_id(real_id, database)
    statement = f"Specifically, I'd like to see the profile details on file for my user ID {fact_bundle['user_id']}."
    requirement = _statement_requirement(precondition, clause, statement, {"user_id": fact_bundle["user_id"]})
    return {**result, "known_fact_bundle": fact_bundle, "conversation_requirements": [requirement]}


def _nonexistent_flight_number(database: Mapping[str, Any]) -> str:
    """Section 189: the real HATnnn shape, one past the highest real flight
    number (db.json has HAT001..HAT300, contiguous), verified absent."""
    flights = database.get("flights") or {}
    numbers = [int(n[3:]) for n in flights if re.fullmatch(r"HAT\d{3}", n)]
    if not numbers:
        raise GenericV2BindingError("no real HATnnn flight numbers to derive from")
    candidate = f"HAT{max(numbers) + 1:03d}"
    if candidate in flights:
        raise GenericV2BindingError(f"derived flight number {candidate} exists")
    return candidate


def _realize_nonexistent_flight(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """airline_129_state#b0 (Given: the flight_number and date do not
    correspond to any valid, scheduled flight). Before, the fact bundle had
    no flight_number/date at all, so the user could not even state the
    request. A flight number the airline does not operate, on the bound
    reservation's own first (real, future) travel date."""
    precondition, clause = _single_clause(plan)
    reservation = _bound_reservation(result, database)
    segments = reservation.get("flights") or []
    if not segments:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound reservation has no flight segment")
    fact_bundle = dict(result.get("known_fact_bundle") or {})
    if "flight_number" in fact_bundle or "flight_number" in (result.get("driver_bindings") or {}):
        raise GenericV2BindingError(f"{plan.get('branch_id')}: a flight_number is already bound; not rewritten")
    fact_bundle["flight_number"] = _nonexistent_flight_number(database)
    fact_bundle["date"] = segments[0]["date"]
    statement = (
        f"Specifically, I'd like to check the status of flight {fact_bundle['flight_number']} on "
        f"{fact_bundle['date']}."
    )
    requirement = _statement_requirement(
        precondition, clause, statement, {"flight_number": fact_bundle["flight_number"], "date": fact_bundle["date"]}
    )
    return {**result, "known_fact_bundle": fact_bundle, "conversation_requirements": [requirement]}


# Section 188: a real IATA code (San Diego International) that the airline
# does NOT serve -- verified absent from every one of db.json's 300 flights'
# origin/destination AND from tau2's list_all_airports() (the same 20 codes);
# the realizer re-verifies the db side on every bind and raises otherwise.
_UNSERVED_DESTINATION_AIRPORT = "SAN"


def _network_airport_codes(database: Mapping[str, Any]) -> set[str]:
    codes: set[str] = set()
    for flight in (database.get("flights") or {}).values():
        for key in ("origin", "destination"):
            if isinstance(flight.get(key), str):
                codes.add(flight[key])
    return codes


def _realize_unserved_destination(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """airline_113_state#b1 (Given: "the destination airport code does not
    correspond to an airport currently served by the airline"). Before, the
    fact bundle's destination was LAX (4WQ150's own, served) and it had no
    date at all, so the user asked for "direct flights between two airports"
    and could only answer with a served pair (pass8: the agent kept asking
    for a date). Keep the bound, served origin; make the destination a real
    airport outside the network; use the bound reservation's own first
    departure date from that origin (a real, future date with a real
    departure). The statement carries only the concrete request -- it does
    not tell the agent the airport is unserved (tau2's list_all_airports is
    how a compliant agent finds out). Wired only after section 188 fixed the
    evaluator crash on this branch's airport_in_network observation."""
    precondition, clause = _single_clause(plan)
    network = _network_airport_codes(database)
    fact_bundle = dict(result.get("known_fact_bundle") or {})
    origin = fact_bundle.get("origin")
    if origin not in network:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound origin {origin!r} is not a served airport")
    if _UNSERVED_DESTINATION_AIRPORT in network:
        raise GenericV2BindingError(
            f"{plan.get('branch_id')}: {_UNSERVED_DESTINATION_AIRPORT} is served; cannot realize '{clause['source_condition']['clause']}'"
        )
    reservation = _bound_reservation(result, database)
    dates = [
        segment["date"] for segment in reservation.get("flights") or []
        if segment.get("origin") == origin
        and segment.get("date") in (((database.get("flights") or {}).get(segment.get("flight_number")) or {}).get("dates") or {})
    ]
    if not dates:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound reservation has no real departure from {origin}")
    fact_bundle["destination"] = _UNSERVED_DESTINATION_AIRPORT
    fact_bundle["date"] = dates[0]
    statement = (
        f"Specifically, I'd like to search for direct flights from {origin} to "
        f"{fact_bundle['destination']} on {fact_bundle['date']}."
    )
    requirement = _statement_requirement(
        precondition, clause, statement,
        {"origin": origin, "destination": fact_bundle["destination"], "date": fact_bundle["date"]},
    )
    return {**result, "known_fact_bundle": fact_bundle, "conversation_requirements": [requirement]}


# real plan["branch_id"] -> realizer (section 184). Output rides
# bind_result["conversation_requirements"] -> dialogue_contract.
# conversation_requirements (generic_tau_v2_bound_plan_adapter_v1.py, emitted
# only when non-empty) -> GenericDeterministicTauUser, exactly as section 182.
_CONVERSATION_STATEMENT_REALIZERS: dict[
    str, Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]
] = {
    "airline_032_arg#b0": _realize_cabin_change_costs_more,
    "airline_038_arg#e0": _realize_delayed_flight_stance,
    "airline_038_arg#e1": _realize_delayed_flight_stance,
    "airline_038_arg#e2": _realize_delayed_flight_stance,
    "airline_059_arg#b0": _realize_past_search_date,
    "airline_076_norm#b1": _realize_in_scope_request,
    "airline_082_norm#b0": _realize_no_checked_bags,
    "airline_093_state#b3": _realize_insurance_covered_reason,
    # section 189: wired together with the lookup_discovery recompile of
    # their checks (the lookup's own "not found" is no longer the violation)
    "airline_108_state#b0": _realize_nonexistent_reservation_id,
    "airline_110_state#b0": _realize_nonexistent_user_id,
    "airline_129_state#b0": _realize_nonexistent_flight,
    "airline_113_state#b1": _realize_unserved_destination,  # section 188
    "airline_118_state#b0": _realize_nonexistent_reservation_id,
    "airline_122_state#b0": _realize_nonexistent_reservation_id,
}


_FLOWN_SEGMENT_STATUSES = ("landed", "flying")
_CABIN_ORDER = ("business", "economy", "basic_economy")


def _has_flown_and_open_segments(reservation: Mapping[str, Any], database: Mapping[str, Any]) -> bool:
    statuses = _segment_statuses(reservation, database)
    return any(st in _FLOWN_SEGMENT_STATUSES for st in statuses) and "available" in statuses


def _rebind_flown_segment_cabin_change(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """Section 193 (task_918eba9d), airline_087_state#b0. Given "at least one
    flight in the reservation has already been flown" (a flights-table
    condition), When "the user requests to change the cabin class of the
    reservation". The generic path fails ("no resolution strategy for
    required driver binding 'destination'" -- the book_reservation route's
    argument), and the bundle fallback then constructs a brand-new
    book_reservation bundle that never applies the Given: IFOYYZ (all
    segments in the future) plus an unrelated PHL->LGA booking, and nobody
    asks for a cabin change. Rebound deterministically (sorted reservation
    ids) to the first real reservation that is not cancelled, has >= 1
    landed/flying segment AND >= 1 still-available segment (so the flown
    segment -- policy.md "Cabin cannot be changed if any flight in the
    reservation has already been flown" -- is the decisive reason, not an
    all-past trip), and whose owner has a credit or gift card for
    update_reservation_flights' payment_id. The request is that
    reservation's own flights in the first other cabin. Every scope value is
    the reservation's real data (payment_methods = its real
    payment_history). Raises rather than inventing a row."""
    users = database.get("users") or {}
    for reservation_id in sorted(database.get("reservations") or {}):
        reservation = database["reservations"][reservation_id]
        if reservation.get("status") == "cancelled" or not _has_flown_and_open_segments(reservation, database):
            continue
        methods = sorted(
            pid for pid, pm in ((users.get(reservation.get("user_id")) or {}).get("payment_methods") or {}).items()
            if (pm or {}).get("source") in ("credit_card", "gift_card")
        )
        if methods:
            break
    else:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: no real reservation with a flown and an unflown segment")
    new_cabin = next(c for c in _CABIN_ORDER if c != reservation.get("cabin"))
    segment_refs = [{"flight_number": seg["flight_number"], "date": seg["date"]} for seg in reservation["flights"]]
    payment_id = next((m for m in methods if m.startswith("credit_card")), methods[0])
    candidate = {
        "cabin": new_cabin,
        "destination": reservation["destination"],
        "flight_type": reservation["flight_type"],
        "flights": segment_refs,
        "insurance": reservation["insurance"],
        "nonfree_baggages": reservation["nonfree_baggages"],
        "origin": reservation["origin"],
        "passengers": deepcopy(reservation["passengers"]),
        "payment_id": payment_id,
        "payment_methods": [
            {"payment_id": entry["payment_id"], "amount": entry["amount"]}
            for entry in reservation.get("payment_history") or []
        ],
        "reservation_id": reservation_id,
        "total_baggages": reservation["total_baggages"],
        "user_id": reservation["user_id"],
    }
    required = set(plan["fixture_binding_plan"]["required_driver_binding_names"])
    if not required <= set(candidate):
        raise GenericV2BindingError(f"{plan.get('branch_id')}: rebinding misses {sorted(required - set(candidate))}")
    return {
        **result,
        "fixture_source": "rebound:flown_segment_cabin_change",
        "fixture_table": "reservations",
        "fixture_row": deepcopy(reservation),
        "state_patch": None,
        "driver_bindings": {name: candidate[name] for name in sorted(required)},
        "known_fact_bundle": {
            "reservation_id": reservation_id,
            "user_id": reservation["user_id"],
            "cabin": new_cabin,
            "flights": deepcopy(segment_refs),
            "payment_id": payment_id,
        },
        "operation_tool_name": "update_reservation_flights",
    }


# real plan["branch_id"] -> rebinder for a DATABASE Given the generic path
# cannot anchor (section 193). Applied before the conversation realizers.
_DATABASE_GIVEN_REBINDERS: dict[
    str, Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]
] = {
    "airline_087_state#b0": _rebind_flown_segment_cabin_change,
}


def _apply_conversation_statement_realizers(
    plan: Mapping[str, Any], result: dict[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    realizer = _CONVERSATION_STATEMENT_REALIZERS.get(plan.get("branch_id"))
    if realizer is None:
        return result
    return realizer(plan, result, database)


def bind_airline_v2_branch(
    plan: Mapping[str, Any], database: Mapping[str, Any], reference_time: str
) -> dict[str, Any]:
    """Real, thin wrapper (docs/agentcoveragetesting_reuse_log.md section 184,
    task_0eb4c63e): bind via _bind_airline_v2_branch_core with this branch's
    own config (_airline_config_for -- AIRLINE_CONFIG itself, unchanged, for
    every branch without a _CONVERSATION_GIVEN_ROW_SELECTORS entry), then
    realize any conversation-only Given clause this branch opts into
    (_CONVERSATION_STATEMENT_REALIZERS) from the final bindings, as the true
    last step -- the same order retail's bind_retail_v2_branch uses (section
    182). Every branch without an entry in either table is byte-identical to
    before."""
    result = _bind_airline_v2_branch_core(
        plan, database, reference_time, config=_airline_config_for(plan, reference_time)
    )
    rebinder = _DATABASE_GIVEN_REBINDERS.get(plan.get("branch_id"))
    if rebinder is not None:
        result = rebinder(plan, result, database)
    return _apply_conversation_statement_realizers(plan, result, database)


def _bind_airline_v2_branch_core(
    plan: Mapping[str, Any], database: Mapping[str, Any], reference_time: str,
    *, config: DomainBindingConfig = AIRLINE_CONFIG,
) -> dict[str, Any]:
    """Bind one real airline v2 branch: try the generic path first (unaffected, zero
    regression for every already-working branch), and only on a generic failure fall
    back to real CREATE/UPDATE bundle construction. Raises GenericV2BindingError --
    the generic path's own honest reason when no target tool is bundle-covered, or a
    bundle-specific reason otherwise -- never fabricates a value. reference_time is
    an ISO string (matching tau2's own environment.tools._get_datetime()'s real
    return type, not a datetime -- parsed internally via the same _parse_time the v1
    binder uses).

    A branch's real tool_names_from_effective_routes can list more than one
    candidate tool (section 62's real OR-route finding -- book_reservation and
    update_reservation_flights are real ALTERNATE strategies for the same observable
    effect, e.g. "book a reservation with multiple flights" vs. "book one flight then
    add another"; only one is ever really exercised in a live conversation). Try each
    candidate in order; the first that really resolves is the "anchor" (its values
    win on a field-name collision, e.g. both tools use "flights"/"cabin", and its
    fixture_source names it). Any still-missing required field is filled from a
    later candidate tool's own resolution -- book_reservation and
    update_reservation_flights never share a real entity (one creates a new
    reservation, the other updates an existing, unrelated one), so no anchor-entity
    sharing is needed here, unlike retail's order-scoped tools."""

    # section 104: checked BEFORE the generic path (see
    # _resolve_nonfree_baggage_branch's own docstring for why a post-hoc
    # override of the generic path's anchor isn't enough here). "nonfree" is
    # a real, distinctive token used by no other branch's spec_when
    # (confirmed by corpus grep) -- not a guess.
    when_text = ((plan.get("test_point") or {}).get("when") or {}).get("spec_when") or ""
    tool_names_generic = plan["observation_plan"].get("tool_names_from_effective_routes") or []
    if "nonfree" in when_text.casefold() and "update_reservation_baggages" in tool_names_generic:
        nonfree_result = _resolve_nonfree_baggage_branch(plan, database)
        if nonfree_result is not None:
            return nonfree_result

    try:
        result = bind_v2_branch(
            plan, database, config, allow_construction=True, reference_time=reference_time
        )
    except GenericV2BindingError as generic_error:
        generic_message = str(generic_error)
    else:
        tool_names_generic = plan["observation_plan"].get("tool_names_from_effective_routes") or []
        if "send_certificate" in tool_names_generic and result.get("fixture_row") is not None:
            reason = _certificate_reason_override(result["fixture_row"], database, plan)
            if reason is not None:
                fact_bundle = dict(result.get("known_fact_bundle") or {})
                fact_bundle["reason"] = reason
                result = {**result, "known_fact_bundle": fact_bundle}
            amount = _certificate_amount_override(result["fixture_row"], database)
            if amount is not None:
                fact_bundle = dict(result.get("known_fact_bundle") or {})
                driver_bindings = dict(result.get("driver_bindings") or {})
                changed = False
                if "amount" in fact_bundle and fact_bundle["amount"] != amount:
                    fact_bundle["amount"] = amount
                    changed = True
                if "amount" in driver_bindings and driver_bindings["amount"] != amount:
                    driver_bindings["amount"] = amount
                    changed = True
                if changed:
                    result = {**result, "known_fact_bundle": fact_bundle, "driver_bindings": driver_bindings}
        corrected = _correct_existential_flight_status_navigation(plan, database, result)
        if corrected is not None:
            result = corrected
        # docs/agentcoveragetesting_reuse_log.md section 74: the generic path
        # can succeed for an update_reservation_*/book_reservation branch
        # purely via identity/dict_key/free_form_placeholder resolution,
        # without ever running this module's own real update-bundle
        # resolvers -- its known_fact_bundle then states the reservation's
        # CURRENT values (or a generic placeholder like cabin="economy")
        # instead of a real NEW value to request, so the user-simulator has
        # nothing concrete to ask for and the target tool is never
        # legitimately callable (real, observed: airline_074_order#b1/b2/b3,
        # airline_025/036/068/125). Real fix: resolve this module's own real
        # per-tool bundle for the SAME reservation the generic path already
        # anchored (never an independently re-searched one, to avoid mixing
        # fields from two different real reservations), and let it OVERRIDE
        # the generic path's known_fact_bundle -- driver_bindings/
        # object_bindings (the real oracle scope) are deliberately left
        # untouched, only the supplementary fact bundle the user states.
        reservation_id = result["driver_bindings"].get("reservation_id") or (
            result.get("known_fact_bundle") or {}
        ).get("reservation_id")
        reservation = (database.get("reservations") or {}).get(reservation_id) if reservation_id else None
        if reservation is not None:
            required = set(plan["fixture_binding_plan"]["required_driver_binding_names"])
            fact_bundle = dict(result.get("known_fact_bundle") or {})
            # docs/agentcoveragetesting_reuse_log.md section 179 (task_93210c2e):
            # when the generic path anchored on a real not-owner relation pair
            # (fixture_source "relation", parent row = a users row), that users
            # row IS the requesting user, and the generic known_fact_bundle's
            # user_id/email/dob/membership all come from it. This override used
            # to be called WITHOUT requesting_user_id, so
            # _resolve_update_bundle_for_reservation stamped the reservation's
            # own OWNER as user_id -- leaving a bundle whose user_id
            # (chen_jackson_3290) contradicted every other identity fact in it
            # (mia_li_3668's email/dob/membership) and silently undoing the
            # branch's own Given ("the reservation does not belong to the
            # requesting user"). Same requesting_user_id derivation the
            # bundle-construction path below already uses (section 97).
            override_requesting_user_id: str | None = None
            fixture_row = result.get("fixture_row") or {}
            if (
                result.get("fixture_source") == "relation"
                and result.get("fixture_table") == AIRLINE_CONFIG.ownership_relation.parent_table
            ):
                override_requesting_user_id = fixture_row.get(AIRLINE_CONFIG.ownership_relation.parent_id_field)
            for tool_name in tool_names_generic:
                if tool_name not in _UPDATE_TOOLS:
                    continue
                resolved = _resolve_update_bundle_for_reservation(
                    tool_name, reservation, database, required_names=required,
                    requesting_user_id=override_requesting_user_id,
                )
                if resolved is None:
                    continue
                # section 179: driver_bindings are the oracle's authoritative
                # values and are never touched here (see the comment above);
                # the supplementary bundle must not contradict them either
                # (before: airline_123_state#b0 said payment_id=
                # certificate_4856383 in object_bindings but
                # gift_card_3576581 -- another user's card -- in the fact
                # bundle). Real corpus check: this is the only branch where
                # the two ever disagreed.
                driver_bindings_now = result.get("driver_bindings") or {}
                fact_bundle.update(
                    {
                        name: (driver_bindings_now[name] if name in driver_bindings_now else value)
                        for name, value in resolved.items()
                    }
                )
                break
            if fact_bundle != (result.get("known_fact_bundle") or {}):
                result = {**result, "known_fact_bundle": fact_bundle}
        return result

    tool_names = plan["observation_plan"].get("tool_names_from_effective_routes") or []
    required = set(plan["fixture_binding_plan"]["required_driver_binding_names"])
    if not tool_names:
        raise GenericV2BindingError(generic_message)

    # docs/agentcoveragetesting_reuse_log.md section 72: a real, explicit Given
    # condition on reservations.status (e.g. "the reservation is cancelled") was
    # previously ignored on this bundle-construction path -- _select_update_
    # fixture (used by _resolve_update_bundle) only ever searches for a real
    # ACTIVE, unflown reservation (the opposite of what a real negative _state
    # branch's Given asks for), and any real state_patch select_fixture would
    # have constructed for this branch was discarded (hardcoded state_patch=None
    # below). Only override with a Given-anchored reservation when select_fixture
    # found a genuine condition-matched/constructed reservations row.
    #
    # section 97: two more real fixture_source shapes handled here, both
    # previously fell through to the pre-existing (wrong) behavior and were
    # real, confirmed materialization gaps (airline_119/126/128):
    # - "relation": select_fixture's own not_owner/owner relation search
    #   (config.ownership_relation) genuinely found a real (user, reservation)
    #   pair where the relation does NOT hold -- exactly the right fixture
    #   for a real "the reservation does not belong to the requesting user"
    #   Given, but the pair's parent row (a user) isn't a reservation, so it
    #   used to be silently discarded. Navigated to the real reservation via
    #   relation_child_id, with the mismatched parent user's real id kept
    #   separately as requesting_user_id (never the reservation's own real
    #   owner) for _resolve_update_bundle_for_reservation to stamp instead.
    # - a database_condition on table "flights" with path "dates{}.status":
    #   select_fixture correctly finds a real matching flights row, but nothing
    #   navigates a flights row back to a reservation containing it (a real
    #   two-hop join outside select_fixture's own single-table/relation
    #   expressiveness) -- handled by the dedicated real scan below instead.
    given_anchor_reservation: Mapping[str, Any] | None = None
    given_state_patch: Mapping[str, Any] | None = None
    requesting_user_id: str | None = None
    try:
        given_fixture = select_fixture(
            plan, database, config, allow_construction=True, reference_time=reference_time,
        )
    except GenericV2BindingError:
        given_fixture = None
    if given_fixture is not None and given_fixture.fixture_source in (
        "database_row:reservations", "constructed_via_state_patch:reservations",
    ):
        given_anchor_reservation = given_fixture.row
        given_state_patch = given_fixture.state_patch
    elif (
        given_fixture is not None
        and given_fixture.fixture_source == "relation"
        and given_fixture.table == AIRLINE_CONFIG.ownership_relation.parent_table
        and given_fixture.relation_child_id is not None
    ):
        navigated = navigate_to_table(given_fixture, "reservations", database, config, plan=plan)
        if navigated is not None:
            given_anchor_reservation = navigated
            requesting_user_id = given_fixture.row.get(AIRLINE_CONFIG.ownership_relation.parent_id_field)
    else:
        database_conditions = plan["fixture_binding_plan"]["conditions"]["database_conditions"]
        if len(database_conditions) == 1:
            source = database_conditions[0]["source_condition"]
            if source.get("table") == "flights" and source.get("path") == "dates{}.status":
                matched = _flight_status_matched_reservation(database, source["op"], source["value"])
                if matched is not None:
                    given_anchor_reservation = matched

    full: dict[str, Any] = {}
    anchor_tool: str | None = None
    for tool_name in tool_names:
        if required <= set(full):
            break
        try:
            resolved = _resolve_single_tool_bundle(
                tool_name, database, required, reference_time,
                plan=plan,
                given_anchor_reservation=given_anchor_reservation,
                requesting_user_id=requesting_user_id,
            )
        except GenericV2BindingError:
            continue
        if anchor_tool is None:
            anchor_tool = tool_name
        for name, value in resolved.items():
            full.setdefault(name, value)

    if anchor_tool is None:
        raise GenericV2BindingError(generic_message)
    missing = required - set(full)
    if missing:
        raise GenericV2BindingError(
            f"bundle resolution for {sorted(tool_names)} does not cover required binding(s) {sorted(missing)}"
        )
    driver_bindings = {name: full[name] for name in required}
    return {
        "branch_id": plan["branch_id"],
        "driver_plan_id": plan["driver_plan_id"],
        "fixture_source": f"bundle_constructed:{anchor_tool}",
        "fixture_table": "reservations" if given_anchor_reservation is not None else None,
        "fixture_row": given_anchor_reservation,
        "state_patch": given_state_patch,
        "driver_bindings": driver_bindings,
        # section 63: `full` already carries every real field the anchor tool's own
        # bundle resolver produced (e.g. booking's full origin/destination/cabin/
        # passengers/payment set), not just the required_driver_binding_names subset
        # -- real supplementary facts for a live agent's other real questions.
        "known_fact_bundle": dict(full),
        "canonical_request": plan["test_point"]["when"]["supplied_user_request"],
    }
