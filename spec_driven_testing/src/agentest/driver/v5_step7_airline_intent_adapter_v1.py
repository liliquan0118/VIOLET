"""Airline-domain adapter for the generic intent-synthesis engine
(v5_step7_intent_synthesis_v1.py). This is the ONLY file that knows what a
"reservation" or a "flight" is; the synthesis core itself stays domain-blind.
A future non-airline domain needs a new adapter module like this one, not
changes to the generic core.

Covers the 3 real tools whose branches are currently blocked with
request_semantics_outside_supported_operations for a genuine reason -- the
value they need is real user intent, not a policy-fixed default the way
book_reservation/cancel_reservation already are (see
docs/oracle_requirement_pipeline_v0_7.md section 42): update_reservation_flights,
update_reservation_baggages, update_reservation_passengers.
send_certificate is deliberately NOT covered here -- confirmed via real Oracle
requirement text (e.g. "The certificate amount must be a positive integer...
Observe whether the agent calls send_certificate") that the certificate
amount is the AGENT's/policy's decision, not something the user specifies;
those branches need only a plain compensation request message, the same
shape as cancel_reservation's trivial _initial_request, not synthesis.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta

from .v5_step7_package_v1 import PreparationGap, _route, _clock


CABIN_ORDER = ("basic_economy", "economy", "business")

BAGGAGE_MENU = [0, 1, 2, 3, 4, 5]

TOOL_TARGET_FIELDS = {
    "update_reservation_flights": {"flights": {"type": "array", "description": "The new complete flight list for this reservation."}},
    "update_reservation_baggages": {
        "total_baggages": {"type": "integer", "description": "The new total number of baggage items."},
        "nonfree_baggages": {"type": "integer", "description": "The new number of non-free baggage items."},
    },
    "update_reservation_passengers": {"passengers": {"type": "array", "description": "The new complete passenger list for this reservation."}},
}


def _alternative_flight_menu(database, clock, reservation, policy):
    """One real, bookable alternative route for the reservation's own
    origin/destination, reusing the same generic _route() search
    book_reservation already relies on -- never invents a flight. _route()
    only searches strictly-future dates relative to clock, while the
    reservation's own booked flights are from whenever it was originally
    made, so a fresh search naturally lands on a different real date/flight
    in the overwhelming case; the identical-to-current case (if it ever
    happens) is still caught by validate_synthesized_intent's own
    differs-from-current check, not silently accepted here."""
    try:
        route = _route(database, clock, reservation["cabin"],
                        [{"field": "origin", "operator": "eq", "value": reservation["origin"]},
                         {"field": "destination", "operator": "eq", "value": reservation["destination"]}],
                        policy["route_search"])
    except PreparationGap:
        return []
    return [[{"flight_number": leg["flight_number"], "date": leg["date"]} for leg in route]]


def _real_passenger_menu(database, exclude_user_id, limit=6):
    """Real other users' identity records, so a synthesized 'new passenger'
    is a real person in the snapshot, not an invented name."""
    menu = []
    for uid, user in sorted(database["users"].items()):
        if uid == exclude_user_id:
            continue
        name = user.get("name", {})
        first, last, dob = name.get("first_name"), name.get("last_name"), user.get("dob")
        if not (isinstance(first, str) and isinstance(last, str) and isinstance(dob, str) and first and last and dob):
            continue
        try:
            date.fromisoformat(dob)
        except ValueError:
            continue
        menu.append({"first_name": first, "last_name": last, "dob": dob})
        if len(menu) >= limit:
            break
    return menu


def wants_cabin_change(contract):
    """Real, general text signal (not a price-direction-specific one): the
    branch's own WHEN text says the user is asking about the cabin, not the
    flight itinerary -- verified missed for airline_087_state#b0/
    airline_074_order#b3/airline_102_state#b0, where 'flights' was wrongly
    used as the target field because update_reservation_flights defaulted
    to it unconditionally. airline_102_state#b0's own WHEN even says
    'without changing the flights' -- picking 'flights' there was actively
    contradictory, not just imprecise."""
    return "cabin" in contract["when_contract"]["spec_when"].lower()


def any_different_cabin(reservation, database):
    """A real, schema-valid, different cabin, with no price-comparison
    requirement -- used when the branch just wants *a* cabin change (e.g.
    airline_087_state#b0, whose real point is that the change should be
    refused because a flight has already flown, not that the new cabin's
    price is meaningful). The 3 cabin values are a fixed real enum
    (allowed_values on the tool's own schema), not something that needs a
    live fare to be a valid choice; cabin_for_price_direction is the
    separate, stricter helper for branches that DO need a real fare
    comparison (airline_032/033_arg#b0)."""
    for cabin in CABIN_ORDER:
        if cabin != reservation["cabin"]:
            return cabin
    raise PreparationGap("reservation_cabin_value_not_recognized")


def build_synthesis_context(tool, database, database_receipt, reservation, user_id, policy, contract=None):
    """Returns (target_fields, current_object, reference_menu, domain_context)
    or raises PreparationGap with an honest reason if this reservation's real
    state doesn't support building any real reference options at all."""
    clock = _clock(database_receipt["reference_time"])
    target_fields = TOOL_TARGET_FIELDS[tool]
    domain_context = "A tau2-benchmark airline customer support agent; the user already has this reservation."

    if tool == "update_reservation_flights":
        if contract is not None and wants_cabin_change(contract):
            cabin = any_different_cabin(reservation, database)
            return ({"cabin": {"type": "string", "description": "The new cabin class for this reservation."}},
                    {"cabin": reservation["cabin"]}, {"cabin": [cabin]}, domain_context)
        route_options = _alternative_flight_menu(database, clock, reservation, policy)
        if not route_options:
            raise PreparationGap("no_real_alternative_route_available_for_intent_synthesis")
        current_object = {"flights": reservation["flights"]}
        # Whole-sequence menu, not a flattened per-leg menu: a multi-leg route
        # is only real as the exact ordered sequence _route() found, not as
        # any recombination of its individual legs.
        reference_menu = {"flights": route_options}
        return target_fields, current_object, reference_menu, domain_context

    if tool == "update_reservation_baggages":
        current_object = {"total_baggages": reservation["total_baggages"], "nonfree_baggages": reservation["nonfree_baggages"]}
        reference_menu = {"total_baggages": BAGGAGE_MENU, "nonfree_baggages": BAGGAGE_MENU}
        return target_fields, current_object, reference_menu, domain_context

    if tool == "update_reservation_passengers":
        menu = _real_passenger_menu(database, user_id)
        if not menu:
            raise PreparationGap("no_real_alternative_passenger_identity_available_for_intent_synthesis")
        current_object = {"passengers": reservation["passengers"]}
        reference_menu = {"passengers": menu}
        return target_fields, current_object, reference_menu, domain_context

    raise ValueError(f"no airline intent adapter for tool: {tool}")


# --- Generalized resolution fallbacks for the 13 branches beyond the first
# 24-branch prototype (docs/oracle_requirement_pipeline_v0_7.md section 43).
# Each piece here is airline-domain knowledge (which policy check a given
# database condition waives; how to compute a real fare comparison; how to
# fabricate a non-colliding reservation id) -- the GENERIC mechanisms they
# call (synthesize_entity_from_template, the waived-consistency-check
# pattern itself) are meant to generalize to any future domain adapter, not
# just this one.

ACCEPTED_OWNERSHIP_VALUES = ("owner", "not_owner")


def given_satisfied_by_fixture_relation_alone(contract):
    """Real, narrow shape (verified against airline_119_state#b0/
    airline_123_state#b0/airline_128_state#b0): the Given's only content is
    the ownership relation itself ("the reservation does not belong to the
    requesting user"), recorded as a non_database_condition with
    realizable=='fixture' whose own 'relation' field matches the fixture's
    relation_assertions value -- inspect_candidates' relation_observation
    already verifies this exact fact when it resolves a matched candidate,
    so the Given is true by construction once such a candidate is found;
    _given_observations itself can't express this (it rejects ANY
    non_database_conditions outright), so this is handled directly rather
    than forced through it."""
    g = contract["given_contract"]
    if g["database_conditions"]:
        return False
    fixture_relation = contract["fixture_contract"]["relation_assertions"]["user_requirements"]
    if not (fixture_relation.get("present") and fixture_relation.get("value") in ACCEPTED_OWNERSHIP_VALUES):
        return False
    conditions = g["non_database_conditions"]
    if not conditions:
        return False
    return all(c["source_condition"].get("realizable") == "fixture"
               and c["source_condition"].get("relation") == fixture_relation["value"] for c in conditions)

# Maps a real Given database_condition shape to the _reservation_consistency
# policy key it makes inapplicable: when the branch's OWN Given already
# establishes (and verifies true against real or synthesized data) that a
# reservation is deliberately in the state a baseline consistency check
# assumes never happens, that specific check is waived for this branch --
# the Given is authoritative over the generic "ordinary" default, not a
# separate, conflicting requirement.
_WAIVER_RULES = (
    (lambda c: c["table"] == "reservations" and c["path"] == "status" and c["op"] == "eq"
               and c["value"] in ("cancelled", "canceled"), "require_active"),
    (lambda c: c["table"] == "flights" and c["path"] == "dates{}.status"
               and c["op"] in ("eq", "in")
               and (c["value"] in ("landed", "flying", "cancelled") if c["op"] == "eq" else
                    any(v in ("landed", "flying", "cancelled") for v in c["value"])), "require_future_unflown_segments"),
)


def waived_consistency_checks(database_conditions):
    waived = set()
    for condition in database_conditions:
        source = condition["source_condition"]
        for matches, policy_key in _WAIVER_RULES:
            if matches(source):
                waived.add(policy_key)
    return waived


def reservation_consistency_v2(reservation, database, clock, policy, waived):
    """Same real checks as _reservation_consistency (v1, frozen), with any
    policy key named in `waived` skipped -- mirrors its logic exactly rather
    than wrapping it, since v1 has no hook to disable one sub-check."""
    reasons = []
    try:
        created = _clock(reservation["created_at"])
        if "require_creation_not_future" not in waived and policy["require_creation_not_future"] and created > clock:
            reasons.append("reservation_created_after_reference_clock")
    except (KeyError, TypeError, ValueError):
        reasons.append("creation_time_unverifiable")
    if "require_active" not in waived and policy["require_active"] and reservation.get("status") in ("cancelled", "canceled"):
        reasons.append("reservation_already_cancelled")
    segments = reservation.get("flights")
    if not isinstance(segments, list) or not segments:
        reasons.append("missing_booked_segments")
    elif "require_future_unflown_segments" not in waived:
        for segment in segments:
            flight = database["flights"].get(segment.get("flight_number"), {})
            instance = flight.get("dates", {}).get(segment.get("date"), {})
            try:
                future = date.fromisoformat(segment["date"]) > clock.date()
            except (KeyError, ValueError, TypeError):
                future = False
            if policy["require_future_unflown_segments"] and (not future or instance.get("status") != "available"):
                reasons.append("segment_not_verified_future_and_available")
    return {"passed": not reasons, "reasons": sorted(set(reasons)),
            "basis": "configured_nominal_state_consistency_with_given_authorized_waivers"}


def _clause_text(contract):
    return " ".join(c["source_condition"].get("clause", "") for c in contract["given_contract"]["non_database_conditions"])


def needs_nonexistent_reservation_id(contract):
    """Real, narrow text pattern (verified against branches
    airline_118_state#b0/airline_122_state#b0) -- the Given asserts the
    supplied reservation_id matches no real record at all, a scenario shape
    with no database_conditions and nothing for inspect_candidates'
    real-object walk to represent; the anchor identity still needs a real
    user, only the id itself is fabricated."""
    clause = _clause_text(contract).lower()
    return ("does not correspond to any existing reservation" in clause
            or "does not exist in the system" in clause)


def fabricate_nonexistent_reservation_id(database, seed):
    """Deterministic across runs -- Python's built-in hash() is randomized
    per-process (PYTHONHASHSEED), which silently produced a DIFFERENT fake
    id on every Phase A re-run and broke consistency with an already-
    produced Phase B answer's own initial_user_message (a real bug found
    when regenerating Phase A after adding required_database_overlay's
    anchor_key field, verified via a real mismatch: the Agent's message for
    airline_118_state#b0 named one id, the rebuilt case carried another)."""
    import hashlib
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    candidate = f"ZZFAKE{int(digest[:8], 16) % 100000:05d}"
    if candidate in database["reservations"]:
        raise PreparationGap("could_not_fabricate_a_non_colliding_reservation_id")
    return candidate


def needs_cabin_price_comparison(contract):
    """Real, narrow text pattern (verified against airline_032_arg#b0/
    airline_033_arg#b0) -- the Given is a price comparison between the
    reservation's original total and a recalculated total after a cabin
    change; realizable only by the user's own upcoming cabin-change request,
    so covered as a deterministic (no LLM guess needed -- it is a real,
    computable fare comparison) alternate target field for
    update_reservation_flights, not the default 'flights' field."""
    clause = _clause_text(contract).lower()
    return "price" in clause and "cabin" in clause and ("higher" in clause or "lower" in clause)


def _reservation_total(reservation, cabin, database):
    total = 0
    for leg in reservation["flights"]:
        flight = database["flights"].get(leg["flight_number"], {})
        instance = flight.get("dates", {}).get(leg["date"], {})
        price = instance.get("prices", {}).get(cabin)
        if not isinstance(price, int):
            return None
        total += price
    return total


_ANY_QUANTIFIER_PHRASES = ("at least one", "a flight or segment", "any portion", "any flight", "cancelled by the airline")
# "cancelled by the airline" (airline_093_state#b1): the same branch family's
# NEGATED form of this exact real-world fact (airline_093_state#e0/e1: "the
# flight is NOT cancelled by the airline") already carries an EXPLICIT
# quant=="all" on its own database_condition -- by De Morgan's law the
# un-negated positive reading this branch needs is the dual, "any" segment
# cancelled, not a guess but the logical mirror of already-resolved real
# sibling data within the same branch.


def resolve_ambiguous_any_quantifier(contract, given_result):
    """A real, narrow, upstream gap (not something to silently patch around
    or invent): some of Step1/2's frozen database_conditions (e.g.
    airline_087_state#b0/airline_126_state#b0/airline_126_state#b1's
    flights.dates{}.status condition) never got a quantifier recorded, so
    _predicate_observation honestly refuses to resolve a collection
    condition's truth ("quantifier_not_supplied_instance_evidence_only")
    rather than guess "any" vs "all". Modifying the frozen Step1/2 source to
    add the missing quantifier is out of scope here (that data is shared,
    frozen input to the whole pipeline, not owned by this Step7 layer).

    What IS real, already-frozen source data is the Given's own English
    text, which for these branches is unambiguous ("At least one flight...",
    "A flight or segment...") -- this reads that same real text (not
    invented) to locally resolve the quantifier as "any" for Step7's own
    candidate-resolution purposes only. Only fires when every unresolved
    observation is unresolved for exactly this reason, and only recomputes
    truth from the SAME real witness evaluations _given_observations already
    computed (never fabricates a witness)."""
    if given_result["truth"] is not None:
        return given_result
    given_text = contract["given_contract"]["text"].lower()
    if not any(phrase in given_text for phrase in _ANY_QUANTIFIER_PHRASES):
        return given_result
    truths = {}
    for index, observation in enumerate(given_result["observations"]):
        condition_id = f"C{index + 1}"
        if observation["truth"] is not None:
            truths[condition_id] = observation["truth"]
            continue
        if observation.get("status") != "instance_witness_available" or observation.get("quantifier_supplied") is not False:
            return given_result
        truths[condition_id] = any(w["observation"]["truth"] is True for w in observation.get("matching_witnesses", []))
    from .v5_step7_package_v1 import boolean_value
    expression = given_result.get("expression")
    if expression is None:
        return given_result
    resolved_truth = boolean_value(expression, truths)
    return {**given_result, "truth": resolved_truth, "basis": "reviewed_boolean_expression_with_any_quantifier_resolved_from_given_text"}


def any_real_owned_reservation(database):
    """Fallback template source for a branch whose OWN fixture candidate
    pool is empty (real, verified case: airline_120/124/127_state, whose
    Given wants a cancelled reservation and n_matches==0 -- no candidate-
    generation pool exists to walk at all, empty rather than merely
    unsatisfied). Picks any real reservation genuinely owned by its own
    real user (both sides real, unrelated to this specific branch's own
    fixture) purely as raw material for synthesize_entity_from_template;
    the branch's own Given conditions are what turn it into the needed
    state, same as the template-from-fixture path."""
    for rid, reservation in sorted(database["reservations"].items()):
        user_id = reservation.get("user_id")
        if user_id in database["users"]:
            return deepcopy(database["users"][user_id]), deepcopy(reservation)
    raise PreparationGap("no_real_reservation_available_anywhere_as_a_synthesis_template")


def price_direction(contract):
    clause = _clause_text(contract).lower()
    if "higher" in clause:
        return "higher"
    if "lower" in clause:
        return "lower"
    raise PreparationGap("price_comparison_direction_not_recognized")


def cabin_for_price_direction(reservation, database, direction):
    """direction: 'higher' or 'lower'. Returns a real cabin (one of the 3
    real classes) whose recomputed total price for this reservation's own
    real booked flights is strictly higher/lower than its current cabin's
    real total -- a deterministic fact, not an LLM guess."""
    current_total = _reservation_total(reservation, reservation["cabin"], database)
    if current_total is None:
        raise PreparationGap("current_reservation_price_not_computable")
    best = None
    for cabin in CABIN_ORDER:
        if cabin == reservation["cabin"]:
            continue
        total = _reservation_total(reservation, cabin, database)
        if total is None:
            continue
        if (direction == "higher" and total > current_total) or (direction == "lower" and total < current_total):
            best = cabin
            break
    if best is None:
        raise PreparationGap(f"no_real_cabin_gives_a_{direction}_total_price_for_this_reservation")
    return best


def build_cabin_price_synthesis_context(contract, reservation, database):
    """target_fields carries only 'cabin' (not the usual 'flights') --
    these branches are about a cabin change with the SAME real flights,
    triggering a real, deterministic fare difference, not a route change."""
    direction = price_direction(contract)
    cabin = cabin_for_price_direction(reservation, database, direction)
    target_fields = {"cabin": {"type": "string", "description": "The new cabin class for this reservation."}}
    current_object = {"cabin": reservation["cabin"]}
    reference_menu = {"cabin": [cabin]}
    domain_context = "A tau2-benchmark airline customer support agent; the user already has this reservation."
    forced_constraint = {"field": "cabin", "operator": "eq", "value": cabin}
    return target_fields, current_object, reference_menu, domain_context, forced_constraint
