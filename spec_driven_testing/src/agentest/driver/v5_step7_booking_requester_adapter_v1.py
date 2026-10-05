"""Mechanical support for book_reservation branches compile_package (v1,
frozen) blocks with requester_not_bound. The real root cause is upstream:
bind_creation_requesters' own _scope_hold (v5_requester_binding_v1.py,
frozen) is a narrower gate than compile_package's own book_reservation
dispatch actually needs -- it only accepts a LITERAL given["text"]=="True"
with root=="reservations" and lookup_status=="anchored", deferring (never
binding a requester) for any conditional Given or any other fixture shape,
even though compile_package's own book_reservation branch does its OWN
independent candidate walk via inspect_candidates and never actually reads
bind_creation_requesters' selected_candidate_index -- actors[bid]["status"]
is consulted only as an early gate. This adapter bypasses that gate and
performs the same real candidate resolution compile_package's own dispatch
would do, extended to the 3 real fixture shapes the 16 real branches in this
corpus need (verified against real data, not assumed):

- Category A (11 branches, airline_040-046_arg#b0/b1/b2): root=="reservations",
  lookup_status=="matched" -- the SUPPLIED candidate pool was already
  upstream-verified (empirically confirmed: all 20 real candidates per
  branch match membership+cabin, vs ~7% for a random sample) to satisfy the
  Given's own database_conditions (membership tier + reservation cabin
  class), a real airline_037/038-style membership/baggage-allowance policy
  matrix. _reservation_consistency's Given check doesn't need to be re-run
  through _given_observations (which can't be safely used here anyway,
  since compile_package always passes reservation=None into it for
  book_reservation); each condition is instead independently re-verified
  directly against the candidate's own real bound (user, reservation) pair
  -- never trusting the "matched" label alone. The Given's own cabin fact is
  then carried into the NEW booking as an explicit cabin constraint (the
  systematic tier x cabin sweep, and the cabin-dependent Oracle requirement
  text, both confirm the Given's cabin is describing the upcoming booking,
  not a fact about a pre-existing reservation the request never mentions).

- Category B (2 branches, airline_079/080_order#b0): root is None (an
  unscoped "any of the real users" pool, lookup_status=="no_conditions"),
  Given=="True" (unconditional) -- _scope_hold's SECOND check (root must be
  "reservations") is what actually blocks these, not the Given; they are
  otherwise completely ordinary book_reservation branches (empty
  constraints, testing agent PROCESS order, not booking content).

- Category C (3 branches, airline_100_state#b0/b1, airline_101_state#b0):
  root=="flights" -- the Given describes a specific real flight's current
  status (delayed/on time/flying), and the real Oracle requirement is a
  PROHIBITION ("flights with this status cannot be booked"). The flight's
  dated instance genuinely has no price/seat data while delayed/on-time/
  flying (only "available" dates carry pricing), so there is no honestly
  computable successful booking to synthesize -- private_operation_bundle
  is left None by design here, mirroring send_certificate's precedent
  (v5_step7_lookup_tools_adapter_v1.py): user_view(), the only thing Step8
  actually reads, never looks at private_operation_bundle at all.

See docs/oracle_requirement_pipeline_v0_7.md section 49.
"""
from __future__ import annotations

from .v5_object_references_v1 import inspect_candidates
from .v5_step7_package_v1 import PreparationGap, _booking


def _database_conditions_hold_for_pair(conditions, user, reservation):
    """Direct, honest re-verification against the candidate's OWN real bound
    (user, reservation) pair -- never trusts lookup_status=="matched" alone.
    Scoped exactly to the real shape observed (plain eq conditions on users
    or reservations); anything else is a genuine gap, not guessed past."""
    for c in conditions:
        source = c["source_condition"]
        if source["op"] != "eq":
            return False
        obj = user if source["table"] == "users" else reservation if source["table"] == "reservations" else None
        if obj is None or obj.get(source["path"]) != source["value"]:
            return False
    return True


def resolve_booking_requester_from_matched_context(contract, store, database, clock, request, policy):
    fixture = contract["fixture_contract"]
    given = contract["given_contract"]
    if fixture["root"] != "reservations" or fixture["lookup"]["lookup_status"] != "matched":
        raise PreparationGap("unsupported_context_role_mapping")
    assertions = fixture["relation_assertions"]
    if any(assertions[k] != {"present": True, "value": "owner"} for k in ("user_requirements", "condition_matches")):
        raise PreparationGap("owner_context_basis_missing_or_conflicting")
    if given["non_database_conditions"]:
        raise PreparationGap("non_database_given_requires_additional_adapter")
    conditions = given["database_conditions"]
    cabin_values = {c["source_condition"]["value"] for c in conditions
                     if c["source_condition"]["table"] == "reservations" and c["source_condition"]["path"] == "cabin"
                     and c["source_condition"]["op"] == "eq"}
    if len(cabin_values) > 1:
        raise PreparationGap("conflicting_cabin_conditions")
    augmented_request = {**request, "constraints": [*request["constraints"],
        *({"field": "cabin", "operator": "eq", "value": v} for v in cabin_values)]}
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "matched":
            continue
        user = store.read(candidate["user_reference"]["handle"])
        reservation = store.read(candidate["root_reference"]["handle"])
        if not _database_conditions_hold_for_pair(conditions, user, reservation):
            continue
        try:
            args, evidence = _booking(user, database, clock, augmented_request, policy)
        except PreparationGap:
            continue
        return candidate, user, args, evidence
    raise PreparationGap("no_supplied_actor_has_complete_supported_booking_facts")


def resolve_booking_requester_from_unscoped_pool(contract, store, database, clock, request, policy):
    fixture = contract["fixture_contract"]
    given = contract["given_contract"]
    if fixture["root"] is not None:
        raise PreparationGap("unsupported_context_role_mapping")
    if given["text"] != "True" or given["database_conditions"] or given["non_database_conditions"]:
        raise PreparationGap("conditional_given_requires_separate_verification")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "not_applicable":
            continue
        user = store.read(candidate["user_reference"]["handle"])
        try:
            args, evidence = _booking(user, database, clock, request, policy)
        except PreparationGap:
            continue
        return candidate, user, args, evidence
    raise PreparationGap("no_supplied_actor_has_complete_supported_booking_facts")


def _flagged_date_for_flight(flight, status):
    for day, instance in sorted(flight.get("dates", {}).items()):
        if instance.get("status") == status:
            return day
    return None


def resolve_booking_requester_for_flagged_flight(contract, store, database):
    """No _booking() call here -- the flagged flight has no price/seat data
    to book with while delayed/on-time/flying, and the real Oracle
    requirement is a prohibition, not a successful-booking check. Just needs
    a real requester identity and the real flagged flight/date."""
    fixture = contract["fixture_contract"]
    if fixture["root"] != "flights" or fixture["lookup"]["lookup_status"] != "matched":
        raise PreparationGap("unsupported_context_role_mapping")
    status_values = {c["source_condition"]["value"] for c in contract["given_contract"]["database_conditions"]
                      if c["source_condition"]["table"] == "flights" and c["source_condition"]["path"] == "dates{}.status"
                      and c["source_condition"]["op"] == "eq"}
    if len(status_values) != 1:
        raise PreparationGap("unsupported_flight_status_condition_shape")
    status = next(iter(status_values))
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "not_applicable":
            continue
        user = store.read(candidate["user_reference"]["handle"])
        flight = store.read(candidate["root_reference"]["handle"])
        passenger = {"first_name": user.get("name", {}).get("first_name"),
                     "last_name": user.get("name", {}).get("last_name"), "dob": user.get("dob")}
        if any(not isinstance(v, str) or not v.strip() for v in passenger.values()):
            continue
        date = _flagged_date_for_flight(flight, status)
        if date is None:
            continue
        return candidate, user, flight, date
    raise PreparationGap("no_real_identity_available_for_flagged_flight_scenario")


def compose_flagged_flight_request(flight, date):
    return (f"Please book a reservation including flight {flight['flight_number']} "
            f"from {flight['origin']} to {flight['destination']} on {date}.")
