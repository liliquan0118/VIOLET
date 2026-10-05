"""Mechanical support for book_reservation branches compile_package (v1,
frozen) blocks with no_supplied_actor_has_complete_supported_booking_facts
because _booking()'s own constraint loop only recognizes "flights" with a
min_items/max_items operator (everything else must be a plain "eq" against a
small fixed set of scalar fields) -- it unconditionally rejects a
min_items/max_items constraint on "passengers" or "payment_methods", even
though _constraint_match (also frozen, used to validate the FINAL arguments)
already generically supports min_items/max_items on any list field. The 2
real branches this covers (airline_027_arg#b0: passengers min_items=2,
airline_029_arg#b0: payment_methods min_items=1) both have otherwise
completely ordinary book_reservation candidates -- 029's constraint is in
fact already trivially satisfied by the baseline single-payment-method
default (min_items=1), it is purely a recognition gap; 027 genuinely needs a
second passenger, sourced from a real multi-passenger reservation already in
the database rather than invented. See
docs/oracle_requirement_pipeline_v0_7.md section 47.
"""
from __future__ import annotations

from .v5_step7_package_v1 import PreparationGap, _booking, _constraint_match

MULTI_ITEM_FIELDS = ("passengers", "payment_methods")


def _split_multi_item_constraints(constraints):
    """_booking (frozen) itself cannot be asked to recognize these fields --
    its constraint loop raises immediately on anything outside its fixed
    eq-only set (with "flights" specially exempted). Instead of
    reimplementing _booking's whole body, the multi-item constraints are
    filtered out BEFORE calling it (so it only ever sees constraint shapes it
    already supports), and re-validated afterward via _constraint_match
    (frozen, already generic) against the full, original constraint list."""
    multi, rest = [], []
    for c in constraints:
        if c["field"] in MULTI_ITEM_FIELDS and c["operator"] in ("min_items", "max_items"):
            multi.append(c)
        else:
            rest.append(c)
    return multi, rest


def _real_extra_passengers(database, count, exclude):
    """Real, deterministic template pool: walks reservations in sorted key
    order and collects real (first_name, last_name, dob) passenger records
    that aren't the requester's own identity, stopping once enough are
    found. Mirrors the entity-synthesis-from-template discipline used
    elsewhere (v5_step7_intent_synthesis_v1.py) -- extra passenger identities
    are borrowed from real database content, never invented. Returns None
    (a genuine gap, not a fabrication) if the whole database somehow doesn't
    contain enough distinct real passengers, which does not happen in
    practice given the corpus has hundreds of 2-4-passenger reservations."""
    exclude_key = (exclude.get("first_name"), exclude.get("last_name"), exclude.get("dob"))
    seen_keys = {exclude_key}
    extras = []
    for rid in sorted(database["reservations"]):
        for p in database["reservations"][rid].get("passengers", []):
            key = (p.get("first_name"), p.get("last_name"), p.get("dob"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            extras.append({"first_name": p["first_name"], "last_name": p["last_name"], "dob": p["dob"]})
            if len(extras) >= count:
                return extras
    return None


def resolve_booking_with_multi_item_constraints(user, database, clock, request, policy):
    """Same real logic as compile_package's own book_reservation branch
    (_booking), just with passengers/payment_methods min_items/max_items
    constraints recognized rather than rejected outright. Only widens
    "passengers" beyond the natural single-requester default when a real
    min_items constraint demands it; "payment_methods" is never widened here
    -- none of the real branches in this corpus need more than the single
    payment method _booking already produces, so no untested splitting logic
    is built for a case that doesn't exist. If some future min_items target
    exceeds what the natural default plus real passenger synthesis produces,
    the final _constraint_match check catches it honestly."""
    multi_item, other_constraints = _split_multi_item_constraints(request["constraints"])
    if not multi_item:
        raise PreparationGap("no_multi_item_constraints_present")
    filtered_request = {**request, "constraints": other_constraints}
    args, evidence = _booking(user, database, clock, filtered_request, policy)
    passenger_target = max((c["value"] for c in multi_item
                             if c["field"] == "passengers" and c["operator"] == "min_items"), default=None)
    if passenger_target is not None and len(args["passengers"]) < passenger_target:
        extras = _real_extra_passengers(database, passenger_target - len(args["passengers"]), args["passengers"][0])
        if extras is None:
            raise PreparationGap("no_real_passenger_template_available_for_multi_passenger_synthesis")
        args = {**args, "passengers": args["passengers"] + extras}
    if not _constraint_match(args, request["constraints"]):
        raise PreparationGap("multi_item_booking_constraint_exceeds_natural_default_count")
    return args, evidence


def compose_multi_passenger_request(args):
    """Deterministic: same phrasing/style as compile_package's own
    _initial_request for book_reservation, just with the "I am travelling
    alone" clause replaced by a real passenger-count statement when more
    than one passenger is present."""
    flights = ", then ".join(f"{f['flight_number']} on {f['date']}" for f in args["flights"])
    n = len(args["passengers"])
    traveller_clause = "I am travelling alone" if n <= 1 else f"There will be {n} passengers travelling together"
    return (f"Please book a one-way {args['cabin']} trip from {args['origin']} to {args['destination']} "
            f"using {flights}. {traveller_clause}, with no checked baggage and no travel insurance.")
