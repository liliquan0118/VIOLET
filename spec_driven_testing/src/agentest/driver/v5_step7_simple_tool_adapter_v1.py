"""Mechanical support for branches compile_package (v1, frozen) blocks with
request_operation_not_determined because their role classification (7C,
already correctly run via the Agent tool in section 41) honestly returned
lifecycle.decision=="not_determined" -- these requests are not about
book_reservation/cancel_reservation at all, so no request-semantics job was
ever dispatched for them. Real investigation of all 33 such branches found
they split into several tool shapes; this module covers the ones needing a
real, schema-valid tool call compile_package itself has no path for at all
(calculate, search_direct_flight, search_onestop_flight,
transfer_to_human_agents, get_flight_status, get_user_details) -- both their
ordinary and deliberate-edge-case (out-of-network destination, far-out-of-
range date, nonexistent flight or user id) real branches. See
docs/oracle_requirement_pipeline_v0_7.md section 51.
"""
from __future__ import annotations

from hashlib import sha256

from .v5_object_references_v1 import inspect_candidates
from .v5_step7_package_v1 import PreparationGap

REFERENCE_DATE = "2024-05-15"
UNSERVED_DESTINATION = "IAD"  # real-world airport code, verified absent from this database's served set


def resolve_generic_identity(contract, store):
    """Any real user, for root==None fixtures (a flat, unscoped pool)."""
    fixture = contract["fixture_contract"]
    if fixture["root"] is not None:
        raise PreparationGap("unsupported_context_role_mapping")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "not_applicable":
            continue
        return store.read(candidate["user_reference"]["handle"])
    raise PreparationGap("no_real_identity_available")


def resolve_flight_anchor(contract, store):
    """A real (user, flight) pair, for root=='flights' fixtures -- both
    roles come from the SAME single fixture/candidate, there is no separate
    identity-only fixture to resolve the requester from."""
    fixture = contract["fixture_contract"]
    if fixture["root"] != "flights":
        raise PreparationGap("unsupported_context_role_mapping")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved":
            continue
        user = store.read(candidate["user_reference"]["handle"])
        flight = store.read(candidate["root_reference"]["handle"])
        return user, flight
    raise PreparationGap("no_real_flight_available")


def resolve_user_anchor(contract, store):
    """A real user, for root=='users' fixtures (self-referential candidate
    pairs: user_id==root_id, so the owner relation is trivially satisfied)."""
    fixture = contract["fixture_contract"]
    if fixture["root"] != "users":
        raise PreparationGap("unsupported_context_role_mapping")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved":
            continue
        return store.read(candidate["user_reference"]["handle"])
    raise PreparationGap("no_real_user_available")


def fabricate_nonexistent_user_id(database, seed):
    """Same discipline as fabricate_nonexistent_reservation_id
    (v5_step7_airline_intent_adapter_v1.py): a deterministic (hashlib, not
    hash()) fake id, distinguishable from any real one and provably absent
    from the database, re-derived identically on every Phase A re-run."""
    for attempt in range(1000):
        candidate = "zzfake_" + sha256(f"{seed}:{attempt}".encode()).hexdigest()[:12]
        if candidate not in database["users"]:
            return candidate
    raise PreparationGap("no_nonexistent_user_id_found")


# -- calculate ---------------------------------------------------------

def compose_calculate_bundle(kind):
    expressions = {"plain": "12 * 3 + 7", "no_sensitive_data": "45 + 55 / 5",
                   "no_division_by_zero": "8 + 4 * 2"}
    expression = expressions[kind]
    return {"tool_name": "calculate", "arguments": {"expression": expression}}, \
        f"Can you calculate {expression} for me?"


# -- search_direct_flight / search_onestop_flight -----------------------

def compose_search_direct_bundle(origin, destination, date):
    args = {"origin": origin, "destination": destination, "date": date}
    return {"tool_name": "search_direct_flight", "arguments": args}, \
        f"Can you search for a direct flight from {origin} to {destination} on {date}?"


def compose_search_onestop_bundle(origin, destination, date):
    args = {"origin": origin, "destination": destination, "date": date}
    return {"tool_name": "search_onestop_flight", "arguments": args}, \
        f"Can you search for a one-stop flight from {origin} to {destination} on {date}?"


# -- transfer_to_human_agents --------------------------------------------

def compose_transfer_bundle():
    args = {"summary": "The user is asking to speak with a human agent."}
    return {"tool_name": "transfer_to_human_agents", "arguments": args}, \
        "I'd like to speak with a human agent, please."


# -- get_flight_status ----------------------------------------------------

def compose_flight_status_bundle(flight_number, date):
    args = {"flight_number": flight_number, "date": date}
    return {"tool_name": "get_flight_status", "arguments": args}, \
        f"What's the status of flight {flight_number} on {date}?"


# -- get_user_details -------------------------------------------------------

def compose_user_details_bundle(user_id):
    args = {"user_id": user_id}
    return {"tool_name": "get_user_details", "arguments": args}, \
        f"Can you pull up the account details for user {user_id}?"
