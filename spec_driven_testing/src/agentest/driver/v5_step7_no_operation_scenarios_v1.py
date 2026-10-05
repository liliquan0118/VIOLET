"""Mechanical support for the remaining request_operation_not_determined
branches whose role classification (7C) honestly returned
lifecycle.decision=="not_determined" and whose real content genuinely has no
single tool call to synthesize -- informational/policy questions, scope
checks, and disjunctive book-or-modify requests. All of these leave
private_operation_bundle as None (user_view(), the only thing Step8 actually
reads, never looks at that field) except the two book-or-modify-baggage/
cancelled-reservation branches which reuse the exact disjunctive-request
entity-synthesis mechanism already built in
v5_step7_undetermined_operation_adapter_v1.py (section 50) for the analogous
airline_130_state#b0v0-v3 branches.

Grounding for the interpretive branches (075, 076v0/v1, 078v0/v1) was pulled
directly from the real compiled policy_statement text in
outputs/step4_support_files_v0_1/semantic_unit_model.json (semantic_unit_id
CA0007::A01 for 076, CA0003::A01/A02 for 078) rather than guessed -- see
docs/oracle_requirement_pipeline_v0_7.md section 52 for the exact quotes.

See docs/oracle_requirement_pipeline_v0_7.md section 52.
"""
from __future__ import annotations

from .v5_object_references_v1 import inspect_candidates
from .v5_step7_package_v1 import PreparationGap


def resolve_any_real_user(contract, store):
    fixture = contract["fixture_contract"]
    if fixture["root"] is not None:
        raise PreparationGap("unsupported_context_role_mapping")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "not_applicable":
            continue
        return store.read(candidate["user_reference"]["handle"])
    raise PreparationGap("no_real_identity_available")


def resolve_reservation_context_user(contract, store):
    """root=='reservations', owner relation -- a real reservation is
    supplied purely as an identity anchor (establishing a real, ordinary
    customer), not as a target the request itself needs to reference."""
    fixture = contract["fixture_contract"]
    if fixture["root"] != "reservations":
        raise PreparationGap("unsupported_context_role_mapping")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "matched":
            continue
        return store.read(candidate["user_reference"]["handle"])
    raise PreparationGap("no_real_identity_available")


def resolve_self_owned_user(contract, store):
    """root=='users', self-referential (user_id==root_id) candidate pairs."""
    fixture = contract["fixture_contract"]
    if fixture["root"] != "users":
        raise PreparationGap("unsupported_context_role_mapping")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved":
            continue
        return store.read(candidate["user_reference"]["handle"])
    raise PreparationGap("no_real_user_available")


def compose_vague_process_message():
    return "I need some help with my booking, or I have a question about my flight."


def compose_insurance_cost_and_refund_message():
    return ("How much does travel insurance cost, and what does it cover? Also, if I need to cancel a flight "
            "for health or weather reasons and I have travel insurance, am I eligible for a refund?")


def compose_baggage_price_message():
    return "How much does it cost to add extra baggage when I book a reservation?"


def compose_generic_policy_violation_message():
    return "I already booked my flight -- can you add travel insurance to it now?"


def compose_out_of_scope_message():
    return "Can you help me book a hotel room for my upcoming trip?"


def compose_in_scope_vague_message():
    return "I have a question about my upcoming reservation."


def compose_advice_seeking_message():
    return "What's the best time of year to visit Paris, and do I need a visa to travel there?"


def compose_complaint_without_compensation_message():
    return "My flight was delayed by three hours and it really disrupted my plans."


def compose_baggage_disjunctive_message():
    return "I'd like to book a new reservation, or if I already have one, please make sure it has no checked bags."


def compose_cancelled_reservation_disjunctive_message(reservation):
    return (f"I'd like to book a new reservation, or make some changes referencing my cancelled "
            f"reservation {reservation['reservation_id']}.")
