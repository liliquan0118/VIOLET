"""Mechanical support for branches compile_package (v1, frozen) blocks with
request_semantics_outside_supported_operations because the real semantic
extraction honestly found tool==None -- not a classification failure, but 3
real, distinct shapes where no single tool genuinely determines the request:

- airline_085_order#b0: the request itself is deliberately abstract ("an
  action that requires a reservation id"), and the real Oracle requirement
  is about PROCESS (the agent should call get_user_details to help locate
  the reservation before doing whatever the user eventually asks for), not
  about any specific tool call. There is no "the" requested operation to
  synthesize -- private_operation_bundle is honestly left None, matching the
  send_certificate precedent (v5_step7_lookup_tools_adapter_v1.py):
  user_view(), the only thing Step8 actually reads, never looks at it.

- airline_088_state#b0: the user asks to add insurance to an ALREADY-BOOKED
  reservation. Real tau2 airline tools have no operation to modify insurance
  on an existing reservation at all (insurance is only ever set at
  book_reservation time) -- this is a genuine, verified capability gap, not
  a missing adapter. Confirmed via the branch's own oracle_contract: its
  requirement is compiled as requirement_type=="semantic_requirement" with
  "No exact tool, argument, message-literal, or temporal runtime anchor is
  present" -- Step6 itself already concluded this needs no Step8 tool-call
  binding. private_operation_bundle is honestly None for the same reason as
  085 above.

- airline_130_state#b0v0-v3: the request text itself lists 4 alternative
  actions ("update the flights, baggages, passengers, or cancel the
  reservation") -- a genuinely disjunctive real request, confirmed by each
  variant's own oracle_contract naming a DIFFERENT pair (or, for v3, single)
  of acceptable tool observations. There is no single canonical operation;
  private_operation_bundle is honestly None. The fixture's candidate pool is
  empty (n_matches==0, a "the reservation has been cancelled" Given with no
  real matching reservation) -- resolved via the same
  any_real_owned_reservation + synthesize_entity_from_template fallback
  already used for airline_106_state#b0/b2 (section 48) and
  airline_120/124/127_state (section 43).

See docs/oracle_requirement_pipeline_v0_7.md section 50.
"""
from __future__ import annotations

from .v5_object_references_v1 import inspect_candidates
from .v5_step7_airline_intent_adapter_v1 import any_real_owned_reservation
from .v5_step7_intent_synthesis_v1 import IntentSynthesisError, synthesize_entity_from_template
from .v5_step7_package_v1 import PreparationGap


def resolve_process_lookup_scenario(contract, store, database):
    fixture = contract["fixture_contract"]
    if fixture["root"] is not None:
        raise PreparationGap("unsupported_context_role_mapping")
    given = contract["given_contract"]
    if not (len(given["non_database_conditions"]) == 1 and not given["database_conditions"]
            and given["non_database_conditions"][0]["source_condition"].get("realizable") == "conversation"):
        raise PreparationGap("conditional_given_requires_separate_verification")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "not_applicable":
            continue
        user = store.read(candidate["user_reference"]["handle"])
        return candidate, user
    raise PreparationGap("no_real_identity_available_for_process_lookup_scenario")


def compose_process_lookup_message():
    return "I need help with something on my reservation, but I don't know my reservation ID."


def resolve_unsupported_capability_scenario(contract, store):
    fixture = contract["fixture_contract"]
    if fixture["root"] != "reservations" or fixture["lookup"]["lookup_status"] != "matched":
        raise PreparationGap("unsupported_context_role_mapping")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "matched":
            continue
        user = store.read(candidate["user_reference"]["handle"])
        reservation = store.read(candidate["root_reference"]["handle"])
        return candidate, user, reservation
    raise PreparationGap("no_supplied_existing_object_passes_given_and_nominal_consistency")


def compose_add_insurance_message(reservation):
    return f"I already have reservation {reservation['reservation_id']} booked -- can you add travel insurance to it now?"


def resolve_disjunctive_request_scenario(contract, database):
    fixture = contract["fixture_contract"]
    if fixture["root"] != "reservations" or fixture["lookup"]["n_matches"] != 0:
        raise PreparationGap("unsupported_context_role_mapping")
    conditions = [c["source_condition"] for c in contract["given_contract"]["database_conditions"]
                  if c["source_condition"]["table"] == "reservations"]
    if not conditions or contract["given_contract"]["non_database_conditions"]:
        raise PreparationGap("no_supplied_existing_object_passes_given_and_nominal_consistency")
    template_user, template_reservation = any_real_owned_reservation(database)
    try:
        synthesized, applied_fields = synthesize_entity_from_template(template_reservation, conditions)
    except IntentSynthesisError:
        raise PreparationGap("no_real_identity_anchor_available_for_synthesis") from None
    overlay = {"table": "reservations", "key": synthesized["reservation_id"], "fields": applied_fields,
               "reason": "no_real_reservation_in_database_satisfies_this_branch_given_state"}
    return template_user, synthesized, overlay


def compose_disjunctive_request_message(reservation):
    return (f"I'd like to update the flights, baggage, or passengers on reservation {reservation['reservation_id']} "
            "-- or if none of that's possible, please just cancel it.")
