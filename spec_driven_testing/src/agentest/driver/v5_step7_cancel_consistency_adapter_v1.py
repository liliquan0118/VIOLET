"""Mechanical support for cancel_reservation branches compile_package (v1,
frozen) blocks with no_supplied_existing_object_passes_given_and_nominal_
consistency because its own baseline _reservation_consistency has no way to
know a branch's Given DELIBERATELY puts the reservation in a state the
baseline policy assumes never happens (an already-flown/cancelled flight
segment, or a reservation that's already cancelled). Real, narrow shape
(verified against the 6 real branches this covers): every one of them is an
otherwise completely ordinary cancel_reservation branch (root=reservations,
owner) whose ONLY real gap is that baseline consistency check.

Two real candidate-pool shapes:
- airline_092_state#b0v0/v1/v2 (Given: "any portion of the flight has
  already been flown", database_conditions status in landed/flying) and
  airline_093_state#b1 (Given: "the flight is cancelled by the airline",
  status eq cancelled) all have 20 real supplied candidates each -- every one
  fails only the baseline require_future_unflown_segments check, which
  waived_consistency_checks/reservation_consistency_v2 (built in section 43
  for the update_* intent-synthesis branches) already recognizes and waives
  for exactly this database_condition shape.
- airline_106_state#b0/b2 (Given: reservation already cancelled/voided) have
  an EMPTY real candidate pool (fixture_contract.lookup.n_matches == 0,
  lookup_status "makeable") -- no real cancelled reservation exists in the
  database matching this fixture's other requirements at all, so this
  branch needs the generic entity-synthesis-from-template fallback (same
  mechanism, same discipline, as airline_120/124/127_state's update_*
  branches: any_real_owned_reservation + synthesize_entity_from_template),
  applied here to cancel_reservation instead of an update_* tool.

See docs/oracle_requirement_pipeline_v0_7.md section 48.
"""
from __future__ import annotations

from .v5_object_references_v1 import inspect_candidates
from .v5_step7_airline_intent_adapter_v1 import (
    any_real_owned_reservation, reservation_consistency_v2, resolve_ambiguous_any_quantifier, waived_consistency_checks,
)
from .v5_step7_intent_synthesis_v1 import IntentSynthesisError, synthesize_entity_from_template
from .v5_step7_package_v1 import PreparationGap, _given_observations, _constraint_match


def resolve_cancel_with_waived_consistency(contract, store, database, parsed_clock, raw_clock, policy, semantic_given, constraints):
    """Stage 1: walk the real supplied candidate pool with the baseline
    consistency check widened by waived_consistency_checks (reused as-is --
    the exact same real Given database_condition shapes it was already built
    to recognize). Stage 2: if the fixture's own candidate pool is genuinely
    empty (not merely unsatisfied), fall back to synthesizing a reservation
    from a real global template and this branch's own database_conditions,
    mirroring synthesize_entity_from_template's already-established use for
    update_* branches in the same situation."""
    fixture = contract["fixture_contract"]
    if fixture["root"] != "reservations":
        raise PreparationGap("unsupported_existing_object_type")
    if fixture["relation_assertions"]["user_requirements"] != {"present": True, "value": "owner"}:
        raise PreparationGap("existing_object_owner_scope_not_supported")
    waived = waived_consistency_checks(contract["given_contract"]["database_conditions"])

    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "matched":
            continue
        user = store.read(candidate["user_reference"]["handle"])
        reservation = store.read(candidate["root_reference"]["handle"])
        consistency = reservation_consistency_v2(reservation, database, parsed_clock, policy["existing_reservation"], waived)
        if not consistency["passed"]:
            continue
        given = _given_observations(contract, candidate, user, reservation, database, raw_clock, semantic_given)
        given = resolve_ambiguous_any_quantifier(contract, given)
        args = {"reservation_id": reservation["reservation_id"]}
        if given["truth"] is True and _constraint_match(args, constraints):
            return candidate, user, reservation, args, None

    if fixture["lookup"]["n_matches"] != 0:
        raise PreparationGap("no_supplied_existing_object_passes_given_and_nominal_consistency")

    template_user, template_reservation = any_real_owned_reservation(database)
    reservation_conditions = [c["source_condition"] for c in contract["given_contract"]["database_conditions"]
                               if c["source_condition"]["table"] == "reservations"]
    if not reservation_conditions or contract["given_contract"]["non_database_conditions"]:
        raise PreparationGap("no_supplied_existing_object_passes_given_and_nominal_consistency")
    try:
        synthesized, applied_fields = synthesize_entity_from_template(template_reservation, reservation_conditions)
    except IntentSynthesisError:
        raise PreparationGap("no_real_identity_anchor_available_for_synthesis") from None
    consistency = reservation_consistency_v2(synthesized, database, parsed_clock, policy["existing_reservation"], waived)
    given = _given_observations(contract, None, template_user, synthesized, database, raw_clock, semantic_given)
    given = resolve_ambiguous_any_quantifier(contract, given)
    args = {"reservation_id": synthesized["reservation_id"]}
    if consistency["passed"] and given["truth"] is True and _constraint_match(args, constraints):
        overlay = {"table": "reservations", "key": synthesized["reservation_id"], "fields": applied_fields,
                   "reason": "no_real_reservation_in_database_satisfies_this_branch_given_state"}
        candidate_stub = {"candidate_index": None,
                           "user_reference": {"handle": None}, "root_reference": {"handle": None}}
        return candidate_stub, template_user, synthesized, args, overlay
    raise PreparationGap("no_supplied_existing_object_passes_given_and_nominal_consistency")
