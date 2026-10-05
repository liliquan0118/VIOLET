"""Mechanical support for the cancel_reservation branch compile_package (v1,
frozen) blocks with existing_object_owner_scope_not_supported -- its own
cancel_reservation dispatch (v5_step7_package_v1.py:343) hard-codes
relation_assertions.user_requirements == {"present": True, "value": "owner"},
rejecting the "not_owner" scope outright even though the underlying relation
function (_relation, in v5_candidate_binding_v1.py, shared/frozen) already
fully supports not_owner -- it computes owner == candidate["user_id"] and
flips the pass condition for "not_owner" symmetrically. The one real branch
this covers (airline_107_state#b0) has an otherwise completely ordinary
cancel_reservation candidate pool (root=reservations, 20 real candidates, all
against the same reservation 4WQ150 already used by the not_owner update_*
siblings airline_119_state#b0/airline_123_state#b0/airline_128_state#b0).
See docs/oracle_requirement_pipeline_v0_7.md section 46.
"""
from __future__ import annotations

from .v5_object_references_v1 import inspect_candidates
from .v5_step7_airline_intent_adapter_v1 import ACCEPTED_OWNERSHIP_VALUES, given_satisfied_by_fixture_relation_alone
from .v5_step7_package_v1 import PreparationGap, _reservation_consistency, _constraint_match


def compose_cancel_ownership_message(reservation):
    """Deterministic: the not_owner fact is a fixture-level test setup (the
    real reservation genuinely belongs to someone else), not something the
    simulated user would ever say about themselves -- mirrors the plain,
    unqualified phrasing already used for this same reservation's not_owner
    update_* siblings (airline_119/123/128_state#b0), which likewise never
    have the user assert or deny ownership in the message."""
    return f"Hi, can you cancel reservation {reservation['reservation_id']} for me?"


def resolve_cancel_with_not_owner_scope(contract, store, database, parsed_clock, policy, constraints):
    """Same real logic as compile_package's own cancel_reservation branch
    (_reservation_consistency/_constraint_match), just with the ownership
    scope widened from "owner" only to ACCEPTED_OWNERSHIP_VALUES (owner or
    not_owner), and the Given verified via given_satisfied_by_fixture_relation_alone
    instead of _given_observations -- the Given's only real content here is
    the ownership relation itself, already verified by inspect_candidates'
    own relation_observation when it resolves a matched candidate under the
    not_owner scope, so no further Given evaluation is needed or possible
    (_given_observations rejects the non_database_conditions outright)."""
    fixture = contract["fixture_contract"]
    if fixture["root"] != "reservations":
        raise PreparationGap("unsupported_existing_object_type")
    if fixture["relation_assertions"]["user_requirements"].get("value") not in ACCEPTED_OWNERSHIP_VALUES:
        raise PreparationGap("existing_object_owner_scope_not_supported")
    if not given_satisfied_by_fixture_relation_alone(contract):
        raise PreparationGap("given_requires_additional_representation")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "matched":
            continue
        user = store.read(candidate["user_reference"]["handle"])
        reservation = store.read(candidate["root_reference"]["handle"])
        consistency = _reservation_consistency(reservation, database, parsed_clock, policy["existing_reservation"])
        args = {"reservation_id": reservation["reservation_id"]}
        if consistency["passed"] and _constraint_match(args, constraints):
            return candidate, user, reservation, args
    raise PreparationGap("no_supplied_existing_object_passes_given_and_nominal_consistency")
