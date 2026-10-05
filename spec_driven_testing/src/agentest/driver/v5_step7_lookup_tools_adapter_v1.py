"""Mechanical support for tools that need NO value synthesis at all --
send_certificate (compensation amount is the agent's/policy's decision, not
user-supplied data) and pure lookup tools (get_reservation_details,
get_user_details: the user just needs to reference a real object, there is
no "new value" to invent). See docs/oracle_requirement_pipeline_v0_7.md
section 44.

Unlike v5_step7_airline_intent_adapter_v1.py, this module never calls the
Agent tool -- every message here is a fixed, deterministic template (the
same discipline compile_package's own cancel_reservation path already uses
for _initial_request), because there is no real interpretive judgment
needed: the user's ask is a plain, ordinary request with no field values to
decide between.
"""
from __future__ import annotations

from copy import deepcopy

from ..compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .v5_object_references_v1 import SnapshotObjects, inspect_candidates
from .v5_step7_package_v1 import PreparationGap, _clock, _given_observations
from .v5_step7_airline_intent_adapter_v1 import (
    ACCEPTED_OWNERSHIP_VALUES, waived_consistency_checks, reservation_consistency_v2,
    resolve_ambiguous_any_quantifier, given_satisfied_by_fixture_relation_alone,
    needs_nonexistent_reservation_id, fabricate_nonexistent_reservation_id,
)

# (branch_id -> real tool name) manual override table for the same reason
# oracle_requirement_pipeline_v7.py's own _TOOL_CALL_FALLBACK_TARGETS exists:
# lexical extraction over request/when text can't find a tool the Oracle
# requirement text itself names explicitly. Here the constraint is stricter
# still -- the semantics-extraction task's own evidence rule requires
# tool_evidence.quote to be an exact substring of the request/when text, and
# neither branch's request/when text ever says "certificate" at all (only
# "complains about cancelled/delayed flights"), so no valid answer could
# ever satisfy that schema for these 2 branches. The real target tool is
# confirmed directly from each branch's own real Oracle requirement text
# (verified, not guessed): "...you can offer a certificate as a gesture
# after confirming the facts... (send_certificate)".
TOOL_RECLASSIFICATION = {
    "airline_037_arg#b0": "send_certificate",
    "airline_038_arg#b0": "send_certificate",
}

MESSAGE_TEMPLATES = {
    "send_certificate": "I'd like to request compensation for an issue with my flight.",
    "get_reservation_details": "Can you pull up the details of my reservation?",
    "get_user_details": "Can you look up my account details?",
}


def _resolved_matched(fixture, store):
    # For root is None fixtures (most send_certificate branches: identity
    # only, no target object at all), inspect_candidates itself reports
    # relation_observation.status=="not_applicable" -- there is genuinely
    # nothing to compare ownership against, which is a real pass, not a
    # failure to match.
    ok_statuses = {"not_applicable"} if fixture["root"] is None else {"matched"}
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] == "resolved" and candidate["relation_observation"]["status"] in ok_statuses:
            yield candidate


def given_is_trivial_conversational_restatement(contract):
    """Real, narrow shape (verified against airline_037/038_arg#b0/e0/e1/e2):
    every non_database_condition is realizable=='conversation' -- a
    dialogue-act fact (the user did or did not complain/ask to cancel, in
    some combination) that Step7's own message construction is responsible
    for embodying, not something _given_observations can or should try to
    verify against stored data. compose_certificate_message reads each
    clause's real polarity (does it contain a negation) to build a message
    consistent with the SPECIFIC branch's own combination -- this function
    only confirms the shape is the "purely conversational" one, it doesn't
    itself require the clause text to literally restate the WHEN (several
    real branches, e.g. the airline_038_arg#e0/e1/e2 trio, share one WHEN
    but differ only in these clauses, so message content has to come from
    the clauses themselves, not a WHEN-text match)."""
    g = contract["given_contract"]
    if g["database_conditions"] or not g["non_database_conditions"]:
        return False
    return all(c["source_condition"].get("realizable") == "conversation" for c in g["non_database_conditions"])


_NEGATION_MARKERS = (" not ", " does not ", " has not ", " do not ", " did not ")


def _clause_is_negative(clause):
    padded = f" {clause.lower()} "
    return any(marker in padded for marker in _NEGATION_MARKERS)


def compose_certificate_message(contract):
    """Deterministic (no LLM needed -- these are two flat booleans, not open
    interpretation): reads each realizable=='conversation' clause's real
    polarity to decide whether to mention complaining and/or wanting to
    change or cancel, matching exactly what that branch's own Given asserts
    -- verified against the real airline_037/038_arg#b0/e0/e1/e2 clause set
    (complaint status and change/cancel-want are the only two topics that
    appear across all of them)."""
    complains = wants_change = None
    for c in contract["given_contract"]["non_database_conditions"]:
        clause = c["source_condition"]["clause"].lower()
        negative = _clause_is_negative(clause)
        if "complain" in clause:
            complains = not negative
        elif "change" in clause or "cancel" in clause:
            wants_change = not negative
    parts = []
    if complains is True:
        parts.append("I wanted to flag an issue with my flight (it was delayed or cancelled)")
    elif complains is False:
        parts.append("I haven't filed a complaint, but I'd like to ask about compensation")
    else:
        parts.append("I'd like to request compensation for an issue with my flight")
    if wants_change is True:
        parts.append("and I'd also like to change or cancel my reservation")
    return ", ".join(parts) + "."


def users_root_mixed_table_given_satisfied(contract, user, database):
    """Only handles the real shape actually observed (verified against all
    4 real branches this covers): a plain conjunction (or a single
    condition) of 'users'-table and 'reservations'-table equality
    conditions, no OR/NOT nesting -- raises rather than silently guessing if
    a branch's real semantic expression isn't that shape, since a
    disjunction here would need picking which specific condition combination
    the scenario should embody, a real judgment call this deterministic
    check does not make."""
    g = contract["given_contract"]
    if g["non_database_conditions"]:
        return False
    conditions = [c["source_condition"] for c in g["database_conditions"]]
    if not conditions:
        return False
    if not all(c["op"] == "eq" for c in conditions):
        return False
    users_conditions = [c for c in conditions if c["table"] == "users"]
    reservations_conditions = [c for c in conditions if c["table"] == "reservations"]
    if len(users_conditions) + len(reservations_conditions) != len(conditions):
        return False  # some other table involved -- not this shape
    if not all(user.get(c["path"]) == c["value"] for c in users_conditions):
        return False
    if reservations_conditions and not _users_root_reservation_conditions_satisfied(user, database, reservations_conditions):
        return False
    return True


def _users_root_reservation_conditions_satisfied(user, database, conditions):
    """Real, narrow structural gap (verified against airline_096_state#b0/
    airline_096_state#b1/airline_098_state#b2/airline_098_state#b3): when a
    fixture's root is 'users' but the Given ALSO has real database
    conditions on the 'reservations' table (e.g. "the user... flies
    economy"), the frozen _predicate_observation has no support at all for
    this table/root combination (it only supports root=='reservations'
    conditions on 'users', the reverse direction) -- and there is no
    reservation bound in the (user_id, root_id) pair anyway, since root IS
    the user record itself. This directly, honestly checks the SAME real
    equality conditions against each of the user's own real reservations
    (a real airline user commonly owns several), returning True as soon as
    one real reservation satisfies every reservation-table condition; never
    guesses which reservation, never fabricates one."""
    owned = [r for rid, r in sorted(database["reservations"].items()) if r.get("user_id") == user["user_id"]]
    for reservation in owned:
        if all(reservation.get(c["path"]) == c["value"] for c in conditions if c["op"] == "eq"):
            return True
    return False


def resolve_lookup_or_certificate_candidate(contract, store, database, clock, policy, semantic_given):
    """Returns (candidate, user, root_object_or_None, overlay_or_None).
    root is the fixture's own root object read via inspect_candidates -- a
    reservation for root=='reservations', a user record for root=='users'
    (same object as the requester in the common self-referential case), or
    None for root is None (identity-only fixtures, e.g. most
    send_certificate branches -- no specific object to bind at all).
    overlay is the section-43.1-style required_database_overlay record when
    root_object was synthesized from a real template rather than resolved
    from a real candidate, else None. candidate is {"candidate_index": None}
    for the synthesis-fallback and nonexistent-id paths, where no real
    fixture candidate index applies.

    Reuses the SAME special-case detectors already built and verified for
    the update_* intent-synthesis tools (not_owner via fixture relation, a
    reservation_id required to provably not exist, and the generic entity-
    synthesis fallback) rather than re-deriving narrower versions -- all
    three real shapes recur here unchanged (airline_109_state#b0/
    airline_108_state#b0/airline_131_state#b0)."""
    fixture = contract["fixture_contract"]
    if fixture["root"] not in (None, "reservations", "users"):
        raise PreparationGap("unsupported_existing_object_type")
    if fixture["relation_assertions"]["user_requirements"].get("value") not in ACCEPTED_OWNERSHIP_VALUES:
        raise PreparationGap("existing_object_owner_scope_not_supported")

    if needs_nonexistent_reservation_id(contract):
        for candidate in _resolved_matched(fixture, store):
            user = store.read(candidate["user_reference"]["handle"])
            anchor = store.read(candidate["root_reference"]["handle"])
            fake_id = fabricate_nonexistent_reservation_id(database, contract["branch_id"])
            overlay = {"table": "reservations", "key": fake_id, "fields": None, "anchor_key": anchor["reservation_id"],
                "reason": "reservation_id chosen to provably not exist in the real snapshot; "
                          "the Given itself requires this id to correspond to no real reservation"}
            return candidate, user, {**anchor, "reservation_id": fake_id}, overlay
        raise PreparationGap("no_real_identity_anchor_available_for_nonexistent_id_scenario")

    relation_only = given_satisfied_by_fixture_relation_alone(contract)
    trivial_given = relation_only or given_is_trivial_conversational_restatement(contract)
    waived = waived_consistency_checks(contract["given_contract"]["database_conditions"])

    for candidate in _resolved_matched(fixture, store):
        user = store.read(candidate["user_reference"]["handle"])
        # root_reference.handle is only ever non-None when fixture["root"]
        # itself is not None (inspect_candidates never resolves a root_id
        # against an unnamed root table), so this single check covers both
        # the "reservations"/"users" root and the identity-only root=None
        # case (root_obj stays None) without needing to branch on the root
        # table name here.
        root_obj = store.read(candidate["root_reference"]["handle"]) if candidate["root_reference"]["handle"] is not None else None

        if fixture["root"] == "reservations" and not trivial_given:
            consistency = reservation_consistency_v2(root_obj, database, clock, policy["existing_reservation"], waived)
            if not consistency["passed"]:
                continue

        if trivial_given:
            given_truth = True
        elif not contract["given_contract"]["database_conditions"] and not contract["given_contract"]["non_database_conditions"]:
            given_truth = True  # literal "True" given, nothing to verify
        elif fixture["root"] == "users" and any(
                c["source_condition"]["table"] == "reservations" for c in contract["given_contract"]["database_conditions"]):
            given_truth = users_root_mixed_table_given_satisfied(contract, user, database)
        else:
            given = _given_observations(contract, candidate, user, root_obj if root_obj is not None else user, database, clock, semantic_given)
            given = resolve_ambiguous_any_quantifier(contract, given)
            given_truth = given["truth"]

        if given_truth is True:
            return candidate, user, root_obj, None

    # Same generic entity-synthesis fallback as section 43.1
    # (synthesize_entity_from_template), reused here rather than
    # reimplemented: a real, verified case (airline_131_state#b0) needs a
    # cancelled reservation and the fixture's own candidate pool is
    # genuinely empty (n_matches==0), so a real owned reservation elsewhere
    # in the snapshot is used as raw material, with the Given's own
    # database_conditions (only op=="eq") applied on top.
    if fixture["root"] == "reservations":
        db_conditions = [c["source_condition"] for c in contract["given_contract"]["database_conditions"]
                         if c["source_condition"]["table"] == "reservations"]
        if db_conditions and not contract["given_contract"]["non_database_conditions"]:
            from .v5_step7_airline_intent_adapter_v1 import any_real_owned_reservation
            from .v5_step7_intent_synthesis_v1 import synthesize_entity_from_template, IntentSynthesisError
            template_user, template_reservation = any_real_owned_reservation(database)
            try:
                synthesized, applied_fields = synthesize_entity_from_template(template_reservation, db_conditions)
            except IntentSynthesisError:
                synthesized = None
            if synthesized is not None:
                consistency = reservation_consistency_v2(synthesized, database, clock, policy["existing_reservation"], waived)
                given = _given_observations(contract, None, template_user, synthesized, database, clock, semantic_given)
                given = resolve_ambiguous_any_quantifier(contract, given)
                if consistency["passed"] and given["truth"] is True:
                    overlay = {"table": "reservations", "key": synthesized["reservation_id"], "fields": applied_fields,
                        "reason": "no_real_reservation_satisfies_this_given_state; synthesized from a real template"}
                    return {"candidate_index": None}, template_user, synthesized, overlay

    raise PreparationGap("no_supplied_existing_object_passes_given_and_nominal_consistency")
