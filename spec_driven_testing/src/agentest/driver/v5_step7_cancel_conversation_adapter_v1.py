"""Mechanical support for cancel_reservation branches compile_package (v1,
frozen) blocks with additional_conversation_constraints_require_adapter --
its universal early gate (`if conversation_preconditions or when_qualifiers:
raise PreparationGap(...)`) fires before even dispatching to the real
book/cancel logic, regardless of which tool. The 2 real branches this covers
(airline_093_state#b3/airline_093_state#e1) have completely ordinary
candidates/consistency/Given verification once past that gate -- the ONLY
thing "additional" here is a conversation_preconditions entry describing
what REASON the user gives for cancelling (covered by insurance, or not), a
real conversational fact Step7's own message construction is responsible
for embodying, the same discipline already used for send_certificate's
complaint-status branches (v5_step7_lookup_tools_adapter_v1.py). See
docs/oracle_requirement_pipeline_v0_7.md section 45.
"""
from __future__ import annotations

from copy import deepcopy

from .v5_object_references_v1 import inspect_candidates
from .v5_step7_package_v1 import PreparationGap, _reservation_consistency, _given_observations, _constraint_match


def compose_cancel_message(contract):
    """Deterministic: reads whether the real conversation_precondition says
    the cancellation reason IS or IS NOT covered by insurance and phrases
    the message accordingly, without inventing a specific policy-sensitive
    reason (this module cannot know which concrete reasons are covered)."""
    covered = None
    for req in contract["user_input_contract"]["conversation_preconditions"]:
        clause = req["requirement"].get("source_clause", "").lower()
        if "covered by insurance" in clause:
            covered = "not" not in clause.split("covered")[0][-8:]
    if covered is True:
        return "I need to cancel my reservation -- it's for a reason that my travel insurance should cover."
    if covered is False:
        return "I'd like to cancel my reservation, though I know my reason isn't something my travel insurance covers."
    return "I'd like to cancel my reservation."


def _non_database_conditions_are_all_trivially_conversational(contract):
    """Local variant of v5_step7_lookup_tools_adapter_v1's
    given_is_trivial_conversational_restatement -- that function ALSO
    requires database_conditions to be empty, because in its own caller
    (resolve_lookup_or_certificate_candidate) "trivial_given" short-circuits
    straight to given_truth=True with no further verification of
    database_conditions at all, so its guard against non-empty
    database_conditions is load-bearing there. Here the caller (below)
    separately, correctly evaluates any real database_conditions afterward,
    so only the non_database_conditions shape itself needs checking: every
    entry must be realizable=='conversation' (true by construction once
    Step7's own message embodies it), and there must be at least one (an
    empty list isn't "trivially conversational", it's just absent)."""
    conditions = contract["given_contract"]["non_database_conditions"]
    return bool(conditions) and all(c["source_condition"]["realizable"] == "conversation" for c in conditions)


def _semantic_given_with_conversational_gate_cleared(contract, semantic_given):
    """_given_observations's second gate (given_logic_missing_or_unrepresented)
    fires whenever the reviewed semantic expression's own unrepresented_text
    is non-empty -- and here it always IS non-empty, because the real semantic
    reviewer correctly reported that its boolean expression covers only the
    database_conditions (C1, C2, ...) and explicitly leaves the conversational
    non-db clause unrepresented (that's not a reviewer mistake, it's an
    accurate report -- there is no database condition for it to map to).
    Only clears unrepresented_text when every entry in it matches (loosely,
    case/whitespace-insensitively) one of the real non_database_conditions'
    own source clauses -- i.e. only when the "unrepresented" part is
    confirmed to be exactly the conversational content already verified
    trivial above, never a silent way to paper over some other real gap in
    the reviewed expression."""
    if semantic_given is None or semantic_given.get("expression") is None:
        return None
    unrepresented = semantic_given.get("unrepresented_text") or []
    if not unrepresented:
        return semantic_given
    known_clauses = {c["source_condition"]["clause"].strip().lower()
                     for c in contract["given_contract"]["non_database_conditions"]}
    if not all(u.strip().lower() in known_clauses for u in unrepresented):
        return None
    patched = deepcopy(semantic_given)
    patched["unrepresented_text"] = []
    return patched


def _given_truth_allowing_conversational_conditions(contract, candidate, user, reservation, database, raw_clock, semantic_given):
    """_given_observations (frozen) rejects ANY non_database_conditions
    outright, before even looking at database_conditions -- but a
    realizable=='conversation' condition (verified here to be the ONLY
    non-database content) is true by construction once Step7's own message
    embodies it, so the real database_conditions still need evaluating
    normally. Builds a local, ephemeral copy of the contract with
    non_database_conditions cleared ONLY to route around that gate for the
    database-condition evaluation -- never mutates the real contract, and
    only used after confirming the omitted part is honestly trivial. Also
    patches the reviewed semantic expression's unrepresented_text the same
    way, and only when that too is confirmed to correspond exactly to the
    trivial conversational content (see
    _semantic_given_with_conversational_gate_cleared)."""
    if not _non_database_conditions_are_all_trivially_conversational(contract):
        return None
    if not contract["given_contract"]["database_conditions"]:
        return True
    patched_semantic = _semantic_given_with_conversational_gate_cleared(contract, semantic_given)
    if patched_semantic is None:
        return None
    stripped = deepcopy(contract)
    stripped["given_contract"]["non_database_conditions"] = []
    given = _given_observations(stripped, candidate, user, reservation, database, raw_clock, patched_semantic)
    return given["truth"]


def resolve_cancel_with_conversation_constraints(contract, store, database, parsed_clock, raw_clock, policy, semantic_given, constraints):
    """Same real logic as compile_package's own cancel_reservation branch
    (_reservation_consistency/_constraint_match), just without the early
    conversation_preconditions gate -- reused directly from v1 rather than
    reimplemented, since nothing about candidate resolution itself needs to
    change. Two clock forms are required because compile_package's own two
    helpers disagree on shape: _reservation_consistency needs the parsed
    datetime, _given_observations needs the raw ISO string -- both real,
    both kept exactly as v1 calls them, not unified into one shape here."""
    fixture = contract["fixture_contract"]
    if fixture["root"] != "reservations":
        raise PreparationGap("unsupported_existing_object_type")
    if fixture["relation_assertions"]["user_requirements"] != {"present": True, "value": "owner"}:
        raise PreparationGap("existing_object_owner_scope_not_supported")
    for candidate in inspect_candidates(fixture, store):
        if candidate["reference_status"] != "resolved" or candidate["relation_observation"]["status"] != "matched":
            continue
        user = store.read(candidate["user_reference"]["handle"])
        reservation = store.read(candidate["root_reference"]["handle"])
        consistency = _reservation_consistency(reservation, database, parsed_clock, policy["existing_reservation"])
        given_truth = _given_truth_allowing_conversational_conditions(contract, candidate, user, reservation, database, raw_clock, semantic_given)
        args = {"reservation_id": reservation["reservation_id"]}
        if consistency["passed"] and given_truth is True and _constraint_match(args, constraints):
            return candidate, user, args
    raise PreparationGap("no_supplied_existing_object_passes_given_and_nominal_consistency")
