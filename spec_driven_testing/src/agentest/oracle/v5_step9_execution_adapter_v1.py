"""Adapter: package a real Step 8 driver trace + its Step 7 case into the `execution`
shape `evaluate_runtime_oracle_branch[_v2]` (runtime_evaluation_v1.py/v2.py) require,
and construct a minimal-but-real branch eligibility gate for unconditional-Given
branches. See docs/v5_step9_execution_adapter_proposal_v0_1.md for the design and the
real runs this was validated against.

No trajectory-format converter needed to be written: `execution["messages"]` is
consumed as-is by the existing `normalize_tau_execution`
(compiler/runtime_observation_binding_v1.py) -- a real Step 8 trace's `messages` list
is already in the OpenAI-style shape that function expects.

Scope: only handles branches whose Given is the literal, unconditional "True" (no
database_conditions, no non_database_conditions) -- `trivial_unconditional_gate` raises
for anything else rather than silently approximating. The full eligibility chain
(`compile_branch_eligibility_contract` + `build_branch_eligibility_witness`, which
needs a bound Driver plan's `fixture_instances[].predicate_witnesses`) is not wired
here: this Step 7 package lineage (`v5_step7_package_v1.py`) has no such artifact.
"""
from copy import deepcopy

from ..compiler.artifacts import content_sha256
from .branch_eligibility_v1 import GATE_VERSION, validate_branch_eligibility_gate


class Step9AdapterError(ValueError):
    pass


def build_execution(case, trace, *, variant_id=None, profile_id=None):
    """`case`: a Step 7 package row (outputs/v5_step7_package_v0_1_final/package.json).
    `trace`: the `raw_transport.trace` object from a Step 8 run's final
    `run_finished_review_required` journal event (e.g. AuthorityBoundSession.finish()).
    """
    if case.get("branch_id") != trace.get("branch_id"):
        raise Step9AdapterError("case_and_trace_branch_id_mismatch")
    if case.get("case_fingerprint") != trace.get("source_case_fingerprint"):
        raise Step9AdapterError("trace_not_produced_from_this_case")

    messages = deepcopy(trace["messages"])
    transport_action_log = []
    for event in trace.get("authority_events", []):
        if not event.get("confirmed_this_turn"):
            continue
        index = event["message_index"]
        transport_action_log.append({
            "action_kind": "confirm_operation",
            "content": messages[index]["content"],
        })

    return {
        "candidate_id": case["branch_id"],
        # Not a real fixture_instances-mechanism id (that lineage has no artifact for
        # this Step 7 package) -- only used here for execution identity/reporting.
        "fixture_instance_id": case["case_fingerprint"],
        "variant_id": variant_id,
        "profile_id": profile_id,
        "driver_bindings": deepcopy(case["driver_bindings"]),
        "messages": messages,
        "runtime_driver": {"transport_action_log": transport_action_log},
    }


def trivial_unconditional_gate(case, execution, branch_id):
    """Only valid when Given is the literal, unconditional "True". Real Given
    conditions must go through the full branch_eligibility_v1 chain instead of this
    shortcut -- that chain isn't wired here (see module docstring).
    """
    given = case.get("source_given_contract") or {}
    if (given.get("text") != "True" or given.get("database_conditions")
            or given.get("non_database_conditions")):
        raise Step9AdapterError("branch_has_real_given_conditions_full_eligibility_chain_required")
    if case.get("branch_id") != branch_id:
        raise Step9AdapterError("case_branch_id_does_not_match_requested_branch_id")

    gate = {
        "schema_version": GATE_VERSION,
        "branch_eligibility_gate_id": f"{branch_id}::gate/trivial_unconditional",
        "oracle_branch_id": branch_id,
        "status": "satisfied",
        "reason": "given_is_unconditional_true_no_predicates_to_check",
        "required_predicate_count": 0,
        "matched_predicate_count": 0,
        "missing_predicate_ids": [],
        "unknown_predicate_ids": [],
        "predicate_mismatches": {},
        "lineage_checks": {},
        "runtime_pre_state_mismatches": {},
        "runtime_given_evidence": {},
        "source_branch_eligibility_contract_fingerprint": "not_built_unconditional_given_shortcut",
        "source_branch_eligibility_witness_fingerprint": "not_built_unconditional_given_shortcut",
        "source_execution_fingerprint": content_sha256(execution),
    }
    gate["branch_eligibility_gate_fingerprint"] = content_sha256(gate)
    # Must genuinely pass the project's own validator, not just look plausible.
    return validate_branch_eligibility_gate(gate)


def given_verified_at_preparation_gate(case, execution, branch_id):
    """For branches with real Given conditions, where the condition was already
    mechanically checked and verified True at Step 7 preparation time
    (`case["given_verification"]`), against the exact same frozen database snapshot
    the Step 8 dialogue actually ran against.

    This is a narrower, different justification than the full
    `compile_branch_eligibility_contract` + `build_branch_eligibility_witness` chain
    (which re-derives Given truth from a bound Driver plan's fixture predicate
    witnesses, plus live reachability milestones from the execution itself -- that
    chain isn't wired here, same reason as `trivial_unconditional_gate`: this Step 7
    package lineage has no `fixture_instances` artifact). What this DOES establish:
    the specific object (`case["driver_bindings"]`, e.g. a reservation_id) the dialogue
    actually operated on is the same one Step 7 verified the Given predicates against --
    `build_execution` already asserts `case["case_fingerprint"] == trace["source_case_fingerprint"]`,
    so this gate does not need to re-derive that binding here. What this does NOT
    establish: live reachability milestones (was the target operation actually
    requested, was identity closed) -- those come for free at the requirement level
    instead, because every evaluator contract used here has `requires_observation:
    True` or an explicit absence check, so a dialogue that stalled before reaching the
    target action still gets a correct fail/absent verdict, not a false pass.

    Restricted to the case where every individual database-condition observation is
    itself True (matches an "all" combinator or a single condition) -- raises rather
    than guess for a partial match under an "any"/"or" combinator, since this project's
    convention is to fail loud on an unhandled shape rather than approximate one.
    """
    given = case.get("source_given_contract") or {}
    verification = case.get("given_verification") or {}
    if given.get("text") == "True" and not given.get("database_conditions") and not given.get("non_database_conditions"):
        raise Step9AdapterError("branch_is_unconditional_use_trivial_unconditional_gate_instead")
    if verification.get("truth") is not True or verification.get("basis") != "reviewed_boolean_expression":
        raise Step9AdapterError("given_not_verified_true_at_preparation")
    observations = verification.get("observations") or []
    if not observations or any(o.get("truth") is not True for o in observations):
        raise Step9AdapterError("given_verification_has_unresolved_or_false_sub_observations")
    if case.get("branch_id") != branch_id:
        raise Step9AdapterError("case_branch_id_does_not_match_requested_branch_id")

    count = len(observations)
    gate = {
        "schema_version": GATE_VERSION,
        "branch_eligibility_gate_id": f"{branch_id}::gate/given_verified_at_preparation",
        "oracle_branch_id": branch_id,
        "status": "satisfied",
        "reason": "given_database_conditions_verified_true_at_step7_preparation_against_same_frozen_snapshot",
        "required_predicate_count": count,
        "matched_predicate_count": count,
        "missing_predicate_ids": [],
        "unknown_predicate_ids": [],
        "predicate_mismatches": {},
        "lineage_checks": {},
        "runtime_pre_state_mismatches": {},
        "runtime_given_evidence": {"source": "case_given_verification_not_live_reachability_witness"},
        "source_branch_eligibility_contract_fingerprint": "not_built_given_verified_at_preparation_shortcut",
        "source_branch_eligibility_witness_fingerprint": "not_built_given_verified_at_preparation_shortcut",
        "source_execution_fingerprint": content_sha256(execution),
    }
    gate["branch_eligibility_gate_fingerprint"] = content_sha256(gate)
    return validate_branch_eligibility_gate(gate)
