"""Step 6, v2: lower Step 5 v2's full live-corpus contracts (four route
kinds: mechanical/semantic_judge/non_decisive/known_gap) into unbound
functional-test plans, offline.

v1 (v5_driver_plan_v1.py, untouched) only ever needed to handle v1's own
three route kinds (mechanical/record_relation/behavior, from the frozen
v5-evidence-step pilot) and never touched a semantic_judge check at all.
v2 reuses the SAME small set of generic, contract-shape-agnostic helpers v1
already imports from driver_plan_lowering_v1.py (_oracle_event_plan,
_prefixed_strings, _environment_references, _target_tool_names,
_requires_confirmation_event) -- confirmed these depend only on the check's
own runtime_observation_binding/evaluation_mode fields, not on the "legacy"
richer contract shape (evidence_bindings/logical_form/fixture_root) that
driver_plan_lowering_v1.py's OWN _lower_one/_lower_all use, which v5's
contracts do not carry and v2 does not attempt to fabricate.

One additional helper is newly reused here that v1 never needed:
_semantic_stimulus_requirements -- also contract-shape-agnostic (only reads
each check's evaluation_mode/semantic_judge_contract), and already maps both
of our real semantic_judge judge_kinds (claim_source_support,
subjective_expression) to a concrete stimulus_kind + constraints a future
driver would need to realize. This is real, already-designed value v1 had no
occasion to use.

known_gap checks get no _oracle_event_plan call at all (confirmed against
real data -- retail_059_order#b1v1::OR02's binding is
extractor_kind=="unbound", which _oracle_event_plan would reject with "no
target event anchor"); an honest placeholder observation is emitted instead,
never fabricating a target. (airline_075_norm#b0::OR01 used to be this
module's real example -- docs/agentcoveragetesting_reuse_log.md section 122
found its real root cause was a missing Step3 policy-lineage assessment, not
an inherently un-groundable claim, and completed that linkage; it now
compiles to a real semantic_judge route instead, and is no longer a
known_gap example.)
"""

from collections import Counter
from copy import deepcopy

from .artifacts import content_sha256
from .driver_plan_lowering_v1 import (
    _oracle_event_plan,
    _prefixed_strings,
    _environment_references,
    _target_tool_names,
    _requires_confirmation_event,
    _semantic_stimulus_requirements,
)
from .then_atomization import ThenAtomizationError
from .v5_branch_test_contract_v2 import SCHEMA as CONTRACT_SCHEMA
from .v5_given_when_preparation_v1 import verify_fingerprint


SCHEMA = "agentspectesting.v5-driver-plan-set/v0.2"


def _seal(document, field):
    document[field] = content_sha256(document)
    return document


def _stimulus_check(check):
    """Adapt a v5 Step5 v2 oracle check into the flat shape
    _semantic_stimulus_requirements expects (evaluation_mode +
    semantic_judge_contract at the top level), without mutating the
    original check."""
    route = check["effective_route"]
    result = {"requirement_id": check["requirement"]["requirement_id"]}
    if route["kind"] == "semantic_judge":
        result["evaluation_mode"] = "semantic_judge"
        result["semantic_judge_contract"] = {
            "semantic_judge_contract_id": route["semantic_judge_contract_id"],
            "program": route["program"],
        }
    else:
        result["evaluation_mode"] = route["kind"]
    return result


_RESERVATION_LOOKUP_TOOLS = ("get_reservation_details", "book_reservation")


def _custom_extraction_observation(route):
    """Real, tailored observation descriptions for the extension predicate_kinds
    whose runtime_binding carries no generic event_filter for
    _oracle_event_plan to anchor on (confirmed against real data and the real
    _extract_*_observations implementations in
    oracle_evaluator_contract_extension_v1.py -- see
    docs/oracle_requirement_pipeline_v0_7.md section 39). Returns None for any
    predicate_kind not handled here -- including routes (e.g. non_decisive,
    see v5_branch_test_contract_v2._route_for) that carry no "program" key at
    all, confirmed real on telecom's live corpus -- so the caller falls back
    to the generic honest placeholder instead of silently mis-describing
    something new or crashing on a missing key.
    """
    program = route.get("program")
    if program is None:
        return None
    predicate = program.get("predicate") or {}
    kind = predicate.get("predicate_kind")

    if kind == "flight_route_field_unchanged":
        # _extract_flight_route_field_observations's real target IS a plain
        # single-event tool_call filter (identified by the predicate's own
        # target_tool_name, not by the binding, which is
        # semantic_event_matcher_deferred for this shape) -- the complexity
        # is only in the VALUE it projects (a static flight-route-table
        # resolution + the most recent prior reservation lookup for the same
        # reservation_id), not in the target event itself.
        tool_name = predicate.get("target_tool_name")
        return {
            "target_event": {"event_filter": {"event_kind": "assistant_tool_call",
                                              "field_equals": {"tool_name": tool_name}}},
            "prerequisite_events": [
                {"event_filter": {"event_kind": "assistant_tool_call",
                                  "field_equals": {"tool_name": name}},
                 "role": "most_recent_prior_reservation_lookup_for_same_reservation_id"}
                for name in _RESERVATION_LOOKUP_TOOLS
            ],
            "projection_requires": "static_flight_route_reference_table",
        }

    if kind == "no_concurrent_message_and_tool_call":
        # _extract_turn_shape_observations groups the WHOLE event stream by
        # source_message_index and checks, per turn, whether that turn
        # produced both an assistant_message and an assistant_tool_call --
        # not a single-event target at all, so this is honestly described as
        # a stream-wide grouping/co-occurrence check instead of forced into
        # the single-event_filter shape.
        return {
            "target_event": {"event_filter": {"event_kind": "any"},
                             "selection": "every_turn_grouped_by_source_message_index"},
            "prerequisite_events": [],
            "co_occurrence_forbidden": ["assistant_message", "assistant_tool_call"],
        }

    if kind == "structurally_unreachable_action":
        # No observation exists to plan: the real tau2 tool schema makes the
        # prohibited action impossible via any tool call, so this predicate
        # evaluates unconditionally true regardless of trajectory content
        # (see docs/oracle_requirement_pipeline_v0_7.md section 27.9/34) --
        # distinct from a gap (there IS a real, correct program) and distinct
        # from "needs custom extraction logic" (there is nothing to extract).
        return {
            "target_event": None, "prerequisite_events": None,
            "no_observation_required": True,
            "reason": "This predicate is unconditionally true for every possible trajectory "
                      "(the real tool schema makes the prohibited action impossible); no event "
                      "needs to be watched at all.",
        }

    return None


def _observation(check):
    route = check["effective_route"]
    kind = route["kind"]
    rid = check["requirement"]["requirement_id"]
    if kind == "known_gap":
        return {
            "requirement_id": rid, "route_kind": kind,
            "role": "observation_only_not_driver_success_milestone",
            "target_event": None, "prerequisite_events": None,
            "expected_observation": None,
            "unavailable_reason": route["reason"],
        }
    binding_input = {
        "binding_id": route["runtime_observation_binding"]["binding_id"],
        "evaluation_mode": "semantic_judge" if kind == "semantic_judge" else "mechanical",
        "runtime_observation_binding": route["runtime_observation_binding"],
    }
    if kind == "semantic_judge":
        binding_input["semantic_judge_contract"] = {
            "semantic_judge_contract_id": route["semantic_judge_contract_id"],
            "program": route["program"],
        }
    try:
        observation = _oracle_event_plan(binding_input)
    except ThenAtomizationError:
        # A real, confirmed shape (see docs/oracle_requirement_pipeline_v0_7.md
        # section 39): some extension predicate_kinds extract their own
        # observation directly from the raw event list at evaluation time
        # (oracle_evaluator_contract_extension_v1.py's own
        # _extract_*_observations functions), bypassing the frozen generic
        # single-event-filter binding shape entirely -- their
        # runtime_binding legitimately carries no event_filter for
        # _oracle_event_plan's generic fallback to anchor on. This is NOT a
        # gap (the program is real and executable); it is a real limit of
        # this generic lowering helper. _custom_extraction_observation gives
        # a real, tailored description for the three predicate_kinds this
        # corpus is confirmed to actually use; anything else falls back to
        # an honest generic placeholder instead of fabricating or crashing.
        #
        # non_decisive routes (confirmed real and common on telecom's live
        # corpus -- see docs/agentcoveragetesting_reuse_log.md section 42)
        # are a distinct case from the above: they are genuinely bound (a
        # real runtime_observation_binding exists) but carry no "program" at
        # all, because _route_for never compiles one for this kind -- there
        # is no predicate, tailored or otherwise, to describe. The honest
        # placeholder for these reuses the real reason already recorded when
        # the route was classified non_decisive, instead of the
        # program-specific message below (which would misleadingly imply a
        # real predicate exists).
        tailored = _custom_extraction_observation(route)
        program = route.get("program")
        if tailored is not None:
            observation = tailored
        elif program is None:
            observation = {
                "target_event": None, "prerequisite_events": None,
                "expected_observation": deepcopy(route["runtime_observation_binding"]["expected_observation"]),
                "custom_extraction_required": None,
                "reason": route["reason"],
            }
        else:
            observation = {
                "target_event": None, "prerequisite_events": None,
                "expected_observation": deepcopy(route["runtime_observation_binding"]["expected_observation"]),
                "custom_extraction_required": (program.get("predicate") or {}).get("predicate_kind")
                    or program.get("evaluator_kind"),
                "reason": "This predicate extracts its own observation directly from the raw event "
                          "list at evaluation time; it has no single declarative event-filter anchor "
                          "for this generic lowering helper to describe, and no tailored description "
                          "has been written for this specific predicate_kind yet.",
            }
    observation.update({
        "requirement_id": rid, "route_kind": kind,
        # section 89: the try-path's _oracle_event_plan already stamps this from
        # binding_input, but the except-path's tailored/generic-placeholder
        # observations (_custom_extraction_observation and its two fallbacks just
        # above) never did -- silently starving build_generic_bound_driver_plan's
        # `binding_id is None` check downstream, which treats a missing binding_id
        # as "no real evaluator/binding to attach" and drops the check from the
        # online bound plan entirely, even though a real evaluator_contract AND
        # runtime_observation_binding both genuinely exist for it. Stamping it
        # here unconditionally (idempotent for the try-path, a real fix for the
        # except-path) is what actually connects these checks to online execution.
        "binding_id": route["runtime_observation_binding"]["binding_id"],
        "role": "observation_only_not_driver_success_milestone",
    })
    return observation


def lower_contracts_v2(source):
    verify_fingerprint(source, "contract_set_fingerprint")
    if source["schema_version"] != CONTRACT_SCHEMA or source["runtime_dispatch_enabled"] is not False:
        raise ValueError("unsupported Step 5 v2 contract input")
    plans, seen = [], set()
    for contract in source["contracts"]:
        verify_fingerprint(contract, "branch_test_contract_fingerprint")
        bid = contract["branch_id"]
        if bid in seen:
            raise ValueError("duplicate branch")
        seen.add(bid)
        if contract["planning_handoff"] != "eligible_with_explicit_obligations" or contract["runtime_ready"] is not False:
            raise ValueError("contract is not unbound planning input")
        checks = contract["oracle_contract"]["checks"]
        no_oracle_content = any(
            g.get("kind") == "branch_has_no_oracle_content" for g in contract["oracle_contract"]["known_gaps"]
        )
        if not checks and not no_oracle_content:
            raise ValueError("missing checks with no honest known_gap explaining why")
        rids = set()
        for check in checks:
            rid = check["requirement"]["requirement_id"]
            if rid in rids or check["requirement"]["branch_id"] != bid:
                raise ValueError("duplicate or misassociated check")
            rids.add(rid)

        observations = [_observation(c) for c in checks]
        tools = _target_tool_names([o for o in observations if o["target_event"] is not None])
        bindable_routes = [
            c["effective_route"]["runtime_observation_binding"]
            for c in checks
            if c["effective_route"]["kind"] != "known_gap"
        ]
        binding_names = {name for rb in bindable_routes for name in _prefixed_strings(rb, "driver_bindings.")}
        # section 89: flight_route_field_unchanged's runtime_observation_binding is
        # semantic_event_matcher_deferred (a generic placeholder with no
        # "driver_bindings.X" template refs for _prefixed_strings to find above --
        # see _extract_flight_route_field_observations's own docstring), even though
        # the predicate's real program genuinely needs a concrete, existing
        # reservation to compare origin/destination/trip_type against. Without this,
        # airline_031_arg#b0v0/v1/v2 (real airline_v2 binder, generic_tau_airline_v2_
        # bundle_resolver_v1.py::bind_airline_v2_branch) never bind any reservation
        # at all (required_driver_binding_names stayed empty), so the check has
        # nothing real to observe against even though it is now correctly wired into
        # the online bound plan (see the binding_id fix above).
        if any(
            ((c["effective_route"].get("program") or {}).get("predicate") or {}).get("predicate_kind")
            == "flight_route_field_unchanged"
            for c in checks
        ):
            binding_names.add("reservation_id")
        # section 92: real cabin-change branches (airline_032/033/074_order#b3/
        # 087/102 -- confirmed by a real corpus scan of every branch whose own
        # when_contract.spec_when mentions "cabin class") never have "cabin" in
        # required_driver_binding_names either, for the same underlying reason
        # as the flight_route_field_unchanged gap above: none of their real
        # compiled checks (compensation_formula/tool_call_error_matches for the
        # price-difference payment/refund requirements, or a near-vacuous
        # enum-membership check) reference "driver_bindings.cabin" directly, so
        # _prefixed_strings above never finds it. Without it, generic_tau_
        # airline_v2_bundle_resolver_v1.py's real cabin_change flag (`"cabin" in
        # required_names`) is always False, so the binder falls through to
        # logic that keeps the SAME cabin (its own correct "pick a genuinely
        # different cabin" branch is right there, just never reached) --
        # landing these branches on reservations already in their target cabin
        # (real, confirmed: 4WQ150 is already "business", the highest tier),
        # so no real price difference ever exists for OR01/OR03's payment/
        # refund checks to meaningfully test.
        if "cabin class" in (contract["when_contract"].get("spec_when") or "").lower():
            binding_names.add("cabin")
        binding_names = sorted(binding_names)
        environment = _environment_references(contract["given_contract"].get("database_conditions") or [])
        fixture = contract["fixture_contract"]
        user = contract["user_input_contract"]
        route_kinds_present = sorted({c["effective_route"]["kind"] for c in checks} - {"mechanical"})
        semantic_stimuli = _semantic_stimulus_requirements([_stimulus_check(c) for c in checks])
        confirmation_observed = _requires_confirmation_event([o for o in observations if o["target_event"] is not None])
        gaps = contract["oracle_contract"]["known_gaps"]

        nodes = [
            {"id": "prepare_bound_scenario", "depends_on": [], "owner_step": 7,
             "task": "bind or construct data, clock and request facts; verify applicable state facts and prepare verification of conversational conditions without assuming they already occurred"},
            {"id": "open_evidence_capture", "depends_on": ["prepare_bound_scenario"], "owner_step": 8,
             "task": "start capture before the first test conversation event, including prerequisite interactions"},
            {"id": "realize_conversation_prerequisites", "depends_on": ["open_evidence_capture"], "owner_step": 8,
             "task": "realize supplied conversation requirements in their bound order; an empty list is a no-op, not a fabricated condition"},
            {"id": "emit_bound_user_request", "depends_on": ["realize_conversation_prerequisites"], "owner_step": 8,
             "task": "emit the supplied user goal and preserve When qualifiers without adding another business goal"},
            {"id": "observe_and_respond", "depends_on": ["emit_bound_user_request"], "owner_step": 8,
             "task": "record responses and tool events; answer with verified bound facts only under configured limits"},
            {"id": "close_and_handoff_trace", "depends_on": ["observe_and_respond"], "owner_step": 8,
             "task": "record termination and evidence completeness, then hand off; never assign pass/fail here"},
        ]

        plan = {
            "driver_plan_id": bid + "::V5DP02", "branch_id": bid,
            "source_contract_id": contract["branch_test_contract_id"],
            "source_contract_fingerprint": contract["branch_test_contract_fingerprint"],
            "test_point": {"given": deepcopy(contract["given_contract"]),
                           "when": deepcopy(contract["when_contract"]),
                           "preconditions_verified": False, "when_reached": None},
            "fixture_binding_plan": {
                "status": "requires_step7_binding", "root": fixture["root"],
                "candidate_source": {"branch_id": bid, "contract_field": "fixture_contract.lookup",
                    "source_contract_fingerprint": contract["branch_test_contract_fingerprint"],
                    "candidate_count": fixture["lookup"]["n_matches"], "lookup_status": fixture["lookup"]["lookup_status"]},
                "conditions": deepcopy(contract["given_contract"]),
                "relation_assertions": deepcopy(fixture["relation_assertions"]),
                "required_driver_binding_names": binding_names,
                "required_environment_references": environment,
                "additional_scope_requirements": ({"request_scope": "identify request and complete window"} if route_kinds_present else {}),
                "selection_policy": "verify candidates or construct a scenario; never assume upstream match proves conditions",
                "selected_candidate": None,
                "verification_policy": "preserve original Given logic; unknown is not true; user assertion is not proof of a database fact"},
            "interaction_plan": {
                "user_requirements": deepcopy(user), "nodes": nodes,
                "semantic_stimulus_requirements": semantic_stimuli,
                "operation_fact_requirements": [{"tool_name": t, "source": "verified_bound_data_and_tool_schema",
                                                 "status": "pending_step7"} for t in tools],
                "response_policy": {"answer_from_verified_bound_facts_only": True,
                    "do_not_change_requested_operation": True,
                    "confirmation": "only after agent requests confirmation for the same bound operation and applicable user constraints allow it",
                    "confirmation_metadata": "record only an actual driver confirmation; never prelabel or fabricate it",
                    "missing_fact": "report unmet binding requirement; do not invent"},
                "confirmation_evidence_required_by_oracle": confirmation_observed},
            "observation_plan": {"events": observations, "tool_names_from_effective_routes": tools,
                "capture_window": {"opens": "before_first_test_conversation_event", "closes": "recorded_terminal_event",
                    "precondition_timing": "preserve source timing; resolve conditional timing during binding, not by assuming every fact is before the request"},
                "reachability": {"user_request_emitted": "runtime_observation_required",
                    "spec_when_reached": "separate_evidence_required_not_inferred_from_oracle_success",
                    "runtime_predicate_binding": "deferred_to_steps7_8"},
                "termination": {"reasons": ["agent_terminal_response", "agent_transfer", "configured_limit", "missing_binding", "incomplete_evidence"],
                    "oracle_target_seen_is_not_an_automatic_stop": True,
                    "budget": "required_run_configuration_not_set_here"}},
            "oracle_handoff": deepcopy(contract["oracle_contract"]),
            "source_required_work": deepcopy(contract["required_work"]),
            "completed_work": ["prepare_abstract_driver_plan"],
            "remaining_work": [deepcopy(w) for w in contract["required_work"] if w["owner_step"] != 6],
            "surface_input_policy": {"allowed": ["user request and supplied conversation requirements", "verified bound facts"],
                "forbidden": ["Then", "Oracle", "expected verdict", "policy correctness conclusion"],
                "full_plan_may_be_sent_to_surface_model": False,
                "enforcement": "requires_allowlisted_projection_at_step7_no_surface_model_called_here"},
            "generation_readiness": {"status": "abstract_plan_prepared", "concrete_fixture_bound": False,
                "surface_prompt_generated": False, "runtime_execution_allowed": False,
                "legacy_binder_accepts_this_schema": False},
            "scope": "functional baseline planning only; no prompt mutations, adversarial probes, feedback search or execution",
        }
        plans.append(_seal(plan, "driver_plan_fingerprint"))
    if len(plans) != source["summary"]["branch_count"]:
        raise ValueError("branch count mismatch")
    return _seal({"schema_version": SCHEMA, "source_contract_set_fingerprint": source["contract_set_fingerprint"],
        "plans": plans,
        "source_deferred_branch_ids": deepcopy(source["source_deferred_branch_ids"]),
        "summary": {"plan_count": len(plans), "requirement_count": sum(len(p["oracle_handoff"]["checks"]) for p in plans),
            "route_counts": dict(Counter(o["route_kind"] for p in plans for o in p["observation_plan"]["events"])),
            "plans_with_known_gaps": sum(bool(p["oracle_handoff"]["known_gaps"]) for p in plans),
            "plans_with_clock_reference": sum(bool(p["fixture_binding_plan"]["required_environment_references"]) for p in plans),
            "plans_with_semantic_stimulus_requirements": sum(bool(p["interaction_plan"]["semantic_stimulus_requirements"]) for p in plans),
            "runtime_ready_count": 0, "external_calls": 0, "target_agent_calls": 0}}, "plan_set_fingerprint")


def render_plans(result):
    lines = ["# Step 6 Abstract Driver Plan (v2, full live Step3/4/5)", "",
             "Only arranges preparation and observation; does not select objects, generate utterances, or execute.", "",
             "| branch | tools involved in observation | object fields to bind | clock references | semantic stimulus requirements | known gaps |",
             "| --- | --- | --- | --- | ---: | ---: |"]
    for p in result["plans"]:
        binding = p["fixture_binding_plan"]
        lines.append(
            f"| {p['branch_id']} | {', '.join(p['observation_plan']['tool_names_from_effective_routes']) or 'request-response window (tools not guessed)'} "
            f"| {', '.join(binding['required_driver_binding_names']) or 'to be bound per scenario/scope'} "
            f"| {', '.join(binding['required_environment_references']) or 'no explicit references'} "
            f"| {len(p['interaction_plan']['semantic_stimulus_requirements'])} "
            f"| {len(p['oracle_handoff']['known_gaps'])} |"
        )
    return "\n".join(lines + [
        "",
        "Each plan: binding preparation → start recording → establish dialogue preconditions → send request → observe and respond → hand off trajectory.",
        "Observing an Oracle target is not a condition for advancing or for stopping automatically. The occurrence of a forbidden operation must not become a Driver goal either.",
        "All actual preconditions, When being reached, evidence completeness, and budgets remain to be verified in the binding/run stage.",
        "`semantic_judge` observation items carry `semantic_stimulus_requirements`, indicating what kind of dialogue content must later be elicited so the judge can obtain a real answer.",
        "`known_gap` observation items have no target_event and must not be forcibly bound or evaluated.",
        "",
    ])
