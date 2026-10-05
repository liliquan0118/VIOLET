"""Lower reviewed v5 contracts into unbound functional-test plans, offline."""

from collections import Counter
from copy import deepcopy

from .artifacts import content_sha256
from .driver_plan_lowering_v1 import _oracle_event_plan, _prefixed_strings, _environment_references, _target_tool_names, _requires_confirmation_event
from .v5_branch_test_contract_v1 import SCHEMA as CONTRACT_SCHEMA
from .v5_given_when_preparation_v1 import verify_fingerprint


SCHEMA = "agentspectesting.v5-driver-plan-set/v0.1"


def _seal(document, field):
    document[field] = content_sha256(document)
    return document


def _observation(check):
    row = check["preparation"]
    route = row["effective_route"]
    kind = route["kind"]
    if kind == "mechanical":
        # Reuse the legacy structural extractor, but not its admission/interaction policy.
        observation = _oracle_event_plan({
            "binding_id": row["legacy_binding"]["binding_id"],
            "evaluation_mode": "mechanical",
            "runtime_observation_binding": {"runtime_binding": route["binding"],
                                            "expected_observation": row["expected_observation"]}})
    elif kind == "record_relation":
        target = route["contract"]["target_binding"]
        if not route.get("mandatory_guard") or not route["record_question_preparer"].endswith(".prepare_guarded_question"):
            raise ValueError("record route lost mandatory guard")
        observation = {"target_event": {"event_filter": {"event_kind": "assistant_tool_call",
            "field_equals": {"tool_name": target["tool_name"]}},
            "projection": {"kind": "field", "path": "arguments." + target["parameter"]}},
            "prerequisite_events": [], "related_evidence": {"adapter": route["evidence_adapter"],
                "identity_policy": "actual_target_event_subjects_and_linked_tool_results",
                "guarded_question_preparer": route["record_question_preparer"]}}
    elif kind == "behavior":
        observation = {"target_event": {"event_filter": {"event_kind": "request_response_window"},
            "selection": "complete_window_for_identified_user_request"}, "prerequisite_events": []}
    else:
        raise ValueError("unsupported effective route; adapter required")
    observation.update({"requirement_id": row["requirement_id"], "route_kind": kind,
                        "role": "observation_only_not_driver_success_milestone"})
    return observation


def lower_contracts(source):
    verify_fingerprint(source, "contract_set_fingerprint")
    if source["schema_version"] != CONTRACT_SCHEMA or source["runtime_dispatch_enabled"] is not False:
        raise ValueError("unsupported Step 5 contract input")
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
        if not checks:
            raise ValueError("missing checks")
        rids = set()
        for check in checks:
            row = check["preparation"]
            rid = row["requirement_id"]
            if rid in rids or row["branch_id"] != bid:
                raise ValueError("duplicate or misassociated check")
            rids.add(rid)
        observations = [_observation(c) for c in checks]
        tools = _target_tool_names(observations)
        binding_names = sorted({name for c in checks if c["preparation"]["effective_route"]["kind"] == "mechanical"
            for name in _prefixed_strings(c["preparation"]["effective_route"]["binding"], "driver_bindings.")})
        environment = _environment_references([c["source_condition"] for c in contract["given_contract"]["database_conditions"]])
        fixture = contract["fixture_contract"]
        user = contract["user_input_contract"]
        supplemental = sorted({c["preparation"]["effective_route"]["kind"] for c in checks} - {"mechanical"})
        confirmation_observed = _requires_confirmation_event(observations)
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
            "driver_plan_id": bid + "::V5DP01", "branch_id": bid,
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
                "additional_scope_requirements": ({"request_scope": "identify request and complete window",
                    "record_scope": "bind actual target event and its subjects if record_relation applies"} if supplemental else {}),
                "selection_policy": "verify candidates or construct a scenario; never assume upstream match proves conditions",
                "selected_candidate": None,
                "verification_policy": "preserve original Given logic; unknown is not true; user assertion is not proof of a database fact"},
            "interaction_plan": {
                "user_requirements": deepcopy(user), "nodes": nodes,
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
        "plans": plans, "out_of_pilot_branch_ids": deepcopy(source["out_of_pilot_branch_ids"]),
        "source_deferred_branch_ids": deepcopy(source["source_deferred_branch_ids"]),
        "summary": {"plan_count": len(plans), "requirement_count": sum(len(p["oracle_handoff"]["checks"]) for p in plans),
            "route_counts": dict(Counter(o["route_kind"] for p in plans for o in p["observation_plan"]["events"])),
            "plans_with_rule_gaps": sum(bool(p["oracle_handoff"]["known_gaps"]) for p in plans),
            "plans_with_clock_reference": sum(bool(p["fixture_binding_plan"]["required_environment_references"]) for p in plans),
            "runtime_ready_count": 0, "external_calls": 0, "target_agent_calls": 0}}, "plan_set_fingerprint")


def render_plans(result):
    lines = ["# Step 6 Abstract Driver Plan", "", "Only arranges preparation and observation; does not select objects, generate utterances, or execute.", "",
             "| branch | tools involved in observation | object fields to bind | clock references | rule gaps |", "| --- | --- | --- | --- | ---: |"]
    for p in result["plans"]:
        binding = p["fixture_binding_plan"]
        lines.append(f"| {p['branch_id']} | {', '.join(p['observation_plan']['tool_names_from_effective_routes']) or 'request-response window (tools not guessed)'} | {', '.join(binding['required_driver_binding_names']) or 'to be bound per scenario/scope'} | {', '.join(binding['required_environment_references']) or 'no explicit references'} | {len(p['oracle_handoff']['known_gaps'])} |")
    return "\n".join(lines + ["", "Each plan: binding preparation → start recording → establish dialogue preconditions → send request → observe and respond → hand off trajectory.",
        "Observing an Oracle target is not a condition for advancing or for stopping automatically. The occurrence of a forbidden operation must not become a Driver goal either.",
        "All actual preconditions, When being reached, evidence completeness, and budgets remain to be verified in the binding/run stage.", ""])
