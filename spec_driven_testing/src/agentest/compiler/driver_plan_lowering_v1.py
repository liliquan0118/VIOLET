"""Lower admitted branch test contracts into generic unbound Driver plans."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import content_sha256
from .branch_test_contract_v1 import validate_branch_test_contract_set
from .then_atomization import ThenAtomizationError


PLAN_SET_VERSION = "agentspectesting.generic-driver-plan-set/v0.1"
PLAN_VERSION = "agentspectesting.generic-driver-plan/v0.1"
LOWERING_VERSION = "branch-contract-driver-lowering/v0.1"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _prefixed_strings(value: Any, prefix: str) -> list[str]:
    return sorted(
        {
            item[len(prefix) :]
            for item in _walk(value)
            if isinstance(item, str) and item.startswith(prefix)
        }
    )


def _environment_references(value: Any) -> list[str]:
    return sorted(
        {item for item in _walk(value) if isinstance(item, str) and item.startswith("$")}
    )


def _event_anchor(event: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(event, Mapping):
        return None
    event_filter = event.get("event_filter")
    if not isinstance(event_filter, Mapping):
        return None
    result = {"event_filter": deepcopy(event_filter)}
    for key in ("scope_constraints", "projection", "matcher", "binding_kind"):
        if key in event:
            result[key] = deepcopy(event[key])
    return result


def _oracle_event_plan(check: Mapping[str, Any]) -> dict[str, Any]:
    binding = check["runtime_observation_binding"]
    runtime = binding["runtime_binding"]
    extractor = runtime.get("extractor_kind")
    if check.get("evaluation_mode") == "semantic_judge":
        semantic = _mapping(
            check.get("semantic_judge_contract"), "$.semantic_judge_contract"
        )
        selector = _mapping(
            semantic.get("program", {}).get("target_selector"),
            "$.semantic_judge_contract.program.target_selector",
        )
        target = {
            "event_filter": {"event_kind": selector["event_kind"]},
            "selection": selector.get("selection"),
            "semantic_judge_contract_id": semantic["semantic_judge_contract_id"],
        }
        prerequisites = []
    elif extractor == "temporal_relation":
        target = _event_anchor(runtime.get("right_event"))
        prerequisites = [
            anchor
            for anchor in [_event_anchor(runtime.get("left_event"))]
            if anchor is not None
        ]
    else:
        target = _event_anchor(runtime)
        prerequisites = []
    if target is None:
        raise ThenAtomizationError(
            f"ready Oracle check has no target event anchor: {check.get('binding_id')}"
        )
    return {
        "binding_id": check["binding_id"],
        "evaluation_mode": check.get("evaluation_mode"),
        "target_event": target,
        "prerequisite_events": prerequisites,
        "expected_observation": deepcopy(binding["expected_observation"]),
    }


def _target_tool_names(event_plans: Sequence[Mapping[str, Any]]) -> list[str]:
    names = set()
    for plan in event_plans:
        event_filter = plan["target_event"].get("event_filter") or {}
        if event_filter.get("event_kind") != "assistant_tool_call":
            continue
        name = (event_filter.get("field_equals") or {}).get("tool_name")
        if isinstance(name, str) and name:
            names.add(name)
    return sorted(names)


def _surface_facts(evidence_bindings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    facts = []
    for evidence in evidence_bindings:
        executable = evidence.get("executable_binding") or {}
        kind = executable.get("kind")
        if kind == "composite_evidence" and isinstance(executable.get("user_fact"), Mapping):
            facts.append(
                {
                    "fact_contract_id": f"{evidence['condition_id']}::USER_FACT",
                    "fact_kind": "user_supplied_fact",
                    "value_contract": deepcopy(executable["user_fact"]),
                    "disclosure_timing": "before_decision_opportunity",
                    "source_binding_fingerprint": evidence["binding_fingerprint"],
                }
            )
        elif kind == "capability_scenario":
            facts.append(
                {
                    "fact_contract_id": f"{evidence['condition_id']}::REQUEST_SCENARIO",
                    "fact_kind": "request_scenario",
                    "value_contract": deepcopy(executable),
                    "disclosure_timing": "as_user_goal",
                    "source_binding_fingerprint": evidence["binding_fingerprint"],
                }
            )
        elif evidence.get("source_kind") in {"user_speech_act", "user_supplied_fact"}:
            facts.append(
                {
                    "fact_contract_id": f"{evidence['condition_id']}::USER_EVIDENCE",
                    "fact_kind": evidence["source_kind"],
                    "value_contract": deepcopy(executable),
                    "disclosure_timing": "before_decision_opportunity",
                    "source_binding_fingerprint": evidence["binding_fingerprint"],
                }
            )
    return facts


def _boundary_opportunities(
    evidence_bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    opportunities = []
    satisfying = {
        "ge": ["equal", "greater_by_adapter_minimum_resolution"],
        "gt": ["greater_by_adapter_minimum_resolution"],
        "le": ["equal", "less_by_adapter_minimum_resolution"],
        "lt": ["less_by_adapter_minimum_resolution"],
    }
    contrast = {
        "ge": "less_by_adapter_minimum_resolution",
        "gt": "equal",
        "le": "greater_by_adapter_minimum_resolution",
        "lt": "equal",
    }
    for evidence in evidence_bindings:
        executable = evidence.get("executable_binding") or {}
        predicate = executable.get("predicate")
        if not isinstance(predicate, Mapping) or predicate.get("op") not in satisfying:
            continue
        value = predicate.get("value")
        if not isinstance(value, Mapping) or not isinstance(value.get("ref"), str):
            continue
        opportunities.append(
            {
                "condition_id": evidence["condition_id"],
                "field": {
                    "table": predicate.get("table"),
                    "path": predicate.get("path"),
                },
                "operator": predicate["op"],
                "boundary_expression": deepcopy(value),
                "satisfying_probe_relations": satisfying[predicate["op"]],
                "nearest_contrast_relation": contrast[predicate["op"]],
                "minimum_resolution": "adapter_defined",
                "semantic_invariant": "selected_probe_must_preserve_this_branch_given_truth",
            }
        )
    return opportunities


def _requires_confirmation_event(event_plans: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        item == "confirm_operation"
        for plan in event_plans
        for item in _walk(plan.get("prerequisite_events") or [])
    )


def _semantic_stimulus_requirements(
    oracle_checks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    requirements = []
    for check in oracle_checks:
        if check.get("evaluation_mode") != "semantic_judge":
            continue
        semantic = check["semantic_judge_contract"]
        judge_kind = semantic["program"]["judge_kind"]
        if judge_kind == "claim_source_support":
            stimulus_kind = "unsupported_information_request"
            constraints = [
                "request_one_specific_factual_answer",
                "answer_not_present_in_prior_user_messages_tools_or_system_policy",
                "user_does_not_supply_the_answer",
            ]
        elif judge_kind == "subjective_expression":
            stimulus_kind = "subjective_recommendation_request"
            constraints = [
                "request_a_preference_recommendation_or_subjective_comment",
                "do_not_embed_a_required_opinion_in_the_user_message",
            ]
        else:
            raise ThenAtomizationError(
                f"unsupported semantic stimulus judge kind: {judge_kind}"
            )
        requirements.append(
            {
                "stimulus_requirement_id": f"{check['requirement_id']}::STIMULUS",
                "stimulus_kind": stimulus_kind,
                "constraints": constraints,
                "materialization_status": "requires_driver_binding",
                "semantic_judge_contract_id": semantic[
                    "semantic_judge_contract_id"
                ],
            }
        )
    return requirements


def _operation_input_probe_requirements(
    oracle_checks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract test-input probes from executable argument predicates.

    The Oracle remains hidden from the surface realizer.  This stage only turns
    an already compiled argument invariant into a typed value-generation
    request for the Binder.  A non-required observation represents a guard: the
    useful test input is immediately outside the allowed region.  A required
    observation represents a nominal operation and stays inside the region.
    """

    probes: dict[tuple[Any, ...], dict[str, Any]] = {}
    for check in oracle_checks:
        if check.get("evaluation_mode") != "mechanical":
            continue
        binding = check.get("runtime_observation_binding") or {}
        runtime = binding.get("runtime_binding") or {}
        event_filter = runtime.get("event_filter") or {}
        tool_name = (event_filter.get("field_equals") or {}).get("tool_name")
        program = (check.get("evaluator_contract") or {}).get("program") or {}
        if not isinstance(tool_name, str) or program.get("evaluator_kind") != "all_matches_predicate":
            continue
        predicate = program.get("predicate") or {}
        requires_observation = bool(program.get("requires_observation"))
        kind = predicate.get("predicate_kind")

        if kind == "field_lte":
            left = predicate.get("left_path")
            right = predicate.get("right_path")
            if not isinstance(left, str) or not isinstance(right, str):
                continue
            relation = "lte" if requires_observation else "gt_by_adapter_minimum_unit"
            key = (tool_name, kind, left, right, relation)
            probes[key] = {
                "probe_requirement_id": f"{check['requirement_id']}::INPUT_PROBE",
                "tool_name": tool_name,
                "probe_kind": "relative_numeric_fields",
                "field_paths": [left, right],
                "required_relation": relation,
                "purpose": "nominal_observation" if requires_observation else "guard_challenge",
                "materialization_status": "requires_driver_binding",
            }
        elif kind == "all_of":
            children = predicate.get("predicates") or []
            length_limits = [
                child
                for child in children
                if isinstance(child, Mapping) and child.get("predicate_kind") == "length_lte"
            ]
            projection = runtime.get("projection") or {}
            path = projection.get("path")
            for child in length_limits:
                limit = child.get("value")
                if not isinstance(path, str) or not isinstance(limit, int):
                    continue
                relation = "lte" if requires_observation else "equal_limit_plus_one"
                key = (tool_name, "collection_length", path, limit, relation)
                probes[key] = {
                    "probe_requirement_id": f"{check['requirement_id']}::INPUT_PROBE",
                    "tool_name": tool_name,
                    "probe_kind": "collection_length",
                    "field_path": path,
                    "limit": limit,
                    "required_relation": relation,
                    "purpose": "nominal_observation" if requires_observation else "guard_challenge",
                    "materialization_status": "requires_driver_binding",
                }
            allowed_sets = [
                child
                for child in children
                if isinstance(child, Mapping) and child.get("predicate_kind") == "in_set"
            ]
            for child in allowed_sets:
                values = child.get("values") or []
                if not isinstance(path, str) or not values:
                    continue
                relation = "member" if requires_observation else "non_member"
                key = (tool_name, "allowed_value_set", path, tuple(values), relation)
                probes[key] = {
                    "probe_requirement_id": f"{check['requirement_id']}::INPUT_PROBE",
                    "tool_name": tool_name,
                    "probe_kind": "allowed_value_set",
                    "field_path": path,
                    "allowed_values": deepcopy(values),
                    "required_relation": relation,
                    "purpose": "nominal_observation" if requires_observation else "guard_challenge",
                    "materialization_status": "requires_driver_binding",
                }
    return list(probes.values())


def _lower_one(contract: Mapping[str, Any]) -> dict[str, Any]:
    branch_id = contract["branch_id"]
    evidence_bindings = contract["given_contract"]["evidence_bindings"]
    oracle_checks = contract["oracle_contract"]["checks"]
    event_plans = [_oracle_event_plan(check) for check in oracle_checks]
    target_tools = _target_tool_names(event_plans)
    required_driver_bindings = _prefixed_strings(oracle_checks, "driver_bindings.")
    environment_refs = _environment_references(evidence_bindings)
    surface_facts = _surface_facts(evidence_bindings)
    semantic_stimuli = _semantic_stimulus_requirements(oracle_checks)
    operation_input_probes = _operation_input_probe_requirements(oracle_checks)
    boundary_opportunities = _boundary_opportunities(evidence_bindings)
    logical_form = contract["given_contract"]["logical_form"]
    active_condition_ids = logical_form["branch_alignment"]["active_condition_ids"]
    confirmation_required = _requires_confirmation_event(event_plans)
    fixture = contract["fixture_contract"]

    plan = {
        "schema_version": PLAN_VERSION,
        "lowering_version": LOWERING_VERSION,
        "driver_plan_id": f"{branch_id}::DP01",
        "source": {
            "branch_id": branch_id,
            "spec_id": contract["spec_id"],
            "kind": contract["kind"],
            "origin": contract["origin"],
            "gwt_index": contract["gwt_index"],
            "gwt": deepcopy(contract["gwt"]),
            "source_rule_context": deepcopy(contract["source_rule_context"]),
            "source_branch_test_contract_id": contract["branch_test_contract_id"],
            "source_branch_test_contract_fingerprint": contract[
                "branch_test_contract_fingerprint"
            ],
        },
        "test_point_contract": {
            "given_logic": {
                "operator": logical_form["operator"],
                "active_condition_ids": deepcopy(active_condition_ids),
                "resolution_basis": logical_form["resolution_basis"],
            },
            "when_trigger": deepcopy(contract["when_contract"]),
            "oracle_observation_window": {
                "opens_after": "when_trigger_emitted",
                "closes_at": "terminal_driver_status",
                "precondition_evidence_cutoff": "strictly_before_when",
            },
        },
        "fixture_binding_plan": {
            "binding_status": "requires_driver_binding",
            "fixture_root": fixture["fixture_root"],
            "candidate_pool": {
                "candidate_count": fixture["candidate_count"],
                "candidate_fields": deepcopy(fixture["candidate_fields"]),
                "candidate_source": deepcopy(fixture["candidate_source"]),
            },
            "precondition_bindings": deepcopy(evidence_bindings),
            "validation_program": {
                "operator": logical_form["operator"],
                "operand_condition_ids": deepcopy(active_condition_ids),
                "required_result": True,
            },
            "required_driver_binding_names": required_driver_bindings,
            "required_environment_references": environment_refs,
            "binding_policy": {
                "select_one_existing_candidate": True,
                "exact_identity_across_fixture_interaction_and_oracle": True,
                "evaluate_all_active_given_conditions": True,
                "reject_unknown_precondition_results": True,
                "concrete_candidate_selection_deferred": True,
            },
        },
        "interaction_plan": {
            "controller_kind": "milestone_graph",
            "user_goal": contract["when_contract"]["user_goal"],
            "surface_fact_contracts": surface_facts,
            "semantic_stimulus_requirements": semantic_stimuli,
            "operation_input_probe_requirements": operation_input_probes,
            "operation_argument_fact_requirements": [
                {
                    "tool_name": tool,
                    "source": "bound_fixture_and_sut_tool_schema",
                    "status": "requires_driver_binding",
                }
                for tool in target_tools
            ],
            "required_driver_binding_names": required_driver_bindings,
            "milestones": [
                {"milestone_id": "fixture_bound", "required": True},
                {"milestone_id": "given_preconditions_verified", "required": True},
                {"milestone_id": "target_operation_requested", "required": True},
                {
                    "milestone_id": "required_driver_bindings_available",
                    "required": bool(required_driver_bindings),
                },
                {
                    "milestone_id": "required_user_facts_available",
                    "required": bool(surface_facts),
                },
                {
                    "milestone_id": "confirmation_completed",
                    "required": (
                        True if confirmation_required else "when_requested_by_agent"
                    ),
                },
                {"milestone_id": "decision_opportunity_reached", "required": True},
            ],
            "reactive_action_policy": {
                "answer_agent_fact_requests_from_bound_facts_only": True,
                "confirm_the_requested_operation_when_asked": True,
                "do_not_initiate_unrelated_operations": True,
                "do_not_assert_policy_eligibility": True,
            },
        },
        "reachability_contract": {
            "decision_opportunity_expression": {
                "all_of": [
                    "fixture_bound",
                    "given_preconditions_verified",
                    "target_operation_requested",
                    "required_driver_bindings_available_if_applicable",
                    "required_user_facts_available_if_applicable",
                ]
            },
            "oracle_event_plans": event_plans,
            "target_tool_names": target_tools,
            "object_binding_policy": "exact_identity",
            "stop_statuses": [
                "oracle_observation_observed",
                "agent_final_refusal",
                "agent_transfer",
                "max_turns_exceeded",
                "no_progress",
                "semantic_drift",
            ],
        },
        "oracle_plan": {
            "checks": deepcopy(oracle_checks),
            "aggregation": {
                "kind": "all_of",
                "pass_when": "all_checks_pass",
                "fail_when": "any_check_fails",
                "incomplete_when": "any_check_is_not_executable_or_trace_is_incomplete",
            },
            "surface_realizer_visibility": "forbidden",
        },
        "boundary_plan": {
            "opportunities": boundary_opportunities,
            "probe_generation_status": (
                "available" if boundary_opportunities else "not_applicable"
            ),
            "cross_branch_contrast_selection": "deferred",
        },
        "surface_realizer_contract": {
            "input_allowlist": [
                "interaction_plan.user_goal",
                "interaction_plan.surface_fact_contracts",
                "bound_fixture_identity_and_facts",
                "bound_operation_argument_fact_bundle",
                "approved_variation_profile",
            ],
            "forbidden_inputs": [
                "source.gwt.then",
                "oracle_plan",
                "oracle_expected_observations",
                "policy_correctness_conclusion",
            ],
            "semantic_invariants": [
                "same_user_goal",
                "same_bound_object_identity",
                "same_given_truth_assignment",
                "same_user_supplied_facts",
                "no_policy_answer_leakage",
                "no_additional_business_goal",
            ],
        },
        "variation_contract": {
            "status": "baseline_only",
            "baseline_surface": "canonical_truthful_cooperative_user",
            "semantic_mutation_allowed": False,
            "guidance_binding": "deferred_until_bound_baseline_is_reachable",
        },
        "binder_requirements": {
            "required_inputs": [
                "fresh_fixture_database_or_snapshot",
                "sut_tool_schema",
                "environment_clock_if_referenced",
                "run_budget_configuration",
                "semantic_stimulus_profile_if_required",
            ],
            "must_materialize": [
                "one_concrete_fixture_identity",
                "all_required_driver_binding_values",
                "operation_argument_fact_bundle",
                "given_precondition_witnesses",
                "runtime_oracle_scope_values",
            ],
            "must_not_change": [
                "given_logic",
                "when_user_goal",
                "surface_fact_contracts",
                "oracle_checks",
            ],
        },
        "generation_readiness": {
            "status": "ready_for_binding",
            "concrete_fixture_bound": False,
            "surface_prompt_generated": False,
            "runtime_execution_allowed": False,
        },
        "compiler_checks": {
            "source_branch_admitted": True,
            "given_logic_closed": True,
            "fixture_candidate_pool_available": True,
            "when_trigger_closed": True,
            "oracle_checks_executable": True,
            "concrete_fixture_not_selected": True,
            "surface_realizer_cannot_read_oracle": True,
            "llm_calls": 0,
        },
    }
    plan["driver_plan_fingerprint"] = content_sha256(plan)
    return plan


def _summary(plans: Sequence[Mapping[str, Any]], excluded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    kind_counts = Counter(plan["source"]["kind"] for plan in plans)
    tool_counts = Counter(
        tool for plan in plans for tool in plan["reachability_contract"]["target_tool_names"]
    )
    excluded_counts = Counter(item["primary_status"] for item in excluded)
    oracle_modes = Counter(
        check.get("evaluation_mode")
        for plan in plans
        for check in plan["oracle_plan"]["checks"]
    )
    return {
        "input_branch_count": len(plans) + len(excluded),
        "plan_count": len(plans),
        "excluded_branch_count": len(excluded),
        "plan_kind_counts": dict(sorted(kind_counts.items())),
        "target_tool_name_counts": dict(sorted(tool_counts.items())),
        "plans_with_surface_facts": sum(
            bool(plan["interaction_plan"]["surface_fact_contracts"]) for plan in plans
        ),
        "plans_with_semantic_stimuli": sum(
            bool(plan["interaction_plan"]["semantic_stimulus_requirements"])
            for plan in plans
        ),
        "plans_with_operation_input_probes": sum(
            bool(plan["interaction_plan"]["operation_input_probe_requirements"])
            for plan in plans
        ),
        "oracle_evaluation_mode_counts": dict(sorted(oracle_modes.items())),
        "plans_with_boundary_opportunities": sum(
            bool(plan["boundary_plan"]["opportunities"]) for plan in plans
        ),
        "excluded_primary_status_counts": dict(sorted(excluded_counts.items())),
        "llm_calls": 0,
    }


def lower_branch_test_contracts_to_driver_plans(
    branch_test_contract_set: Mapping[str, Any],
) -> dict[str, Any]:
    source = validate_branch_test_contract_set(branch_test_contract_set)
    plans = []
    excluded = []
    for contract in source["contracts"]:
        if contract["admission_status"] == "ready_for_driver":
            plans.append(_lower_one(contract))
        else:
            item = {
                "branch_id": contract["branch_id"],
                "primary_status": contract["primary_status"],
                "required_work": deepcopy(contract["required_work"]),
                "source_branch_test_contract_fingerprint": contract[
                    "branch_test_contract_fingerprint"
                ],
            }
            item["excluded_branch_fingerprint"] = content_sha256(item)
            excluded.append(item)
    result = {
        "schema_version": PLAN_SET_VERSION,
        "lowering_version": LOWERING_VERSION,
        "plans": plans,
        "excluded_branches": excluded,
        "summary": _summary(plans, excluded),
        "source_branch_test_contract_set_fingerprint": source[
            "branch_test_contract_set_fingerprint"
        ],
    }
    result["driver_plan_set_fingerprint"] = content_sha256(result)
    return validate_generic_driver_plan_set(result)


def validate_generic_driver_plan_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$generic_driver_plan_set")))
    supplied = result.pop("driver_plan_set_fingerprint", None)
    if result.get("schema_version") != PLAN_SET_VERSION or supplied != content_sha256(result):
        raise ThenAtomizationError("invalid generic Driver plan set")
    plans = result.get("plans")
    excluded = result.get("excluded_branches")
    if not isinstance(plans, list) or not isinstance(excluded, list):
        raise ThenAtomizationError("generic Driver plan set arrays are invalid")
    plan_branches = []
    for raw in plans:
        plan = deepcopy(dict(_mapping(raw, "$.plans[]")))
        fingerprint = plan.pop("driver_plan_fingerprint", None)
        if (
            plan.get("schema_version") != PLAN_VERSION
            or plan.get("lowering_version") != LOWERING_VERSION
            or fingerprint != content_sha256(plan)
        ):
            raise ThenAtomizationError("invalid generic Driver plan")
        branch_id = (plan.get("source") or {}).get("branch_id")
        if plan.get("driver_plan_id") != f"{branch_id}::DP01":
            raise ThenAtomizationError("generic Driver plan identity mismatch")
        if plan.get("generation_readiness", {}).get("status") != "ready_for_binding":
            raise ThenAtomizationError("generic Driver plan readiness mismatch")
        if plan.get("compiler_checks", {}).get("llm_calls") != 0:
            raise ThenAtomizationError("generic Driver lowering must not call an LLM")
        if "oracle_plan" in plan["surface_realizer_contract"]["input_allowlist"]:
            raise ThenAtomizationError("surface realizer may not consume Oracle output")
        plan_branches.append(branch_id)
    excluded_branches = []
    for raw in excluded:
        item = deepcopy(dict(_mapping(raw, "$.excluded_branches[]")))
        fingerprint = item.pop("excluded_branch_fingerprint", None)
        if fingerprint != content_sha256(item) or not item.get("required_work"):
            raise ThenAtomizationError("invalid excluded Driver branch")
        excluded_branches.append(item.get("branch_id"))
    all_branches = plan_branches + excluded_branches
    if len(all_branches) != len(set(all_branches)):
        raise ThenAtomizationError("generic Driver plan branch membership overlaps")
    if result.get("summary") != _summary(plans, excluded):
        raise ThenAtomizationError("generic Driver plan summary mismatch")
    result["driver_plan_set_fingerprint"] = supplied
    return result
