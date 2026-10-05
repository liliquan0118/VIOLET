"""Compile accepted policy artifacts and a Test Intent into an unbound plan."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    CAP_COVERAGE_MATCH_MANIFEST,
    CAP_DECISION_CONFIGURATION_MODEL,
    CAP_EXECUTION_MATCHING_CONTRACTS,
    CAP_OPERATION_POLICY_CLOSURES,
    artifact_document_for_capability,
    artifact_identity_view,
    content_sha256,
    provenance_view,
    resolve_artifact_bundle_file,
)
from .experiment_planner import derive_test_design, plan_experiment
from .profiles import (
    load_configuration_file,
    profile_identity,
    validate_compiler_profile,
    validate_sut_adapter_profile,
)
from .selected_cell import SelectedCellError, materialize_selected_cell_contract, selected_cell
from .source_loader import load_source_spec_record
from .source_resolver import resolve_source_spec_target
from .source_contracts import make_resolved_spec_target
from .evaluation_resolver import resolve_evaluation_target
from .capabilities import DEFAULT_PIPELINE_CAPABILITIES
from .target_selector import (
    ISOLATED_ATOMIC_BRANCH_WITNESS,
    select_target_cells,
)


COMPILER_REQUEST_SCHEMA_VERSION = "agentspectesting.compiler-request/v0.1"
COMPILER_REQUEST_SCHEMA_VERSION_V2 = "agentspectesting.compiler-request/v0.2"
COMPILER_REQUEST_SCHEMA_VERSION_V3 = "agentspectesting.compiler-request/v0.3"
COMPILED_PLAN_SCHEMA_VERSION = "agentspectesting.compiled-test-plan/v0.1"
COMPILED_PLAN_SCHEMA_VERSION_V2 = "agentspectesting.compiled-test-plan/v0.2"
COMPILED_PLAN_SCHEMA_VERSION_V3 = "agentspectesting.compiled-test-plan/v0.3"
COMPILER_VERSION = "accepted-model-compiler/v0.1"
COMPILER_VERSION_V2 = "source-resolved-accepted-model-compiler/v0.2"
COMPILER_VERSION_V3 = "profile-driven-source-resolved-compiler/v0.3"


class CompilerRequestError(ValueError):
    """Raised when a Test Intent is inconsistent with accepted policy artifacts."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompilerRequestError(f"{path} must be an object")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompilerRequestError(f"{path} must be a non-empty string")
    return value


def _condition_control_role(condition: Mapping[str, Any]) -> str:
    roles = set((condition.get("runtime_binding") or {}).get("semantic_roles") or [])
    if "user_fact" in roles:
        return "dialogue_controlled"
    if "decision_operation_selector" in roles:
        return "interaction_triggered"
    if "policy_derived" in roles:
        return "policy_derived"
    if "domain_state" in roles:
        return "fixture_controlled"
    return "unclassified"


def _condition_observation_role(condition: Mapping[str, Any]) -> str:
    roles = set((condition.get("runtime_binding") or {}).get("semantic_roles") or [])
    if "user_fact" in roles:
        return "driver_makes_fact_available"
    if "decision_operation_selector" in roles:
        return "driver_requests_target_operation"
    if "policy_derived" in roles:
        return "verifier_derives_from_policy_evidence"
    if "domain_state" in roles:
        return "environment_and_tool_evidence_available"
    return "bounded_evidence_required"


def _compiled_condition(condition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "predicate_id": condition.get("predicate_id"),
        "required_truth_value": condition.get("required_truth_value"),
        "surface_text": condition.get("surface_text"),
        "evaluation_class": condition.get("evaluation_class"),
        "semantic_roles": deepcopy(
            (condition.get("runtime_binding") or {}).get("semantic_roles") or []
        ),
        "temporal_role": condition.get("temporal_role"),
        "control_role": _condition_control_role(condition),
        "observation_role": _condition_observation_role(condition),
        "obligation_role": "configuration_witness_only",
        "typed_expression": deepcopy(condition.get("typed_expression")),
        "runtime_binding": deepcopy(condition.get("runtime_binding")),
        "required_observation_values": deepcopy(
            condition.get("required_observation_values") or []
        ),
        "literal_constraints": deepcopy(condition.get("literal_constraints") or []),
    }


def _target_from_contract(
    *,
    contract: Mapping[str, Any],
    cell: Mapping[str, Any],
    target_name: str,
    focus_predicate_ref: str,
) -> dict[str, Any]:
    operation = _mapping(contract.get("target_operation"), "$.target_operation")
    conditions = [_compiled_condition(value) for value in contract.get("required_conditions") or []]
    preconditions = [
        value["predicate_id"]
        for value in conditions
        if value["temporal_role"] == "pre_or_activation_condition"
    ]
    postconditions = [
        value["predicate_id"]
        for value in conditions
        if value["temporal_role"] == "post_operation_state"
    ]
    return {
        "target_name": target_name,
        "subject": {
            "operation_policy_family_id": contract.get("operation_policy_family_id"),
            "canonical_operation_ids": deepcopy(operation.get("canonical_operation_ids") or []),
            "canonical_operation_context_ids": deepcopy(
                operation.get("canonical_operation_context_ids") or []
            ),
            "operation_selector_id": operation.get("operation_selector_id"),
            "operation_expressions": deepcopy(operation.get("operation_expressions") or []),
            "verified_tool_names": deepcopy(operation.get("verified_tool_names") or []),
        },
        "focal_configuration": {
            "decision_region_id": cell.get("decision_region_id"),
            "coverage_cell_id": contract.get("target_coverage_cell_id"),
            "cell_kind": contract.get("cell_kind"),
            "focus_predicate_ref": focus_predicate_ref,
            "required_factor_values": [
                {
                    "predicate_id": value["predicate_id"],
                    "expected_truth_value": value["required_truth_value"],
                }
                for value in conditions
            ],
            "dont_care_predicate_ids": deepcopy(
                contract.get("dont_care_predicate_ids") or []
            ),
            "expected_operation_decision": contract.get("expected_operation_decision"),
        },
        "lifecycle_scope": {
            "pre_or_activation": preconditions,
            "at_decision_opportunity": preconditions,
            "focal_execution": deepcopy(operation.get("verified_tool_names") or []),
            "post_operation_state": postconditions,
        },
        "conditions": conditions,
        "generation_contract": deepcopy(dict(contract)),
    }


def _validate_target_intent(
    *,
    target_spec: Mapping[str, Any],
    target: Mapping[str, Any],
    path: str,
) -> None:
    focus = _nonempty_string(target_spec.get("focus_predicate_ref"), f"{path}.focus_predicate_ref")
    desired_truth = target_spec.get("desired_truth")
    if not isinstance(desired_truth, bool):
        raise CompilerRequestError(f"{path}.desired_truth must be boolean")
    matches = [
        value
        for value in target["conditions"]
        if value.get("predicate_id") == focus
    ]
    if len(matches) != 1:
        raise CompilerRequestError(
            f"{path}: focus predicate {focus!r} occurs {len(matches)} times in selected cell"
        )
    if matches[0]["required_truth_value"] is not desired_truth:
        raise CompilerRequestError(
            f"{path}: desired truth {desired_truth!r} conflicts with selected cell value "
            f"{matches[0]['required_truth_value']!r}"
        )
    desired_decision = target_spec.get("desired_policy_decision")
    actual_decision = target["focal_configuration"]["expected_operation_decision"]
    if desired_decision is not None and desired_decision != actual_decision:
        raise CompilerRequestError(
            f"{path}: desired decision {desired_decision!r} conflicts with accepted cell "
            f"decision {actual_decision!r}"
        )


def _compile_one_target(
    *,
    domain: str,
    manifest: dict[str, Any],
    request_id: str,
    target_name: str,
    target_spec: Mapping[str, Any],
    path: str,
) -> dict[str, Any]:
    coverage_cell_id = _nonempty_string(
        target_spec.get("coverage_cell_id"), f"{path}.coverage_cell_id"
    )
    focus = _nonempty_string(
        target_spec.get("focus_predicate_ref"), f"{path}.focus_predicate_ref"
    )
    cell = selected_cell(manifest, coverage_cell_id)
    contract = materialize_selected_cell_contract(
        domain=domain,
        manifest=manifest,
        coverage_cell_id=coverage_cell_id,
        selection_id=f"{request_id}.{target_name}",
    )
    target = _target_from_contract(
        contract=contract,
        cell=cell,
        target_name=target_name,
        focus_predicate_ref=focus,
    )
    _validate_target_intent(target_spec=target_spec, target=target, path=path)
    return target


def _evaluate_comparison(operator: str, left: float, right: float) -> bool:
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    raise CompilerRequestError(f"unsupported boundary comparison operator: {operator!r}")


def _duration_seconds(value: Mapping[str, Any]) -> float:
    raw = value.get("value")
    unit = value.get("unit")
    if not isinstance(raw, (int, float)):
        raise CompilerRequestError("duration threshold value must be numeric")
    multipliers = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
    if unit not in multipliers:
        raise CompilerRequestError(f"unsupported duration unit: {unit!r}")
    return float(raw) * multipliers[unit]


def _focus_condition(target: Mapping[str, Any]) -> Mapping[str, Any]:
    focus = target["focal_configuration"]["focus_predicate_ref"]
    return next(value for value in target["conditions"] if value["predicate_id"] == focus)


def _compile_boundary_refinements(
    *,
    refinements: list[Any],
    targets: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    compiled = []
    for index, raw in enumerate(refinements):
        path = f"$.test_intent.boundary_refinements[{index}]"
        item = _mapping(raw, path)
        refinement_id = _nonempty_string(item.get("refinement_id"), f"{path}.refinement_id")
        target_name = _nonempty_string(item.get("target_name"), f"{path}.target_name")
        if target_name not in targets:
            raise CompilerRequestError(f"{path}.target_name references unknown target {target_name!r}")
        offset = item.get("offset_seconds_from_threshold")
        if not isinstance(offset, (int, float)):
            raise CompilerRequestError(
                f"{path}.offset_seconds_from_threshold must be numeric"
            )
        target = targets[target_name]
        condition = _focus_condition(target)
        expression = _mapping(condition.get("typed_expression"), f"{path}.typed_expression")
        if expression.get("kind") != "comparison":
            raise CompilerRequestError(f"{path}: focus predicate is not a comparison")
        right = _mapping(expression.get("right"), f"{path}.typed_expression.right")
        if right.get("kind") != "literal" or right.get("value_type") != "duration":
            raise CompilerRequestError(f"{path}: comparison threshold is not a duration literal")
        threshold_seconds = _duration_seconds(right)
        observed_seconds = threshold_seconds + float(offset)
        truth = _evaluate_comparison(
            str(expression.get("operator")), observed_seconds, threshold_seconds
        )
        required_truth = condition["required_truth_value"]
        if truth is not required_truth:
            raise CompilerRequestError(
                f"{path}: computed predicate truth {truth!r} does not match target "
                f"{target_name!r} required truth {required_truth!r}"
            )
        compiled.append(
            {
                "refinement_id": refinement_id,
                "target_name": target_name,
                "target_coverage_cell_id": target["focal_configuration"]["coverage_cell_id"],
                "focus_predicate_ref": condition["predicate_id"],
                "operator": expression.get("operator"),
                "threshold_seconds": threshold_seconds,
                "offset_seconds_from_threshold": offset,
                "observed_duration_seconds": observed_seconds,
                "predicate_truth": truth,
                "expected_operation_decision": target["focal_configuration"][
                    "expected_operation_decision"
                ],
            }
        )
    return compiled


def _fixture_requirements(target: Mapping[str, Any]) -> dict[str, Any]:
    conditions = target["conditions"]
    return {
        "binding_status": "unbound",
        "concrete_object_ids_are_forbidden": True,
        "state_constraints": [
            deepcopy(value)
            for value in conditions
            if value["control_role"] == "fixture_controlled"
        ],
        "dialogue_controlled_constraints": [
            deepcopy(value)
            for value in conditions
            if value["control_role"] == "dialogue_controlled"
        ],
        "policy_derived_constraints": [
            deepcopy(value)
            for value in conditions
            if value["control_role"] == "policy_derived"
        ],
        "interaction_trigger_constraints": [
            deepcopy(value)
            for value in conditions
            if value["control_role"] == "interaction_triggered"
        ],
        "isolation_predicate_truth": [
            {
                "predicate_id": value["predicate_id"],
                "required_truth_value": value["required_truth_value"],
            }
            for value in conditions
        ],
        "postcondition_transition_sources": [
            deepcopy(value)
            for value in conditions
            if value["temporal_role"] == "post_operation_state"
        ],
    }


def _interaction_requirements(target: Mapping[str, Any]) -> dict[str, Any]:
    contract = target["generation_contract"]
    return {
        "plan_kind": "milestone_graph",
        "milestones": [
            {"milestone_id": "fixture_valid", "required": True},
            {"milestone_id": "target_operation_requested", "required": True},
            {"milestone_id": "requesting_identity_bound", "required": True},
            {"milestone_id": "target_entity_bound", "required": True},
            {"milestone_id": "required_user_facts_available", "required": True},
            {"milestone_id": "required_agent_evidence_available", "required": True},
            {
                "milestone_id": "required_confirmation_completed",
                "required": "when_required_by_prerequisite_or_agent_request",
            },
            {"milestone_id": "decision_opportunity_reached", "required": True},
        ],
        "required_prerequisites": deepcopy(contract.get("required_prerequisites") or []),
        "natural_language_generation_allowed": False,
    }


def _reachability_contract(target: Mapping[str, Any]) -> dict[str, Any]:
    subject = target["subject"]
    return {
        "decision_opportunity_expression": {
            "all_of": [
                "fixture_valid",
                "target_operation_requested",
                "requesting_identity_bound",
                "target_entity_bound",
                "required_user_facts_available",
                "required_agent_evidence_available",
                "required_confirmation_completed_if_applicable",
            ]
        },
        "focal_evidence": {
            "verified_target_tool_names": deepcopy(subject["verified_tool_names"]),
            "allow_explicit_final_policy_decision": True,
            "allow_required_behavior_evidence": True,
        },
        "object_binding_policy": "exact_identity",
        "evidence_cutoff": "before_or_at_focal_execution",
        "stop_statuses": [
            "decision_opportunity_reached",
            "target_action_observed",
            "agent_refused_before_target",
            "transferred_before_target",
            "max_turns_exceeded",
            "semantic_drift",
        ],
    }


def _oracle_contract(target: Mapping[str, Any]) -> dict[str, Any]:
    contract = target["generation_contract"]
    return {
        "fixture_oracle": {
            "required_predicate_truth": deepcopy(
                contract["coverage_oracle"]["required_predicate_truth"]
            ),
            "exact_target_object_binding": True,
        },
        "coverage_oracle": deepcopy(contract["coverage_oracle"]),
        "reachability_oracle": _reachability_contract(target),
        "correctness_oracle": {
            "status": contract.get("correctness_oracle_status"),
            "expected_operation_decision": contract.get("expected_operation_decision"),
            "required_prerequisites": deepcopy(contract.get("required_prerequisites") or []),
            "required_outcome_monitors": deepcopy(
                contract.get("required_outcome_monitors") or []
            ),
            "required_path_match_contracts": deepcopy(
                contract.get("required_path_match_contracts") or []
            ),
        },
    }


def _variation_contract(request: Mapping[str, Any]) -> dict[str, Any]:
    policy = deepcopy(dict(_mapping(request.get("variation_policy") or {}, "$.variation_policy")))
    policy.setdefault("semantic_invariants", [])
    policy.setdefault("mutable_dimensions", [])
    policy.setdefault("forbidden_mutations", [])
    policy.setdefault("max_changed_dimensions_per_candidate", 1)
    policy.setdefault("require_baseline", True)
    policy.setdefault("require_semantic_invariant_check", True)
    return policy


def _guidance_contract(variation: Mapping[str, Any]) -> dict[str, Any]:
    mutable = variation.get("mutable_dimensions") or []
    return {
        "ordered_progress_states": [
            "fixture_valid",
            "target_configuration_satisfied",
            "target_entity_bound",
            "required_facts_available",
            "confirmation_completed",
            "decision_opportunity_reached",
            "target_action_observed",
            "terminal_state_correct",
        ],
        "allowed_mutation_catalog": deepcopy(mutable),
        "mutation_selection_policy": "earliest_unresolved_progress_state",
        "max_changed_dimensions_per_candidate": variation.get(
            "max_changed_dimensions_per_candidate", 1
        ),
        "semantic_drift_rejection_rules": deepcopy(
            variation.get("forbidden_mutations") or []
        ),
        "stop_conditions": [
            "reproducible_minimal_valid_failure_found",
            "compiled_candidate_budget_exhausted",
            "no_fixture_can_satisfy_target",
            "all_remaining_candidates_violate_invariants",
        ],
    }


def _validate_target_artifact_closure(
    *,
    target: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
) -> None:
    family_id = target["subject"]["operation_policy_family_id"]
    cell_id = target["focal_configuration"]["coverage_cell_id"]
    closures = artifact_document_for_capability(
        resolved_artifacts,
        CAP_OPERATION_POLICY_CLOSURES,
    )
    decisions = artifact_document_for_capability(
        resolved_artifacts,
        CAP_DECISION_CONFIGURATION_MODEL,
    )
    matching = artifact_document_for_capability(
        resolved_artifacts,
        CAP_EXECUTION_MATCHING_CONTRACTS,
    )
    closure_family_ids = {
        value.get("operation_policy_family_id")
        for value in closures.get("operation_policy_families") or []
    }
    matching_family_ids = {
        value.get("operation_policy_family_id")
        for value in matching.get("operation_family_contracts") or []
    }
    decision_family_ids = set(decisions.get("selected_operation_policy_family_ids") or [])
    if family_id not in closure_family_ids:
        raise CompilerRequestError(
            f"selected family {family_id!r} is absent from operation policy closures"
        )
    if family_id not in matching_family_ids:
        raise CompilerRequestError(
            f"selected family {family_id!r} is absent from execution matching contracts"
        )
    if family_id not in decision_family_ids:
        raise CompilerRequestError(
            f"selected family {family_id!r} is absent from decision configuration model"
        )
    denominator = decisions.get("coverage_denominator") or {}
    decision_cells = set(denominator.get("decision_configuration_cell_ids") or [])
    unconditional_cells = set(denominator.get("unconditional_requirement_cell_ids") or [])
    if cell_id not in decision_cells | unconditional_cells:
        raise CompilerRequestError(
            f"selected cell {cell_id!r} is absent from decision configuration denominator"
        )


def _fingerprint_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Remove machine-local locator paths while preserving content identity."""

    payload = deepcopy(dict(plan))
    payload.pop("compiled_plan_fingerprint", None)
    provenance = payload.get("provenance") or {}
    provenance.pop("artifact_bundle_path", None)
    for artifact in (provenance.get("artifacts") or {}).values():
        artifact.pop("path", None)
    return payload


def _compile_test_plan_v1(
    request: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile one legacy selected-cell request without binding a fixture."""

    if request.get("schema_version") != COMPILER_REQUEST_SCHEMA_VERSION:
        raise CompilerRequestError(
            f"unsupported compiler request schema: {request.get('schema_version')!r}"
        )
    request_id = _nonempty_string(request.get("request_id"), "$.request_id")
    domain = _nonempty_string(request.get("domain"), "$.domain")
    if domain != resolved_artifacts.get("domain"):
        raise CompilerRequestError(
            f"request domain {domain!r} does not match artifact domain "
            f"{resolved_artifacts.get('domain')!r}"
        )
    manifest = deepcopy(
        artifact_document_for_capability(
            resolved_artifacts,
            CAP_COVERAGE_MATCH_MANIFEST,
        )
    )
    intent = _mapping(request.get("test_intent"), "$.test_intent")
    primary_spec = _mapping(intent.get("primary_target"), "$.test_intent.primary_target")

    targets: dict[str, dict[str, Any]] = {
        "primary": _compile_one_target(
            domain=domain,
            manifest=manifest,
            request_id=request_id,
            target_name="primary",
            target_spec=primary_spec,
            path="$.test_intent.primary_target",
        )
    }
    raw_contrasts = intent.get("contrast_targets") or []
    if not isinstance(raw_contrasts, list):
        raise CompilerRequestError("$.test_intent.contrast_targets must be an array")
    for index, raw in enumerate(raw_contrasts):
        path = f"$.test_intent.contrast_targets[{index}]"
        spec = _mapping(raw, path)
        name = _nonempty_string(spec.get("target_name"), f"{path}.target_name")
        if name == "primary" or name in targets:
            raise CompilerRequestError(f"{path}.target_name is duplicated: {name!r}")
        targets[name] = _compile_one_target(
            domain=domain,
            manifest=manifest,
            request_id=request_id,
            target_name=name,
            target_spec=spec,
            path=path,
        )

    for target in targets.values():
        _validate_target_artifact_closure(
            target=target,
            resolved_artifacts=resolved_artifacts,
        )

    refinements = intent.get("boundary_refinements") or []
    if not isinstance(refinements, list):
        raise CompilerRequestError("$.test_intent.boundary_refinements must be an array")
    compiled_refinements = _compile_boundary_refinements(
        refinements=refinements,
        targets=targets,
    )

    primary = targets["primary"]
    variation = _variation_contract(request)
    semantic_conditions = [
        value["predicate_id"]
        for value in primary["conditions"]
        if value["evaluation_class"] != "mechanical"
    ]
    diagnostics = []
    if semantic_conditions:
        diagnostics.append(
            {
                "diagnostic_id": "COMPILER::READINESS::BOUNDED_SEMANTIC",
                "severity": "warning",
                "stage": "condition_classification",
                "code": "bounded_semantic_fixture_design_required",
                "predicate_ids": semantic_conditions,
            }
        )

    plan = {
        "schema_version": COMPILED_PLAN_SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
        "compiled_plan_id": request_id,
        "provenance": provenance_view(resolved_artifacts),
        "source_anchors": [deepcopy(request.get("source_anchor") or {})],
        "test_objective": {
            "failure_hypotheses": deepcopy(intent.get("failure_hypotheses") or []),
            "isolation_policy": intent.get("isolation_policy"),
        },
        "target": deepcopy(primary),
        "contrast_targets": [
            deepcopy(value) for name, value in targets.items() if name != "primary"
        ],
        "boundary_refinements": compiled_refinements,
        "fixture_requirements": _fixture_requirements(primary),
        "interaction_requirements": _interaction_requirements(primary),
        "reachability_contract": _reachability_contract(primary),
        "oracle_contract": _oracle_contract(primary),
        "variation_contract": variation,
        "guidance_contract": _guidance_contract(variation),
        "environment_requirements": deepcopy(
            request.get("environment_capabilities") or {}
        ),
        "generation_readiness": {
            "status": "ready_for_driver_binding",
            "fixture_binding": primary["generation_contract"].get(
                "generation_readiness"
            ),
            "interaction_binding": "milestone_graph_compiled",
            "oracle_binding": primary["generation_contract"].get(
                "correctness_oracle_status"
            ),
            "blocking_issues": [],
            "warnings": [value["code"] for value in diagnostics],
        },
        "budget": deepcopy(request.get("budget") or {}),
        "compiler_diagnostics": diagnostics,
        "compiler_checks": {
            "artifact_hashes_verified": True,
            "selected_cell_exists": True,
            "focus_truth_matches_selected_cell": True,
            "expected_decision_matches_selected_cell": True,
            "concrete_fixture_ids_absent": True,
            "coverage_reachability_correctness_separated": True,
            "llm_calls": 0,
        },
    }
    plan["compiled_plan_fingerprint"] = content_sha256(_fingerprint_payload(plan))
    return plan


def _focus_truth_from_selected_target(
    selected_target: Mapping[str, Any],
    focus_predicate_id: str,
    path: str,
) -> bool:
    matches = [
        value
        for value in selected_target.get("required_factor_values") or []
        if isinstance(value, Mapping)
        and value.get("predicate_id") == focus_predicate_id
    ]
    if len(matches) != 1 or not isinstance(
        matches[0].get("expected_truth_value"), bool
    ):
        raise CompilerRequestError(
            f"{path} must assign focus predicate {focus_predicate_id!r} exactly once"
        )
    return matches[0]["expected_truth_value"]


def _compile_test_plan_v2(
    request: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
    *,
    source_base_dir: str | Path | None,
) -> dict[str, Any]:
    """Resolve a selected source branch and reuse the accepted cell composer."""

    if "test_intent" in request or "source_anchor" in request:
        raise CompilerRequestError(
            "compiler-request/v0.2 must use selected_spec_ref and must not provide "
            "legacy source_anchor or test_intent"
        )
    request_id = _nonempty_string(request.get("request_id"), "$.request_id")
    domain = _nonempty_string(request.get("domain"), "$.domain")
    if domain != resolved_artifacts.get("domain"):
        raise CompilerRequestError(
            f"request domain {domain!r} does not match artifact domain "
            f"{resolved_artifacts.get('domain')!r}"
        )
    selected_spec_ref = _mapping(
        request.get("selected_spec_ref"), "$.selected_spec_ref"
    )
    selection_policy = request.get(
        "target_selection_policy", ISOLATED_ATOMIC_BRANCH_WITNESS
    )
    if not isinstance(selection_policy, str) or not selection_policy:
        raise CompilerRequestError(
            "$.target_selection_policy must be a non-empty string"
        )

    source_record = load_source_spec_record(
        selected_spec_ref,
        base_dir=source_base_dir,
    )
    resolved_target = resolve_source_spec_target(source_record, resolved_artifacts)
    if resolved_target["resolution_status"] != "resolved_unique":
        raise CompilerRequestError(
            "source resolution did not produce a unique model binding: "
            f"{resolved_target['resolution_status']}; "
            f"diagnostics={resolved_target['diagnostics']!r}"
        )
    selected_target = select_target_cells(
        resolved_target,
        resolved_artifacts,
        selection_policy=selection_policy,
    )
    focus_id = selected_target["focus_predicate_id"]
    primary = selected_target["primary_target"]
    contrast = selected_target["contrast_target"]

    probe_configuration = _mapping(
        request.get("probe_configuration") or {}, "$.probe_configuration"
    )
    refinements = probe_configuration.get("boundary_refinements") or []
    if not isinstance(refinements, list):
        raise CompilerRequestError(
            "$.probe_configuration.boundary_refinements must be an array"
        )
    search_objective = _mapping(
        request.get("search_objective") or {}, "$.search_objective"
    )
    failure_hypotheses = search_objective.get("failure_hypotheses") or []
    if not isinstance(failure_hypotheses, list):
        raise CompilerRequestError(
            "$.search_objective.failure_hypotheses must be an array"
        )

    adapted_request = {
        "schema_version": COMPILER_REQUEST_SCHEMA_VERSION,
        "request_id": request_id,
        "domain": domain,
        "source_anchor": deepcopy(dict(selected_spec_ref)),
        "test_intent": {
            "primary_target": {
                "coverage_cell_id": primary["coverage_cell_id"],
                "focus_predicate_ref": focus_id,
                "desired_truth": _focus_truth_from_selected_target(
                    primary, focus_id, "$.selected_coverage_target.primary_target"
                ),
                "desired_policy_decision": primary[
                    "expected_operation_decision"
                ],
            },
            "contrast_targets": [
                {
                    "target_name": "contrast",
                    "coverage_cell_id": contrast["coverage_cell_id"],
                    "focus_predicate_ref": focus_id,
                    "desired_truth": _focus_truth_from_selected_target(
                        contrast,
                        focus_id,
                        "$.selected_coverage_target.contrast_target",
                    ),
                    "desired_policy_decision": contrast[
                        "expected_operation_decision"
                    ],
                }
            ],
            "isolation_policy": selection_policy,
            "failure_hypotheses": deepcopy(failure_hypotheses),
            "boundary_refinements": deepcopy(refinements),
        },
        "environment_capabilities": deepcopy(
            request.get("environment_capabilities") or {}
        ),
        "variation_policy": deepcopy(request.get("variation_policy") or {}),
        "budget": deepcopy(request.get("budget") or {}),
    }
    plan = _compile_test_plan_v1(adapted_request, resolved_artifacts)
    plan["schema_version"] = COMPILED_PLAN_SCHEMA_VERSION_V2
    plan["compiler_version"] = COMPILER_VERSION_V2
    plan["source_anchors"] = [deepcopy(dict(selected_spec_ref))]
    plan["source_spec_record"] = deepcopy(source_record)
    plan["resolved_spec_target"] = deepcopy(resolved_target)
    plan["selected_coverage_target"] = deepcopy(selected_target)
    plan["test_objective"]["selection_policy"] = selection_policy
    plan["test_objective"]["target_origin"] = "derived_from_selected_spec_ref"
    plan["compiler_checks"].update(
        {
            "source_anchor_loaded": True,
            "source_record_fingerprint_verified": True,
            "source_to_model_resolution_unique": True,
            "target_cell_derived_not_supplied": True,
            "minimal_contrast_derived_not_supplied": True,
            "source_resolution_artifact_identity_matches": True,
        }
    )
    plan["compiled_plan_fingerprint"] = content_sha256(_fingerprint_payload(plan))
    return plan


def _derived_request_id(
    *,
    source_record: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
    compiler_profile: Mapping[str, Any],
    sut_profile: Mapping[str, Any],
) -> str:
    artifact_identity = artifact_identity_view(resolved_artifacts)
    identity = {
        "source_record_fingerprint": source_record["source_record_fingerprint"],
        "artifact_bundle_id": artifact_identity["artifact_bundle_id"],
        "compiler_profile": profile_identity(compiler_profile),
        "sut_adapter_profile": profile_identity(sut_profile),
    }
    branch = str(source_record["branch_id"]).replace("#", "_").replace("/", "_")
    return f"{branch}.auto.{content_sha256(identity)[:16]}"


def _derived_variation_contract(
    design: Mapping[str, Any],
    compiler_profile: Mapping[str, Any],
) -> dict[str, Any]:
    mutation_policy = compiler_profile["mutation_policy"]
    invariants = deepcopy(design["semantic_invariants"])
    rejection_rules = [
        {
            "rule_id": "reject_operation_identity_change",
            "basis_invariant_id": "operation_identity_immutable",
        },
        {
            "rule_id": "reject_target_object_identity_change",
            "basis_invariant_id": "target_object_identity_immutable",
        },
        {
            "rule_id": "reject_immutable_predicate_truth_change",
            "basis": "all predicate invariants with mutation_permission=immutable",
        },
        {
            "rule_id": "reject_focus_change_outside_compiled_probe",
            "basis": "focus predicate is probe_controlled_only",
        },
        {
            "rule_id": "reject_fabricated_tool_or_environment_evidence",
            "basis": "accepted runtime evidence channels",
        },
    ]
    return {
        "derivation_status": "fully_derived_from_profiles_and_target_contract",
        "semantic_invariants": invariants,
        "mutable_dimensions": deepcopy(design["mutation_catalog"]),
        "baseline_values": {
            item["name"]: item["baseline_value"]
            for item in design["mutation_catalog"]
        },
        "rejected_mutation_operators": deepcopy(
            design["rejected_mutation_operators"]
        ),
        "forbidden_mutations": rejection_rules,
        "max_changed_dimensions_per_candidate": mutation_policy[
            "max_changed_dimensions_per_candidate"
        ],
        "require_baseline": mutation_policy["require_baseline"],
        "require_semantic_invariant_check": mutation_policy[
            "require_semantic_invariant_check"
        ],
    }


def _derived_guidance_contract(
    variation: Mapping[str, Any],
    compiler_profile: Mapping[str, Any],
) -> dict[str, Any]:
    policy = compiler_profile["guidance_policy"]
    return {
        "derivation_status": "instantiated_from_guidance_profile",
        "ordered_progress_states": [
            "fixture_valid",
            "target_operation_requested",
            "requesting_identity_bound",
            "target_entity_bound",
            "required_user_facts_available",
            "required_agent_evidence_available",
            "required_confirmation_completed",
            "decision_opportunity_reached",
            "target_action_observed",
            "terminal_state_correct",
        ],
        "allowed_mutation_catalog": deepcopy(variation["mutable_dimensions"]),
        "mutation_selection_policy": policy["selection_strategy"],
        "prefer_single_dimension_changes": bool(
            policy.get("prefer_single_dimension_changes", True)
        ),
        "require_baseline_first": bool(policy.get("require_baseline_first", True)),
        "require_contrast_replay": bool(policy.get("require_contrast_replay", True)),
        "minimize_confirmed_failures": bool(
            policy.get("minimize_confirmed_failures", True)
        ),
        "max_changed_dimensions_per_candidate": variation[
            "max_changed_dimensions_per_candidate"
        ],
        "semantic_drift_rejection_rules": deepcopy(
            variation["forbidden_mutations"]
        ),
        "stop_conditions": deepcopy(policy["stop_conditions"]),
    }


def _compile_test_plan_v3(
    request: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
    *,
    source_base_dir: str | Path | None,
    compiler_profile: Mapping[str, Any] | None,
    sut_adapter_profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compile a source-only case request using versioned general profiles."""

    allowed_request_fields = {"schema_version", "selected_spec_ref"}
    extras = sorted(set(request) - allowed_request_fields)
    if extras:
        raise CompilerRequestError(
            "compiler-request/v0.3 permits only selected_spec_ref as case input; "
            "unsupported fields: " + ", ".join(extras)
        )
    if compiler_profile is None:
        raise CompilerRequestError(
            "compiler-request/v0.3 requires an external compiler profile"
        )
    if sut_adapter_profile is None:
        raise CompilerRequestError(
            "compiler-request/v0.3 requires an external SUT adapter profile"
        )
    compiler_config = validate_compiler_profile(compiler_profile)
    sut_config = validate_sut_adapter_profile(sut_adapter_profile)
    artifact_domain = _nonempty_string(
        resolved_artifacts.get("domain"), "$.resolved_artifacts.domain"
    )
    if sut_config["domain"] != artifact_domain:
        raise CompilerRequestError(
            f"SUT adapter domain {sut_config['domain']!r} does not match artifact "
            f"domain {artifact_domain!r}"
        )

    selected_spec_ref = _mapping(
        request.get("selected_spec_ref"), "$.selected_spec_ref"
    )
    source_record = load_source_spec_record(
        selected_spec_ref,
        base_dir=source_base_dir,
    )
    resolved_evaluation_target = resolve_evaluation_target(
        source_record, resolved_artifacts
    )
    if resolved_evaluation_target["resolution_status"] != "resolved_unique":
        raise CompilerRequestError(
            "evaluation resolution did not produce a unique runtime binding: "
            f"{resolved_evaluation_target['resolution_status']}; "
            f"resolver={resolved_evaluation_target['resolver_id']!r}; "
            f"diagnostics={resolved_evaluation_target['diagnostics']!r}"
        )
    evaluation_binding = resolved_evaluation_target["evaluation_binding"]
    if evaluation_binding.get("target_archetype") != "operation_decision":
        raise CompilerRequestError(
            "no registered compiled-plan lowering for evaluation archetype: "
            f"{evaluation_binding.get('target_archetype')!r}"
        )
    legacy_binding = evaluation_binding.get("legacy_model_binding")
    if not isinstance(legacy_binding, Mapping):
        raise CompilerRequestError(
            "operation-decision evaluation target lacks its compatibility binding"
        )
    resolved_target = make_resolved_spec_target(
        source_spec_record=source_record,
        resolution_status="resolved_unique",
        model_binding=legacy_binding,
        resolution_evidence=resolved_evaluation_target["resolution_evidence"],
        diagnostics=resolved_evaluation_target["diagnostics"],
    )
    experiment_plan = plan_experiment(resolved_target, compiler_config)
    selection_policy = experiment_plan["selected_witness_builder"]
    selected_target = select_target_cells(
        resolved_target,
        resolved_artifacts,
        selection_policy=selection_policy,
    )
    focus_id = selected_target["focus_predicate_id"]
    primary_selection = selected_target["primary_target"]
    contrast_selection = selected_target["contrast_target"]
    request_id = _derived_request_id(
        source_record=source_record,
        resolved_artifacts=resolved_artifacts,
        compiler_profile=compiler_config,
        sut_profile=sut_config,
    )

    adapted_request = {
        "schema_version": COMPILER_REQUEST_SCHEMA_VERSION,
        "request_id": request_id,
        "domain": artifact_domain,
        "source_anchor": deepcopy(dict(selected_spec_ref)),
        "test_intent": {
            "primary_target": {
                "coverage_cell_id": primary_selection["coverage_cell_id"],
                "focus_predicate_ref": focus_id,
                "desired_truth": _focus_truth_from_selected_target(
                    primary_selection,
                    focus_id,
                    "$.selected_coverage_target.primary_target",
                ),
                "desired_policy_decision": primary_selection[
                    "expected_operation_decision"
                ],
            },
            "contrast_targets": [
                {
                    "target_name": "contrast",
                    "coverage_cell_id": contrast_selection["coverage_cell_id"],
                    "focus_predicate_ref": focus_id,
                    "desired_truth": _focus_truth_from_selected_target(
                        contrast_selection,
                        focus_id,
                        "$.selected_coverage_target.contrast_target",
                    ),
                    "desired_policy_decision": contrast_selection[
                        "expected_operation_decision"
                    ],
                }
            ],
            "isolation_policy": selection_policy,
            "failure_hypotheses": [],
            "boundary_refinements": [],
        },
        "environment_capabilities": {},
        "variation_policy": {},
        "budget": {},
    }
    plan = _compile_test_plan_v1(adapted_request, resolved_artifacts)
    primary = plan["target"]
    contrast = plan["contrast_targets"][0]
    design = derive_test_design(
        resolved_target=resolved_target,
        primary=primary,
        contrast=contrast,
        compiler_profile=compiler_config,
        sut_profile=sut_config,
    )
    refinements = design["probe_synthesis"]["boundary_refinements"]
    plan["boundary_refinements"] = _compile_boundary_refinements(
        refinements=refinements,
        targets={"primary": primary, "contrast": contrast},
    )
    plan["probe_contract"] = deepcopy(design["probe_contract"])
    variation = _derived_variation_contract(design, compiler_config)

    plan["schema_version"] = COMPILED_PLAN_SCHEMA_VERSION_V3
    plan["compiler_version"] = COMPILER_VERSION_V3
    plan["source_anchors"] = [deepcopy(dict(selected_spec_ref))]
    plan["source_spec_record"] = deepcopy(source_record)
    plan["resolved_spec_target"] = deepcopy(resolved_target)
    plan["resolved_evaluation_target"] = deepcopy(resolved_evaluation_target)
    plan["experiment_plan"] = deepcopy(experiment_plan)
    plan["selected_coverage_target"] = deepcopy(selected_target)
    plan["derived_test_design"] = deepcopy(design)
    plan["test_objective"] = {
        "target_origin": "derived_from_selected_spec_ref",
        "experiment_origin": "derived_from_compiler_profile_and_model_structure",
        "failure_hypotheses": [
            item["hypothesis_id"] for item in design["failure_hypotheses"]
        ],
        "isolation_policy": selection_policy,
    }
    plan["variation_contract"] = variation
    plan["guidance_contract"] = _derived_guidance_contract(
        variation, compiler_config
    )
    plan["environment_requirements"] = {
        "derivation_status": "declared_by_sut_adapter_profile",
        "sut": sut_config["sut"],
        "domain": sut_config["domain"],
        "fixture_operations": deepcopy(sut_config["fixture_operations"]),
        "observable_channels": deepcopy(sut_config["observable_channels"]),
        "controllable_channels": deepcopy(sut_config["controllable_channels"]),
        "runtime_observation_catalog": deepcopy(
            sut_config["runtime_observation_catalog"]
        ),
        "scalar_resolutions": deepcopy(sut_config["scalar_resolutions"]),
        "fixture_binding_catalog": deepcopy(
            sut_config["fixture_binding_catalog"]
        ),
    }
    plan["configuration"] = {
        "compiler_profile": profile_identity(compiler_config),
        "sut_adapter_profile": profile_identity(sut_config),
        "run_configuration": {
            "status": "external_to_compiled_plan",
            "affects_compiled_plan_fingerprint": False,
        },
    }
    probe_family = design["probe_contract"]["probe_family"]
    capability_context = {
        "target_archetype": evaluation_binding["target_archetype"],
        "sut": sut_config["sut"],
        "domain": sut_config["domain"],
        "sut_profile_id": sut_config["profile_id"],
        "verified_tool_names": evaluation_binding["evaluation_subject"].get(
            "verified_tool_names"
        )
        or [],
        "probe_family": probe_family,
    }
    capability_stages = (
        "experiment_planner",
        "probe_synthesizer",
        "fixture_binder",
        "runtime_protocol",
        "surface_realizer",
        "outcome_oracle",
        "mutation_provider",
        "guidance_controller",
    )
    plan["capability_contract"] = {
        "registry_schema_version": DEFAULT_PIPELINE_CAPABILITIES.snapshot()[
            "schema_version"
        ],
        "target_archetype": evaluation_binding["target_archetype"],
        "probe_family": probe_family,
        "stage_bindings": {
            stage: (
                capability.capability_id
                if (
                    capability := DEFAULT_PIPELINE_CAPABILITIES.select(
                        stage, capability_context
                    )
                )
                is not None
                else None
            )
            for stage in capability_stages
        },
    }
    installed_fixture_binder = plan["capability_contract"]["stage_bindings"].get(
        "fixture_binder"
    )
    if installed_fixture_binder is not None:
        plan["generation_readiness"]["fixture_binding"] = (
            "registered_fixture_binder_ready"
        )
        plan["compiler_diagnostics"] = [
            diagnostic
            for diagnostic in plan["compiler_diagnostics"]
            if diagnostic.get("code")
            != "bounded_semantic_fixture_design_required"
        ]
        plan["generation_readiness"]["warnings"] = [
            code
            for code in plan["generation_readiness"]["warnings"]
            if code != "bounded_semantic_fixture_design_required"
        ]
    plan.pop("budget", None)
    if design["probe_contract"]["probe_family"] == "unsupported":
        plan["compiler_diagnostics"].append(
            {
                "diagnostic_id": "COMPILER::PROBE::NOT_APPLICABLE",
                "severity": "info",
                "stage": "probe_synthesis",
                "code": design["probe_synthesis"]["reason"],
            }
        )
    plan["compiler_checks"].update(
        {
            "source_anchor_loaded": True,
            "source_record_fingerprint_verified": True,
            "source_to_model_resolution_unique": True,
            "source_to_evaluation_resolution_unique": True,
            "evaluation_archetype_lowered": True,
            "pipeline_capabilities_declared": True,
            "experiment_plan_derived_not_supplied": True,
            "target_cell_derived_not_supplied": True,
            "minimal_contrast_derived_not_supplied": True,
            "boundary_probes_derived_not_supplied": True,
            "probe_contract_derived_not_supplied": True,
            "failure_hypotheses_derived_not_supplied": True,
            "semantic_invariants_derived_not_supplied": True,
            "mutation_catalog_derived_not_supplied": True,
            "environment_capabilities_from_adapter_profile": True,
            "case_specific_request_fields_absent": True,
            "run_budget_excluded_from_compiled_plan": True,
            "source_resolution_artifact_identity_matches": True,
        }
    )
    plan["compiled_plan_fingerprint"] = content_sha256(_fingerprint_payload(plan))
    return plan


def compile_test_plan(
    request: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
    *,
    source_base_dir: str | Path | None = None,
    compiler_profile: Mapping[str, Any] | None = None,
    sut_adapter_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile legacy or profile-driven source-resolved requests."""

    schema_version = request.get("schema_version")
    if schema_version == COMPILER_REQUEST_SCHEMA_VERSION:
        return _compile_test_plan_v1(request, resolved_artifacts)
    if schema_version == COMPILER_REQUEST_SCHEMA_VERSION_V2:
        return _compile_test_plan_v2(
            request,
            resolved_artifacts,
            source_base_dir=source_base_dir,
        )
    if schema_version == COMPILER_REQUEST_SCHEMA_VERSION_V3:
        return _compile_test_plan_v3(
            request,
            resolved_artifacts,
            source_base_dir=source_base_dir,
            compiler_profile=compiler_profile,
            sut_adapter_profile=sut_adapter_profile,
        )
    raise CompilerRequestError(
        f"unsupported compiler request schema: {schema_version!r}"
    )


def compile_request_file(
    *,
    request_path: str | Path,
    artifact_bundle_path: str | Path,
    output_path: str | Path | None = None,
    source_base_dir: str | Path | None = None,
    compiler_profile_path: str | Path | None = None,
    sut_adapter_profile_path: str | Path | None = None,
) -> dict[str, Any]:
    request_file = Path(request_path)
    try:
        request = json.loads(request_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompilerRequestError(f"cannot load compiler request {request_file}: {exc}") from exc
    if not isinstance(request, dict):
        raise CompilerRequestError("compiler request must contain one JSON object")
    resolved = resolve_artifact_bundle_file(artifact_bundle_path)
    compiler_profile = (
        load_configuration_file(compiler_profile_path, kind="compiler_profile")
        if compiler_profile_path is not None
        else None
    )
    sut_adapter_profile = (
        load_configuration_file(
            sut_adapter_profile_path, kind="sut_adapter_profile"
        )
        if sut_adapter_profile_path is not None
        else None
    )
    plan = compile_test_plan(
        request,
        resolved,
        source_base_dir=source_base_dir,
        compiler_profile=compiler_profile,
        sut_adapter_profile=sut_adapter_profile,
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return plan
