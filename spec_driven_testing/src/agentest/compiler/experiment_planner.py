"""Deterministically instantiate experiments from accepted model structure."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .profiles import ConfigurationError
from .target_selector import ISOLATED_ATOMIC_BRANCH_WITNESS


EXPERIMENT_PLAN_SCHEMA_VERSION = "agentspectesting.experiment-plan/v0.1"
DERIVED_TEST_DESIGN_SCHEMA_VERSION = "agentspectesting.derived-test-design/v0.1"
PROBE_CONTRACT_SCHEMA_VERSION = "agentspectesting.probe-contract/v0.1"


class ExperimentPlanningError(ValueError):
    """Raised when enabled general methods cannot instantiate an experiment."""


_DURATION_SECONDS = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentPlanningError(f"{path} must be an object")
    return value


def _focus_condition(target: Mapping[str, Any]) -> Mapping[str, Any]:
    focus = target["focal_configuration"]["focus_predicate_ref"]
    matches = [
        condition
        for condition in target.get("conditions") or []
        if isinstance(condition, Mapping) and condition.get("predicate_id") == focus
    ]
    if len(matches) != 1:
        raise ExperimentPlanningError(
            f"focus predicate {focus!r} must have exactly one compiled condition"
        )
    return matches[0]


def _focus_truth(target: Mapping[str, Any]) -> bool:
    condition = _focus_condition(target)
    truth = condition.get("required_truth_value")
    if not isinstance(truth, bool):
        raise ExperimentPlanningError("focus predicate truth must be boolean")
    return truth


def plan_experiment(
    resolved_target: Mapping[str, Any],
    compiler_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Select an algorithm from model structure, never from a case request."""

    binding = _mapping(resolved_target.get("model_binding"), "$resolved_target.model_binding")
    enabled = set(
        compiler_profile["experiment_policy"]["enabled_experiment_families"]
    )
    if binding.get("required_truth_value") is not True:
        raise ExperimentPlanningError(
            "current isolated atomic planner requires a positive source branch; "
            "no request-side policy override is allowed"
        )
    if not {"nominal_witness", "minimal_contrast"}.issubset(enabled):
        raise ConfigurationError(
            "compiler profile does not enable required witness and contrast experiments"
        )
    plan = {
        "schema_version": EXPERIMENT_PLAN_SCHEMA_VERSION,
        "planner_version": "accepted-structure-experiment-planner/v0.1",
        "source_branch_id": resolved_target.get("source_branch_id"),
        "focus_predicate_id": binding.get("focus_predicate_id"),
        "policy_modality": binding.get("policy_modality"),
        "required_truth_value": binding.get("required_truth_value"),
        "expected_operation_decision": binding.get("expected_operation_decision"),
        "selected_witness_builder": ISOLATED_ATOMIC_BRANCH_WITNESS,
        "selected_contrast_objective": "minimum_policy_factor_hamming_distance",
        "enabled_experiment_families": sorted(enabled),
        "selection_basis": [
            "resolved_source_branch_is_positive",
            "focus_is_accepted_operation_outcome_decision_factor",
            "profile_prefers_isolated_focus",
            "profile_requires_unique_minimum_difference_contrast",
        ],
    }
    plan["experiment_plan_fingerprint"] = content_sha256(plan)
    return plan


def _evaluate(operator: str, left: float, right: float) -> bool:
    operations = {
        "<": lambda: left < right,
        "<=": lambda: left <= right,
        ">": lambda: left > right,
        ">=": lambda: left >= right,
        "==": lambda: left == right,
        "!=": lambda: left != right,
    }
    try:
        return operations[operator]()
    except KeyError as exc:
        raise ExperimentPlanningError(
            f"unsupported comparison operator: {operator!r}"
        ) from exc


def _duration_seconds(value: Mapping[str, Any], path: str) -> float:
    raw = value.get("value")
    unit = value.get("unit")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ExperimentPlanningError(f"{path}.value must be numeric")
    try:
        multiplier = _DURATION_SECONDS[str(unit)]
    except KeyError as exc:
        raise ExperimentPlanningError(f"{path}.unit is unsupported: {unit!r}") from exc
    return float(raw) * multiplier


def _duration_resolution_seconds(sut_profile: Mapping[str, Any]) -> float:
    duration = _mapping(
        sut_profile["scalar_resolutions"]["duration"],
        "$sut_adapter_profile.scalar_resolutions.duration",
    )
    return _duration_seconds(duration, "$sut_adapter_profile.scalar_resolutions.duration")


def _target_name_for_truth(
    truth: bool,
    primary: Mapping[str, Any],
    contrast: Mapping[str, Any],
) -> str:
    matches = [
        name
        for name, target in (("primary", primary), ("contrast", contrast))
        if _focus_truth(target) is truth
    ]
    if len(matches) != 1:
        raise ExperimentPlanningError(
            f"probe truth {truth!r} must map to exactly one primary/contrast target"
        )
    return matches[0]


def synthesize_boundary_refinements(
    primary: Mapping[str, Any],
    contrast: Mapping[str, Any],
    compiler_profile: Mapping[str, Any],
    sut_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Instantiate generic comparison strategies using SUT scalar resolution."""

    condition = _focus_condition(primary)
    expression = condition.get("typed_expression")
    if not isinstance(expression, Mapping) or expression.get("kind") != "comparison":
        return {
            "applicability": "not_applicable",
            "reason": "focus_predicate_is_not_a_comparison",
            "boundary_refinements": [],
        }
    right = expression.get("right")
    if (
        not isinstance(right, Mapping)
        or right.get("kind") != "literal"
        or right.get("value_type") != "duration"
    ):
        return {
            "applicability": "not_applicable",
            "reason": "comparison_threshold_is_not_a_supported_duration_literal",
            "boundary_refinements": [],
        }

    threshold = _duration_seconds(right, "$focus_expression.right")
    epsilon = _duration_resolution_seconds(sut_profile)
    operator = str(expression.get("operator"))
    offsets = {
        "just_below_threshold": -epsilon,
        "at_threshold": 0.0,
        "just_above_threshold": epsilon,
    }
    refinements: list[dict[str, Any]] = []
    for strategy in compiler_profile["probe_policy"]["comparison_strategies"]:
        offset = offsets[strategy]
        truth = _evaluate(operator, threshold + offset, threshold)
        refinements.append(
            {
                "refinement_id": strategy,
                "target_name": _target_name_for_truth(truth, primary, contrast),
                "offset_seconds_from_threshold": offset,
            }
        )
    return {
        "applicability": "applicable",
        "reason": "focus_predicate_is_supported_duration_comparison",
        "operator": operator,
        "threshold_seconds": threshold,
        "epsilon_seconds": epsilon,
        "epsilon_origin": "sut_adapter.smallest_reliably_controllable_unit",
        "boundary_refinements": refinements,
    }


def _probe_target_instance(
    *,
    target_name: str,
    target: Mapping[str, Any],
    control_contract: Mapping[str, Any],
) -> dict[str, Any]:
    focus = str(target["focal_configuration"]["focus_predicate_ref"])
    return {
        "probe_id": f"{target_name}::{focus}",
        "target_name": target_name,
        "coverage_cell_id": target["focal_configuration"]["coverage_cell_id"],
        "focus_predicate_id": focus,
        "required_focus_truth": _focus_truth(target),
        "expected_operation_decision": target["focal_configuration"][
            "expected_operation_decision"
        ],
        "control_contract": deepcopy(dict(control_contract)),
    }


def synthesize_probe_contract(
    primary: Mapping[str, Any],
    contrast: Mapping[str, Any],
    boundary_synthesis: Mapping[str, Any],
) -> dict[str, Any]:
    """Lower one accepted focus expression into a generic probe contract."""

    condition = _focus_condition(primary)
    expression = _mapping(
        condition.get("typed_expression"), "$.focus_condition.typed_expression"
    )
    focus_id = str(condition["predicate_id"])
    instances: list[dict[str, Any]] = []

    if boundary_synthesis.get("applicability") == "applicable":
        family = "duration_boundary"
        for refinement in boundary_synthesis.get("boundary_refinements") or []:
            target_name = str(refinement["target_name"])
            target = primary if target_name == "primary" else contrast
            instances.append(
                {
                    "probe_id": str(refinement["refinement_id"]),
                    "target_name": target_name,
                    "coverage_cell_id": target["focal_configuration"][
                        "coverage_cell_id"
                    ],
                    "focus_predicate_id": focus_id,
                    "required_focus_truth": _focus_truth(target),
                    "expected_operation_decision": target["focal_configuration"][
                        "expected_operation_decision"
                    ],
                    "control_contract": {
                        "kind": "temporal_offset",
                        "offset_seconds_from_threshold": refinement[
                            "offset_seconds_from_threshold"
                        ],
                    },
                }
            )
    elif expression.get("kind") == "semantic_evidence_predicate":
        runtime = _mapping(condition.get("runtime_binding"), "$.focus.runtime_binding")
        operand_ids = runtime.get("condition_operand_ids") or []
        if len(operand_ids) != 1 or not isinstance(operand_ids[0], str):
            raise ExperimentPlanningError(
                "semantic boolean focus requires exactly one bound condition operand"
            )
        family = "semantic_boolean_contrast"
        control = {
            "kind": "semantic_predicate_truth",
            "condition_operand_id": operand_ids[0],
            "observation_value_ids": deepcopy(
                runtime.get("observation_value_ids") or []
            ),
        }
        instances = [
            _probe_target_instance(
                target_name="primary", target=primary, control_contract=control
            ),
            _probe_target_instance(
                target_name="contrast", target=contrast, control_contract=control
            ),
        ]
    elif (
        expression.get("kind") == "comparison"
        and expression.get("operator") in {"==", "!="}
        and isinstance(expression.get("left"), Mapping)
        and expression["left"].get("kind") == "observation_value_ref"
        and isinstance(expression.get("right"), Mapping)
        and expression["right"].get("kind") == "literal"
        and expression["right"].get("value_type") == "enum"
    ):
        family = "enum_equality_contrast"
        control = {
            "kind": "observation_comparison",
            "observation_value_id": expression["left"]["observation_value_id"],
            "operator": expression["operator"],
            "literal": deepcopy(dict(expression["right"])),
        }
        instances = [
            _probe_target_instance(
                target_name="primary", target=primary, control_contract=control
            ),
            _probe_target_instance(
                target_name="contrast", target=contrast, control_contract=control
            ),
        ]
    else:
        family = "unsupported"

    payload = {
        "schema_version": PROBE_CONTRACT_SCHEMA_VERSION,
        "focus_predicate_id": focus_id,
        "probe_family": family,
        "probe_instances": instances,
    }
    payload["probe_contract_fingerprint"] = content_sha256(payload)
    return payload


def validate_probe_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    item = deepcopy(dict(_mapping(value, "$probe_contract")))
    if item.get("schema_version") != PROBE_CONTRACT_SCHEMA_VERSION:
        raise ExperimentPlanningError("unsupported probe contract schema")
    expected = item.pop("probe_contract_fingerprint", None)
    if not isinstance(expected, str) or not expected:
        raise ExperimentPlanningError("probe contract fingerprint is required")
    family = item.get("probe_family")
    if family not in {
        "duration_boundary",
        "semantic_boolean_contrast",
        "enum_equality_contrast",
        "unsupported",
    }:
        raise ExperimentPlanningError(f"unsupported probe family: {family!r}")
    focus = item.get("focus_predicate_id")
    if not isinstance(focus, str) or not focus:
        raise ExperimentPlanningError("probe contract focus predicate is required")
    instances = item.get("probe_instances")
    if not isinstance(instances, list):
        raise ExperimentPlanningError("probe contract instances must be an array")
    seen = set()
    for index, raw in enumerate(instances):
        instance = _mapping(raw, f"$.probe_instances[{index}]")
        probe_id = instance.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id or probe_id in seen:
            raise ExperimentPlanningError("probe instance IDs must be unique strings")
        seen.add(probe_id)
        if instance.get("focus_predicate_id") != focus:
            raise ExperimentPlanningError("probe instance focus predicate drift")
        if not isinstance(instance.get("required_focus_truth"), bool):
            raise ExperimentPlanningError("probe instance truth must be boolean")
        _mapping(instance.get("control_contract"), "$.probe.control_contract")
    actual = content_sha256(item)
    if actual != expected:
        raise ExperimentPlanningError(
            f"probe contract fingerprint mismatch: expected {expected}, computed {actual}"
        )
    item["probe_contract_fingerprint"] = expected
    return item


def _derive_failure_hypotheses(
    resolved_target: Mapping[str, Any],
    primary: Mapping[str, Any],
    contrast: Mapping[str, Any],
    compiler_profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    enabled = set(
        compiler_profile["hypothesis_policy"]["enabled_hypothesis_families"]
    )
    binding = resolved_target["model_binding"]
    focus = _focus_condition(primary)
    expression = focus.get("typed_expression") or {}
    hypotheses: list[dict[str, Any]] = []

    def add(identifier: str, family: str, evidence: list[str]) -> None:
        if family in enabled:
            hypotheses.append(
                {
                    "hypothesis_id": identifier,
                    "family": family,
                    "derivation_evidence": evidence,
                }
            )

    decision = primary["focal_configuration"]["expected_operation_decision"]
    if decision == "permitted":
        add(
            "false_refusal",
            "decision_error",
            ["primary_expected_operation_decision=permitted"],
        )
    elif decision in {"prohibited", "refused"}:
        add(
            "unsafe_allow",
            "decision_error",
            [f"primary_expected_operation_decision={decision}"],
        )
    add(
        "focus_condition_ignored_across_minimal_contrast",
        "decision_error",
        [
            f"focus_predicate={binding['focus_predicate_id']}",
            "primary_and_contrast_differ_only_on_focus",
        ],
    )
    if expression.get("kind") == "comparison":
        operator = expression.get("operator")
        identifier = {
            "<=": "inclusive_boundary_treated_as_exclusive",
            ">=": "inclusive_boundary_treated_as_exclusive",
            "<": "exclusive_boundary_treated_as_inclusive",
            ">": "exclusive_boundary_treated_as_inclusive",
            "==": "equality_boundary_misclassified",
            "!=": "inequality_boundary_misclassified",
        }.get(operator, "comparison_boundary_misclassified")
        add(identifier, "boundary_error", [f"accepted_operator={operator}"])
    if primary["subject"].get("verified_tool_names"):
        add(
            "verbal_compliance_without_verified_focal_execution",
            "execution_error",
            [
                "verified_tools="
                + ",".join(primary["subject"]["verified_tool_names"])
            ],
        )
    if any(
        condition.get("runtime_binding", {}).get("evidence_channels")
        for condition in primary.get("conditions") or []
    ):
        add(
            "decision_without_required_evidence",
            "evidence_error",
            ["one_or_more_target_conditions_require_runtime_evidence"],
        )
    return hypotheses


def _derive_semantic_invariants(
    primary: Mapping[str, Any],
    contrast: Mapping[str, Any],
) -> list[dict[str, Any]]:
    focus = primary["focal_configuration"]["focus_predicate_ref"]
    primary_values = {
        item["predicate_id"]: item["expected_truth_value"]
        for item in primary["focal_configuration"]["required_factor_values"]
    }
    contrast_values = {
        item["predicate_id"]: item["expected_truth_value"]
        for item in contrast["focal_configuration"]["required_factor_values"]
    }
    if set(primary_values) != set(contrast_values):
        raise ExperimentPlanningError("primary and contrast factor sets differ")
    changed = [
        predicate_id
        for predicate_id in primary_values
        if primary_values[predicate_id] is not contrast_values[predicate_id]
    ]
    if changed != [focus]:
        raise ExperimentPlanningError(
            "minimal contrast must change exactly the focus predicate"
        )
    invariants: list[dict[str, Any]] = [
        {
            "invariant_id": "operation_identity_immutable",
            "kind": "operation_identity",
            "operation_policy_family_id": primary["subject"][
                "operation_policy_family_id"
            ],
            "verified_tool_names": deepcopy(primary["subject"]["verified_tool_names"]),
            "mutation_permission": "immutable",
        },
        {
            "invariant_id": "target_object_identity_immutable",
            "kind": "object_identity",
            "binding_policy": "exact_identity",
            "mutation_permission": "immutable",
        },
    ]
    for predicate_id in sorted(primary_values):
        if predicate_id == focus:
            invariants.append(
                {
                    "invariant_id": f"predicate::{predicate_id}",
                    "kind": "predicate_truth",
                    "predicate_id": predicate_id,
                    "allowed_truth_values": sorted(
                        {primary_values[predicate_id], contrast_values[predicate_id]}
                    ),
                    "mutation_permission": "probe_controlled_only",
                }
            )
        else:
            invariants.append(
                {
                    "invariant_id": f"predicate::{predicate_id}",
                    "kind": "predicate_truth",
                    "predicate_id": predicate_id,
                    "required_truth_value": primary_values[predicate_id],
                    "mutation_permission": "immutable",
                }
            )
    return invariants


def _mutation_applicable(
    applicability: str,
    primary: Mapping[str, Any],
    sut_profile: Mapping[str, Any],
) -> tuple[bool, str]:
    focus = _focus_condition(primary)
    expression = focus.get("typed_expression") or {}
    controllable = set(sut_profile.get("controllable_channels") or [])
    prerequisites = primary["generation_contract"].get("required_prerequisites") or []
    checks = {
        "always": (True, "operator_is_unconditionally_available"),
        "temporal_focus": (
            expression.get("kind") == "comparison"
            and (expression.get("right") or {}).get("value_type") == "duration",
            "focus_is_duration_comparison",
        ),
        "user_message_control": (
            "user_message" in controllable,
            "sut_adapter_controls_user_message",
        ),
        "confirmation_milestone": (
            "user_message" in controllable,
            "compiled_interaction_contains_conditional_confirmation_milestone",
        ),
        "user_facts_or_prerequisites": (
            bool(prerequisites)
            or any(
                condition.get("control_role") == "dialogue_controlled"
                for condition in primary.get("conditions") or []
            ),
            "target_requires_prerequisite_or_dialogue_fact_disclosure",
        ),
    }
    try:
        return checks[applicability]
    except KeyError as exc:
        raise ExperimentPlanningError(
            f"unsupported mutation applicability rule: {applicability!r}"
        ) from exc


def _derive_mutation_catalog(
    primary: Mapping[str, Any],
    compiler_profile: Mapping[str, Any],
    sut_profile: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    focus_id = primary["focal_configuration"]["focus_predicate_ref"]
    for operator in compiler_profile["mutation_policy"]["operator_catalog"]:
        applicable, basis = _mutation_applicable(
            operator["applicability"], primary, sut_profile
        )
        record = {
            "name": operator["name"],
            "baseline_value": operator["baseline_value"],
            "allowed_values": deepcopy(operator["allowed_values"]),
            "targets_progress_states": deepcopy(operator["targets_progress_states"]),
            "applicability_basis": basis,
            "binding": {
                "focus_predicate_id": focus_id
                if operator["applicability"] == "temporal_focus"
                else None
            },
        }
        if applicable:
            selected.append(record)
        else:
            record["rejection_reason"] = "applicability_condition_not_satisfied"
            rejected.append(record)
    return selected, rejected


def derive_test_design(
    *,
    resolved_target: Mapping[str, Any],
    primary: Mapping[str, Any],
    contrast: Mapping[str, Any],
    compiler_profile: Mapping[str, Any],
    sut_profile: Mapping[str, Any],
) -> dict[str, Any]:
    probes = synthesize_boundary_refinements(
        primary, contrast, compiler_profile, sut_profile
    )
    probe_contract = synthesize_probe_contract(primary, contrast, probes)
    hypotheses = _derive_failure_hypotheses(
        resolved_target, primary, contrast, compiler_profile
    )
    invariants = _derive_semantic_invariants(primary, contrast)
    mutations, rejected_mutations = _derive_mutation_catalog(
        primary, compiler_profile, sut_profile
    )
    design = {
        "schema_version": DERIVED_TEST_DESIGN_SCHEMA_VERSION,
        "probe_synthesis": probes,
        "probe_contract": probe_contract,
        "failure_hypotheses": hypotheses,
        "semantic_invariants": invariants,
        "mutation_catalog": mutations,
        "rejected_mutation_operators": rejected_mutations,
    }
    design["derived_test_design_fingerprint"] = content_sha256(design)
    return design
