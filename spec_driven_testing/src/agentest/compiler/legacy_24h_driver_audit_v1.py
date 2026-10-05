"""Mechanically compare generic 24h Driver plans with the legacy specialist plan."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .driver_plan_lowering_v1 import validate_generic_driver_plan_set
from .then_atomization import ThenAtomizationError


AUDIT_VERSION = "agentspectesting.legacy-24h-driver-audit/v0.1"
BRANCH_IDS = (
    "airline_093_state#b0",
    "airline_093_state#b1",
    "airline_093_state#b2",
    "airline_093_state#b3",
    "airline_093_state#e0",
    "airline_093_state#e1",
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _new_predicates(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = []
    for item in plan["fixture_binding_plan"]["precondition_bindings"]:
        predicate = (item.get("executable_binding") or {}).get("predicate")
        if isinstance(predicate, Mapping):
            result.append(predicate)
    return result


def _expected_operators(plan: Mapping[str, Any]) -> list[str]:
    return [
        check["runtime_observation_binding"]["expected_observation"]["operator"]
        for check in plan["oracle_plan"]["checks"]
    ]


def audit_legacy_24h_driver_compatibility(
    generic_plan_set: Mapping[str, Any], legacy_plan: Mapping[str, Any]
) -> dict[str, Any]:
    generic = validate_generic_driver_plan_set(generic_plan_set)
    legacy = deepcopy(dict(_mapping(legacy_plan, "$legacy_plan")))
    if legacy.get("schema_version") != "agentspectesting.compiled-test-plan/v0.3":
        raise ThenAtomizationError("legacy comparison requires CompiledTestPlan v0.3")
    plans = {item["source"]["branch_id"]: item for item in generic["plans"]}
    missing = [branch_id for branch_id in BRANCH_IDS if branch_id not in plans]
    if missing:
        raise ThenAtomizationError("generic 24h Driver plans are missing: " + ", ".join(missing))

    b0 = plans["airline_093_state#b0"]
    e0 = plans["airline_093_state#e0"]
    e1 = plans["airline_093_state#e1"]
    b0_created = next(
        item
        for item in _new_predicates(b0)
        if item.get("table") == "reservations" and item.get("path") == "created_at"
    )
    e0_created = next(
        item
        for item in _new_predicates(e0)
        if item.get("table") == "reservations" and item.get("path") == "created_at"
    )
    e1_created = next(
        item
        for item in _new_predicates(e1)
        if item.get("table") == "reservations" and item.get("path") == "created_at"
    )
    focus_id = legacy["target"]["focal_configuration"]["focus_predicate_ref"]
    legacy_focus = next(
        item for item in legacy["target"]["conditions"] if item["predicate_id"] == focus_id
    )
    typed = legacy_focus["typed_expression"]
    threshold = typed["right"]
    legacy_refinements = {
        item["refinement_id"]: item for item in legacy.get("boundary_refinements") or []
    }
    legacy_positive_tools = sorted(legacy["target"]["subject"]["verified_tool_names"])
    new_positive_tools = sorted(b0["reachability_contract"]["target_tool_names"])
    legacy_contrast_decision = legacy["contrast_targets"][0]["focal_configuration"][
        "expected_operation_decision"
    ]
    new_negative_operators = {
        "e0": _expected_operators(e0),
        "e1": _expected_operators(e1),
    }

    checks = {
        "positive_threshold_semantics_equivalent": (
            typed.get("operator") == "<="
            and threshold.get("value") == 24
            and threshold.get("unit") == "hour"
            and b0_created.get("op") == "ge"
            and b0_created.get("value")
            == {"ref": "$NOW", "offset": {"hours": -24}}
        ),
        "positive_target_tool_equivalent": (
            legacy_positive_tools == new_positive_tools == ["cancel_reservation"]
        ),
        "positive_outcome_compatible": (
            legacy["target"]["focal_configuration"]["expected_operation_decision"]
            == "permitted"
            and _expected_operators(b0) == ["present"]
        ),
        "threshold_regions_represented": (
            e0_created.get("op") == "lt"
            and e1_created.get("op") == "lt"
            and e0_created.get("value") == b0_created.get("value")
            and e1_created.get("value") == b0_created.get("value")
            and legacy_refinements.get("just_below_threshold", {}).get(
                "observed_duration_seconds"
            )
            == 86399.0
            and legacy_refinements.get("at_threshold", {}).get(
                "observed_duration_seconds"
            )
            == 86400.0
            and legacy_refinements.get("just_above_threshold", {}).get(
                "observed_duration_seconds"
            )
            == 86401.0
        ),
        "negative_outcome_semantics_equivalent": (
            legacy_contrast_decision == "prohibited"
            and new_negative_operators == {"e0": ["absent"], "e1": ["absent"]}
        ),
        "concrete_fixture_equivalence_testable": False,
    }
    result = {
        "schema_version": AUDIT_VERSION,
        "compatibility_status": "partial_with_known_differences",
        "direct_branch_mapping": {
            "airline_093_state#b0": {
                "legacy_target": "primary",
                "mapping_status": "semantically_compatible_unbound",
            },
            "airline_093_state#e0": {
                "legacy_target": "contrast",
                "mapping_status": "threshold_region_only_not_full_contract",
            },
            "airline_093_state#b1": {"mapping_status": "not_represented_by_legacy_24h_plan"},
            "airline_093_state#b2": {"mapping_status": "not_represented_by_legacy_24h_plan"},
            "airline_093_state#b3": {"mapping_status": "not_represented_by_legacy_24h_plan"},
            "airline_093_state#e1": {"mapping_status": "not_represented_by_legacy_24h_plan"},
        },
        "checks": checks,
        "findings": [
            {
                "code": "legacy_contrast_outcome_is_weaker",
                "legacy_value": legacy_contrast_decision,
                "generic_value": new_negative_operators,
                "meaning": "The legacy coverage contrast says unspecified, while the source-GWT Oracle requires the scoped cancellation call to be absent.",
            },
            {
                "code": "legacy_specialist_plan_has_narrower_branch_coverage",
                "unrepresented_branch_ids": [
                    "airline_093_state#b1",
                    "airline_093_state#b2",
                    "airline_093_state#b3",
                    "airline_093_state#e1",
                ],
                "meaning": "The legacy 24h plan does not represent the other permission alternatives or the insured-but-uncovered negative branch.",
            },
            {
                "code": "fixture_comparison_deferred_until_binding",
                "meaning": "Generic plans intentionally contain no concrete fixture, so concrete state equality can only be checked after Generic Driver Binding.",
            },
        ],
        "source_generic_driver_plan_set_fingerprint": generic[
            "driver_plan_set_fingerprint"
        ],
        "source_legacy_compiled_plan_identity": {
            "compiled_plan_id": legacy.get("compiled_plan_id"),
            "compiled_plan_fingerprint": legacy.get("compiled_plan_fingerprint"),
        },
    }
    result["audit_fingerprint"] = content_sha256(result)
    return validate_legacy_24h_driver_audit(result)


def validate_legacy_24h_driver_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$legacy_24h_driver_audit")))
    supplied = result.pop("audit_fingerprint", None)
    if result.get("schema_version") != AUDIT_VERSION or supplied != content_sha256(result):
        raise ThenAtomizationError("invalid legacy 24h Driver audit")
    checks = _mapping(result.get("checks"), "$.checks")
    expected_keys = {
        "positive_threshold_semantics_equivalent",
        "positive_target_tool_equivalent",
        "positive_outcome_compatible",
        "threshold_regions_represented",
        "negative_outcome_semantics_equivalent",
        "concrete_fixture_equivalence_testable",
    }
    if set(checks) != expected_keys or any(not isinstance(value, bool) for value in checks.values()):
        raise ThenAtomizationError("legacy 24h Driver audit checks are invalid")
    if result.get("compatibility_status") != "partial_with_known_differences":
        raise ThenAtomizationError("legacy 24h Driver compatibility status is invalid")
    result["audit_fingerprint"] = supplied
    return result
