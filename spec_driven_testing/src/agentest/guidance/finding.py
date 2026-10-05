"""Deterministic export of a terminal Guidance search as one finding bundle."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256
from .controller import decide_next_experiment


FINAL_FINDING_SCHEMA_VERSION = "agentspectesting.final-finding-bundle/v0.1"
FINAL_FINDING_GENERATOR_VERSION = "terminal-guidance-finding-exporter/v0.1"


class FinalFindingError(ValueError):
    """Raised when a terminal, lineage-closed finding cannot be exported."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalFindingError(f"{path} must be an object")
    return value


def _bound_identity(bound_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bound_driver_plan_id": bound_plan.get("bound_driver_plan_id"),
        "bound_driver_plan_fingerprint": bound_plan.get(
            "bound_driver_plan_fingerprint"
        ),
    }


def _compiled_identity(compiled_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "compiled_plan_id": compiled_plan.get("compiled_plan_id"),
        "compiled_plan_fingerprint": compiled_plan.get(
            "compiled_plan_fingerprint"
        ),
    }


def _is_quarantined(observation: Mapping[str, Any]) -> bool:
    explicit = observation.get("evidence_status")
    if explicit is not None:
        if explicit not in {"accepted", "quarantined"}:
            raise FinalFindingError("observation has unsupported evidence_status")
        return explicit == "quarantined"
    classification = str(
        (_mapping(observation.get("oracle_result"), "$.observation.oracle_result"))
        .get("classification", "")
    )
    return classification.startswith(("fixture_error:", "harness_error:"))


def _coordinate(oracle: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    fixture_id = oracle.get("fixture_instance_id")
    if not isinstance(fixture_id, str) or not fixture_id:
        raise FinalFindingError("Oracle evidence has no fixture_instance_id")
    selection = _mapping(
        oracle.get("variation_selection") or {}, "$.oracle.variation_selection"
    )
    return fixture_id, tuple(sorted((str(key), str(value)) for key, value in selection.items()))


def _effective_variation(
    bound_plan: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, str]:
    contract = _mapping(bound_plan.get("variation_contract"), "$.variation_contract")
    baseline = dict(
        _mapping(contract.get("baseline_values"), "$.variation_contract.baseline_values")
    )
    result = {str(key): str(value) for key, value in baseline.items()}
    for key, value in selection.items():
        if key not in result:
            raise FinalFindingError(f"unknown variation dimension in evidence: {key}")
        result[str(key)] = str(value)
    return dict(sorted(result.items()))


def _fixture(bound_plan: Mapping[str, Any], fixture_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in bound_plan.get("fixture_instances") or []
        if isinstance(item, Mapping) and item.get("fixture_instance_id") == fixture_id
    ]
    if len(matches) != 1:
        raise FinalFindingError(
            f"finding fixture must resolve exactly once: {fixture_id!r}"
        )
    return deepcopy(dict(matches[0]))


def _evidence_record(observation: Mapping[str, Any]) -> dict[str, Any]:
    oracle = deepcopy(
        dict(_mapping(observation.get("oracle_result"), "$.observation.oracle_result"))
    )
    quarantined = _is_quarantined(observation)
    return {
        "observation_id": observation.get("observation_id"),
        "evidence_status": "quarantined" if quarantined else "accepted",
        "coordinate": {
            "fixture_instance_id": oracle.get("fixture_instance_id"),
            "refinement_id": oracle.get("refinement_id"),
            "coverage_cell_id": oracle.get("coverage_cell_id"),
            "variation_selection": deepcopy(oracle.get("variation_selection") or {}),
        },
        "classification": oracle.get("classification"),
        "correctness_assessed": oracle.get("correctness_assessed"),
        "policy_decision_correct": oracle.get("policy_decision_correct"),
        "strict_execution_passed": oracle.get("strict_execution_passed"),
        "oracle_result_fingerprint": oracle.get("oracle_result_fingerprint"),
        "actual_outcome": deepcopy(oracle.get("actual_outcome")),
        "failure_record": deepcopy(oracle.get("failure_record")),
        "quarantine": deepcopy(observation.get("quarantine")),
        "oracle_result": oracle,
    }


def build_final_finding_bundle(
    compiled_plan: Mapping[str, Any],
    bound_plan: Mapping[str, Any],
    history: Mapping[str, Any],
    terminal_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standalone finding only from a confirmed terminal search."""

    compiled = deepcopy(dict(_mapping(compiled_plan, "$.compiled_plan")))
    bound = deepcopy(dict(_mapping(bound_plan, "$.bound_plan")))
    search = deepcopy(dict(_mapping(history, "$.history")))
    if compiled.get("schema_version") != "agentspectesting.compiled-test-plan/v0.3":
        raise FinalFindingError("Final Finding requires CompiledTestPlan v0.3")
    if bound.get("schema_version") != "agentspectesting.bound-driver-plan/v0.1":
        raise FinalFindingError("Final Finding requires BoundDriverPlan v0.1")
    compiled_identity = _compiled_identity(compiled)
    if bound.get("compiled_plan_identity") != compiled_identity:
        raise FinalFindingError("compiled and bound plan lineage does not close")
    bound_identity = _bound_identity(bound)
    if search.get("bound_driver_plan_identity") != bound_identity:
        raise FinalFindingError("history and bound plan lineage does not close")

    terminal_source = "supplied_checkpoint" if terminal_decision is not None else "derived"
    terminal = (
        deepcopy(dict(_mapping(terminal_decision, "$.terminal_decision")))
        if terminal_decision is not None
        else decide_next_experiment(bound, search)
    )
    if terminal.get("search_id") != search.get("search_id"):
        raise FinalFindingError("terminal decision belongs to a different search")
    if terminal.get("bound_driver_plan_identity") != bound_identity:
        raise FinalFindingError("terminal decision belongs to a different bound plan")
    terminal_fingerprint = terminal.get("guidance_decision_fingerprint")
    terminal_payload = deepcopy(terminal)
    terminal_payload.pop("guidance_decision_fingerprint", None)
    if terminal_fingerprint != content_sha256(terminal_payload):
        raise FinalFindingError("terminal Guidance decision fingerprint is invalid")
    if not (
        terminal.get("status") == "stop"
        and terminal.get("phase") == "confirmed"
        and terminal.get("reason_code")
        == "reproducible_minimal_valid_failure_found"
    ):
        raise FinalFindingError(
            "finding export requires reproducible_minimal_valid_failure_found"
        )

    raw_observations = search.get("observations")
    if not isinstance(raw_observations, list) or not raw_observations:
        raise FinalFindingError("terminal history has no observations")
    evidence_records = [
        _evidence_record(_mapping(item, "$.history.observations[]"))
        for item in raw_observations
    ]
    accepted = [item for item in evidence_records if item["evidence_status"] == "accepted"]
    violations = [
        item
        for item in accepted
        if str(item["classification"]).startswith("policy_violation:")
    ]
    if not violations:
        raise FinalFindingError("terminal finding has no accepted policy violation")

    groups: dict[
        tuple[str, tuple[tuple[str, str], ...]], list[dict[str, Any]]
    ] = defaultdict(list)
    first_violation_index: dict[
        tuple[str, tuple[tuple[str, str], ...]], int
    ] = {}
    for index, item in enumerate(accepted):
        coordinate = _coordinate(item["oracle_result"])
        groups[coordinate].append(item)
        if str(item["classification"]).startswith("policy_violation:"):
            first_violation_index.setdefault(coordinate, index)
    trigger_coordinate = sorted(
        first_violation_index,
        key=lambda value: (
            len(value[1]),
            first_violation_index[value],
            value,
        ),
    )[0]
    trigger_attempts = groups[trigger_coordinate]
    trigger_violations = [
        item
        for item in trigger_attempts
        if str(item["classification"]).startswith("policy_violation:")
    ]
    required_confirmations = int(
        _mapping(search.get("configuration"), "$.history.configuration").get(
            "required_confirmations"
        )
    )
    if len(trigger_violations) < required_confirmations:
        raise FinalFindingError("accepted trigger evidence is below confirmation threshold")

    fixture_id, selection_items = trigger_coordinate
    trigger_selection = dict(selection_items)
    trigger_fixture = _fixture(bound, fixture_id)
    trigger_refinement = trigger_fixture.get("refinement_id")
    boundary_refinement = next(
        (
            deepcopy(dict(item))
            for item in bound.get("search_design", {}).get(
                "boundary_refinements", []
            )
            if isinstance(item, Mapping)
            and item.get("refinement_id") == trigger_refinement
        ),
        None,
    )
    if boundary_refinement is None:
        raise FinalFindingError("trigger fixture has no boundary refinement lineage")

    passing_surface_contrasts = [
        item
        for item in accepted
        if item["classification"] == "pass:permitted_action_committed"
        and item["coordinate"]["fixture_instance_id"] == fixture_id
        and dict(item["coordinate"]["variation_selection"]) != trigger_selection
    ]
    same_surface_boundary_contrasts = [
        item
        for item in accepted
        if item["coordinate"]["fixture_instance_id"] != fixture_id
        and dict(item["coordinate"]["variation_selection"]) == trigger_selection
    ]
    hypotheses = [
        deepcopy(dict(item))
        for item in bound.get("search_design", {}).get("failure_hypotheses", [])
        if isinstance(item, Mapping)
    ]
    supported_hypothesis_ids: list[str] = []
    classification = str(trigger_violations[0]["classification"])
    for hypothesis in hypotheses:
        identifier = str(hypothesis.get("hypothesis_id"))
        if identifier == classification.split(":", 1)[-1]:
            supported_hypothesis_ids.append(identifier)
        if (
            hypothesis.get("family") == "boundary_error"
            and trigger_refinement == "at_threshold"
            and boundary_refinement.get("operator") in {"<=", ">="}
        ):
            supported_hypothesis_ids.append(identifier)
    supported_hypothesis_ids = list(dict.fromkeys(supported_hypothesis_ids))

    source_record = deepcopy(
        dict(_mapping(compiled.get("source_spec_record"), "$.source_spec_record"))
    )
    resolved_target = deepcopy(
        dict(
            _mapping(
                compiled.get("resolved_spec_target"), "$.resolved_spec_target"
            )
        )
    )
    target = deepcopy(dict(_mapping(compiled.get("target"), "$.target")))
    focus_predicate_id = target.get("focal_configuration", {}).get(
        "focus_predicate_ref"
    )
    focus_condition = next(
        (
            deepcopy(dict(item))
            for item in target.get("conditions", [])
            if isinstance(item, Mapping)
            and item.get("predicate_id") == focus_predicate_id
        ),
        None,
    )
    if focus_condition is None:
        raise FinalFindingError("compiled target has no focus condition")

    counts = {
        "observed_experiments": len(evidence_records),
        "accepted_evidence_experiments": len(accepted),
        "quarantined_experiments": len(evidence_records) - len(accepted),
        "accepted_policy_violations": sum(
            str(item["classification"]).startswith("policy_violation:")
            for item in accepted
        ),
        "accepted_passes": sum(
            str(item["classification"]).startswith("pass:") for item in accepted
        ),
        "accepted_reachability_failures": sum(
            str(item["classification"]).startswith("reachability_failure:")
            for item in accepted
        ),
        "accepted_coverage_only": sum(
            str(item["classification"]).startswith("coverage_only:")
            for item in accepted
        ),
    }
    payload = {
        "schema_version": FINAL_FINDING_SCHEMA_VERSION,
        "generator_version": FINAL_FINDING_GENERATOR_VERSION,
        "finding_status": "confirmed",
        "finding_type": "policy_violation",
        "lineage": {
            "source_spec_record": source_record,
            "resolved_spec_target": resolved_target,
            "compiled_plan_identity": compiled_identity,
            "bound_driver_plan_identity": bound_identity,
            "search_id": search.get("search_id"),
            "terminal_guidance_decision_fingerprint": terminal.get(
                "guidance_decision_fingerprint"
            ),
            "terminal_guidance_decision_source": terminal_source,
        },
        "focal_target": {
            "subject": deepcopy(target.get("subject")),
            "focal_configuration": deepcopy(target.get("focal_configuration")),
            "focus_condition": focus_condition,
            "boundary_refinement": boundary_refinement,
        },
        "failure_signature": {
            "classification": classification,
            "analysis_axis": trigger_violations[0]["oracle_result"].get(
                "analysis_axis"
            ),
            "supported_hypothesis_ids": supported_hypothesis_ids,
            "expected_operation_decision": trigger_fixture.get(
                "expected_operation_decision"
            ),
            "actual_decision": (
                trigger_violations[0]["actual_outcome"] or {}
            ).get("decision"),
            "representation_sensitive": bool(passing_surface_contrasts),
        },
        "minimal_trigger": {
            "fixture_instance_id": fixture_id,
            "refinement_id": trigger_refinement,
            "coverage_cell_id": trigger_fixture.get("coverage_cell_id"),
            "object_bindings": deepcopy(trigger_fixture.get("object_bindings")),
            "observation_bindings": deepcopy(
                trigger_fixture.get("observation_bindings")
            ),
            "variation_selection": trigger_selection,
            "effective_variation_profile": _effective_variation(
                bound, trigger_selection
            ),
            "minimality": terminal.get("evidence_basis", {}).get("minimality"),
        },
        "confirmation": {
            "required_confirmations": required_confirmations,
            "accepted_attempt_count": len(trigger_attempts),
            "accepted_violation_count": len(trigger_violations),
            "trigger_observation_ids": [
                item["observation_id"] for item in trigger_violations
            ],
            "confirmation_rule": "count_based",
        },
        "passing_surface_contrasts": [
            {
                "observation_id": item["observation_id"],
                "classification": item["classification"],
                "coordinate": deepcopy(item["coordinate"]),
                "effective_variation_profile": _effective_variation(
                    bound, item["coordinate"]["variation_selection"]
                ),
                "actual_outcome": deepcopy(item["actual_outcome"]),
                "oracle_result_fingerprint": item["oracle_result_fingerprint"],
            }
            for item in passing_surface_contrasts
        ],
        "same_surface_boundary_contrasts": [
            {
                "observation_id": item["observation_id"],
                "classification": item["classification"],
                "coordinate": deepcopy(item["coordinate"]),
                "actual_outcome": deepcopy(item["actual_outcome"]),
                "oracle_result_fingerprint": item["oracle_result_fingerprint"],
            }
            for item in same_surface_boundary_contrasts
        ],
        "search_summary": {
            "terminal_status": terminal.get("status"),
            "terminal_phase": terminal.get("phase"),
            "terminal_reason_code": terminal.get("reason_code"),
            "budget": deepcopy(terminal.get("budget")),
            "counts": counts,
        },
        "evidence_records": evidence_records,
        "method_limits": [
            "count_based_confirmation_not_statistical_confidence",
            "minimality_limited_to_baseline_or_one_compiled_dimension",
            "no_population_failure_rate_estimate",
        ],
        "llm_calls": 0,
    }
    payload["finding_id"] = "FIND::" + content_sha256(payload)[:16]
    payload["finding_fingerprint"] = content_sha256(payload)
    return payload


def build_final_finding_bundle_files(
    *,
    compiled_plan_path: str | Path,
    bound_plan_path: str | Path,
    history_path: str | Path,
    output_path: str | Path,
    terminal_decision_path: str | Path | None = None,
) -> dict[str, Any]:
    def load(path: str | Path, label: str) -> dict[str, Any]:
        file = Path(path)
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FinalFindingError(f"cannot load {label} {file}: {exc}") from exc
        return deepcopy(dict(_mapping(value, f"$.{label}")))

    finding = build_final_finding_bundle(
        load(compiled_plan_path, "compiled_plan"),
        load(bound_plan_path, "bound_plan"),
        load(history_path, "history"),
        (
            load(terminal_decision_path, "terminal_decision")
            if terminal_decision_path is not None
            else None
        ),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(finding, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return finding


__all__ = [
    "FINAL_FINDING_GENERATOR_VERSION",
    "FINAL_FINDING_SCHEMA_VERSION",
    "FinalFindingError",
    "build_final_finding_bundle",
    "build_final_finding_bundle_files",
]
