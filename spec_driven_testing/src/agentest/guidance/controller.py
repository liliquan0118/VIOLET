"""Deterministic Guidance v0.1.

Guidance is a feedback-to-experiment controller.  It never writes user text and
never calls a model.  It chooses one bound fixture and at most one compiled
variation dimension for the runtime Driver to realize.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256


GUIDANCE_CONTROLLER_VERSION = "deterministic-guidance-controller/v0.1"
GUIDANCE_HISTORY_SCHEMA_VERSION = "agentspectesting.guidance-history/v0.1"
GUIDANCE_DECISION_SCHEMA_VERSION = "agentspectesting.guidance-decision/v0.1"
DEFAULT_MAX_INVALID_ATTEMPTS_PER_COORDINATE = 2


class GuidanceError(ValueError):
    """Raised when feedback cannot safely determine a next experiment."""


_LEGACY_BASELINE_VALUES = {
    "fact_disclosure_order": "required_identity_first",
    "temporal_surface_form": "exact_observation_values",
    "turn_decomposition": "one_required_fact_per_turn",
    "confirmation_style": "direct",
    "surface_paraphrase": "canonical",
}

# Old compiled plans used aggregate names.  Expanding them here makes Guidance
# backward compatible while newly compiled plans use Runtime's canonical names.
_PROGRESS_STATE_ALIASES = {
    "target_configuration_satisfied": {"fixture_valid"},
    "required_facts_available": {
        "required_user_facts_available",
        "required_agent_evidence_available",
    },
    "confirmation_completed": {"required_confirmation_completed"},
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GuidanceError(f"{path} must be an object")
    return value


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise GuidanceError(f"{path} must be a positive integer")
    return value


def _bound_identity(bound_plan: Mapping[str, Any]) -> dict[str, Any]:
    if bound_plan.get("schema_version") != "agentspectesting.bound-driver-plan/v0.1":
        raise GuidanceError("Guidance requires BoundDriverPlan v0.1")
    identifier = bound_plan.get("bound_driver_plan_id")
    fingerprint = bound_plan.get("bound_driver_plan_fingerprint")
    if not isinstance(identifier, str) or not identifier:
        raise GuidanceError("bound plan has no identity")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise GuidanceError("bound plan has no fingerprint")
    return {
        "bound_driver_plan_id": identifier,
        "bound_driver_plan_fingerprint": fingerprint,
    }


def _catalog(bound_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = _mapping(bound_plan.get("variation_contract"), "$.variation_contract")
    raw = contract.get("mutable_dimensions")
    if not isinstance(raw, list) or not raw:
        raise GuidanceError("variation contract has no mutable dimensions")
    explicit = contract.get("baseline_values")
    baseline_values = dict(explicit) if isinstance(explicit, Mapping) else {}
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        value = _mapping(item, f"$.variation_contract.mutable_dimensions[{index}]")
        name = value.get("name")
        allowed = value.get("allowed_values")
        if not isinstance(name, str) or not name or name in names:
            raise GuidanceError("variation dimension names must be unique strings")
        if not isinstance(allowed, list) or not allowed:
            raise GuidanceError(f"variation dimension {name!r} has no allowed values")
        baseline = value.get("baseline_value") or baseline_values.get(name)
        if baseline is None:
            baseline = _LEGACY_BASELINE_VALUES.get(name)
        if baseline not in allowed:
            raise GuidanceError(
                f"variation dimension {name!r} has no valid compiled baseline"
            )
        states: set[str] = set()
        for state in value.get("targets_progress_states") or []:
            states.update(_PROGRESS_STATE_ALIASES.get(str(state), {str(state)}))
        result.append(
            {
                **deepcopy(dict(value)),
                "name": name,
                "allowed_values": list(allowed),
                "baseline_value": baseline,
                "normalized_target_progress_states": sorted(states),
            }
        )
        names.add(name)
    return result


def _fixture_index(bound_plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fixtures = bound_plan.get("fixture_instances")
    if not isinstance(fixtures, list) or not fixtures:
        raise GuidanceError("bound plan has no fixture instances")
    result: dict[str, dict[str, Any]] = {}
    for item in fixtures:
        fixture = dict(_mapping(item, "$.fixture_instances[]"))
        identifier = fixture.get("fixture_instance_id")
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise GuidanceError("fixture instance IDs must be unique strings")
        result[identifier] = fixture
    return result


def new_guidance_history(
    bound_plan: Mapping[str, Any],
    *,
    max_experiments: int,
    required_confirmations: int = 2,
    max_attempts_per_candidate: int = 3,
    max_invalid_attempts_per_coordinate: int = (
        DEFAULT_MAX_INVALID_ATTEMPTS_PER_COORDINATE
    ),
) -> dict[str, Any]:
    """Create the only mutable search input: an append-only observation history."""

    identity = _bound_identity(bound_plan)
    maximum = _positive_int(max_experiments, "$.max_experiments")
    confirmations = _positive_int(required_confirmations, "$.required_confirmations")
    attempts = _positive_int(
        max_attempts_per_candidate, "$.max_attempts_per_candidate"
    )
    invalid_attempts = _positive_int(
        max_invalid_attempts_per_coordinate,
        "$.max_invalid_attempts_per_coordinate",
    )
    if attempts < confirmations:
        raise GuidanceError("max attempts must be at least required confirmations")
    return {
        "schema_version": GUIDANCE_HISTORY_SCHEMA_VERSION,
        "guidance_controller_version": GUIDANCE_CONTROLLER_VERSION,
        "search_id": "guidance::" + content_sha256(
            {"bound": identity, "max_experiments": maximum}
        )[:16],
        "bound_driver_plan_identity": identity,
        "configuration": {
            "max_experiments": maximum,
            "required_confirmations": confirmations,
            "max_attempts_per_candidate": attempts,
            "max_invalid_attempts_per_coordinate": invalid_attempts,
        },
        "observations": [],
        "llm_calls": 0,
    }


def _validate_history(
    bound_plan: Mapping[str, Any], history: Mapping[str, Any]
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    value = deepcopy(dict(_mapping(history, "$.history")))
    if value.get("schema_version") != GUIDANCE_HISTORY_SCHEMA_VERSION:
        raise GuidanceError("unsupported Guidance history schema")
    if value.get("bound_driver_plan_identity") != _bound_identity(bound_plan):
        raise GuidanceError("Guidance history belongs to a different bound plan")
    configuration = _mapping(value.get("configuration"), "$.history.configuration")
    _positive_int(configuration.get("max_experiments"), "$.configuration.max_experiments")
    required = _positive_int(
        configuration.get("required_confirmations"),
        "$.configuration.required_confirmations",
    )
    attempts = _positive_int(
        configuration.get("max_attempts_per_candidate"),
        "$.configuration.max_attempts_per_candidate",
    )
    if attempts < required:
        raise GuidanceError("max attempts must be at least required confirmations")
    invalid_attempts = configuration.get(
        "max_invalid_attempts_per_coordinate",
        DEFAULT_MAX_INVALID_ATTEMPTS_PER_COORDINATE,
    )
    configuration = dict(configuration)
    configuration["max_invalid_attempts_per_coordinate"] = _positive_int(
        invalid_attempts,
        "$.configuration.max_invalid_attempts_per_coordinate",
    )
    value["configuration"] = configuration
    observations = value.get("observations")
    if not isinstance(observations, list):
        raise GuidanceError("$.history.observations must be an array")
    return value, observations


def record_guidance_observation(
    bound_plan: Mapping[str, Any],
    history: Mapping[str, Any],
    oracle_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one Bound Oracle result after validating its experiment identity."""

    value, observations = _validate_history(bound_plan, history)
    result = deepcopy(dict(_mapping(oracle_result, "$.oracle_result")))
    if result.get("bound_driver_plan_identity") != _bound_identity(bound_plan):
        raise GuidanceError("Oracle result belongs to a different bound plan")
    fixture_id = result.get("fixture_instance_id")
    if fixture_id not in _fixture_index(bound_plan):
        raise GuidanceError("Oracle result names an unknown fixture")
    if not isinstance(result.get("classification"), str):
        raise GuidanceError("Oracle result has no classification")
    _normalized_selection(bound_plan, result.get("variation_selection") or {})
    classification = str(result["classification"])
    quarantined = classification.startswith(("fixture_error:", "harness_error:"))
    observation = {
        "observation_id": f"OBS{len(observations) + 1:04d}",
        "evidence_status": "quarantined" if quarantined else "accepted",
        "quarantine": (
            {
                "reason_code": classification,
                "retryable": True,
            }
            if quarantined
            else None
        ),
        "oracle_result": result,
    }
    value["observations"].append(observation)
    value["history_fingerprint"] = content_sha256(value)
    return value


def _baseline_values(bound_plan: Mapping[str, Any]) -> dict[str, str]:
    return {item["name"]: str(item["baseline_value"]) for item in _catalog(bound_plan)}


def _normalized_selection(
    bound_plan: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, str]:
    catalog = {item["name"]: item for item in _catalog(bound_plan)}
    baseline = _baseline_values(bound_plan)
    requested = dict(_mapping(selection, "$.variation_selection"))
    unknown = sorted(set(requested) - set(catalog))
    if unknown:
        raise GuidanceError("unknown variation dimensions: " + ", ".join(unknown))
    normalized: dict[str, str] = {}
    for dimension, value in requested.items():
        if value not in catalog[dimension]["allowed_values"]:
            raise GuidanceError(
                f"variation {dimension!r} does not allow value {value!r}"
            )
        if value != baseline[dimension]:
            normalized[dimension] = str(value)
    maximum = bound_plan["variation_contract"].get(
        "max_changed_dimensions_per_candidate", 1
    )
    if len(normalized) > maximum:
        raise GuidanceError("feedback contains too many changed variation dimensions")
    return dict(sorted(normalized.items()))


def _results(observations: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        item = _mapping(observation, f"$.observations[{index}]")
        oracle = deepcopy(
            dict(_mapping(item.get("oracle_result"), f"$.observations[{index}].oracle_result"))
        )
        oracle["_observation_id"] = item.get("observation_id")
        oracle["_observation_index"] = index
        classification = str(oracle.get("classification", ""))
        default_status = (
            "quarantined"
            if classification.startswith(("fixture_error:", "harness_error:"))
            else "accepted"
        )
        evidence_status = item.get("evidence_status", default_status)
        if evidence_status not in {"accepted", "quarantined"}:
            raise GuidanceError(
                f"$.observations[{index}].evidence_status is unsupported"
            )
        oracle["_evidence_status"] = evidence_status
        result.append(oracle)
    return result


def _coordinate(
    bound_plan: Mapping[str, Any], result: Mapping[str, Any]
) -> tuple[str, tuple[tuple[str, str], ...]]:
    fixture_id = result.get("fixture_instance_id")
    if fixture_id not in _fixture_index(bound_plan):
        raise GuidanceError("history contains an unknown fixture")
    selection = _normalized_selection(
        bound_plan, result.get("variation_selection") or {}
    )
    return str(fixture_id), tuple(sorted(selection.items()))


def _selection_from_coordinate(
    coordinate: tuple[str, tuple[tuple[str, str], ...]]
) -> dict[str, str]:
    return dict(coordinate[1])


def _fixture_priority(fixture: Mapping[str, Any], index: int) -> tuple[int, int]:
    refinement = fixture.get("refinement_id")
    priorities = {
        "at_threshold": 0,
        "just_below_threshold": 1,
        "just_above_threshold": 2,
    }
    return priorities.get(str(refinement), 3), index


def _ordered_fixture_ids(bound_plan: Mapping[str, Any], *, baseline: bool) -> list[str]:
    fixtures = list(_fixture_index(bound_plan).values())
    if baseline:
        # Baseline follows compiled probe order; exploration starts at the exact boundary.
        return [str(item["fixture_instance_id"]) for item in fixtures]
    ordered = sorted(
        enumerate(fixtures), key=lambda pair: _fixture_priority(pair[1], pair[0])
    )
    return [str(item[1]["fixture_instance_id"]) for item in ordered]


def _hypotheses(bound_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    search = bound_plan.get("search_design") or {}
    raw = search.get("failure_hypotheses") if isinstance(search, Mapping) else []
    return [deepcopy(dict(item)) for item in raw or [] if isinstance(item, Mapping)]


def _hypotheses_for_dimension(
    bound_plan: Mapping[str, Any], dimension: Mapping[str, Any]
) -> list[str]:
    focus_bound = bool((dimension.get("binding") or {}).get("focus_predicate_id"))
    states = set(dimension.get("normalized_target_progress_states") or [])
    selected: list[str] = []
    for hypothesis in _hypotheses(bound_plan):
        family = hypothesis.get("family")
        if focus_bound and family in {"boundary_error", "decision_error"}:
            selected.append(str(hypothesis.get("hypothesis_id")))
        elif family == "evidence_error" and states & {
            "required_user_facts_available",
            "required_agent_evidence_available",
        }:
            selected.append(str(hypothesis.get("hypothesis_id")))
        elif family == "execution_error" and "decision_opportunity_reached" in states:
            selected.append(str(hypothesis.get("hypothesis_id")))
    return list(dict.fromkeys(selected))


def _decision(
    bound_plan: Mapping[str, Any],
    history: Mapping[str, Any],
    *,
    status: str,
    phase: str,
    reason_code: str,
    experiment: Mapping[str, Any] | None = None,
    evidence_basis: Mapping[str, Any] | None = None,
    retained_dimensions: list[str] | None = None,
    target_hypotheses: list[str] | None = None,
    successful_driver_path: bool = False,
) -> dict[str, Any]:
    catalog = _catalog(bound_plan)
    retained = list(retained_dimensions or [])
    all_dimensions = [item["name"] for item in catalog]
    observations = history.get("observations") or []
    quarantined_count = sum(
        1
        for item in observations
        if (
            item.get("evidence_status") == "quarantined"
            or (
                item.get("evidence_status") is None
                and str((item.get("oracle_result") or {}).get("classification", ""))
                .startswith(("fixture_error:", "harness_error:"))
            )
        )
    )
    configuration = history["configuration"]
    result = {
        "schema_version": GUIDANCE_DECISION_SCHEMA_VERSION,
        "guidance_controller_version": GUIDANCE_CONTROLLER_VERSION,
        "decision_id": f"GD{len(observations) + 1:04d}",
        "search_id": history["search_id"],
        "bound_driver_plan_identity": _bound_identity(bound_plan),
        "status": status,
        "phase": phase,
        "reason_code": reason_code,
        "experiment": deepcopy(dict(experiment)) if experiment else None,
        "preserve": {
            "successful_driver_path": successful_driver_path,
            "semantic_invariant_ids": [
                item.get("invariant_id")
                for item in bound_plan["variation_contract"].get(
                    "semantic_invariants", []
                )
            ],
            "max_changed_dimensions": bound_plan["variation_contract"].get(
                "max_changed_dimensions_per_candidate", 1
            ),
        },
        "evidence_basis": deepcopy(dict(evidence_basis or {})),
        "search_space_update": {
            "retained_dimensions": retained,
            "temporarily_pruned_dimensions": [
                item for item in all_dimensions if item not in retained
            ],
            "target_hypotheses": list(target_hypotheses or []),
        },
        "budget": {
            "max_experiments": configuration["max_experiments"],
            "observed_experiments": len(observations),
            "accepted_evidence_experiments": len(observations) - quarantined_count,
            "quarantined_experiments": quarantined_count,
            "remaining_before_decision": max(
                0, configuration["max_experiments"] - len(observations)
            ),
            "proposed_experiments": 1 if status == "continue" else 0,
        },
        "generator_contract": {
            "producer": "deterministic_guidance_controller",
            "surface_realizer": "runtime_driver",
            "llm_calls": 0,
        },
    }
    result["guidance_decision_fingerprint"] = content_sha256(result)
    return result


def _experiment(
    fixtures: Mapping[str, Mapping[str, Any]],
    fixture_id: str,
    selection: Mapping[str, str],
    *,
    purpose: str,
) -> dict[str, Any]:
    fixture = fixtures[fixture_id]
    return {
        "fixture_instance_id": fixture_id,
        "refinement_id": fixture.get("refinement_id"),
        "coverage_cell_id": fixture.get("coverage_cell_id"),
        "variation_selection": deepcopy(dict(selection)),
        "purpose": purpose,
    }


def _policy_anchor(
    bound_plan: Mapping[str, Any], results: list[Mapping[str, Any]]
) -> tuple[
    tuple[str, tuple[tuple[str, str], ...]], list[Mapping[str, Any]]
] | None:
    groups: dict[
        tuple[str, tuple[tuple[str, str], ...]], list[Mapping[str, Any]]
    ] = defaultdict(list)
    first_violation: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
    for result in results:
        coordinate = _coordinate(bound_plan, result)
        groups[coordinate].append(result)
        if str(result.get("classification", "")).startswith("policy_violation:"):
            first_violation.setdefault(coordinate, int(result["_observation_index"]))
    if not first_violation:
        return None
    ordered = sorted(
        first_violation,
        key=lambda item: (len(item[1]), first_violation[item], item),
    )
    return ordered[0], groups[ordered[0]]


def _latest_per_fixture(
    results: list[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for result in results:
        latest[str(result.get("fixture_instance_id"))] = result
    return latest


def decide_next_experiment(
    bound_plan: Mapping[str, Any], history: Mapping[str, Any]
) -> dict[str, Any]:
    """Choose one next experiment or a typed stop/quarantine decision."""

    value, raw_observations = _validate_history(bound_plan, history)
    fixtures = _fixture_index(bound_plan)
    catalog = _catalog(bound_plan)
    results = _results(raw_observations)
    evidence_results = [
        result for result in results if result["_evidence_status"] == "accepted"
    ]
    coordinates = {
        _coordinate(bound_plan, result) for result in evidence_results
    }

    if results:
        latest = results[-1]
        classification = str(latest.get("classification", ""))
        if latest["_evidence_status"] == "quarantined":
            coordinate = _coordinate(bound_plan, latest)
            invalid_attempts = [
                item
                for item in results
                if item["_evidence_status"] == "quarantined"
                and _coordinate(bound_plan, item) == coordinate
            ]
            limit = value["configuration"][
                "max_invalid_attempts_per_coordinate"
            ]
            evidence = {
                "quarantined_observation_ids": [
                    item.get("_observation_id") for item in invalid_attempts
                ],
                "latest_reason_code": classification,
                "invalid_attempt_count": len(invalid_attempts),
                "max_invalid_attempts_per_coordinate": limit,
            }
            if len(invalid_attempts) < limit:
                selection = _selection_from_coordinate(coordinate)
                return _decision(
                    bound_plan,
                    value,
                    status="continue",
                    phase="invalid_run_retry",
                    reason_code="retry_quarantined_coordinate",
                    experiment=_experiment(
                        fixtures,
                        coordinate[0],
                        selection,
                        purpose="retry_quarantined_run",
                    ),
                    evidence_basis=evidence,
                    retained_dimensions=list(selection),
                )
            return _decision(
                bound_plan,
                value,
                status="quarantine",
                phase="invalid_run",
                reason_code="invalid_retry_limit_exhausted",
                evidence_basis=evidence,
            )
        if classification.startswith("execution_failure:"):
            return _decision(
                bound_plan,
                value,
                status="stop",
                phase="runtime_investigation",
                reason_code=classification,
                evidence_basis={"observation_id": latest.get("_observation_id")},
            )

    maximum = value["configuration"]["max_experiments"]
    if len(results) >= maximum:
        return _decision(
            bound_plan,
            value,
            status="stop",
            phase="exhausted",
            reason_code="run_budget_exhausted",
        )

    # A canonical run for every compiled probe is required before feedback can
    # justify any surface mutation.
    for fixture_id in _ordered_fixture_ids(bound_plan, baseline=True):
        coordinate = (fixture_id, ())
        if coordinate not in coordinates:
            return _decision(
                bound_plan,
                value,
                status="continue",
                phase="baseline",
                reason_code="required_baseline_missing",
                experiment=_experiment(
                    fixtures, fixture_id, {}, purpose="establish_canonical_baseline"
                ),
            )

    anchor = _policy_anchor(bound_plan, evidence_results)
    if anchor is not None:
        coordinate, attempts = anchor
        violations = [
            item
            for item in attempts
            if str(item.get("classification", "")).startswith("policy_violation:")
        ]
        required = value["configuration"]["required_confirmations"]
        maximum_attempts = value["configuration"]["max_attempts_per_candidate"]
        selection = _selection_from_coordinate(coordinate)
        fixture_id = coordinate[0]
        evidence = {
            "trigger_observation_ids": [
                item.get("_observation_id") for item in violations
            ],
            "classification": violations[0].get("classification"),
            "attempt_count": len(attempts),
            "violation_count": len(violations),
        }
        if len(violations) < required and len(attempts) < maximum_attempts:
            return _decision(
                bound_plan,
                value,
                status="continue",
                phase="confirmation",
                reason_code="replay_policy_violation_candidate",
                experiment=_experiment(
                    fixtures, fixture_id, selection, purpose="exact_failure_replay"
                ),
                evidence_basis=evidence,
                retained_dimensions=list(selection),
                successful_driver_path=True,
            )
        if len(violations) >= required:
            # First hold the prompt transformation fixed across probe cells.
            for contrast_fixture_id in _ordered_fixture_ids(
                bound_plan, baseline=False
            ):
                contrast_coordinate = (
                    contrast_fixture_id,
                    tuple(sorted(selection.items())),
                )
                if contrast_coordinate not in coordinates:
                    return _decision(
                        bound_plan,
                        value,
                        status="continue",
                        phase="contrast",
                        reason_code="required_boundary_contrast_missing",
                        experiment=_experiment(
                            fixtures,
                            contrast_fixture_id,
                            selection,
                            purpose="same_surface_boundary_contrast",
                        ),
                        evidence_basis=evidence,
                        retained_dimensions=list(selection),
                        successful_driver_path=True,
                    )

            # Then vary values inside the already implicated dimension.  A
            # baseline failure uses only focus-bound dimensions, preventing an
            # unconstrained sweep of unrelated dialogue changes.
            implicated = list(selection)
            dimensions = (
                [item for item in catalog if item["name"] in implicated]
                if implicated
                else [
                    item
                    for item in catalog
                    if (item.get("binding") or {}).get("focus_predicate_id")
                ]
            )
            for dimension in dimensions:
                name = dimension["name"]
                for allowed in dimension["allowed_values"]:
                    candidate = (
                        {}
                        if allowed == dimension["baseline_value"]
                        else {name: str(allowed)}
                    )
                    candidate_coordinate = (
                        fixture_id,
                        tuple(sorted(candidate.items())),
                    )
                    if candidate_coordinate not in coordinates:
                        return _decision(
                            bound_plan,
                            value,
                            status="continue",
                            phase="minimization",
                            reason_code="mutation_value_contrast_missing",
                            experiment=_experiment(
                                fixtures,
                                fixture_id,
                                candidate,
                                purpose="isolate_minimal_trigger_value",
                            ),
                            evidence_basis=evidence,
                            retained_dimensions=[name],
                            target_hypotheses=_hypotheses_for_dimension(
                                bound_plan, dimension
                            ),
                            successful_driver_path=True,
                        )
            return _decision(
                bound_plan,
                value,
                status="stop",
                phase="confirmed",
                reason_code="reproducible_minimal_valid_failure_found",
                evidence_basis={
                    **evidence,
                    "minimality": "baseline_or_single_dimension_relative_to_baseline",
                },
                retained_dimensions=list(selection),
                successful_driver_path=True,
            )

    # Reachability is routed only from the latest result for each fixture.  A
    # later successful path therefore closes an earlier reachability failure.
    latest_by_fixture = _latest_per_fixture(evidence_results)
    for fixture_id in _ordered_fixture_ids(bound_plan, baseline=False):
        latest = latest_by_fixture.get(fixture_id)
        if latest is None:
            continue
        classification = str(latest.get("classification", ""))
        if not classification.startswith(("reachability_failure:", "completion_failure:")):
            continue
        reachability = (latest.get("reachability_gate") or {}).get("result") or {}
        unresolved = (
            (latest.get("failure_record") or {}).get(
                "earliest_unresolved_milestone"
            )
            or reachability.get("earliest_unresolved_milestone")
        )
        canonical_states = sorted(
            _PROGRESS_STATE_ALIASES.get(str(unresolved), {str(unresolved)})
        )
        canonical = canonical_states[0]
        eligible = [
            item
            for item in catalog
            if set(canonical_states)
            & set(item["normalized_target_progress_states"])
        ]
        for dimension in eligible:
            name = dimension["name"]
            for allowed in dimension["allowed_values"]:
                if allowed == dimension["baseline_value"]:
                    continue
                selection = {name: str(allowed)}
                if (fixture_id, tuple(sorted(selection.items()))) in coordinates:
                    continue
                return _decision(
                    bound_plan,
                    value,
                    status="continue",
                    phase="reachability_recovery",
                    reason_code="earliest_unresolved_milestone_routing",
                    experiment=_experiment(
                        fixtures,
                        fixture_id,
                        selection,
                        purpose=f"advance_{canonical}",
                    ),
                    evidence_basis={
                        "observation_id": latest.get("_observation_id"),
                        "earliest_unresolved_milestone": canonical,
                    },
                    retained_dimensions=[name],
                    target_hypotheses=_hypotheses_for_dimension(
                        bound_plan, dimension
                    ),
                )
        return _decision(
            bound_plan,
            value,
            status="stop",
            phase="reachability_exhausted",
            reason_code="no_untried_mutation_targets_unresolved_milestone",
            evidence_basis={
                "observation_id": latest.get("_observation_id"),
                "earliest_unresolved_milestone": canonical,
            },
        )

    # Information-gain ordering is structural: finish a partial same-surface
    # contrast family first, then start focus-bound dimensions, then other
    # compiled dimensions.  No empirical magic weights are used.
    focus_first = sorted(
        enumerate(catalog),
        key=lambda pair: (
            0 if (pair[1].get("binding") or {}).get("focus_predicate_id") else 1,
            pair[0],
        ),
    )
    fixture_order = _ordered_fixture_ids(bound_plan, baseline=False)
    mutation_families: list[tuple[dict[str, Any], dict[str, str]]] = []
    for _, dimension in focus_first:
        for allowed in dimension["allowed_values"]:
            if allowed != dimension["baseline_value"]:
                mutation_families.append(
                    (dimension, {dimension["name"]: str(allowed)})
                )
    partial: list[tuple[dict[str, Any], dict[str, str]]] = []
    unstarted: list[tuple[dict[str, Any], dict[str, str]]] = []
    for dimension, selection in mutation_families:
        observed = {
            fixture_id
            for fixture_id in fixture_order
            if (fixture_id, tuple(sorted(selection.items()))) in coordinates
        }
        (partial if observed else unstarted).append((dimension, selection))
    for dimension, selection in [*partial, *unstarted]:
        for fixture_id in fixture_order:
            coordinate = (fixture_id, tuple(sorted(selection.items())))
            if coordinate in coordinates:
                continue
            return _decision(
                bound_plan,
                value,
                status="continue",
                phase="exploration",
                reason_code=(
                    "complete_minimal_contrast_family"
                    if (dimension, selection) in partial
                    else "highest_structural_information_gain"
                ),
                experiment=_experiment(
                    fixtures,
                    fixture_id,
                    selection,
                    purpose="test_compiled_single_dimension_variation",
                ),
                retained_dimensions=[dimension["name"]],
                target_hypotheses=_hypotheses_for_dimension(
                    bound_plan, dimension
                ),
            )

    return _decision(
        bound_plan,
        value,
        status="stop",
        phase="exhausted",
        reason_code="all_compiled_single_dimension_candidates_exhausted",
    )
