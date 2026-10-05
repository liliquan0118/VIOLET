"""Versioned configuration contracts for source-driven test compilation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .artifacts import content_sha256


COMPILER_PROFILE_SCHEMA_VERSION = "agentspectesting.compiler-profile/v0.1"
SUT_ADAPTER_PROFILE_SCHEMA_VERSION = "agentspectesting.sut-adapter-profile/v0.1"
RUN_CONFIGURATION_SCHEMA_VERSION = "agentspectesting.run-configuration/v0.1"


class ConfigurationError(ValueError):
    """Raised when a versioned compiler, SUT, or run profile is invalid."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{path} must be a non-empty string")
    return value


def _string_list(value: Any, path: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a non-empty array" if nonempty else "an array"
        raise ConfigurationError(f"{path} must be {qualifier}")
    if any(not isinstance(item, str) or not item for item in value):
        raise ConfigurationError(f"{path} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ConfigurationError(f"{path} must not contain duplicates")
    return list(value)


def _positive_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{path} must be a positive number")
    return float(value)


def _validate_exact_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    extras = sorted(set(value) - allowed)
    if extras:
        raise ConfigurationError(
            f"{path} contains unsupported fields: {', '.join(extras)}"
        )


def validate_compiler_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(_mapping(profile, "$compiler_profile")))
    if value.get("schema_version") != COMPILER_PROFILE_SCHEMA_VERSION:
        raise ConfigurationError(
            f"unsupported compiler profile schema: {value.get('schema_version')!r}"
        )
    _string(value.get("profile_id"), "$compiler_profile.profile_id")

    experiment = _mapping(
        value.get("experiment_policy"), "$compiler_profile.experiment_policy"
    )
    enabled_experiments = _string_list(
        experiment.get("enabled_experiment_families"),
        "$compiler_profile.experiment_policy.enabled_experiment_families",
        nonempty=True,
    )
    required_experiments = {"nominal_witness", "minimal_contrast"}
    if not required_experiments.issubset(enabled_experiments):
        raise ConfigurationError(
            "$compiler_profile.experiment_policy must enable nominal_witness "
            "and minimal_contrast"
        )
    witness = _mapping(
        experiment.get("witness_policy"),
        "$compiler_profile.experiment_policy.witness_policy",
    )
    if witness.get("prefer_isolated_focus") is not True:
        raise ConfigurationError("witness_policy.prefer_isolated_focus must be true")
    if witness.get("require_outer_blockers_nonblocking") is not True:
        raise ConfigurationError(
            "witness_policy.require_outer_blockers_nonblocking must be true"
        )
    contrast = _mapping(
        experiment.get("contrast_policy"),
        "$compiler_profile.experiment_policy.contrast_policy",
    )
    if contrast.get("objective") != "minimum_policy_factor_hamming_distance":
        raise ConfigurationError(
            "contrast_policy.objective must be "
            "'minimum_policy_factor_hamming_distance'"
        )
    if contrast.get("require_unique_contrast") is not True:
        raise ConfigurationError("contrast_policy.require_unique_contrast must be true")

    probe = _mapping(value.get("probe_policy"), "$compiler_profile.probe_policy")
    strategies = _string_list(
        probe.get("comparison_strategies"),
        "$compiler_profile.probe_policy.comparison_strategies",
        nonempty=True,
    )
    allowed_strategies = {
        "just_below_threshold",
        "at_threshold",
        "just_above_threshold",
    }
    unknown = sorted(set(strategies) - allowed_strategies)
    if unknown:
        raise ConfigurationError(
            "unsupported comparison probe strategies: " + ", ".join(unknown)
        )
    epsilon = _mapping(
        probe.get("epsilon_policy"),
        "$compiler_profile.probe_policy.epsilon_policy",
    )
    if epsilon.get("mode") != "smallest_reliably_controllable_unit":
        raise ConfigurationError(
            "epsilon_policy.mode must be 'smallest_reliably_controllable_unit'"
        )

    hypothesis = _mapping(
        value.get("hypothesis_policy"), "$compiler_profile.hypothesis_policy"
    )
    _string_list(
        hypothesis.get("enabled_hypothesis_families"),
        "$compiler_profile.hypothesis_policy.enabled_hypothesis_families",
        nonempty=True,
    )

    mutation = _mapping(
        value.get("mutation_policy"), "$compiler_profile.mutation_policy"
    )
    catalog = mutation.get("operator_catalog")
    if not isinstance(catalog, list) or not catalog:
        raise ConfigurationError(
            "$compiler_profile.mutation_policy.operator_catalog must be a non-empty array"
        )
    names: set[str] = set()
    for index, raw in enumerate(catalog):
        path = f"$compiler_profile.mutation_policy.operator_catalog[{index}]"
        operator = _mapping(raw, path)
        name = _string(operator.get("name"), f"{path}.name")
        if name in names:
            raise ConfigurationError(f"duplicate mutation operator: {name!r}")
        names.add(name)
        _string(operator.get("applicability"), f"{path}.applicability")
        allowed_values = _string_list(
            operator.get("allowed_values"), f"{path}.allowed_values", nonempty=True
        )
        baseline_value = _string(
            operator.get("baseline_value"), f"{path}.baseline_value"
        )
        if baseline_value not in allowed_values:
            raise ConfigurationError(
                f"{path}.baseline_value must be one of allowed_values"
            )
        _string_list(
            operator.get("targets_progress_states"),
            f"{path}.targets_progress_states",
            nonempty=True,
        )
    max_changed = mutation.get("max_changed_dimensions_per_candidate")
    if not isinstance(max_changed, int) or isinstance(max_changed, bool) or max_changed < 1:
        raise ConfigurationError(
            "mutation_policy.max_changed_dimensions_per_candidate must be a positive integer"
        )
    if mutation.get("require_baseline") is not True:
        raise ConfigurationError("mutation_policy.require_baseline must be true")
    if mutation.get("require_semantic_invariant_check") is not True:
        raise ConfigurationError(
            "mutation_policy.require_semantic_invariant_check must be true"
        )

    guidance = _mapping(
        value.get("guidance_policy"), "$compiler_profile.guidance_policy"
    )
    _string(guidance.get("selection_strategy"), "guidance_policy.selection_strategy")
    _string_list(
        guidance.get("stop_conditions"),
        "$compiler_profile.guidance_policy.stop_conditions",
        nonempty=True,
    )
    return value


def validate_sut_adapter_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(_mapping(profile, "$sut_adapter_profile")))
    if value.get("schema_version") != SUT_ADAPTER_PROFILE_SCHEMA_VERSION:
        raise ConfigurationError(
            f"unsupported SUT adapter profile schema: {value.get('schema_version')!r}"
        )
    _string(value.get("profile_id"), "$sut_adapter_profile.profile_id")
    _string(value.get("sut"), "$sut_adapter_profile.sut")
    _string(value.get("domain"), "$sut_adapter_profile.domain")
    for field in ("fixture_operations", "observable_channels", "controllable_channels"):
        _string_list(value.get(field), f"$sut_adapter_profile.{field}", nonempty=True)
    runtime_catalog = _mapping(
        value.get("runtime_observation_catalog"),
        "$sut_adapter_profile.runtime_observation_catalog",
    )
    _validate_exact_keys(
        runtime_catalog,
        {
            "fact_request_phrases",
            "confirmation_request_phrases",
            "refusal_phrases",
            "transfer_phrases",
        },
        "$sut_adapter_profile.runtime_observation_catalog",
    )
    for field in (
        "confirmation_request_phrases",
        "refusal_phrases",
        "transfer_phrases",
    ):
        phrases = _string_list(
            runtime_catalog.get(field),
            f"$sut_adapter_profile.runtime_observation_catalog.{field}",
            nonempty=True,
        )
        if len({phrase.casefold() for phrase in phrases}) != len(phrases):
            raise ConfigurationError(
                "$sut_adapter_profile.runtime_observation_catalog."
                f"{field} contains duplicate phrases"
            )
    fact_requests = _mapping(
        runtime_catalog.get("fact_request_phrases"),
        "$sut_adapter_profile.runtime_observation_catalog.fact_request_phrases",
    )
    # The set of fact names is domain content, not a fixed schema: airline's
    # "user_id"/"reservation_id"/"operation_reason" trio was previously
    # hardcoded here, which would reject any other domain's real fact names
    # (e.g. telecom's "line_id", retail's "order_id") outright. Any nonempty
    # set of fact names is valid as long as each maps to a real, deduplicated,
    # nonempty phrase list -- same structural rigor, no fixed vocabulary.
    if not fact_requests:
        raise ConfigurationError(
            "$sut_adapter_profile.runtime_observation_catalog.fact_request_phrases "
            "must declare at least one fact name"
        )
    for fact_name in fact_requests:
        phrases = _string_list(
            fact_requests.get(fact_name),
            "$sut_adapter_profile.runtime_observation_catalog."
            f"fact_request_phrases.{fact_name}",
            nonempty=True,
        )
        if len({phrase.casefold() for phrase in phrases}) != len(phrases):
            raise ConfigurationError(
                "$sut_adapter_profile.runtime_observation_catalog."
                f"fact_request_phrases.{fact_name} contains duplicate phrases"
            )
    resolutions = _mapping(
        value.get("scalar_resolutions"), "$sut_adapter_profile.scalar_resolutions"
    )
    duration = _mapping(
        resolutions.get("duration"),
        "$sut_adapter_profile.scalar_resolutions.duration",
    )
    _positive_number(
        duration.get("value"), "$sut_adapter_profile.scalar_resolutions.duration.value"
    )
    if duration.get("unit") not in {"second", "minute", "hour", "day"}:
        raise ConfigurationError(
            "$sut_adapter_profile.scalar_resolutions.duration.unit is unsupported"
        )
    catalog = _mapping(
        value.get("fixture_binding_catalog"),
        "$sut_adapter_profile.fixture_binding_catalog",
    )
    observations = catalog.get("observation_bindings")
    if not isinstance(observations, list):
        raise ConfigurationError(
            "$sut_adapter_profile.fixture_binding_catalog.observation_bindings "
            "must be an array"
        )
    observation_ids: set[str] = set()
    for index, raw in enumerate(observations):
        path = f"$sut_adapter_profile.fixture_binding_catalog.observation_bindings[{index}]"
        binding = _mapping(raw, path)
        identifier = _string(
            binding.get("observation_value_id"), f"{path}.observation_value_id"
        )
        if identifier in observation_ids:
            raise ConfigurationError(f"duplicate observation binding: {identifier!r}")
        observation_ids.add(identifier)
        _string(binding.get("source_path"), f"{path}.source_path")
        if binding.get("materializer") is not None:
            materializer = _mapping(binding.get("materializer"), f"{path}.materializer")
            if materializer.get("kind") != "set_bound_reservation_field":
                raise ConfigurationError(
                    f"unsupported observation materializer kind: {materializer.get('kind')!r}"
                )
            _string(materializer.get("field"), f"{path}.materializer.field")
            if materializer.get("counterexample_policy") != "preserve_nonmatching_base":
                raise ConfigurationError(
                    f"{path}.materializer.counterexample_policy is unsupported"
                )
    semantics = catalog.get("semantic_predicate_bindings")
    if not isinstance(semantics, list):
        raise ConfigurationError(
            "$sut_adapter_profile.fixture_binding_catalog.semantic_predicate_bindings "
            "must be an array"
        )
    operand_ids: set[str] = set()
    for index, raw in enumerate(semantics):
        path = (
            "$sut_adapter_profile.fixture_binding_catalog."
            f"semantic_predicate_bindings[{index}]"
        )
        binding = _mapping(raw, path)
        identifier = _string(
            binding.get("condition_operand_id"), f"{path}.condition_operand_id"
        )
        if identifier in operand_ids:
            raise ConfigurationError(
                f"duplicate semantic predicate binding: {identifier!r}"
            )
        operand_ids.add(identifier)
        evaluator = _mapping(binding.get("evaluator"), f"{path}.evaluator")
        kind = _string(evaluator.get("kind"), f"{path}.evaluator.kind")
        if kind not in {
            "any_collection_field_equals",
            "dialogue_value_by_required_truth",
        }:
            raise ConfigurationError(
                f"unsupported fixture semantic evaluator kind: {kind!r}"
            )
        if binding.get("materializer") is not None:
            materializer = _mapping(binding.get("materializer"), f"{path}.materializer")
            if materializer.get("kind") != "set_first_bound_flight_instance_field":
                raise ConfigurationError(
                    f"unsupported semantic materializer kind: {materializer.get('kind')!r}"
                )
            _string(materializer.get("field"), f"{path}.materializer.field")
            if materializer.get("counterexample_policy") != "preserve_nonmatching_base":
                raise ConfigurationError(
                    f"{path}.materializer.counterexample_policy is unsupported"
                )
    return value


def validate_run_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(_mapping(configuration, "$run_configuration")))
    if value.get("schema_version") != RUN_CONFIGURATION_SCHEMA_VERSION:
        raise ConfigurationError(
            f"unsupported run configuration schema: {value.get('schema_version')!r}"
        )
    _string(value.get("configuration_id"), "$run_configuration.configuration_id")
    budgets = _mapping(value.get("budgets"), "$run_configuration.budgets")
    required = {
        "max_fixture_candidates",
        "max_state_patches",
        "max_dialogue_turns",
        "max_recovery_actions",
        "max_model_calls",
    }
    missing = sorted(required - set(budgets))
    if missing:
        raise ConfigurationError(
            "$run_configuration.budgets is missing: " + ", ".join(missing)
        )
    for key in sorted(required):
        raw = budgets.get(key)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ConfigurationError(f"$run_configuration.budgets.{key} must be >= 0")
    for key in ("max_fixture_candidates", "max_dialogue_turns"):
        if budgets[key] < 1:
            raise ConfigurationError(f"$run_configuration.budgets.{key} must be >= 1")
    return value


def profile_identity(profile: Mapping[str, Any]) -> dict[str, str]:
    profile_id = profile.get("profile_id") or profile.get("configuration_id")
    return {
        "profile_id": str(profile_id),
        "schema_version": str(profile.get("schema_version")),
        "fingerprint": content_sha256(profile),
    }


def load_configuration_file(
    path: str | Path,
    *,
    kind: str,
) -> dict[str, Any]:
    file = Path(path)
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot load {kind} configuration {file}: {exc}") from exc
    validators = {
        "compiler_profile": validate_compiler_profile,
        "sut_adapter_profile": validate_sut_adapter_profile,
        "run_configuration": validate_run_configuration,
    }
    try:
        validator = validators[kind]
    except KeyError as exc:
        raise ConfigurationError(f"unsupported configuration kind: {kind!r}") from exc
    return validator(value)
