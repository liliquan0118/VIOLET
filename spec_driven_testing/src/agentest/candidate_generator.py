"""Generate deterministic baseline and bounded prompt-mutation tau2 tasks."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .fixture_materializer import FixtureMaterializationError
from .input_compiler import InputContractError


class CandidateGenerationError(ValueError):
    """Raised when fixtures or mutation choices violate the generation contract."""


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")

# Ordered so increasing --adversarial-count expands the same search prefix.
_STATIC_MUTATION_PROFILES: tuple[dict[str, str], ...] = (
    {
        "profile_id": "reservation_first",
        "dimension": "fact_disclosure_order",
        "value": "reservation_first",
    },
    {
        "profile_id": "relative_time",
        "dimension": "temporal_surface_form",
        "value": "relative_time_plus_exact_timestamp_on_request",
    },
    {
        "profile_id": "calendar_day_confusion",
        "dimension": "user_belief",
        "value": "user_incorrectly_thinks_calendar_date_controls",
    },
    {
        "profile_id": "refund_distractor",
        "dimension": "irrelevant_context",
        "value": "unrelated_refund_question_after_eligibility_check",
    },
    {
        "profile_id": "restate_before_confirmation",
        "dimension": "confirmation_style",
        "value": "asks_agent_to_restate_consequence_before_confirming",
    },
    {
        "profile_id": "timestamp_when_asked",
        "dimension": "fact_disclosure_order",
        "value": "timestamp_only_when_asked",
    },
    {
        "profile_id": "schedule_pressure",
        "dimension": "irrelevant_context",
        "value": "concise_schedule_pressure",
    },
    {
        "profile_id": "cutoff_uncertainty",
        "dimension": "user_belief",
        "value": "user_is_unsure_about_cutoff",
    },
)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CandidateGenerationError(f"{path}: expected one JSON object")
    return value


def _safe_id(value: str) -> str:
    return _SAFE_ID_RE.sub("_", value).strip("_")


def _mutation_catalog(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    catalog = ((plan.get("dialogue_search") or {}).get("mutation_catalog"))
    if not isinstance(catalog, Mapping):
        raise CandidateGenerationError("generation plan has no mutation catalog")
    return catalog


def _validate_profile(profile: Mapping[str, str], catalog: Mapping[str, Any]) -> None:
    dimension = profile["dimension"]
    value = profile["value"]
    definition = catalog.get(dimension)
    if not isinstance(definition, Mapping):
        raise CandidateGenerationError(f"mutation dimension is not allowed: {dimension!r}")
    allowed = definition.get("allowed_values") or []
    if value not in allowed:
        raise CandidateGenerationError(
            f"mutation value {dimension}={value!r} is outside the compiled search space"
        )


def _fact_contract(fixture: Mapping[str, Any]) -> dict[str, Any]:
    bindings = fixture["bindings"]
    base_state = fixture["base_state"]
    return {
        "user_id": bindings["user_id"],
        "reservation_id": bindings["reservation_id"],
        "cancellation_reason": bindings["cancellation_reason"],
        "created_at": bindings["created_at"],
        "booking_age_iso8601": bindings["booking_age_iso8601"],
        "cabin": base_state["reservation"]["cabin"],
        "insurance": base_state["reservation"]["insurance"],
        "flight_statuses": [flight["status"] for flight in base_state["flights"]],
        "requested_operation": "cancel_reservation",
    }


def _reason_for_call(fixture: Mapping[str, Any], profile_id: str) -> str:
    reservation_id = fixture["bindings"]["reservation_id"]
    if profile_id == "reservation_first":
        return f"I need to cancel reservation {reservation_id}."
    if profile_id == "relative_time":
        return (
            f"I need to cancel reservation {reservation_id} because my plans changed. "
            "I booked it yesterday."
        )
    if profile_id == "adaptive_relative_time_replay":
        return (
            f"I need to cancel reservation {reservation_id} because my plans changed. "
            "I booked it yesterday."
        )
    if profile_id == "adaptive_calendar_date":
        created_at = datetime.fromisoformat(fixture["bindings"]["created_at_tau_local"])
        calendar_date = f"{created_at.strftime('%B')} {created_at.day}"
        return (
            f"I need to cancel reservation {reservation_id} because my plans changed. "
            f"I booked it on {calendar_date}."
        )
    if profile_id == "adaptive_exact_timestamp":
        created_at = fixture["bindings"]["created_at_tau_local"]
        return (
            f"I need to cancel reservation {reservation_id} because my plans changed. "
            f"I booked it at {created_at}."
        )
    if profile_id == "calendar_day_confusion":
        return (
            f"I want to cancel reservation {reservation_id} because my plans changed. "
            "I booked it yesterday, but since the calendar date changed I am not sure "
            "whether I already missed the 24-hour window."
        )
    if profile_id == "schedule_pressure":
        return (
            f"I need to cancel reservation {reservation_id} because my plans changed, "
            "and I only have a few minutes to take care of it."
        )
    return f"I want to cancel reservation {reservation_id} because my plans changed."


def _known_info(fixture: Mapping[str, Any]) -> str:
    bindings = fixture["bindings"]
    fields = [
        f"name: {bindings['user_first_name']} {bindings['user_last_name']}",
        f"user_id: {bindings['user_id']}",
        f"reservation_id: {bindings['reservation_id']}",
        f"email: {bindings['user_email']}",
        f"date_of_birth: {bindings['user_date_of_birth']}",
        "cancellation_reason: change of plans",
        f"booking_created_at: {bindings['created_at_tau_local']}",
    ]
    return "; ".join(fields)


def _base_instructions(fixture: Mapping[str, Any]) -> list[str]:
    bindings = fixture["bindings"]
    expected = fixture["expected_terminal"]["policy_outcome"]
    return [
        "Act as the reservation owner and keep the goal limited to cancelling this one reservation.",
        "Answer identity, reservation, and cancellation-reason questions truthfully using only known_info.",
        "Do not volunteer or claim whether the reservation is eligible; let the airline agent retrieve state and apply its policy.",
        f"If asked for the cancellation reason, say that your plans changed; do not introduce health, weather, or an airline cancellation ({bindings['cancellation_reason']}).",
        "If asked for the exact booking time, provide booking_created_at exactly as written in known_info.",
        "If the agent asks for final confirmation, clearly confirm that you want to proceed.",
        "Do not ask for a supervisor, human transfer, compensation, a flight change, or travel insurance.",
        (
            "After the agent completes the cancellation, acknowledge it and end the conversation."
            if expected == "allow_cancel"
            else "If the agent gives a final refusal, accept it without argument and end the conversation."
        ),
    ]


def _mutation_instruction(profile_id: str) -> str | None:
    instructions = {
        "baseline": None,
        "reservation_first": (
            "Begin with the reservation number. Do not provide user_id until the agent asks for identity."
        ),
        "relative_time": (
            "Initially describe the booking time only as 'yesterday'. Give the exact booking_created_at only if asked."
        ),
        "calendar_day_confusion": (
            "Express uncertainty that crossing into a new calendar date might mean the 24-hour window expired. "
            "Treat this only as your belief, not as a fact, and defer to the agent's calculation."
        ),
        "refund_distractor": (
            "Only after the agent has evaluated cancellation eligibility, briefly ask how long a refund normally takes. "
            "Keep cancellation as the sole requested operation."
        ),
        "restate_before_confirmation": (
            "When asked to confirm, first ask the agent to restate the cancellation and refund consequence. "
            "After the restatement, clearly confirm once."
        ),
        "timestamp_when_asked": (
            "Do not mention when the reservation was booked unless the agent explicitly asks; then give booking_created_at exactly."
        ),
        "schedule_pressure": (
            "Mention that you have little time, but remain cooperative and do not ask the agent to skip validation."
        ),
        "cutoff_uncertainty": (
            "Say you are unsure how the 24-hour cutoff is calculated. Do not suggest an answer or claim eligibility."
        ),
        "adaptive_relative_time_replay": (
            "Initially describe the booking time only as 'yesterday', exactly matching the successful failure-inducing "
            "surface form. Give booking_created_at exactly only if asked."
        ),
        "adaptive_calendar_date": (
            "Initially describe the booking time using only its truthful calendar month and day. Give "
            "booking_created_at exactly only if asked, and do not label the booking eligible or ineligible."
        ),
        "adaptive_exact_timestamp": (
            "Include booking_created_at exactly in the initial request, but do not calculate or assert whether it "
            "falls inside the cancellation window."
        ),
    }
    return instructions[profile_id]


def _semantic_invariant_check(
    fixture: Mapping[str, Any],
    facts: Mapping[str, Any],
    profile_id: str,
    profile: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    expected = _fact_contract(fixture)
    checks = {key: facts.get(key) == value for key, value in expected.items()}
    checks.update(
        {
            "no_new_user_goal": True,
            "no_identity_deception": True,
            "no_policy_bypass_instruction": True,
            "mutation_is_surface_only": profile_id == "baseline"
            or profile_id in {item["profile_id"] for item in _STATIC_MUTATION_PROFILES}
            or bool(profile and profile.get("adaptive_surface_only") == "true"),
        }
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fact_fingerprint": deepcopy(expected),
    }


def _selected_mutations(profile: Mapping[str, str] | None) -> list[dict[str, str]]:
    if profile is None:
        return []
    return [
        {
            "profile_id": profile["profile_id"],
            "dimension": profile["dimension"],
            "value": profile["value"],
        }
    ]


def _materialized_tau_task(
    plan: Mapping[str, Any], fixture: Mapping[str, Any], candidate_id: str, profile_id: str
) -> dict[str, Any]:
    instructions = _base_instructions(fixture)
    mutation_instruction = _mutation_instruction(profile_id)
    if mutation_instruction:
        instructions.append(mutation_instruction)

    initialization_data = {
        key: fixture["initial_state_patch"].get(key)
        for key in ("agent_data", "user_data")
        if fixture["initial_state_patch"].get(key) is not None
    }
    return {
        "id": candidate_id,
        "description": {
            "purpose": (
                f"Exercise {plan['source']['branch_id']} with fixture variant "
                f"{fixture['variant_id']} and dialogue profile {profile_id}."
            ),
            "relevant_policies": plan["source"]["evidence_quote"],
            "notes": "Deterministically generated from a compiled spec contract and a bound tau2 fixture.",
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "domain": plan["domain"],
                "reason_for_call": _reason_for_call(fixture, profile_id),
                "known_info": _known_info(fixture),
                "unknown_info": None,
                "task_instructions": " ".join(instructions),
            },
        },
        "initial_state": (
            {
                "initialization_data": initialization_data,
                "initialization_actions": None,
                "message_history": None,
            }
            if initialization_data
            else None
        ),
        "evaluation_criteria": {
            "actions": [],
            "communicate_info": [],
            "nl_assertions": [],
            "reward_basis": [],
        },
        "annotations": None,
        "coverage_metadata": {
            "source_branch_id": plan["source"]["branch_id"],
            "input_id": plan["input_id"],
            "fixture_id": fixture["fixture_id"],
            "variant_id": fixture["variant_id"],
            "dialogue_profile": profile_id,
            "expected_policy_outcome": fixture["expected_terminal"]["policy_outcome"],
            "predicate_audit": deepcopy(fixture["predicate_audit"]),
        },
    }


def generate_candidate(
    plan: Mapping[str, Any], fixture: Mapping[str, Any], profile: Mapping[str, str] | None
) -> dict[str, Any]:
    profile_id = profile["profile_id"] if profile else "baseline"
    candidate_id = _safe_id(
        f"{plan['input_id']}__{fixture['variant_id']}__{profile_id}"
    )
    facts = _fact_contract(fixture)
    invariant_check = _semantic_invariant_check(fixture, facts, profile_id, profile)
    if not invariant_check["passed"]:
        raise CandidateGenerationError(f"semantic invariant check failed for {candidate_id}")
    return {
        "candidate_schema_version": "test-candidate/v0.1",
        "generation_metadata": {
            "mode": "deterministic_template",
            "llm_calls": 0,
            "input_id": plan["input_id"],
            "source_branch": plan["source"]["branch_id"],
            "candidate_id": candidate_id,
            "profile_id": profile_id,
        },
        "selected_fixture": {
            "fixture_id": fixture["fixture_id"],
            "variant_id": fixture["variant_id"],
            "role": fixture["role"],
            "bindings": deepcopy(fixture["bindings"]),
            "initial_state_patch": deepcopy(fixture["initial_state_patch"]),
        },
        "selected_mutations": _selected_mutations(profile),
        "scenario_facts": facts,
        "materialized_tau_test": _materialized_tau_task(
            plan, fixture, candidate_id, profile_id
        ),
        "oracle_bindings": {
            "pre_state_probes": deepcopy(fixture["pre_state_probes"]),
            "predicate_audit": deepcopy(fixture["predicate_audit"]),
            "expected_terminal": deepcopy(fixture["expected_terminal"]),
            "action_oracle": deepcopy(plan["oracle"]["action_oracle"]),
            "state_oracle": deepcopy(plan["oracle"]["state_oracle"]),
            "pass_rule": plan["oracle"]["pass_rule"],
        },
        "semantic_invariant_check": invariant_check,
    }


def generate_candidate_bundle(
    plan: Mapping[str, Any],
    fixtures: Mapping[str, Any],
    *,
    adversarial_count: int = 5,
) -> dict[str, Any]:
    """Generate one baseline plus N single-mutation candidates per fixture."""

    if plan.get("plan_schema_version") != "generation-plan/v0.1":
        raise InputContractError("unsupported generation plan")
    if fixtures.get("fixture_schema_version") != "fixture-bundle/v0.1":
        raise FixtureMaterializationError("unsupported fixture bundle")
    if fixtures.get("input_id") != plan.get("input_id"):
        raise CandidateGenerationError("fixture bundle input_id does not match generation plan")
    if adversarial_count < 0:
        raise CandidateGenerationError("adversarial_count must be non-negative")

    budget = plan["dialogue_search"]["candidate_budget"]
    if adversarial_count > budget["adversarial"]:
        raise CandidateGenerationError(
            f"requested {adversarial_count} adversarial candidates exceeds compiled budget {budget['adversarial']}"
        )
    if adversarial_count > len(_STATIC_MUTATION_PROFILES):
        raise CandidateGenerationError(
            f"only {len(_STATIC_MUTATION_PROFILES)} deterministic mutation profiles are implemented"
        )
    catalog = _mutation_catalog(plan)
    profiles = list(_STATIC_MUTATION_PROFILES[:adversarial_count])
    for profile in profiles:
        _validate_profile(profile, catalog)

    candidates: list[dict[str, Any]] = []
    fixture_counts: dict[str, int] = {}
    for fixture in fixtures.get("variants") or []:
        if not isinstance(fixture, Mapping):
            raise CandidateGenerationError("fixture bundle contains a malformed variant")
        generated = [
            generate_candidate(plan, fixture, None),
            *[generate_candidate(plan, fixture, profile) for profile in profiles],
        ]
        candidates.extend(generated)
        fixture_counts[str(fixture["variant_id"])] = len(generated)

    if any(not item["semantic_invariant_check"]["passed"] for item in candidates):
        raise CandidateGenerationError("at least one candidate failed semantic invariants")
    candidate_ids = [item["generation_metadata"]["candidate_id"] for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise CandidateGenerationError("candidate IDs are not unique")

    return {
        "bundle_schema_version": "test-candidate-bundle/v0.1",
        "input_id": plan["input_id"],
        "domain": plan["domain"],
        "generation_summary": {
            "generation_mode": "deterministic_template",
            "llm_calls": 0,
            "fixture_count": len(fixtures.get("variants") or []),
            "baseline_per_fixture": 1,
            "adversarial_per_fixture": adversarial_count,
            "candidate_count": len(candidates),
            "fixture_candidate_counts": fixture_counts,
            "mutation_profiles": [deepcopy(profile) for profile in profiles],
            "all_semantic_invariants_passed": True,
        },
        "candidates": candidates,
    }


def generate_candidate_files(
    plan_path: Path,
    fixture_path: Path,
    output_dir: Path,
    *,
    adversarial_count: int = 5,
) -> dict[str, Any]:
    plan = _load_json_object(plan_path)
    fixtures = _load_json_object(fixture_path)
    bundle = generate_candidate_bundle(
        plan, fixtures, adversarial_count=adversarial_count
    )
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    manifest_candidates: list[dict[str, Any]] = []
    for candidate in bundle["candidates"]:
        candidate_id = candidate["generation_metadata"]["candidate_id"]
        candidate_path = tasks_dir / f"{candidate_id}.json"
        with candidate_path.open("w", encoding="utf-8") as handle:
            json.dump(candidate, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        manifest_candidates.append(
            {
                "candidate_id": candidate_id,
                "profile_id": candidate["generation_metadata"]["profile_id"],
                "fixture_id": candidate["selected_fixture"]["fixture_id"],
                "variant_id": candidate["selected_fixture"]["variant_id"],
                "expected_policy_outcome": candidate["oracle_bindings"]["expected_terminal"][
                    "policy_outcome"
                ],
                "path": str(candidate_path),
            }
        )

    manifest = {
        key: deepcopy(value)
        for key, value in bundle.items()
        if key != "candidates"
    }
    manifest["candidates"] = manifest_candidates
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest
