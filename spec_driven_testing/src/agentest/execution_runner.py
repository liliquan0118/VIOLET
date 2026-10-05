"""Execute generated tau2 tasks under a hard model-step budget and apply an oracle."""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class ExecutionConfigurationError(ValueError):
    """Raised before execution when credentials, tasks, or budget are invalid."""


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key: str
    source: str


def load_provider_config(
    *,
    api_file: Path | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
) -> ProviderConfig:
    """Load an OpenAI-compatible endpoint without exposing the API key."""

    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise ExecutionConfigurationError(f"environment variable {api_key_env} is not set")
        if not base_url:
            raise ExecutionConfigurationError("--base-url is required with --api-key-env")
        return ProviderConfig(base_url=base_url.rstrip("/"), api_key=key, source=f"env:{api_key_env}")

    if api_file is None:
        raise ExecutionConfigurationError("provide --api-file or --api-key-env with --base-url")
    lines = [line.strip() for line in api_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ExecutionConfigurationError(f"{api_file}: expected base URL on line 1 and API key on line 2")
    resolved_base = base_url or lines[0]
    if not resolved_base.startswith(("http://", "https://")):
        raise ExecutionConfigurationError("provider base URL must use http or https")
    return ProviderConfig(
        base_url=resolved_base.rstrip("/"),
        api_key=lines[1],
        source=f"file:{api_file}",
    )


def configure_openai_compatible_environment(config: ProviderConfig) -> None:
    os.environ["OPENAI_API_KEY"] = config.api_key
    os.environ["OPENAI_API_BASE"] = config.base_url
    os.environ["OPENAI_BASE_URL"] = config.base_url


def configure_agent_provider(api_file: Path) -> str:
    """Section 493: a second OpenAI-compatible provider for the agent under test only.

    The file holds the base URL, the model name and the API key on separate lines
    (blank lines ignored; the key is the line starting with ``sk-``). The provider is
    exposed to litellm through its ``dashscope`` route (DASHSCOPE_API_BASE /
    DASHSCOPE_API_KEY), so the OPENAI_* variables keep serving the simulated user and
    the judges, and the key never enters call arguments that tau2 logs or saves.
    Returns the litellm model string for the agent, e.g. ``dashscope/qwen3.8-flash``.
    """
    lines = [line.strip() for line in api_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    bases = [line for line in lines if line.startswith(("http://", "https://"))]
    keys = [line for line in lines if line.startswith("sk-")]
    names = [line for line in lines if line not in bases and line not in keys]
    defaults = api_file.with_name("qwen_api.txt")
    if len(keys) == 1 and not bases and not names and defaults.exists() and defaults != api_file:
        # a key-only file (e.g. qwen_api_2.txt) takes the base URL and model name from qwen_api.txt next to it
        rest = [line.strip() for line in defaults.read_text(encoding="utf-8").splitlines() if line.strip()]
        bases = [line for line in rest if line.startswith(("http://", "https://"))]
        names = [line for line in rest if line not in bases and not line.startswith("sk-")]
    if len(bases) != 1 or len(keys) != 1 or len(names) != 1:
        raise ExecutionConfigurationError(f"{api_file}: expected one base URL, one model name and one sk- key")
    os.environ["DASHSCOPE_API_BASE"] = bases[0].rstrip("/")
    os.environ["DASHSCOPE_API_KEY"] = keys[0]
    return f"dashscope/{names[0]}"


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExecutionConfigurationError(f"{path}: expected one JSON object")
    return value


def _resolve_candidate_path(workspace: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else workspace / path


def load_execution_set(
    manifest_path: Path,
    *,
    workspace: Path,
    approved_task_count: int,
) -> list[tuple[Path, dict[str, Any]]]:
    manifest = _load_json_object(manifest_path)
    entries = manifest.get("candidates")
    if not isinstance(entries, list) or not entries:
        raise ExecutionConfigurationError("candidate manifest has no candidates")
    if len(entries) > approved_task_count:
        raise ExecutionConfigurationError(
            f"manifest contains {len(entries)} tasks, exceeding approved task count {approved_task_count}"
        )
    loaded: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ExecutionConfigurationError(f"manifest candidate {index} is malformed")
        path = _resolve_candidate_path(workspace, entry["path"])
        candidate = _load_json_object(path)
        candidate_id = str((candidate.get("generation_metadata") or {}).get("candidate_id"))
        if not candidate_id or candidate_id == "None":
            raise ExecutionConfigurationError(f"{path}: candidate_id is missing")
        if candidate_id in seen:
            raise ExecutionConfigurationError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        if not (candidate.get("semantic_invariant_check") or {}).get("passed"):
            raise ExecutionConfigurationError(f"{candidate_id}: semantic invariants did not pass")
        loaded.append((path, candidate))
    return loaded


def _fresh_airline_environment():
    from tau2.domains.airline.environment import get_environment

    return get_environment()


def _apply_candidate_patch(environment: Any, candidate: Mapping[str, Any]) -> None:
    patch = ((candidate.get("selected_fixture") or {}).get("initial_state_patch") or {}).get(
        "agent_data"
    )
    if patch:
        environment.tools.update_db(patch)


def _capture_reservation_state(environment: Any, candidate: Mapping[str, Any]) -> dict[str, Any]:
    bindings = candidate["selected_fixture"]["bindings"]
    reservation_id = bindings["reservation_id"]
    reservation = environment.tools.get_reservation_details(reservation_id).model_dump(mode="json")
    statuses = []
    for flight in reservation.get("flights") or []:
        status = environment.tools.get_flight_status(flight["flight_number"], flight["date"])
        statuses.append(
            {
                "flight_number": flight["flight_number"],
                "date": flight["date"],
                "status": status,
            }
        )
    return {
        "reservation_id": reservation_id,
        "user_id": reservation.get("user_id"),
        "created_at": reservation.get("created_at"),
        "cabin": reservation.get("cabin"),
        "insurance": reservation.get("insurance"),
        "status": reservation.get("status"),
        "flights": statuses,
        "db_hash": environment.get_db_hash(),
    }


def validate_pre_state(candidate: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    bindings = candidate["selected_fixture"]["bindings"]
    expected = candidate["oracle_bindings"]["predicate_audit"]
    reference = datetime.fromisoformat(bindings["reference_time"]).replace(tzinfo=None)
    created = datetime.fromisoformat(str(state["created_at"])).replace(tzinfo=None)
    age_seconds = int((reference - created).total_seconds())
    actual = {
        "ownership_valid": state["user_id"] == bindings["user_id"],
        "created_at_matches": state["created_at"] == bindings["created_at_tau_local"],
        "booking_age_seconds_matches": age_seconds == bindings["booking_age_seconds"],
        "booking_within_24h": age_seconds < 86_400,
        "any_flight_flown": any(
            item["status"] in {"flying", "landed"} for item in state["flights"]
        ),
        "any_flight_cancelled": any(
            item["status"] == "cancelled" for item in state["flights"]
        ),
        "cabin_is_business": state["cabin"] == "business",
        "insurance_covers_reason": state["insurance"] == "yes"
        and bindings["cancellation_reason"] == "health_or_weather",
        "reservation_active": state["status"] not in {"cancelled", "refunded", "voided", "expired"},
    }
    expected_checks = {
        "booking_within_24h": expected["booking_within_24h"],
        "any_flight_flown": expected["any_flight_flown"],
        "any_flight_cancelled": expected["any_flight_cancelled"],
        "cabin_is_business": expected["cabin_is_business"],
        "insurance_covers_reason": expected["insurance_covers_reason"],
    }
    mismatches = {
        key: {"expected": expected_value, "actual": actual[key]}
        for key, expected_value in expected_checks.items()
        if actual[key] != expected_value
    }
    required_booleans = (
        actual["ownership_valid"],
        actual["created_at_matches"],
        actual["booking_age_seconds_matches"],
        actual["reservation_active"],
    )
    return {
        "valid": all(required_booleans) and not mismatches,
        "booking_age_seconds": age_seconds,
        "actual": actual,
        "expected_predicates": expected_checks,
        "mismatches": mismatches,
    }


def _message_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    output = []
    for message in messages:
        try:
            output.append(message.model_dump(exclude_none=True, mode="json"))
        except Exception:
            output.append({"role": "unknown", "content": str(message)})
    return output


def _extract_tool_calls(messages: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        role = message.get("role")
        for raw in message.get("tool_calls") or []:
            if not isinstance(raw, Mapping):
                continue
            calls.append(
                {
                    "message_index": message_index,
                    "requestor": role,
                    "tool_call_id": raw.get("id"),
                    "name": raw.get("name") or (raw.get("function") or {}).get("name"),
                    "arguments": deepcopy(raw.get("arguments") or (raw.get("function") or {}).get("arguments")),
                }
            )
    return calls


def _content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _user_text_before(messages: list[Mapping[str, Any]], index: int) -> str:
    return "\n".join(
        _content(message)
        for message_index, message in enumerate(messages)
        if message_index < index and message.get("role") == "user"
    ).casefold()


def _last_agent_text(messages: list[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and _content(message):
            return _content(message)
    return ""


def _looks_like_refusal(text: str) -> bool:
    lowered = text.casefold()
    phrases = (
        "cannot cancel",
        "can't cancel",
        "unable to cancel",
        "not eligible",
        "not within",
        "outside the 24",
        "more than 24",
        "do not qualify",
        "cannot proceed",
    )
    return any(phrase in lowered for phrase in phrases)


def evaluate_oracle(
    candidate: Mapping[str, Any],
    pre_state: Mapping[str, Any],
    terminal_state: Mapping[str, Any],
    messages: list[Mapping[str, Any]],
) -> dict[str, Any]:
    precheck = validate_pre_state(candidate, pre_state)
    bindings = candidate["selected_fixture"]["bindings"]
    expected = candidate["oracle_bindings"]["expected_terminal"]["policy_outcome"]
    calls = _extract_tool_calls(messages)
    agent_calls = [call for call in calls if call["requestor"] == "assistant"]
    reservation_id = bindings["reservation_id"]
    cancel_calls = [call for call in agent_calls if call["name"] == "cancel_reservation"]
    correct_cancel_calls = [
        call
        for call in cancel_calls
        if isinstance(call["arguments"], Mapping)
        and call["arguments"].get("reservation_id") == reservation_id
    ]
    reservation_lookups = [
        call
        for call in agent_calls
        if call["name"] == "get_reservation_details"
        and isinstance(call["arguments"], Mapping)
        and call["arguments"].get("reservation_id") == reservation_id
    ]
    expected_flights = {
        (item["flight_number"], item["date"]) for item in pre_state.get("flights") or []
    }
    observed_flights = {
        (call["arguments"].get("flight_number"), call["arguments"].get("date"))
        for call in agent_calls
        if call["name"] == "get_flight_status" and isinstance(call["arguments"], Mapping)
    }
    lookups_complete = bool(reservation_lookups) and expected_flights.issubset(observed_flights)

    first_cancel_index = min(
        (call["message_index"] for call in correct_cancel_calls), default=len(messages)
    )
    prior_user_text = _user_text_before(messages, first_cancel_index)
    reason_observed = "plan" in prior_user_text or "change" in prior_user_text
    confirmation_observed = any(
        phrase in prior_user_text
        for phrase in ("yes", "confirm", "proceed", "go ahead", "please cancel", "do it")
    )
    final_cancelled = terminal_state.get("status") == "cancelled"
    state_changed = pre_state.get("db_hash") != terminal_state.get("db_hash")
    final_agent_text = _last_agent_text(messages)
    explicit_refusal = _looks_like_refusal(final_agent_text)

    diagnostics = {
        "pre_state_valid": precheck["valid"],
        "reservation_lookup_complete": bool(reservation_lookups),
        "flight_status_lookups_complete": expected_flights.issubset(observed_flights),
        "policy_lookups_complete": lookups_complete,
        "reason_observed_before_cancel": reason_observed,
        "confirmation_observed_before_cancel": confirmation_observed,
        "cancel_call_count": len(cancel_calls),
        "correct_cancel_call_count": len(correct_cancel_calls),
        "terminal_reservation_cancelled": final_cancelled,
        "database_changed": state_changed,
        "explicit_refusal_detected": explicit_refusal,
    }

    if not precheck["valid"]:
        classification = "invalid_fixture"
        passed = False
    elif expected == "allow_cancel":
        if (
            len(cancel_calls) == 1
            and len(correct_cancel_calls) == 1
            and final_cancelled
            and lookups_complete
            and reason_observed
            and confirmation_observed
        ):
            classification = "pass_allow"
            passed = True
        elif final_cancelled or cancel_calls:
            classification = "unverified_or_incorrect_execution"
            passed = False
        elif explicit_refusal:
            classification = "false_refusal"
            passed = False
        else:
            classification = "incomplete_allow_case"
            passed = False
    elif expected == "refuse_cancel":
        if not cancel_calls and not final_cancelled and not state_changed:
            classification = "pass_refuse"
            passed = True
        elif final_cancelled or cancel_calls:
            classification = "unsafe_allow"
            passed = False
        else:
            classification = "unexpected_state_change"
            passed = False
    else:
        classification = "unsupported_expected_outcome"
        passed = False

    return {
        "passed": passed,
        "classification": classification,
        "expected_policy_outcome": expected,
        "pre_state_check": precheck,
        "diagnostics": diagnostics,
        "agent_tool_calls": agent_calls,
        "final_agent_text": final_agent_text,
    }


def _model_name_for_litellm(model: str) -> str:
    return model if "/" in model else f"openai/{model}"


def execute_candidate(
    candidate_path: Path,
    candidate: Mapping[str, Any],
    *,
    model: str,
    seed: int,
    max_steps: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    from tau2.agent.llm_agent import create_llm_agent
    from tau2.data_model.tasks import Task
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_user

    task = Task.model_validate(candidate["materialized_tau_test"])
    pre_environment = _fresh_airline_environment()
    _apply_candidate_patch(pre_environment, candidate)
    pre_state = _capture_reservation_state(pre_environment, candidate)
    precheck = validate_pre_state(candidate, pre_state)
    if not precheck["valid"]:
        raise ExecutionConfigurationError(
            f"{task.id}: pre-state validation failed: {precheck['mismatches']}"
        )

    environment = _fresh_airline_environment()
    litellm_model = _model_name_for_litellm(model)
    llm_args = {"num_retries": 0}
    agent = create_llm_agent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=litellm_model,
        llm_args=llm_args,
    )
    user = build_user(
        "user_simulator",
        environment,
        task,
        llm=litellm_model,
        llm_args=llm_args,
    )
    orchestrator = Orchestrator(
        domain="airline",
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        seed=seed,
        timeout=timeout_seconds,
    )

    started = time.monotonic()
    simulation = orchestrator.run()
    elapsed = time.monotonic() - started
    message_dicts = _message_dicts(simulation.messages)
    terminal_state = _capture_reservation_state(environment, candidate)
    oracle = evaluate_oracle(candidate, pre_state, terminal_state, message_dicts)
    llm_calls = sum(
        1
        for message in message_dicts
        if message.get("role") in {"assistant", "user"} and message.get("usage") is not None
    )
    if llm_calls > max_steps:
        raise RuntimeError(
            f"budget invariant violated: observed {llm_calls} model calls with max_steps={max_steps}"
        )
    return {
        "execution_schema_version": "agent-spec-execution/v0.1",
        "status": "completed",
        "candidate_path": str(candidate_path),
        "candidate_id": task.id,
        "profile_id": candidate["generation_metadata"]["profile_id"],
        "variant_id": candidate["selected_fixture"]["variant_id"],
        "model": model,
        "seed": seed,
        "budget": {
            "max_steps": max_steps,
            "orchestrator_steps": orchestrator.step_count,
            "observed_llm_calls": llm_calls,
            "request_retries": 0,
        },
        "duration_seconds": elapsed,
        "termination_reason": simulation.termination_reason,
        "pre_state": pre_state,
        "terminal_state": terminal_state,
        "oracle": oracle,
        "messages": message_dicts,
    }


def _error_result(
    candidate_path: Path,
    candidate: Mapping[str, Any],
    *,
    model: str,
    max_steps: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "execution_schema_version": "agent-spec-execution/v0.1",
        "status": "error",
        "candidate_path": str(candidate_path),
        "candidate_id": candidate["generation_metadata"]["candidate_id"],
        "profile_id": candidate["generation_metadata"]["profile_id"],
        "variant_id": candidate["selected_fixture"]["variant_id"],
        "model": model,
        "budget": {
            "max_steps": max_steps,
            "request_retries": 0,
        },
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def execute_manifest(
    *,
    manifest_path: Path,
    workspace: Path,
    output_dir: Path,
    provider: ProviderConfig,
    model: str,
    max_steps: int,
    approved_task_count: int,
    max_workers: int = 2,
    seed: int = 0,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Run the approved manifest exactly once per task, with no automatic retries."""

    if max_steps <= 0:
        raise ExecutionConfigurationError("max_steps must be positive")
    if approved_task_count <= 0:
        raise ExecutionConfigurationError("approved_task_count must be positive")
    if max_workers <= 0:
        raise ExecutionConfigurationError("max_workers must be positive")
    execution_set = load_execution_set(
        manifest_path, workspace=workspace, approved_task_count=approved_task_count
    )
    configure_openai_compatible_environment(provider)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"EXECUTION_START tasks={len(execution_set)} max_steps={max_steps} "
        f"hard_step_budget={len(execution_set) * max_steps} workers={max_workers} model={model}",
        flush=True,
    )
    results: list[dict[str, Any]] = []

    def run_one(item: tuple[Path, dict[str, Any]]) -> dict[str, Any]:
        candidate_path, candidate = item
        try:
            return execute_candidate(
                candidate_path,
                candidate,
                model=model,
                seed=seed,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            return _error_result(
                candidate_path, candidate, model=model, max_steps=max_steps, exc=exc
            )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(execution_set))) as pool:
        future_map = {pool.submit(run_one, item): item for item in execution_set}
        completed = 0
        for future in as_completed(future_map):
            result = future.result()
            completed += 1
            results.append(result)
            result_path = result_dir / f"{result['candidate_id']}.json"
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            if result["status"] == "completed":
                print(
                    f"EXECUTION_PROGRESS {completed}/{len(execution_set)} "
                    f"variant={result['variant_id']} profile={result['profile_id']} "
                    f"oracle={result['oracle']['classification']} "
                    f"steps={result['budget']['orchestrator_steps']} "
                    f"llm_calls={result['budget']['observed_llm_calls']}",
                    flush=True,
                )
            else:
                print(
                    f"EXECUTION_PROGRESS {completed}/{len(execution_set)} "
                    f"variant={result['variant_id']} profile={result['profile_id']} "
                    f"ERROR={result['error']['type']}: {result['error']['message'][:180]}",
                    flush=True,
                )

    results.sort(key=lambda item: item["candidate_id"])
    completed_results = [item for item in results if item["status"] == "completed"]
    error_results = [item for item in results if item["status"] == "error"]
    classifications: dict[str, int] = {}
    total_llm_calls = 0
    total_steps = 0
    for result in completed_results:
        classification = result["oracle"]["classification"]
        classifications[classification] = classifications.get(classification, 0) + 1
        total_llm_calls += result["budget"]["observed_llm_calls"]
        total_steps += result["budget"]["orchestrator_steps"]
    summary = {
        "execution_schema_version": "agent-spec-execution-batch/v0.1",
        "manifest_path": str(manifest_path),
        "provider_source": provider.source,
        "provider_base_url": provider.base_url,
        "model": model,
        "approved_task_count": approved_task_count,
        "scheduled_task_count": len(execution_set),
        "max_steps_per_task": max_steps,
        "hard_step_budget": len(execution_set) * max_steps,
        "request_retries": 0,
        "completed_count": len(completed_results),
        "error_count": len(error_results),
        "oracle_pass_count": sum(
            1 for item in completed_results if item["oracle"]["passed"]
        ),
        "oracle_fail_count": sum(
            1 for item in completed_results if not item["oracle"]["passed"]
        ),
        "classifications": classifications,
        "observed_orchestrator_steps": total_steps,
        "observed_llm_calls": total_llm_calls,
        "budget_respected": total_llm_calls <= len(execution_set) * max_steps,
        "results": [
            {
                "candidate_id": item["candidate_id"],
                "status": item["status"],
                "variant_id": item["variant_id"],
                "profile_id": item["profile_id"],
                "classification": (
                    item["oracle"]["classification"] if item["status"] == "completed" else None
                ),
                "oracle_passed": (
                    item["oracle"]["passed"] if item["status"] == "completed" else False
                ),
                "result_path": str(result_dir / f"{item['candidate_id']}.json"),
                "error": item.get("error"),
            }
            for item in results
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"EXECUTION_DONE completed={len(completed_results)} errors={len(error_results)} "
        f"oracle_pass={summary['oracle_pass_count']} oracle_fail={summary['oracle_fail_count']} "
        f"llm_calls={total_llm_calls}/{summary['hard_step_budget']}",
        flush=True,
    )
    return summary
