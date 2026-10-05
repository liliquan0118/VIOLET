"""Approval-bounded execution loop for deterministic Guidance.

The loop is deliberately batch-bounded.  It cannot execute unless both an
experiment count and a per-experiment target-agent call budget are supplied.
When that allowance is consumed it returns control to the caller, even when
Guidance has another candidate ready.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

from ..compiler.artifacts import content_sha256
from ..compiler.profiles import validate_run_configuration
from ..driver.tau_online import run_tau_online_driver
from .controller import (
    GuidanceError,
    decide_next_experiment,
    record_guidance_observation,
)


GUIDANCE_BATCH_SCHEMA_VERSION = "agentspectesting.guidance-batch/v0.1"
GUIDANCE_BATCH_RUNNER_VERSION = "approval-bounded-guidance-runner/v0.1"

GuidanceExecutor = Callable[..., Mapping[str, Any]]
StepCallback = Callable[[Mapping[str, Any], Mapping[str, Any]], None]
DecisionCallback = Callable[[int, Mapping[str, Any]], None]
ErrorCallback = Callable[[int, Mapping[str, Any], Exception], None]


class GuidanceLoopError(ValueError):
    """Raised when approval, execution, or checkpoint contracts are violated."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GuidanceLoopError(f"{path} must be an object")
    return value


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise GuidanceLoopError(f"{path} must be a positive integer")
    return value


def _experiment_coordinate(decision: Mapping[str, Any]) -> dict[str, Any]:
    experiment = _mapping(decision.get("experiment"), "$.decision.experiment")
    fixture_id = experiment.get("fixture_instance_id")
    selection = experiment.get("variation_selection")
    if not isinstance(fixture_id, str) or not fixture_id:
        raise GuidanceLoopError("Guidance decision has no fixture instance ID")
    if not isinstance(selection, Mapping):
        raise GuidanceLoopError("Guidance decision has no variation selection")
    return {
        "fixture_instance_id": fixture_id,
        "variation_selection": deepcopy(dict(selection)),
    }


def _validate_execution(
    execution: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    approved_agent_call_budget: int,
) -> dict[str, Any]:
    value = deepcopy(dict(_mapping(execution, "$.execution")))
    expected = _experiment_coordinate(decision)
    actual = {
        "fixture_instance_id": value.get("fixture_instance_id"),
        "variation_selection": deepcopy(dict(value.get("variation_selection") or {})),
    }
    if actual != expected:
        raise GuidanceLoopError(
            "executor returned a different experiment coordinate than Guidance selected"
        )
    budget = _mapping(value.get("budget"), "$.execution.budget")
    observed_calls = budget.get("observed_agent_calls")
    if not isinstance(observed_calls, int) or isinstance(observed_calls, bool):
        raise GuidanceLoopError("execution has no integer observed-agent call count")
    if observed_calls < 0 or observed_calls > approved_agent_call_budget:
        raise GuidanceLoopError("execution exceeded the approved target-agent call budget")
    if budget.get("user_model_calls") != 0:
        raise GuidanceLoopError("Guidance loop requires the deterministic zero-model user")
    oracle = _mapping(value.get("bound_oracle"), "$.execution.bound_oracle")
    if oracle.get("fixture_instance_id") != expected["fixture_instance_id"]:
        raise GuidanceLoopError("Bound Oracle result does not match the executed fixture")
    if dict(oracle.get("variation_selection") or {}) != expected["variation_selection"]:
        raise GuidanceLoopError("Bound Oracle result does not match the executed variation")
    return value


def run_approved_guidance_batch(
    bound_plan: Mapping[str, Any],
    run_configuration: Mapping[str, Any],
    history: Mapping[str, Any],
    *,
    approved_experiment_count: int,
    approved_agent_call_budget_per_experiment: int,
    model: str,
    seed: int = 0,
    timeout_seconds: float = 300.0,
    executor: GuidanceExecutor = run_tau_online_driver,
    on_step: StepCallback | None = None,
    on_decision: DecisionCallback | None = None,
    on_error: ErrorCallback | None = None,
) -> dict[str, Any]:
    """Execute at most one explicitly approved batch and feed results back.

    `executor` is injectable so the full loop can be tested offline.  Production
    callers use `run_tau_online_driver`, which performs the real target calls.
    """

    approved_count = _positive_int(
        approved_experiment_count, "$.approved_experiment_count"
    )
    per_experiment = _positive_int(
        approved_agent_call_budget_per_experiment,
        "$.approved_agent_call_budget_per_experiment",
    )
    run = validate_run_configuration(run_configuration)
    if per_experiment > run["budgets"]["max_model_calls"]:
        raise GuidanceLoopError(
            "approved per-experiment agent-call budget exceeds RunConfiguration.max_model_calls"
        )
    if not isinstance(model, str) or not model:
        raise GuidanceLoopError("$.model must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise GuidanceLoopError("$.seed must be an integer")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise GuidanceLoopError("$.timeout_seconds must be positive")

    current_history = deepcopy(dict(history))
    steps: list[dict[str, Any]] = []
    observed_agent_calls = 0
    for batch_index in range(approved_count):
        decision = decide_next_experiment(bound_plan, current_history)
        if decision["status"] != "continue":
            break
        coordinate = _experiment_coordinate(decision)
        if on_decision is not None:
            on_decision(batch_index, deepcopy(decision))
        try:
            execution = executor(
                bound_plan,
                run,
                fixture_instance_id=coordinate["fixture_instance_id"],
                variation_selection=coordinate["variation_selection"],
                model=model,
                approved_agent_call_budget=per_experiment,
                seed=seed + batch_index,
                timeout_seconds=float(timeout_seconds),
            )
            validated = _validate_execution(
                execution,
                decision,
                approved_agent_call_budget=per_experiment,
            )
            updated_history = record_guidance_observation(
                bound_plan, current_history, validated["bound_oracle"]
            )
        except Exception as exc:
            if on_error is not None:
                on_error(batch_index, deepcopy(decision), exc)
            raise
        current_history = updated_history
        calls = int(validated["budget"]["observed_agent_calls"])
        observed_agent_calls += calls
        step = {
            "step_id": f"STEP{batch_index + 1:04d}",
            "batch_index": batch_index,
            "decision": deepcopy(decision),
            "execution": validated,
            "oracle_classification": validated["bound_oracle"].get(
                "classification"
            ),
            "approved_agent_call_budget": per_experiment,
            "observed_agent_calls": calls,
            "history_observation_id": current_history["observations"][-1][
                "observation_id"
            ],
        }
        step["step_fingerprint"] = content_sha256(step)
        steps.append(step)
        if on_step is not None:
            on_step(deepcopy(step), deepcopy(current_history))

    next_decision = decide_next_experiment(bound_plan, current_history)
    if next_decision["status"] == "continue" and len(steps) >= approved_count:
        batch_status = "awaiting_approval"
        stop_reason = "approved_experiment_count_consumed"
    elif next_decision["status"] == "continue":
        # This is defensive: the only non-terminal exit from the loop should be
        # allowance exhaustion.
        batch_status = "awaiting_approval"
        stop_reason = "batch_returned_before_next_external_call"
    else:
        batch_status = "guidance_terminal"
        stop_reason = next_decision["reason_code"]

    result = {
        "schema_version": GUIDANCE_BATCH_SCHEMA_VERSION,
        "runner_version": GUIDANCE_BATCH_RUNNER_VERSION,
        "search_id": current_history.get("search_id"),
        "batch_status": batch_status,
        "stop_reason": stop_reason,
        "approval": {
            "approved_experiment_count": approved_count,
            "approved_agent_call_budget_per_experiment": per_experiment,
            "maximum_approved_agent_calls": approved_count * per_experiment,
        },
        "consumption": {
            "executed_experiments": len(steps),
            "observed_agent_calls": observed_agent_calls,
            "remaining_approved_experiments": approved_count - len(steps),
            "external_calls_made": observed_agent_calls,
        },
        "steps": steps,
        "updated_history": current_history,
        "next_guidance_decision": next_decision,
    }
    result["batch_fingerprint"] = content_sha256(result)
    return result


def run_approved_guidance_batch_files(
    *,
    bound_plan_path: str | Path,
    run_configuration_path: str | Path,
    history_path: str | Path,
    output_dir: str | Path,
    approved_experiment_count: int,
    approved_agent_call_budget_per_experiment: int,
    model: str,
    seed: int = 0,
    timeout_seconds: float = 300.0,
    executor: GuidanceExecutor = run_tau_online_driver,
) -> dict[str, Any]:
    """File entry point with a durable checkpoint after every completed step."""

    def load(path: str | Path, label: str) -> dict[str, Any]:
        file = Path(path)
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GuidanceLoopError(f"cannot load {label} {file}: {exc}") from exc
        return deepcopy(dict(_mapping(value, f"$.{label}")))

    bound = load(bound_plan_path, "bound_plan")
    run = load(run_configuration_path, "run_configuration")
    history = load(history_path, "history")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise GuidanceLoopError(
            f"output directory must be absent or empty: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)

    def write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    write_json(root / "history.initial.json", history)

    def checkpoint(step: Mapping[str, Any], updated: Mapping[str, Any]) -> None:
        step_root = root / str(step["step_id"]).lower()
        write_json(step_root / "execution.json", step["execution"])
        write_json(step_root / "oracle.json", step["execution"]["bound_oracle"])
        write_json(step_root / "step.json", step)
        write_json(root / "history.json", updated)

    def checkpoint_decision(
        batch_index: int, decision: Mapping[str, Any]
    ) -> None:
        step_root = root / f"step{batch_index + 1:04d}"
        write_json(step_root / "decision.json", decision)

    def checkpoint_error(
        batch_index: int,
        decision: Mapping[str, Any],
        exc: Exception,
    ) -> None:
        step_root = root / f"step{batch_index + 1:04d}"
        write_json(
            step_root / "error.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "decision_id": decision.get("decision_id"),
                "history_updated": False,
            },
        )

    result = run_approved_guidance_batch(
        bound,
        run,
        history,
        approved_experiment_count=approved_experiment_count,
        approved_agent_call_budget_per_experiment=(
            approved_agent_call_budget_per_experiment
        ),
        model=model,
        seed=seed,
        timeout_seconds=timeout_seconds,
        executor=executor,
        on_step=checkpoint,
        on_decision=checkpoint_decision,
        on_error=checkpoint_error,
    )
    write_json(root / "history.json", result["updated_history"])
    write_json(root / "next_guidance_decision.json", result["next_guidance_decision"])
    # Each step already owns its full execution.  The top-level summary keeps
    # only compact step evidence to avoid duplicating potentially large traces.
    summary = deepcopy(result)
    summary["steps"] = [
        {key: value for key, value in step.items() if key != "execution"}
        for step in result["steps"]
    ]
    summary.pop("updated_history", None)
    write_json(root / "batch_summary.json", summary)
    result["output_dir"] = str(root)
    return result
