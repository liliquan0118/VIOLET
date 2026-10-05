"""Online tau-bench adapter for the deterministic Runtime Driver.

The module keeps tau2 imports lazy so compilation, replay, and unit tests do not
require the tau-bench environment.  The adapter has no user-model dependency:
only the target agent may make model calls during an explicitly approved run.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..compiler.profiles import validate_run_configuration
from .runtime import (
    RuntimeDriverError,
    apply_next_driver_action,
    ingest_runtime_observation,
    initialize_runtime_driver,
    reachability_result,
)


TAU_ONLINE_ADAPTER_VERSION = "tau-online-runtime-adapter/v0.1"
TAU_ONLINE_EXECUTION_SCHEMA_VERSION = "agentspectesting.tau-online-execution/v0.1"


class TauOnlineAdapterError(ValueError):
    """Raised for uncorrelated tau messages or invalid online-driver state."""


def _is_text_tool_call_message(message: Any) -> bool:
    """Return whether a tau message contains a tool preamble and tool calls.

    OpenAI-compatible endpoints may legally return a short assistant preamble
    together with tool calls.  tau2's half-duplex protocol rejects that shape
    by default even though routing it to the environment is unambiguous.
    """

    try:
        return bool(message.is_tool_call() and message.has_text_content())
    except AttributeError:
        value = _message_dict(message)
        content = value.get("content")
        return bool(
            value.get("tool_calls")
            and isinstance(content, str)
            and content.strip()
        )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TauOnlineAdapterError(f"{path} must be an object")
    return value


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return deepcopy(dict(message))
    try:
        value = message.model_dump(exclude_none=True, mode="json")
    except Exception as exc:
        raise TauOnlineAdapterError(
            f"cannot normalize tau message {type(message).__name__}: {exc}"
        ) from exc
    return deepcopy(dict(_mapping(value, "$.tau_message")))


def _tool_call(raw: Mapping[str, Any]) -> dict[str, Any]:
    function = raw.get("function")
    nested = function if isinstance(function, Mapping) else {}
    name = raw.get("name") or nested.get("name")
    arguments = raw.get("arguments")
    if arguments is None:
        arguments = nested.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise TauOnlineAdapterError(
                f"tau tool call arguments are not valid JSON: {exc}"
            ) from exc
    if not isinstance(name, str) or not name:
        raise TauOnlineAdapterError("tau tool call has no name")
    if not isinstance(arguments, Mapping):
        raise TauOnlineAdapterError(f"tau tool call {name!r} has non-object arguments")
    identifier = raw.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise TauOnlineAdapterError(f"tau tool call {name!r} has no call id")
    return {
        "tool_call_id": identifier,
        "tool_name": name,
        "arguments": deepcopy(dict(arguments)),
        "requestor": raw.get("requestor", "assistant"),
    }


class TauMessageNormalizer:
    """Convert tau2 messages to RuntimeObservation events with exact correlation."""

    def __init__(self) -> None:
        self.pending_tool_calls: dict[str, dict[str, Any]] = {}

    def normalize(self, message: Any) -> list[dict[str, Any]]:
        value = _message_dict(message)
        role = value.get("role")
        if role == "assistant":
            return self._assistant(value)
        if role == "tool":
            if isinstance(value.get("tool_messages"), list):
                events: list[dict[str, Any]] = []
                for raw in value["tool_messages"]:
                    events.extend(self.normalize(raw))
                return events
            return self._tool_result(value)
        if role in {"user", "system"}:
            return []
        raise TauOnlineAdapterError(f"unsupported tau message role: {role!r}")

    def _assistant(self, message: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw_calls = message.get("tool_calls") or []
        content = message.get("content")
        events: list[dict[str, Any]] = []
        if isinstance(content, str) and content.strip():
            events.append(
                {"event_kind": "assistant_message", "content": content}
            )
        for raw in raw_calls:
            call = _tool_call(_mapping(raw, "$.tau_message.tool_calls[]"))
            if call["requestor"] != "assistant":
                continue
            identifier = call["tool_call_id"]
            if identifier in self.pending_tool_calls:
                raise TauOnlineAdapterError(
                    f"duplicate unresolved tau tool call id: {identifier!r}"
                )
            self.pending_tool_calls[identifier] = call
            events.append(
                {
                    "event_kind": "assistant_tool_call",
                    "tool_call_id": identifier,
                    "tool_name": call["tool_name"],
                    "arguments": deepcopy(call["arguments"]),
                }
            )
        if raw_calls:
            return events
        if not isinstance(content, str) or not content.strip():
            raise TauOnlineAdapterError("tau assistant text message is empty")
        events.append({"event_kind": "assistant_turn_completed"})
        return events

    def _tool_result(self, message: Mapping[str, Any]) -> list[dict[str, Any]]:
        if message.get("requestor", "assistant") != "assistant":
            return []
        identifier = message.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise TauOnlineAdapterError("tau tool result has no call id")
        try:
            call = self.pending_tool_calls.pop(identifier)
        except KeyError as exc:
            raise TauOnlineAdapterError(
                f"tau tool result {identifier!r} has no pending assistant call"
            ) from exc
        return [
            {
                "event_kind": "tool_result",
                "tool_call_id": identifier,
                "tool_name": call["tool_name"],
                "arguments": deepcopy(call["arguments"]),
                "success": message.get("error") is not True,
                "content": message.get("content"),
            }
        ]

    def assert_closed(self) -> None:
        if self.pending_tool_calls:
            raise TauOnlineAdapterError(
                "online trace ended with unresolved tool calls: "
                + ", ".join(sorted(self.pending_tool_calls))
            )


class TauRuntimeEventAdapter:
    """Own the runtime session while tau-bench is running."""

    def __init__(self, session: Mapping[str, Any]) -> None:
        self.session = deepcopy(dict(session))
        self.normalizer = TauMessageNormalizer()
        self.event_log: list[dict[str, Any]] = []
        self.transport_action_log: list[dict[str, Any]] = []
        self.recovery_action_count = 0

    @property
    def terminal(self) -> bool:
        return self.session["state"]["status"] == "terminal"

    @property
    def stop_reason(self) -> str | None:
        return self.session["state"]["stop_reason"]

    def observe_tau_message(self, message: Any) -> list[dict[str, Any]]:
        events = self.normalizer.normalize(message)
        for event in events:
            record = {
                "event_index": len(self.event_log),
                "event": deepcopy(event),
                "runtime_applied": False,
                "runtime_skip_reason": None,
            }
            if self.terminal:
                record["runtime_skip_reason"] = "runtime_already_terminal"
            else:
                self.session = ingest_runtime_observation(self.session, event)
                record["runtime_applied"] = True
            self.event_log.append(record)
        return events

    def next_user_content(self) -> str:
        """Return one tau-compatible deterministic user message."""

        if self.terminal or self.session["next_action"]["action_kind"] == "stop":
            return "###STOP###"
        action_kind = self.session["next_action"]["action_kind"]
        if action_kind == "wait_for_agent":
            maximum = self.session["runtime_context"]["max_recovery_actions"]
            if self.recovery_action_count >= maximum:
                self.session = ingest_runtime_observation(
                    self.session,
                    {
                        "event_kind": "terminal_state",
                        "reason": "recovery_budget_exceeded",
                    },
                )
                return "###STOP###"
            content = "Please continue checking the cancellation request."
            self.transport_action_log.append(
                {
                    "transport_action_index": len(self.transport_action_log),
                    "action_kind": "prompt_agent_to_continue",
                    "content": content,
                }
            )
            self.recovery_action_count += 1
            return content
        self.session = apply_next_driver_action(self.session)
        realized = self.session["state"]["action_log"][-1]["realized"]
        content = realized["message"]["content"]
        self.transport_action_log.append(
            {
                "transport_action_index": len(self.transport_action_log),
                "action_kind": action_kind,
                "content": content,
            }
        )
        return content

    def terminate_if_active(self, reason: str) -> None:
        if not self.terminal:
            self.session = ingest_runtime_observation(
                self.session,
                {"event_kind": "terminal_state", "reason": reason},
            )

    def result(self) -> dict[str, Any]:
        return {
            "adapter_version": TAU_ONLINE_ADAPTER_VERSION,
            "runtime_session": deepcopy(self.session),
            "reachability_result": reachability_result(self.session),
            "normalized_event_log": deepcopy(self.event_log),
            "transport_action_log": deepcopy(self.transport_action_log),
            "normalizer_open_tool_call_ids": sorted(
                self.normalizer.pending_tool_calls
            ),
            "checks": {
                "user_model_calls": 0,
                "tool_results_correlated_by_call_id": True,
            },
        }


class TauRuntimeDriverUser:
    """A tau half-duplex user backed only by Runtime Driver templates."""

    def __init__(self, adapter: TauRuntimeEventAdapter) -> None:
        self.adapter = adapter

    def get_init_state(self, message_history: list[Any] | None = None) -> dict[str, Any]:
        return {"messages": list(message_history or [])}

    def generate_next_message(
        self, message: Any, state: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        from tau2.data_model.message import UserMessage

        content = self.adapter.next_user_content()
        user_message = UserMessage(role="user", content=content, cost=0.0)
        state["messages"].append(user_message)
        return user_message, state

    def set_seed(self, seed: int) -> None:
        return None

    def stop(self, message: Any = None, state: Any = None) -> None:
        return None


def materialize_tau_runtime_task_value(
    bound_plan: Mapping[str, Any], fixture_instance_id: str
) -> dict[str, Any]:
    """Build the minimal tau Task transport object from one bound fixture."""

    fixtures = [
        item
        for item in bound_plan.get("fixture_instances", [])
        if item.get("fixture_instance_id") == fixture_instance_id
    ]
    if len(fixtures) != 1:
        raise TauOnlineAdapterError(
            f"fixture_instance_id must resolve exactly once: {fixture_instance_id!r}"
        )
    fixture = fixtures[0]
    objects = fixture["object_bindings"]
    patch = fixture["initial_state_patch"].get("agent_data")
    return {
        "id": f"online::{fixture_instance_id}",
        "description": {
            "purpose": "Drive one bound AgentSpecTesting fixture to its focal decision point.",
            "relevant_policies": None,
            "notes": "User messages are generated by deterministic-runtime-driver/v0.1.",
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "domain": "airline",
                "reason_for_call": "Runtime Driver controlled cancellation test.",
                "known_info": (
                    f"user_id: {objects['user_id']}; "
                    f"reservation_id: {objects['reservation_id']}"
                ),
                "unknown_info": None,
                "task_instructions": (
                    "Transport metadata only. The deterministic Runtime Driver, not a user "
                    "simulator, controls every user message."
                ),
            },
        },
        "initial_state": {
            "initialization_data": (
                {"agent_data": deepcopy(patch)} if patch is not None else None
            ),
            "initialization_actions": None,
            "message_history": None,
        },
        "evaluation_criteria": {
            "actions": [],
            "communicate_info": [],
            "nl_assertions": [],
            "reward_basis": [],
        },
        "annotations": None,
        "coverage_metadata": {
            "bound_driver_plan_id": bound_plan.get("bound_driver_plan_id"),
            "bound_driver_plan_fingerprint": bound_plan.get(
                "bound_driver_plan_fingerprint"
            ),
            "fixture_instance_id": fixture_instance_id,
            "coverage_cell_id": fixture.get("coverage_cell_id"),
            "expected_operation_decision": fixture.get(
                "expected_operation_decision"
            ),
        },
    }


def _message_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    return [_message_dict(message) for message in messages]


def _capture_bound_reservation_state(
    environment: Any, fixture: Mapping[str, Any]
) -> dict[str, Any]:
    reservation_id = fixture["object_bindings"]["reservation_id"]
    reservation = environment.tools.get_reservation_details(reservation_id)
    try:
        value = reservation.model_dump(mode="json")
    except Exception:
        value = dict(reservation)
    flights = []
    for flight in value.get("flights") or []:
        status = environment.tools.get_flight_status(
            flight["flight_number"], flight["date"]
        )
        flights.append(
            {
                "flight_number": flight["flight_number"],
                "date": flight["date"],
                "status": status,
            }
        )
    return {
        "reservation_id": reservation_id,
        "user_id": value.get("user_id"),
        "status": value.get("status"),
        "created_at": value.get("created_at"),
        "cabin": value.get("cabin"),
        "insurance": value.get("insurance"),
        "flights": flights,
        "db_hash": environment.get_db_hash(),
    }


def create_instrumented_tau_orchestrator(
    *,
    domain: str,
    agent: Any,
    user: TauRuntimeDriverUser,
    environment: Any,
    task: Any,
    fixture: Mapping[str, Any],
    adapter: TauRuntimeEventAdapter,
    approved_agent_call_budget: int,
    max_steps: int,
    seed: int,
    timeout_seconds: float,
) -> Any:
    """Create an Orchestrator subclass that streams trajectory deltas to Runtime Driver."""

    from tau2.data_model.simulation import TerminationReason
    from tau2.orchestrator.orchestrator import Orchestrator, Role

    if approved_agent_call_budget < 1:
        raise TauOnlineAdapterError("approved_agent_call_budget must be positive")

    class InstrumentedRuntimeOrchestrator(Orchestrator):
        def __init__(self) -> None:
            super().__init__(
                domain=domain,
                agent=agent,
                user=user,
                environment=environment,
                task=task,
                max_steps=max_steps,
                seed=seed,
                timeout=timeout_seconds,
                validate_communication=True,
            )
            self.runtime_adapter = adapter
            self.runtime_cursor = 0
            self.observed_agent_calls = 0
            self.runtime_stop_after_environment = False
            self.runtime_pre_state = None

        def _check_communication_error(self) -> None:
            # The target endpoint can emit a natural-language preamble and a
            # tool call in one valid OpenAI-compatible assistant message.  The
            # Runtime Driver normalizes both parts, while tau2 can safely route
            # the tool calls to ENV.  Preserve all other tau2 communication
            # validation, including empty-message and user-protocol checks.
            if self.from_role == Role.AGENT and _is_text_tool_call_message(
                self.message
            ):
                return
            super()._check_communication_error()

        def _observe_new_messages(self) -> None:
            new_messages = self.trajectory[self.runtime_cursor :]
            self.runtime_cursor = len(self.trajectory)
            for message in new_messages:
                value = _message_dict(message)
                if value.get("role") == "assistant" and value.get("cost") is not None:
                    if value.get("cost") != 0.0 or value.get("usage") is not None:
                        self.observed_agent_calls += 1
                events = self.runtime_adapter.observe_tau_message(message)
                if any(event["event_kind"] == "assistant_tool_call" for event in events):
                    if self.runtime_adapter.terminal and self.to_role == Role.ENV:
                        self.runtime_stop_after_environment = True

        def _apply_runtime_stop(self) -> None:
            if self.runtime_stop_after_environment:
                if self.from_role == Role.ENV:
                    self.done = True
                    self.termination_reason = TerminationReason.USER_STOP
                    self.runtime_stop_after_environment = False
                return
            if self.runtime_adapter.terminal:
                self.done = True
                self.termination_reason = TerminationReason.USER_STOP

        def _apply_agent_call_budget(self) -> None:
            if self.observed_agent_calls < approved_agent_call_budget:
                return
            if self.to_role == Role.ENV:
                self.runtime_adapter.terminate_if_active(
                    "approved_agent_call_budget_exhausted"
                )
                self.runtime_stop_after_environment = True
            else:
                self.runtime_adapter.terminate_if_active(
                    "approved_agent_call_budget_exhausted"
                )
                self.done = True
                self.termination_reason = TerminationReason.MAX_STEPS

        def initialize(self) -> None:
            super().initialize()
            self.runtime_pre_state = _capture_bound_reservation_state(
                environment, fixture
            )
            self._observe_new_messages()
            self._apply_runtime_stop()

        def step(self) -> None:
            super().step()
            self._observe_new_messages()
            self._apply_agent_call_budget()
            self._apply_runtime_stop()

    return InstrumentedRuntimeOrchestrator()


def run_tau_online_driver(
    bound_plan: Mapping[str, Any],
    run_configuration: Mapping[str, Any],
    *,
    fixture_instance_id: str,
    variation_selection: Mapping[str, Any] | None,
    model: str,
    approved_agent_call_budget: int,
    seed: int = 0,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Execute one bound fixture with a deterministic user and a target LLM agent."""

    run = validate_run_configuration(run_configuration)
    if approved_agent_call_budget < 1:
        raise TauOnlineAdapterError("approved agent-call budget must be positive")
    if approved_agent_call_budget > run["budgets"]["max_model_calls"]:
        raise TauOnlineAdapterError(
            "approved agent-call budget exceeds RunConfiguration.max_model_calls"
        )
    session = initialize_runtime_driver(
        bound_plan,
        run,
        fixture_instance_id=fixture_instance_id,
        variation_selection=variation_selection,
    )
    from tau2.agent.llm_agent import create_llm_agent
    from tau2.data_model.tasks import Task
    from tau2.domains.airline.environment import get_environment

    adapter = TauRuntimeEventAdapter(session)
    user = TauRuntimeDriverUser(adapter)
    environment = get_environment()
    task_value = materialize_tau_runtime_task_value(bound_plan, fixture_instance_id)
    task = Task.model_validate(task_value)
    fixture = next(
        item
        for item in bound_plan["fixture_instances"]
        if item["fixture_instance_id"] == fixture_instance_id
    )
    litellm_model = model if "/" in model else f"openai/{model}"
    agent = create_llm_agent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=litellm_model,
        llm_args={"num_retries": 0},
    )
    max_steps = max(4, approved_agent_call_budget * 4 + 4)
    orchestrator = create_instrumented_tau_orchestrator(
        domain="airline",
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        fixture=fixture,
        adapter=adapter,
        approved_agent_call_budget=approved_agent_call_budget,
        max_steps=max_steps,
        seed=seed,
        timeout_seconds=timeout_seconds,
    )
    started = time.monotonic()
    simulation = orchestrator.run()
    duration = time.monotonic() - started
    if orchestrator.observed_agent_calls > approved_agent_call_budget:
        raise RuntimeError("approved target-agent call budget was exceeded")
    online = adapter.result()
    execution = {
        "schema_version": TAU_ONLINE_EXECUTION_SCHEMA_VERSION,
        "adapter_version": TAU_ONLINE_ADAPTER_VERSION,
        "bound_driver_plan_identity": deepcopy(
            session["bound_driver_plan_identity"]
        ),
        "fixture_instance_id": fixture_instance_id,
        "variation_selection": deepcopy(dict(variation_selection or {})),
        "model": model,
        "seed": seed,
        "budget": {
            "approved_agent_call_budget": approved_agent_call_budget,
            "observed_agent_calls": orchestrator.observed_agent_calls,
            "user_model_calls": 0,
            "orchestrator_steps": orchestrator.step_count,
            "request_retries": 0,
        },
        "duration_seconds": duration,
        "tau_termination_reason": simulation.termination_reason,
        "runtime_stop_reason": online["reachability_result"]["stop_reason"],
        "runtime_driver": online,
        "pre_state": deepcopy(orchestrator.runtime_pre_state),
        "terminal_state": _capture_bound_reservation_state(environment, fixture),
        "messages": _message_dicts(simulation.messages),
        "materialized_tau_task": task_value,
    }
    from ..oracle import evaluate_bound_online_execution

    execution["bound_oracle"] = evaluate_bound_online_execution(
        bound_plan, execution
    )
    return execution


def run_tau_online_driver_file(
    *,
    bound_plan_path: str | Path,
    run_configuration_path: str | Path,
    fixture_instance_id: str,
    variation_selection: Mapping[str, Any] | None,
    model: str,
    approved_agent_call_budget: int,
    seed: int,
    timeout_seconds: float,
    output_path: str | Path,
) -> dict[str, Any]:
    def load(path: str | Path, label: str) -> dict[str, Any]:
        file = Path(path)
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TauOnlineAdapterError(f"cannot load {label} {file}: {exc}") from exc
        return deepcopy(dict(_mapping(value, f"$.{label}")))

    execution = run_tau_online_driver(
        load(bound_plan_path, "bound_plan"),
        load(run_configuration_path, "run_configuration"),
        fixture_instance_id=fixture_instance_id,
        variation_selection=variation_selection,
        model=model,
        approved_agent_call_budget=approved_agent_call_budget,
        seed=seed,
        timeout_seconds=timeout_seconds,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(execution, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return execution
