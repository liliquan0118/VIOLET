"""Driver binding contracts and environment adapters."""

from .tau_airline import (
    DriverBindingError,
    bind_tau_airline_driver,
    bind_tau_airline_driver_file,
)
from .registry import bind_registered_driver_file
from .runtime import (
    RuntimeDriverError,
    apply_next_driver_action,
    ingest_runtime_observation,
    initialize_runtime_driver,
    reachability_result,
    realize_driver_action,
    replay_runtime_trace,
    replay_runtime_trace_file,
)
from .tau_online import (
    TauMessageNormalizer,
    TauOnlineAdapterError,
    TauRuntimeDriverUser,
    TauRuntimeEventAdapter,
    create_instrumented_tau_orchestrator,
    materialize_tau_runtime_task_value,
    run_tau_online_driver,
    run_tau_online_driver_file,
)
from .generic_tau_airline_v1 import (
    GenericDriverBindingError,
    bind_generic_tau_airline_plans,
    bind_generic_tau_airline_plans_file,
    validate_generic_bound_driver_plan_set,
)
from .generic_tau_online_v1 import (
    GenericTauOnlineError,
    GenericDeterministicTauUser,
    evaluate_generic_mechanical_oracle,
    materialize_generic_tau_task,
    run_generic_tau_online_plan,
)

__all__ = [
    "DriverBindingError",
    "bind_tau_airline_driver",
    "bind_tau_airline_driver_file",
    "bind_registered_driver_file",
    "RuntimeDriverError",
    "initialize_runtime_driver",
    "realize_driver_action",
    "apply_next_driver_action",
    "ingest_runtime_observation",
    "reachability_result",
    "replay_runtime_trace",
    "replay_runtime_trace_file",
    "TauOnlineAdapterError",
    "TauMessageNormalizer",
    "TauRuntimeEventAdapter",
    "TauRuntimeDriverUser",
    "materialize_tau_runtime_task_value",
    "create_instrumented_tau_orchestrator",
    "run_tau_online_driver",
    "run_tau_online_driver_file",
    "GenericDriverBindingError",
    "bind_generic_tau_airline_plans",
    "bind_generic_tau_airline_plans_file",
    "validate_generic_bound_driver_plan_set",
    "GenericTauOnlineError",
    "GenericDeterministicTauUser",
    "evaluate_generic_mechanical_oracle",
    "materialize_generic_tau_task",
    "run_generic_tau_online_plan",
]
