"""Dispatch fixture binding through the compiled capability contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tau_airline import DriverBindingError, bind_tau_airline_driver_file


_FILE_BINDERS = {
    "binder.tau-airline.cancel-duration/v0.1": bind_tau_airline_driver_file,
    "binder.tau-airline.cancel-focus-control/v0.1": bind_tau_airline_driver_file,
}


def bind_registered_driver_file(
    *,
    compiled_plan_path: str | Path,
    run_configuration_path: str | Path,
    output_path: str | Path,
    database_path: str | Path | None = None,
    reference_time: str | None = None,
) -> dict[str, Any]:
    plan_path = Path(compiled_plan_path)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriverBindingError(f"cannot load compiled plan {plan_path}: {exc}") from exc
    capability = (
        (plan.get("capability_contract") or {})
        .get("stage_bindings", {})
        .get("fixture_binder")
    )
    if not isinstance(capability, str) or not capability:
        raise DriverBindingError(
            "compiled plan has no registered fixture-binder capability"
        )
    binder = _FILE_BINDERS.get(capability)
    if binder is None:
        raise DriverBindingError(
            f"fixture-binder capability is not installed: {capability!r}"
        )
    return binder(
        compiled_plan_path=compiled_plan_path,
        run_configuration_path=run_configuration_path,
        output_path=output_path,
        database_path=database_path,
        reference_time=reference_time,
    )


__all__ = ["bind_registered_driver_file"]
