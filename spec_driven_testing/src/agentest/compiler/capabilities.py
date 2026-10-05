"""Versioned capability registry for the profile-driven test pipeline.

The original vertical slice selected its implementations implicitly: a
duration comparison eventually reached the tau-airline cancellation binder,
runtime and oracle.  This module makes those choices explicit without changing
the behaviour of that slice.  New spec families must register a capability
instead of adding case-specific branches to the shared compiler.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


CAPABILITY_REGISTRY_VERSION = "agentspectesting.capability-registry/v0.1"


class CapabilityError(ValueError):
    """Raised when capability selection is invalid or ambiguous."""


@dataclass(frozen=True)
class PipelineCapability:
    capability_id: str
    stage: str
    target_archetypes: tuple[str, ...] = ()
    sut: str | None = None
    domain: str | None = None
    sut_profile_ids: tuple[str, ...] = ()
    verified_tool_names: tuple[str, ...] = ()
    probe_families: tuple[str, ...] = ()
    implementation: str = ""

    def matches(self, context: Mapping[str, Any]) -> bool:
        archetype = context.get("target_archetype")
        if self.target_archetypes and archetype not in self.target_archetypes:
            return False
        if self.sut is not None and context.get("sut") != self.sut:
            return False
        if self.domain is not None and context.get("domain") != self.domain:
            return False
        if self.sut_profile_ids and context.get("sut_profile_id") not in self.sut_profile_ids:
            return False
        if self.verified_tool_names:
            actual = tuple(sorted(str(value) for value in context.get("verified_tool_names") or []))
            if actual != tuple(sorted(self.verified_tool_names)):
                return False
        if self.probe_families:
            if context.get("probe_family") not in self.probe_families:
                return False
        return True

    def specificity(self) -> int:
        return sum(
            (
                bool(self.target_archetypes),
                self.sut is not None,
                self.domain is not None,
                bool(self.sut_profile_ids),
                bool(self.verified_tool_names),
                bool(self.probe_families),
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "stage": self.stage,
            "target_archetypes": list(self.target_archetypes),
            "sut": self.sut,
            "domain": self.domain,
            "sut_profile_ids": list(self.sut_profile_ids),
            "verified_tool_names": list(self.verified_tool_names),
            "probe_families": list(self.probe_families),
            "implementation": self.implementation,
        }


class CapabilityRegistry:
    """Deterministically select one registered implementation per stage."""

    def __init__(self, capabilities: list[PipelineCapability]) -> None:
        identifiers = [item.capability_id for item in capabilities]
        if len(set(identifiers)) != len(identifiers):
            raise CapabilityError("capability IDs must be unique")
        self._capabilities = tuple(capabilities)

    def select(
        self, stage: str, context: Mapping[str, Any]
    ) -> PipelineCapability | None:
        matches = [
            item
            for item in self._capabilities
            if item.stage == stage and item.matches(context)
        ]
        if not matches:
            return None
        highest = max(item.specificity() for item in matches)
        selected = [item for item in matches if item.specificity() == highest]
        if len(selected) != 1:
            raise CapabilityError(
                f"capability selection for stage {stage!r} is ambiguous: "
                + ", ".join(sorted(item.capability_id for item in selected))
            )
        return selected[0]

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_REGISTRY_VERSION,
            "capabilities": [
                deepcopy(item.as_dict())
                for item in sorted(
                    self._capabilities, key=lambda value: (value.stage, value.capability_id)
                )
            ],
        }


DEFAULT_PIPELINE_CAPABILITIES = CapabilityRegistry(
    [
        PipelineCapability(
            capability_id="planner.operation-decision.isolated-atomic/v0.1",
            stage="experiment_planner",
            target_archetypes=("operation_decision",),
            implementation="agentest.compiler.experiment_planner.plan_experiment",
        ),
        PipelineCapability(
            capability_id="probe.duration-comparison-boundary/v0.1",
            stage="probe_synthesizer",
            target_archetypes=("operation_decision",),
            probe_families=("duration_boundary",),
            implementation=(
                "agentest.compiler.experiment_planner."
                "synthesize_boundary_refinements"
            ),
        ),
        PipelineCapability(
            capability_id="probe.semantic-boolean-contrast/v0.1",
            stage="probe_synthesizer",
            target_archetypes=("operation_decision",),
            probe_families=("semantic_boolean_contrast",),
            implementation=(
                "agentest.compiler.experiment_planner."
                "synthesize_probe_contract"
            ),
        ),
        PipelineCapability(
            capability_id="probe.enum-equality-contrast/v0.1",
            stage="probe_synthesizer",
            target_archetypes=("operation_decision",),
            probe_families=("enum_equality_contrast",),
            implementation=(
                "agentest.compiler.experiment_planner."
                "synthesize_probe_contract"
            ),
        ),
        PipelineCapability(
            capability_id="binder.tau-airline.cancel-duration/v0.1",
            stage="fixture_binder",
            target_archetypes=("operation_decision",),
            sut="tau_bench",
            domain="airline",
            verified_tool_names=("cancel_reservation",),
            probe_families=("duration_boundary",),
            implementation="agentest.driver.tau_airline.bind_tau_airline_driver",
        ),
        PipelineCapability(
            capability_id="binder.tau-airline.cancel-focus-control/v0.1",
            stage="fixture_binder",
            target_archetypes=("operation_decision",),
            sut="tau_bench",
            domain="airline",
            sut_profile_ids=("tau-bench-airline/v0.2",),
            verified_tool_names=("cancel_reservation",),
            probe_families=(
                "semantic_boolean_contrast",
                "enum_equality_contrast",
            ),
            implementation="agentest.driver.tau_airline.bind_tau_airline_driver",
        ),
        PipelineCapability(
            capability_id="runtime.tau-airline.cancel/v0.1",
            stage="runtime_protocol",
            target_archetypes=("operation_decision",),
            sut="tau_bench",
            domain="airline",
            verified_tool_names=("cancel_reservation",),
            implementation="agentest.driver.runtime",
        ),
        PipelineCapability(
            capability_id="surface.tau-airline.cancel/v0.1",
            stage="surface_realizer",
            target_archetypes=("operation_decision",),
            sut="tau_bench",
            domain="airline",
            verified_tool_names=("cancel_reservation",),
            implementation="agentest.driver.runtime.realize_driver_action",
        ),
        PipelineCapability(
            capability_id="oracle.tau-airline.cancel/v0.1",
            stage="outcome_oracle",
            target_archetypes=("operation_decision",),
            sut="tau_bench",
            domain="airline",
            verified_tool_names=("cancel_reservation",),
            implementation="agentest.oracle.bound.evaluate_bound_online_execution",
        ),
        PipelineCapability(
            capability_id="mutation.operation-decision.structural/v0.1",
            stage="mutation_provider",
            target_archetypes=("operation_decision",),
            implementation="agentest.compiler.experiment_planner._derive_mutation_catalog",
        ),
        PipelineCapability(
            capability_id="guidance.deterministic-feedback/v0.1",
            stage="guidance_controller",
            target_archetypes=("operation_decision",),
            implementation="agentest.guidance.controller.decide_next_experiment",
        ),
    ]
)


__all__ = [
    "CAPABILITY_REGISTRY_VERSION",
    "CapabilityError",
    "CapabilityRegistry",
    "DEFAULT_PIPELINE_CAPABILITIES",
    "PipelineCapability",
]
