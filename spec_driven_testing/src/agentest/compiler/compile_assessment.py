"""Non-throwing, layer-by-layer capability assessment for one selected spec."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .artifacts import content_sha256, resolve_artifact_bundle_file
from .capabilities import DEFAULT_PIPELINE_CAPABILITIES, CapabilityRegistry
from .evaluation_contracts import make_compile_assessment
from .evaluation_resolver import resolve_evaluation_target
from .target_resolution import make_target_resolution_report
from .policy_lineage import resolve_policy_lineage
from .runtime_projection import (
    materialize_runtime_projection_target,
    resolve_runtime_projection,
)
from .profiles import load_configuration_file, validate_compiler_profile, validate_sut_adapter_profile
from .source_loader import list_selected_spec_refs, load_source_spec_record


COMPILE_ASSESSMENT_SET_SCHEMA_VERSION = (
    "agentspectesting.compile-assessment-set/v0.2"
)


def _probe_family(binding: Mapping[str, Any]) -> str:
    assertion = binding.get("assertion_contract") or {}
    expression = assertion.get("typed_expression") or {}
    right = expression.get("right") if isinstance(expression, Mapping) else None
    if expression.get("kind") == "semantic_evidence_predicate":
        return "semantic_boolean_contrast"
    if (
        isinstance(expression, Mapping)
        and expression.get("kind") == "comparison"
        and isinstance(right, Mapping)
        and right.get("value_type") == "duration"
    ):
        return "duration_boundary"
    if (
        isinstance(expression, Mapping)
        and expression.get("kind") == "comparison"
        and expression.get("operator") in {"==", "!="}
        and isinstance(right, Mapping)
        and right.get("value_type") == "enum"
    ):
        return "enum_equality_contrast"
    return "nominal"


def _stage(
    status: str,
    *,
    reason_code: str,
    capability_id: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason_code,
        "capability_id": capability_id,
    }


def assess_selected_spec(
    request: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
    *,
    source_base_dir: str | Path | None,
    compiler_profile: Mapping[str, Any],
    sut_adapter_profile: Mapping[str, Any],
    capability_registry: CapabilityRegistry = DEFAULT_PIPELINE_CAPABILITIES,
) -> dict[str, Any]:
    """Assess every pipeline layer without pretending unsupported work compiled."""

    validate_compiler_profile(compiler_profile)
    sut = validate_sut_adapter_profile(sut_adapter_profile)
    selected = request.get("selected_spec_ref")
    source = load_source_spec_record(selected, base_dir=source_base_dir)
    target = resolve_evaluation_target(source, resolved_artifacts)
    policy_lineage = resolve_policy_lineage(source, resolved_artifacts)
    runtime_projection = resolve_runtime_projection(
        source, policy_lineage, resolved_artifacts
    )
    target = materialize_runtime_projection_target(
        source, target, runtime_projection, resolved_artifacts
    )
    resolution_report = make_target_resolution_report(
        source_spec_record=source,
        resolved_evaluation_target=target,
        policy_lineage_report=policy_lineage,
        runtime_projection_report=runtime_projection,
    )
    binding = target.get("evaluation_binding") or {}
    archetype = binding.get("target_archetype")
    subject = binding.get("evaluation_subject") or {}
    tools = subject.get("verified_tool_names") or []
    probe_family = _probe_family(binding) if binding else None
    context = {
        "target_archetype": archetype,
        "sut": sut["sut"],
        "domain": sut["domain"],
        "sut_profile_id": sut["profile_id"],
        "verified_tool_names": tools,
        "probe_family": probe_family,
    }

    stages: dict[str, dict[str, Any]] = {
        "source_loading": _stage("ready", reason_code="selected_source_loaded"),
    }
    if resolution_report["readiness"]["runtime_target_ready"]:
        stages["semantic_resolution"] = _stage(
            "ready", reason_code="runtime_evaluation_target_ready"
        )
    elif resolution_report["readiness"]["target_ready"]:
        stages["semantic_resolution"] = _stage(
            "partial", reason_code="abstract_target_ready_runtime_binding_incomplete"
        )
    else:
        stages["semantic_resolution"] = _stage(
            "blocked", reason_code=target["resolution_status"]
        )

    downstream = (
        resolution_report["readiness"]["runtime_target_ready"]
        and isinstance(archetype, str)
    )
    stage_specs = (
        ("experiment_planning", "experiment_planner"),
        ("probe_synthesis", "probe_synthesizer"),
        ("fixture_binding", "fixture_binder"),
        ("runtime_protocol", "runtime_protocol"),
        ("surface_realization", "surface_realizer"),
        ("outcome_oracle", "outcome_oracle"),
        ("mutation_provider", "mutation_provider"),
        ("guidance", "guidance_controller"),
    )
    for output_name, registry_stage in stage_specs:
        if not downstream:
            stages[output_name] = _stage(
                "not_assessed", reason_code="semantic_resolution_not_ready"
            )
            continue
        capability = capability_registry.select(registry_stage, context)
        if capability is None:
            stages[output_name] = _stage(
                "blocked",
                reason_code=f"missing_{registry_stage}_capability",
            )
        else:
            stages[output_name] = _stage(
                "ready",
                reason_code="registered_capability_selected",
                capability_id=capability.capability_id,
            )

    snapshot = capability_registry.snapshot()
    registry_identity = {
        "schema_version": snapshot["schema_version"],
        "fingerprint": content_sha256(snapshot),
        "capability_count": len(snapshot["capabilities"]),
    }
    return make_compile_assessment(
        source_spec_record=source,
        resolved_evaluation_target=target,
        target_resolution_report=resolution_report,
        stages=stages,
        capability_registry=registry_identity,
        diagnostics=target["diagnostics"],
    )


def assess_request_file(
    *,
    request_path: str | Path,
    artifact_bundle_path: str | Path,
    source_base_dir: str | Path | None,
    compiler_profile_path: str | Path,
    sut_adapter_profile_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    request_file = Path(request_path)
    request = json.loads(request_file.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("compiler request must contain one JSON object")
    result = assess_selected_spec(
        request,
        resolve_artifact_bundle_file(artifact_bundle_path),
        source_base_dir=source_base_dir,
        compiler_profile=load_configuration_file(
            compiler_profile_path, kind="compiler_profile"
        ),
        sut_adapter_profile=load_configuration_file(
            sut_adapter_profile_path, kind="sut_adapter_profile"
        ),
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def assess_spec_set(
    *,
    workbook: str | Path,
    sheet: str,
    review_ok: int | None,
    resolved_artifacts: Mapping[str, Any],
    source_base_dir: str | Path | None,
    compiler_profile: Mapping[str, Any],
    sut_adapter_profile: Mapping[str, Any],
    capability_registry: CapabilityRegistry = DEFAULT_PIPELINE_CAPABILITIES,
) -> dict[str, Any]:
    """Assess every selected workbook branch with the same frozen inputs."""

    refs = list_selected_spec_refs(
        workbook=workbook,
        sheet=sheet,
        base_dir=source_base_dir,
        review_ok=review_ok,
    )
    items = []
    for ref in refs:
        assessment = assess_selected_spec(
            {
                "schema_version": "agentspectesting.compiler-request/v0.3",
                "selected_spec_ref": ref,
            },
            resolved_artifacts,
            source_base_dir=source_base_dir,
            compiler_profile=compiler_profile,
            sut_adapter_profile=sut_adapter_profile,
            capability_registry=capability_registry,
        )
        items.append(
            {
                "selected_spec_ref": ref,
                "assessment": assessment,
            }
        )

    resolution_counts = Counter(
        item["assessment"]["resolved_evaluation_target_identity"][
            "resolution_status"
        ]
        for item in items
    )
    resolver_counts = Counter(
        item["assessment"]["resolved_evaluation_target_identity"]["resolver_id"]
        for item in items
    )
    archetype_counts = Counter(
        item["assessment"]["target_archetype"] or "unresolved"
        for item in items
    )
    source_kind_counts = Counter(
        item["assessment"]["source_kind"] or "unclassified"
        for item in items
    )
    resolution_by_source_kind: dict[str, Counter[str]] = {}
    for item in items:
        assessment = item["assessment"]
        kind = assessment["source_kind"] or "unclassified"
        resolution_by_source_kind.setdefault(kind, Counter()).update(
            [
                assessment["resolved_evaluation_target_identity"][
                    "resolution_status"
                ]
            ]
        )
    dimension_counts = {
        dimension: Counter(
            item["assessment"]["target_resolution_report"]["dimensions"][
                dimension
            ]["status"]
            for item in items
        )
        for dimension in (
            "target_compilation",
            "accepted_binding",
            "lineage",
            "assertion_formalization",
            "observation_binding",
        )
    }
    policy_lineage_counts = Counter(
        (
            item["assessment"]["target_resolution_report"].get(
                "policy_lineage_report"
            )
            or {"status": "not_available"}
        )["status"]
        for item in items
    )
    runtime_projection_counts = Counter(
        (
            item["assessment"]["target_resolution_report"].get(
                "runtime_projection_report"
            )
            or {"status": "not_available"}
        )["status"]
        for item in items
    )
    readiness_counts = {
        readiness: sum(
            bool(
                item["assessment"]["target_resolution_report"]["readiness"][
                    readiness
                ]
            )
            for item in items
        )
        for readiness in (
            "target_ready",
            "accepted_binding_ready",
            "assertion_ready",
            "runtime_target_ready",
        )
    }
    payload = {
        "schema_version": COMPILE_ASSESSMENT_SET_SCHEMA_VERSION,
        "selection": {
            "workbook": str(workbook),
            "sheet": sheet,
            "review_ok": review_ok,
        },
        "summary": {
            "spec_count": len(items),
            "resolution_status_counts": dict(sorted(resolution_counts.items())),
            "resolver_counts": dict(sorted(resolver_counts.items())),
            "target_archetype_counts": dict(sorted(archetype_counts.items())),
            "source_kind_counts": dict(sorted(source_kind_counts.items())),
            "resolution_by_source_kind": {
                kind: dict(sorted(counts.items()))
                for kind, counts in sorted(resolution_by_source_kind.items())
            },
            "target_resolution_dimension_counts": {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(dimension_counts.items())
            },
            "policy_lineage_status_counts": dict(
                sorted(policy_lineage_counts.items())
            ),
            "runtime_projection_status_counts": dict(
                sorted(runtime_projection_counts.items())
            ),
            "target_resolution_readiness_counts": readiness_counts,
            "end_to_end_ready_count": sum(
                bool(item["assessment"]["end_to_end_ready"])
                for item in items
            ),
        },
        "items": items,
        "llm_calls": 0,
    }
    payload["compile_assessment_set_fingerprint"] = content_sha256(payload)
    return payload


def assess_spec_set_file(
    *,
    workbook: str | Path,
    sheet: str,
    review_ok: int | None,
    artifact_bundle_path: str | Path,
    source_base_dir: str | Path | None,
    compiler_profile_path: str | Path,
    sut_adapter_profile_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    result = assess_spec_set(
        workbook=workbook,
        sheet=sheet,
        review_ok=review_ok,
        resolved_artifacts=resolve_artifact_bundle_file(artifact_bundle_path),
        source_base_dir=source_base_dir,
        compiler_profile=load_configuration_file(
            compiler_profile_path, kind="compiler_profile"
        ),
        sut_adapter_profile=load_configuration_file(
            sut_adapter_profile_path, kind="sut_adapter_profile"
        ),
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


__all__ = [
    "COMPILE_ASSESSMENT_SET_SCHEMA_VERSION",
    "assess_request_file",
    "assess_selected_spec",
    "assess_spec_set",
    "assess_spec_set_file",
]
