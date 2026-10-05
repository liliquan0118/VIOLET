"""General source-to-evaluation contracts.

Unlike ``ResolvedSpecTarget v0.1``, this contract does not assume that every
spec is an operation-outcome predicate.  It can represent argument, trace,
state, content and output assertions while retaining a compatibility projection
for the existing operation-decision compiler.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .source_contracts import source_record_metadata, validate_source_spec_record


RESOLVED_EVALUATION_TARGET_SCHEMA_VERSION = (
    "agentspectesting.resolved-evaluation-target/v0.2"
)
COMPILE_ASSESSMENT_SCHEMA_VERSION = "agentspectesting.compile-assessment/v0.2"

EVALUATION_RESOLUTION_STATUSES = frozenset(
    {
        "resolved_unique",
        "source_grounded",
        "needs_adjudication",
        "unresolved",
        "ambiguous",
        "lineage_gap",
        "semantic_conflict",
    }
)

TARGET_ARCHETYPES = frozenset(
    {
        "operation_decision",
        "tool_argument",
        "interaction_order",
        "state_transition",
        "response_content",
        "output_set",
        "path_requirement",
        "source_assertion",
    }
)


class EvaluationContractError(ValueError):
    """Raised when a general evaluation artifact is malformed."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationContractError(f"{path} must be a non-empty string")
    return value


def _objects(value: Any, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise EvaluationContractError(f"{path} must be an array")
    return [
        deepcopy(dict(_mapping(item, f"{path}[{index}]")))
        for index, item in enumerate(value)
    ]


def _evaluation_binding(value: Mapping[str, Any], path: str) -> dict[str, Any]:
    item = deepcopy(dict(_mapping(value, path)))
    _string(item.get("binding_id"), f"{path}.binding_id")
    archetype = _string(item.get("target_archetype"), f"{path}.target_archetype")
    if archetype not in TARGET_ARCHETYPES:
        raise EvaluationContractError(
            f"{path}.target_archetype is unsupported: {archetype!r}"
        )
    subject = _mapping(item.get("evaluation_subject"), f"{path}.evaluation_subject")
    assertion = _mapping(item.get("assertion_contract"), f"{path}.assertion_contract")
    evidence = _mapping(item.get("evidence_contract"), f"{path}.evidence_contract")
    _string(subject.get("kind"), f"{path}.evaluation_subject.kind")
    _string(assertion.get("kind"), f"{path}.assertion_contract.kind")
    _string(evidence.get("kind"), f"{path}.evidence_contract.kind")
    refs = _mapping(item.get("accepted_artifact_refs"), f"{path}.accepted_artifact_refs")
    if any(not isinstance(key, str) or not key for key in refs):
        raise EvaluationContractError(
            f"{path}.accepted_artifact_refs keys must be non-empty strings"
        )
    return item


def _validate_binding_grounding(
    binding: Mapping[str, Any],
    *,
    accepted: bool,
    path: str,
) -> None:
    refs = _mapping(binding.get("accepted_artifact_refs"), f"{path}.accepted_artifact_refs")
    if accepted and not refs:
        raise EvaluationContractError(
            f"{path}.accepted_artifact_refs must identify at least one accepted contract"
        )
    if not accepted and refs:
        raise EvaluationContractError(
            f"{path}.accepted_artifact_refs must be empty for source-grounded targets"
        )


def _validate_accepted_identity_evidence(
    evidence: list[Mapping[str, Any]],
    *,
    required: bool,
) -> None:
    identities = [
        item
        for item in evidence
        if item.get("evidence_kind") == "accepted_artifact_identity"
    ]
    if required and len(identities) != 1:
        raise EvaluationContractError(
            "accepted target resolution requires exactly one accepted_artifact_identity evidence item"
        )
    for index, identity in enumerate(identities):
        for field in (
            "artifact_bundle_id",
            "compile_run_id",
            "artifact_catalog_fingerprint",
        ):
            _string(identity.get(field), f"$.resolution_evidence[{index}].{field}")
        hashes = _mapping(
            identity.get("artifact_hashes"),
            f"$.resolution_evidence[{index}].artifact_hashes",
        )
        if not hashes:
            raise EvaluationContractError(
                "accepted_artifact_identity.artifact_hashes must not be empty"
            )


def make_resolved_evaluation_target(
    *,
    source_spec_record: Mapping[str, Any],
    resolution_status: str,
    resolver_id: str,
    evaluation_binding: Mapping[str, Any] | None = None,
    candidate_bindings: list[Mapping[str, Any]] | None = None,
    resolution_evidence: list[Mapping[str, Any]] | None = None,
    diagnostics: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    source = validate_source_spec_record(source_spec_record)
    if resolution_status not in EVALUATION_RESOLUTION_STATUSES:
        raise EvaluationContractError(
            f"unsupported evaluation resolution status: {resolution_status!r}"
        )
    carries_binding = resolution_status in {"resolved_unique", "source_grounded"}
    if carries_binding:
        if evaluation_binding is None:
            raise EvaluationContractError(
                "evaluation_binding is required for resolved/source-grounded targets"
            )
        binding = _evaluation_binding(evaluation_binding, "$.evaluation_binding")
        _validate_binding_grounding(
            binding,
            accepted=resolution_status == "resolved_unique",
            path="$.evaluation_binding",
        )
    else:
        if evaluation_binding is not None:
            raise EvaluationContractError(
                "evaluation_binding must be absent for unresolved targets"
            )
        binding = None
    candidates = [
        _evaluation_binding(value, f"$.candidate_bindings[{index}]")
        for index, value in enumerate(candidate_bindings or [])
    ]
    for index, candidate in enumerate(candidates):
        _validate_binding_grounding(
            candidate,
            accepted=True,
            path=f"$.candidate_bindings[{index}]",
        )
    if resolution_status == "ambiguous" and len(candidates) < 2:
        raise EvaluationContractError(
            "ambiguous evaluation resolution requires at least two candidates"
        )
    if resolution_status == "needs_adjudication" and len(candidates) != 1:
        raise EvaluationContractError(
            "needs_adjudication evaluation resolution requires exactly one candidate"
        )
    if carries_binding and candidates:
        raise EvaluationContractError(
            "resolved/source-grounded target must not retain candidates"
        )
    evidence = _objects(resolution_evidence or [], "$.resolution_evidence")
    _validate_accepted_identity_evidence(
        evidence,
        required=resolution_status in {
            "resolved_unique",
            "needs_adjudication",
            "ambiguous",
            "lineage_gap",
            "semantic_conflict",
        },
    )
    payload = {
        "schema_version": RESOLVED_EVALUATION_TARGET_SCHEMA_VERSION,
        "source_record_fingerprint": source["source_record_fingerprint"],
        "source_branch_id": source["branch_id"],
        "resolution_status": resolution_status,
        "resolver_id": _string(resolver_id, "$.resolver_id"),
        "evaluation_binding": binding,
        "candidate_bindings": candidates,
        "resolution_evidence": evidence,
        "diagnostics": _objects(diagnostics or [], "$.diagnostics"),
    }
    payload["resolved_evaluation_target_fingerprint"] = content_sha256(payload)
    return payload


def validate_resolved_evaluation_target(
    value: Mapping[str, Any],
    *,
    source_spec_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = _mapping(value, "$resolved_evaluation_target")
    if item.get("schema_version") != RESOLVED_EVALUATION_TARGET_SCHEMA_VERSION:
        raise EvaluationContractError("unsupported resolved evaluation target schema")
    expected = _string(
        item.get("resolved_evaluation_target_fingerprint"),
        "$.resolved_evaluation_target_fingerprint",
    )
    if source_spec_record is None:
        source_fingerprint = _string(
            item.get("source_record_fingerprint"), "$.source_record_fingerprint"
        )
        source_branch_id = _string(item.get("source_branch_id"), "$.source_branch_id")
        source = None
    else:
        source = validate_source_spec_record(source_spec_record)
        source_fingerprint = source["source_record_fingerprint"]
        source_branch_id = source["branch_id"]
        if item.get("source_record_fingerprint") != source_fingerprint:
            raise EvaluationContractError(
                "resolved evaluation target does not belong to source record"
            )
        if item.get("source_branch_id") != source_branch_id:
            raise EvaluationContractError(
                "resolved evaluation target branch does not match source record"
            )
    if source is not None:
        rebuilt = make_resolved_evaluation_target(
            source_spec_record=source,
            resolution_status=str(item.get("resolution_status")),
            resolver_id=item.get("resolver_id"),
            evaluation_binding=item.get("evaluation_binding"),
            candidate_bindings=item.get("candidate_bindings") or [],
            resolution_evidence=item.get("resolution_evidence") or [],
            diagnostics=item.get("diagnostics") or [],
        )
        actual = rebuilt["resolved_evaluation_target_fingerprint"]
        if actual != expected:
            raise EvaluationContractError(
                f"resolved evaluation target fingerprint mismatch: expected {expected}, computed {actual}"
            )
        return rebuilt

    status = item.get("resolution_status")
    if status not in EVALUATION_RESOLUTION_STATUSES:
        raise EvaluationContractError(f"unsupported evaluation resolution status: {status!r}")
    binding = item.get("evaluation_binding")
    carries_binding = status in {"resolved_unique", "source_grounded"}
    if carries_binding:
        if binding is None:
            raise EvaluationContractError("resolved target requires evaluation_binding")
        normalized_binding = _evaluation_binding(binding, "$.evaluation_binding")
        _validate_binding_grounding(
            normalized_binding,
            accepted=status == "resolved_unique",
            path="$.evaluation_binding",
        )
    else:
        if binding is not None:
            raise EvaluationContractError("unresolved target must not carry evaluation_binding")
        normalized_binding = None
    candidates = [
        _evaluation_binding(candidate, f"$.candidate_bindings[{index}]")
        for index, candidate in enumerate(item.get("candidate_bindings") or [])
    ]
    for index, candidate in enumerate(candidates):
        _validate_binding_grounding(
            candidate,
            accepted=True,
            path=f"$.candidate_bindings[{index}]",
        )
    if status == "ambiguous" and len(candidates) < 2:
        raise EvaluationContractError("ambiguous target requires at least two candidates")
    if status == "needs_adjudication" and len(candidates) != 1:
        raise EvaluationContractError(
            "needs_adjudication target requires exactly one candidate"
        )
    evidence = _objects(item.get("resolution_evidence") or [], "$.resolution_evidence")
    _validate_accepted_identity_evidence(
        evidence,
        required=status in {
            "resolved_unique",
            "needs_adjudication",
            "ambiguous",
            "lineage_gap",
            "semantic_conflict",
        },
    )
    payload = {
        "schema_version": RESOLVED_EVALUATION_TARGET_SCHEMA_VERSION,
        "source_record_fingerprint": source_fingerprint,
        "source_branch_id": source_branch_id,
        "resolution_status": status,
        "resolver_id": _string(item.get("resolver_id"), "$.resolver_id"),
        "evaluation_binding": normalized_binding,
        "candidate_bindings": candidates,
        "resolution_evidence": evidence,
        "diagnostics": _objects(item.get("diagnostics") or [], "$.diagnostics"),
    }
    actual = content_sha256(payload)
    if actual != expected:
        raise EvaluationContractError(
            f"resolved evaluation target fingerprint mismatch: expected {expected}, computed {actual}"
        )
    payload["resolved_evaluation_target_fingerprint"] = actual
    return payload


def make_compile_assessment(
    *,
    source_spec_record: Mapping[str, Any],
    resolved_evaluation_target: Mapping[str, Any],
    target_resolution_report: Mapping[str, Any],
    stages: Mapping[str, Any],
    capability_registry: Mapping[str, Any],
    diagnostics: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    source = validate_source_spec_record(source_spec_record)
    target = validate_resolved_evaluation_target(
        resolved_evaluation_target, source_spec_record=source
    )
    # Local import avoids a module-level cycle: target_resolution builds on the
    # compatibility ResolvedEvaluationTarget contract defined in this module.
    from .target_resolution import validate_target_resolution_report

    resolution_report = validate_target_resolution_report(
        target_resolution_report,
        source_spec_record=source,
        resolved_evaluation_target=target,
    )
    stage_values = deepcopy(dict(_mapping(stages, "$.stages")))
    required = {
        "source_loading",
        "semantic_resolution",
        "experiment_planning",
        "probe_synthesis",
        "fixture_binding",
        "runtime_protocol",
        "surface_realization",
        "outcome_oracle",
        "mutation_provider",
        "guidance",
    }
    missing = sorted(required - set(stage_values))
    if missing:
        raise EvaluationContractError(
            "compile assessment is missing stages: " + ", ".join(missing)
        )
    for name, stage in stage_values.items():
        value = _mapping(stage, f"$.stages.{name}")
        if value.get("status") not in {"ready", "partial", "blocked", "not_assessed"}:
            raise EvaluationContractError(
                f"$.stages.{name}.status is unsupported: {value.get('status')!r}"
            )
    end_to_end = all(
        stage_values[name]["status"] == "ready"
        for name in required
    )
    payload = {
        "schema_version": COMPILE_ASSESSMENT_SCHEMA_VERSION,
        "source_record_fingerprint": source["source_record_fingerprint"],
        "source_branch_id": source["branch_id"],
        "source_kind": source_record_metadata(source).get("kind"),
        "resolved_evaluation_target_identity": {
            "fingerprint": target["resolved_evaluation_target_fingerprint"],
            "resolution_status": target["resolution_status"],
            "resolver_id": target["resolver_id"],
        },
        "target_resolution_report": resolution_report,
        "target_archetype": (
            (target.get("evaluation_binding") or {}).get("target_archetype")
        ),
        "stages": stage_values,
        "end_to_end_ready": end_to_end,
        "capability_registry": deepcopy(
            dict(_mapping(capability_registry, "$.capability_registry"))
        ),
        "diagnostics": _objects(diagnostics or [], "$.diagnostics"),
        "llm_calls": 0,
    }
    payload["compile_assessment_fingerprint"] = content_sha256(payload)
    return payload


__all__ = [
    "COMPILE_ASSESSMENT_SCHEMA_VERSION",
    "EVALUATION_RESOLUTION_STATUSES",
    "EvaluationContractError",
    "RESOLVED_EVALUATION_TARGET_SCHEMA_VERSION",
    "TARGET_ARCHETYPES",
    "make_compile_assessment",
    "make_resolved_evaluation_target",
    "validate_resolved_evaluation_target",
]
