"""Multi-axis target compilation and accepted-binding assessment.

This module deliberately separates "what the reviewed source asks us to test"
from "how completely the current accepted runtime model can execute it".  The
older ``ResolvedEvaluationTarget`` remains the compatibility binding used by
existing downstream code; ``TargetResolutionReport`` is the public Step-3
assessment.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .evaluation_contracts import validate_resolved_evaluation_target
from .source_contracts import source_record_metadata, validate_source_spec_record


ABSTRACT_EVALUATION_TARGET_SCHEMA_VERSION = (
    "agentspectesting.abstract-evaluation-target/v0.1"
)
TARGET_RESOLUTION_REPORT_SCHEMA_VERSION = (
    "agentspectesting.target-resolution-report/v0.1"
)

SOURCE_TARGET_FAMILIES = {
    "ARG": "argument_constraint",
    "ORDER": "interaction_constraint",
    "STATE": "state_or_operation_constraint",
    "NORM": "normative_response_constraint",
}

DIMENSION_STATUSES = {
    "target_compilation": frozenset({"complete", "invalid"}),
    "accepted_binding": frozenset(
        {"unique", "multiple", "partial", "missing", "conflict"}
    ),
    "lineage": frozenset({"closed", "gap", "conflict", "not_assessed"}),
    "assertion_formalization": frozenset(
        {"complete", "partial", "unformalized", "conflict", "not_assessed"}
    ),
    "observation_binding": frozenset(
        {"complete", "partial", "missing", "conflict", "not_assessed"}
    ),
}


class TargetResolutionError(ValueError):
    """Raised when a target-resolution artifact violates its contract."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetResolutionError(f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TargetResolutionError(f"{path} must be a non-empty string")
    return value


def _objects(value: Any, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TargetResolutionError(f"{path} must be an array")
    return [
        deepcopy(dict(_mapping(item, f"{path}[{index}]")))
        for index, item in enumerate(value)
    ]


def make_abstract_evaluation_target(
    source_spec_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile every valid reviewed source branch into a source-authoritative IR."""

    source = validate_source_spec_record(source_spec_record)
    metadata = source_record_metadata(source)
    source_kind = str(metadata.get("kind") or "").upper()
    target_family = SOURCE_TARGET_FAMILIES.get(source_kind, "source_assertion")
    review = _mapping(metadata.get("review") or {}, "$.source_metadata.review")
    payload = {
        "schema_version": ABSTRACT_EVALUATION_TARGET_SCHEMA_VERSION,
        "source_record_fingerprint": source["source_record_fingerprint"],
        "source_branch_id": source["branch_id"],
        "source_classification": {
            "kind": source_kind or None,
            "origin": metadata.get("origin"),
            "review_status": review.get("status"),
        },
        "target_family": target_family,
        "subject_contract": {
            "kind": "source_described_subject",
            "source_when": source["when"],
        },
        "activation_contract": {
            "kind": "reviewed_given_when",
            "given": source["given"],
            "when": source["when"],
        },
        "normative_assertion": {
            "kind": "reviewed_then_with_modality",
            "rule_text": source["rule_text"],
            "then": source["then"],
            "deontic": source["deontic"],
        },
        "source_evidence": {
            "kind": "reviewed_source_record",
            "evidence_quote": metadata.get("evidence_quote"),
            "review_note": review.get("note"),
        },
    }
    payload["abstract_evaluation_target_fingerprint"] = content_sha256(payload)
    return payload


def validate_abstract_evaluation_target(
    value: Mapping[str, Any],
    *,
    source_spec_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = _mapping(value, "$abstract_evaluation_target")
    if item.get("schema_version") != ABSTRACT_EVALUATION_TARGET_SCHEMA_VERSION:
        raise TargetResolutionError("unsupported abstract evaluation target schema")
    expected = _string(
        item.get("abstract_evaluation_target_fingerprint"),
        "$.abstract_evaluation_target_fingerprint",
    )
    if source_spec_record is not None:
        rebuilt = make_abstract_evaluation_target(source_spec_record)
        if rebuilt["abstract_evaluation_target_fingerprint"] != expected:
            raise TargetResolutionError(
                "abstract evaluation target does not match the supplied source record"
            )
        return rebuilt
    payload = deepcopy(dict(item))
    payload.pop("abstract_evaluation_target_fingerprint", None)
    _string(payload.get("source_record_fingerprint"), "$.source_record_fingerprint")
    _string(payload.get("source_branch_id"), "$.source_branch_id")
    _string(payload.get("target_family"), "$.target_family")
    for field in (
        "source_classification",
        "subject_contract",
        "activation_contract",
        "normative_assertion",
        "source_evidence",
    ):
        _mapping(payload.get(field), f"$.{field}")
    actual = content_sha256(payload)
    if actual != expected:
        raise TargetResolutionError(
            f"abstract evaluation target fingerprint mismatch: expected {expected}, computed {actual}"
        )
    payload["abstract_evaluation_target_fingerprint"] = actual
    return payload


def _dimension(
    name: str,
    status: str,
    *,
    reason_code: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in DIMENSION_STATUSES[name]:
        raise TargetResolutionError(
            f"unsupported {name} status: {status!r}"
        )
    return {
        "status": status,
        "reason_code": reason_code,
        "evidence": deepcopy(dict(evidence or {})),
    }


def _quality_checks(target: Mapping[str, Any]) -> list[dict[str, Any]]:
    for evidence in target.get("resolution_evidence") or []:
        if evidence.get("evidence_kind") == "resolution_quality_gate":
            return _objects(evidence.get("checks") or [], "$.quality_checks")
    return []


def _candidate_ids(target: Mapping[str, Any]) -> list[str]:
    bindings = []
    if target.get("evaluation_binding") is not None:
        bindings.append(target["evaluation_binding"])
    bindings.extend(target.get("candidate_bindings") or [])
    return sorted(
        str(binding.get("binding_id"))
        for binding in bindings
        if binding.get("binding_id")
    )


def _quality_statuses(
    checks: list[Mapping[str, Any]], *check_ids: str
) -> list[str]:
    wanted = set(check_ids)
    return [
        str(check.get("status"))
        for check in checks
        if check.get("check_id") in wanted
    ]


def _accepted_binding_dimension(target: Mapping[str, Any]) -> dict[str, Any]:
    legacy = str(target["resolution_status"])
    candidates = _candidate_ids(target)
    evidence = {
        "resolver_id": target["resolver_id"],
        "candidate_binding_ids": candidates,
        "candidate_count": len(candidates),
    }
    if legacy in {"resolved_unique", "needs_adjudication"}:
        return _dimension(
            "accepted_binding",
            "unique",
            reason_code="one_accepted_binding_identified",
            evidence=evidence,
        )
    if legacy == "ambiguous":
        return _dimension(
            "accepted_binding",
            "multiple",
            reason_code="multiple_accepted_bindings_remain",
            evidence=evidence,
        )
    if legacy == "lineage_gap":
        return _dimension(
            "accepted_binding",
            "partial",
            reason_code="accepted_binding_identity_is_incomplete",
            evidence=evidence,
        )
    if legacy == "semantic_conflict":
        return _dimension(
            "accepted_binding",
            "conflict",
            reason_code="source_conflicts_with_accepted_binding",
            evidence=evidence,
        )
    return _dimension(
        "accepted_binding",
        "missing",
        reason_code="no_accepted_binding_identified",
        evidence=evidence,
    )


def _lineage_dimension(target: Mapping[str, Any]) -> dict[str, Any]:
    legacy = str(target["resolution_status"])
    if legacy == "source_grounded":
        return _dimension(
            "lineage",
            "not_assessed",
            reason_code="accepted_binding_absent",
        )
    if legacy == "lineage_gap":
        return _dimension(
            "lineage",
            "gap",
            reason_code="accepted_artifact_lineage_not_closed",
        )
    return _dimension(
        "lineage",
        "closed",
        reason_code="accepted_candidate_lineage_is_closed",
    )


def _assertion_dimension(
    target: Mapping[str, Any], checks: list[Mapping[str, Any]]
) -> dict[str, Any]:
    legacy = str(target["resolution_status"])
    evidence = {
        "quality_checks": [
            deepcopy(dict(check))
            for check in checks
            if check.get("check_id")
            in {
                "assertion_scope_and_branch",
                "semantic_judgment",
                "tool_argument_assertion",
            }
        ]
    }
    if legacy == "resolved_unique":
        return _dimension(
            "assertion_formalization",
            "complete",
            reason_code="accepted_assertion_is_mechanically_closed",
            evidence=evidence,
        )
    if legacy == "source_grounded":
        return _dimension(
            "assertion_formalization",
            "unformalized",
            reason_code="only_reviewed_natural_language_assertion_available",
        )
    if legacy == "semantic_conflict":
        return _dimension(
            "assertion_formalization",
            "conflict",
            reason_code="source_assertion_conflicts_with_accepted_semantics",
        )
    if legacy in {"ambiguous", "lineage_gap"}:
        return _dimension(
            "assertion_formalization",
            "not_assessed",
            reason_code="accepted_target_identity_not_ready_for_assertion_check",
        )
    if legacy == "needs_adjudication":
        assertion_statuses = _quality_statuses(
            checks,
            "assertion_scope_and_branch",
            "semantic_judgment",
            "tool_argument_assertion",
        )
        if assertion_statuses and all(value == "pass" for value in assertion_statuses):
            return _dimension(
                "assertion_formalization",
                "complete",
                reason_code="assertion_closed_but_another_runtime_dimension_is_partial",
                evidence=evidence,
            )
        return _dimension(
            "assertion_formalization",
            "partial",
            reason_code="accepted_subject_found_but_assertion_has_unformalized_facets",
            evidence=evidence,
        )
    return _dimension(
        "assertion_formalization",
        "not_assessed",
        reason_code="assertion_status_not_available",
    )


def _observation_dimension(
    target: Mapping[str, Any], checks: list[Mapping[str, Any]]
) -> dict[str, Any]:
    legacy = str(target["resolution_status"])
    runtime_checks = [
        deepcopy(dict(check))
        for check in checks
        if check.get("check_id") == "runtime_oracle_closure"
    ]
    evidence = {"runtime_oracle_checks": runtime_checks}
    if legacy == "resolved_unique":
        return _dimension(
            "observation_binding",
            "complete",
            reason_code="runtime_observation_contract_is_closed",
            evidence=evidence,
        )
    if legacy in {"source_grounded", "lineage_gap"}:
        return _dimension(
            "observation_binding",
            "missing",
            reason_code="runtime_observation_contract_not_bound",
        )
    if legacy in {"ambiguous", "semantic_conflict"}:
        return _dimension(
            "observation_binding",
            "not_assessed",
            reason_code="accepted_target_identity_not_ready_for_observation_binding",
        )
    if legacy == "needs_adjudication":
        statuses = [str(check.get("status")) for check in runtime_checks]
        if statuses and all(value == "pass" for value in statuses):
            return _dimension(
                "observation_binding",
                "complete",
                reason_code="runtime_observation_exists_despite_other_partial_dimensions",
                evidence=evidence,
            )
        if statuses:
            return _dimension(
                "observation_binding",
                "partial",
                reason_code="runtime_observation_binding_requires_additional_evaluator",
                evidence=evidence,
            )
        candidate = (target.get("candidate_bindings") or [{}])[0]
        observation = _mapping(
            candidate.get("evidence_contract") or {},
            "$.candidate_bindings[0].evidence_contract",
        )
        if observation.get("runtime_observable") is True:
            return _dimension(
                "observation_binding",
                "complete",
                reason_code="accepted_endpoints_are_runtime_observable",
                evidence={"evidence_contract": deepcopy(dict(observation))},
            )
        return _dimension(
            "observation_binding",
            "partial",
            reason_code="runtime_observation_binding_not_fully_proven",
            evidence=evidence,
        )
    return _dimension(
        "observation_binding",
        "not_assessed",
        reason_code="observation_status_not_available",
    )


def _apply_policy_lineage_dimensions(
    dimensions: dict[str, dict[str, Any]],
    policy_lineage_report: Mapping[str, Any] | None,
    runtime_projection_report: Mapping[str, Any] | None,
) -> None:
    """Refine a legacy source-grounded result with accepted policy lineage.

    Exact policy evidence establishes accepted identity and source-to-policy
    lineage.  It does not by itself establish branch-level assertion or runtime
    observation closure, so those dimensions remain deliberately partial.
    """

    if not policy_lineage_report:
        return
    status = policy_lineage_report.get("status")
    candidates = list(policy_lineage_report.get("candidates") or [])
    evidence = {
        "policy_lineage_report_fingerprint": policy_lineage_report.get(
            "policy_lineage_report_fingerprint"
        ),
        "policy_ref_ids": [item.get("policy_ref_id") for item in candidates],
    }
    projection_status = (
        runtime_projection_report.get("status")
        if runtime_projection_report is not None
        else None
    )
    projection_evidence = (
        {
            "runtime_projection_report_fingerprint": runtime_projection_report.get(
                "runtime_projection_report_fingerprint"
            ),
            "runtime_projection_status": projection_status,
        }
        if runtime_projection_report is not None
        else {}
    )
    if status == "exact_unique":
        projection_conflict_reasons = {
            "operation_scope_conflict": "source_operation_conflicts_with_accepted_runtime_family",
            "invalid_source_branch": "source_branch_is_internally_inconsistent",
            "assertion_scope_conflict": "source_assertion_scope_conflicts_with_accepted_runtime_oracle",
            "condition_scope_conflict": "source_condition_scope_is_broader_than_accepted_runtime_path",
        }
        dimensions["accepted_binding"] = _dimension(
            "accepted_binding",
            "unique",
            reason_code="accepted_policy_ref_identified_by_exact_source_evidence",
            evidence=evidence,
        )
        dimensions["lineage"] = _dimension(
            "lineage",
            "closed",
            reason_code="source_policy_semantic_path_lineage_is_closed",
            evidence=evidence,
        )
        dimensions["assertion_formalization"] = _dimension(
            "assertion_formalization",
            "conflict" if projection_status in projection_conflict_reasons else "partial",
            reason_code=(
                projection_conflict_reasons[projection_status]
                if projection_status in projection_conflict_reasons
                else (
                    "runtime_projection_found_but_resolved_binding_migration_is_pending"
                    if projection_status in {"resolved_unique", "resolved_atomic"}
                    else "accepted_policy_found_but_atomic_branch_binding_is_pending"
                )
            ),
            evidence={
                **evidence,
                **projection_evidence,
                "semantic_unit_ids": candidates[0].get("semantic_unit_ids") or [],
                "path_test_specification_ids": candidates[0].get(
                    "path_test_specification_ids"
                )
                or [],
            },
        )
        runtime_refs = candidates[0].get("runtime_contract_refs") or []
        projection_missing = projection_status in {
            "runtime_projection_missing",
            "operation_scope_conflict",
            "invalid_source_branch",
            "assertion_scope_conflict",
            "condition_scope_conflict",
        }
        dimensions["observation_binding"] = _dimension(
            "observation_binding",
            "partial" if runtime_refs and not projection_missing else "missing",
            reason_code=(
                "branch_runtime_projection_found_but_binding_migration_is_pending"
                if projection_status == "resolved_unique"
                else (
                    "policy_runtime_contract_exists_but_atomic_branch_binding_is_pending"
                    if runtime_refs and not projection_missing
                    else "accepted_path_has_no_runtime_contract_projection"
                )
            ),
            evidence={
                **evidence,
                **projection_evidence,
                "runtime_contract_refs": runtime_refs,
            },
        )
    elif status == "exact_multiple":
        dimensions["accepted_binding"] = _dimension(
            "accepted_binding",
            "multiple",
            reason_code="source_evidence_supports_multiple_accepted_policy_refs",
            evidence=evidence,
        )
        dimensions["lineage"] = _dimension(
            "lineage",
            "closed",
            reason_code="candidate_policy_lineages_are_individually_closed",
            evidence=evidence,
        )
        dimensions["assertion_formalization"] = _dimension(
            "assertion_formalization",
            "not_assessed",
            reason_code="policy_identity_requires_branch_disambiguation",
            evidence=evidence,
        )
        dimensions["observation_binding"] = _dimension(
            "observation_binding",
            "not_assessed",
            reason_code="policy_identity_requires_branch_disambiguation",
            evidence=evidence,
        )


def _missing_requirements(
    target: Mapping[str, Any], dimensions: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    requirements = []
    for name, dimension in dimensions.items():
        if name == "target_compilation" or dimension["status"] in {
            "complete",
            "unique",
            "closed",
        }:
            continue
        requirements.append(
            {
                "dimension": name,
                "status": dimension["status"],
                "reason_code": dimension["reason_code"],
            }
        )
    for diagnostic in target.get("diagnostics") or []:
        if diagnostic.get("code"):
            requirements.append(
                {
                    "dimension": "legacy_resolver",
                    "status": target["resolution_status"],
                    "reason_code": diagnostic["code"],
                    "details": deepcopy(dict(diagnostic)),
                }
            )
    deduplicated = {}
    for item in requirements:
        key = (item["dimension"], item["status"], item["reason_code"])
        deduplicated.setdefault(key, item)
    return [deduplicated[key] for key in sorted(deduplicated)]


def make_target_resolution_report(
    *,
    source_spec_record: Mapping[str, Any],
    resolved_evaluation_target: Mapping[str, Any],
    policy_lineage_report: Mapping[str, Any] | None = None,
    runtime_projection_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = validate_source_spec_record(source_spec_record)
    abstract = make_abstract_evaluation_target(source)
    target = validate_resolved_evaluation_target(
        resolved_evaluation_target,
        source_spec_record=source,
    )
    checks = _quality_checks(target)
    if policy_lineage_report is not None:
        from .policy_lineage import validate_policy_lineage_report

        policy_lineage = validate_policy_lineage_report(policy_lineage_report)
        if policy_lineage["source_record_fingerprint"] != source[
            "source_record_fingerprint"
        ]:
            raise TargetResolutionError(
                "policy lineage report does not belong to source record"
            )
    else:
        policy_lineage = None
    if runtime_projection_report is not None:
        from .runtime_projection import validate_runtime_projection_report

        runtime_projection = validate_runtime_projection_report(
            runtime_projection_report
        )
        if runtime_projection["source_record_fingerprint"] != source[
            "source_record_fingerprint"
        ]:
            raise TargetResolutionError(
                "runtime projection report does not belong to source record"
            )
    else:
        runtime_projection = None
    dimensions = {
        "target_compilation": _dimension(
            "target_compilation",
            "complete",
            reason_code="reviewed_source_compiled_to_abstract_target",
            evidence={
                "abstract_evaluation_target_fingerprint": abstract[
                    "abstract_evaluation_target_fingerprint"
                ]
            },
        ),
        "accepted_binding": _accepted_binding_dimension(target),
        "lineage": _lineage_dimension(target),
        "assertion_formalization": _assertion_dimension(target, checks),
        "observation_binding": _observation_dimension(target, checks),
    }
    if target["resolution_status"] == "source_grounded":
        _apply_policy_lineage_dimensions(
            dimensions, policy_lineage, runtime_projection
        )
    accepted_binding_ready = (
        dimensions["accepted_binding"]["status"] == "unique"
        and dimensions["lineage"]["status"] == "closed"
    )
    assertion_ready = (
        accepted_binding_ready
        and dimensions["assertion_formalization"]["status"] == "complete"
    )
    runtime_target_ready = (
        assertion_ready
        and dimensions["observation_binding"]["status"] == "complete"
    )
    payload = {
        "schema_version": TARGET_RESOLUTION_REPORT_SCHEMA_VERSION,
        "source_record_fingerprint": source["source_record_fingerprint"],
        "source_branch_id": source["branch_id"],
        "abstract_evaluation_target": abstract,
        "legacy_resolution_identity": {
            "fingerprint": target["resolved_evaluation_target_fingerprint"],
            "resolution_status": target["resolution_status"],
            "resolver_id": target["resolver_id"],
        },
        "policy_lineage_report": deepcopy(policy_lineage),
        "runtime_projection_report": deepcopy(runtime_projection),
        "dimensions": dimensions,
        "readiness": {
            "target_ready": True,
            "accepted_binding_ready": accepted_binding_ready,
            "assertion_ready": assertion_ready,
            "runtime_target_ready": runtime_target_ready,
        },
        "missing_requirements": _missing_requirements(target, dimensions),
        "llm_calls": 0,
    }
    payload["target_resolution_report_fingerprint"] = content_sha256(payload)
    return payload


def validate_target_resolution_report(
    value: Mapping[str, Any],
    *,
    source_spec_record: Mapping[str, Any] | None = None,
    resolved_evaluation_target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = _mapping(value, "$target_resolution_report")
    if item.get("schema_version") != TARGET_RESOLUTION_REPORT_SCHEMA_VERSION:
        raise TargetResolutionError("unsupported target resolution report schema")
    expected = _string(
        item.get("target_resolution_report_fingerprint"),
        "$.target_resolution_report_fingerprint",
    )
    if source_spec_record is not None and resolved_evaluation_target is not None:
        rebuilt = make_target_resolution_report(
            source_spec_record=source_spec_record,
            resolved_evaluation_target=resolved_evaluation_target,
            policy_lineage_report=item.get("policy_lineage_report"),
            runtime_projection_report=item.get("runtime_projection_report"),
        )
        if rebuilt["target_resolution_report_fingerprint"] != expected:
            raise TargetResolutionError(
                "target resolution report does not match supplied source/legacy target"
            )
        return rebuilt
    payload = deepcopy(dict(item))
    payload.pop("target_resolution_report_fingerprint", None)
    validate_abstract_evaluation_target(payload.get("abstract_evaluation_target") or {})
    if payload.get("policy_lineage_report") is not None:
        from .policy_lineage import validate_policy_lineage_report

        validate_policy_lineage_report(payload["policy_lineage_report"])
    if payload.get("runtime_projection_report") is not None:
        from .runtime_projection import validate_runtime_projection_report

        validate_runtime_projection_report(payload["runtime_projection_report"])
    dimensions = _mapping(payload.get("dimensions"), "$.dimensions")
    for name, allowed in DIMENSION_STATUSES.items():
        dimension = _mapping(dimensions.get(name), f"$.dimensions.{name}")
        if dimension.get("status") not in allowed:
            raise TargetResolutionError(
                f"$.dimensions.{name}.status is unsupported: {dimension.get('status')!r}"
            )
        _string(dimension.get("reason_code"), f"$.dimensions.{name}.reason_code")
    _mapping(payload.get("readiness"), "$.readiness")
    _objects(payload.get("missing_requirements") or [], "$.missing_requirements")
    actual = content_sha256(payload)
    if actual != expected:
        raise TargetResolutionError(
            f"target resolution report fingerprint mismatch: expected {expected}, computed {actual}"
        )
    payload["target_resolution_report_fingerprint"] = actual
    return payload


__all__ = [
    "ABSTRACT_EVALUATION_TARGET_SCHEMA_VERSION",
    "DIMENSION_STATUSES",
    "SOURCE_TARGET_FAMILIES",
    "TARGET_RESOLUTION_REPORT_SCHEMA_VERSION",
    "TargetResolutionError",
    "make_abstract_evaluation_target",
    "make_target_resolution_report",
    "validate_abstract_evaluation_target",
    "validate_target_resolution_report",
]
