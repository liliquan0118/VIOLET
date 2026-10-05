"""Data contracts at the source-spec to accepted-model boundary.

This module deliberately contains no workbook I/O and no model-resolution
logic.  It defines and validates the values that those later stages exchange.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256


SELECTED_SPEC_REF_SCHEMA_VERSION = "agentspectesting.selected-spec-ref/v0.1"
SOURCE_SPEC_RECORD_SCHEMA_VERSION = "agentspectesting.source-spec-record/v0.1"
SOURCE_SPEC_RECORD_V2_SCHEMA_VERSION = "agentspectesting.source-spec-record/v0.2"
RESOLVED_SPEC_TARGET_SCHEMA_VERSION = "agentspectesting.resolved-spec-target/v0.1"

RESOLUTION_STATUSES = frozenset(
    {
        "resolved_unique",
        "unresolved",
        "ambiguous",
        "lineage_gap",
        "semantic_conflict",
    }
)

MODEL_BINDING_FIELDS = (
    "source_span_id",
    "source_evidence_id",
    "operation_policy_family_id",
    "focus_predicate_id",
    "required_truth_value",
    "policy_modality",
    "expected_operation_decision",
)


class SourceContractError(ValueError):
    """Raised when a source-boundary value violates its data contract."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceContractError(f"{path} must be an object")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceContractError(f"{path} must be a non-empty string")
    return value


def _array_of_objects(value: Any, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SourceContractError(f"{path} must be an array")
    result = []
    for index, item in enumerate(value):
        result.append(deepcopy(dict(_mapping(item, f"{path}[{index}]"))))
    return result


def _source_scalar(value: Any, path: str, *, optional: bool = False) -> Any:
    if value is None and optional:
        return None
    if not isinstance(value, (str, int, float, bool)):
        suffix = " or null" if optional else ""
        raise SourceContractError(f"{path} must be a scalar{suffix}")
    return value


def _source_columns(value: Any, path: str) -> dict[str, Any]:
    item = _mapping(value, path)
    result: dict[str, Any] = {}
    for key, column_value in item.items():
        name = _nonempty_string(key, f"{path}.<column-name>")
        result[name] = _source_scalar(
            column_value,
            f"{path}.{name}",
            optional=True,
        )
    return result


def validate_selected_spec_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a workbook branch locator.

    The locator identifies one requested branch.  It contains no accepted-model
    identifiers such as predicate or coverage-cell IDs.
    """

    item = _mapping(value, "$")
    if item.get("schema_version") != SELECTED_SPEC_REF_SCHEMA_VERSION:
        raise SourceContractError(
            "unsupported selected spec ref schema: "
            f"{item.get('schema_version')!r}"
        )
    workbook = _nonempty_string(item.get("workbook"), "$.workbook")
    sheet = _nonempty_string(item.get("sheet"), "$.sheet")
    row = item.get("row")
    if not isinstance(row, int) or isinstance(row, bool) or row < 1:
        raise SourceContractError("$.row must be a positive integer")
    branch_id = _nonempty_string(item.get("branch_id"), "$.branch_id")
    return {
        "schema_version": SELECTED_SPEC_REF_SCHEMA_VERSION,
        "workbook": workbook,
        "sheet": sheet,
        "row": row,
        "branch_id": branch_id,
    }


def make_source_spec_record(
    *,
    selected_spec_ref: Mapping[str, Any],
    spec_id: str,
    branch_id: str,
    rule_text: str,
    given: str,
    when: str,
    then: str,
    deontic: str,
    additional_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fingerprinted record from already-extracted branch values."""

    ref = validate_selected_spec_ref(selected_spec_ref)
    normalized_branch_id = _nonempty_string(branch_id, "$.branch_id")
    if normalized_branch_id != ref["branch_id"]:
        raise SourceContractError(
            "$.branch_id does not match $.selected_spec_ref.branch_id: "
            f"{normalized_branch_id!r} != {ref['branch_id']!r}"
        )
    extras = additional_fields or {}
    extras = deepcopy(dict(_mapping(extras, "$.additional_fields")))
    payload = {
        "schema_version": SOURCE_SPEC_RECORD_SCHEMA_VERSION,
        "selected_spec_ref": ref,
        "spec_id": _nonempty_string(spec_id, "$.spec_id"),
        "branch_id": normalized_branch_id,
        "rule_text": _nonempty_string(rule_text, "$.rule_text"),
        "given": _nonempty_string(given, "$.given"),
        "when": _nonempty_string(when, "$.when"),
        "then": _nonempty_string(then, "$.then"),
        "deontic": _nonempty_string(deontic, "$.deontic"),
        "additional_fields": extras,
    }
    payload["source_record_fingerprint"] = content_sha256(payload)
    return payload


def make_source_spec_record_v2(
    *,
    selected_spec_ref: Mapping[str, Any],
    spec_id: str,
    branch_id: str,
    source_kind: str,
    source_origin: str,
    rule_text: str,
    given: str,
    when: str,
    then: str,
    deontic: str,
    evidence_quote: str | None,
    review_status: Any,
    review_note: Any,
    source_columns: Mapping[str, Any],
    loader_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the lossless, typed Source Anchor record used by new loaders.

    The top-level statement fields remain deliberately simple so later
    resolvers do not need to know about XLSX.  Classification, review metadata,
    the complete source row, and loader provenance are explicit rather than
    hidden in an untyped ``additional_fields`` bag.
    """

    ref = validate_selected_spec_ref(selected_spec_ref)
    normalized_branch_id = _nonempty_string(branch_id, "$.branch_id")
    if normalized_branch_id != ref["branch_id"]:
        raise SourceContractError(
            "$.branch_id does not match $.selected_spec_ref.branch_id: "
            f"{normalized_branch_id!r} != {ref['branch_id']!r}"
        )
    metadata = {
        "kind": _nonempty_string(source_kind, "$.source_metadata.kind"),
        "origin": _nonempty_string(source_origin, "$.source_metadata.origin"),
        "evidence_quote": (
            _nonempty_string(evidence_quote, "$.source_metadata.evidence_quote")
            if evidence_quote is not None
            else None
        ),
        "review": {
            "status": _source_scalar(
                review_status,
                "$.source_metadata.review.status",
                optional=True,
            ),
            "note": _source_scalar(
                review_note,
                "$.source_metadata.review.note",
                optional=True,
            ),
        },
    }
    columns = _source_columns(source_columns, "$.source_columns")
    provenance = deepcopy(
        dict(_mapping(loader_provenance, "$.loader_provenance"))
    )
    _nonempty_string(
        provenance.get("adapter_id"),
        "$.loader_provenance.adapter_id",
    )
    payload = {
        "schema_version": SOURCE_SPEC_RECORD_V2_SCHEMA_VERSION,
        "selected_spec_ref": ref,
        "spec_id": _nonempty_string(spec_id, "$.spec_id"),
        "branch_id": normalized_branch_id,
        "rule_text": _nonempty_string(rule_text, "$.rule_text"),
        "given": _nonempty_string(given, "$.given"),
        "when": _nonempty_string(when, "$.when"),
        "then": _nonempty_string(then, "$.then"),
        "deontic": _nonempty_string(deontic, "$.deontic"),
        "source_metadata": metadata,
        "source_columns": columns,
        "loader_provenance": provenance,
    }
    expected_columns = {
        "spec_id": payload["spec_id"],
        "branch_id": payload["branch_id"],
        "kind": metadata["kind"],
        "origin": metadata["origin"],
        "deontic": payload["deontic"],
        "rule_text": payload["rule_text"],
        "given": payload["given"],
        "when": payload["when"],
        "then": payload["then"],
        "evidence_quote": metadata["evidence_quote"],
        "review_ok": metadata["review"]["status"],
        "review_note": metadata["review"]["note"],
    }
    mismatches = []
    for name, expected in expected_columns.items():
        if name not in columns:
            continue
        actual = columns[name]
        if name == "evidence_quote" and actual == "" and expected is None:
            continue
        if actual != expected:
            mismatches.append(name)
    mismatches.sort()
    if mismatches:
        raise SourceContractError(
            "canonical source fields disagree with $.source_columns: "
            f"{mismatches!r}"
        )
    payload["source_record_fingerprint"] = content_sha256(payload)
    return payload


def validate_source_spec_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a source record, including its content fingerprint."""

    item = _mapping(value, "$source_spec_record")
    schema_version = item.get("schema_version")
    if schema_version not in {
        SOURCE_SPEC_RECORD_SCHEMA_VERSION,
        SOURCE_SPEC_RECORD_V2_SCHEMA_VERSION,
    }:
        raise SourceContractError(
            "unsupported source spec record schema: "
            f"{schema_version!r}"
        )
    expected = _nonempty_string(
        item.get("source_record_fingerprint"),
        "$source_spec_record.source_record_fingerprint",
    )
    common = {
        "selected_spec_ref": _mapping(
            item.get("selected_spec_ref"),
            "$source_spec_record.selected_spec_ref",
        ),
        "spec_id": item.get("spec_id"),
        "branch_id": item.get("branch_id"),
        "rule_text": item.get("rule_text"),
        "given": item.get("given"),
        "when": item.get("when"),
        "then": item.get("then"),
        "deontic": item.get("deontic"),
    }
    if schema_version == SOURCE_SPEC_RECORD_SCHEMA_VERSION:
        normalized = make_source_spec_record(
            **common,
            additional_fields=item.get("additional_fields") or {},
        )
    else:
        metadata = _mapping(
            item.get("source_metadata"),
            "$source_spec_record.source_metadata",
        )
        review = _mapping(
            metadata.get("review"),
            "$source_spec_record.source_metadata.review",
        )
        normalized = make_source_spec_record_v2(
            **common,
            source_kind=metadata.get("kind"),
            source_origin=metadata.get("origin"),
            evidence_quote=metadata.get("evidence_quote"),
            review_status=review.get("status"),
            review_note=review.get("note"),
            source_columns=_mapping(
                item.get("source_columns"),
                "$source_spec_record.source_columns",
            ),
            loader_provenance=_mapping(
                item.get("loader_provenance"),
                "$source_spec_record.loader_provenance",
            ),
        )
    actual = normalized["source_record_fingerprint"]
    if actual != expected:
        raise SourceContractError(
            "source spec record fingerprint mismatch: "
            f"expected {expected}, computed {actual}"
        )
    return normalized


def source_record_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return one stable metadata view for legacy and v0.2 source records."""

    source = validate_source_spec_record(value)
    if source["schema_version"] == SOURCE_SPEC_RECORD_V2_SCHEMA_VERSION:
        return deepcopy(source["source_metadata"])
    extras = _mapping(source.get("additional_fields"), "$.additional_fields")
    columns = _mapping(extras.get("source_columns") or {}, "$.source_columns")
    return {
        "kind": columns.get("kind"),
        "origin": columns.get("origin"),
        "evidence_quote": columns.get("evidence_quote"),
        "review": {
            "status": columns.get("review_ok"),
            "note": columns.get("review_note"),
        },
    }


def source_record_columns(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete source-row column view independent of schema version."""

    source = validate_source_spec_record(value)
    if source["schema_version"] == SOURCE_SPEC_RECORD_V2_SCHEMA_VERSION:
        return deepcopy(source["source_columns"])
    extras = _mapping(source.get("additional_fields"), "$.additional_fields")
    columns = deepcopy(
        dict(_mapping(extras.get("source_columns") or {}, "$.source_columns"))
    )
    for name in (
        "spec_id",
        "branch_id",
        "rule_text",
        "given",
        "when",
        "then",
        "deontic",
    ):
        columns[name] = source[name]
    return columns


def _validate_model_binding(value: Mapping[str, Any], path: str) -> dict[str, Any]:
    item = _mapping(value, path)
    normalized = {
        "source_span_id": _nonempty_string(
            item.get("source_span_id"), f"{path}.source_span_id"
        ),
        "source_evidence_id": _nonempty_string(
            item.get("source_evidence_id"), f"{path}.source_evidence_id"
        ),
        "operation_policy_family_id": _nonempty_string(
            item.get("operation_policy_family_id"),
            f"{path}.operation_policy_family_id",
        ),
        "focus_predicate_id": _nonempty_string(
            item.get("focus_predicate_id"), f"{path}.focus_predicate_id"
        ),
        "required_truth_value": item.get("required_truth_value"),
        "policy_modality": _nonempty_string(
            item.get("policy_modality"), f"{path}.policy_modality"
        ),
        "expected_operation_decision": _nonempty_string(
            item.get("expected_operation_decision"),
            f"{path}.expected_operation_decision",
        ),
    }
    if not isinstance(normalized["required_truth_value"], bool):
        raise SourceContractError(f"{path}.required_truth_value must be boolean")
    return normalized


def make_resolved_spec_target(
    *,
    source_spec_record: Mapping[str, Any],
    resolution_status: str,
    model_binding: Mapping[str, Any] | None = None,
    candidate_bindings: list[Mapping[str, Any]] | None = None,
    resolution_evidence: list[Mapping[str, Any]] | None = None,
    diagnostics: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a resolver result without performing resolution.

    Only ``resolved_unique`` may carry a selected model binding.  All failure or
    ambiguity outcomes remain explicit and therefore cannot silently drive cell
    selection.
    """

    source = validate_source_spec_record(source_spec_record)
    if resolution_status not in RESOLUTION_STATUSES:
        raise SourceContractError(
            f"unsupported resolution status: {resolution_status!r}"
        )
    if resolution_status == "resolved_unique":
        if model_binding is None:
            raise SourceContractError(
                "$.model_binding is required when resolution_status is resolved_unique"
            )
        normalized_binding: dict[str, Any] | None = _validate_model_binding(
            model_binding, "$.model_binding"
        )
    else:
        if model_binding is not None:
            raise SourceContractError(
                "$.model_binding must be absent unless resolution_status is resolved_unique"
            )
        normalized_binding = None

    normalized_candidates = [
        _validate_model_binding(value, f"$.candidate_bindings[{index}]")
        for index, value in enumerate(candidate_bindings or [])
    ]
    if resolution_status == "ambiguous" and len(normalized_candidates) < 2:
        raise SourceContractError(
            "ambiguous resolution must contain at least two candidate_bindings"
        )
    if resolution_status == "resolved_unique" and normalized_candidates:
        raise SourceContractError(
            "resolved_unique must not retain candidate_bindings"
        )

    payload = {
        "schema_version": RESOLVED_SPEC_TARGET_SCHEMA_VERSION,
        "source_record_fingerprint": source["source_record_fingerprint"],
        "source_branch_id": source["branch_id"],
        "resolution_status": resolution_status,
        "model_binding": normalized_binding,
        "candidate_bindings": normalized_candidates,
        "resolution_evidence": _array_of_objects(
            resolution_evidence or [], "$.resolution_evidence"
        ),
        "diagnostics": _array_of_objects(diagnostics or [], "$.diagnostics"),
    }
    payload["resolved_target_fingerprint"] = content_sha256(payload)
    return payload


def validate_resolved_spec_target(
    value: Mapping[str, Any],
    *,
    source_spec_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a resolver result and optionally bind it to its source record."""

    item = _mapping(value, "$resolved_spec_target")
    if item.get("schema_version") != RESOLVED_SPEC_TARGET_SCHEMA_VERSION:
        raise SourceContractError(
            "unsupported resolved spec target schema: "
            f"{item.get('schema_version')!r}"
        )
    expected = _nonempty_string(
        item.get("resolved_target_fingerprint"),
        "$resolved_spec_target.resolved_target_fingerprint",
    )
    if source_spec_record is None:
        source_fingerprint = _nonempty_string(
            item.get("source_record_fingerprint"),
            "$resolved_spec_target.source_record_fingerprint",
        )
        source_branch_id = _nonempty_string(
            item.get("source_branch_id"),
            "$resolved_spec_target.source_branch_id",
        )
    else:
        source = validate_source_spec_record(source_spec_record)
        source_fingerprint = source["source_record_fingerprint"]
        source_branch_id = source["branch_id"]
        if item.get("source_record_fingerprint") != source_fingerprint:
            raise SourceContractError(
                "resolved target does not belong to the supplied source spec record"
            )
        if item.get("source_branch_id") != source_branch_id:
            raise SourceContractError(
                "resolved target branch does not match the supplied source spec record"
            )

    status = item.get("resolution_status")
    if status not in RESOLUTION_STATUSES:
        raise SourceContractError(f"unsupported resolution status: {status!r}")
    binding = item.get("model_binding")
    if status == "resolved_unique":
        if binding is None:
            raise SourceContractError(
                "$.model_binding is required when resolution_status is resolved_unique"
            )
        normalized_binding = _validate_model_binding(binding, "$.model_binding")
    else:
        if binding is not None:
            raise SourceContractError(
                "$.model_binding must be absent unless resolution_status is resolved_unique"
            )
        normalized_binding = None

    candidates = item.get("candidate_bindings") or []
    if not isinstance(candidates, list):
        raise SourceContractError("$.candidate_bindings must be an array")
    normalized_candidates = [
        _validate_model_binding(candidate, f"$.candidate_bindings[{index}]")
        for index, candidate in enumerate(candidates)
    ]
    if status == "ambiguous" and len(normalized_candidates) < 2:
        raise SourceContractError(
            "ambiguous resolution must contain at least two candidate_bindings"
        )
    if status == "resolved_unique" and normalized_candidates:
        raise SourceContractError(
            "resolved_unique must not retain candidate_bindings"
        )

    payload = {
        "schema_version": RESOLVED_SPEC_TARGET_SCHEMA_VERSION,
        "source_record_fingerprint": source_fingerprint,
        "source_branch_id": source_branch_id,
        "resolution_status": status,
        "model_binding": normalized_binding,
        "candidate_bindings": normalized_candidates,
        "resolution_evidence": _array_of_objects(
            item.get("resolution_evidence") or [], "$.resolution_evidence"
        ),
        "diagnostics": _array_of_objects(item.get("diagnostics") or [], "$.diagnostics"),
    }
    actual = content_sha256(payload)
    if actual != expected:
        raise SourceContractError(
            "resolved target fingerprint mismatch: "
            f"expected {expected}, computed {actual}"
        )
    payload["resolved_target_fingerprint"] = actual
    return payload


__all__ = [
    "MODEL_BINDING_FIELDS",
    "RESOLUTION_STATUSES",
    "RESOLVED_SPEC_TARGET_SCHEMA_VERSION",
    "SELECTED_SPEC_REF_SCHEMA_VERSION",
    "SOURCE_SPEC_RECORD_SCHEMA_VERSION",
    "SOURCE_SPEC_RECORD_V2_SCHEMA_VERSION",
    "SourceContractError",
    "make_resolved_spec_target",
    "make_source_spec_record",
    "make_source_spec_record_v2",
    "source_record_columns",
    "source_record_metadata",
    "validate_resolved_spec_target",
    "validate_selected_spec_ref",
    "validate_source_spec_record",
]
