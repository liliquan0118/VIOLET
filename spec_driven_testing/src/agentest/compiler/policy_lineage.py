"""Deterministic Source-Spec to accepted-policy lineage resolution.

This layer uses the upstream source evidence IDs that were actually consumed by
policy construction.  It does not infer policy identity from a runtime monitor's
surface wording and it deliberately preserves multiple policies when one source
quote spans more than one accepted proposition.
"""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import (
    CAP_ACCEPTED_POLICY_MODEL,
    CAP_EXECUTION_MATCHING_CONTRACTS,
    CAP_POLICY_PATH_TEST_SPECIFICATIONS,
    CAP_POLICY_SEMANTIC_UNITS,
    CAP_POLICY_SOURCE_EVIDENCE,
    artifact_capability_status,
    artifact_document_for_capability,
    content_sha256,
)
from .source_contracts import source_record_metadata, validate_source_spec_record


POLICY_LINEAGE_REPORT_SCHEMA_VERSION = "agentspectesting.policy-lineage-report/v0.1"

POLICY_LINEAGE_STATUSES = frozenset(
    {
        "exact_unique",
        "exact_multiple",
        "no_exact_evidence_match",
        "outside_policy_authority",
        "capability_unavailable",
    }
)


class PolicyLineageError(ValueError):
    """Raised when a policy-lineage report violates its contract."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyLineageError(f"{path} must be an object")
    return value


def _normalized_source_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def _evidence_match_basis(source_quote: str, accepted_text: str) -> str | None:
    source = _normalized_source_text(source_quote)
    accepted = _normalized_source_text(accepted_text)
    if not source or not accepted:
        return None
    if source == accepted:
        return "normalized_evidence_exact"
    if accepted in source:
        return "accepted_evidence_contained_in_source_quote"
    if source in accepted:
        return "source_quote_contained_in_accepted_evidence"
    return None


def _capability_readiness(resolved_artifacts: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        artifact_capability_status(resolved_artifacts, capability)
        for capability in (
            CAP_POLICY_SOURCE_EVIDENCE,
            CAP_ACCEPTED_POLICY_MODEL,
            CAP_POLICY_SEMANTIC_UNITS,
            CAP_POLICY_PATH_TEST_SPECIFICATIONS,
            CAP_EXECUTION_MATCHING_CONTRACTS,
        )
    ]


def _policy_records(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for proposal in document.get("proposal_results") or []:
        if not isinstance(proposal, Mapping):
            continue
        passage_id = str(proposal.get("target_source_passage_id") or "")
        for policy in proposal.get("policies") or []:
            if not isinstance(policy, Mapping) or not policy.get("policy_id"):
                continue
            result.append(
                {
                    "policy_ref_id": f"{passage_id}::{policy['policy_id']}",
                    "target_source_passage_id": passage_id,
                    "policy": policy,
                }
            )
    return result


def _evidence_index(document: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for proposal in document.get("policy_proposal_inputs") or []:
        if not isinstance(proposal, Mapping):
            continue
        passage_id = str(proposal.get("target_source_passage_id") or "")
        proposal_id = str(proposal.get("policy_proposal_input_id") or "")
        for evidence in proposal.get("target_evidence") or []:
            if not isinstance(evidence, Mapping) or not evidence.get("evidence_id"):
                continue
            item = deepcopy(dict(evidence))
            item["policy_proposal_input_id"] = proposal_id
            result[(passage_id, str(evidence["evidence_id"]))] = item
    return result


def _semantic_index(document: Mapping[str, Any]) -> dict[str, list[str]]:
    result = {}
    for item in document.get("policy_index") or []:
        if not isinstance(item, Mapping) or not item.get("policy_ref_id"):
            continue
        result[str(item["policy_ref_id"])] = sorted(
            str(value) for value in item.get("semantic_unit_ids") or []
        )
    return result


def _path_spec_index(document: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in document.get("path_test_specifications") or []:
        if not isinstance(item, Mapping) or not item.get("path_test_specification_id"):
            continue
        specification_id = str(item["path_test_specification_id"])
        for policy_ref_id in item.get("policy_ref_ids") or []:
            result.setdefault(str(policy_ref_id), []).append(specification_id)
    return {key: sorted(set(values)) for key, values in result.items()}


def _contains_policy_ref(value: Any, policy_ref_id: str) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_policy_ref(item, policy_ref_id) for item in value.values())
    if isinstance(value, list):
        return any(_contains_policy_ref(item, policy_ref_id) for item in value)
    return value == policy_ref_id


def _runtime_contract_refs(
    document: Mapping[str, Any],
    policy_ref_id: str,
    path_test_specification_ids: list[str],
) -> list[dict[str, Any]]:
    path_ids = set(path_test_specification_ids)
    result = []
    for contract in document.get("path_match_contracts") or []:
        if not isinstance(contract, Mapping):
            continue
        path_id = str(contract.get("path_test_specification_id") or "")
        # A path-match contract normally points back to policy lineage through
        # its path-test-specification ID; it need not duplicate policy_ref_id.
        # Preserve the direct-ref fallback for older artifact variants.
        if path_id not in path_ids and not _contains_policy_ref(
            contract, policy_ref_id
        ):
            continue
        result.append(
            {
                "contract_kind": "path_match_contract",
                "contract_id": contract.get("path_match_contract_id"),
                "operation_policy_family_id": contract.get(
                    "owner_operation_policy_family_id"
                ),
                "path_test_specification_id": path_id,
                "binding_basis": (
                    "path_test_specification_lineage"
                    if path_id in path_ids
                    else "embedded_policy_ref"
                ),
            }
        )
    for family in document.get("operation_family_contracts") or []:
        if not isinstance(family, Mapping):
            continue
        family_id = family.get("operation_policy_family_id")
        for field, kind, identifier in (
            ("prerequisite_monitors", "prerequisite_monitor", "prerequisite_monitor_id"),
            ("outcome_monitors", "outcome_monitor", "outcome_monitor_id"),
            ("fact_contracts", "fact_contract", "predicate_id"),
        ):
            for contract in family.get(field) or []:
                if not isinstance(contract, Mapping) or not _contains_policy_ref(
                    contract, policy_ref_id
                ):
                    continue
                result.append(
                    {
                        "contract_kind": kind,
                        "contract_id": contract.get(identifier),
                        "operation_policy_family_id": family_id,
                        "path_test_specification_id": None,
                        "binding_basis": "embedded_policy_ref",
                    }
                )
    deduplicated = {
        (
            item["contract_kind"],
            str(item["contract_id"]),
            str(item["operation_policy_family_id"]),
        ): item
        for item in result
    }
    return [deduplicated[key] for key in sorted(deduplicated)]


def resolve_policy_lineage(
    source_spec_record: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve exact source-evidence lineage through accepted policy artifacts."""

    source = validate_source_spec_record(source_spec_record)
    metadata = source_record_metadata(source)
    origin = str(metadata.get("origin") or "").casefold()
    readiness = _capability_readiness(resolved_artifacts)
    unavailable = [item for item in readiness if item["status"] != "available_unique"]
    candidates: list[dict[str, Any]] = []

    if origin != "prompt":
        status = "outside_policy_authority"
        reason_code = "source_origin_is_not_system_prompt_policy"
    elif unavailable:
        status = "capability_unavailable"
        reason_code = "policy_lineage_capability_not_available_unique"
    else:
        evidence_document = artifact_document_for_capability(
            resolved_artifacts, CAP_POLICY_SOURCE_EVIDENCE
        )
        policy_document = artifact_document_for_capability(
            resolved_artifacts, CAP_ACCEPTED_POLICY_MODEL
        )
        semantic_document = artifact_document_for_capability(
            resolved_artifacts, CAP_POLICY_SEMANTIC_UNITS
        )
        path_document = artifact_document_for_capability(
            resolved_artifacts, CAP_POLICY_PATH_TEST_SPECIFICATIONS
        )
        runtime_document = artifact_document_for_capability(
            resolved_artifacts, CAP_EXECUTION_MATCHING_CONTRACTS
        )
        evidence = _evidence_index(evidence_document)
        semantic = _semantic_index(semantic_document)
        paths = _path_spec_index(path_document)
        source_quote = str(metadata.get("evidence_quote") or "")
        for record in _policy_records(policy_document):
            policy = record["policy"]
            evidence_ids = list(
                policy.get("semantic_support_evidence_ids")
                or policy.get("target_evidence_ids")
                or []
            )
            matches = []
            for evidence_id in evidence_ids:
                accepted = evidence.get(
                    (record["target_source_passage_id"], str(evidence_id))
                )
                if accepted is None:
                    continue
                basis = _evidence_match_basis(source_quote, str(accepted.get("text") or ""))
                if basis is None:
                    continue
                matches.append(
                    {
                        "policy_proposal_input_id": accepted.get(
                            "policy_proposal_input_id"
                        ),
                        "evidence_id": evidence_id,
                        "source_span_id": accepted.get("source_span_id"),
                        "match_basis": basis,
                        "accepted_evidence_text": accepted.get("text"),
                    }
                )
            if not matches:
                continue
            policy_ref_id = record["policy_ref_id"]
            candidates.append(
                {
                    "policy_ref_id": policy_ref_id,
                    "target_source_passage_id": record[
                        "target_source_passage_id"
                    ],
                    "accepted_policy_statement": policy.get("statement"),
                    "source_evidence_matches": matches,
                    "semantic_unit_ids": semantic.get(policy_ref_id, []),
                    "path_test_specification_ids": paths.get(policy_ref_id, []),
                    "runtime_contract_refs": _runtime_contract_refs(
                        runtime_document,
                        policy_ref_id,
                        paths.get(policy_ref_id, []),
                    ),
                }
            )
        candidates.sort(key=lambda item: str(item["policy_ref_id"]))
        if not candidates:
            status = "no_exact_evidence_match"
            reason_code = "source_quote_does_not_match_accepted_semantic_evidence"
        elif len(candidates) == 1:
            status = "exact_unique"
            reason_code = "one_policy_ref_has_exact_source_evidence_lineage"
        else:
            status = "exact_multiple"
            reason_code = "source_evidence_supports_multiple_accepted_policies"

    payload = {
        "schema_version": POLICY_LINEAGE_REPORT_SCHEMA_VERSION,
        "source_record_fingerprint": source["source_record_fingerprint"],
        "source_branch_id": source["branch_id"],
        "source_origin": origin or None,
        "status": status,
        "reason_code": reason_code,
        "capability_readiness": readiness,
        "candidates": candidates,
        "llm_calls": 0,
    }
    payload["policy_lineage_report_fingerprint"] = content_sha256(payload)
    return payload


def validate_policy_lineage_report(value: Mapping[str, Any]) -> dict[str, Any]:
    item = _mapping(value, "$policy_lineage_report")
    if item.get("schema_version") != POLICY_LINEAGE_REPORT_SCHEMA_VERSION:
        raise PolicyLineageError("unsupported policy lineage report schema")
    if item.get("status") not in POLICY_LINEAGE_STATUSES:
        raise PolicyLineageError(f"unsupported policy lineage status: {item.get('status')!r}")
    expected = item.get("policy_lineage_report_fingerprint")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise PolicyLineageError("policy lineage fingerprint must be a SHA-256 digest")
    payload = deepcopy(dict(item))
    payload.pop("policy_lineage_report_fingerprint", None)
    actual = content_sha256(payload)
    if actual != expected:
        raise PolicyLineageError(
            f"policy lineage fingerprint mismatch: expected {expected}, computed {actual}"
        )
    payload["policy_lineage_report_fingerprint"] = actual
    return payload


__all__ = [
    "POLICY_LINEAGE_REPORT_SCHEMA_VERSION",
    "POLICY_LINEAGE_STATUSES",
    "PolicyLineageError",
    "resolve_policy_lineage",
    "validate_policy_lineage_report",
]
