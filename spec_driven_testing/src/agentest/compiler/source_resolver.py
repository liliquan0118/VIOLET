"""Resolve a loaded source-spec branch into accepted-model lineage."""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import (
    CAP_COVERAGE_MATCH_MANIFEST,
    CAP_DECISION_CONFIGURATION_MODEL,
    CAP_EXECUTION_MATCHING_CONTRACTS,
    CAP_OPERATION_POLICY_CLOSURES,
    artifact_capability_status,
    artifact_document_for_capability,
    artifact_identity_view,
)
from .source_contracts import (
    make_resolved_spec_target,
    validate_source_spec_record,
)


REQUIRED_RESOLUTION_CAPABILITIES = {
    "operation_policy_closures": CAP_OPERATION_POLICY_CLOSURES,
    "coverage_match_manifest": CAP_COVERAGE_MATCH_MANIFEST,
    "decision_configuration_model": CAP_DECISION_CONFIGURATION_MODEL,
    "execution_matching_contracts": CAP_EXECUTION_MATCHING_CONTRACTS,
}

ARTICLE_TOKENS = frozenset({"a", "an", "the"})
OPERATION_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "agent",
        "are",
        "be",
        "can",
        "for",
        "help",
        "is",
        "may",
        "must",
        "of",
        "or",
        "request",
        "requests",
        "the",
        "to",
        "user",
        "will",
    }
)
TOKEN_ALIASES = {
    "hr": "hour",
    "hrs": "hours",
}

MODALITY_DECISION_COMPATIBILITY = {
    "permission": frozenset({"permitted"}),
    "prohibition": frozenset({"blocked", "forbidden", "prohibited"}),
    "obligation": frozenset({"required"}),
}


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(
        TOKEN_ALIASES.get(token, token)
        for token in re.findall(r"[a-z0-9]+", normalized)
    )


def _normalized_text(value: str) -> str:
    return " ".join(_tokens(value))


def _article_insensitive_text(value: str) -> str:
    return " ".join(token for token in _tokens(value) if token not in ARTICLE_TOKENS)


def _match_basis(left: str, right: str) -> str | None:
    if _normalized_text(left) == _normalized_text(right):
        return "normalized_lexical_exact"
    if _article_insensitive_text(left) == _article_insensitive_text(right):
        return "article_insensitive_lexical_exact"
    return None


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _artifact_identity(resolved_artifacts: Mapping[str, Any]) -> dict[str, Any]:
    identity = artifact_identity_view(resolved_artifacts)
    return {
        "evidence_kind": "accepted_artifact_identity",
        **identity,
    }


def _artifact_documents(
    resolved_artifacts: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    documents = {}
    diagnostics = []
    for name, capability in REQUIRED_RESOLUTION_CAPABILITIES.items():
        status = artifact_capability_status(resolved_artifacts, capability)
        if status["status"] != "available_unique":
            diagnostics.append(
                {
                    "code": "resolution_artifact_capability_unavailable",
                    "logical_artifact_role": name,
                    "required_capability": capability,
                    "capability_status": status["status"],
                    "provider_artifact_ids": status["provider_artifact_ids"],
                }
            )
        else:
            documents[name] = artifact_document_for_capability(
                resolved_artifacts,
                capability,
            )
    return documents, diagnostics


def _walk_source_atoms(value: Any):
    if isinstance(value, Mapping):
        if value.get("kind") == "source_atom":
            yield value
        for child in value.values():
            yield from _walk_source_atoms(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_source_atoms(child)


def _source_atom_candidates(
    source_text: str,
    closures: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    families = closures.get("operation_policy_families") or []
    candidates_by_basis: dict[str, list[dict[str, Any]]] = {
        "normalized_lexical_exact": [],
        "article_insensitive_lexical_exact": [],
    }
    for family in families:
        if not isinstance(family, Mapping):
            continue
        family_id = family.get("operation_policy_family_id")
        for gate in family.get("eligibility_gates") or []:
            if not isinstance(gate, Mapping):
                continue
            for branch in gate.get("branches") or []:
                if not isinstance(branch, Mapping):
                    continue
                for atom in _walk_source_atoms(branch.get("guard")):
                    accepted_text = atom.get("source_text")
                    if not isinstance(accepted_text, str) or not accepted_text:
                        continue
                    basis = _match_basis(source_text, accepted_text)
                    if basis is None:
                        continue
                    candidates_by_basis[basis].append(
                        {
                            "operation_policy_family_id": family_id,
                            "operation_expressions": deepcopy(
                                family.get("operation_expressions") or []
                            ),
                            "verified_tool_names": deepcopy(
                                family.get("verified_tool_names") or []
                            ),
                            "eligibility_gate_id": gate.get("eligibility_gate_id"),
                            "eligibility_branch_id": branch.get(
                                "eligibility_branch_id"
                            ),
                            "accepted_branch_decision": branch.get("decision"),
                            "accepted_source_text": accepted_text,
                            "source_evidence_id": atom.get("source_evidence_id"),
                            "source_span_id": atom.get("source_span_id"),
                            "match_basis": basis,
                        }
                    )
    for basis in (
        "normalized_lexical_exact",
        "article_insensitive_lexical_exact",
    ):
        raw = candidates_by_basis[basis]
        if not raw:
            continue
        deduplicated = {}
        for candidate in raw:
            key = (
                candidate.get("operation_policy_family_id"),
                candidate.get("eligibility_branch_id"),
                candidate.get("accepted_branch_decision"),
                candidate.get("accepted_source_text"),
                candidate.get("source_evidence_id"),
                candidate.get("source_span_id"),
            )
            deduplicated.setdefault(key, candidate)
        ordered_keys = sorted(
            deduplicated,
            key=lambda key: tuple("" if value is None else str(value) for value in key),
        )
        return [deduplicated[key] for key in ordered_keys], basis
    return [], None


def _unique_records(
    records: Any,
    field: str,
    expected: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(records, list):
        return []
    return [
        record
        for record in records
        if isinstance(record, Mapping) and record.get(field) == expected
    ]


def _candidate_bindings(
    *,
    source: Mapping[str, Any],
    atom: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics = []
    evidence = []
    family_id = atom.get("operation_policy_family_id")
    span_id = atom.get("source_span_id")
    evidence_id = atom.get("source_evidence_id")
    decision = atom.get("accepted_branch_decision")
    for field, value in (
        ("operation_policy_family_id", family_id),
        ("source_span_id", span_id),
        ("source_evidence_id", evidence_id),
        ("accepted_branch_decision", decision),
    ):
        if not isinstance(value, str) or not value:
            diagnostics.append(
                {
                    "code": "source_atom_lineage_field_missing",
                    "field": field,
                    "accepted_source_text": atom.get("accepted_source_text"),
                }
            )
    if diagnostics:
        return [], evidence, diagnostics

    closures = documents["operation_policy_closures"]
    closure_families = _unique_records(
        closures.get("operation_policy_families"),
        "operation_policy_family_id",
        family_id,
    )
    manifest = documents["coverage_match_manifest"]
    manifest_families = _unique_records(
        manifest.get("operation_family_match_manifests"),
        "operation_policy_family_id",
        family_id,
    )
    execution = documents["execution_matching_contracts"]
    execution_families = _unique_records(
        execution.get("operation_family_contracts"),
        "operation_policy_family_id",
        family_id,
    )
    for artifact_name, matches in (
        ("operation_policy_closures", closure_families),
        ("coverage_match_manifest", manifest_families),
        ("execution_matching_contracts", execution_families),
    ):
        if len(matches) != 1:
            diagnostics.append(
                {
                    "code": "operation_family_lineage_not_unique",
                    "artifact_name": artifact_name,
                    "operation_policy_family_id": family_id,
                    "match_count": len(matches),
                }
            )
    if diagnostics:
        return [], evidence, diagnostics

    accepted_text = atom["accepted_source_text"]
    factor_matches_by_basis: dict[str, list[Mapping[str, Any]]] = {
        "normalized_lexical_exact": [],
        "article_insensitive_lexical_exact": [],
    }
    for factor in manifest_families[0].get("factor_evaluator_contracts") or []:
        if not isinstance(factor, Mapping):
            continue
        surface = factor.get("surface_text")
        if not isinstance(surface, str):
            continue
        basis = _match_basis(accepted_text, surface)
        if basis:
            factor_matches_by_basis[basis].append(factor)
    factor_basis = None
    factor_matches = []
    for basis in (
        "normalized_lexical_exact",
        "article_insensitive_lexical_exact",
    ):
        if factor_matches_by_basis[basis]:
            factor_basis = basis
            factor_matches = factor_matches_by_basis[basis]
            break
    if not factor_matches:
        diagnostics.append(
            {
                "code": "source_atom_has_no_factor_contract",
                "operation_policy_family_id": family_id,
                "source_span_id": span_id,
                "accepted_source_text": accepted_text,
            }
        )
        return [], evidence, diagnostics

    decision_model = documents["decision_configuration_model"]
    bindings = []
    for factor in factor_matches:
        predicate_id = factor.get("predicate_id")
        if not isinstance(predicate_id, str) or not predicate_id:
            diagnostics.append(
                {
                    "code": "factor_contract_predicate_missing",
                    "operation_policy_family_id": family_id,
                }
            )
            continue
        role_matches = [
            record
            for record in decision_model.get("predicate_role_inventory") or []
            if isinstance(record, Mapping)
            and record.get("operation_policy_family_id") == family_id
            and record.get("predicate_id") == predicate_id
        ]
        fact_matches = _unique_records(
            execution_families[0].get("fact_contracts"),
            "predicate_id",
            predicate_id,
        )
        if len(role_matches) != 1 or len(fact_matches) != 1:
            diagnostics.append(
                {
                    "code": "predicate_lineage_not_closed",
                    "operation_policy_family_id": family_id,
                    "predicate_id": predicate_id,
                    "decision_role_match_count": len(role_matches),
                    "execution_fact_match_count": len(fact_matches),
                }
            )
            continue
        fact_surface = fact_matches[0].get("surface_text")
        if not isinstance(fact_surface, str) or _match_basis(
            factor.get("surface_text") or "", fact_surface
        ) is None:
            diagnostics.append(
                {
                    "code": "factor_execution_surface_conflict",
                    "operation_policy_family_id": family_id,
                    "predicate_id": predicate_id,
                }
            )
            continue
        binding = {
            "source_span_id": span_id,
            "source_evidence_id": evidence_id,
            "operation_policy_family_id": family_id,
            "focus_predicate_id": predicate_id,
            "required_truth_value": True,
            "policy_modality": source["deontic"],
            "expected_operation_decision": decision,
        }
        bindings.append(binding)
        evidence.append(
            {
                "evidence_kind": "source_atom_to_predicate_lineage",
                "source_field": "given",
                "source_text": source["given"],
                "accepted_source_text": accepted_text,
                "source_atom_match_basis": atom.get("match_basis"),
                "factor_surface_match_basis": factor_basis,
                "source_span_id": span_id,
                "source_evidence_id": evidence_id,
                "operation_policy_family_id": family_id,
                "eligibility_gate_id": atom.get("eligibility_gate_id"),
                "eligibility_branch_id": atom.get("eligibility_branch_id"),
                "accepted_branch_decision": decision,
                "focus_predicate_id": predicate_id,
                "operation_expressions": deepcopy(
                    atom.get("operation_expressions") or []
                ),
                "verified_tool_names": deepcopy(
                    atom.get("verified_tool_names") or []
                ),
                "closure_checks": {
                    "operation_policy_family": True,
                    "coverage_factor_contract": True,
                    "decision_predicate_role": True,
                    "execution_fact_contract": True,
                },
            }
        )
    return bindings, evidence, diagnostics


def _deduplicate_bindings(bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "source_span_id",
        "source_evidence_id",
        "operation_policy_family_id",
        "focus_predicate_id",
        "required_truth_value",
        "policy_modality",
        "expected_operation_decision",
    )
    unique = {}
    for binding in bindings:
        key = tuple(binding.get(field) for field in fields)
        unique.setdefault(key, binding)
    return [unique[key] for key in sorted(unique)]


def _semantic_conflict(
    binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    modality = str(binding.get("policy_modality") or "").casefold()
    decision = str(binding.get("expected_operation_decision") or "").casefold()
    accepted = MODALITY_DECISION_COMPATIBILITY.get(modality)
    if accepted is None:
        return {
            "code": "unsupported_modality_for_eligibility_resolution",
            "policy_modality": modality,
            "accepted_branch_decision": decision,
        }
    if decision not in accepted:
        return {
            "code": "source_modality_conflicts_with_accepted_decision",
            "policy_modality": modality,
            "accepted_branch_decision": decision,
            "compatible_decisions": sorted(accepted),
        }
    return None


def _operation_token(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _operation_tokens(values: list[str]) -> set[str]:
    result = set()
    for value in values:
        for token in _tokens(value.replace("_", " ")):
            canonical = _operation_token(token)
            if canonical not in OPERATION_STOP_TOKENS:
                result.add(canonical)
    return result


def _operation_scope_check(
    source: Mapping[str, Any],
    binding: Mapping[str, Any],
    lineage_evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    matching = next(
        (
            item
            for item in lineage_evidence
            if item.get("focus_predicate_id") == binding.get("focus_predicate_id")
            and item.get("operation_policy_family_id")
            == binding.get("operation_policy_family_id")
        ),
        {},
    )
    accepted_values = [
        value
        for value in (
            list(matching.get("operation_expressions") or [])
            + list(matching.get("verified_tool_names") or [])
        )
        if isinstance(value, str)
    ]
    source_tokens = _operation_tokens([source["when"], source["then"]])
    accepted_tokens = _operation_tokens(accepted_values)
    overlap = sorted(source_tokens & accepted_tokens)
    required_overlap = 1 if len(accepted_tokens) <= 1 else 2
    check = {
        "evidence_kind": "operation_scope_lexical_check",
        "source_fields": ["when", "then"],
        "source_operation_tokens": sorted(source_tokens),
        "accepted_operation_tokens": sorted(accepted_tokens),
        "overlap_tokens": overlap,
        "required_overlap_count": required_overlap,
        "satisfied": len(overlap) >= required_overlap,
    }
    if check["satisfied"]:
        return check, None
    return check, {
        "code": "source_operation_conflicts_with_accepted_family",
        "operation_policy_family_id": binding.get("operation_policy_family_id"),
        "overlap_tokens": overlap,
        "required_overlap_count": required_overlap,
    }


def resolve_source_spec_target(
    source_spec_record: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one loaded branch using exact accepted-artifact lineage.

    The resolver uses conservative lexical equivalence only.  It never chooses
    among multiple candidates and never repairs missing upstream lineage.
    """

    source = validate_source_spec_record(source_spec_record)
    identity = _artifact_identity(resolved_artifacts)
    documents, artifact_diagnostics = _artifact_documents(resolved_artifacts)
    if artifact_diagnostics:
        return make_resolved_spec_target(
            source_spec_record=source,
            resolution_status="lineage_gap",
            resolution_evidence=[identity],
            diagnostics=artifact_diagnostics,
        )

    atoms, atom_basis = _source_atom_candidates(
        source["given"], documents["operation_policy_closures"]
    )
    match_evidence = {
        "evidence_kind": "source_atom_search",
        "source_field": "given",
        "source_text": source["given"],
        "normalized_source_text": _normalized_text(source["given"]),
        "selected_match_basis": atom_basis,
        "candidate_count": len(atoms),
    }
    if not atoms:
        return make_resolved_spec_target(
            source_spec_record=source,
            resolution_status="unresolved",
            resolution_evidence=[identity, match_evidence],
            diagnostics=[
                {
                    "code": "no_accepted_source_atom_match",
                    "source_field": "given",
                }
            ],
        )

    bindings = []
    lineage_evidence = []
    lineage_diagnostics = []
    for atom in atoms:
        candidate_bindings, candidate_evidence, candidate_diagnostics = (
            _candidate_bindings(source=source, atom=atom, documents=documents)
        )
        bindings.extend(candidate_bindings)
        lineage_evidence.extend(candidate_evidence)
        lineage_diagnostics.extend(candidate_diagnostics)
    bindings = _deduplicate_bindings(bindings)
    evidence = [identity, match_evidence, *lineage_evidence]

    if lineage_diagnostics:
        return make_resolved_spec_target(
            source_spec_record=source,
            resolution_status="lineage_gap",
            candidate_bindings=bindings,
            resolution_evidence=evidence,
            diagnostics=lineage_diagnostics,
        )
    if len(bindings) > 1:
        return make_resolved_spec_target(
            source_spec_record=source,
            resolution_status="ambiguous",
            candidate_bindings=bindings,
            resolution_evidence=evidence,
            diagnostics=[
                {
                    "code": "multiple_closed_model_bindings",
                    "candidate_count": len(bindings),
                }
            ],
        )
    if not bindings:
        return make_resolved_spec_target(
            source_spec_record=source,
            resolution_status="lineage_gap",
            resolution_evidence=evidence,
            diagnostics=[{"code": "no_closed_model_binding"}],
        )

    binding = bindings[0]
    operation_check, operation_conflict = _operation_scope_check(
        source, binding, lineage_evidence
    )
    evidence.append(operation_check)
    if operation_conflict is not None:
        return make_resolved_spec_target(
            source_spec_record=source,
            resolution_status="semantic_conflict",
            candidate_bindings=[binding],
            resolution_evidence=evidence,
            diagnostics=[operation_conflict],
        )
    conflict = _semantic_conflict(binding)
    if conflict is not None:
        return make_resolved_spec_target(
            source_spec_record=source,
            resolution_status="semantic_conflict",
            candidate_bindings=[binding],
            resolution_evidence=evidence,
            diagnostics=[conflict],
        )
    return make_resolved_spec_target(
        source_spec_record=source,
        resolution_status="resolved_unique",
        model_binding=binding,
        resolution_evidence=evidence,
        diagnostics=[],
    )


__all__ = [
    "MODALITY_DECISION_COMPATIBILITY",
    "REQUIRED_RESOLUTION_CAPABILITIES",
    "resolve_source_spec_target",
]
