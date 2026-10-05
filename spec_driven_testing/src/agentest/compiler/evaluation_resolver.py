"""Registry-based resolution from one reviewed source branch to an evaluation target."""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .artifacts import (
    CAP_AGENT_TOOL_SCHEMAS,
    CAP_COVERAGE_MATCH_MANIFEST,
    CAP_DECISION_CONFIGURATION_MODEL,
    CAP_EXECUTION_MATCHING_CONTRACTS,
    CAP_OPERATION_POLICY_CLOSURES,
    CAP_TOOL_OBSERVABLE_ENDPOINTS,
    artifact_capability_status,
    artifact_document_for_capability,
    artifact_identity_view,
)
from .evaluation_contracts import TARGET_ARCHETYPES, make_resolved_evaluation_target
from .source_contracts import source_record_metadata, validate_source_spec_record
from .source_resolver import resolve_source_spec_target


EVALUATION_RESOLVER_REGISTRY_VERSION = "accepted-evaluation-resolver-registry/v0.3"
OPERATION_FACTOR_RESOLVER_ID = "accepted-operation-factor-resolver/v0.1"
TOOL_ARGUMENT_RESOLVER_ID = "accepted-tool-argument-resolver/v0.1"
PATH_CONTRACT_RESOLVER_ID = "accepted-path-contract-resolver/v0.2"
SOURCE_FALLBACK_RESOLVER_ID = "reviewed-source-grounding-resolver/v0.1"


@dataclass(frozen=True)
class EvaluationResolverRegistration:
    """One ordered resolver capability in the source-to-target registry."""

    resolver_id: str
    source_kinds: tuple[str, ...]
    source_origins: tuple[str, ...]
    target_archetypes: tuple[str, ...]
    required_artifact_capabilities: tuple[str, ...]
    resolve: Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]

    def applies_to(self, source: Mapping[str, Any]) -> bool:
        metadata = source_record_metadata(source)
        kind = str(metadata.get("kind") or "").upper()
        origin = str(metadata.get("origin") or "").casefold()
        kind_applies = not self.source_kinds or kind in self.source_kinds
        origin_applies = not self.source_origins or origin in self.source_origins
        return kind_applies and origin_applies

    def capability_readiness(
        self, resolved_artifacts: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            artifact_capability_status(resolved_artifacts, capability)
            for capability in self.required_artifact_capabilities
        ]


_ALIASES = {
    "hrs": "hours",
    "hr": "hour",
    "purchase": "buy",
    "purchased": "buy",
    "booking": "reservation",
    "bookings": "reservation",
}

_NON_SEMANTIC_TOKENS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "before",
        "by", "can", "for", "from", "if", "in", "is", "it", "must",
        "not", "of", "on", "only", "or", "should", "such", "than", "that",
        "the", "their", "them", "then", "to", "user", "users", "when",
        "with", "within", "agent",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _artifact_identity_evidence(
    resolved_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "evidence_kind": "accepted_artifact_identity",
        **artifact_identity_view(resolved_artifacts),
    }


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(
        _ALIASES.get(token, token)
        for token in re.findall(r"[a-z0-9]+", normalized)
    )


def _normalized(value: str) -> str:
    return " ".join(_tokens(value))


def _article_insensitive(value: str) -> str:
    return " ".join(token for token in _tokens(value) if token not in {"a", "an", "the"})


def _informative(value: str) -> frozenset[str]:
    return frozenset(token for token in _tokens(value) if token not in _NON_SEMANTIC_TOKENS)


def _match_basis(source_text: str, accepted_text: str) -> tuple[int, str] | None:
    if _normalized(source_text) == _normalized(accepted_text):
        return 0, "normalized_lexical_exact"
    if _article_insensitive(source_text) == _article_insensitive(accepted_text):
        return 1, "article_insensitive_lexical_exact"
    source_tokens = _informative(source_text)
    accepted_tokens = _informative(accepted_text)
    shorter = min(len(source_tokens), len(accepted_tokens))
    if shorter >= 4 and (
        source_tokens.issubset(accepted_tokens)
        or accepted_tokens.issubset(source_tokens)
    ):
        return 2, "informative_token_subset"
    return None


def _documents(resolved_artifacts: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    logical_capabilities = {
        "agent_tool_schemas": CAP_AGENT_TOOL_SCHEMAS,
        "tool_observable_endpoints": CAP_TOOL_OBSERVABLE_ENDPOINTS,
        "operation_policy_closures": CAP_OPERATION_POLICY_CLOSURES,
        "coverage_match_manifest": CAP_COVERAGE_MATCH_MANIFEST,
        "decision_configuration_model": CAP_DECISION_CONFIGURATION_MODEL,
        "execution_matching_contracts": CAP_EXECUTION_MATCHING_CONTRACTS,
    }
    result = {}
    for name, capability in logical_capabilities.items():
        if artifact_capability_status(resolved_artifacts, capability)["status"] != "available_unique":
            continue
        result[name] = artifact_document_for_capability(
            resolved_artifacts,
            capability,
        )
    return result


def _operation_family_contract(
    documents: Mapping[str, Mapping[str, Any]], family_id: str
) -> Mapping[str, Any]:
    execution = documents.get("execution_matching_contracts") or {}
    matches = [
        item
        for item in execution.get("operation_family_contracts") or []
        if isinstance(item, Mapping)
        and item.get("operation_policy_family_id") == family_id
    ]
    return matches[0] if len(matches) == 1 else {}


def _factor_contract(
    documents: Mapping[str, Mapping[str, Any]], family_id: str, predicate_id: str
) -> Mapping[str, Any]:
    manifest = documents.get("coverage_match_manifest") or {}
    families = [
        item
        for item in manifest.get("operation_family_match_manifests") or []
        if isinstance(item, Mapping)
        and item.get("operation_policy_family_id") == family_id
    ]
    if len(families) != 1:
        return {}
    matches = [
        item
        for item in families[0].get("factor_evaluator_contracts") or []
        if isinstance(item, Mapping) and item.get("predicate_id") == predicate_id
    ]
    return matches[0] if len(matches) == 1 else {}


def _operation_binding(
    legacy_binding: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    family_id = str(legacy_binding["operation_policy_family_id"])
    predicate_id = str(legacy_binding["focus_predicate_id"])
    family = _operation_family_contract(documents, family_id)
    selector = deepcopy(dict(_mapping(family.get("operation_selector"))))
    factor = _factor_contract(documents, family_id, predicate_id)
    return {
        "binding_id": f"operation-decision::{family_id}::{predicate_id}",
        "target_archetype": "operation_decision",
        "evaluation_subject": {
            "kind": "operation_outcome",
            "operation_policy_family_id": family_id,
            "operation_expressions": deepcopy(selector.get("operation_expressions") or []),
            "verified_tool_names": deepcopy(selector.get("verified_tool_names") or []),
        },
        "assertion_contract": {
            "kind": "predicate_truth_implies_operation_decision",
            "predicate_id": predicate_id,
            "required_truth_value": legacy_binding["required_truth_value"],
            "expected_operation_decision": legacy_binding[
                "expected_operation_decision"
            ],
            "policy_modality": legacy_binding["policy_modality"],
            "surface_text": factor.get("surface_text"),
            "typed_expression": deepcopy(factor.get("typed_expression")),
        },
        "evidence_contract": {
            "kind": "accepted_operation_factor_lineage",
            "source_span_id": legacy_binding["source_span_id"],
            "source_evidence_id": legacy_binding["source_evidence_id"],
            "runtime_binding": deepcopy(factor.get("runtime_binding")),
        },
        "accepted_artifact_refs": {
            "operation_policy_family_id": family_id,
            "predicate_id": predicate_id,
        },
        "legacy_model_binding": deepcopy(dict(legacy_binding)),
    }


def _operation_factor_target(
    source: Mapping[str, Any], resolved_artifacts: Mapping[str, Any]
) -> dict[str, Any]:
    legacy = resolve_source_spec_target(source, resolved_artifacts)
    documents = _documents(resolved_artifacts)
    status = legacy["resolution_status"]
    if status == "resolved_unique":
        binding = _operation_binding(legacy["model_binding"], documents)
        return make_resolved_evaluation_target(
            source_spec_record=source,
            resolution_status="resolved_unique",
            resolver_id=OPERATION_FACTOR_RESOLVER_ID,
            evaluation_binding=binding,
            resolution_evidence=legacy["resolution_evidence"],
            diagnostics=legacy["diagnostics"],
        )
    candidates = [
        _operation_binding(item, documents)
        for item in legacy.get("candidate_bindings") or []
    ]
    return make_resolved_evaluation_target(
        source_spec_record=source,
        resolution_status=status,
        resolver_id=OPERATION_FACTOR_RESOLVER_ID,
        candidate_bindings=candidates,
        resolution_evidence=legacy["resolution_evidence"],
        diagnostics=legacy["diagnostics"],
    )


_SCOPE_TOKEN_ALIASES = {
    "agents": "agent",
    "baggages": "baggage",
    "books": "book",
    "booked": "book",
    "flights": "flight",
    "passengers": "passenger",
    "reservations": "reservation",
    "updates": "update",
}


def _scope_tokens(value: str) -> frozenset[str]:
    return frozenset(
        _SCOPE_TOKEN_ALIASES.get(token, token)
        for token in re.findall(
            r"[a-z0-9]+", unicodedata.normalize("NFKC", value).casefold()
        )
        if token not in {"a", "an", "the", "to", "tool"}
    )


def _accepted_tool_endpoints(
    documents: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    catalog = documents.get("tool_observable_endpoints") or {}
    agent_spec = documents.get("agent_tool_schemas") or {}
    agent_tools = {
        str(item.get("name")): item
        for item in agent_spec.get("tools") or []
        if isinstance(item, Mapping) and item.get("name")
    }
    endpoints = []
    for tool in catalog.get("tools") or []:
        if not isinstance(tool, Mapping) or not tool.get("tool_name"):
            continue
        for endpoint in tool.get("observable_endpoints") or []:
            if not isinstance(endpoint, Mapping):
                continue
            if endpoint.get("source_kind") != "tool_argument":
                continue
            item = deepcopy(dict(endpoint))
            item.setdefault("tool_name", tool.get("tool_name"))
            endpoints.append(item)
    return endpoints, agent_tools


def _endpoint_lineage_errors(
    endpoint: Mapping[str, Any],
    agent_tools: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    tool_name = str(endpoint.get("tool_name") or "")
    source_path = str(endpoint.get("source_path") or "")
    tool = agent_tools.get(tool_name)
    if tool is None:
        return ["tool_missing_from_accepted_agent_spec"]
    parameter = _mapping(tool.get("parameters")).get(source_path)
    if not isinstance(parameter, Mapping):
        return ["argument_missing_from_accepted_agent_spec"]
    errors = []
    if parameter.get("type") != endpoint.get("type"):
        errors.append("argument_type_disagrees_across_accepted_artifacts")
    required = source_path in (tool.get("required_parameters") or [])
    if bool(endpoint.get("required")) != required:
        errors.append("argument_requiredness_disagrees_across_accepted_artifacts")
    endpoint_values = list(endpoint.get("allowed_values") or [])
    schema_values = list(parameter.get("enum") or [])
    if endpoint_values != schema_values:
        errors.append("argument_enum_disagrees_across_accepted_artifacts")
    if _normalized(str(parameter.get("description") or "")) != _normalized(
        str(endpoint.get("description") or "")
    ):
        errors.append("argument_description_disagrees_across_accepted_artifacts")
    return errors


def _tool_argument_source_matches(
    source: Mapping[str, Any],
    endpoints: list[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    metadata = source_record_metadata(source)
    quotes = [
        part.strip()
        for part in str(metadata.get("evidence_quote") or "").split("|")
        if part.strip()
    ]
    source_assertion = " ".join(
        str(source.get(field) or "") for field in ("rule_text", "then")
    )
    assertion_tokens = _scope_tokens(source_assertion)
    matches = []
    for endpoint in endpoints:
        description = str(endpoint.get("description") or "")
        description_normalized = _normalized(description)
        bases = []
        for quote in quotes:
            quote_normalized = _normalized(quote)
            if quote_normalized == description_normalized:
                bases.append("normalized_evidence_exact")
            elif description_normalized and description_normalized in quote_normalized:
                bases.append("accepted_description_in_labeled_evidence")
        if not bases:
            continue
        source_path = str(endpoint.get("source_path") or "")
        argument_token = source_path.rsplit(".", 1)[-1].replace("[]", "")
        argument_tokens = _scope_tokens(argument_token)
        if argument_tokens and not argument_tokens.issubset(assertion_tokens):
            continue
        basis = (
            "normalized_evidence_exact"
            if "normalized_evidence_exact" in bases
            else "accepted_description_in_labeled_evidence"
        )
        matches.append((basis, endpoint))

    if not matches:
        return []
    all_source_text = " ".join(
        str(source.get(field) or "")
        for field in ("rule_text", "given", "when", "then")
    )
    explicit_tools = {
        str(endpoint.get("tool_name"))
        for _, endpoint in matches
        if re.search(
            rf"(?<![a-z0-9_]){re.escape(str(endpoint.get('tool_name')))}(?![a-z0-9_])",
            all_source_text,
            flags=re.IGNORECASE,
        )
    }
    if explicit_tools:
        matches = [
            item for item in matches if item[1].get("tool_name") in explicit_tools
        ]
    else:
        source_scope = _scope_tokens(all_source_text)
        scope_tools = {
            str(endpoint.get("tool_name"))
            for _, endpoint in matches
            if _scope_tokens(str(endpoint.get("tool_name") or "")).issubset(
                source_scope
            )
        }
        if scope_tools:
            matches = [
                item for item in matches if item[1].get("tool_name") in scope_tools
            ]
    deduplicated = {}
    for basis, endpoint in matches:
        deduplicated.setdefault(str(endpoint.get("endpoint_id")), (basis, endpoint))
    return [deduplicated[key] for key in sorted(deduplicated)]


def _tool_argument_binding(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    endpoint_id = str(endpoint.get("endpoint_id"))
    return {
        "binding_id": f"tool-argument::{endpoint_id}",
        "target_archetype": "tool_argument",
        "evaluation_subject": {
            "kind": "tool_invocation_argument",
            "tool_name": endpoint.get("tool_name"),
            "observable_endpoint_id": endpoint_id,
            "argument_path": endpoint.get("source_path"),
        },
        "assertion_contract": {
            "kind": "accepted_tool_argument_schema",
            "required": bool(endpoint.get("required")),
            "value_type": endpoint.get("type"),
            "allowed_values": deepcopy(endpoint.get("allowed_values") or []),
            "accepted_description": endpoint.get("description"),
        },
        "evidence_contract": {
            "kind": "accepted_tool_observable_endpoint",
            "observable_endpoint_id": endpoint_id,
            "observation_channel": "tool_call.arguments",
            "runtime_observable": True,
        },
        "accepted_artifact_refs": {
            "tool_name": endpoint.get("tool_name"),
            "observable_endpoint_id": endpoint_id,
        },
    }


def _tool_argument_set_binding(
    endpoints: list[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(endpoints, key=lambda item: str(item.get("endpoint_id")))
    endpoint_ids = [str(item.get("endpoint_id")) for item in ordered]
    tool_names = sorted({str(item.get("tool_name")) for item in ordered})
    return {
        "binding_id": "tool-argument-set::" + "::".join(endpoint_ids),
        "target_archetype": "tool_argument",
        "evaluation_subject": {
            "kind": "tool_invocation_argument_set",
            "tool_names": tool_names,
            "observable_endpoint_ids": endpoint_ids,
            "argument_paths": [item.get("source_path") for item in ordered],
        },
        "assertion_contract": {
            "kind": "source_argument_relation_not_formalized",
            "accepted_argument_schemas": [
                {
                    "observable_endpoint_id": item.get("endpoint_id"),
                    "required": bool(item.get("required")),
                    "value_type": item.get("type"),
                    "allowed_values": deepcopy(item.get("allowed_values") or []),
                }
                for item in ordered
            ],
        },
        "evidence_contract": {
            "kind": "accepted_tool_observable_endpoint_set",
            "observable_endpoint_ids": endpoint_ids,
            "observation_channel": "tool_call.arguments",
            "runtime_observable": True,
            "relation_runtime_predicate_available": False,
        },
        "accepted_artifact_refs": {
            "tool_names": tool_names,
            "observable_endpoint_ids": endpoint_ids,
        },
    }


def _tool_argument_target(
    source: Mapping[str, Any], resolved_artifacts: Mapping[str, Any]
) -> dict[str, Any]:
    documents = _documents(resolved_artifacts)
    endpoints, agent_tools = _accepted_tool_endpoints(documents)
    matches = _tool_argument_source_matches(source, endpoints)
    identity = _artifact_identity_evidence(resolved_artifacts)
    if not matches:
        return make_resolved_evaluation_target(
            source_spec_record=source,
            resolution_status="unresolved",
            resolver_id=TOOL_ARGUMENT_RESOLVER_ID,
            diagnostics=[{"code": "no_exact_accepted_tool_argument_match"}],
        )

    lineage_failures = []
    valid_matches = []
    for basis, endpoint in matches:
        errors = _endpoint_lineage_errors(endpoint, agent_tools)
        if errors:
            lineage_failures.append(
                {
                    "observable_endpoint_id": endpoint.get("endpoint_id"),
                    "reason_codes": errors,
                }
            )
        else:
            valid_matches.append((basis, endpoint))
    if lineage_failures:
        return make_resolved_evaluation_target(
            source_spec_record=source,
            resolution_status="lineage_gap",
            resolver_id=TOOL_ARGUMENT_RESOLVER_ID,
            resolution_evidence=[identity],
            diagnostics=[
                {
                    "code": "tool_argument_accepted_lineage_not_closed",
                    "failures": lineage_failures,
                }
            ],
        )

    bindings = [_tool_argument_binding(endpoint) for _, endpoint in valid_matches]
    evidence = [identity] + [
        {
            "evidence_kind": "source_to_tool_argument_match",
            "match_basis": basis,
            "observable_endpoint_id": endpoint.get("endpoint_id"),
        }
        for basis, endpoint in valid_matches
    ]
    if len(bindings) > 1:
        matched_tools = {
            str(endpoint.get("tool_name")) for _, endpoint in valid_matches
        }
        if len(matched_tools) == 1:
            composite = _tool_argument_set_binding(
                [endpoint for _, endpoint in valid_matches]
            )
            return make_resolved_evaluation_target(
                source_spec_record=source,
                resolution_status="needs_adjudication",
                resolver_id=TOOL_ARGUMENT_RESOLVER_ID,
                candidate_bindings=[composite],
                resolution_evidence=evidence,
                diagnostics=[
                    {
                        "code": "multi_argument_relation_not_formalized_by_tool_schema",
                        "observable_endpoint_ids": [
                            endpoint.get("endpoint_id")
                            for _, endpoint in valid_matches
                        ],
                    }
                ],
            )
        return make_resolved_evaluation_target(
            source_spec_record=source,
            resolution_status="ambiguous",
            resolver_id=TOOL_ARGUMENT_RESOLVER_ID,
            candidate_bindings=bindings,
            resolution_evidence=evidence,
            diagnostics=[
                {
                    "code": "multiple_exact_tool_argument_matches",
                    "match_count": len(bindings),
                }
            ],
        )

    provisional = make_resolved_evaluation_target(
        source_spec_record=source,
        resolution_status="resolved_unique",
        resolver_id=TOOL_ARGUMENT_RESOLVER_ID,
        evaluation_binding=bindings[0],
        resolution_evidence=evidence,
    )
    from .resolution_quality import assess_resolution_quality

    quality = assess_resolution_quality(source, provisional, resolved_artifacts)
    quality_evidence = evidence + [
        {
            "evidence_kind": "resolution_quality_gate",
            "verdict": quality["verdict"],
            "quality_fingerprint": quality["resolution_quality_fingerprint"],
            "checks": deepcopy(quality["checks"]),
        }
    ]
    if quality["verdict"] == "verified":
        return make_resolved_evaluation_target(
            source_spec_record=source,
            resolution_status="resolved_unique",
            resolver_id=TOOL_ARGUMENT_RESOLVER_ID,
            evaluation_binding=bindings[0],
            resolution_evidence=quality_evidence,
        )
    if quality["verdict"] == "needs_review":
        return make_resolved_evaluation_target(
            source_spec_record=source,
            resolution_status="needs_adjudication",
            resolver_id=TOOL_ARGUMENT_RESOLVER_ID,
            candidate_bindings=bindings,
            resolution_evidence=quality_evidence,
            diagnostics=[
                {
                    "code": "tool_argument_assertion_not_mechanically_closed",
                    "nonpass_reason_codes": [
                        item["reason_code"]
                        for item in quality["checks"]
                        if item["status"] != "pass"
                    ],
                }
            ],
        )
    return make_resolved_evaluation_target(
        source_spec_record=source,
        resolution_status="semantic_conflict",
        resolver_id=TOOL_ARGUMENT_RESOLVER_ID,
        resolution_evidence=quality_evidence,
        diagnostics=[
            {
                "code": "tool_argument_candidate_rejected_by_quality_gate",
                "rejected_binding_id": bindings[0]["binding_id"],
                "failure_reason_codes": [
                    item["reason_code"]
                    for item in quality["checks"]
                    if item["status"] == "fail"
                ],
            }
        ],
    )


def _source_texts(source: Mapping[str, Any]) -> list[tuple[str, str]]:
    result = [
        ("then", str(source["then"])),
        ("rule_text", str(source["rule_text"])),
        ("when", str(source["when"])),
        ("given", str(source["given"])),
    ]
    quote = source_record_metadata(source).get("evidence_quote")
    if isinstance(quote, str):
        result.extend(
            ("evidence_quote", part.strip())
            for part in quote.split("|")
            if part.strip()
        )
    return result


def _path_surfaces(contract: Mapping[str, Any]) -> list[tuple[str, str]]:
    trace = _mapping(contract.get("trace_requirements"))
    monitors = _mapping(contract.get("oracle_monitors"))
    result: list[tuple[str, str]] = []
    for item in trace.get("action_event_requirements") or []:
        if isinstance(item, Mapping) and isinstance(item.get("action_text"), str):
            result.append(("action_event_requirement", str(item["action_text"])))
    for collection, label in (
        (monitors.get("action_monitors") or [], "action_monitor"),
        (monitors.get("relation_monitors") or [], "relation_monitor"),
    ):
        for item in collection:
            if not isinstance(item, Mapping):
                continue
            for key in ("action_text", "requirement_text"):
                if isinstance(item.get(key), str):
                    result.append((label, str(item[key])))
    return list(dict.fromkeys(result))


def _path_binding(
    contract: Mapping[str, Any], family: Mapping[str, Any]
) -> dict[str, Any]:
    selector = _mapping(family.get("operation_selector"))
    trace = deepcopy(dict(_mapping(contract.get("trace_requirements"))))
    monitors = deepcopy(dict(_mapping(contract.get("oracle_monitors"))))
    has_tool = bool(trace.get("tool_invocation_requirements"))
    has_actions = bool(trace.get("action_event_requirements"))
    archetype = "interaction_order" if has_actions and has_tool else "path_requirement"
    return {
        "binding_id": f"path-contract::{contract.get('path_match_contract_id')}",
        "target_archetype": archetype,
        "evaluation_subject": {
            "kind": "interaction_trace" if has_actions else "path_behavior",
            "operation_policy_family_id": contract.get(
                "owner_operation_policy_family_id"
            ),
            "operation_expressions": deepcopy(selector.get("operation_expressions") or []),
            "verified_tool_names": deepcopy(selector.get("verified_tool_names") or []),
        },
        "assertion_contract": {
            "kind": "accepted_path_requirements",
            "activation_monitor": deepcopy(contract.get("activation_monitor")),
            "trace_requirements": trace,
        },
        "evidence_contract": {
            "kind": "accepted_path_oracle_monitors",
            "oracle_monitors": monitors,
            "requires_semantic_judgment": bool(
                contract.get("requires_semantic_judgment")
            ),
        },
        "accepted_artifact_refs": {
            "path_match_contract_id": contract.get("path_match_contract_id"),
            "path_test_specification_id": contract.get(
                "path_test_specification_id"
            ),
            "focal_semantic_path_id": contract.get("focal_semantic_path_id"),
            "operation_policy_family_id": contract.get(
                "owner_operation_policy_family_id"
            ),
        },
    }


def _path_contract_target(
    source: Mapping[str, Any], resolved_artifacts: Mapping[str, Any]
) -> dict[str, Any]:
    documents = _documents(resolved_artifacts)
    execution = documents.get("execution_matching_contracts") or {}
    families = {
        item.get("operation_policy_family_id"): item
        for item in execution.get("operation_family_contracts") or []
        if isinstance(item, Mapping)
    }
    matches: list[tuple[int, str, str, str, Mapping[str, Any], Mapping[str, Any]]] = []
    for contract in execution.get("path_match_contracts") or []:
        if not isinstance(contract, Mapping):
            continue
        family = families.get(contract.get("owner_operation_policy_family_id"))
        if not isinstance(family, Mapping):
            continue
        for source_field, source_text in _source_texts(source):
            for accepted_field, accepted_text in _path_surfaces(contract):
                basis = _match_basis(source_text, accepted_text)
                if basis is None:
                    continue
                rank, name = basis
                matches.append(
                    (
                        rank,
                        str(contract.get("path_match_contract_id")),
                        source_field,
                        f"{accepted_field}:{name}",
                        contract,
                        family,
                    )
                )
    if not matches:
        return make_resolved_evaluation_target(
            source_spec_record=source,
            resolution_status="unresolved",
            resolver_id=PATH_CONTRACT_RESOLVER_ID,
            diagnostics=[{"code": "no_accepted_path_contract_match"}],
        )
    best_rank = min(item[0] for item in matches)
    best = [item for item in matches if item[0] == best_rank]
    by_id: dict[str, tuple[int, str, str, str, Mapping[str, Any], Mapping[str, Any]]] = {}
    for item in sorted(best, key=lambda value: (value[1], value[2], value[3])):
        by_id.setdefault(item[1], item)
    bindings = [_path_binding(item[4], item[5]) for item in by_id.values()]
    evidence = [_artifact_identity_evidence(resolved_artifacts)] + [
        {
            "evidence_kind": "source_to_path_contract_match",
            "source_field": item[2],
            "match_basis": item[3],
            "path_match_contract_id": item[1],
        }
        for item in by_id.values()
    ]
    if len(bindings) == 1:
        provisional = make_resolved_evaluation_target(
            source_spec_record=source,
            resolution_status="resolved_unique",
            resolver_id=PATH_CONTRACT_RESOLVER_ID,
            evaluation_binding=bindings[0],
            resolution_evidence=evidence,
        )
        # Imported here to keep the quality module usable as an independent
        # auditor while avoiding a module-level resolver/quality cycle.
        from .resolution_quality import assess_resolution_quality

        quality = assess_resolution_quality(source, provisional, resolved_artifacts)
        quality_evidence = evidence + [
            {
                "evidence_kind": "resolution_quality_gate",
                "verdict": quality["verdict"],
                "quality_fingerprint": quality[
                    "resolution_quality_fingerprint"
                ],
                "checks": deepcopy(quality["checks"]),
            }
        ]
        if quality["verdict"] == "verified":
            return make_resolved_evaluation_target(
                source_spec_record=source,
                resolution_status="resolved_unique",
                resolver_id=PATH_CONTRACT_RESOLVER_ID,
                evaluation_binding=bindings[0],
                resolution_evidence=quality_evidence,
            )
        if quality["verdict"] == "needs_review":
            return make_resolved_evaluation_target(
                source_spec_record=source,
                resolution_status="needs_adjudication",
                resolver_id=PATH_CONTRACT_RESOLVER_ID,
                candidate_bindings=bindings,
                resolution_evidence=quality_evidence,
                diagnostics=[
                    {
                        "code": "unique_lexical_candidate_needs_adjudication",
                        "nonpass_reason_codes": [
                            check["reason_code"]
                            for check in quality["checks"]
                            if check["status"] != "pass"
                        ],
                    }
                ],
            )
        return make_resolved_evaluation_target(
            source_spec_record=source,
            resolution_status="unresolved",
            resolver_id=PATH_CONTRACT_RESOLVER_ID,
            resolution_evidence=quality_evidence,
            diagnostics=[
                {
                    "code": "unique_lexical_candidate_rejected_by_quality_gate",
                    "rejected_binding_id": bindings[0]["binding_id"],
                    "failure_reason_codes": [
                        check["reason_code"]
                        for check in quality["checks"]
                        if check["status"] == "fail"
                    ],
                }
            ],
        )
    return make_resolved_evaluation_target(
        source_spec_record=source,
        resolution_status="ambiguous",
        resolver_id=PATH_CONTRACT_RESOLVER_ID,
        candidate_bindings=bindings,
        resolution_evidence=evidence,
        diagnostics=[
            {
                "code": "multiple_accepted_path_contract_matches",
                "match_count": len(bindings),
            }
        ],
    )


def _source_grounded_target(
    source: Mapping[str, Any], resolved_artifacts: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = source_record_metadata(source)
    kind = str(metadata.get("kind") or "").upper()
    archetype = {
        "ARG": "tool_argument",
        "ORDER": "interaction_order",
        "STATE": "state_transition",
        "NORM": "response_content",
    }.get(kind, "source_assertion")
    binding = {
        "binding_id": f"reviewed-source::{source['branch_id']}",
        "target_archetype": archetype,
        "evaluation_subject": {
            "kind": "unlowered_reviewed_source_rule",
            "source_kind": kind or None,
            "source_origin": metadata.get("origin"),
        },
        "assertion_contract": {
            "kind": "source_text_not_yet_lowered",
            "given": source["given"],
            "when": source["when"],
            "then": source["then"],
            "deontic": source["deontic"],
        },
        "evidence_contract": {
            "kind": "reviewed_source_record",
            "review_ok": _mapping(metadata.get("review")).get("status"),
            "source_record_fingerprint": source["source_record_fingerprint"],
        },
        "accepted_artifact_refs": {},
    }
    return make_resolved_evaluation_target(
        source_spec_record=source,
        resolution_status="source_grounded",
        resolver_id=SOURCE_FALLBACK_RESOLVER_ID,
        evaluation_binding=binding,
        resolution_evidence=[_artifact_identity_evidence(resolved_artifacts)],
        diagnostics=[
            {
                "code": "accepted_runtime_contract_not_resolved",
                "target_archetype_hint": archetype,
            }
        ],
    )


def _attach_registry_trace(
    target: Mapping[str, Any],
    source: Mapping[str, Any],
    attempts: list[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence = deepcopy(target.get("resolution_evidence") or [])
    evidence.append(
        {
            "evidence_kind": "resolver_registry_trace",
            "registry_version": EVALUATION_RESOLVER_REGISTRY_VERSION,
            "attempts": deepcopy(attempts),
        }
    )
    return make_resolved_evaluation_target(
        source_spec_record=source,
        resolution_status=str(target["resolution_status"]),
        resolver_id=str(target["resolver_id"]),
        evaluation_binding=target.get("evaluation_binding"),
        candidate_bindings=target.get("candidate_bindings") or [],
        resolution_evidence=evidence,
        diagnostics=target.get("diagnostics") or [],
    )


DEFAULT_EVALUATION_RESOLVERS = (
    EvaluationResolverRegistration(
        resolver_id=OPERATION_FACTOR_RESOLVER_ID,
        source_kinds=("STATE",),
        source_origins=(),
        target_archetypes=("operation_decision",),
        required_artifact_capabilities=(
            CAP_OPERATION_POLICY_CLOSURES,
            CAP_COVERAGE_MATCH_MANIFEST,
            CAP_DECISION_CONFIGURATION_MODEL,
            CAP_EXECUTION_MATCHING_CONTRACTS,
        ),
        resolve=_operation_factor_target,
    ),
    EvaluationResolverRegistration(
        resolver_id=TOOL_ARGUMENT_RESOLVER_ID,
        source_kinds=("ARG",),
        source_origins=("schema",),
        target_archetypes=("tool_argument",),
        required_artifact_capabilities=(
            CAP_AGENT_TOOL_SCHEMAS,
            CAP_TOOL_OBSERVABLE_ENDPOINTS,
        ),
        resolve=_tool_argument_target,
    ),
    EvaluationResolverRegistration(
        resolver_id=PATH_CONTRACT_RESOLVER_ID,
        source_kinds=("ARG", "NORM", "ORDER", "STATE"),
        source_origins=(),
        target_archetypes=("interaction_order", "path_requirement"),
        required_artifact_capabilities=(CAP_EXECUTION_MATCHING_CONTRACTS,),
        resolve=_path_contract_target,
    ),
    EvaluationResolverRegistration(
        resolver_id=SOURCE_FALLBACK_RESOLVER_ID,
        source_kinds=(),
        source_origins=(),
        target_archetypes=(
            "tool_argument",
            "interaction_order",
            "state_transition",
            "response_content",
            "source_assertion",
        ),
        required_artifact_capabilities=(),
        resolve=_source_grounded_target,
    ),
)


def resolve_evaluation_target(
    source_spec_record: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
    *,
    resolver_registry: tuple[EvaluationResolverRegistration, ...] = (
        DEFAULT_EVALUATION_RESOLVERS
    ),
) -> dict[str, Any]:
    """Resolve with explicit precedence and a source-grounded final state.

    A closed operation-factor lineage remains authoritative.  Ambiguity,
    semantic conflict and lineage gaps are never bypassed by a weaker resolver.
    Only a genuine no-match proceeds to accepted path contracts and then to a
    reviewed-source target that is deliberately marked as not runtime-lowered.
    """

    source = validate_source_spec_record(source_spec_record)
    if not resolver_registry:
        raise ValueError("evaluation resolver registry must not be empty")
    identifiers = [entry.resolver_id for entry in resolver_registry]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("evaluation resolver IDs must be unique")
    for entry in resolver_registry:
        if not entry.resolver_id:
            raise ValueError("evaluation resolver ID must be non-empty")
        if not entry.target_archetypes:
            raise ValueError(
                f"resolver {entry.resolver_id!r} must declare target archetypes"
            )
        unsupported = sorted(set(entry.target_archetypes) - TARGET_ARCHETYPES)
        if unsupported:
            raise ValueError(
                f"resolver {entry.resolver_id!r} declares unsupported target archetypes: "
                f"{unsupported!r}"
            )
        if len(set(entry.required_artifact_capabilities)) != len(
            entry.required_artifact_capabilities
        ):
            raise ValueError(
                f"resolver {entry.resolver_id!r} repeats required capabilities"
            )

    attempts: list[dict[str, Any]] = []
    capability_barriers: list[dict[str, Any]] = []
    for registration in resolver_registry:
        if not registration.applies_to(source):
            attempts.append(
                {
                    "resolver_id": registration.resolver_id,
                    "status": "not_applicable",
                    "declared_source_kinds": list(registration.source_kinds),
                    "declared_source_origins": list(registration.source_origins),
                    "declared_target_archetypes": list(
                        registration.target_archetypes
                    ),
                }
            )
            continue
        readiness = registration.capability_readiness(resolved_artifacts)
        unavailable = [
            item for item in readiness if item["status"] != "available_unique"
        ]
        if unavailable:
            barrier = {
                "resolver_id": registration.resolver_id,
                "required_artifact_capabilities": readiness,
            }
            capability_barriers.append(barrier)
            attempts.append(
                {
                    "resolver_id": registration.resolver_id,
                    "status": "capability_unavailable",
                    "declared_target_archetypes": list(
                        registration.target_archetypes
                    ),
                    "required_artifact_capabilities": readiness,
                }
            )
            continue
        target = registration.resolve(source, resolved_artifacts)
        if target["resolver_id"] != registration.resolver_id:
            raise ValueError(
                "resolver registry identity mismatch: "
                f"registered {registration.resolver_id!r}, returned "
                f"{target['resolver_id']!r}"
            )
        returned_bindings = []
        if target.get("evaluation_binding") is not None:
            returned_bindings.append(target["evaluation_binding"])
        returned_bindings.extend(target.get("candidate_bindings") or [])
        returned_archetypes = {
            str(binding.get("target_archetype"))
            for binding in returned_bindings
        }
        undeclared = sorted(
            returned_archetypes - set(registration.target_archetypes)
        )
        if undeclared:
            raise ValueError(
                f"resolver {registration.resolver_id!r} returned undeclared target "
                f"archetypes: {undeclared!r}"
            )
        attempts.append(
            {
                "resolver_id": registration.resolver_id,
                "status": target["resolution_status"],
                "declared_target_archetypes": list(
                    registration.target_archetypes
                ),
                "required_artifact_capabilities": readiness,
                "diagnostics": deepcopy(target.get("diagnostics") or []),
            }
        )
        if target["resolution_status"] != "unresolved":
            if (
                capability_barriers
                and target["resolution_status"] != "source_grounded"
            ):
                return _attach_registry_trace(
                    make_resolved_evaluation_target(
                        source_spec_record=source,
                        resolution_status="lineage_gap",
                        resolver_id=str(capability_barriers[0]["resolver_id"]),
                        resolution_evidence=[
                            _artifact_identity_evidence(resolved_artifacts)
                        ],
                        diagnostics=[
                            {
                                "code": "higher_priority_resolver_capability_unavailable",
                                "barriers": deepcopy(capability_barriers),
                                "suppressed_lower_priority_result": {
                                    "resolver_id": target["resolver_id"],
                                    "resolution_status": target[
                                        "resolution_status"
                                    ],
                                },
                            }
                        ],
                    ),
                    source,
                    attempts,
                )
            return _attach_registry_trace(target, source, attempts)
    raise ValueError(
        "evaluation resolver registry exhausted without a terminal result; "
        "register a source-grounding fallback"
    )


__all__ = [
    "DEFAULT_EVALUATION_RESOLVERS",
    "EVALUATION_RESOLVER_REGISTRY_VERSION",
    "EvaluationResolverRegistration",
    "OPERATION_FACTOR_RESOLVER_ID",
    "PATH_CONTRACT_RESOLVER_ID",
    "SOURCE_FALLBACK_RESOLVER_ID",
    "TOOL_ARGUMENT_RESOLVER_ID",
    "resolve_evaluation_target",
]
