"""Deterministic acceptance checks for resolver outputs.

The resolver answers "which accepted contract is the best candidate?".  This
module answers the stricter question "is that candidate justified strongly
enough to be called a unique runtime resolution?".  It intentionally contains
no LLM fallback: an unprovable match is reported for review rather than being
silently accepted.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import (
    CAP_EXECUTION_MATCHING_CONTRACTS,
    CAP_TOOL_OBSERVABLE_ENDPOINTS,
    artifact_capability_status,
    artifact_document_for_capability,
)
from .artifacts import content_sha256
from .evaluation_contracts import validate_resolved_evaluation_target
from .evaluation_resolver import (
    OPERATION_FACTOR_RESOLVER_ID,
    PATH_CONTRACT_RESOLVER_ID,
    TOOL_ARGUMENT_RESOLVER_ID,
    _informative,
    _match_basis,
    _path_surfaces,
)
from .source_contracts import validate_source_spec_record


RESOLUTION_QUALITY_SCHEMA_VERSION = "agentspectesting.resolution-quality/v0.1"
RESOLUTION_QUALITY_SET_SCHEMA_VERSION = (
    "agentspectesting.resolution-quality-set/v0.1"
)

QUALITY_VERDICTS = frozenset({"verified", "needs_review", "rejected"})

_ACTION_ALIASES = {
    "booking": "book",
    "booked": "book",
    "modify": "update",
    "modified": "update",
    "modifying": "update",
    "change": "update",
    "changed": "update",
    "changing": "update",
    "cancellation": "cancel",
    "cancelled": "cancel",
    "canceled": "cancel",
    "refunded": "refund",
    "refundable": "refund",
}

_OPERATION_ACTIONS = frozenset(
    {
        "add",
        "book",
        "cancel",
        "compensate",
        "create",
        "delete",
        "exchange",
        "locate",
        "refund",
        "remove",
        "search",
        "update",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _check(
    check_id: str,
    status: str,
    *,
    reason_code: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"pass", "review", "fail"}:
        raise ValueError(f"unsupported quality-check status: {status!r}")
    return {
        "check_id": check_id,
        "status": status,
        "reason_code": reason_code,
        "evidence": deepcopy(dict(evidence or {})),
    }


def _operation_tokens(value: str) -> frozenset[str]:
    return frozenset(_ACTION_ALIASES.get(token, token) for token in _informative(value))


def _operation_scope_check(
    source: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    subject = _mapping(binding.get("evaluation_subject"))
    accepted_texts = [
        str(value) for value in subject.get("operation_expressions") or []
    ]
    tools = [str(value) for value in subject.get("verified_tool_names") or []]
    accepted_tokens = set()
    for value in accepted_texts:
        accepted_tokens.update(_operation_tokens(value))
    for tool in tools:
        accepted_tokens.update(_operation_tokens(tool.replace("_", " ")))
    accepted_actions = accepted_tokens & _OPERATION_ACTIONS

    source_text = " ".join((str(source["when"]), str(source["then"])))
    source_tokens = set(_operation_tokens(source_text))
    source_actions = source_tokens & _OPERATION_ACTIONS
    overlap = sorted(accepted_actions & source_actions)
    evidence = {
        "accepted_operation_actions": sorted(accepted_actions),
        "source_operation_actions": sorted(source_actions),
        "overlap": overlap,
        "operation_expressions": accepted_texts,
        "verified_tool_names": tools,
    }
    if overlap:
        return _check(
            "operation_scope",
            "pass",
            reason_code="source_and_contract_share_operation_action",
            evidence=evidence,
        )
    if accepted_actions and source_actions:
        return _check(
            "operation_scope",
            "fail",
            reason_code="source_operation_conflicts_with_contract_family",
            evidence=evidence,
        )
    return _check(
        "operation_scope",
        "review",
        reason_code="operation_scope_not_mechanically_provable",
        evidence=evidence,
    )


def _source_expectation(source: Mapping[str, Any]) -> str:
    then = str(source["then"]).casefold()
    deontic = str(source["deontic"]).casefold()
    if deontic == "prohibition" or any(
        marker in then for marker in ("must not", "cannot", "may not", "prohibited")
    ):
        return "prohibited"
    if deontic == "permission" or any(
        marker in then for marker in (" may ", " can ", "is permitted")
    ):
        return "permitted"
    if deontic == "obligation" or "must" in then or "should" in then:
        return "required"
    return "unknown"


def _contract_expectations(contract: Mapping[str, Any]) -> frozenset[str]:
    monitors = _mapping(contract.get("oracle_monitors"))
    result = set()
    for item in monitors.get("action_monitors") or []:
        expectation = str(_mapping(item).get("policy_expectation") or "")
        if expectation == "must_not_perform":
            result.add("prohibited")
        elif expectation == "must_perform":
            result.add("required")
    for item in monitors.get("relation_monitors") or []:
        relation = _mapping(item)
        expectation = str(relation.get("operation_expectation") or "")
        if expectation == "must_not_perform":
            result.add("prohibited")
        elif expectation == "may_perform":
            result.add("permitted")
        modality = str(relation.get("modality") or "")
        if modality == "required":
            result.add("required")
        elif modality == "permitted":
            result.add("permitted")
    return frozenset(result)


def _best_then_match(
    source: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[int, str, str] | None:
    matches = []
    for accepted_field, accepted_text in _path_surfaces(contract):
        basis = _match_basis(str(source["then"]), accepted_text)
        if basis is not None:
            matches.append((basis[0], basis[1], accepted_field))
    return min(matches) if matches else None


def _best_rule_match(
    source: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[int, str, str] | None:
    matches = []
    for accepted_field, accepted_text in _path_surfaces(contract):
        basis = _match_basis(str(source["rule_text"]), accepted_text)
        if basis is not None:
            matches.append((basis[0], basis[1], accepted_field))
    return min(matches) if matches else None


def _compound_scope_narrowed(
    source: Mapping[str, Any], contract: Mapping[str, Any]
) -> bool:
    """Detect a split source branch mapped back to an unsplit compound rule."""

    then_text = str(source["then"])
    for _field, accepted_text in _path_surfaces(contract):
        compound = accepted_text.count(",") >= 1 and " and " in accepted_text.casefold()
        repeated_limit = accepted_text.casefold().count("at most") >= 2
        if not (compound or repeated_limit):
            continue
        if _match_basis(then_text, accepted_text) is None:
            return True
    return False


def _assertion_check(
    source: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    then_match = _best_then_match(source, contract)
    rule_match = _best_rule_match(source, contract)
    source_expectation = _source_expectation(source)
    accepted_expectations = _contract_expectations(contract)
    evidence = {
        "then_match_basis": then_match[1] if then_match else None,
        "rule_match_basis": rule_match[1] if rule_match else None,
        "source_expectation": source_expectation,
        "contract_expectations": sorted(accepted_expectations),
    }
    if _compound_scope_narrowed(source, contract):
        return _check(
            "assertion_scope_and_branch",
            "fail",
            reason_code="source_branch_is_narrower_than_compound_contract",
            evidence=evidence,
        )
    if (
        source_expectation != "unknown"
        and accepted_expectations
        and source_expectation not in accepted_expectations
    ):
        return _check(
            "assertion_scope_and_branch",
            "fail",
            reason_code="source_branch_expectation_conflicts_with_contract",
            evidence=evidence,
        )
    if then_match is not None:
        return _check(
            "assertion_scope_and_branch",
            "pass",
            reason_code="source_assertion_directly_matches_contract_surface",
            evidence=evidence,
        )
    if rule_match is not None and source_expectation in accepted_expectations:
        return _check(
            "assertion_scope_and_branch",
            "pass",
            reason_code="exact_rule_match_with_compatible_branch_expectation",
            evidence=evidence,
        )
    return _check(
        "assertion_scope_and_branch",
        "review",
        reason_code="source_assertion_equivalence_not_mechanically_provable",
        evidence=evidence,
    )


def _runtime_oracle_check(contract: Mapping[str, Any]) -> dict[str, Any]:
    monitors = _mapping(contract.get("oracle_monitors"))
    relation_monitors = [
        _mapping(item) for item in monitors.get("relation_monitors") or []
    ]
    action_monitors = [
        _mapping(item) for item in monitors.get("action_monitors") or []
    ]
    operation_monitors = [
        _mapping(item) for item in monitors.get("operation_decision_monitors") or []
    ]
    statuses = [item.get("runtime_binding_status") for item in relation_monitors]
    evidence = {
        "relation_runtime_binding_statuses": statuses,
        "action_monitor_count": len(action_monitors),
        "operation_decision_monitor_count": len(operation_monitors),
    }
    if any(status == "runtime_target_not_bound" for status in statuses):
        return _check(
            "runtime_oracle_closure",
            "fail",
            reason_code="accepted_contract_explicitly_lacks_runtime_target",
            evidence=evidence,
        )
    if relation_monitors and not operation_monitors and any(
        status not in {"runtime_target_bound"} for status in statuses
    ):
        return _check(
            "runtime_oracle_closure",
            "review",
            reason_code="relation_monitor_has_no_closed_runtime_binding",
            evidence=evidence,
        )
    if relation_monitors or action_monitors or operation_monitors:
        return _check(
            "runtime_oracle_closure",
            "pass",
            reason_code="accepted_contract_has_monitorable_runtime_evidence",
            evidence=evidence,
        )
    return _check(
        "runtime_oracle_closure",
        "fail",
        reason_code="accepted_contract_has_no_runtime_oracle_monitor",
        evidence=evidence,
    )


def _semantic_judgment_check(contract: Mapping[str, Any]) -> dict[str, Any]:
    required = bool(contract.get("requires_semantic_judgment"))
    return _check(
        "semantic_judgment",
        "review" if required else "pass",
        reason_code=(
            "accepted_contract_requires_semantic_adjudication"
            if required
            else "accepted_contract_declares_no_semantic_adjudication"
        ),
        evidence={"requires_semantic_judgment": required},
    )


def _path_contracts(resolved_artifacts: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    status = artifact_capability_status(
        resolved_artifacts,
        CAP_EXECUTION_MATCHING_CONTRACTS,
    )
    if status["status"] != "available_unique":
        return {}
    execution = _mapping(
        artifact_document_for_capability(
            resolved_artifacts,
            CAP_EXECUTION_MATCHING_CONTRACTS,
        )
    )
    return {
        str(item["path_match_contract_id"]): item
        for item in execution.get("path_match_contracts") or []
        if isinstance(item, Mapping) and item.get("path_match_contract_id")
    }


def _tool_argument_endpoints(
    resolved_artifacts: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    status = artifact_capability_status(
        resolved_artifacts,
        CAP_TOOL_OBSERVABLE_ENDPOINTS,
    )
    if status["status"] != "available_unique":
        return {}
    catalog = _mapping(
        artifact_document_for_capability(
            resolved_artifacts,
            CAP_TOOL_OBSERVABLE_ENDPOINTS,
        )
    )
    result = {}
    for tool in catalog.get("tools") or []:
        for endpoint in _mapping(tool).get("observable_endpoints") or []:
            item = _mapping(endpoint)
            endpoint_id = item.get("endpoint_id")
            if item.get("source_kind") == "tool_argument" and endpoint_id:
                result[str(endpoint_id)] = item
    return result


def _tool_argument_scope_check(
    source: Mapping[str, Any], endpoint: Mapping[str, Any]
) -> dict[str, Any]:
    source_path = str(endpoint.get("source_path") or "")
    argument_tokens = _informative(source_path.replace("_", " "))
    source_tokens = _informative(
        " ".join((str(source["rule_text"]), str(source["then"])))
    )
    evidence = {
        "source_path": source_path,
        "argument_tokens": sorted(argument_tokens),
        "source_argument_tokens": sorted(source_tokens),
    }
    if argument_tokens and argument_tokens.issubset(source_tokens):
        return _check(
            "tool_argument_scope",
            "pass",
            reason_code="source_assertion_names_accepted_argument",
            evidence=evidence,
        )
    return _check(
        "tool_argument_scope",
        "fail",
        reason_code="source_assertion_does_not_name_accepted_argument",
        evidence=evidence,
    )


_UNFORMALIZED_ARGUMENT_MARKERS = frozenset(
    {
        "before",
        "after",
        "different",
        "exceed",
        "format",
        "formatted",
        "greater",
        "iata",
        "less",
        "operator",
        "operators",
        "parentheses",
        "positive",
        "valid",
    }
)


def _tool_argument_schema_check(
    source: Mapping[str, Any], endpoint: Mapping[str, Any]
) -> dict[str, Any]:
    rule_tokens = _informative(str(source["rule_text"]).replace("_", " "))
    value_type = str(endpoint.get("type") or "").casefold()
    allowed_values = [str(value) for value in endpoint.get("allowed_values") or []]
    type_proved = bool(value_type) and value_type in rule_tokens
    enum_proved = bool(allowed_values) and all(
        _informative(value.replace("_", " ")).issubset(rule_tokens)
        for value in allowed_values
    )
    unformalized = sorted(rule_tokens & _UNFORMALIZED_ARGUMENT_MARKERS)
    evidence = {
        "accepted_value_type": value_type or None,
        "accepted_allowed_values": allowed_values,
        "source_mentions_type": type_proved,
        "source_mentions_all_allowed_values": enum_proved,
        "unformalized_source_markers": unformalized,
    }
    if unformalized:
        return _check(
            "tool_argument_assertion",
            "review",
            reason_code="source_assertion_contains_constraint_not_formalized_by_tool_schema",
            evidence=evidence,
        )
    if enum_proved:
        return _check(
            "tool_argument_assertion",
            "pass",
            reason_code="source_enum_equals_accepted_argument_enum",
            evidence=evidence,
        )
    if type_proved:
        return _check(
            "tool_argument_assertion",
            "pass",
            reason_code="source_type_equals_accepted_argument_type",
            evidence=evidence,
        )
    return _check(
        "tool_argument_assertion",
        "review",
        reason_code="source_argument_constraint_not_mechanically_equivalent_to_schema",
        evidence=evidence,
    )


def assess_resolution_quality(
    source_spec_record: Mapping[str, Any],
    resolved_evaluation_target: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit one current ``resolved_unique`` result without changing it."""

    source = validate_source_spec_record(source_spec_record)
    target = validate_resolved_evaluation_target(
        resolved_evaluation_target, source_spec_record=source
    )
    if target["resolution_status"] != "resolved_unique":
        raise ValueError("quality acceptance only audits resolved_unique targets")

    resolver_id = target["resolver_id"]
    binding = _mapping(target.get("evaluation_binding"))
    checks = []
    if resolver_id == OPERATION_FACTOR_RESOLVER_ID:
        checks.extend(
            [
                _check(
                    "accepted_lineage_closure",
                    "pass",
                    reason_code="operation_factor_resolver_proved_closed_lineage",
                ),
                _check(
                    "runtime_oracle_closure",
                    "pass",
                    reason_code="factor_contract_contains_runtime_binding",
                ),
            ]
        )
    elif resolver_id == TOOL_ARGUMENT_RESOLVER_ID:
        refs = _mapping(binding.get("accepted_artifact_refs"))
        endpoint_id = str(refs.get("observable_endpoint_id") or "")
        endpoint = _tool_argument_endpoints(resolved_artifacts).get(endpoint_id)
        if endpoint is None:
            checks.append(
                _check(
                    "accepted_contract_identity",
                    "fail",
                    reason_code="selected_tool_argument_endpoint_not_found",
                    evidence={"observable_endpoint_id": endpoint_id},
                )
            )
        else:
            checks.extend(
                [
                    _check(
                        "accepted_contract_identity",
                        "pass",
                        reason_code="selected_tool_argument_endpoint_exists_uniquely",
                        evidence={"observable_endpoint_id": endpoint_id},
                    ),
                    _tool_argument_scope_check(source, endpoint),
                    _tool_argument_schema_check(source, endpoint),
                    _check(
                        "runtime_oracle_closure",
                        "pass",
                        reason_code="tool_argument_is_observable_in_tool_call_trace",
                        evidence={
                            "observation_channel": "tool_call.arguments",
                            "source_path": endpoint.get("source_path"),
                        },
                    ),
                ]
            )
    elif resolver_id == PATH_CONTRACT_RESOLVER_ID:
        refs = _mapping(binding.get("accepted_artifact_refs"))
        path_id = str(refs.get("path_match_contract_id") or "")
        contract = _path_contracts(resolved_artifacts).get(path_id)
        if contract is None:
            checks.append(
                _check(
                    "accepted_contract_identity",
                    "fail",
                    reason_code="selected_path_contract_not_found",
                    evidence={"path_match_contract_id": path_id},
                )
            )
        else:
            checks.extend(
                [
                    _check(
                        "accepted_contract_identity",
                        "pass",
                        reason_code="selected_path_contract_exists_uniquely",
                        evidence={"path_match_contract_id": path_id},
                    ),
                    _operation_scope_check(source, binding),
                    _assertion_check(source, contract),
                    _runtime_oracle_check(contract),
                    _semantic_judgment_check(contract),
                ]
            )
    else:
        checks.append(
            _check(
                "resolver_support",
                "fail",
                reason_code="resolved_unique_came_from_unsupported_resolver",
                evidence={"resolver_id": resolver_id},
            )
        )

    statuses = {item["status"] for item in checks}
    verdict = "rejected" if "fail" in statuses else (
        "needs_review" if "review" in statuses else "verified"
    )
    payload = {
        "schema_version": RESOLUTION_QUALITY_SCHEMA_VERSION,
        "source_record_fingerprint": source["source_record_fingerprint"],
        "source_branch_id": source["branch_id"],
        "resolved_evaluation_target_fingerprint": target[
            "resolved_evaluation_target_fingerprint"
        ],
        "resolver_id": resolver_id,
        "binding_id": binding.get("binding_id"),
        "verdict": verdict,
        "checks": checks,
        "llm_calls": 0,
    }
    payload["resolution_quality_fingerprint"] = content_sha256(payload)
    return payload


def make_resolution_quality_set(
    *,
    items: list[Mapping[str, Any]],
    selection: Mapping[str, Any],
    resolved_unique_population_size: int,
) -> dict[str, Any]:
    normalized_items = [deepcopy(dict(item)) for item in items]
    verdict_counts = Counter(item["quality"]["verdict"] for item in normalized_items)
    reason_counts = Counter(
        check["reason_code"]
        for item in normalized_items
        for check in item["quality"]["checks"]
        if check["status"] != "pass"
    )
    audited_count = len(normalized_items)
    payload = {
        "schema_version": RESOLUTION_QUALITY_SET_SCHEMA_VERSION,
        "selection": deepcopy(dict(selection)),
        "acceptance_policy": {
            "scope": "all_current_resolved_unique_targets",
            "pass_condition": "every audited target has verdict=verified",
            "unprovable_semantics_policy": "needs_review_not_verified",
            "llm_fallback": False,
        },
        "summary": {
            "resolved_unique_population_size": resolved_unique_population_size,
            "audited_count": audited_count,
            "population_complete": audited_count == resolved_unique_population_size,
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "nonpass_reason_counts": dict(sorted(reason_counts.items())),
            "acceptance_passed": (
                audited_count == resolved_unique_population_size
                and audited_count > 0
                and verdict_counts.get("verified", 0) == audited_count
            ),
        },
        "items": normalized_items,
        "llm_calls": 0,
    }
    payload["resolution_quality_set_fingerprint"] = content_sha256(payload)
    return payload


__all__ = [
    "QUALITY_VERDICTS",
    "RESOLUTION_QUALITY_SCHEMA_VERSION",
    "RESOLUTION_QUALITY_SET_SCHEMA_VERSION",
    "assess_resolution_quality",
    "make_resolution_quality_set",
]
