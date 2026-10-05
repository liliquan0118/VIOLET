"""Policy-scoped branch projection onto accepted runtime contracts.

The global lexical resolver searches a small set of runtime surface strings.
This module instead joins through accepted policy and path IDs first, and only
then compares the reviewed branch with the structured condition/assertion slots
of those candidates.  Its report is intentionally diagnostic: a path can be in
lineage while its oracle is still too coarse for the reviewed branch.
"""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import (
    CAP_EXECUTION_MATCHING_CONTRACTS,
    CAP_POLICY_PATH_TEST_SPECIFICATIONS,
    artifact_document_for_capability,
    artifact_identity_view,
    content_sha256,
)
from .evaluation_contracts import (
    make_resolved_evaluation_target,
    validate_resolved_evaluation_target,
)
from .policy_lineage import validate_policy_lineage_report
from .source_contracts import validate_source_spec_record


RUNTIME_PROJECTION_REPORT_SCHEMA_VERSION = (
    "agentspectesting.runtime-projection-report/v0.1"
)
POLICY_RUNTIME_PROJECTION_RESOLVER_ID = (
    "accepted-policy-runtime-projection-resolver/v0.1"
)


class RuntimeProjectionError(ValueError):
    """Raised when a runtime-projection report violates its contract."""


_PHRASES = {
    "basic economy": "basic_economy",
    "business class": "business_class",
    "checked bag": "checked_bag",
    "checked bags": "checked_bag",
    "credit card": "credit_card",
    "gift card": "gift_card",
    "original payment method": "original_payment_method",
    "reservation id": "reservation_id",
    "travel certificate": "travel_certificate",
    "travel insurance": "travel_insurance",
    "trip type": "trip_type",
    "user id": "user_id",
}

_ALIASES = {
    "bags": "bag",
    "baggage": "bag",
    "checked_bags": "checked_bag",
    "booked": "book",
    "booking": "book",
    "canceling": "cancel",
    "cancellation": "cancel",
    "cancellations": "cancel",
    "cancelled": "cancel",
    "canceled": "cancel",
    "cancels": "cancel",
    "confirming": "confirm",
    "cards": "card",
    "certificates": "certificate",
    "changed": "change",
    "changing": "change",
    "days": "day",
    "enables": "provide",
    "enable": "provide",
    "facts": "fact",
    "flights": "flight",
    "grant": "get",
    "grants": "get",
    "hours": "hour",
    "hrs": "hour",
    "modified": "modify",
    "modifying": "modify",
    "modification": "modify",
    "offered": "offer",
    "offering": "offer",
    "passengers": "passenger",
    "prices": "price",
    "provides": "provide",
    "provided": "provide",
    "proceeding": "proceed",
    "reasons": "reason",
    "receives": "get",
    "receive": "get",
    "refunded": "refund",
    "refunding": "refund",
    "reservations": "reservation",
}

_STOP = frozenset(
    "a an the is are was were be been being this that it its of to from by in on "
    "at as and or each all any some their they them with into than agent must may "
    "can should for requests request does not state will".split()
)

_COORDINATE_GENERIC = frozenset(
    {"flight", "for", "has", "member", "passenger", "reservation", "user"}
)

_OPERATION_ALIASES = {
    "booking": "book",
    "booked": "book",
    "cancellation": "cancel",
    "cancelled": "cancel",
    "canceling": "cancel",
    "changed": "modify",
    "changing": "modify",
    "change": "modify",
    "editing": "modify",
    "edit": "modify",
    "modified": "modify",
    "modifying": "modify",
    "modification": "modify",
    "refunding": "refund",
    "refunded": "refund",
    "updates": "modify",
    "updating": "modify",
    "update": "modify",
}

_OPERATION_CUES = frozenset(
    {"book", "cancel", "compensation", "locate", "modify", "offer", "refund"}
)


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    for phrase, replacement in _PHRASES.items():
        text = text.replace(phrase, replacement)
    return " ".join(re.findall(r"[a-z0-9_]+", text))


def _tokens(value: Any) -> frozenset[str]:
    return frozenset(
        _ALIASES.get(token, token)
        for token in _normalized(value).split()
        if token not in _STOP
    )


def _operation_tokens(value: Any) -> frozenset[str]:
    return frozenset(
        _OPERATION_ALIASES.get(token, token)
        for token in _normalized(value).split()
        if _OPERATION_ALIASES.get(token, token) in _OPERATION_CUES
    )


def _assertion_action_tokens(value: Any) -> frozenset[str]:
    normalized = _normalized(value)
    match = re.search(
        r"\b(?:agent|user)\s+(?:must|may|can|should|will)\s+(?:not\s+)?([a-z_]+)",
        normalized,
    )
    if match is None:
        return frozenset()
    action = _OPERATION_ALIASES.get(match.group(1), match.group(1))
    return frozenset({action}) if action in _OPERATION_CUES else frozenset()


def _structured_match(source: str, accepted: str) -> dict[str, Any] | None:
    source_normalized = _normalized(source)
    accepted_normalized = _normalized(accepted)
    if not source_normalized or not accepted_normalized:
        return None
    source_tokens = _tokens(source)
    accepted_tokens = _tokens(accepted)
    if not source_tokens or not accepted_tokens:
        return None
    shared = source_tokens & accepted_tokens
    source_numbers = {token for token in source_tokens if token.isdigit()}
    accepted_numbers = {token for token in accepted_tokens if token.isdigit()}
    if source_numbers and accepted_numbers and source_numbers != accepted_numbers:
        return None
    if source_tokens == accepted_tokens:
        basis = "canonical_structured_token_exact"
        score = 4.5 + 0.1 * min(len(accepted_tokens), 6)
    elif accepted_normalized in source_normalized:
        basis = "accepted_slot_contained_in_source_field"
        score = 4.0 + 0.15 * min(len(accepted_tokens), 6)
    elif source_normalized in accepted_normalized:
        basis = "source_field_contained_in_accepted_slot"
        score = 3.8 + 0.15 * min(len(source_tokens), 6)
    else:
        smaller_coverage = len(shared) / min(len(source_tokens), len(accepted_tokens))
        accepted_coverage = len(shared) / len(accepted_tokens)
        if (
            source_numbers
            and source_numbers == accepted_numbers
            and len(shared - source_numbers) >= 2
        ):
            basis = "numeric_range_and_subject_signature"
            score = round(3.0 + 0.5 * accepted_coverage, 6)
        elif (
            "," in str(accepted)
            and (" and " in accepted_normalized or " or " in accepted_normalized)
            and (shared - _COORDINATE_GENERIC)
            and len(source_tokens) < len(accepted_tokens)
        ):
            basis = "coordinated_relation_member_signature"
            score = round(2.5 + 0.5 * accepted_coverage, 6)
        elif len(shared) < 2 or smaller_coverage < 0.66:
            return None
        else:
            basis = "canonical_structured_token_signature"
            score = round(2.0 + smaller_coverage + 0.5 * accepted_coverage, 6)
    return {
        "basis": basis,
        "score": score,
        "source_text": source,
        "accepted_text": accepted,
        "shared_signature_tokens": sorted(shared),
        "source_token_count": len(source_tokens),
        "accepted_token_count": len(accepted_tokens),
        "source_coverage": round(len(shared) / len(source_tokens), 6),
        "accepted_coverage": round(len(shared) / len(accepted_tokens), 6),
    }


def _collect_text_slots(value: Any, keys: frozenset[str]) -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in keys and isinstance(child, str) and child.strip():
                result.append(child.strip())
            else:
                result.extend(_collect_text_slots(child, keys))
    elif isinstance(value, list):
        for child in value:
            result.extend(_collect_text_slots(child, keys))
    return list(dict.fromkeys(result))


def _contains_logical_negation(value: Any) -> bool:
    if isinstance(value, Mapping):
        if str(value.get("kind") or "").startswith("not_"):
            return True
        return any(_contains_logical_negation(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_logical_negation(child) for child in value)
    return False


def _best_match(source_fields: list[str], accepted_slots: list[str]) -> dict[str, Any] | None:
    matches = [
        match
        for source in source_fields
        for accepted in accepted_slots
        if (match := _structured_match(source, accepted)) is not None
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda item: (
            float(item["score"]),
            len(item["shared_signature_tokens"]),
            item["accepted_text"],
        ),
    )


def _component_matches(
    source_fields: list[str], accepted_slots: list[str]
) -> list[dict[str, Any]]:
    """Keep the best accepted match for every source assertion clause."""

    result = []
    full_source = source_fields[0] if source_fields else None
    for source in source_fields:
        if source == full_source and len(source_fields) > 1:
            continue
        match = _best_match([source], accepted_slots)
        if match is not None:
            result.append(match)
    if not result and full_source is not None:
        match = _best_match([full_source], accepted_slots)
        if match is not None:
            result.append(match)
    deduplicated = {}
    for match in result:
        key = (match["source_text"], match["accepted_text"])
        deduplicated.setdefault(key, match)
    return list(deduplicated.values())


def _candidate_assertion_signature(candidate: Mapping[str, Any]) -> set[str]:
    evidence = candidate.get("match_evidence") or {}
    matches = list(evidence.get("assertion_components") or [])
    if evidence.get("assertion") is not None:
        matches.append(evidence["assertion"])
    return set().union(
        *(set(match.get("shared_signature_tokens") or []) for match in matches)
    ) if matches else set()


def _candidate_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_kind": candidate.get("candidate_kind"),
        "path_test_specification_id": candidate.get(
            "path_test_specification_id"
        ),
        "semantic_unit_ids": deepcopy(candidate.get("semantic_unit_ids") or []),
        "policy_ref_ids": deepcopy(candidate.get("policy_ref_ids") or []),
    }


def _logical_polarity(value: Any) -> str:
    normalized = _normalized(value)
    if re.search(r"\b(?:not|no|never|cannot|without|unable)\b", normalized):
        return "negative"
    return "positive"


def _source_assertion_polarity(source: Mapping[str, Any]) -> str:
    then = _normalized(source.get("then") or "")
    if re.search(r"\b(?:must|may|can|should|will) not\b", then):
        return "negative"
    if str(source.get("deontic") or "") == "prohibition":
        return "negative"
    return "positive"


def _accepted_assertion_polarity(specification: Mapping[str, Any]) -> str | None:
    expected = specification.get("expected_results") or {}
    oracle = specification.get("observation_oracle") or {}
    expectations = {
        str(value)
        for _, value in _collect_key_values(
            {"expected": expected, "oracle": oracle}, frozenset({"expectation"})
        )
    }
    if any(value.startswith("must_not") for value in expectations):
        return "negative"
    if expectations & {
        "must_perform",
        "must_call",
        "may_perform",
        "may_call_for_operation",
    }:
        return "positive"
    assertion_text = " ".join(
        _collect_text_slots(
            {"expected": expected, "oracle": oracle},
            frozenset(
                {"action_text", "exact_relation_text", "relation_text"}
            ),
        )
    )
    normalized = _normalized(assertion_text)
    if re.search(r"\b(?:prohibited|must not|cannot|not allowed)\b", normalized):
        return "negative"
    if re.search(r"\b(?:permitted|allowed|can be|may)\b", normalized):
        return "positive"
    return None


def _collect_key_values(value: Any, keys: frozenset[str]) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in keys and isinstance(child, (str, bool, int, float)):
                result.append((key, child))
            else:
                result.extend(_collect_key_values(child, keys))
    elif isinstance(value, list):
        for child in value:
            result.extend(_collect_key_values(child, keys))
    return result


def _content_stem(token: str) -> str:
    value = _ALIASES.get(token, _OPERATION_ALIASES.get(token, token))
    if value.endswith("ion") and len(value) > 6:
        return value[:-3]
    if value.endswith("ing") and len(value) > 6:
        value = value[:-3]
    if value.endswith("ed") and len(value) > 5:
        value = value[:-2]
    if value.endswith("e") and len(value) > 5:
        value = value[:-1]
    return value


def _source_action_verb(value: Any) -> str | None:
    normalized = _normalized(value)
    match = re.search(
        r"\b(?:agent|user)\s+(?:must|may|can|should|will)\s+(?:not\s+)?([a-z_]+)",
        normalized,
    )
    return match.group(1) if match is not None else None


def _deontic_nominalization_texts(
    source_action_stem: str | None, accepted_texts: list[str]
) -> list[str]:
    """Find accepted assertions that predicate modality of the same action.

    A shared derivational stem is insufficient: ``verify cancellation rules``
    mentions cancellation but does not assert that cancellation is permitted.
    The accepted text must also carry an explicit modal/deontic predicate.  This
    keeps nominalization repair structural rather than a loose bag-of-words
    match.
    """

    if source_action_stem is None:
        return []
    result = []
    for text in accepted_texts:
        normalized = _normalized(text)
        stems = {
            _content_stem(token) for token in normalized.split()
        }
        if source_action_stem not in stems:
            continue
        if not re.search(
            r"\b(?:must|may|can|cannot|should|required|permitted|prohibited|allowed)\b",
            normalized,
        ):
            continue
        result.append(text)
    return result


def _condition_texts(specification: Mapping[str, Any]) -> list[str]:
    activation = (specification.get("input_condition") or {}).get(
        "activation", {}
    )
    return _collect_text_slots(
        activation,
        frozenset({"condition_text", "predicate_text", "source_text"}),
    )


def _condition_discriminator_tokens(
    candidate: Mapping[str, Any],
    specification: Mapping[str, Any] | None,
) -> set[str]:
    if specification is not None:
        activation = (specification.get("input_condition") or {}).get(
            "activation", {}
        )
        exact_values = [
            str(item.get("value_text"))
            for item in _collect_mappings(activation)
            if item.get("kind") == "value_equals" and item.get("value_text")
        ]
        if exact_values:
            return set(_tokens(" ".join(exact_values)))
        return set(_tokens(" ".join(_condition_texts(specification))))
    match = (candidate.get("match_evidence") or {}).get("condition") or {}
    return set(_tokens(match.get("accepted_text") or ""))


def _collect_mappings(value: Any) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        result.append(value)
        for child in value.values():
            result.extend(_collect_mappings(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_collect_mappings(child))
    return result


def _candidate_path_specification(
    candidate: Mapping[str, Any],
    path_index: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    path_id = candidate.get("path_test_specification_id")
    return path_index.get(str(path_id)) if path_id else None


def _candidate_condition_polarity(
    candidate: Mapping[str, Any],
    specification: Mapping[str, Any] | None,
) -> str | None:
    if specification is not None:
        texts = _condition_texts(specification)
    else:
        match = (candidate.get("match_evidence") or {}).get("condition") or {}
        texts = [str(match.get("accepted_text") or "")]
    texts = [text for text in texts if text]
    if not texts:
        return None
    polarity = _logical_polarity(" ".join(texts))
    if candidate.get("condition_truth_value") is False:
        polarity = "positive" if polarity == "negative" else "negative"
    return polarity


def _candidate_focal_signature(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    evidence = candidate.get("match_evidence") or {}
    assertion = evidence.get("assertion") or {}
    accepted_text = assertion.get("accepted_text")
    return tuple(sorted(_tokens(accepted_text))) if accepted_text else ()


def _source_condition_is_broader_disjunction(
    candidate: Mapping[str, Any],
) -> bool:
    """Detect a reviewed OR condition that strictly widens one accepted atom."""

    condition = (candidate.get("match_evidence") or {}).get("condition") or {}
    source_text = str(condition.get("source_text") or "")
    accepted_text = str(condition.get("accepted_text") or "")
    source_normalized = _normalized(source_text)
    accepted_normalized = _normalized(accepted_text)
    if " or " not in f" {source_normalized} ":
        return False
    if " or " in f" {accepted_normalized} ":
        return False
    source_tokens = set(_tokens(source_text))
    accepted_tokens = set(_tokens(accepted_text))
    extra_tokens = source_tokens - accepted_tokens - set(_COORDINATE_GENERIC)
    return bool(extra_tokens)


def _source_asserted_tool_name(source: Mapping[str, Any]) -> str | None:
    action = _source_action_verb(source.get("then"))
    return action if action and "_" in action else None


def _runtime_tool_grounding(
    candidate: Mapping[str, Any], runtime_document: Mapping[str, Any]
) -> tuple[set[str], set[str]]:
    """Return family-verified tools and path-oracle-observed tools."""

    family_ids = {
        str(item.get("operation_policy_family_id"))
        for item in candidate.get("runtime_contract_refs") or []
        if item.get("operation_policy_family_id")
    }
    contract_ids = {
        str(item.get("contract_id"))
        for item in candidate.get("runtime_contract_refs") or []
        if item.get("contract_id")
    }
    verified = {
        str(tool_name)
        for family in runtime_document.get("operation_family_contracts") or []
        if str(family.get("operation_policy_family_id")) in family_ids
        for tool_name in (
            (family.get("operation_selector") or {}).get("verified_tool_names")
            or []
        )
    }
    observed: set[str] = set()
    for contract in runtime_document.get("path_match_contracts") or []:
        if str(contract.get("path_match_contract_id")) not in contract_ids:
            continue
        for monitor in _collect_mappings(contract.get("oracle_monitors") or {}):
            if monitor.get("tool_name"):
                observed.add(str(monitor["tool_name"]))
            observed.update(
                str(tool_name) for tool_name in monitor.get("tool_names") or []
            )
    return verified, observed


def _path_only_gap_reason_codes(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    specification: Mapping[str, Any] | None,
    runtime_document: Mapping[str, Any],
    *,
    action_aligned: bool,
) -> list[str]:
    """Explain why an in-lineage path cannot cover the reviewed assertion."""

    reasons: list[str] = []
    then = _normalized(source.get("then") or "")
    if re.search(r"\b(?:before|after)\b", then):
        trace_constraints = (
            specification.get("trace_constraints") or {}
            if specification is not None
            else {}
        )
        pre_trace = (
            specification.get("pre_trace_setup") or {}
            if specification is not None
            else {}
        )
        has_precedence_evidence = any(
            bool(value) for value in trace_constraints.values()
        ) or bool(pre_trace.get("dependency_requirements"))
        if not has_precedence_evidence:
            reasons.append("source_precedence_not_covered_by_path_oracle")

    source_tool = _source_asserted_tool_name(source)
    if source_tool is not None:
        verified_tools, observed_tools = _runtime_tool_grounding(
            candidate, runtime_document
        )
        if source_tool in verified_tools and source_tool not in observed_tools:
            reasons.append("source_tool_oracle_not_runtime_grounded")

    if not action_aligned:
        reasons.append("source_action_not_covered_by_accepted_oracle")
    return list(dict.fromkeys(reasons))


def _source_precedence_is_ungrounded(source: Mapping[str, Any]) -> bool:
    """Return true when Then invents ordering absent from its source evidence."""

    then = _normalized(source.get("then") or "")
    then_markers = set(re.findall(r"\b(?:before|after)\b", then))
    if not then_markers:
        return False
    metadata = source.get("source_metadata") or {}
    grounding_text = _normalized(
        f"{source.get('rule_text') or ''} "
        f"{metadata.get('evidence_quote') or ''}"
    )
    grounding_markers = set(
        re.findall(r"\b(?:before|after)\b", grounding_text)
    )
    if "prior to" in grounding_text:
        grounding_markers.add("before")
    if "subsequent to" in grounding_text:
        grounding_markers.add("after")
    return not then_markers <= grounding_markers


def _action_clause_with_deontic_marker(
    text: Any, action_stem: str
) -> str | None:
    """Return the smallest clause that explicitly predicates an action.

    Accepted semantic extraction is intentionally conservative and can miss a
    nominal passive such as ``a transfer is needed``.  Recovery is only safe
    when the action and its modality occur in the same local clause; merely
    mentioning the action elsewhere in a policy is insufficient.
    """

    raw = str(text or "").strip()
    if not raw:
        return None
    clauses = [
        clause.strip(" .()")
        for clause in re.split(r"[.;]|\b(?:and|or)\b", raw, flags=re.I)
        if clause.strip(" .()")
    ]
    for clause in clauses:
        normalized = _normalized(clause)
        stems = {_content_stem(token) for token in normalized.split()}
        if action_stem not in stems:
            continue
        if re.search(
            r"\b(?:must|may|can|cannot|should|need|needs|needed|required|"
            r"permitted|prohibited|allowed)\b",
            normalized,
        ):
            return clause
    return None


def _actions_are_coordinated(
    text: Any, source_action_stem: str, accepted_action_stems: set[str]
) -> bool:
    """Require the recovered and accepted actions in one coordinated sentence."""

    normalized = _normalized(text)
    if not normalized or not accepted_action_stems:
        return False
    tokens = normalized.split()
    source_positions = [
        index
        for index, token in enumerate(tokens)
        if _content_stem(token) == source_action_stem
    ]
    accepted_positions = [
        index
        for index, token in enumerate(tokens)
        if _content_stem(token) in accepted_action_stems
    ]
    return any(
        any(
            marker in {"and", "then"}
            for marker in tokens[min(left, right) + 1 : max(left, right)]
        )
        for left in source_positions
        for right in accepted_positions
        if left != right
    )


def _accepted_focal_action_stems(texts: list[str]) -> set[str]:
    """Extract only the focal predicate of each accepted assertion slot."""

    result: set[str] = set()
    modality_tokens = {
        "allowed",
        "cannot",
        "may",
        "must",
        "need",
        "needed",
        "permitted",
        "prohibited",
        "required",
        "should",
        "will",
    }
    for text in texts:
        normalized = _normalized(text)
        modal_action = re.search(
            r"\b(?:agent|user)\s+(?:must|may|can|should|will)\s+"
            r"(?:not\s+)?([a-z_]+)",
            normalized,
        )
        if modal_action is not None:
            result.add(_content_stem(modal_action.group(1)))
            continue
        content = [
            token
            for token in normalized.split()
            if token not in _STOP and token not in modality_tokens
        ]
        if content:
            result.add(_content_stem(content[0]))
    return result


def _source_evidence_action_recovery(
    source: Mapping[str, Any],
    accepted_assertion_texts: list[str],
    accepted_policy_statements: list[str],
) -> dict[str, Any] | None:
    """Recover one source action omitted by accepted semantic atomization.

    This is not free-form inference.  The reviewed Then must state a mandatory
    action, the reviewed source evidence and accepted policy must independently
    state that action with a local deontic marker, and both must coordinate it
    with the action already monitored by the accepted path.
    """

    then = _normalized(source.get("then") or "")
    source_action = _source_action_verb(source.get("then"))
    if source_action is None or not re.search(
        r"\b(?:must|should|will)\s+(?:not\s+)?[a-z_]+", then
    ):
        return None
    source_action_stem = _content_stem(source_action)
    accepted_action_stems = _accepted_focal_action_stems(
        accepted_assertion_texts
    )
    if source_action_stem in accepted_action_stems or not accepted_action_stems:
        return None

    metadata = source.get("source_metadata") or {}
    source_grounding_texts = list(
        dict.fromkeys(
            str(value).strip()
            for value in (
                source.get("rule_text"),
                metadata.get("evidence_quote"),
            )
            if str(value or "").strip()
        )
    )
    source_clause = next(
        (
            clause
            for text in source_grounding_texts
            if (
                clause := _action_clause_with_deontic_marker(
                    text, source_action_stem
                )
            )
            and _actions_are_coordinated(
                text, source_action_stem, accepted_action_stems
            )
        ),
        None,
    )
    accepted_match = next(
        (
            (statement, clause)
            for statement in accepted_policy_statements
            if (
                clause := _action_clause_with_deontic_marker(
                    statement, source_action_stem
                )
            )
            and _actions_are_coordinated(
                statement, source_action_stem, accepted_action_stems
            )
        ),
        None,
    )
    if source_clause is None or accepted_match is None:
        return None

    accepted_statement, accepted_clause = accepted_match
    expectation = (
        "must_not_observe"
        if re.search(
            rf"\b(?:must|should|will)\s+not\s+{re.escape(source_action)}\b",
            then,
        )
        else "must_observe"
    )
    return {
        "kind": "source_evidence_action",
        "source_action": source_action,
        "source_action_stem": source_action_stem,
        "source_action_text": source.get("then"),
        "expectation": expectation,
        "observation_target": {
            "kind": "assistant_response_semantic_action",
            "action_text": source.get("then"),
            "expectation": expectation,
            "semantic_match_required": True,
        },
        "grounding": {
            "source_clause": source_clause,
            "accepted_policy_clause": accepted_clause,
            "accepted_policy_statement": accepted_statement,
            "accepted_path_action_texts": deepcopy(
                accepted_assertion_texts
            ),
        },
    }


def _adjudicate_candidates(
    source: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    path_document: Mapping[str, Any],
    runtime_document: Mapping[str, Any],
    accepted_policy_statements: list[str],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Classify candidate scope and repair mechanically equivalent wording.

    Most calls arrive with several candidates, but a single path-only candidate
    also needs this pass.  In particular, accepted relations often express the
    focal behavior as a noun (``compensation``) while a reviewed branch uses the
    corresponding action verb (``compensate``).  The same structured checks used
    to adjudicate ambiguity can prove that equivalence without an LLM.
    """

    path_index = {
        str(item.get("path_test_specification_id")): item
        for item in path_document.get("path_test_specifications") or []
        if isinstance(item, Mapping) and item.get("path_test_specification_id")
    }
    source_assertion_polarity = _source_assertion_polarity(source)
    source_condition_polarity = _logical_polarity(
        source.get("given")
        if _normalized(source.get("given")) not in {"", "true"}
        else source.get("when")
    )
    source_action = _source_action_verb(source.get("then"))
    source_action_stem = _content_stem(source_action) if source_action else None
    then_has_modal_negation = bool(
        re.search(
            r"\b(?:must|may|can|should|will)\s+not\b",
            _normalized(source.get("then") or ""),
        )
    )
    if (
        str(source.get("deontic") or "") == "permission"
        and then_has_modal_negation
    ):
        rejected = [
            {
                **_candidate_identity(candidate),
                "reason_codes": ["source_deontic_then_polarity_conflict"],
            }
            for candidate in candidates
        ]
        return (
            "invalid_source_branch",
            candidates,
            rejected,
            {
                "status": "invalid_source_branch",
                "reason_codes": ["source_deontic_then_polarity_conflict"],
                "source_deontic": source.get("deontic"),
                "source_assertion_polarity": source_assertion_polarity,
            },
        )

    specifications = [
        _candidate_path_specification(candidate, path_index)
        for candidate in candidates
    ]
    discriminator_sets = [
        _condition_discriminator_tokens(candidate, specification)
        for candidate, specification in zip(candidates, specifications)
    ]
    nonempty_discriminators = [value for value in discriminator_sets if value]
    common_discriminators = (
        set.intersection(*nonempty_discriminators)
        if nonempty_discriminators
        else set()
    )
    generic_discriminators = set(_COORDINATE_GENERIC) | {
        "action",
        "condition",
        "know",
        "scope",
    }
    distinctive_sets = [
        value - common_discriminators - generic_discriminators
        for value in discriminator_sets
    ]
    source_condition_tokens = set(
        _tokens(
            f"{source.get('given') or ''} {source.get('when') or ''}"
        )
    )
    discriminator_overlaps = [
        len(value & source_condition_tokens) for value in distinctive_sets
    ]
    max_discriminator_overlap = max(discriminator_overlaps, default=0)

    evaluations = []
    surviving = []
    rejected = []
    for candidate_index, candidate in enumerate(candidates):
        specification = specifications[candidate_index]
        condition_polarity = _candidate_condition_polarity(
            candidate, specification
        )
        assertion_polarity = (
            _accepted_assertion_polarity(specification)
            if specification is not None
            else source_assertion_polarity
        )
        reason_codes = []
        if _source_condition_is_broader_disjunction(candidate):
            reason_codes.append(
                "source_condition_broader_than_accepted_condition"
            )
        if (
            condition_polarity is not None
            and condition_polarity != source_condition_polarity
        ):
            reason_codes.append("condition_polarity_conflict")
        if (
            assertion_polarity is not None
            and assertion_polarity != source_assertion_polarity
        ):
            reason_codes.append("assertion_polarity_conflict")
        if (
            max_discriminator_overlap > 0
            and distinctive_sets[candidate_index]
            and discriminator_overlaps[candidate_index] == 0
        ):
            reason_codes.append("condition_discriminator_conflict")
        accepted_assertion_texts = (
            _collect_text_slots(
                {
                    "expected": specification.get("expected_results"),
                    "oracle": specification.get("observation_oracle"),
                },
                frozenset(
                    {"action_text", "exact_relation_text", "relation_text"}
                ),
            )
            if specification is not None
            else [
                str(
                    ((candidate.get("match_evidence") or {}).get("assertion") or {}).get(
                        "accepted_text"
                    )
                    or ""
                )
            ]
        )
        accepted_stems = {
            _content_stem(token)
            for text in accepted_assertion_texts
            for token in _normalized(text).split()
        }
        action_aligned = bool(
            source_action_stem and source_action_stem in accepted_stems
        )
        source_action_recovery = _source_evidence_action_recovery(
            source,
            accepted_assertion_texts,
            accepted_policy_statements,
        )
        gap_reason_codes = _path_only_gap_reason_codes(
            source,
            candidate,
            specification,
            runtime_document,
            action_aligned=action_aligned,
        )
        if source_action_recovery is not None:
            gap_reason_codes = [
                reason
                for reason in gap_reason_codes
                if reason != "source_action_not_covered_by_accepted_oracle"
            ]
        nominalization_texts = _deontic_nominalization_texts(
            source_action_stem, accepted_assertion_texts
        )
        evaluation = {
            **_candidate_identity(candidate),
            "condition_polarity": condition_polarity,
            "assertion_polarity": assertion_polarity,
            "source_action": source_action,
            "source_action_stem": source_action_stem,
            "action_aligned": action_aligned,
            "source_evidence_action_recovery": deepcopy(
                source_action_recovery
            ),
            "condition_discriminator_tokens": sorted(
                distinctive_sets[candidate_index]
            ),
            "condition_discriminator_overlap": discriminator_overlaps[
                candidate_index
            ],
            "reason_codes": reason_codes,
            "gap_reason_codes": gap_reason_codes,
        }
        evaluations.append(evaluation)
        if reason_codes:
            rejected.append(evaluation)
        else:
            normalized_candidate = deepcopy(candidate)
            if condition_polarity is not None:
                match_evidence = normalized_candidate.setdefault(
                    "match_evidence", {}
                )
                if match_evidence.get("condition_polarity") is not None:
                    match_evidence["pre_adjudication_condition_polarity"] = deepcopy(
                        match_evidence["condition_polarity"]
                    )
                match_evidence["condition_polarity"] = {
                    "basis": "structured_ambiguity_adjudication",
                    "source_logical_polarity": source_condition_polarity,
                    "accepted_logical_polarity": condition_polarity,
                    "condition_truth_value": normalized_candidate.get(
                        "condition_truth_value"
                    ),
                    "matches": True,
                }
            if (
                normalized_candidate.get("projection_quality") == "path_only"
                and nominalization_texts
            ):
                accepted_text = min(
                    nominalization_texts,
                    key=len,
                )
                normalized_candidate["projection_quality"] = "complete"
                normalized_candidate["selected_semantic_unit_ids"] = deepcopy(
                    normalized_candidate.get("semantic_unit_ids") or []
                )
                normalized_candidate.setdefault("match_evidence", {})[
                    "assertion"
                ] = {
                    "basis": "deontic_subject_nominalization",
                    "score": 4.0,
                    "source_text": source.get("then"),
                    "accepted_text": accepted_text,
                    "shared_signature_tokens": [source_action_stem],
                    "source_token_count": len(_tokens(source.get("then"))),
                    "accepted_token_count": len(_tokens(accepted_text)),
                    "source_coverage": None,
                    "accepted_coverage": None,
                }
            elif (
                normalized_candidate.get("projection_quality") == "path_only"
                and source_action_recovery is not None
            ):
                normalized_candidate["projection_quality"] = "complete_atomic"
                normalized_candidate["atomic_binding"] = deepcopy(
                    source_action_recovery
                )
                normalized_candidate.setdefault("match_evidence", {})[
                    "assertion"
                ] = {
                    "basis": "source_evidence_coordinated_action_recovery",
                    "score": 4.0,
                    "source_text": source.get("then"),
                    "accepted_text": source_action_recovery["grounding"][
                        "accepted_policy_clause"
                    ],
                    "shared_signature_tokens": [source_action_stem],
                    "source_token_count": len(_tokens(source.get("then"))),
                    "accepted_token_count": len(
                        _tokens(
                            source_action_recovery["grounding"][
                                "accepted_policy_clause"
                            ]
                        )
                    ),
                    "source_coverage": None,
                    "accepted_coverage": None,
                }
            surviving.append(normalized_candidate)

    if len(surviving) == 1:
        resolved_status = (
            "resolved_atomic"
            if surviving[0].get("projection_quality") == "complete_atomic"
            else (
                "path_only"
                if surviving[0].get("projection_quality") == "path_only"
                else "resolved_unique"
            )
        )
        if (
            resolved_status == "path_only"
            and _source_precedence_is_ungrounded(source)
            and any(
                "source_precedence_not_covered_by_path_oracle"
                in item["gap_reason_codes"]
                for item in evaluations
                if not item["reason_codes"]
            )
        ):
            rejection = {
                **_candidate_identity(surviving[0]),
                "reason_codes": [
                    "source_precedence_not_grounded_in_source_evidence"
                ],
            }
            return (
                "assertion_scope_conflict",
                surviving,
                [*rejected, rejection],
                {
                    "status": "assertion_scope_conflict",
                    "reason_codes": [
                        "source_precedence_not_grounded_in_source_evidence"
                    ],
                    "source_rule_text": source.get("rule_text"),
                    "source_evidence_quote": (
                        (source.get("source_metadata") or {}).get(
                            "evidence_quote"
                        )
                    ),
                    "candidate_evaluations": evaluations,
                },
            )
        if resolved_status == "path_only":
            reason_codes = list(
                dict.fromkeys(
                    reason
                    for item in evaluations
                    if not item["reason_codes"]
                    for reason in item["gap_reason_codes"]
                )
            ) or ["accepted_path_does_not_cover_source_assertion"]
        elif len(candidates) == 1:
            if (
                candidates[0].get("projection_quality") == "path_only"
                and surviving[0].get("projection_quality") == "complete"
            ):
                reason_codes = ["single_candidate_deontic_nominalization"]
            elif (
                candidates[0].get("projection_quality") == "path_only"
                and surviving[0].get("projection_quality") == "complete_atomic"
                and (surviving[0].get("atomic_binding") or {}).get("kind")
                == "source_evidence_action"
            ):
                reason_codes = [
                    "single_candidate_source_evidence_action_recovery"
                ]
            else:
                reason_codes = ["single_candidate_structured_review"]
        else:
            reason_codes = ["incompatible_candidates_eliminated"]
        return (
            resolved_status,
            surviving,
            rejected,
            {
                "status": (
                    resolved_status
                    if len(candidates) == 1 or resolved_status == "path_only"
                    else "resolved_unique"
                ),
                "reason_codes": reason_codes,
                "candidate_evaluations": evaluations,
            },
        )
    if not surviving:
        if any(
            "source_condition_broader_than_accepted_condition"
            in item["reason_codes"]
            for item in evaluations
        ):
            return (
                "condition_scope_conflict",
                candidates,
                rejected,
                {
                    "status": "condition_scope_conflict",
                    "reason_codes": [
                        "source_condition_broader_than_accepted_condition"
                    ],
                    "candidate_evaluations": evaluations,
                },
            )
        if all(
            candidate.get("projection_quality") == "path_only"
            for candidate in candidates
        ) and not any(item["action_aligned"] for item in evaluations):
            return (
                "path_only",
                candidates,
                rejected,
                {
                    "status": "path_only",
                    "reason_codes": list(
                        dict.fromkeys(
                            reason
                            for item in evaluations
                            for reason in item["gap_reason_codes"]
                        )
                    )
                    or ["accepted_path_does_not_cover_source_assertion"],
                    "candidate_evaluations": evaluations,
                },
            )
        weak_scope_match = any(
            (
                (candidate.get("match_evidence") or {}).get("assertion") or {}
            ).get("basis")
            == "coordinated_relation_member_signature"
            for candidate in candidates
        )
        status = (
            "assertion_scope_conflict"
            if weak_scope_match or source_action is not None
            else "ambiguous"
        )
        return (
            status,
            candidates,
            rejected,
            {
                "status": status,
                "reason_codes": ["all_candidates_conflict_with_source_contract"],
                "candidate_evaluations": evaluations,
            },
        )

    signatures = {_candidate_focal_signature(candidate) for candidate in surviving}
    assertion_polarities = {
        item["assertion_polarity"]
        for item in evaluations
        if not item["reason_codes"] and item["assertion_polarity"] is not None
    }
    if len(signatures) == 1 and () not in signatures and len(assertion_polarities) <= 1:
        return (
            "resolved_alternative",
            surviving,
            rejected,
            {
                "status": "resolved_alternative",
                "reason_codes": ["equivalent_focal_assertion_multiple_realizations"],
                "focal_signature_tokens": list(next(iter(signatures))),
                "candidate_evaluations": evaluations,
            },
        )
    return (
        "ambiguous",
        surviving,
        rejected,
        {
            "status": "remains_ambiguous",
            "reason_codes": ["multiple_non_equivalent_candidates_remain"],
            "candidate_evaluations": evaluations,
        },
    )


def _assertion_segments(value: Any) -> list[str]:
    """Return independently matchable clauses while retaining the full text."""

    text = str(value or "").strip()
    if not text:
        return []
    parts = [
        part.strip(" .()")
        for part in re.split(r"\s+(?:and|before|after)\s+|[,;]", text, flags=re.I)
        if part.strip(" .()")
    ]
    return list(dict.fromkeys([text, *parts]))


def _path_runtime_index(document: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for item in document.get("path_match_contracts") or []:
        if not isinstance(item, Mapping) or not item.get("path_test_specification_id"):
            continue
        path_id = str(item["path_test_specification_id"])
        result.setdefault(path_id, []).append(
            {
                "contract_kind": "path_match_contract",
                "contract_id": item.get("path_match_contract_id"),
                "operation_policy_family_id": item.get(
                    "owner_operation_policy_family_id"
                ),
                "path_test_specification_id": path_id,
            }
        )
    return result


def _semantic_assertion_slots(specification: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    text_keys = frozenset(
        {
            "action_text",
            "exact_relation_text",
            "member_text",
            "relation_text",
            "requirement_text",
            "tool_name",
        }
    )

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            semantic_id = value.get("action_semantic_unit_id") or value.get(
                "relation_semantic_unit_id"
            )
            if semantic_id:
                key = str(semantic_id)
                result[key] = list(
                    dict.fromkeys(
                        result.get(key, [])
                        + _collect_text_slots(value, text_keys)
                    )
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(specification)
    return result


def _family_index(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["operation_policy_family_id"]): item
        for item in document.get("operation_family_contracts") or []
        if isinstance(item, Mapping) and item.get("operation_policy_family_id")
    }


def _path_candidates(
    *,
    policy_ref_ids: set[str],
    allowed_path_ids: set[str],
    path_document: Mapping[str, Any],
    runtime_document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runtime_index = _path_runtime_index(runtime_document)
    families = _family_index(runtime_document)
    result = []
    for specification in path_document.get("path_test_specifications") or []:
        if not isinstance(specification, Mapping):
            continue
        path_id = str(specification.get("path_test_specification_id") or "")
        if path_id not in allowed_path_ids or path_id not in runtime_index:
            continue
        refs = sorted(policy_ref_ids & set(specification.get("policy_ref_ids") or []))
        if not refs:
            continue
        assertion_slots = _collect_text_slots(
            {
                "focal_execution": specification.get("focal_execution"),
                "expected_results": specification.get("expected_results"),
                "observation_oracle": specification.get("observation_oracle"),
                "pre_trace_setup": specification.get("pre_trace_setup"),
                "trace_constraints": specification.get("trace_constraints"),
            },
            frozenset(
                {
                    "action_text",
                    "exact_relation_text",
                    "member_text",
                    "relation_text",
                    "requirement_text",
                    "tool_name",
                }
            ),
        )
        activation = (specification.get("input_condition") or {}).get(
            "activation", {}
        )
        selector_condition = activation.get(
            "eligibility_witness_expression"
        ) or activation.get("expression") or activation.get("source_conditions")
        condition_slots = _collect_text_slots(
            selector_condition,
            frozenset({"condition_text", "predicate_text", "source_text"}),
        )
        inventory = (
            specification.get("focal_execution", {}).get(
                "operation_endpoint_inventory", {}
            )
            if isinstance(specification.get("focal_execution"), Mapping)
            else {}
        )
        operation_slots = []
        for runtime_ref in runtime_index[path_id]:
            family = families.get(str(runtime_ref["operation_policy_family_id"]))
            if not isinstance(family, Mapping):
                continue
            selector = family.get("operation_selector") or {}
            operation_slots.extend(selector.get("operation_expressions") or [])
        result.append(
            {
                "candidate_kind": "path_contract",
                "policy_ref_ids": refs,
                "path_test_specification_id": path_id,
                "semantic_unit_ids": sorted(
                    set(inventory.get("action_semantic_unit_ids") or [])
                    | set(inventory.get("relation_semantic_unit_ids") or [])
                ),
                "source_branch_seed_ids": deepcopy(
                    specification.get("input_condition", {}).get(
                        "source_branch_seed_ids", []
                    )
                ),
                "condition_truth_value": specification.get(
                    "input_condition", {}
                ).get("activation", {}).get("condition_truth_value"),
                "activation_expression_is_negated": _contains_logical_negation(
                    selector_condition
                ),
                "assertion_slots": assertion_slots,
                "semantic_assertion_slots": _semantic_assertion_slots(
                    specification
                ),
                "condition_slots": condition_slots,
                "operation_slots": list(dict.fromkeys(operation_slots)),
                "runtime_contract_refs": runtime_index[path_id],
            }
        )
    return result


def _prerequisite_candidates(
    *, policy_ref_ids: set[str], runtime_document: Mapping[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for family in runtime_document.get("operation_family_contracts") or []:
        if not isinstance(family, Mapping):
            continue
        family_id = str(family.get("operation_policy_family_id") or "")
        selector = family.get("operation_selector") or {}
        operation_slots = list(selector.get("operation_expressions") or [])
        for monitor in family.get("prerequisite_monitors") or []:
            if not isinstance(monitor, Mapping):
                continue
            refs = sorted(policy_ref_ids & set(monitor.get("policy_ref_ids") or []))
            if not refs:
                continue
            for event in monitor.get("required_prior_events") or []:
                if not isinstance(event, Mapping) or not event.get("action_text"):
                    continue
                semantic_id = str(event.get("action_semantic_unit_id") or "")
                action_text = str(event["action_text"])
                for ref in refs:
                    key = (ref, semantic_id, _normalized(action_text))
                    item = grouped.setdefault(
                        key,
                        {
                            "candidate_kind": "prerequisite_event",
                            "policy_ref_ids": [ref],
                            "path_test_specification_id": None,
                            "semantic_unit_ids": [semantic_id] if semantic_id else [],
                            "source_branch_seed_ids": [],
                            "condition_truth_value": True,
                            "assertion_slots": [action_text],
                            "condition_slots": _collect_text_slots(
                                event,
                                frozenset({"condition_text", "source_text"}),
                            ),
                            "operation_slots": [],
                            "runtime_contract_refs": [],
                        },
                    )
                    item["operation_slots"] = list(
                        dict.fromkeys(item["operation_slots"] + operation_slots)
                    )
                    item["runtime_contract_refs"].append(
                        {
                            "contract_kind": "prerequisite_monitor",
                            "contract_id": monitor.get("prerequisite_monitor_id"),
                            "event_requirement_id": event.get(
                                "event_requirement_id"
                            ),
                            "operation_policy_family_id": family_id,
                            "path_test_specification_id": None,
                        }
                    )
    return list(grouped.values())


def resolve_runtime_projection(
    source_spec_record: Mapping[str, Any],
    policy_lineage_report: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a reviewed branch only inside its accepted policy lineage."""

    source = validate_source_spec_record(source_spec_record)
    policy = validate_policy_lineage_report(policy_lineage_report)
    if policy["source_record_fingerprint"] != source["source_record_fingerprint"]:
        raise RuntimeProjectionError("policy lineage report does not belong to source")

    policy_ref_ids = {
        str(item["policy_ref_id"]) for item in policy.get("candidates") or []
    }
    accepted_policy_statements = list(
        dict.fromkeys(
            str(item.get("accepted_policy_statement")).strip()
            for item in policy.get("candidates") or []
            if str(item.get("accepted_policy_statement") or "").strip()
        )
    )
    allowed_path_ids = {
        str(path_id)
        for item in policy.get("candidates") or []
        for path_id in item.get("path_test_specification_ids") or []
    }
    if not policy_ref_ids:
        status = "not_applicable"
        candidates: list[dict[str, Any]] = []
        rejected_candidates: list[dict[str, Any]] = []
    else:
        path_document = artifact_document_for_capability(
            resolved_artifacts, CAP_POLICY_PATH_TEST_SPECIFICATIONS
        )
        runtime_document = artifact_document_for_capability(
            resolved_artifacts, CAP_EXECUTION_MATCHING_CONTRACTS
        )
        raw = _path_candidates(
            policy_ref_ids=policy_ref_ids,
            allowed_path_ids=allowed_path_ids,
            path_document=path_document,
            runtime_document=runtime_document,
        ) + _prerequisite_candidates(
            policy_ref_ids=policy_ref_ids,
            runtime_document=runtime_document,
        )
        assertion_fields = _assertion_segments(source.get("then") or "")
        given_text = str(source.get("given") or "")
        condition_fields = (
            [str(source.get("when") or "")]
            if _normalized(given_text) in {"", "true"}
            else [given_text]
        )
        operation_fields = [str(source.get("when") or "")]
        candidates = []
        rejected_candidates = []
        source_operation_tokens = _operation_tokens(source.get("when") or "") | (
            _assertion_action_tokens(source.get("then") or "")
        )
        for item in raw:
            assertion = _best_match(assertion_fields, item["assertion_slots"])
            assertion_components = _component_matches(
                assertion_fields, item["assertion_slots"]
            )
            condition = _best_match(condition_fields, item["condition_slots"])
            operation = _best_match(
                operation_fields, list(item.get("operation_slots") or [])
            )
            accepted_operation_tokens = frozenset().union(
                *(
                    _operation_tokens(slot)
                    for slot in item.get("operation_slots") or []
                )
            )
            if (
                source_operation_tokens
                and accepted_operation_tokens
                and source_operation_tokens.isdisjoint(accepted_operation_tokens)
            ):
                rejected_candidates.append(
                    {
                        "reason_code": "source_operation_conflicts_with_runtime_family",
                        "candidate_kind": item["candidate_kind"],
                        "policy_ref_ids": item["policy_ref_ids"],
                        "path_test_specification_id": item.get(
                            "path_test_specification_id"
                        ),
                        "semantic_unit_ids": item["semantic_unit_ids"],
                        "source_operation_tokens": sorted(source_operation_tokens),
                        "accepted_operation_tokens": sorted(
                            accepted_operation_tokens
                        ),
                        "runtime_contract_refs": item["runtime_contract_refs"],
                    }
                )
                continue
            if assertion is None and condition is None:
                continue
            semantic_matches = {
                semantic_id: match
                for semantic_id, slots in item.get(
                    "semantic_assertion_slots", {}
                ).items()
                if semantic_id in set(item["semantic_unit_ids"])
                if (match := _best_match(assertion_fields, slots)) is not None
            }
            selected_semantic_ids: list[str] = []
            if semantic_matches:
                best_semantic_score = max(
                    float(match["score"]) for match in semantic_matches.values()
                )
                selected_semantic_ids = sorted(
                    semantic_id
                    for semantic_id, match in semantic_matches.items()
                    if best_semantic_score - float(match["score"]) < 0.25
                )
            quality = "complete" if assertion is not None else "path_only"
            atomic_binding = None
            coordinated_matches = {
                semantic_id: semantic_matches[semantic_id]
                for semantic_id in selected_semantic_ids
                if semantic_matches[semantic_id]["basis"]
                == "coordinated_relation_member_signature"
            }
            if coordinated_matches:
                context_tokens = _tokens(" ".join(condition_fields))
                member_tokens = sorted(
                    set().union(
                        *(
                            set(match["shared_signature_tokens"])
                            for match in coordinated_matches.values()
                        )
                    )
                    - set(context_tokens)
                )
                atomic_binding = {
                    "kind": "relation_member_signature",
                    "parent_semantic_unit_ids": sorted(coordinated_matches),
                    "member_signature_tokens": member_tokens,
                    "selection_matches": coordinated_matches,
                }
                quality = "complete_atomic"
            elif selected_semantic_ids and set(selected_semantic_ids) < set(
                item["semantic_unit_ids"]
            ):
                atomic_binding = {
                    "kind": "semantic_unit_subset",
                    "selected_semantic_unit_ids": selected_semantic_ids,
                    "selection_matches": {
                        semantic_id: semantic_matches[semantic_id]
                        for semantic_id in selected_semantic_ids
                    },
                }
                quality = "complete_atomic"
            elif (
                assertion is not None
                and selected_semantic_ids
                and assertion["basis"]
                in {
                    "source_field_contained_in_accepted_slot",
                    "coordinated_relation_member_signature",
                }
                and assertion["source_token_count"] < assertion["accepted_token_count"]
            ):
                atomic_binding = {
                    "kind": "relation_member_signature",
                    "parent_semantic_unit_ids": selected_semantic_ids,
                    "member_signature_tokens": assertion[
                        "shared_signature_tokens"
                    ],
                    "selection_match": assertion,
                }
                quality = "complete_atomic"
            candidate = deepcopy(item)
            candidate.pop("assertion_slots", None)
            candidate.pop("semantic_assertion_slots", None)
            candidate.pop("condition_slots", None)
            candidate.pop("operation_slots", None)
            candidate["selected_semantic_unit_ids"] = selected_semantic_ids
            candidate["atomic_binding"] = atomic_binding
            candidate["projection_quality"] = quality
            candidate["match_evidence"] = {
                "assertion": assertion,
                "assertion_components": assertion_components,
                "condition": condition,
                "operation": operation,
            }
            candidate["selection_score"] = round(
                (float(assertion["score"]) if assertion else 0.0)
                + 0.65
                * (
                    float(condition["score"])
                    + 2.0 * float(condition["source_coverage"])
                    if condition
                    else 0.0
                )
                + 0.35 * (float(operation["score"]) if operation else 0.0),
                6,
            )
            if item["candidate_kind"] == "path_contract" and condition is not None:
                source_is_negated = bool(
                    re.search(r"\b(?:not|no|without)\b", given_text, re.IGNORECASE)
                )
                activation_is_negated = bool(
                    item.get("activation_expression_is_negated")
                )
                truth_value = item.get("condition_truth_value")
                expected_source_negation = activation_is_negated
                if truth_value is False:
                    expected_source_negation = not expected_source_negation
                polarity_matches = source_is_negated == expected_source_negation
                candidate["match_evidence"]["condition_polarity"] = {
                    "source_is_negated": source_is_negated,
                    "activation_expression_is_negated": activation_is_negated,
                    "condition_truth_value": truth_value,
                    "matches": polarity_matches,
                }
                candidate["selection_score"] = round(
                    float(candidate["selection_score"])
                    + (0.8 if polarity_matches else -0.8),
                    6,
                )
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                -float(item["selection_score"]),
                item["candidate_kind"],
                str(item.get("path_test_specification_id") or ""),
                tuple(item["semantic_unit_ids"]),
                tuple(item["policy_ref_ids"]),
            )
        )
        if not candidates:
            status = (
                "operation_scope_conflict"
                if rejected_candidates
                else "runtime_projection_missing"
            )
        else:
            best = float(candidates[0]["selection_score"])
            ranked_candidates = candidates
            candidates = [
                item for item in candidates if best - float(item["selection_score"]) < 1.0
            ]
            candidates = [
                item
                for item in candidates
                if best - float(item["selection_score"]) < 0.5
            ]
            assertion_text = _normalized(source.get("then") or "")
            has_composition_marker = any(
                marker in f" {assertion_text} "
                for marker in (" and ", " before ", " after ")
            )
            if has_composition_marker:
                covered_tokens = set().union(
                    *(
                        _candidate_assertion_signature(item)
                        for item in candidates
                    )
                )
                for item in ranked_candidates:
                    if item in candidates:
                        continue
                    signature = _candidate_assertion_signature(item)
                    if not signature:
                        continue
                    if signature - covered_tokens and covered_tokens - signature:
                        candidates.append(item)
                        covered_tokens.update(signature)
                candidates = [
                    candidate
                    for candidate in candidates
                    if not any(
                        _candidate_assertion_signature(candidate)
                        < _candidate_assertion_signature(other)
                        for other in candidates
                        if other is not candidate
                    )
                ]
            qualities = {item["projection_quality"] for item in candidates}
            identities = {
                (
                    item["candidate_kind"],
                    tuple(item["semantic_unit_ids"]),
                    item.get("path_test_specification_id"),
                )
                for item in candidates
            }
            if len(identities) > 1:
                signatures = [
                    _candidate_assertion_signature(item)
                    for item in candidates
                ]
                contributes_distinct_requirements = all(
                    left - right and right - left
                    for index, left in enumerate(signatures)
                    for right in signatures[index + 1 :]
                    if left and right
                )
                status = (
                    "resolved_composite"
                    if has_composition_marker and contributes_distinct_requirements
                    else "ambiguous"
                )
            elif "complete_atomic" in qualities:
                status = "resolved_atomic"
            elif "path_only" in qualities:
                status = "path_only"
            elif len(identities) == 1:
                status = "resolved_unique"
            else:
                status = "ambiguous"

    requires_condition_scope_review = any(
        _source_condition_is_broader_disjunction(candidate)
        for candidate in candidates
    )
    if status in {"ambiguous", "path_only"} or requires_condition_scope_review:
        (
            status,
            candidates,
            adjudication_rejections,
            adjudication,
        ) = _adjudicate_candidates(
            source,
            candidates,
            path_document,
            runtime_document,
            accepted_policy_statements,
        )
        rejected_candidates.extend(adjudication_rejections)
    else:
        adjudication = {
            "status": "not_required",
            "reason_codes": ["projection_was_not_ambiguous"],
        }

    payload = {
        "schema_version": RUNTIME_PROJECTION_REPORT_SCHEMA_VERSION,
        "source_record_fingerprint": source["source_record_fingerprint"],
        "source_branch_id": source["branch_id"],
        "policy_lineage_report_fingerprint": policy[
            "policy_lineage_report_fingerprint"
        ],
        "status": status,
        "candidates": candidates,
        "rejected_candidates": rejected_candidates,
        "adjudication": adjudication,
        "llm_calls": 0,
    }
    payload["runtime_projection_report_fingerprint"] = content_sha256(payload)
    return payload


def validate_runtime_projection_report(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeProjectionError("$runtime_projection_report must be an object")
    item = deepcopy(dict(value))
    if item.get("schema_version") != RUNTIME_PROJECTION_REPORT_SCHEMA_VERSION:
        raise RuntimeProjectionError("unsupported runtime projection report schema")
    expected = item.pop("runtime_projection_report_fingerprint", None)
    actual = content_sha256(item)
    if expected != actual:
        raise RuntimeProjectionError("runtime projection report fingerprint mismatch")
    item["runtime_projection_report_fingerprint"] = actual
    return item


def _legacy_quality_rejections(
    target: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for evidence in target.get("resolution_evidence") or []:
        if evidence.get("evidence_kind") != "resolver_registry_trace":
            continue
        for attempt in evidence.get("attempts") or []:
            for diagnostic in attempt.get("diagnostics") or []:
                if (
                    diagnostic.get("code")
                    == "unique_lexical_candidate_rejected_by_quality_gate"
                ):
                    result.append(deepcopy(dict(diagnostic)))
    return result


def runtime_projection_admission(
    source_spec_record: Mapping[str, Any],
    legacy_target: Mapping[str, Any],
    runtime_projection_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide whether a diagnostic projection may become an accepted target.

    This is deliberately stricter than projection discovery.  Discovery may
    retain ambiguous, path-only, or conflict candidates for diagnosis; target
    materialization accepts either one runtime-observable branch or one
    mechanically proven set of complementary branches.  It never converts an
    ambiguous candidate set into a composite target.
    """

    source = validate_source_spec_record(source_spec_record)
    target = validate_resolved_evaluation_target(
        legacy_target, source_spec_record=source
    )
    report = validate_runtime_projection_report(runtime_projection_report)
    if report["source_record_fingerprint"] != source["source_record_fingerprint"]:
        raise RuntimeProjectionError(
            "runtime projection report does not belong to source"
        )
    reasons = []
    if target["resolution_status"] != "source_grounded":
        reasons.append("legacy_target_is_not_source_grounded")
    if report["status"] not in {
        "resolved_unique",
        "resolved_atomic",
        "resolved_composite",
        "resolved_alternative",
    }:
        reasons.append("projection_status_is_not_uniquely_materializable")
    candidates = list(report.get("candidates") or [])
    if report["status"] == "resolved_composite":
        if len(candidates) < 2:
            reasons.append("composite_projection_has_fewer_than_two_members")
        policy_sets = [
            set(candidate.get("policy_ref_ids") or [])
            for candidate in candidates
        ]
        if policy_sets and not set.intersection(*policy_sets):
            reasons.append("composite_projection_has_no_shared_policy_scope")
    elif report["status"] == "resolved_alternative":
        if len(candidates) < 2:
            reasons.append("alternative_projection_has_fewer_than_two_members")
    elif len(candidates) != 1:
        reasons.append("projection_does_not_have_exactly_one_candidate")
    for candidate in candidates:
        if candidate.get("projection_quality") not in {
            "complete",
            "complete_atomic",
        }:
            reasons.append("projection_assertion_is_not_complete")
        if not candidate.get("runtime_contract_refs"):
            reasons.append("projection_has_no_runtime_contract")
        polarity = (candidate.get("match_evidence") or {}).get(
            "condition_polarity"
        )
        if isinstance(polarity, Mapping) and polarity.get("matches") is not True:
            reasons.append("projection_condition_polarity_conflicts")
        if candidate.get("projection_quality") != "complete_atomic":
            continue
        atomic = candidate.get("atomic_binding") or {}
        atomic_kind = atomic.get("kind")
        if atomic_kind == "semantic_unit_subset":
            if not atomic.get("selected_semantic_unit_ids"):
                reasons.append("atomic_semantic_subset_is_empty")
        elif atomic_kind == "relation_member_signature":
            if not atomic.get("parent_semantic_unit_ids"):
                reasons.append("atomic_relation_parent_is_missing")
            if not atomic.get("member_signature_tokens"):
                reasons.append("atomic_relation_member_signature_is_empty")
        elif atomic_kind == "source_evidence_action":
            observation_target = atomic.get("observation_target") or {}
            grounding = atomic.get("grounding") or {}
            if not atomic.get("source_action"):
                reasons.append("source_evidence_action_is_missing")
            if (
                observation_target.get("kind")
                != "assistant_response_semantic_action"
                or not observation_target.get("action_text")
            ):
                reasons.append(
                    "source_evidence_action_observation_target_is_missing"
                )
            if not grounding.get("source_clause") or not grounding.get(
                "accepted_policy_clause"
            ):
                reasons.append("source_evidence_action_grounding_is_incomplete")
        else:
            reasons.append("atomic_binding_kind_is_not_supported")
    quality_rejections = _legacy_quality_rejections(target)
    disallowed_quality_reasons = sorted(
        {
            str(reason)
            for rejection in quality_rejections
            for reason in rejection.get("failure_reason_codes") or []
            if reason != "source_branch_is_narrower_than_compound_contract"
        }
    )
    if disallowed_quality_reasons:
        reasons.append("legacy_semantic_quality_rejection_is_not_repaired")
    return {
        "admitted": not reasons,
        "reason_codes": reasons,
        "projection_status": report["status"],
        "projection_candidate_count": len(candidates),
        "legacy_resolution_status": target["resolution_status"],
        "legacy_quality_rejections": quality_rejections,
        "disallowed_legacy_quality_reason_codes": disallowed_quality_reasons,
    }


def _verified_tool_names(
    candidate: Mapping[str, Any], resolved_artifacts: Mapping[str, Any]
) -> list[str]:
    runtime_document = artifact_document_for_capability(
        resolved_artifacts, CAP_EXECUTION_MATCHING_CONTRACTS
    )
    family_ids = {
        str(item.get("operation_policy_family_id"))
        for item in candidate.get("runtime_contract_refs") or []
        if item.get("operation_policy_family_id")
    }
    return sorted(
        {
            str(tool_name)
            for family in runtime_document.get("operation_family_contracts") or []
            if str(family.get("operation_policy_family_id")) in family_ids
            for tool_name in (
                (family.get("operation_selector") or {}).get(
                    "verified_tool_names"
                )
                or []
            )
        }
    )


def materialize_runtime_projection_target(
    source_spec_record: Mapping[str, Any],
    legacy_target: Mapping[str, Any],
    runtime_projection_report: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote one admitted policy projection to the downstream target API."""

    source = validate_source_spec_record(source_spec_record)
    target = validate_resolved_evaluation_target(
        legacy_target, source_spec_record=source
    )
    report = validate_runtime_projection_report(runtime_projection_report)
    if report["source_record_fingerprint"] != source["source_record_fingerprint"]:
        raise RuntimeProjectionError(
            "runtime projection report does not belong to source"
        )
    admission = runtime_projection_admission(source, target, report)
    if not admission["admitted"]:
        return target

    candidates = deepcopy(report["candidates"])
    candidate = candidates[0]
    legacy_binding = target.get("evaluation_binding") or {}
    runtime_refs = [
        deepcopy(runtime_ref)
        for member in candidates
        for runtime_ref in member.get("runtime_contract_refs") or []
    ]
    deduplicated_runtime_refs = {}
    for runtime_ref in runtime_refs:
        key = (
            runtime_ref.get("contract_kind"),
            runtime_ref.get("contract_id"),
            runtime_ref.get("event_requirement_id"),
            runtime_ref.get("operation_policy_family_id"),
        )
        deduplicated_runtime_refs.setdefault(key, runtime_ref)
    runtime_refs = list(deduplicated_runtime_refs.values())
    path_ids = sorted(
        {
            str(member.get("path_test_specification_id"))
            for member in candidates
            if member.get("path_test_specification_id")
        }
    )
    path_id = path_ids[0] if len(path_ids) == 1 else None
    semantic_ids = sorted(
        {
            str(semantic_id)
            for member in candidates
            for semantic_id in member.get("semantic_unit_ids") or []
        }
    )
    selected_semantic_ids = sorted(
        {
            str(semantic_id)
            for member in candidates
            for semantic_id in member.get("selected_semantic_unit_ids") or []
        }
    )
    runtime_contract_ids = sorted(
        {
            str(item.get("contract_id"))
            for item in runtime_refs
            if item.get("contract_id")
        }
    )
    accepted_refs: dict[str, Any] = {
        "policy_ref_ids": sorted(
            {
                str(policy_ref_id)
                for member in candidates
                for policy_ref_id in member.get("policy_ref_ids") or []
            }
        ),
        "semantic_unit_ids": semantic_ids,
        "runtime_contract_ids": runtime_contract_ids,
        "runtime_projection_report_fingerprint": report[
            "runtime_projection_report_fingerprint"
        ],
    }
    if path_ids:
        accepted_refs["path_test_specification_ids"] = path_ids
    if selected_semantic_ids:
        accepted_refs["selected_semantic_unit_ids"] = selected_semantic_ids

    binding = {
        "binding_id": (
            f"policy-runtime-projection::{source['branch_id']}::"
            f"{str(path_id or report['status'])}"
        ),
        "target_archetype": legacy_binding.get(
            "target_archetype", "source_assertion"
        ),
        "evaluation_subject": {
            "kind": "accepted_policy_runtime_projection",
            "candidate_kind": (
                candidate.get("candidate_kind")
                if len(candidates) == 1
                else (
                    "alternatives"
                    if report["status"] == "resolved_alternative"
                    else "composite"
                )
            ),
            "policy_ref_ids": accepted_refs["policy_ref_ids"],
            "path_test_specification_id": path_id,
            "semantic_unit_ids": semantic_ids,
            "selected_semantic_unit_ids": selected_semantic_ids,
            "verified_tool_names": sorted(
                {
                    tool_name
                    for member in candidates
                    for tool_name in _verified_tool_names(
                        member, resolved_artifacts
                    )
                }
            ),
            "projection_members": [
                {
                    "candidate_kind": member.get("candidate_kind"),
                    "policy_ref_ids": deepcopy(member.get("policy_ref_ids") or []),
                    "path_test_specification_id": member.get(
                        "path_test_specification_id"
                    ),
                    "semantic_unit_ids": deepcopy(
                        member.get("semantic_unit_ids") or []
                    ),
                    "runtime_contract_refs": deepcopy(
                        member.get("runtime_contract_refs") or []
                    ),
                }
                for member in candidates
            ],
        },
        "assertion_contract": {
            "kind": (
                "accepted_policy_scoped_composite_projection"
                if report["status"] == "resolved_composite"
                else (
                    "accepted_policy_scoped_alternative_projection"
                    if report["status"] == "resolved_alternative"
                    else "accepted_policy_scoped_branch_projection"
                )
            ),
            "given": source["given"],
            "when": source["when"],
            "then": source["then"],
            "deontic": source["deontic"],
            "projection_status": report["status"],
            "condition_truth_value": (
                candidate.get("condition_truth_value")
                if len(candidates) == 1
                else None
            ),
            "projection_quality": (
                candidate.get("projection_quality")
                if len(candidates) == 1
                else (
                    "complete_alternative"
                    if report["status"] == "resolved_alternative"
                    else "complete_composite"
                )
            ),
            "atomic_binding": (
                deepcopy(candidate.get("atomic_binding"))
                if len(candidates) == 1
                else None
            ),
            "member_assertions": [
                {
                    "condition_truth_value": member.get(
                        "condition_truth_value"
                    ),
                    "projection_quality": member.get("projection_quality"),
                    "atomic_binding": deepcopy(member.get("atomic_binding")),
                }
                for member in candidates
            ],
        },
        "evidence_contract": {
            "kind": "accepted_runtime_projection_evidence",
            "runtime_observable": True,
            "runtime_contract_refs": runtime_refs,
            "member_match_evidence": [
                deepcopy(member.get("match_evidence") or {})
                for member in candidates
            ],
        },
        "accepted_artifact_refs": accepted_refs,
    }
    evidence = [
        {
            "evidence_kind": "accepted_artifact_identity",
            **artifact_identity_view(resolved_artifacts),
        },
        {
            "evidence_kind": "policy_runtime_projection_migration",
            "runtime_projection_report_fingerprint": report[
                "runtime_projection_report_fingerprint"
            ],
            "legacy_target_fingerprint": target[
                "resolved_evaluation_target_fingerprint"
            ],
            "admission": admission,
        },
    ]
    return make_resolved_evaluation_target(
        source_spec_record=source,
        resolution_status="resolved_unique",
        resolver_id=POLICY_RUNTIME_PROJECTION_RESOLVER_ID,
        evaluation_binding=binding,
        resolution_evidence=evidence,
        diagnostics=[],
    )


__all__ = [
    "POLICY_RUNTIME_PROJECTION_RESOLVER_ID",
    "RUNTIME_PROJECTION_REPORT_SCHEMA_VERSION",
    "RuntimeProjectionError",
    "materialize_runtime_projection_target",
    "resolve_runtime_projection",
    "runtime_projection_admission",
    "validate_runtime_projection_report",
]
