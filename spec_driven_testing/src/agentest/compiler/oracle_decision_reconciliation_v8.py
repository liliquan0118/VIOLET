"""Deterministic reconciliation for complete v0.8 oracle decisions."""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping

from .oracle_decision_pipeline_v8 import (
    DECISIONS,
    decision_packet_members,
    validate_oracle_decision_packet_set,
)
from .oracle_requirement_pipeline_v7 import (
    RESPONSE_SET_VERSION as COMPATIBLE_RESPONSE_SET_VERSION,
    validate_oracle_requirement_packet_set,
)
from .then_atomization import ThenAtomizationError


def _norm(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", str(text).casefold()))


def _canonical_observation(candidate: Mapping[str, Any]) -> str:
    kind = candidate.get("candidate_kind")
    contract = candidate.get("observation_contract") or {}
    text = str(candidate.get("requirement_text") or "")
    if kind == "tool_call":
        return f"tool_call::{contract.get('tool_name')}"
    if kind == "assistant_literal":
        return f"assistant_literal::{_norm(contract.get('literal'))}"
    if kind == "semantic_requirement":
        tool = re.search(r"\btool\s+call\s+to\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.I)
        if tool:
            return f"tool_call::{tool.group(1)}"
        message = re.search(r"\bsend\s+the\s+message\s+(['\"])(.+?)\1", text, re.I)
        if message:
            return f"assistant_literal::{_norm(message.group(2))}"
    return str(candidate.get("candidate_key") or f"text::{_norm(text)}")


def _parameter_count(candidate: Mapping[str, Any]) -> int:
    contract = candidate.get("observation_contract") or {}
    if candidate.get("candidate_kind") == "tool_argument_constraint":
        return len(contract.get("parameters") or [])
    if candidate.get("candidate_kind") == "tool_argument":
        return 1
    return 0


def _specificity(candidate: Mapping[str, Any]) -> tuple[int, int, str]:
    scores = {
        "tool_argument_constraint": 60,
        "tool_argument": 60,
        "temporal_relation": 60,
        "tool_call": 50,
        "assistant_literal": 50,
        "semantic_requirement": 10,
        "unbound_branch_assertion": 0,
    }
    # Within the tool_argument_constraint/tool_argument tier, a candidate that
    # binds MORE parameters is a strictly more complete restatement of the
    # same requirement_text than one binding fewer (e.g. a 3-parameter
    # composite vs. a 1-parameter scalar both restating one Then) -- without
    # this, two same-kind-scored candidates fall straight to the candidate_id
    # string tiebreak below, which has no relationship to completeness and
    # can pick the less complete one. Found by diffing this project's own
    # full-155-branch reconciliation output branch-by-branch after widening
    # candidate generation (see docs/oracle_requirement_pipeline_v0_7.md) --
    # 6 real branches, including one an existing Step4 mechanism depends on
    # by exact parameter list, lost their composite to a scalar this way.
    return (
        scores.get(str(candidate.get("candidate_kind")), 20),
        _parameter_count(candidate),
        str(candidate.get("candidate_id") or ""),
    )


def reconcile_oracle_decisions(
    candidate_packet_set: Mapping[str, Any],
    decision_packet_set: Mapping[str, Any],
    response_set: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = validate_oracle_requirement_packet_set(candidate_packet_set)
    decisions = validate_oracle_decision_packet_set(decision_packet_set)
    candidate_index = {
        (item["branch_id"], item["candidate_id"]): item for item in candidates["packets"]
    }
    decision_index = {}
    for packet in decisions["packets"]:
        for member in decision_packet_members(packet):
            expanded = deepcopy(packet)
            expanded.update(member)
            decision_index[(member["branch_id"], member["candidate_id"])] = expanded
    if set(candidate_index) != set(decision_index):
        raise ThenAtomizationError("v0.8 decision/candidate identities do not form a closed set")
    for identity, packet in decision_index.items():
        if (
            packet.get("source_candidate_packet_fingerprint")
            != candidate_index[identity].get("packet_fingerprint")
        ):
            raise ThenAtomizationError(f"v0.8 candidate lineage mismatch: {identity}")

    raw_responses = response_set.get("responses")
    if not isinstance(raw_responses, list):
        raise ThenAtomizationError("v0.8 responses must be an array")
    response_index: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in raw_responses:
        if not isinstance(raw, Mapping):
            raise ThenAtomizationError("v0.8 response must be an object")
        identity = (raw.get("branch_id"), raw.get("candidate_id"))
        if identity not in decision_index or identity in response_index:
            raise ThenAtomizationError(f"v0.8 response identity is unknown or duplicate: {identity}")
        if raw.get("decision") not in DECISIONS:
            raise ThenAtomizationError(f"v0.8 response decision is invalid: {identity}")
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ThenAtomizationError(f"v0.8 response reason is empty: {identity}")
        response_index[identity] = deepcopy(dict(raw))
    missing = sorted(set(decision_index) - set(response_index))
    if missing:
        raise ThenAtomizationError(f"v0.8 response set is incomplete: {missing}")

    reconciled = deepcopy(response_index)
    changes: list[dict[str, Any]] = []

    equivalent: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for identity, packet in decision_index.items():
        model_input = packet["model_input"]
        equivalent[
            (model_input["then_requirement"], model_input["candidate_observation"])
        ].append(identity)
    conflicted_branches: set[str] = set()
    for key, identities in equivalent.items():
        observed = {response_index[identity]["decision"] for identity in identities}
        if len(observed) <= 1:
            continue
        for identity in identities:
            previous = reconciled[identity]["decision"]
            reconciled[identity]["decision"] = "ambiguous"
            reconciled[identity]["reason"] = (
                "Equivalent model inputs received conflicting decisions; deterministic "
                "reconciliation requires adjudication."
            )
            conflicted_branches.add(identity[0])
            changes.append(
                {
                    "identity": list(identity),
                    "change_kind": "equivalent_input_conflict",
                    "from": previous,
                    "to": "ambiguous",
                    "equivalence_key": list(key),
                }
            )

    # A real gap found and fixed in this round (see
    # docs/oracle_requirement_pipeline_v0_7.md section 31): the conflict
    # check above only fires when members of the SAME rendered judge question
    # (`equivalent[key]`) DISAGREE. When they all agree "yes" -- the SAME
    # question was independently generated by two different candidate paths
    # in the SAME branch (e.g. a generic spec_binding restatement and a
    # specific exact-Then-text one) and both, correctly, got the same real
    # answer -- nothing previously deduplicated them: `deduplicate()` below
    # keys on the CANDIDATE's own `candidate_key`/`requirement_text`, which
    # differs between the two phrasings, so it never catches this shape
    # either. Both then materialize as separate accepted requirements for
    # the identical real observation, one of which is pure duplication (see
    # airline_133_state#b0v0/v1's OC01/OC08, OC02/OC10, OC03/OC09 -- found by
    # scripts/detect_redundant_requirements_v0_1.py, which correctly
    # identified them as content-redundant but could not be used to remove
    # them: flipping only one member's raw response away from "yes" makes it
    # disagree with its own literal duplicate-question sibling, which is
    # exactly what the conflict check above exists to catch).
    #
    # Fixed the same way the two existing dedup passes below already handle
    # a same-branch group of "yes" candidates that are duplicates of ONE
    # ANOTHER by some other measure: keep the most specific member
    # (_specificity, already used by `deduplicate()`), demote the rest to
    # "no". Deliberately scoped PER BRANCH within each equivalence group
    # (`equivalent[key]` can and does legitimately span MULTIPLE branches --
    # e.g. airline_037_arg#e0/airline_038_arg#e0/e1/e2 share several
    # identical rendered questions, each needing its OWN accepted
    # requirement since they are different branches; verified against the
    # real full corpus that grouping by branch here affects only the 3
    # airline_133 pairs and none of the other 12 real multi-member
    # same-question "yes" groups, all of which are legitimately cross-branch).
    for key, identities in equivalent.items():
        by_branch: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for identity in identities:
            by_branch[identity[0]].append(identity)
        for branch_id, branch_identities in by_branch.items():
            if len(branch_identities) <= 1:
                continue
            if any(reconciled[identity]["decision"] != "yes" for identity in branch_identities):
                continue
            keep = max(
                branch_identities,
                key=lambda identity: _specificity(
                    candidate_index[identity]["task_input"]["proposed_observation"]
                ),
            )
            for identity in branch_identities:
                if identity == keep:
                    continue
                reconciled[identity]["decision"] = "no"
                reconciled[identity]["reason"] = (
                    f"Deterministically deduplicated against {keep[1]} in the same branch "
                    "(identical rendered judge question)."
                )
                changes.append(
                    {
                        "identity": list(identity),
                        "change_kind": "same_rendered_question_duplicate",
                        "from": "yes",
                        "to": "no",
                        "kept_identity": list(keep),
                        "equivalence_key": list(key),
                    }
                )

    def deduplicate(group_kind: str, key_for: Any) -> None:
        grouped: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for identity, response in reconciled.items():
            if response["decision"] != "yes":
                continue
            candidate = candidate_index[identity]["task_input"]["proposed_observation"]
            key = key_for(candidate)
            if key:
                grouped[(identity[0], key)].append(identity)
        for (branch_id, key), identities in grouped.items():
            if len(identities) <= 1:
                continue
            keep = max(
                identities,
                key=lambda identity: _specificity(
                    candidate_index[identity]["task_input"]["proposed_observation"]
                ),
            )
            for identity in identities:
                if identity == keep:
                    continue
                reconciled[identity]["decision"] = "no"
                reconciled[identity]["reason"] = (
                    f"Deterministically deduplicated against {keep[1]} in the same branch."
                )
                changes.append(
                    {
                        "identity": list(identity),
                        "change_kind": group_kind,
                        "from": "yes",
                        "to": "no",
                        "kept_identity": list(keep),
                        "deduplication_key": key,
                    }
                )

    deduplicate("canonical_observation_duplicate", _canonical_observation)
    deduplicate(
        "same_requirement_duplicate",
        lambda candidate: _norm(candidate.get("requirement_text")),
    )

    ordered = [
        reconciled[(item["branch_id"], item["candidate_id"])]
        for item in candidates["packets"]
    ]
    counts = defaultdict(int)
    for item in ordered:
        counts[item["decision"]] += 1
    compatible = {
        "schema_version": COMPATIBLE_RESPONSE_SET_VERSION,
        "responses": ordered,
    }
    return {
        "schema_version": "agentspectesting.oracle-decision-reconciliation/v0.8",
        "compatible_response_set": compatible,
        "summary": {
            "response_count": len(ordered),
            "decision_counts": dict(sorted(counts.items())),
            "change_count": len(changes),
            "equivalent_input_conflict_change_count": sum(
                item["change_kind"] == "equivalent_input_conflict" for item in changes
            ),
            "deduplication_change_count": sum(
                item["change_kind"] != "equivalent_input_conflict" for item in changes
            ),
            "ready_branch_count": len(
                {identity[0] for identity in decision_index} - conflicted_branches
            ),
            "adjudication_branch_count": len(conflicted_branches),
        },
        "adjudication_branches": sorted(conflicted_branches),
        "changes": changes,
    }
