"""Select source-qualified branches for condition review, never execution."""

from copy import deepcopy

from .artifacts import content_sha256


def _verify(document, key):
    payload = deepcopy(document)
    fingerprint = payload.pop(key)
    if content_sha256(payload) != fingerprint:
        raise ValueError(f"{key} mismatch")
    return fingerprint


def _index(rows):
    indexed = {row["branch_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate branch IDs")
    return indexed


def build_scope(intake, handoff, semantics):
    """Preserve handoff obligations and version-review caveats in a scoped index."""
    source_hash = _verify(intake, "intake_fingerprint")
    handoff_hash = _verify(handoff, "handoff_fingerprint")
    if any(doc["source_intake_fingerprint"] != source_hash for doc in (handoff, semantics)):
        raise ValueError("source intake fingerprint mismatch")
    _verify(semantics, "review_fingerprint")
    sources = _index(intake["branches"])
    entries = _index(handoff["branches"])
    reviews = _index(semantics["reviews"])
    if sources.keys() != entries.keys() or not reviews.keys() <= sources.keys():
        raise ValueError("source branch sets mismatch")
    shared = _index(intake["version_comparison"]["shared_id_comparisons"])
    if reviews.keys() != shared.keys():
        raise ValueError("semantic review branch set differs from shared source IDs")
    admitted, deferred = [], []
    for branch_id, entry in entries.items():
        expected_ref = {"source_intake_fingerprint": source_hash, "branch_id": branch_id}
        if entry["source_branch_ref"] != expected_ref:
            raise ValueError("branch source reference mismatch")
        status = entry["handoff_status"]
        reasons = entry["remaining_review_reasons"]
        if status not in ("ready_for_condition_review", "needs_input_review"):
            raise ValueError("unknown handoff status")
        if (status == "needs_input_review") != bool(reasons):
            raise ValueError("handoff status and reasons disagree")
        review = reviews.get(branch_id)
        record = {
            "branch_id": branch_id,
            "spec_id": sources[branch_id]["source_fields"]["spec_id"],
            "source_refs": deepcopy(sources[branch_id]["source_refs"]),
            "handoff": deepcopy(entry),
            "version_semantics": ({
                "verdict": review["verdict"],
                "rationale": review.get("rationale"),
                "assumptions": deepcopy(review.get("assumptions", [])),
            } if review else {"verdict": "no_shared_id_review"}),
            "old_oracle_reuse_authorized": False,
            "condition_compilation": "not_performed",
            "test_readiness": "not_assessed",
        }
        (admitted if status == "ready_for_condition_review" else deferred).append(record)
    result = {
        "schema_version": "agentspectesting.v5-condition-review-scope/v0.1",
        "scope": "second-step input selection only; no condition compilation or execution",
        "source_intake_fingerprint": source_hash,
        "source_handoff_fingerprint": handoff_hash,
        "source_semantic_review_fingerprint": content_sha256(semantics),
        "selection_rule": "ready_for_condition_review with no remaining_review_reasons",
        "feedback_document": "docs/v5_upstream_feedback_pending_conditions.md",
        "summary": {
            "total_branch_count": len(entries),
            "admitted_branch_count": len(admitted),
            "deferred_branch_count": len(deferred),
            "admitted_spec_count": len({x["spec_id"] for x in admitted}),
            "deferred_spec_count": len({x["spec_id"] for x in deferred}),
            "external_llm_calls": 0,
        },
        "admitted": admitted,
        "deferred": deferred,
    }
    result["scope_fingerprint"] = content_sha256(result)
    return result
