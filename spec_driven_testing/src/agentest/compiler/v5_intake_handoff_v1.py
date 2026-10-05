"""First-step source handoff overlay; preserves initial intake diagnostics."""

from collections import Counter
from copy import deepcopy

from .artifacts import content_sha256


def relation_handoff(evidence):
    """Separate upstream relationship assertions from runtime proof."""
    expected = deepcopy(evidence["user_requirements"])
    lookup = deepcopy(evidence["condition_matches"])
    known_expected = expected.get("present") and expected.get("value") in ("owner", "not_owner")
    known_lookup = lookup.get("present") and lookup.get("value") in ("owner", "not_owner")
    if known_expected and not lookup.get("present"):
        status = "expected_relation_only_lookup_field_missing"
    elif known_expected and known_lookup and expected["value"] == lookup["value"]:
        status = "upstream_labels_agree_not_runtime_verified"
    elif known_expected and known_lookup:
        status = "conflicting_upstream_labels"
    else:
        status = "unspecified_or_null_requires_review"
    return {
        "expected_relation_assertion": expected,
        "lookup_relation_assertion": lookup,
        "relationship_status": status,
        "runtime_verification": "not_performed",
        "downstream_obligation": "First confirm which kind of target object the relation applies to; if applicable, verify the relation when binding the concrete user and object. Agreement between the two upstream labels is not runtime proof.",
    }


def _when_handoff(branch, review):
    actual_when = branch["source_fields"]["gwt"]["when"]
    user = branch["user_requirements"]
    if actual_when != review["reviewed_when"] or user.get("errors") != review["expected_errors"]:
        raise ValueError("When review no longer matches source text or diagnostics")
    if user.get("trigger", {}).get("trigger") != actual_when:
        raise ValueError("review assumes trigger retains the full When")
    if user.get("when_qualifiers") != []:
        raise ValueError("review assumes the upstream qualifier list is empty")
    spans = []
    for quote in review["source_quotes"]:
        if not quote or actual_when.count(quote) != 1:
            raise ValueError("review quote must be a unique verbatim source span")
        start = actual_when.index(quote)
        spans.append({"source_field": "source_fields.gwt.when", "start": start,
                      "end": start + len(quote), "offset_unit": "unicode_code_points",
                      "text": quote})
    if not spans:
        raise ValueError("When source review must supply evidence")
    return {
        "status": "source_information_retained_qualifier_materialization_pending",
        "authoritative_text": actual_when, "source_path": "source_fields.gwt.when",
        "trigger_equals_when": True,
        "upstream_when_qualifiers": deepcopy(user["when_qualifiers"]),
        "empty_qualifiers_mean_no_constraints": False,
        "source_quote_evidence": spans, "finding": review["finding"],
        "downstream_obligation": review["handoff_obligation"],
        "semantic_condition_compilation": "not_performed",
    }


def close_intake(intake, review):
    payload = deepcopy(intake)
    fingerprint = payload.pop("intake_fingerprint")
    if content_sha256(payload) != fingerprint or review["source_intake_fingerprint"] != fingerprint:
        raise ValueError("intake/closure source fingerprint mismatch")
    reviewed = {x["branch_id"]: x for x in review["when_reviews"]}
    if len(reviewed) != len(review["when_reviews"]):
        raise ValueError("duplicate When review IDs")
    branches_by_id = {x["branch_id"]: x for x in intake["branches"]}
    if not set(reviewed) <= branches_by_id.keys():
        raise ValueError("When review refers to an unknown branch")
    entries = []
    for branch in intake["branches"]:
        relation = relation_handoff(branch["relation_evidence"])
        when = _when_handoff(branch, reviewed[branch["branch_id"]]) if branch["branch_id"] in reviewed else None
        addressed, remaining = [], []
        for reason in branch["review_reasons"]:
            resolved_relation = (reason.get("code") == "missing_on_one_side" and reason.get("field") == "relation"
                                 and relation["relationship_status"] == "expected_relation_only_lookup_field_missing")
            resolved_when = (when is not None and reason.get("code") == "upstream_errors"
                             and reason.get("source") == "user_requirements"
                             and reason.get("value") == reviewed[branch["branch_id"]]["expected_errors"])
            if resolved_relation or resolved_when:
                addressed.append({"original_reason": deepcopy(reason),
                                  "disposition": "documented_expectation_without_lookup_proof" if resolved_relation else "use_retained_full_when_not_empty_qualifier_list"})
            else:
                remaining.append(deepcopy(reason))
        entries.append({
            "branch_id": branch["branch_id"],
            "source_branch_ref": {"source_intake_fingerprint": fingerprint, "branch_id": branch["branch_id"]},
            "original_intake_status": branch["intake_status"],
            "original_review_reasons": deepcopy(branch["review_reasons"]),
            "relation_handoff": relation, "when_handoff": when,
            "addressed_intake_reasons": addressed, "remaining_review_reasons": remaining,
            "handoff_status": "needs_input_review" if remaining else "ready_for_condition_review",
            "test_readiness": "not_assessed",
        })
    summary = {
        "branch_count": len(entries),
        "handoff_status_counts": dict(sorted(Counter(x["handoff_status"] for x in entries).items())),
        "relation_status_counts": dict(sorted(Counter(x["relation_handoff"]["relationship_status"] for x in entries).items())),
        "addressed_relation_missing_count": sum(any(y["disposition"] == "documented_expectation_without_lookup_proof" for y in x["addressed_intake_reasons"]) for x in entries),
        "reviewed_when_error_branch_count": sum(x["when_handoff"] is not None for x in entries),
        "remaining_reason_counts": dict(sorted(Counter(y["code"] for x in entries for y in x["remaining_review_reasons"]).items())),
        "external_llm_calls": 0,
    }
    result = {"schema_version": "agentspectesting.v5-intake-handoff/v0.1",
              "source_intake_fingerprint": fingerprint,
              "source_review_fingerprint": content_sha256(review),
              "scope": "first-step closure overlay; not second-step implementation",
              "summary": summary, "branches": entries}
    result["handoff_fingerprint"] = content_sha256(result)
    return result


def render_handoff(result):
    import json

    lines = ["# Step 1 close-out acceptance: relation fields and When reference errors", "",
             "This report is a supplementary record to the original intake; it keeps the original diagnostics and does not overwrite the two upstream JSON files. ready_for_condition_review only means the source issues now have a basis for handling; it must not be read as the conditions or tests having passed.", "",
             "## 1. Result", "", "```json", json.dumps(result["summary"], ensure_ascii=False, indent=2), "```", "",
             "## 2. Handling rules for relation", "",
             "The user-requirement-side owner/not_owner is stored as the upstream expected relation; the same-named field on the candidate side is stored as a retrieval-side label. The 14 candidate-side missing fields are no longer treated as semantic conflicts, and owner is not backfilled. Actual ownership for every branch was not verified in Step 1; even when the two labels agree, later steps must still first determine which object the relation applies to, then verify it at binding time.", "",
             "## 3. Review of the two When entries", ""]
    for branch in result["branches"]:
        when = branch["when_handoff"]
        if when:
            lines += [f"### {branch['branch_id']}", "", when["finding"], "",
                      "Full When retained:", "", "> " + when["authoritative_text"], "",
                      "Handed to Step 2: " + when["downstream_obligation"], ""]
    lines += ["## 4. Input diagnostics still to be handled", "", "| branch | Reason |", "| --- | --- |"]
    for branch in result["branches"]:
        if branch["remaining_review_reasons"]:
            reasons = ", ".join(x["code"] for x in branch["remaining_review_reasons"])
            lines.append(f"| {branch['branch_id']} | {reasons} |")
    lines += ["", "The 5 warnings and 6 upstream untestability declarations are still retained and are handed to condition-evidence analysis. This round did not judge them testable or untestable. The equivalence assumptions and ambiguities from the earlier semantic review also remain in effect; this round did not approve reusing the old Oracle.", "",
              "## 5. What this round actually resolved", "",
              "Missing relation fields now have a clear downstream meaning; the two errors are confirmed to be structured-reference issues, and the source-text information is still complete. Step 2 must read the full When and must not treat an empty when_qualifiers as meaning no qualifiers. The original errors remain traceable; structured condition extraction has not yet been implemented.", ""]
    return "\n".join(lines)
