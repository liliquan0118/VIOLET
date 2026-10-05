"""Prepare lossless Given/When evidence for review; do not generate test inputs."""

from collections import Counter
from copy import deepcopy
import json
import re

from .artifacts import content_sha256


PATH_COMPONENT = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)(\[\]|\{\})?")
# count_gt/count_ge: real retail data (e.g. retail_044_arg#b0:
# orders.payment_history[].payment_method_id count_gt 0) compares the size
# of a collection at path against value -- a real operator family the
# airline generation never used. This module only validates operator
# syntax, never operator semantics (interpreting "count of X" happens at a
# later binding stage), so admitting these two tokens here is the same
# narrow fix as admitting a real new enum value anywhere else in this file.
OPS = {"eq", "ne", "gt", "ge", "lt", "le", "in", "not_in", "contains", "exists", "count_gt", "count_ge"}
TIME_UNITS = {"seconds", "minutes", "hours", "days", "weeks"}


def verify_fingerprint(value, key):
    payload = deepcopy(value)
    expected = payload.pop(key)
    if content_sha256(payload) != expected:
        raise ValueError(f"{key} mismatch")
    return expected


def source_spans(text, quote):
    """Exact string matching only; absence is not a semantic mismatch judgment."""
    starts = []
    cursor = 0
    if isinstance(quote, str) and quote:
        while True:
            start = text.find(quote, cursor)
            if start < 0:
                break
            starts.append(start)
            cursor = start + 1
    return {
        "status": "exact_unique" if len(starts) == 1 else "exact_multiple" if starts else "not_verbatim",
        "spans": [{"start": start, "end": start + len(quote), "text": quote,
                   "offset_unit": "unicode_code_points"} for start in starts],
        "semantic_equivalence": "not_assessed",
    }


def parse_database_condition(raw):
    """Validate a limited expression grammar, not schema, meaning or runtime truth."""
    issues = []
    if not isinstance(raw, dict):
        return {"source_condition": deepcopy(raw), "syntax_status": "needs_review",
                "issues": ["condition_not_object"], "expression": None}
    required = {"table", "path", "op", "value"}
    if not required <= raw.keys():
        issues.append("missing_condition_fields")
    if set(raw) - (required | {"quant"}):
        issues.append("uninterpreted_condition_fields")
    if not isinstance(raw.get("table"), str) or not raw["table"].strip():
        issues.append("invalid_table")
    steps = []
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        issues.append("invalid_path")
    else:
        for component in path.split("."):
            match = PATH_COMPONENT.fullmatch(component)
            if not match:
                issues.append("unsupported_path_syntax")
                break
            steps.append({"field": match[1], "traversal": {
                None: "field", "[]": "array_items", "{}": "object_values"}[match[2]]})
    op = raw.get("op")
    if not isinstance(op, str) or op not in OPS:
        issues.append("unsupported_operator")
    if "quant" in raw and raw["quant"] not in ("all", "any"):
        issues.append("unsupported_quantifier")
    value = raw.get("value")
    value_expression = {"kind": "literal", "value": deepcopy(value)}
    if isinstance(value, dict):
        # "offset" is optional -- real telecom data uses a bare {"ref":
        # "$TODAY"} (no arithmetic, just "compare against today"), which the
        # airline generation never needed (its only real relative-time
        # conditions always carried a real, non-empty day/hour offset). A
        # missing offset means zero, not "unsupported" -- $NOW/$TODAY alone
        # is a real, meaningful comparison. "$TODAY" is accepted alongside
        # "$NOW" as a second real reference token (both resolve against the
        # same real bound clock downstream; which literal word the upstream
        # generation used is preserved, not silently collapsed to "$NOW").
        ref = value.get("ref")
        offset = value.get("offset", {})
        valid = (set(value) <= {"ref", "offset"} and ref in ("$NOW", "$TODAY")
                 and isinstance(offset, dict) and set(offset) <= TIME_UNITS
                 and all(type(number) in (int, float) for number in offset.values()))
        if not valid:
            issues.append("unsupported_value_expression")
        else:
            value_expression = {"kind": "relative_test_time", "ref": ref,
                                "offset": deepcopy(offset), "clock_binding": "pending"}
    elif isinstance(value, list):
        if any(type(item) not in (str, int, float, bool, type(None)) for item in value):
            issues.append("unsupported_literal_list")
    elif type(value) not in (str, int, float, bool, type(None)):
        issues.append("unsupported_literal")
    if op in ("in", "not_in") and not isinstance(value, list):
        issues.append("membership_requires_list")
    if op == "exists" and type(value) is not bool:
        issues.append("exists_requires_boolean")
    has_collection = any(step["traversal"] != "field" for step in steps)
    expression = {
        "kind": "database_predicate",
        "subject": {"table": raw.get("table"), "path_steps": steps,
                    "object_binding": "pending"},
        "operator": op, "right": value_expression,
        "quantifier": {"present": "quant" in raw, **({"value": raw["quant"]} if "quant" in raw else {})},
    }
    return {
        "source_condition": deepcopy(raw),
        "syntax_status": "needs_review" if issues else "parsed",
        "issues": issues, "expression": None if issues else expression,
        "collection_scope_review_required": has_collection,
        "missing_collection_quantifier": has_collection and "quant" not in raw,
        "schema_validation": "not_performed", "runtime_validation": "not_performed",
        "given_alignment": "not_assessed",
    }


def _ref(intake, branch, field, item=None, *, expected):
    positions = branch["source_refs"]["user_requirements"]
    if len(positions) != 1:
        raise ValueError("admitted source is not uniquely located")
    index = positions[0]
    document = intake["sources"]["user_requirements"]["document"]
    if type(index) is not int or not 0 <= index < len(document):
        raise ValueError("invalid source index")
    if document[index]["gwt"]["branch_id"] != branch["branch_id"]:
        raise ValueError("source position and branch differ")
    actual = document[index]
    for key in field.split("/"):
        if not isinstance(actual, dict) or key not in actual:
            raise ValueError(f"source field missing: {field}")
        actual = actual[key]
    if item is not None:
        if not isinstance(actual, list) or type(item) is not int or not 0 <= item < len(actual):
            raise ValueError(f"source item missing: {field}[{item}]")
        actual = actual[item]
    # Canonical JSON comparison distinguishes true from 1 and preserves null.
    if content_sha256(actual) != content_sha256(expected):
        raise ValueError(f"source field content mismatch: {branch['branch_id']}:{field}")
    pointer = f"/sources/user_requirements/document/{index}/{field}"
    return {"intake_fingerprint": intake["intake_fingerprint"],
            "json_pointer": pointer if item is None else f"{pointer}/{item}"}


def _prepare_branch(intake, branch, selection):
    fields, user = branch["source_fields"], branch["user_requirements"]
    # Validate entire collections as well as individual references: an omitted
    # item or an emptied list otherwise has no per-item reference to check.
    for key in ("spec_id", "kind", "origin", "rule_text", "gwt", "root",
                "DBconditions", "nonDBconditions"):
        _ref(intake, branch, key, expected=fields[key])
    for key in ("trigger", "conversation_preconditions", "when_qualifiers"):
        _ref(intake, branch, key, expected=user[key])
    branch_id = branch["branch_id"]
    given, when = fields["gwt"]["given"], fields["gwt"]["when"]
    db, non_db, conversation, qualifiers = [], [], [], []
    for index, raw in enumerate(fields["DBconditions"]):
        db.append({"condition_id": f"{branch_id}::DB::{index + 1}",
                   "source_ref": _ref(intake, branch, "DBconditions", index, expected=raw),
                   **parse_database_condition(raw)})
    for index, raw in enumerate(fields["nonDBconditions"]):
        non_db.append({
            "condition_id": f"{branch_id}::NONDB::{index + 1}",
            "source_ref": _ref(intake, branch, "nonDBconditions", index, expected=raw),
            "source_condition": deepcopy(raw),
            "given_quote_match": source_spans(given, raw["clause"]),
            "evidence_source": "needs_review",
            "upstream_realizable_is_proof": False,
        })
    for index, raw in enumerate(user["conversation_preconditions"]):
        links = [x["condition_id"] for x in non_db
                 if raw["source_clause"] == x["source_condition"]["clause"]]
        conversation.append({
            "requirement_id": f"{branch_id}::CONVERSATION::{index + 1}",
            "source_ref": _ref(intake, branch, "conversation_preconditions", index, expected=raw),
            "upstream_requirement": deepcopy(raw),
            "exact_nonDB_links": links,
            "link_status": "exact_unique" if len(links) == 1 else "exact_multiple" if links else "not_exactly_linked",
            "given_quote_match": source_spans(given, raw["source_clause"]),
            "user_information_vs_business_fact": "needs_review",
            "satisfies_given": "not_assessed",
        })
    for index, raw in enumerate(user["when_qualifiers"]):
        qualifiers.append({
            "requirement_id": f"{branch_id}::WHEN_QUALIFIER::{index + 1}",
            "source_ref": _ref(intake, branch, "when_qualifiers", index, expected=raw),
            "upstream_requirement": deepcopy(raw),
            "when_quote_match": source_spans(when, raw["source_quote"]),
            "interpretation": "needs_review",
        })
    constant = given.strip().casefold() == "true"
    constant_conflict = constant and bool(db or non_db or conversation)
    resolved_constant = constant and not constant_conflict
    handoff_when = selection["handoff"]["when_handoff"]
    if handoff_when and handoff_when["authoritative_text"] != when:
        raise ValueError("When handoff differs from source")
    tasks = []
    if not resolved_constant:
        tasks.append("given_condition_coverage_and_logic")
    if non_db or conversation:
        tasks.append("user_information_and_fact_evidence")
    if any(x.get("collection_scope_review_required") for x in db):
        tasks.append("collection_object_date_and_quantifier_scope")
    if any(x.get("expression", {}).get("right", {}).get("kind") == "relative_test_time"
           for x in db if x.get("expression")):
        tasks.append("test_clock_binding")
    tasks.append("when_event_and_qualifier_coverage")
    proposed_trigger = user["trigger"]["trigger"]
    if when != proposed_trigger:
        tasks.append("raw_when_vs_suggested_user_trigger")
    structural_issues = (["true_given_has_attached_conditions"] if constant_conflict else [])
    for item in db:
        structural_issues.extend(f"{item['condition_id']}:{code}" for code in item["issues"])
    result = {
        "branch_id": branch_id, "spec_id": fields["spec_id"],
        "source_context": {"rule_text": fields["rule_text"], "origin": fields["origin"],
                           "kind": fields["kind"], "gwt": deepcopy(fields["gwt"])},
        "given": {
            "source_text": given, "source_ref": _ref(intake, branch, "gwt/given", expected=given),
            "logic_status": "constant_resolved" if resolved_constant else "needs_review",
            "logical_expression": {"kind": "constant", "value": True} if resolved_constant else None,
            "database_conditions": db, "non_database_conditions": non_db,
            "conversation_requirements": conversation,
            "missing_upstream_mapping": not constant and not db and not non_db,
            "candidate_lists_imply_conjunction": False,
        },
        "when": {
            "source_text": when, "source_ref": _ref(intake, branch, "gwt/when", expected=when),
            "suggested_user_trigger_text": proposed_trigger,
            "suggested_trigger_ref": _ref(intake, branch, "trigger/trigger", expected=proposed_trigger),
            "trigger_text_identical": when == proposed_trigger,
            "qualifier_requirements": qualifiers,
            "empty_qualifiers_mean_no_constraints": False,
            "event_and_qualifier_resolution": "needs_review",
            "source_handoff": deepcopy(handoff_when),
        },
        "relation_handoff": deepcopy(selection["handoff"]["relation_handoff"]),
        "version_semantics": deepcopy(selection["version_semantics"]),
        "review_tasks": tasks, "structural_issues": structural_issues,
        "preparation_status": "needs_structure_review" if structural_issues else "prepared_for_semantic_review",
        "condition_compilation_status": "incomplete",
        "old_oracle_reuse_authorized": False, "test_readiness": "not_assessed",
        "source_branch_fingerprint": content_sha256(branch),
    }
    result["preparation_fingerprint"] = content_sha256(result)
    return result


def prepare_given_when(intake, scope):
    verify_fingerprint(intake, "intake_fingerprint")
    verify_fingerprint(scope, "scope_fingerprint")
    if scope["source_intake_fingerprint"] != intake["intake_fingerprint"]:
        raise ValueError("scope/intake source mismatch")
    sources = {x["branch_id"]: x for x in intake["branches"]}
    selections = scope["admitted"] + scope["deferred"]
    ids = [x["branch_id"] for x in selections]
    if len(sources) != len(intake["branches"]) or len(ids) != len(set(ids)) or set(ids) != sources.keys():
        raise ValueError("scope partition differs from source branches")
    for group, expected in ((scope["admitted"], "ready_for_condition_review"),
                            (scope["deferred"], "needs_input_review")):
        for selection in group:
            handoff = selection["handoff"]
            if handoff["handoff_status"] != expected or bool(handoff["remaining_review_reasons"]) != (expected == "needs_input_review"):
                raise ValueError("invalid scope admission")
            if handoff["branch_id"] != selection["branch_id"] or selection["source_refs"] != sources[selection["branch_id"]]["source_refs"]:
                raise ValueError("scope branch reference mismatch")
    branches = [_prepare_branch(intake, sources[x["branch_id"]], x) for x in scope["admitted"]]
    counts = lambda key: dict(sorted(Counter(x[key] for x in branches).items()))
    db = [c for x in branches for c in x["given"]["database_conditions"]]
    non_db = [c for x in branches for c in x["given"]["non_database_conditions"]]
    conversation = [c for x in branches for c in x["given"]["conversation_requirements"]]
    qualifiers = [c for x in branches for c in x["when"]["qualifier_requirements"]]
    result = {
        "schema_version": "agentspectesting.v5-given-when-preparation/v0.1",
        "scope": "step 2A: expression and evidence preparation; not final Given/When contracts",
        "source_intake_fingerprint": intake["intake_fingerprint"],
        "source_scope_fingerprint": scope["scope_fingerprint"],
        "branches": branches, "deferred_branch_ids": [x["branch_id"] for x in scope["deferred"]],
        "summary": {
            "branch_count": len(branches), "deferred_branch_count": len(scope["deferred"]),
            "given_logic_status_counts": dict(Counter(x["given"]["logic_status"] for x in branches)),
            "preparation_status_counts": counts("preparation_status"),
            "database_condition_count": len(db), "database_syntax_status_counts": dict(Counter(c["syntax_status"] for c in db)),
            "non_database_condition_count": len(non_db),
            "conversation_requirement_count": len(conversation),
            "conversation_link_counts": dict(Counter(c["link_status"] for c in conversation)),
            "when_qualifier_count": len(qualifiers),
            "when_quote_match_counts": dict(Counter(c["when_quote_match"]["status"] for c in qualifiers)),
            "raw_when_trigger_difference_count": sum(not x["when"]["trigger_text_identical"] for x in branches),
            "collection_scope_review_condition_count": sum(c.get("collection_scope_review_required", False) for c in db),
            "missing_collection_quantifier_count": sum(c.get("missing_collection_quantifier", False) for c in db),
            "compiled_complete_branch_count": 0, "external_llm_calls": 0,
        },
    }
    result["preparation_set_fingerprint"] = content_sha256(result)
    return result


def render_preparation(result):
    lines = ["# Step 2 (2A): Given/When structure preparation results", "",
             "Source text, structured representations, and review items have been prepared; no requests were generated, no database was operated on, and no tests were executed. prepared_for_semantic_review does not mean the conditions are correct or executable.", "",
             "## Summary", "", "```json", json.dumps(result["summary"], ensure_ascii=False, indent=2), "```", "",
             "## Branch review index", "", "| Branch | Given | DB / non-DB / conversation / When qualifier counts | Follow-up review items |", "| --- | --- | --- | --- |"]
    for branch in result["branches"]:
        given, when = branch["given"], branch["when"]
        sizes = [len(given[name]) for name in ("database_conditions", "non_database_conditions", "conversation_requirements")]
        sizes.append(len(when["qualifier_requirements"]))
        lines.append(f"| {branch['branch_id']} | {given['logic_status']} | {' / '.join(map(str, sizes))} | {', '.join(branch['review_tasks'])} |")
    lines += ["", "See preparation.json in the same directory for the full Given/When, source locations, original quantifiers, reference matches, and pending-review reasons.", "",
              "Note: an exact string link only proves that the two upstream fields reference the same span of text; it does not prove that a single user utterance is enough to make the business fact hold.", ""]
    return "\n".join(lines)
