"""Assemble supplied test requirements; no semantic re-review or runtime proof.

This is a new step-2 contract, intentionally not a legacy compiled predicate
contract. Natural-language logic stays in its authoritative source text.
"""

from collections import Counter
from copy import deepcopy
import json

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import parse_database_condition, verify_fingerprint


SCHEMA = "agentspectesting.v5-given-when-assembly/v0.1"
SHARED_FIELDS = ("spec_id", "kind", "origin", "rule_text", "gwt_index", "gwt",
                 "root", "DBconditions", "nonDBconditions")


def _seal(value, key):
    value[key] = content_sha256(value)
    return value


def _index(rows):
    indexed = {row["branch_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate branch IDs")
    return indexed


def _source_row(intake, branch, source):
    positions = branch["source_refs"][source]
    document = intake["sources"][source]["document"]
    if len(positions) != 1 or type(positions[0]) is not int or not 0 <= positions[0] < len(document):
        raise ValueError("source row must be uniquely located")
    index = positions[0]
    row = document[index]
    if not isinstance(row, dict):
        raise ValueError("source row must be an object")
    gwt = row.get("gwt")
    source_id = gwt.get("branch_id") if isinstance(gwt, dict) else None
    # Missing/malformed GWT is a per-branch input error, reported below by
    # _validate_rows. A different explicit ID is still a lineage failure:
    # never silently attribute another branch's record to this selection.
    if isinstance(source_id, str) and source_id.strip() and source_id != branch["branch_id"]:
        raise ValueError("source row/branch identity mismatch")
    return row, {
        "intake_fingerprint": intake["intake_fingerprint"],
        "json_pointer": f"/sources/{source}/document/{index}",
        "record_fingerprint": content_sha256(row),
    }


def _field_ref(ref, field):
    return {**ref, "json_pointer": ref["json_pointer"] + "/" + field}


def _validate_rows(user, lookup, selection):
    """Only explicit structural gaps/conflicts, not inferred language differences."""
    issues = []
    def issue(code, field):
        issues.append({"code": code, "field": field})
    for field in SHARED_FIELDS:
        if field not in user or field not in lookup:
            issue("missing_shared_field", field)
        elif content_sha256(user[field]) != content_sha256(lookup[field]):
            issue("shared_field_mismatch", field)
    for label, row in (("user_requirements", user), ("condition_matches", lookup)):
        for field in ("spec_id", "kind", "origin", "rule_text"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                issue("required_text_missing", label + "/" + field)
        gwt = row.get("gwt")
        if not isinstance(gwt, dict):
            issue("gwt_not_object", label + "/gwt")
        else:
            for field in ("given", "when", "then", "branch_id"):
                if not isinstance(gwt.get(field), str) or not gwt[field].strip():
                    issue("required_text_missing", label + "/gwt/" + field)
        for field in ("DBconditions", "nonDBconditions"):
            if not isinstance(row.get(field), list):
                issue("required_list_missing", label + "/" + field)
        if "root" in row and row["root"] is not None and (not isinstance(row["root"], str) or not row["root"].strip()):
            issue("root_type_invalid", label + "/root")
        # A missing relation remains allowed by the source-step handoff.
        # This checks shape, not whether a supplied relation holds in data.
        if "relation" in row and (not isinstance(row["relation"], str) or not row["relation"].strip()):
            issue("relation_type_invalid", label + "/relation")
    if user.get("spec_id") != selection["spec_id"]:
        issue("scope_spec_mismatch", "spec_id")
    trigger = user.get("trigger")
    if not isinstance(trigger, dict) or not isinstance(trigger.get("trigger"), str) or not trigger["trigger"].strip():
        issue("user_trigger_missing", "trigger/trigger")
    for field, texts in (("conversation_preconditions", ("condition", "source_clause")),
                         ("when_qualifiers", ("condition", "source_quote")),
                         ("nonDBconditions", ("clause",))):
        items = user.get(field)
        if not isinstance(items, list):
            issue("required_list_missing", field)
        else:
            for index, item in enumerate(items):
                if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip() for k in texts):
                    issue("invalid_requirement_item", f"{field}/{index}")
    for index, item in enumerate(user.get("DBconditions", []) if isinstance(user.get("DBconditions"), list) else []):
        if not isinstance(item, dict) or not {"table", "path", "op", "value"} <= item.keys():
            issue("database_condition_fields_missing", f"DBconditions/{index}")
        elif any(not isinstance(item.get(k), str) or not item[k].strip() for k in ("table", "path", "op")):
            issue("database_condition_identifier_invalid", f"DBconditions/{index}")
    if not isinstance(lookup.get("lookup_status"), str) or not lookup["lookup_status"].strip():
        issue("lookup_status_missing", "lookup_status")
    if not isinstance(lookup.get("matches"), list):
        issue("candidate_list_missing", "matches")
    else:
        for index, candidate in enumerate(lookup["matches"]):
            pointer = f"condition_matches/matches/{index}"
            if not isinstance(candidate, dict):
                issue("candidate_not_object", pointer)
                continue
            # Current upstream candidate format carries user_id and root_id.
            # root=null legitimately has root_id=null (user-only anchors).
            for field in ("user_id", "root_id"):
                if field not in candidate:
                    issue("candidate_identifier_missing", pointer + "/" + field)
                    continue
                value = candidate[field]
                if field == "root_id" and lookup.get("root") is None and value is None:
                    continue
                if not isinstance(value, str) or not value.strip():
                    issue("candidate_identifier_invalid", pointer + "/" + field)
    if type(lookup.get("n_matches")) is not int or lookup["n_matches"] < 0:
        issue("candidate_count_invalid", "n_matches")
    # n_matches may describe the total search result, while matches is a sample.
    # Never require equality without an upstream enumeration guarantee.
    if isinstance(lookup.get("matches"), list) and type(lookup.get("n_matches")) is int and lookup["n_matches"] < len(lookup["matches"]):
        issue("candidate_count_less_than_supplied", "n_matches")
    if "relation" in user and "relation" in lookup and content_sha256(user["relation"]) != content_sha256(lookup["relation"]):
        issue("explicit_relation_conflict", "relation")
    return issues


def _assemble_branch(intake, branch, selection):
    user, user_ref = _source_row(intake, branch, "user_requirements")
    lookup, lookup_ref = _source_row(intake, branch, "condition_matches")
    issues = _validate_rows(user, lookup, selection)
    result = {
        "branch_id": branch["branch_id"], "spec_id": selection["spec_id"],
        "source_refs": {"user_requirements": user_ref, "condition_matches": lookup_ref},
        "input_validation": {"status": "blocked" if issues else "passed", "issues": issues,
                             "semantic_re_review_performed": False},
        "source_step_handoff": deepcopy(selection["handoff"]),
        "version_semantics": deepcopy(selection["version_semantics"]),
        "assembly_status": "blocked_input" if issues else "ready_for_binding",
        "test_readiness": "not_assessed", "runtime_validation": "not_performed",
        "old_oracle_reuse_authorized": False,
    }
    if issues:
        return _seal(result, "assembly_fingerprint")
    given, when = user["gwt"]["given"], user["gwt"]["when"]
    db = []
    for index, raw in enumerate(user["DBconditions"]):
        parsed = parse_database_condition(raw)
        # Unsupported local encoding is a consumer capability issue, not a
        # claim that the supplied upstream requirement is semantically wrong.
        db.append({"condition_id": f"{branch['branch_id']}::DB::{index + 1}",
                   "source_condition": deepcopy(raw),
                   "source_ref": _field_ref(user_ref, f"DBconditions/{index}"),
                   "optional_encoding": {"status": "available" if parsed["syntax_status"] == "parsed" else "not_supported",
                                         "expression": parsed["expression"], "issues": parsed["issues"]},
                   "evidence_requirement": "evaluate_on_bound_test_state", "satisfied": None})
    non_db = [{"condition_id": f"{branch['branch_id']}::NONDB::{index + 1}",
               "source_condition": deepcopy(raw),
               "source_ref": _field_ref(user_ref, f"nonDBconditions/{index}"),
               "satisfied": None}
              for index, raw in enumerate(user["nonDBconditions"])]
    def requirements(field):
        rows = []
        for index, raw in enumerate(user[field]):
            rows.append({"requirement": deepcopy(raw), "source_ref": _field_ref(user_ref, f"{field}/{index}"),
                         "realization_status": "not_bound",
                         "non_db_links": [x["condition_id"] for x in non_db
                                          if raw.get("source_clause") == x["source_condition"]["clause"]]})
        return rows
    result.update({
        "source_context": {k: deepcopy(user[k]) for k in ("rule_text", "origin", "kind", "gwt")},
        "given_requirements": {
            "text": given, "source_ref": _field_ref(user_ref, "gwt/given"),
            "logic": {"representation": "source_text", "text": given,
                      "structured_list_implies_and": False, "formal_predicate_compilation": "not_performed"},
            "database_conditions": db, "non_database_conditions": non_db,
            "upstream_interpretation": "used_as_supplied_not_rejudged",
        },
        "user_input_requirements": {
            "request_description": user["trigger"]["trigger"],
            "trigger_record": deepcopy(user["trigger"]),
            "trigger_ref": _field_ref(user_ref, "trigger"),
            "conversation_preconditions": requirements("conversation_preconditions"),
            "when_qualifiers": requirements("when_qualifiers"),
            "full_when_context": when,
            "realization_status": "not_bound",
            "user_assertion_automatically_proves_business_fact": False,
        },
        "trigger_requirements": {
            "spec_when": when, "source_ref": _field_ref(user_ref, "gwt/when"),
            "supplied_user_request": user["trigger"]["trigger"],
            "event_kind_gate_required": False, "text_equivalence_gate_required": False,
            "user_request_and_agent_receipt": "same_test_interaction",
            "target_action_replaced_by_request": False,
            "driver_responsibility": "realize_supplied_request_and_check_reachability_of_spec_when",
            "runtime_reachability": "not_assessed",
        },
        "fixture_requirements": {
            "root": deepcopy(user["root"]),
            "relation_assertions": {side: {"present": "relation" in row,
                                            **({"value": deepcopy(row["relation"])} if "relation" in row else {})}
                                    for side, row in (("user_requirements", user), ("condition_matches", lookup))},
            "lookup": {k: deepcopy(lookup[k]) for k in ("lookup_status", "n_matches", "matches")},
            "lookup_ref": lookup_ref,
            "candidate_semantics": "upstream_candidates_not_verified_fixtures",
            "selected_candidate": None,
            "runtime_relation_verification": "not_performed",
            "provenance": {side: {k: deepcopy(intake["sources"][side].get(k))
                                   for k in ("sha256", "database_snapshot", "reference_time", "generation_time", "provenance_status")}
                           for side in ("user_requirements", "condition_matches")},
        },
        "downstream_requirements": {
            "fixture_binding": {"required": True, "instruction": "bind or construct a concrete scenario; check root/relationship applicability and supplied conditions"},
            "user_input_binding": {"required": True, "instruction": "realize the supplied request, conversation prerequisites and full When without replacing business facts by assertions"},
            "precondition_verification": {"required": True, "instruction": "verify the original Given and When requirements against bound data, input and trace; do not infer AND from a list"},
            "trigger_reachability": {"required": True, "instruction": "observe whether the spec When is reached; a request alone does not prove an Agent tool action"},
        },
        "upstream_diagnostics": {
            "user_errors": deepcopy(user.get("errors", [])),
            "lookup_errors": deepcopy(lookup.get("errors", [])),
            "lookup_warnings": deepcopy(lookup.get("warnings", [])),
            "handling": "source_step_admission_and_handoff_preserved_not_reopened",
        },
        "excluded_from_test_input": ["violation_supplies"],
        "oracle_status": "not_compiled_here",
    })
    return _seal(result, "assembly_fingerprint")


def assemble_given_when(intake, scope):
    verify_fingerprint(intake, "intake_fingerprint")
    verify_fingerprint(scope, "scope_fingerprint")
    if scope["source_intake_fingerprint"] != intake["intake_fingerprint"]:
        raise ValueError("scope/intake mismatch")
    sources = _index(intake["branches"])
    selected = _index(scope["admitted"] + scope["deferred"])
    if sources.keys() != selected.keys():
        raise ValueError("scope partition mismatch")
    for group, status in ((scope["admitted"], "ready_for_condition_review"),
                          (scope["deferred"], "needs_input_review")):
        for row in group:
            handoff = row["handoff"]
            if handoff["handoff_status"] != status or bool(handoff["remaining_review_reasons"]) != (status == "needs_input_review"):
                raise ValueError("scope admission mismatch")
            if handoff["branch_id"] != row["branch_id"] or row["source_refs"] != sources[row["branch_id"]]["source_refs"]:
                raise ValueError("scope source reference mismatch")
            if handoff["source_branch_ref"] != {"source_intake_fingerprint": intake["intake_fingerprint"], "branch_id": row["branch_id"]}:
                raise ValueError("handoff source identity mismatch")
    branches = [_assemble_branch(intake, sources[s["branch_id"]], s) for s in scope["admitted"]]
    ready = [b for b in branches if b["assembly_status"] == "ready_for_binding"]
    counts = lambda field: sum(len(b["given_requirements"][field]) for b in ready)
    result = {
        "schema_version": SCHEMA,
        "scope": "step 2: upstream requirement assembly, not runtime validation or formal predicate compilation",
        "source_intake_fingerprint": intake["intake_fingerprint"], "source_scope_fingerprint": scope["scope_fingerprint"],
        "branches": branches, "deferred_branch_ids": [s["branch_id"] for s in scope["deferred"]],
        "policy": {"upstream_requirements_are_inputs": True, "universal_semantic_review_gate": False,
                   "old_review_answers_required": False, "automatic_downstream_execution": False},
        "summary": {"branch_count": len(branches), "deferred_branch_count": len(scope["deferred"]),
                    "assembly_status_counts": dict(Counter(b["assembly_status"] for b in branches)),
                    "database_condition_count": counts("database_conditions"),
                    "non_database_condition_count": counts("non_database_conditions"),
                    "conversation_requirement_count": sum(len(b["user_input_requirements"]["conversation_preconditions"]) for b in ready),
                    "when_qualifier_count": sum(len(b["user_input_requirements"]["when_qualifiers"]) for b in ready),
                    "lookup_status_counts": dict(Counter(b["fixture_requirements"]["lookup"]["lookup_status"] for b in ready)),
                    "database_optional_encoding_counts": dict(Counter(c["optional_encoding"]["status"] for b in ready for c in b["given_requirements"]["database_conditions"])),
                    "semantic_review_calls_required": 0, "external_llm_calls": 0, "test_ready_branch_count": 0},
    }
    return _seal(result, "assembly_set_fingerprint")


def render_assembly(result):
    lines = ["# Step 2: Upstream requirement assembly results", "", "ready_for_binding means the inputs have been assembled and can proceed to subsequent binding; it does not mean the data has been verified, the logic compiled, or the tests made executable.", "",
             "```json", json.dumps(result["summary"], ensure_ascii=False, indent=2), "```", "",
             "| branch | assembly status | DB / non-DB / dialogue / When qualifiers | candidate status |", "| --- | --- | --- | --- |"]
    for b in result["branches"]:
        if b["assembly_status"] == "blocked_input":
            lines.append(f"| {b['branch_id']} | blocked_input | see input_validation.issues | — |")
            continue
        g, u = b["given_requirements"], b["user_input_requirements"]
        counts = [len(g["database_conditions"]), len(g["non_database_conditions"]), len(u["conversation_preconditions"]), len(u["when_qualifiers"])]
        lines.append(f"| {b['branch_id']} | ready_for_binding | {' / '.join(map(str, counts))} | {b['fixture_requirements']['lookup']['lookup_status']} |")
    return "\n".join(lines) + "\n"
