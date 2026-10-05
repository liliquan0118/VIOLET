"""Lossless v5 source intake. No condition execution, Oracle or test generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .artifacts import content_sha256


VERSION = "agentspectesting.v5-source-intake/v0.1"
COMMON = ("spec_id", "kind", "origin", "rule_text", "gwt_index", "gwt",
          "root", "DBconditions", "nonDBconditions")
USER_FIELDS = ("trigger", "conversation_preconditions", "when_qualifiers",
               "violation_supplies", "untestable_in_domain", "untestable_reasons", "errors")
LOOKUP_FIELDS = ("lookup_status", "n_matches", "matches", "warnings", "errors")
# "patchable" (no real, already-existing object satisfies the branch's Given
# state, but one could be synthesized via a real database patch/overlay --
# see the real "patch" field these rows carry, conceptually the same idea as
# Step7's own required_database_overlay mechanism) never appeared in the
# airline domain's generation, so it was missing here -- confirmed via real
# telecom (17 rows) and retail (1 row) intake data, all of which were the
# ONLY source of each domain's invalid_row_structure count. Step2
# (v5_given_when_assembly_v1.py) only checks lookup_status is a non-empty
# string, no allow-list, so this was the sole place a new, real, legitimate
# status value needed registering.
STATUSES = {"matched", "anchored", "makeable", "no_conditions", "unmakeable", "patchable"}


def _present(row: dict, field: str) -> dict:
    return {"present": field in row, **({"value": deepcopy(row[field])} if field in row else {})}


def _read(path: Path) -> dict:
    raw = path.read_bytes()

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key: {key!r} in {path}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError(f"non-JSON constant {value!r} in {path}")

    return {
        "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "document": json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                               parse_constant=reject_constant),
        "generation_time": None, "database_snapshot": None, "reference_time": None,
        "provenance_status": "file_identity_only_upstream_metadata_not_supplied",
    }


def _row_errors(row: Any, role: str) -> list[dict]:
    errors = []

    def check(obj, key, expected, path, nullable=False):
        if not isinstance(obj, dict) or key not in obj:
            errors.append({"path": f"{path}.{key}", "code": "missing_field"})
            return None
        value = obj[key]
        valid = value is None and nullable
        valid = valid or (type(value) is expected)
        if expected is str and value == "":
            valid = False
        if not valid:
            errors.append({"path": f"{path}.{key}", "code": "invalid_type_or_empty",
                           "expected": expected.__name__, "actual": type(value).__name__})
        return value

    if not isinstance(row, dict):
        return [{"path": "$", "code": "row_not_object"}]
    for field in ("spec_id", "kind", "origin", "rule_text"):
        check(row, field, str, "$")
    index = check(row, "gwt_index", int, "$")
    if type(index) is int and index < 0:
        errors.append({"path": "$.gwt_index", "code": "negative_index"})
    check(row, "root", str, "$", nullable=True)
    gwt = check(row, "gwt", dict, "$")
    if isinstance(gwt, dict):
        for field in ("given", "when", "then", "branch_id"):
            check(gwt, field, str, "$.gwt")
    for field in ("DBconditions", "nonDBconditions", "errors"):
        check(row, field, list, "$")
    for field in ("errors", "warnings"):
        if isinstance(row.get(field), list):
            for i, value in enumerate(row[field]):
                if not isinstance(value, str):
                    errors.append({"path": f"$.{field}[{i}]", "code": "diagnostic_not_string"})
    for field in ("DBconditions", "nonDBconditions"):
        for i, item in enumerate(row.get(field, []) if isinstance(row.get(field), list) else []):
            if not isinstance(item, dict):
                errors.append({"path": f"$.{field}[{i}]", "code": "item_not_object"})
                continue
            if field == "DBconditions":
                for name in ("table", "path", "op"):
                    check(item, name, str, f"$.{field}[{i}]")
                if "value" not in item:
                    errors.append({"path": f"$.{field}[{i}].value", "code": "missing_field"})
            else:
                check(item, "clause", str, f"$.{field}[{i}]")
    if "relation" in row:
        relation = row["relation"]
        if relation not in (None, "owner", "not_owner"):
            errors.append({"path": "$.relation", "code": "unknown_relation"})
    if role == "user_requirements":
        trigger = check(row, "trigger", dict, "$")
        if isinstance(trigger, dict):
            check(trigger, "trigger", str, "$.trigger")
            if "then_kind" in trigger:
                check(trigger, "then_kind", str, "$.trigger")
        check(row, "untestable_in_domain", bool, "$")
        for field in ("conversation_preconditions", "when_qualifiers", "violation_supplies", "untestable_reasons"):
            values = check(row, field, list, "$")
            if isinstance(values, list):
                for i, value in enumerate(values):
                    if not isinstance(value, dict):
                        errors.append({"path": f"$.{field}[{i}]", "code": "item_not_object"})
                    else:
                        names = {"conversation_preconditions": ("condition", "source_clause"),
                                 "when_qualifiers": ("condition", "source_quote"),
                                 "violation_supplies": ("requirement", "breaks")}.get(field, ())
                        for name in names:
                            check(value, name, str, f"$.{field}[{i}]")
    else:
        status = check(row, "lookup_status", str, "$")
        if isinstance(status, str) and status not in STATUSES:
            errors.append({"path": "$.lookup_status", "code": "unknown_lookup_status"})
        check(row, "warnings", list, "$")
        matches = check(row, "matches", list, "$")
        count = check(row, "n_matches", int, "$")
        if isinstance(matches, list):
            if type(count) is int and count != len(matches):
                errors.append({"path": "$.n_matches", "code": "match_count_mismatch"})
            for i, match in enumerate(matches):
                if not isinstance(match, dict):
                    errors.append({"path": f"$.matches[{i}]", "code": "item_not_object"})
                else:
                    check(match, "user_id", str, f"$.matches[{i}]")
                    check(match, "root_id", str, f"$.matches[{i}]", nullable=True)
    return errors


def _index(rows: list, role: str) -> tuple[dict, list]:
    index = defaultdict(list)
    diagnostics = []
    for i, row in enumerate(rows):
        issues = _row_errors(row, role)
        branch = (row.get("gwt") or {}).get("branch_id") if isinstance(row, dict) and isinstance(row.get("gwt"), dict) else None
        if isinstance(branch, str) and branch:
            index[branch].append(i)
        else:
            branch = None
        if issues:
            diagnostics.append({"source": role, "row_index": i, "branch_id": branch, "issues": issues})
    return dict(index), diagnostics


def _old_index(document: dict) -> dict:
    if not isinstance(document, dict) or not isinstance(document.get("specs"), list):
        raise ValueError("v3 source must contain a specs array")
    result = {}
    for spec in document["specs"]:
        if not isinstance(spec, dict) or not isinstance(spec.get("gwt"), list):
            raise ValueError("invalid v3 spec/GWT structure")
        for gwt in spec["gwt"]:
            if not isinstance(gwt, dict) or not isinstance(gwt.get("branch_id"), str):
                raise ValueError("v3 GWT branch ID missing")
            branch = gwt["branch_id"]
            if not branch or branch in result:
                raise ValueError("v3 branch IDs must be unique and nonempty")
            result[branch] = {**{k: deepcopy(spec.get(k)) for k in ("spec_id", "kind", "origin", "rule_text")},
                              "gwt": {k: deepcopy(gwt.get(k)) for k in ("given", "when", "then")}}
    return result


def _version_comparison(old: dict, branches: list[dict]) -> dict:
    new = {b["branch_id"]: b for b in branches}
    comparisons = []
    for key in sorted(old.keys() & new.keys()):
        source = new[key]["source_fields"]
        if not source or any(x["code"] == "invalid_row_structure" for x in new[key]["review_reasons"]):
            comparisons.append({"branch_id": key, "status": "uncomparable_input_issue", "differences": []})
            continue
        normalized = {k: source.get(k) for k in ("spec_id", "kind", "origin", "rule_text")}
        normalized["gwt"] = {k: source["gwt"].get(k) for k in ("given", "when", "then")}
        differences = []
        for field in ("spec_id", "kind", "origin", "rule_text", "gwt.given", "gwt.when", "gwt.then"):
            def get(value):
                return value["gwt"][field[4:]] if field.startswith("gwt.") else value[field]
            if get(old[key]) != get(normalized):
                differences.append({"field": field, "v3": get(old[key]), "v5": get(normalized)})
        comparisons.append({"branch_id": key, "status": "text_changed" if differences else "source_text_equal",
                            "differences": differences})
    return {
        "comparison_scope": "exact IDs and spec_id/kind/origin/rule_text/GWT text only; no semantic mapping or Oracle reuse approval",
        "v3_branch_count": len(old), "v5_branch_count": len(new),
        "shared_id_count": len(comparisons), "v3_only": sorted(old.keys() - new.keys()),
        "v5_only": sorted(new.keys() - old.keys()), "shared_id_comparisons": comparisons,
        "comparison_status_counts": dict(sorted(Counter(x["status"] for x in comparisons).items())),
    }


def build_intake(user_source: dict, match_source: dict, v3_source: dict | None = None) -> dict:
    """Preserve originals; classify only intake quality, never test readiness.

    v3_source is an optional prior-generation spec file, used only to render
    a "did this drift from a known-good earlier run" diff report
    (version_comparison) -- it never affects intake_status, summary, or any
    branch's review_reasons, all of which are computed from user_source/
    match_source alone. A brand-new domain (e.g. telecom, retail) has no
    real prior generation to compare against; omitting v3_source (rather
    than fabricating a fake placeholder "prior version") is the honest way
    to represent that, and version_comparison is reported as not_applicable
    instead of a vacuous all-zero comparison against nothing real."""
    sources = {"user_requirements": deepcopy(user_source), "condition_matches": deepcopy(match_source),
               "v3_reference": deepcopy(v3_source) if v3_source is not None else None}
    for role in ("user_requirements", "condition_matches"):
        if not isinstance(sources[role]["document"], list):
            raise ValueError(f"{role} must be an array")
    users, matches = sources["user_requirements"]["document"], sources["condition_matches"]["document"]
    ui, ue = _index(users, "user_requirements")
    mi, me = _index(matches, "condition_matches")
    diagnostics = ue + me
    malformed = {x["branch_id"] for x in diagnostics}
    branches = []
    for key in sorted(ui.keys() | mi.keys()):
        ur, mr = ui.get(key, []), mi.get(key, [])
        issues = []
        for role, refs in (("user_requirements", ur), ("condition_matches", mr)):
            if len(refs) != 1:
                issues.append({"code": "missing_peer" if not refs else "duplicate_branch_id", "source": role, "row_indices": refs})
        left, right = (users[ur[0]] if len(ur) == 1 else {}), (matches[mr[0]] if len(mr) == 1 else {})
        field_differences = []
        if len(ur) == len(mr) == 1:
            for field in (*COMMON, "relation"):
                a, b = _present(left, field), _present(right, field)
                # Python considers True == 1 and False == 0. Source JSON types
                # must remain distinct when comparing upstream conditions.
                if content_sha256(a) != content_sha256(b):
                    field_differences.append({"field": field, "kind": "missing_on_one_side" if a["present"] != b["present"] else "value_conflict",
                                              "user_requirements": a, "condition_matches": b})
        issues.extend({"code": x["kind"], "field": x["field"]} for x in field_differences)
        if key in malformed:
            issues.append({"code": "invalid_row_structure"})
        for role, row in (("user_requirements", left), ("condition_matches", right)):
            for field in ("errors", "warnings"):
                if row.get(field):
                    issues.append({"code": "upstream_" + field, "source": role, "value": deepcopy(row[field])})
        if left.get("untestable_in_domain") is True:
            issues.append({"code": "upstream_untestable_claim", "value": deepcopy(left.get("untestable_reasons"))})
        branches.append({
            "branch_id": key, "source_refs": {"user_requirements": ur, "condition_matches": mr},
            "source_fields": {k: deepcopy(left[k]) for k in COMMON if k in left},
            "user_requirements": {k: deepcopy(left[k]) for k in USER_FIELDS if k in left},
            "fixture_lookup": {k: deepcopy(right[k]) for k in LOOKUP_FIELDS if k in right},
            "relation_evidence": {"user_requirements": _present(left, "relation"), "condition_matches": _present(right, "relation")},
            "field_differences": field_differences, "review_reasons": issues,
            "intake_status": "needs_input_review" if issues else "ready_for_condition_review",
            "semantic_validation": "not_performed", "test_readiness": "not_assessed",
        })
    summary = {
        "user_record_count": len(users), "match_record_count": len(matches), "branch_count": len(branches),
        "paired_unique_count": sum(len(b["source_refs"]["user_requirements"]) == len(b["source_refs"]["condition_matches"]) == 1 for b in branches),
        "malformed_row_count": len(diagnostics),
        "unidentified_row_count": sum(x["branch_id"] is None for x in diagnostics),
        "intake_status_counts": dict(sorted(Counter(b["intake_status"] for b in branches).items())),
        "review_reason_counts": dict(sorted(Counter(i["code"] for b in branches for i in b["review_reasons"]).items())),
        "llm_calls": 0,
    }
    version_comparison = (_version_comparison(_old_index(v3_source["document"]), branches) if v3_source is not None
        # Real downstream consumer (v5_condition_review_scope_v1.py::build_scope)
        # iterates shared_id_comparisons unconditionally -- None crashes it.
        # Zero/empty here is not a fabricated claim of "zero drift"; it
        # honestly reflects that there is no v3 at all to compare against
        # (a brand-new domain), not that a real comparison found nothing.
        else {"comparison_scope": "not_applicable_no_v3_reference_supplied", "v3_branch_count": 0,
              "v5_branch_count": len(branches), "shared_id_count": 0, "v3_only": [],
              "v5_only": [], "shared_id_comparisons": [], "comparison_status_counts": {}})
    result = {"schema_version": VERSION, "scope": "source intake only",
              "sources": sources, "branches": branches, "row_diagnostics": diagnostics,
              "summary": summary, "version_comparison": version_comparison}
    result["intake_fingerprint"] = content_sha256(result)
    return result


def intake_files(user_path: Path, match_path: Path, v3_path: Path | None = None) -> dict:
    return build_intake(_read(user_path), _read(match_path), _read(v3_path) if v3_path is not None else None)


def audit_view(intake: dict) -> dict:
    return {"schema_version": "agentspectesting.v5-source-audit/v0.1",
            "source_intake_fingerprint": intake["intake_fingerprint"],
            "sources": {k: ({f: v for f, v in source.items() if f != "document"} if source is not None else None)
                        for k, source in intake["sources"].items()},
            "summary": deepcopy(intake["summary"]), "row_diagnostics": deepcopy(intake["row_diagnostics"]),
            "branches_needing_review": [deepcopy(b) for b in intake["branches"] if b["review_reasons"]],
            "version_comparison": deepcopy(intake["version_comparison"])}


def render_review(intake: dict) -> str:
    summary, comparison = intake["summary"], intake["version_comparison"]
    lines = ["# Part 1: v5 source intake acceptance results", "",
             "This report only checks sources, structure, and correspondences. ready_for_condition_review means the conditions can proceed to review; it does not mean the semantics are correct, the fixtures are usable, or the tests are executable.", "",
             "## 1. Full results", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
             "## 2. Branches needing review", "", "Reasons may overlap. Original errors, warnings, untestability declarations, and missing fields are all retained, with no automatic adjudication.", "",
             "| branch | reason |", "| --- | --- |"]
    for branch in intake["branches"]:
        if branch["review_reasons"]:
            reasons = sorted({x["code"] + (":" + x["field"] if "field" in x else "") for x in branch["review_reasons"]})
            safe_id = branch["branch_id"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {safe_id} | {', '.join(reasons)} |")
    if comparison.get("comparison_scope") == "not_applicable_no_v3_reference_supplied":
        lines += ["", "## 3. Old/new version correspondence", "",
                  f"No v3 reference file was supplied, so the old/new version comparison is skipped (v5 has {comparison['v5_branch_count']} entries in total, all treated as new; this does not indicate a quality problem).", ""]
    else:
        lines += ["", "## 3. Old/new version correspondence", "",
                  f"v3: {comparison['v3_branch_count']} entries, v5: {comparison['v5_branch_count']} entries, shared IDs: {comparison['shared_id_count']}; v3 only: {len(comparison['v3_only'])}, v5 only: {len(comparison['v5_only'])}.", "",
                  "```json", json.dumps(comparison["comparison_status_counts"], ensure_ascii=False, indent=2), "```", "",
                  "source_text_equal only means the fields listed in this report are identical verbatim; not all upstream material was compared, and it does not authorize reusing the Oracle. Differing IDs were not similarity-matched or merged, and no reasons for deletion were inferred.", ""]
    lines += ["## 4. Representative branches", ""]
    for key in ("airline_093_state#b0", "airline_093_state#e1", "airline_094_norm#b0", "airline_030_arg#b0"):
        branch = next((b for b in intake["branches"] if b["branch_id"] == key), None)
        if branch:
            example = {k: deepcopy(branch[k]) for k in ("branch_id", "source_fields", "relation_evidence", "intake_status", "review_reasons")}
            example["user_requirements"] = {k: v for k, v in branch["user_requirements"].items() if k != "violation_supplies"}
            lookup = branch["fixture_lookup"]
            example["fixture_lookup"] = {k: v for k, v in lookup.items() if k != "matches"}
            example["candidate_preview"] = lookup.get("matches", [])[:1]
            lines += [f"### {key}", "", "```json", json.dumps(example, ensure_ascii=False, indent=2), "```", ""]
    lines += ["## 5. Scope of this round", "",
              "The original documents are stored in full in sources.*.document of intake.json; branches refer back to them via array indices in source_refs. Unknown fields are not deleted, missing fields are not filled with null, and there is no adjudication of may/obligation or of whether candidate inputs are valid.", "",
              "audit.json stores the per-entry differences and the full version comparison. No condition evaluation, data construction, Oracle, test case generation, or external model calls were run. Proceed to step 2 only after this part has been reviewed.", ""]
    return "\n".join(lines)
