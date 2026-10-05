"""Prepare Then source inputs for Step 3, offline.

No observations, expectations, prompts, or executable tests are generated.

Used to compare against a borrowed older method release's own Given/When/Then
text and reference candidate/parameter bindings (an audit trail this method's
own steps never needed -- Step1/Step2 never touch a tool catalog or produce
bindings). Removed entirely: it never drove any behavior downstream of Step3a
candidate generation, and would not exist at all for a new agent/domain. See
docs/oracle_requirement_pipeline_v0_7.md sections 15-16.
"""

from collections import Counter
from copy import deepcopy
import json

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import verify_fingerprint
from .v5_given_when_assembly_v1 import SCHEMA as ASSEMBLY_SCHEMA


SCHEMA = "agentspectesting.v5-step3-intake/v0.1"
FIELDS = ("spec_id", "kind", "origin", "rule_text", "given", "when", "then")


def seal(value, field):
    value[field] = content_sha256(value)
    return value


def index(rows, key):
    result = {}
    for row in rows:
        identity = row.get(key)
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"missing or duplicate {key}")
        result[identity] = row
    return result


def prepare_step3_intake(assembly):
    verify_fingerprint(assembly, "assembly_set_fingerprint")
    if assembly.get("schema_version") != ASSEMBLY_SCHEMA:
        raise ValueError("expected frozen Step 2 assembly schema")
    current = index(assembly["branches"], "branch_id")
    deferred = assembly["deferred_branch_ids"]
    if len(set(deferred)) != len(deferred) or set(current) & set(deferred):
        raise ValueError("invalid deferred partition")

    rows = []
    blocked = []
    for bid, branch in current.items():
        verify_fingerprint(branch, "assembly_fingerprint")
        if branch["assembly_status"] == "blocked_input":
            # A real, honestly-tracked exclusion -- not a silent skip. Step2
            # itself already decided this branch cannot be assembled (see
            # v5_given_when_assembly_v1.py's own "blocked_input" status);
            # Step3 intake was never designed to receive a mix of statuses
            # because airline's own frozen assembly happened to have zero
            # blocked_input branches, but telecom/retail's real assemblies do
            # (3 and 5 respectively) -- raising here would make every other
            # ready branch in the same assembly unusable too, which is not
            # what "blocked" means for this one branch.
            blocked.append(bid)
            continue
        if branch["assembly_status"] != "ready_for_binding":
            raise ValueError(f"unexpected assembly_status: {branch['assembly_status']!r}")
        context = {"spec_id": branch["spec_id"], **deepcopy(branch["source_context"])}
        if context["gwt"]["branch_id"] != bid:
            raise ValueError("current GWT branch identity mismatch")
        for k in ("given", "when", "then"):
            if not isinstance(context["gwt"].get(k), str) or not context["gwt"][k].strip():
                raise ValueError("current GWT text missing")
        user = branch["user_input_requirements"]
        rows.append(seal({
            "branch_id": bid, "source_assembly_fingerprint": branch["assembly_fingerprint"],
            "source_refs": deepcopy(branch["source_refs"]),
            "then_input": {
                "source_context": context,
                "given_requirements": deepcopy(branch["given_requirements"]),
                "original_when": branch["trigger_requirements"]["spec_when"],
                "supplied_user_request": user["request_description"],
                "user_request_source_ref": deepcopy(user["trigger_ref"]),
                "conversation_preconditions": deepcopy(user["conversation_preconditions"]),
                "when_qualifiers": deepcopy(user["when_qualifiers"]),
                "upstream_then_kind_hint": user["trigger_record"].get("then_kind"),
                "then_kind_is_not_normative_authority": True,
                "origin_interpretation": "domain_hypothesis_not_promoted_to_system_rule" if context["origin"] == "domain_knowledge" else "source_origin_preserved_not_revalidated",
            },
            "intake_status": "prepared_for_step3_review",
            "new_oracle_requirements": [], "new_expectations": [],
            "oracle_compilation_status": "not_started", "test_readiness": "not_assessed",
        }, "step3_input_fingerprint"))
    return seal({
        "schema_version": SCHEMA, "source_assembly_set_fingerprint": assembly["assembly_set_fingerprint"],
        "branches": rows, "deferred_branch_ids": deepcopy(deferred),
        "blocked_branch_ids": sorted(blocked),
        "scope": "Step 3A input preparation, not semantic reuse adjudication or Oracle compilation",
        "summary": {
            "branch_count": len(rows), "deferred_branch_count": len(deferred),
            "blocked_branch_count": len(blocked),
            "source_origin_counts": dict(Counter(r["then_input"]["source_context"]["origin"] for r in rows)),
            "current_tool_mapping_validated": 0, "oracle_reuse_authorized": 0,
            "new_oracle_requirements": 0, "new_expectations": 0,
            "external_llm_calls": 0, "target_agent_calls": 0,
        },
    }, "step3_intake_fingerprint")


def render_step3_intake(result):
    lines = ["# Step 3A: Then Input Inventory", "",
             "This artifact is not an LLM call package and does not contain comparison information from the old-version method package (stripped; see "
             "docs/oracle_requirement_pipeline_v0_7.md section 16).", "",
             "```json", json.dumps(result["summary"], ensure_ascii=False, indent=2), "```", "",
             "| Branch | Source |", "| --- | --- |"]
    for r in result["branches"]:
        lines.append(f"| {r['branch_id']} | {r['then_input']['source_context']['origin']} |")
    return "\n".join(lines) + "\n"
