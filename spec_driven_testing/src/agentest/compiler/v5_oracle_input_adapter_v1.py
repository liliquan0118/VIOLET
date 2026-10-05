"""Thin v5 source adapter for the existing v0.7/v0.8 Oracle workflow.

This prepares observation candidates, not executable test inputs or verdicts.

`target_action`/`bindings`/`relevant_tools` are never threaded in from a
borrowed, older, unrelated method release (agentspectesting-method-v1.1.0's
v3-era specs) -- that data was never a product of this method's own earlier
steps, so it would not exist for a new agent/domain, and Step3 intake no
longer even carries it (see v5_step3_intake_v1.py's own docstring).
`build_oracle_requirement_packets` treats these fields as optional overrides
(`None` -> derive fresh from the current branch's own text + the real tool
catalog); every branch uses that in-method derivation uniformly, airline
included. See docs/oracle_requirement_pipeline_v0_7.md sections 15-16 for the
real numbers behind this decision and why the two in-method mechanisms it
triggers are adequate replacements.
"""

from copy import deepcopy

from .artifacts import content_sha256
from .oracle_requirement_pipeline_v7 import build_oracle_requirement_packets
from .oracle_decision_pipeline_v8 import build_oracle_decision_packets
from .v5_given_when_preparation_v1 import verify_fingerprint
from .v5_step3_intake_v1 import SCHEMA, index


def prepare_oracle_inputs(intake, assessments, semantic_units, tool_catalog, profile,
                          *, selected_branch_ids=None):
    verify_fingerprint(intake, "step3_intake_fingerprint")
    if intake.get("schema_version") != SCHEMA:
        raise ValueError("expected Step 3 intake")
    rows = index(intake["branches"], "branch_id")
    selected = list(rows) if selected_branch_ids is None else list(selected_branch_ids)
    if not selected or len(set(selected)) != len(selected) or set(selected) - rows.keys():
        raise ValueError("select nonempty unique admitted branch IDs")
    specs, contexts = [], {}
    for bid in selected:
        row = rows[bid]
        verify_fingerprint(row, "step3_input_fingerprint")
        current = row["then_input"]
        context = current["source_context"]
        if context["gwt"]["branch_id"] != bid or current["original_when"] != context["gwt"]["when"]:
            raise ValueError("current GWT identity/When mismatch")
        # One record per branch supports different current rule texts for siblings.
        # target_action/bindings/relevant_tools intentionally not supplied --
        # see module docstring.
        specs.append({
            **{k: deepcopy(context[k]) for k in ("spec_id", "kind", "origin", "rule_text")},
            "gwt": [deepcopy(context["gwt"])],
        })
        request = current["supplied_user_request"]
        if not isinstance(request, str):
            raise ValueError("supplied user request must be text")
        contexts[bid] = {
            "user_request": {"text": request, "source_ref": deepcopy(current["user_request_source_ref"])},
            "v5_source": {
                "step3_input_fingerprint": row["step3_input_fingerprint"],
                "source_assembly_fingerprint": row["source_assembly_fingerprint"],
                "source_refs": deepcopy(row["source_refs"]),
            },
        }
    spec_document = {"specs": specs}
    candidates = build_oracle_requirement_packets(
        spec_document, assessments, semantic_units, tool_catalog, profile,
        selected_branch_ids=selected,
    )
    # The original generator/selector remains unchanged. Carry source context
    # through its existing branch_context extension point, outside model_input.
    for packet in candidates["packets"]:
        packet["task_input"]["branch_context"].update(deepcopy(contexts[packet["branch_id"]]))
        packet.pop("packet_fingerprint")
        packet["packet_fingerprint"] = content_sha256(packet)
    candidates.pop("packet_set_fingerprint")
    candidates["packet_set_fingerprint"] = content_sha256(candidates)
    decisions = build_oracle_decision_packets(candidates)
    report = {
        "source_intake_fingerprint": intake["step3_intake_fingerprint"],
        "source_support_fingerprints": {k: content_sha256(v) for k, v in {
            "assessments": assessments, "semantic_units": semantic_units,
            "tool_catalog": tool_catalog, "profile": profile,
        }.items()},
        "candidate_packet_set_fingerprint": candidates["packet_set_fingerprint"],
        "decision_packet_set_fingerprint": decisions["packet_set_fingerprint"],
        "summary": {
            "branch_count": len(selected),
            "candidate_count": len(candidates["packets"]),
            "unique_membership_questions": len(decisions["packets"]),
            "fallback_branch_count": sum(
                "gwt_then_fallback" in b["candidate_origins"] for b in candidates["branch_summaries"]),
            "external_calls": 0, "reused_old_answers": 0,
        },
        "status": "candidates_prepared_membership_not_run",
        "limits": [
            "Candidate generation is not a completeness or acceptance check.",
            "Given constraints remain in frozen Step 2; database objects are not bound here.",
            "No runtime extraction, Driver, test execution or evaluation is performed.",
        ],
    }
    return {"specs": spec_document, "candidates": candidates, "decisions": decisions, "report": report}
