"""Step 3 review handoff: source rules, task policy and historical evidence stay separate.

This is a review contract, not the accepted-requirements schema or an evaluator.
No natural-language interpretation, legacy acceptance promotion or runtime execution.
"""

from collections import Counter
from copy import deepcopy
import json

from .v5_given_when_preparation_v1 import verify_fingerprint
from .v5_step3_intake_v1 import index, seal


SCHEMA = "agentspectesting.v5-oracle-review-contract-set/v0.1"
EVIDENCE_SCHEMA = "agentspectesting.v5-oracle-scope-evidence/v0.1"


def execution_assumptions(chain):
    """Recognize explicit historical config fields, not whether the spec entails them."""
    expectation = chain["expectation"]
    runtime = chain["runtime_binding"]["runtime_binding"]
    program = chain["evaluator"]["program"] or {}
    facts = []

    def add(code, path, value):
        facts.append({"code": code, "historical_json_pointer": path, "value": deepcopy(value),
                      "source_support": "not_established", "active_in_v5": False})

    if expectation["normative_mode"] != expectation["effective_mode"]:
        add("historical_modality_conversion", "/expectation",
            {"normative_mode": expectation["normative_mode"], "effective_mode": expectation["effective_mode"],
             "permission_rule": expectation["derivation"].get("permission_rule"),
             "policy_id": expectation["derivation"].get("expectation_policy_id")})
    if expectation["expected_observation"].get("requires_observation") is True:
        add("historical_observation_required", "/expectation/expected_observation", expectation["expected_observation"])
    matcher = runtime.get("left_event", {}).get("matcher", {})
    if matcher.get("kind") == "contains_right_call_argument_values":
        add("historical_argument_disclosure_matcher", "/runtime_binding/runtime_binding/left_event/matcher", matcher)
    left = runtime.get("left_event", {})
    if left.get("binding_kind") == "driver_action_event":
        add("historical_driver_label_evidence", "/runtime_binding/runtime_binding/left_event", left)
    if runtime.get("event_filter", {}).get("event_kind") == "assistant_tool_call" and program.get("evaluator_kind") == "match_cardinality":
        add("historical_call_count_only", "/evaluator/program", program)
    return facts


def build_contracts(evidence):
    verify_fingerprint(evidence, "evidence_fingerprint")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("expected scope evidence, not a model checklist")
    index(evidence["branches"], "branch_id")
    rows = []
    for source in evidence["branches"]:
        context = source["current_source"]
        candidates = index(source["legacy_candidates"], "candidate_id")
        chains = index(source["legacy_chains"], "requirement_id")
        observations = []
        for cid, candidate in candidates.items():
            linked = [r for r in chains.values() if cid in r["source_candidate_ids"]]
            observations.append({"candidate_id": cid,
                "proposal": deepcopy(candidate["candidate"]),
                "source_grounding": "not_adjudicated_for_v5",
                "scope_review": "pending",
                "legacy_reference": {"selection": deepcopy(candidate["historical_selection"]),
                                     "chain_ids": [r["requirement_id"] for r in linked]},
                "accepted_for_v5": False})
        hints = [{"requirement_id": r["requirement_id"],
                  "evidence_pointer": f"/branches/{len(rows)}/legacy_chains/{i}",
                  "assumptions": execution_assumptions(r)} for i, r in enumerate(chains.values())]
        row = {
            "branch_id": source["branch_id"],
            "source_step3_input_fingerprint": source["source_step3_input_fingerprint"],
            "spec_requirement": {
                "source_context": deepcopy(context["source_context"]),
                "interpretation_status": "source_preserved_not_compiled",
                "active_assertions": [],
                "legacy_mode_is_not_current_rule": True,
            },
            "applicability": {
                "given_requirements": deepcopy(context["given_requirements"]),
                "when": context["original_when"],
                "conversation_preconditions": deepcopy(context["conversation_preconditions"]),
                "when_qualifiers": deepcopy(context["when_qualifiers"]),
                "reachability_contract": None,
                "status": "not_compiled",
                "tool_call_presence_implies_reachability": False,
            },
            "task_completion": {
                "source_user_request": context["supplied_user_request"],
                "user_request_source_ref": deepcopy(context["user_request_source_ref"]),
                "policy_status": "not_selected_for_v5",
                "active_expectations": [],
                "legacy_policy_reference": deepcopy(evidence["legacy_expectation_policy"]),
                "changes_spec_modality": False,
            },
            "observation_review": {
                "candidates": observations,
                "coverage_status": "not_adjudicated_for_v5",
                "historical_execution_assumptions": hints,
                "unknown_matchers_are_supported": False,
            },
            "handoff": {
                "allowed_use": "step4_review_only",
                "runtime_lowering_allowed": False,
                "executable_programs": [],
                "blocking_requirements": ["spec_assertions_and_scope_not_compiled",
                                          "applicability_not_compiled",
                                          "candidate_coverage_not_adjudicated",
                                          "observation_grounding_not_adjudicated"],
                "task_policy_pending_is_not_a_spec_violation": True,
            },
        }
        rows.append(seal(row, "contract_fingerprint"))
    facts = [a for r in rows for h in r["observation_review"]["historical_execution_assumptions"] for a in h["assumptions"]]
    return seal({"schema_version": SCHEMA,
                 "source_evidence_fingerprint": evidence["evidence_fingerprint"],
                 "branches": rows,
                 "summary": {"branch_count": len(rows), "candidate_count": sum(len(r["observation_review"]["candidates"]) for r in rows),
                             "historical_assumption_counts": dict(Counter(f["code"] for f in facts)),
                             "active_spec_assertions": 0, "active_task_expectations": 0,
                             "ready_for_runtime_lowering": 0, "external_llm_calls": 0, "target_agent_calls": 0}},
                "contract_set_fingerprint")


def validate_contracts(contract, evidence):
    """Check exact reconstruction too: re-signing a changed readiness flag is not acceptance."""
    verify_fingerprint(contract, "contract_set_fingerprint")
    if contract != build_contracts(evidence):
        raise ValueError("review contract differs from evidence-derived separation")
    return contract


def step4_review_input(contract, evidence, branch_id):
    validate_contracts(contract, evidence)
    rows = index(contract["branches"], "branch_id")
    if branch_id not in rows:
        raise ValueError("branch not in reviewed batch")
    return {"schema_version": "agentspectesting.v5-step4-review-input/v0.1",
            "source_contract_set_fingerprint": contract["contract_set_fingerprint"],
            "contract": deepcopy(rows[branch_id]), "runtime_lowering_allowed": False}


def render_contracts(contract):
    lines = ["# Step 3 Layered Review Contract", "", "This is only a review handoff, not an accepted or executable Oracle.", "",
             "| Branch | Candidates | Historical execution assumptions | Executable |", "| --- | --- | --- | --- |"]
    for row in contract["branches"]:
        obs = row["observation_review"]
        facts = [a["code"] for h in obs["historical_execution_assumptions"] for a in h["assumptions"]]
        lines.append(f"| {row['branch_id']} | {len(obs['candidates'])} | {', '.join(facts) or 'no recognized configuration; does not mean no assumptions'} | No |")
    lines += ["", "```json", json.dumps(contract["summary"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)
