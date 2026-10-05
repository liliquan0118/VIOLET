"""Prepare narrowly scoped text-review packets; no model or Agent invocation."""

from collections import Counter
from copy import deepcopy

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import verify_fingerprint


DECISIONS = {
    "given_structure": {"single_condition", "multiple_conditions", "unclear"},
    "when_event_equivalence": {"same_event", "different_event", "unclear"},
}


def _stratum(branch, task):
    if task == "when_event_equivalence":
        return "when:constant_given" if branch["given"]["logic_status"] == "constant_resolved" else "when:nonconstant_given"
    db = len(branch["given"]["database_conditions"])
    non_db = len(branch["given"]["non_database_conditions"])
    if db and non_db:
        return "given:mixed"
    if db:
        return "given:db_single" if db == 1 else "given:db_multiple"
    if non_db:
        return "given:nonDB_single" if non_db == 1 else "given:nonDB_multiple"
    return "given:unmapped"


def _model_view(branch, task):
    if task == "given_structure":
        return {"given": branch["given"]["source_text"]}
    return {"given_context": branch["given"]["source_text"],
            "original_when": branch["when"]["source_text"],
            "suggested_user_request": branch["when"]["suggested_user_trigger_text"]}


def _messages(view, task, template):
    if task == "given_structure":
        text = "Given (text to review):\n" + view["given"]
    else:
        text = ("Given context (text to review):\n" + view["given_context"]
                + "\n\nOriginal When:\n" + view["original_when"]
                + "\n\nSuggested user request:\n" + view["suggested_user_request"])
    return [{"role": "system", "content": template.strip()}, {"role": "user", "content": text}]


def build_review_packets(preparation, templates):
    verify_fingerprint(preparation, "preparation_set_fingerprint")
    if preparation.get("schema_version") != "agentspectesting.v5-given-when-preparation/v0.1":
        raise ValueError("unsupported preparation schema")
    if set(templates) != set(DECISIONS) or any(not isinstance(v, str) or not v.strip() for v in templates.values()):
        raise ValueError("both nonempty task templates are required")
    branches = preparation["branches"]
    ids = [b["branch_id"] for b in branches]
    if len(ids) != len(set(ids)) or set(ids) & set(preparation["deferred_branch_ids"]):
        raise ValueError("invalid preparation branch partition")
    groups, mechanical, exclusions = {}, [], []
    for branch in sorted(branches, key=lambda b: b["branch_id"]):
        verify_fingerprint(branch, "preparation_fingerprint")
        if branch["structural_issues"] or branch["preparation_status"] != "prepared_for_semantic_review":
            exclusions.append({"branch_id": branch["branch_id"], "reason": "structure_review_required"})
            continue
        tasks = []
        if branch["given"]["logic_status"] == "constant_resolved":
            if branch["given"]["source_text"].strip().casefold() != "true" or any(branch["given"][key] for key in ("database_conditions", "non_database_conditions", "conversation_requirements")):
                raise ValueError("invalid constant Given bypass")
            mechanical.append({"branch_id": branch["branch_id"], "task": "given_structure",
                               "basis": "explicit_unconditional_true", "result": "no_split_needed"})
        else:
            tasks.append("given_structure")
        same_text = branch["when"]["source_text"] == branch["when"]["suggested_user_trigger_text"]
        if same_text != branch["when"]["trigger_text_identical"]:
            raise ValueError("When identity flag differs from text")
        if same_text:
            mechanical.append({"branch_id": branch["branch_id"], "task": "when_event_equivalence",
                               "basis": "identical_text", "result": "same_event_description_only"})
        else:
            tasks.append("when_event_equivalence")
        for task in tasks:
            view = _model_view(branch, task)
            messages = _messages(view, task, templates[task])
            group_key = content_sha256({"task": task, "messages": messages})
            packet = groups.setdefault(group_key, {
                "packet_id": f"V5-2B1::{task}::{group_key[:20]}", "task": task,
                "model_view": view, "messages": messages,
                "template_fingerprint": content_sha256(templates[task]),
                "response_decisions": sorted(DECISIONS[task]), "consumers": [],
                "result_status": "not_run", "approval_status": "not_requested",
            })
            packet["consumers"].append({
                "branch_id": branch["branch_id"], "preparation_fingerprint": branch["preparation_fingerprint"],
                "stratum": _stratum(branch, task),
                "source_refs": ([deepcopy(branch["given"]["source_ref"])] if task == "given_structure"
                                else [deepcopy(branch["given"]["source_ref"]), deepcopy(branch["when"]["source_ref"]),
                                      deepcopy(branch["when"]["suggested_trigger_ref"])]),
            })
    packets = list(groups.values())
    for packet in packets:
        packet["packet_fingerprint"] = content_sha256(packet)
    result = {
        "schema_version": "agentspectesting.v5-semantic-review-packets/v0.1",
        "stage": "2B1 source-text review only",
        "source_preparation_fingerprint": preparation["preparation_set_fingerprint"],
        "packets": packets, "mechanical_decisions": mechanical,
        "excluded_for_structure": exclusions, "deferred_branch_ids": deepcopy(preparation["deferred_branch_ids"]),
        "downstream_policy": {
            "given_structure": "Single -> clause mapping review; multiple -> separate source-preserving split task; unclear -> clarification. Never infer AND/OR here.",
            "when_event_equivalence": "Different -> retain separate driving request and observed event; same -> only event equivalence reviewed. Qualifier coverage and reachability remain pending.",
            "format_valid_is_semantically_accepted": False,
            "automatically_promote_test_readiness": False,
        },
        "remaining_2B_work": ["Given clause split where necessary", "clause-to-condition mapping and coverage",
                              "full Given logical structure", "authoritative evidence and missing-context resolution",
                              "When qualifier coverage and driving-event relationship"],
        "summary": {
            "branch_count": len(branches), "deferred_branch_count": len(preparation["deferred_branch_ids"]),
            "packet_counts": dict(sorted(Counter(p["task"] for p in packets).items())),
            "candidate_calls_full_pool": len(packets),
            "branch_task_pairs": sum(len(p["consumers"]) for p in packets),
            "exact_input_dedup_savings": sum(len(p["consumers"]) - 1 for p in packets),
            "mechanical_decision_counts": dict(sorted(Counter(p["task"] for p in mechanical).items())),
            "external_llm_calls": 0,
        },
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return result


def select_pilot(packet_set, policy):
    verify_fingerprint(packet_set, "packet_set_fingerprint")
    if (policy.get("selection_method") != "first_by_branch_id_per_observed_structure"
            or type(policy.get("max_calls")) is not int or policy["max_calls"] <= 0
            or policy.get("max_calls_per_packet") != 1 or policy.get("automatic_retries") != 0):
        raise ValueError("invalid pilot selection policy")
    strata = policy["strata"]
    if not isinstance(strata, list) or any(not isinstance(s, str) for s in strata) or len(strata) != len(set(strata)):
        raise ValueError("invalid pilot strata")
    selected, coverage = [], []
    for stratum in strata:
        eligible = [p for p in packet_set["packets"] if any(c["stratum"] == stratum for c in p["consumers"])]
        eligible.sort(key=lambda p: (min(c["branch_id"] for c in p["consumers"] if c["stratum"] == stratum), p["packet_id"]))
        chosen = next((p for p in eligible if p["packet_id"] in selected), None)
        if chosen is None and eligible and len(selected) < policy["max_calls"]:
            chosen = eligible[0]
            selected.append(chosen["packet_id"])
        coverage.append({"stratum": stratum, "status": "selected" if chosen else "absent" if not eligible else "budget_excluded",
                         "packet_id": chosen["packet_id"] if chosen else None})
    selected_packets = [deepcopy(next(p for p in packet_set["packets"] if p["packet_id"] == key)) for key in selected]
    result = {
        "schema_version": "agentspectesting.v5-semantic-pilot/v0.1",
        "source_packet_set_fingerprint": packet_set["packet_set_fingerprint"],
        "selection_policy": deepcopy(policy), "coverage": coverage, "packets": selected_packets,
        "approval_status": "not_requested", "approved_call_count": 0,
        "proposed_model": "deepseek-v4-flash", "max_calls_per_packet": 1, "automatic_retries": 0,
        "target_agent_calls": 0,
        "summary": {"proposed_call_count": len(selected_packets),
                    "task_counts": dict(sorted(Counter(p["task"] for p in selected_packets).items())),
                    "covered_branch_count": len({c["branch_id"] for p in selected_packets for c in p["consumers"]}),
                    "external_llm_calls": 0},
    }
    result["pilot_fingerprint"] = content_sha256(result)
    return result


def validate_text_review_response(packet, response):
    """Validate output shape only; never label a response semantically accepted."""
    verify_fingerprint(packet, "packet_fingerprint")
    if not isinstance(response, dict) or set(response) != {"decision", "reason"}:
        raise ValueError("response must contain exactly decision and reason")
    if not isinstance(response["decision"], str) or response["decision"] not in DECISIONS[packet["task"]]:
        raise ValueError("invalid task decision")
    if not isinstance(response["reason"], str) or not response["reason"].strip() or len(response["reason"]) > 600:
        raise ValueError("reason must be a short nonempty sentence")
    return {"packet_id": packet["packet_id"], "source_packet_fingerprint": packet["packet_fingerprint"],
            "response": deepcopy(response), "format_status": "valid", "semantic_acceptance": "pending_review",
            "test_readiness": "not_assessed"}


def render_pilot(pilot):
    lines = ["# 2B1 first-batch semantic review: pre-call preview", "", "The model has not been called yet, and approval for this batch of calls has not yet been obtained. Below is only a preview of the messages actually proposed to be sent.", "",
             f"Proposed calls: {pilot['summary']['proposed_call_count']}, at most one per question, no automatic retries; target Agent calls: 0.", "",
             "Internal branch IDs, summaries, structural groupings and earlier review conclusions are not sent to the model. Calls are merged only when the complete inputs are identical; outputs still require human review.", ""]
    for i, packet in enumerate(pilot["packets"], 1):
        lines += [f"## {i}. {packet['task']}", "", "Related branches: " + ", ".join(c["branch_id"] for c in packet["consumers"]), "",
                  "### System question", "", "```text", packet["messages"][0]["content"], "```", "",
                  "### Input text (verbatim)", "", "```text", packet["messages"][1]["content"], "```", ""]
    return "\n".join(lines)
