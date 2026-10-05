"""Split event-kind review from restriction review; preserve a reviewed-result gate."""

from copy import deepcopy

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import verify_fingerprint


KIND = "when_event_kind"
RESTRICTIONS = "when_restriction_preservation"
DECISIONS = {KIND: {"same_kind", "different_kind", "unclear"},
             RESTRICTIONS: {"preserved", "changed", "unclear"}}


def _packet(branch, task, template, dependency=None):
    view = {"original_when": branch["when"]["source_text"],
            "suggested_description": branch["when"]["suggested_user_trigger_text"]}
    messages = [{"role": "system", "content": template.strip()},
                {"role": "user", "content": "Original description:\n" + view["original_when"]
                 + "\n\nSuggested description:\n" + view["suggested_description"]}]
    identity = {"branch_id": branch["branch_id"], "task": task, "messages": messages,
                "source_preparation_fingerprint": branch["preparation_fingerprint"], "dependency": dependency}
    result = {"packet_id": f"V5-WHEN-SPLIT::{task}::{content_sha256(identity)[:20]}",
              "task": task, "branch_id": branch["branch_id"], "model_view": view, "messages": messages,
              "source_refs": [deepcopy(branch["when"]["source_ref"]), deepcopy(branch["when"]["suggested_trigger_ref"])],
              "source_preparation_fingerprint": branch["preparation_fingerprint"],
              "dependency": deepcopy(dependency), "response_decisions": sorted(DECISIONS[task]),
              "approval_status": "not_requested", "result_status": "not_run"}
    result["packet_fingerprint"] = content_sha256(result)
    return result


def build_kind_packets(preparation, template):
    verify_fingerprint(preparation, "preparation_set_fingerprint")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("event-kind template required")
    if preparation.get("schema_version") != "agentspectesting.v5-given-when-preparation/v0.1":
        raise ValueError("unexpected preparation schema")
    ids = [b["branch_id"] for b in preparation["branches"]]
    if len(ids) != len(set(ids)) or set(ids) & set(preparation["deferred_branch_ids"]):
        raise ValueError("invalid preparation branch partition")
    packets, mechanical, blocked = [], [], []
    for branch in sorted(preparation["branches"], key=lambda b: b["branch_id"]):
        verify_fingerprint(branch, "preparation_fingerprint")
        if branch["structural_issues"] or branch["preparation_status"] != "prepared_for_semantic_review":
            blocked.append(branch["branch_id"])
            continue
        same = branch["when"]["source_text"] == branch["when"]["suggested_user_trigger_text"]
        if same != branch["when"]["trigger_text_identical"]:
            raise ValueError("When identity flag differs from text")
        if same:
            mechanical.append({"branch_id": branch["branch_id"], "basis": "identical_text",
                               "event_kind": "same_kind", "restriction_comparison": "preserved",
                               "when_qualifier_extraction": "pending", "reachability": "not_assessed"})
        else:
            packets.append(_packet(branch, KIND, template))
    result = {"schema_version": "agentspectesting.v5-when-kind-packets/v0.1",
              "source_preparation_set_fingerprint": preparation["preparation_set_fingerprint"],
              "packets": packets, "mechanical_decisions": mechanical, "blocked_branch_ids": blocked,
              "deferred_branch_ids": deepcopy(preparation["deferred_branch_ids"]),
              "summary": {"event_kind_calls_full_pool": len(packets), "identical_text_branches": len(mechanical),
                          "restriction_calls_ready": 0, "external_llm_calls": 0},
              "restriction_gate": "accepted same_kind result from this exact packet is required"}
    result["packet_set_fingerprint"] = content_sha256(result)
    return result


def select_previous_when_pilot(kind_set, previous_pilot):
    """Reuse the earlier sample, never its model answers or semantic labels."""
    verify_fingerprint(kind_set, "packet_set_fingerprint")
    verify_fingerprint(previous_pilot, "pilot_fingerprint")
    selected_ids = {c["branch_id"] for p in previous_pilot["packets"]
                    if p["task"] == "when_event_equivalence" for c in p["consumers"]}
    candidates = {p["branch_id"]: p for p in kind_set["packets"]}
    if not selected_ids or not selected_ids <= candidates.keys():
        raise ValueError("previous When pilot is not fully represented in new packet set")
    for packet in kind_set["packets"]:
        verify_fingerprint(packet, "packet_fingerprint")
    for old_packet in previous_pilot["packets"]:
        verify_fingerprint(old_packet, "packet_fingerprint")
        if old_packet["task"] == "when_event_equivalence":
            for consumer in old_packet["consumers"]:
                if consumer["preparation_fingerprint"] != candidates[consumer["branch_id"]]["source_preparation_fingerprint"]:
                    raise ValueError("previous pilot preparation source mismatch")
    packets = [deepcopy(candidates[key]) for key in sorted(selected_ids)]
    result = {"schema_version": "agentspectesting.v5-when-split-pilot/v0.1",
              "source_packet_set_fingerprint": kind_set["packet_set_fingerprint"],
              "previous_pilot_fingerprint": previous_pilot["pilot_fingerprint"],
              "selection_reason": "same previous When sample, revised question; no old answer reuse",
              "packets": packets, "proposed_model": "deepseek-v4-flash",
              "approval_status": "not_requested", "approved_call_count": 0,
              "max_calls_per_packet": 1, "automatic_retries": 0, "target_agent_calls": 0,
              "summary": {"proposed_event_kind_calls": len(packets), "restriction_calls_ready": 0,
                          "external_llm_calls": 0}}
    result["pilot_fingerprint"] = content_sha256(result)
    return result


def validate_split_response(packet, response):
    verify_fingerprint(packet, "packet_fingerprint")
    task = packet["task"]
    if task not in DECISIONS:
        raise ValueError("old or unknown When task cannot answer a split question")
    fields = {"decision", "reason"} | ({"original_quote", "suggested_quote"} if task == RESTRICTIONS else set())
    if not isinstance(response, dict) or set(response) != fields:
        raise ValueError("unexpected response fields")
    if not isinstance(response["decision"], str) or response["decision"] not in DECISIONS[task]:
        raise ValueError("invalid split-task decision")
    if not isinstance(response["reason"], str) or not response["reason"].strip() or len(response["reason"]) > 600:
        raise ValueError("short nonempty reason required")
    if task == RESTRICTIONS:
        for key, source in (("original_quote", "original_when"), ("suggested_quote", "suggested_description")):
            quote = response[key]
            if quote is not None and (not isinstance(quote, str) or not quote or quote not in packet["model_view"][source]):
                raise ValueError("restriction evidence must be a verbatim source quote or null")
        if response["decision"] == "changed" and not (response["original_quote"] or response["suggested_quote"]):
            raise ValueError("changed restrictions need at least one source quote")
    return {"packet_id": packet["packet_id"], "source_packet_fingerprint": packet["packet_fingerprint"],
            "response": deepcopy(response), "format_status": "valid", "semantic_acceptance": "pending_review",
            "test_readiness": "not_assessed"}


def build_restriction_packets(preparation, kind_set, reviewed_results, template):
    """Gate on explicitly reviewed new-task decisions, not raw or legacy responses."""
    verify_fingerprint(preparation, "preparation_set_fingerprint")
    verify_fingerprint(kind_set, "packet_set_fingerprint")
    verify_fingerprint(reviewed_results, "review_fingerprint")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("restriction template required")
    if kind_set.get("schema_version") != "agentspectesting.v5-when-kind-packets/v0.1":
        raise ValueError("new event-kind packet set required")
    if reviewed_results.get("schema_version") != "agentspectesting.v5-when-kind-reviewed/v0.1":
        raise ValueError("new event-kind review schema required")
    if kind_set["source_preparation_set_fingerprint"] != preparation["preparation_set_fingerprint"] or reviewed_results["source_packet_set_fingerprint"] != kind_set["packet_set_fingerprint"]:
        raise ValueError("restriction gate source mismatch")
    packets = {p["packet_id"]: p for p in kind_set["packets"]}
    if len(packets) != len(kind_set["packets"]):
        raise ValueError("duplicate event-kind packet")
    reviews = {r["packet_id"]: r for r in reviewed_results["reviews"]}
    if len(reviews) != len(reviewed_results["reviews"]) or not reviews.keys() <= packets.keys():
        raise ValueError("duplicate or unknown event-kind review")
    branches = {b["branch_id"]: b for b in preparation["branches"]}
    ready, held = [], []
    for key, packet in packets.items():
        verify_fingerprint(packet, "packet_fingerprint")
        branch = branches[packet["branch_id"]]
        verify_fingerprint(branch, "preparation_fingerprint")
        if packet["task"] != KIND or packet["source_preparation_fingerprint"] != branch["preparation_fingerprint"]:
            raise ValueError("event-kind packet source mismatch")
        review = reviews.get(key)
        if review is None:
            reason = "event_kind_not_reviewed"
        else:
            if review["source_packet_fingerprint"] != packet["packet_fingerprint"]:
                raise ValueError("reviewed packet fingerprint mismatch")
            validate_split_response(packet, review["response"])
            if review["semantic_acceptance"] != "accepted_for_narrow_question":
                reason = "event_kind_review_not_accepted"
            elif review["response"]["decision"] == "different_kind":
                reason = "different_kind_requires_driving_to_observation_analysis"
            elif review["response"]["decision"] == "unclear":
                reason = "event_kind_unclear"
            else:
                ready.append(_packet(branch, RESTRICTIONS, template, dependency={
                    "event_kind_packet_id": key, "review_fingerprint": reviewed_results["review_fingerprint"]}))
                continue
        held.append({"branch_id": packet["branch_id"], "reason": reason})
    result = {"schema_version": "agentspectesting.v5-when-restriction-packets/v0.1", "packets": ready,
              "held": held, "source_review_fingerprint": reviewed_results["review_fingerprint"],
              "source_packet_set_fingerprint": kind_set["packet_set_fingerprint"],
              "summary": {"restriction_calls_ready": len(ready), "held_count": len(held), "external_llm_calls": 0},
              "downstream_rule": "changed means an explicit restriction difference, not an invalid test; contextual acceptability requires separate evidence",
              "automatic_contract_promotion": False}
    result["packet_set_fingerprint"] = content_sha256(result)
    return result


def render_split_pilot(pilot, restriction_template):
    lines = ["# When Split Review: Pre-call Preview", "", "No calls have been made for this batch yet; it requires separate approval and cannot reuse the quota of earlier batches.", "",
             "## Question 1: Event kind", ""]
    for packet in pilot["packets"]:
        lines += ["### " + packet["branch_id"], "", "```text", packet["messages"][0]["content"], "```", "",
                  "```text", packet["messages"][1]["content"], "```", ""]
    lines += ["## Question 2: Is the restriction preserved (not yet released)", "", "Messages for a branch are generated only when Question 1 yields a review-accepted same_kind. Different kinds and unclear do not enter this direct-comparison path.", "",
              "```text", restriction_template.strip(), "```", "",
              "Question 2 still uses the original two descriptions and does not rewrite the original text using the model's reasoning from Question 1. This preview has no model judgments pre-filled.", ""]
    return "\n".join(lines)
