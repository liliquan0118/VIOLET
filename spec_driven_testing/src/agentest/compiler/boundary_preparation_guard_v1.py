"""Step 4 preparation gate for reviewed, unspecified comparison boundaries.

It cannot issue Agent verdicts, execute tools, or accept an unreviewed LLM rule.
"""
from copy import deepcopy

from .artifacts import content_sha256
from .scoped_requirement_judge_v1 import check_seal
from .record_relation_judge_v2 import factual_comparison, prepare_record_question, record_index, validate_response
from .step4_evidence_extensions_v1 import sealed


def compile_guard(packet, checked_response, review, *, value_type):
    check_seal(packet, "packet_fingerprint")
    if packet.get("task") != "explicit_comparison_boundary_v1":
        raise ValueError("boundary question required")
    if validate_response(packet, checked_response["answer"]) != checked_response:
        raise ValueError("changed boundary answer metadata")
    if (review.get("schema_version") != "agentspectesting.boundary-review/v0.1"
            or review.get("decision") != "accepted_no_explicit_equality_rule"
            or review.get("source_packet_fingerprint") != packet["packet_fingerprint"]
            or review.get("source_answer_fingerprint") != content_sha256(checked_response["answer"])
            or not review.get("reviewer") or not review.get("review_document")):
        raise ValueError("matching semantic review required")
    if checked_response["answer"]["boundary"] != "not_specified":
        raise ValueError("this gate does not compile allowed or forbidden policies")
    comparison = packet["model_input"]["comparison"]
    if (comparison["relationship"] != "equal" or comparison["left"]["record"] != "previous record"
            or comparison["right"]["record"] != "next record"):
        raise ValueError("unsupported record-role selector; do not guess scope")
    if value_type not in {"number", "common_clock_datetime"}:
        raise ValueError("supported source value type required")
    return sealed({
        "schema_version": "agentspectesting.boundary-preparation-guard/v0.1",
        "status": "prepared_with_known_rule_gap",
        "source_contract_fingerprint": packet["source_contract_fingerprint"],
        "criterion": packet["model_input"]["requirement"],
        "source_boundary_packet_fingerprint": packet["packet_fingerprint"],
        "source_answer_fingerprint": content_sha256(checked_response["answer"]),
        "source_review_fingerprint": content_sha256(review),
        "review_document": review["review_document"],
        "selector": {"pairing": "adjacent", "left_field": comparison["left"]["field"],
                     "right_field": comparison["right"]["field"], "value_type": value_type},
        "unresolved_relation": "eq",
        "when_matched": {"preparation_status": "insufficient", "reason": "rule_basis_missing", "prepare_model_question": False},
        "when_not_matched": "continue_other_preparation_checks_not_automatic_pass",
        "runtime_dispatch_enabled": False,
    })


def prepare_guarded_question(guard, old_packet, comparison_plan):
    """Prepare a v2 question only if this gate and factual preparation succeed.

    Always recompute the guarded adjacent pairs from record values. Omitting
    an edge from comparison_plan cannot bypass the gate; stored fact labels
    and old model answers are not used. No global trajectory dispatch is added.
    """
    check_seal(guard)
    check_seal(old_packet, "packet_fingerprint")
    if guard.get("schema_version") != "agentspectesting.boundary-preparation-guard/v0.1":
        raise ValueError("unsupported gate")
    if (old_packet.get("mode") != "record_relation"
            or old_packet.get("contract_fingerprint") != guard["source_contract_fingerprint"]
            or old_packet["model_input"]["requirement"] != guard["criterion"]):
        raise ValueError("gate cannot apply to a different criterion or contract")
    selector = guard["selector"]
    if selector["pairing"] != "adjacent" or guard["unresolved_relation"] != "eq":
        raise ValueError("unsupported guard selector")
    records = old_packet["model_input"]["ordered_records"]
    facts, diagnostics, matches = [], [], []
    try:
        record_index(records)
        if not records:
            raise ValueError("no records supplied")
        for left, right in zip(records, records[1:]):
            fact = factual_comparison(records,
                {"ref": left["ref"], "field": selector["left_field"]},
                {"ref": right["ref"], "field": selector["right_field"]}, selector["value_type"])
            facts.append(fact)
            if fact["relation"] == guard["unresolved_relation"]:
                matches.append(deepcopy(fact))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        diagnostics.append({"code": "evidence_insufficient", "detail": str(exc)})
    if matches:
        diagnostics.append({"code": "rule_basis_missing", "detail": "Reviewed requirement does not specify this boundary."})
    question = None
    if not diagnostics:
        try:
            question = prepare_record_question(old_packet, comparison_plan)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            diagnostics.append({"code": "comparison_preparation_incomplete", "detail": str(exc)})
    return sealed({
        "source_guard_fingerprint": guard["fingerprint"],
        "source_packet_fingerprint": old_packet["packet_fingerprint"],
        "comparison_plan_fingerprint": content_sha256(comparison_plan),
        "preparation_status": "insufficient" if diagnostics else "judge_question_prepared",
        "diagnostics": diagnostics, "guarded_comparisons": facts, "matched_boundaries": matches,
        "packet": question, "external_calls": 0, "agent_verdict": None,
        "scope": "Step 4 preparation gate only; no runtime evaluator or Agent verdict",
    })
