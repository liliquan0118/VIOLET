"""Structured evidence references and a separate explicit-boundary question.

Comparison results are facts, never inferred policy operators or verdicts.
This module does not mutate v1 packets, process trajectories or call a model.
"""
from copy import deepcopy
from datetime import datetime
import json
import math

from .scoped_requirement_judge_v1 import check_seal
from .step4_evidence_extensions_v1 import sealed


def record_index(records):
    result = {}
    for record in records:
        ref = record.get("ref")
        if not isinstance(ref, str) or not ref or ref in result:
            raise ValueError("unique nonempty record references required")
        result[ref] = record
    return result


def resolve_field(records, pointer):
    if not isinstance(pointer, dict) or set(pointer) != {"ref", "field"}:
        raise ValueError("field evidence requires ref and field only")
    ref, field = pointer["ref"], pointer["field"]
    if (not isinstance(ref, str) or not isinstance(field, str) or field == "ref"
            or ref not in records or field not in records[ref] or records[ref][field] is None):
        raise ValueError("unknown or missing evidence field")
    value = records[ref][field]
    if isinstance(value, (dict, list)):
        raise ValueError("scalar evidence field required")
    return deepcopy(value)


def factual_comparison(records, left, right, value_type):
    index = record_index(records)
    values = [resolve_field(index, side) for side in (left, right)]
    if value_type == "number":
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("finite numeric values required")
        compared = values
    elif value_type == "common_clock_datetime":
        if any(not isinstance(v, str) or "T" not in v for v in values):
            raise ValueError("normalized dated timestamps required")
        compared = [datetime.fromisoformat(v) for v in values]
        if (compared[0].utcoffset() is None) != (compared[1].utcoffset() is None):
            raise ValueError("mixed clock conventions")
    else:
        raise ValueError("unsupported comparison value type")
    a, b = compared
    relation = "lt" if a < b else "gt" if a > b else "eq"
    return {"left": deepcopy(left), "right": deepcopy(right), "left_value": values[0], "right_value": values[1],
            "relation": relation, "meaning": "factual_order_only_not_allowed_or_forbidden"}


def prepare_record_question(old_packet, comparisons):
    check_seal(old_packet, "packet_fingerprint")
    if old_packet.get("mode") != "record_relation":
        raise ValueError("this revision must not replace behavior calibration")
    old = old_packet["model_input"]
    facts = [factual_comparison(old["ordered_records"], c["left"], c["right"], c["value_type"]) for c in comparisons]
    if not facts:
        raise ValueError("explicit comparison plan required")
    return sealed({"task": "record_requirement_check_v2", "source_packet_fingerprint": old_packet["packet_fingerprint"],
        "source_contract_fingerprint": old_packet["contract_fingerprint"],
        "comparison_plan": deepcopy(comparisons),
        "model_input": {"requirement": old["requirement"], "ordered_records": deepcopy(old["ordered_records"]),
                        "time_convention": old["time_convention"], "computed_facts": facts},
        "scope": "same_previously_gated_target_records_not_full_Given_When_or_execution"}, "packet_fingerprint")


def prepare_boundary_question(old_packet, left_role, left_field, right_role, right_field):
    check_seal(old_packet, "packet_fingerprint")
    if old_packet.get("mode") != "record_relation":
        raise ValueError("record relation source required")
    for value in (left_role, left_field, right_role, right_field):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("explicit comparison roles required")
    return sealed({"task": "explicit_comparison_boundary_v1", "source_packet_fingerprint": old_packet["packet_fingerprint"],
        "source_contract_fingerprint": old_packet["contract_fingerprint"],
        "model_input": {"requirement": old_packet["model_input"]["requirement"],
                        "comparison": {"left": {"record": left_role, "field": left_field},
                                       "right": {"record": right_role, "field": right_field}, "relationship": "equal"}},
        "acceptance_boundary": "text_explicitness_proposal_not_new_domain_policy_or_Agent_verdict"}, "packet_fingerprint")


def render_prompt(packet):
    check_seal(packet, "packet_fingerprint")
    if packet["task"] == "record_requirement_check_v2":
        question = (
            "Do these ordered records satisfy the stated requirement?\n"
            "The computed facts describe order, not policy. Use only the supplied requirement; do not invent thresholds. "
            "If a missing rule changes the answer, return insufficient. Given/When applicability is checked separately.\n"
            "Return JSON with decision (satisfied, violated, insufficient), evidence (array of {ref, field}), and reason (one short sentence). "
            "Cite existing record IDs and field names only; do not copy values or JSON text. satisfied/violated need evidence.\n"
        )
    elif packet["task"] == "explicit_comparison_boundary_v1":
        question = (
            "Does the requirement explicitly allow or forbid the stated equality?\n"
            "Read the wording only, without adding domain assumptions. If it does not settle equality, choose not_specified. "
            "This asks what the text states, not whether an actual itinerary is feasible.\n"
            "Return JSON with boundary (allowed, forbidden, not_specified), source_span "
            "(an exact requirement excerpt establishing allowed/forbidden, or null for not_specified), and reason (one short sentence).\n"
        )
    else:
        raise ValueError("unsupported task")
    return question + "\n" + json.dumps(packet["model_input"], ensure_ascii=False, indent=2)


def validate_response(packet, answer):
    check_seal(packet, "packet_fingerprint")
    if not isinstance(answer, dict) or not isinstance(answer.get("reason"), str) or not answer["reason"].strip():
        raise ValueError("response and reason required")
    if packet["task"] == "record_requirement_check_v2":
        if set(answer) != {"decision", "evidence", "reason"} or answer["decision"] not in {"satisfied", "violated", "insufficient"}:
            raise ValueError("invalid decision response")
        if not isinstance(answer["evidence"], list) or (not answer["evidence"] and answer["decision"] != "insufficient"):
            raise ValueError("evidence required")
        index = record_index(packet["model_input"]["ordered_records"])
        resolved, seen = [], set()
        for pointer in answer["evidence"]:
            value = resolve_field(index, pointer)
            key = pointer["ref"], pointer["field"]
            if key in seen:
                raise ValueError("duplicate evidence pointer")
            seen.add(key)
            resolved.append({**pointer, "value": value})
        return {"source_packet_fingerprint": packet["packet_fingerprint"], "answer": deepcopy(answer),
                "resolved_evidence": resolved, "validation": "field_identity_and_values_not_semantic_correctness"}
    if packet["task"] == "explicit_comparison_boundary_v1":
        if set(answer) != {"boundary", "source_span", "reason"} or answer["boundary"] not in {"allowed", "forbidden", "not_specified"}:
            raise ValueError("invalid boundary response")
        span = answer["source_span"]
        if answer["boundary"] == "not_specified":
            if span is not None:
                raise ValueError("not_specified requires null span")
        elif not isinstance(span, str) or not span.strip() or span not in packet["model_input"]["requirement"]:
            raise ValueError("boundary needs an exact requirement span")
        return {"source_packet_fingerprint": packet["packet_fingerprint"], "answer": deepcopy(answer),
                "policy_status": "proposal_requires_semantic_review", "agent_verdict": None}
    raise ValueError("unsupported task")
