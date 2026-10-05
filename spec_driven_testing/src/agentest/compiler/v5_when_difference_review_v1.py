"""Source-bound directional difference discovery, not an equivalence oracle."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import verify_fingerprint
from .v5_json_mode_preflight_v1 import validate_json_object_messages
from .v5_restriction_batch_v1 import validate_restriction_sources

DISCOVER = "when_directional_difference_discovery"
PILOT_SCHEMA = "agentspectesting.v5-when-difference-pilot/v0.1"
PREFLIGHT_SCHEMA = "agentspectesting.v5-when-difference-preflight/v0.1"
DIRECTIONS = (
    ("original_to_suggested", "original_when", "suggested_description"),
    ("suggested_to_original", "suggested_description", "original_when"),
)


def seal(value, field):
    value[field] = content_sha256(value)
    return value


def build_difference_pool(parent, gate, template):
    verify_fingerprint(parent, "pilot_fingerprint")
    verify_fingerprint(gate, "packet_set_fingerprint")
    if gate.get("schema_version") != "agentspectesting.v5-when-restriction-packets/v0.1":
        raise ValueError("reviewed event-kind gate required")
    if parent["source_packet_set_fingerprint"] != gate["packet_set_fingerprint"]:
        raise ValueError("parent and gate differ")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("difference discovery template required")
    allowed = {p["packet_id"]: p for p in gate["packets"]}
    if len(allowed) != len(gate["packets"]):
        raise ValueError("duplicate gate packet")
    packets, seen = [], set()
    for source in parent["packets"]:
        verify_fingerprint(source, "packet_fingerprint")
        if source != allowed.get(source["packet_id"]):
            raise ValueError("parent packet not admitted by gate")
        branch = source["branch_id"]
        if branch in seen or source["task"] != "when_restriction_preservation":
            raise ValueError("duplicate branch or unexpected parent task")
        if source["dependency"]["review_fingerprint"] != gate["source_review_fingerprint"]:
            raise ValueError("review dependency mismatch")
        seen.add(branch)
        for direction, source_key, other_key in DIRECTIONS:
            view = {"source_text": source["model_view"][source_key],
                    "comparison_text": source["model_view"][other_key]}
            if any(not isinstance(v, str) or not v.strip() for v in view.values()):
                raise ValueError("two nonempty complete texts required")
            messages = [{"role": "system", "content": template.strip()},
                        {"role": "user", "content": "Text to check:\n" + view["source_text"]
                         + "\n\nComparison text:\n" + view["comparison_text"]}]
            validate_json_object_messages(messages)
            provenance = {"branch_id": branch, "direction": direction,
                          "parent_pilot_fingerprint": parent["pilot_fingerprint"],
                          "source_gate_fingerprint": gate["packet_set_fingerprint"],
                          "source_packet_fingerprint": source["packet_fingerprint"]}
            identity = {"task": DISCOVER, "messages": messages, "provenance": provenance}
            packets.append(seal({"packet_id": "V5-DIFFERENCE::" + content_sha256(identity)[:24],
                                 "task": DISCOVER, "model_view": view, "messages": messages,
                                 "provenance": provenance, "approval_status": "not_requested",
                                 "result_status": "not_run"}, "packet_fingerprint"))
    if not packets:
        raise ValueError("no selected admitted branches")
    return seal({"schema_version": "agentspectesting.v5-when-difference-pool/v0.1",
                 "parent_pilot_fingerprint": parent["pilot_fingerprint"], "packets": packets,
                 "summary": {"branches": len(seen), "proposed_discovery_calls": len(packets),
                             "candidate_verification_calls_ready": 0, "external_llm_calls": 0},
                 "policy": {"empty_results_are_not_equivalence": True,
                            "automatic_contract_promotion": False, "coverage_review_required": True}},
                "packet_set_fingerprint")


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 1200


def _quote(value, text, *, nullable):
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value.strip() or value not in text:
        raise ValueError("quote must be verbatim from its assigned complete text")


def validate_difference_response(packet, response):
    verify_fingerprint(packet, "packet_fingerprint")
    if packet["task"] != DISCOVER:
        raise ValueError("not a difference discovery packet")
    if not isinstance(response, dict) or set(response) != {"differences", "uncertainties"}:
        raise ValueError("differences and uncertainties required; no equivalence verdict")
    view, seen = packet["model_view"], set()
    for kind in ("differences", "uncertainties"):
        if not isinstance(response[kind], list):
            raise ValueError("result collections must be lists")
        label = "difference" if kind == "differences" else "reason"
        for item in response[kind]:
            if not isinstance(item, dict) or set(item) != {label, "source_quote", "comparison_quote"}:
                raise ValueError("unexpected finding fields")
            if not _text(item[label]):
                raise ValueError("short nonempty finding description required")
            _quote(item["source_quote"], view["source_text"], nullable=kind == "uncertainties")
            _quote(item["comparison_quote"], view["comparison_text"], nullable=True)
            if item["source_quote"] is None and item["comparison_quote"] is None:
                raise ValueError("uncertainty needs evidence from at least one text")
            identity = content_sha256({"kind": kind, "item": item})
            if identity in seen:
                raise ValueError("exact duplicate finding")
            seen.add(identity)
    candidates = [
        {"candidate_id": "V5-DIFF-CANDIDATE::" + content_sha256(
            {"packet_fingerprint": packet["packet_fingerprint"], "finding": item})[:24],
         "finding": deepcopy(item), "candidate_validity": "not_reviewed"}
        for item in response["differences"]
    ]
    return {"packet_id": packet["packet_id"], "source_packet_fingerprint": packet["packet_fingerprint"],
            "response": deepcopy(response), "candidates": candidates, "format_status": "valid",
            "semantic_acceptance": "pending_review", "coverage_review": "not_reviewed",
            "pair_review": "incomplete", "test_readiness": "not_assessed",
            "candidate_verification_calls_ready": 0, "automatic_contract_promotion": False}


def validate_difference_sources(pilot, preflight):
    docs = {}
    for key in ("parent_pilot", "parent_preflight", "restriction_gate", "difference_template"):
        ref = preflight["input_files"][key]
        data = Path(ref["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("difference source changed: " + key)
        docs[key] = data.decode("utf-8") if key == "difference_template" else json.loads(data)
    parent, parent_check = docs["parent_pilot"], docs["parent_preflight"]
    if parent["pilot_fingerprint"] != parent_check["pilot_fingerprint"]:
        raise ValueError("parent pilot identity mismatch")
    validate_restriction_sources(parent, parent_check)
    rebuilt = build_difference_pool(parent, docs["restriction_gate"], docs["difference_template"])
    if rebuilt["packet_set_fingerprint"] != pilot["source_packet_set_fingerprint"]:
        raise ValueError("difference pool source mismatch")
    if pilot["packets"] != rebuilt["packets"]:
        raise ValueError("batch must contain both exact directions for each selected source pair")
    return {"source_reconstruction": "matched", "packet_count": len(rebuilt["packets"])}
