"""Single-text evidence inventories, followed by reviewed explicit alignment."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import verify_fingerprint
from .v5_json_mode_preflight_v1 import validate_json_object_messages
from .v5_restriction_batch_v1 import validate_restriction_sources

EXTRACT = "when_explicit_details"
EXTRACT_V2 = "when_explicit_detail_questions"
ALIGN = "when_explicit_alignment"
PILOT_SCHEMA = "agentspectesting.v5-when-details-pilot/v0.1"
PREFLIGHT_SCHEMA = "agentspectesting.v5-when-details-preflight/v0.1"
PILOT_SCHEMA_V2 = "agentspectesting.v5-when-details-pilot/v0.2"
PREFLIGHT_SCHEMA_V2 = "agentspectesting.v5-when-details-preflight/v0.2"
RELATIONS = {"same_explicit_information", "different_explicit_information", "original_only", "suggested_only", "unclear"}


def _seal(value, field):
    value[field] = content_sha256(value)
    return value


def _packet(task, view, template, user_text, provenance):
    if not isinstance(template, str) or not template.strip():
        raise ValueError("explicit-information prompt required")
    messages = [{"role": "system", "content": template.strip()}, {"role": "user", "content": user_text}]
    validate_json_object_messages(messages)
    identity = {"task": task, "messages": messages, "provenance": provenance}
    return _seal({"packet_id": "V5-EXPLICIT::" + task + "::" + content_sha256(identity)[:20],
                  "task": task, "model_view": deepcopy(view), "messages": messages,
                  "provenance": deepcopy(provenance), "approval_status": "not_requested", "result_status": "not_run"},
                 "packet_fingerprint")


def build_detail_pool(gate, template, *, task=EXTRACT):
    if task not in {EXTRACT, EXTRACT_V2}:
        raise ValueError("unknown extraction task version")
    verify_fingerprint(gate, "packet_set_fingerprint")
    if gate.get("schema_version") != "agentspectesting.v5-when-restriction-packets/v0.1":
        raise ValueError("reviewed restriction gate required")
    packets = []
    branches = set()
    for parent in gate["packets"]:
        verify_fingerprint(parent, "packet_fingerprint")
        if parent["task"] != "when_restriction_preservation" or parent["branch_id"] in branches:
            raise ValueError("unexpected or duplicate source branch")
        if parent["dependency"]["review_fingerprint"] != gate["source_review_fingerprint"]:
            raise ValueError("source gate review mismatch")
        branches.add(parent["branch_id"])
        for side, key in (("original", "original_when"), ("suggested", "suggested_description")):
            sentence = parent["model_view"][key]
            packets.append(_packet(task, {"sentence": sentence}, template, "Sentence:\n" + sentence,
                {"branch_id": parent["branch_id"], "side": side,
                 "source_gate_fingerprint": gate["packet_set_fingerprint"],
                 "source_packet_fingerprint": parent["packet_fingerprint"]}))
    version = "0.2" if task == EXTRACT_V2 else "0.1"
    return _seal({"schema_version": "agentspectesting.v5-when-details-pool/v" + version,
                  "source_gate_fingerprint": gate["packet_set_fingerprint"], "packets": packets,
                  "summary": {"branches": len(branches), "extraction_questions": len(packets),
                              "alignment_questions_ready": 0, "external_llm_calls": 0},
                  "policy": {"not_stated_is_not_unrestricted": True, "automatic_contract_promotion": False}},
                 "packet_set_fingerprint")


def _short_text(value, maximum=600):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def validate_explicit_response(packet, response):
    verify_fingerprint(packet, "packet_fingerprint")
    if not isinstance(response, dict):
        raise ValueError("JSON object required")
    if packet["task"] == EXTRACT_V2:
        if set(response) != {"details"} or not isinstance(response["details"], list):
            raise ValueError("v0.2 requires only the details list")
        sentence, seen = packet["model_view"]["sentence"], set()
        for item in response["details"]:
            if not isinstance(item, dict) or set(item) != {"question", "answer", "quote", "status"}:
                raise ValueError("v0.2 detail requires question, answer, quote and status")
            if not _short_text(item["question"]) or not _short_text(item["answer"]):
                raise ValueError("short question and answer required")
            if not isinstance(item["status"], str) or item["status"] not in {"explicit", "unclear"}:
                raise ValueError("explicit or unclear status required")
            if not _short_text(item["quote"], len(sentence)) or item["quote"] not in sentence:
                raise ValueError("detail evidence must be an exact quote from this sentence")
            identity = content_sha256(item)
            if identity in seen:
                raise ValueError("duplicate detail")
            seen.add(identity)
    elif packet["task"] == EXTRACT:
        if set(response) != {"details", "uncertain_quotes"} or not isinstance(response["details"], list) or not isinstance(response["uncertain_quotes"], list):
            raise ValueError("details and uncertain_quotes lists required")
        sentence = packet["model_view"]["sentence"]
        seen = set()
        for item in response["details"]:
            if not isinstance(item, dict) or set(item) != {"statement", "quote"} or not _short_text(item["statement"]):
                raise ValueError("each detail needs a short statement and quote")
            quote = item["quote"]
            if not _short_text(quote, len(sentence)) or quote not in sentence:
                raise ValueError("detail evidence must be an exact quote from this sentence")
            identity = (item["statement"], quote)
            if identity in seen:
                raise ValueError("duplicate detail")
            seen.add(identity)
        quotes = response["uncertain_quotes"]
        if any(not _short_text(q, len(sentence)) or q not in sentence for q in quotes):
            raise ValueError("uncertainty must cite this sentence")
        if len(quotes) != len(set(quotes)):
            raise ValueError("duplicate uncertain quote")
    elif packet["task"] == ALIGN:
        if set(response) != {"groups"} or not isinstance(response["groups"], list):
            raise ValueError("alignment groups required")
        expected = {side: {d["id"] for d in packet["model_view"][side]["details"]} for side in ("original", "suggested")}
        seen = {"original": set(), "suggested": set()}
        for group in response["groups"]:
            if not isinstance(group, dict) or set(group) != {"original_ids", "suggested_ids", "relation", "reason"}:
                raise ValueError("unexpected alignment fields")
            relation = group["relation"]
            if not isinstance(relation, str) or relation not in RELATIONS or not _short_text(group["reason"]):
                raise ValueError("alignment needs a relation and short reason")
            for side in seen:
                ids = group[side + "_ids"]
                if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                    raise ValueError("detail IDs must be lists of strings")
                if len(ids) != len(set(ids)) or not set(ids) <= expected[side] or set(ids) & seen[side]:
                    raise ValueError("unknown or repeated detail ID")
                seen[side].update(ids)
            has_original, has_suggested = bool(group["original_ids"]), bool(group["suggested_ids"])
            required = {"original_only": (True, False), "suggested_only": (False, True)}.get(relation, (True, True))
            if (has_original, has_suggested) != required:
                raise ValueError("relation does not match populated sides")
        if seen != expected:
            raise ValueError("every extracted detail must be accounted for")
    else:
        raise ValueError("unknown explicit-information question")
    return {"packet_id": packet["packet_id"], "source_packet_fingerprint": packet["packet_fingerprint"],
            "response": deepcopy(response), "format_status": "valid", "semantic_acceptance": "pending_review",
            "test_readiness": "not_assessed"}


def build_alignment_pool(detail_pool, reviewed, template):
    verify_fingerprint(detail_pool, "packet_set_fingerprint")
    verify_fingerprint(reviewed, "review_fingerprint")
    if reviewed.get("schema_version") != "agentspectesting.v5-when-details-reviewed/v0.1" or reviewed["source_packet_set_fingerprint"] != detail_pool["packet_set_fingerprint"]:
        raise ValueError("review must belong to this detail pool")
    source = {p["packet_id"]: p for p in detail_pool["packets"]}
    reviews = {r["packet_id"]: r for r in reviewed["reviews"]}
    if len(source) != len(detail_pool["packets"]) or len(reviews) != len(reviewed["reviews"]) or not reviews.keys() <= source.keys():
        raise ValueError("duplicate or unknown reviewed packet")
    branches = {}
    for packet in source.values():
        verify_fingerprint(packet, "packet_fingerprint")
        if packet["task"] not in {EXTRACT, EXTRACT_V2}:
            raise ValueError("only extraction packets may feed alignment")
        entry = branches.setdefault(packet["provenance"]["branch_id"], {})
        side = packet["provenance"]["side"]
        if side not in {"original", "suggested"} or side in entry:
            raise ValueError("invalid or duplicate source side")
        entry[side] = packet
    ready, held = [], []
    for branch, sides in branches.items():
        view, blockers = {}, []
        for side in ("original", "suggested"):
            packet = sides.get(side)
            review = reviews.get(packet["packet_id"]) if packet else None
            if review is None:
                blockers.append(side + "_not_reviewed")
                continue
            if review["source_packet_fingerprint"] != packet["packet_fingerprint"]:
                raise ValueError("reviewed extraction fingerprint mismatch")
            validate_explicit_response(packet, review["response"])
            if review["semantic_acceptance"] != "accepted_for_narrow_question":
                blockers.append(side + "_not_accepted")
                continue
            uncertain = (any(d["status"] == "unclear" for d in review["response"]["details"])
                         if packet["task"] == EXTRACT_V2 else bool(review["response"]["uncertain_quotes"]))
            if uncertain:
                blockers.append(side + "_uncertainty_unresolved")
                continue
            if not review["response"]["details"]:
                blockers.append(side + "_no_details_extracted")
                continue
            prefix = "O" if side == "original" else "S"
            details = []
            for i, d in enumerate(review["response"]["details"], 1):
                item = {"id": prefix + str(i), **deepcopy(d)}
                if packet["task"] == EXTRACT_V2:
                    item["statement"] = d["question"] + " " + d["answer"]
                details.append(item)
            view[side] = {"sentence": packet["model_view"]["sentence"], "details": details}
        if blockers:
            held.append({"branch_id": branch, "reasons": blockers})
            continue
        user = []
        for side in ("original", "suggested"):
            user += [side.title() + " sentence:", view[side]["sentence"], "Explicit details:"]
            user += [d["id"] + ": " + d["statement"] + "\nSource quote: " + d["quote"] for d in view[side]["details"]]
            if not view[side]["details"]:
                user.append("No explicit details extracted; this does not mean unrestricted.")
            user.append("")
        ready.append(_packet(ALIGN, view, template, "\n".join(user),
            {"branch_id": branch, "detail_pool_fingerprint": detail_pool["packet_set_fingerprint"],
             "review_fingerprint": reviewed["review_fingerprint"]}))
    return _seal({"schema_version": "agentspectesting.v5-when-alignment-pool/v0.1", "packets": ready, "held": held,
                  "summary": {"alignment_questions_ready": len(ready), "external_llm_calls": 0},
                  "automatic_contract_promotion": False}, "packet_set_fingerprint")


def validate_detail_sources(pilot, preflight):
    docs = {}
    for key in ("parent_pilot", "parent_preflight", "restriction_gate", "detail_template"):
        ref = preflight["input_files"][key]
        data = Path(ref["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("explicit detail source changed: " + key)
        docs[key] = data.decode("utf-8") if key == "detail_template" else json.loads(data)
    parent, parent_check = docs["parent_pilot"], docs["parent_preflight"]
    if parent["pilot_fingerprint"] != parent_check["pilot_fingerprint"]:
        raise ValueError("parent pilot identity mismatch")
    validate_restriction_sources(parent, parent_check)
    gate = docs["restriction_gate"]
    if gate["packet_set_fingerprint"] != parent["source_packet_set_fingerprint"]:
        raise ValueError("explicit extraction gate mismatch")
    task = EXTRACT_V2 if pilot["schema_version"] == PILOT_SCHEMA_V2 else EXTRACT
    rebuilt = build_detail_pool(gate, docs["detail_template"], task=task)
    if rebuilt["packet_set_fingerprint"] != pilot["source_packet_set_fingerprint"]:
        raise ValueError("explicit detail pool mismatch")
    if pilot["packets"] != rebuilt["packets"]:
        raise ValueError("extraction pilot must contain the complete source-bound two-sided pool")
    return {"source_reconstruction": "matched", "packet_count": len(pilot["packets"])}
