"""Recheck second-question provenance before any provider configuration is read."""

import hashlib
import json
from pathlib import Path

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import verify_fingerprint
from .v5_when_split_review_v1 import build_restriction_packets

PILOT_SCHEMA = "agentspectesting.v5-when-restriction-pilot/v0.1"
PREFLIGHT_SCHEMA = "agentspectesting.v5-when-restriction-preflight/v0.1"


def validate_restriction_sources(pilot, preflight):
    """Reconstruct the semantic gate from hash-bound local source artifacts."""
    if pilot.get("schema_version") != PILOT_SCHEMA or preflight.get("schema_version") != PREFLIGHT_SCHEMA:
        raise ValueError("restriction batch schemas required")
    verify_fingerprint(pilot, "pilot_fingerprint")
    verify_fingerprint(preflight, "preflight_fingerprint")
    refs = preflight["input_files"]
    documents = {}
    for key in ("preparation", "kind_pool", "kind_review", "restriction_template", "restriction_gate"):
        ref = refs[key]
        data = Path(ref["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("restriction source file changed: " + key)
        documents[key] = data.decode("utf-8") if key == "restriction_template" else json.loads(data)
    review = documents["kind_review"]
    rebuilt = build_restriction_packets(documents["preparation"], documents["kind_pool"], review,
                                       documents["restriction_template"])
    if content_sha256(rebuilt) != content_sha256(documents["restriction_gate"]):
        raise ValueError("restriction gate does not match source reconstruction")
    if pilot["source_packet_set_fingerprint"] != rebuilt["packet_set_fingerprint"] or preflight["source_packet_set_fingerprint"] != rebuilt["packet_set_fingerprint"]:
        raise ValueError("restriction batch references a different gate")
    if preflight["source_review_fingerprint"] != review["review_fingerprint"]:
        raise ValueError("restriction batch review mismatch")
    allowed = {p["packet_id"]: p for p in rebuilt["packets"]}
    ids = [p["packet_id"] for p in pilot["packets"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate restriction batch")
    for packet in pilot["packets"]:
        if packet["packet_id"] not in allowed or content_sha256(packet) != content_sha256(allowed[packet["packet_id"]]):
            raise ValueError("restriction packet is not exactly admitted by reviewed gate")
    return {"source_reconstruction": "matched", "packet_count": len(ids),
            "source_review_fingerprint": review["review_fingerprint"]}
