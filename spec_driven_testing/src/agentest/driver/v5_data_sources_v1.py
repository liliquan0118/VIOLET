"""Read-only data-source packaging. No scenario selection or predicate evaluation."""

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from agentest.compiler.artifacts import content_sha256
from agentest.compiler.v5_field_projection_v1 import validate_snapshot
from agentest.compiler.v5_given_when_preparation_v1 import verify_fingerprint


SCHEMA = "agentspectesting.v5-data-source-bundle/v0.1"
PAYLOADS = ("database_snapshot.json", "tool_schema_snapshot.json", "clock.json")


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def required_schema_tools(plans):
    verify_fingerprint(plans, "plan_set_fingerprint")
    if plans["schema_version"] != "agentspectesting.v5-driver-plan-set/v0.1":
        raise ValueError("unsupported plan schema")
    names = set()
    for plan in plans["plans"]:
        verify_fingerprint(plan, "driver_plan_fingerprint")
        names.update(plan["observation_plan"]["tool_names_from_effective_routes"])
        for check in plan["oracle_handoff"]["checks"]:
            route = check["preparation"]["effective_route"]
            if route["kind"] == "record_relation":
                # Public evidence-producing methods from the already reviewed source basis.
                names.update(n for n in route["evidence_source_basis"]["methods"] if not n.startswith("_"))
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("invalid tool name")
    return sorted(names)


def _check_clock(value):
    if not isinstance(value, str) or "T" not in value:
        raise ValueError("reference time must be an explicit ISO datetime, not a default date")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return "offset_explicit_preserved" if parsed.tzinfo else "timezone_unspecified_not_assumed_utc"


def _check_database(database):
    if not isinstance(database, dict):
        raise ValueError("database must be an object")
    for table in ("users", "reservations", "flights"):
        rows = database.get(table)
        if not isinstance(rows, dict) or any(not isinstance(r, dict) for r in rows.values()):
            raise ValueError(f"invalid airline table shape: {table}")
    # Shape only. Do not resolve identities, relations, candidates or business conditions.


def _check_schemas(snapshot, requested):
    validate_snapshot(snapshot, requested)
    def walk(value, root):
        if isinstance(value, dict):
            ref = value.get("$ref")
            if ref is not None and (not isinstance(ref, str) or not ref.startswith("#")):
                raise ValueError("external schema references are not allowed in an offline bundle")
            if ref is not None and ref != "#":
                if not ref.startswith("#/"):
                    raise ValueError("only local JSON-pointer schema references are supported")
                current = root
                try:
                    for part in ref[2:].split("/"):
                        current = current[part.replace("~1", "/").replace("~0", "~")]
                except (KeyError, TypeError):
                    raise ValueError("unresolved local schema reference") from None
            for child in value.values():
                walk(child, root)
        elif isinstance(value, list):
            for child in value:
                walk(child, root)
    for record in snapshot["tools"].values():
        walk(record["parameters"], record["parameters"])
        Draft202012Validator.check_schema(record["parameters"])


def prepare_bundle(database, database_receipt, tool_snapshot, plans, assembly_fingerprint, source_files):
    verify_fingerprint(database_receipt, "binding_set_fingerprint")
    if database_receipt["database_fingerprint"] != content_sha256(database):
        raise ValueError("database differs from historical receipt")
    if database_receipt["source_assembly_set_fingerprint"] != assembly_fingerprint:
        raise ValueError("historical database receipt refers to another assembly")
    _check_database(database)
    requested = required_schema_tools(plans)
    _check_schemas(tool_snapshot, requested)
    value = database_receipt["reference_time"]
    timezone_status = _check_clock(value)
    clock = {"value": value, "timezone_status": timezone_status,
             "source_kind": "historical_environment_reference_time",
             "source_receipt_fingerprint": database_receipt["binding_set_fingerprint"],
             "provider": deepcopy(database_receipt["source_files"]["clock_provider"]),
             "wall_clock_used": False, "provider_executed_this_stage": False}
    provider = clock["provider"]
    schema_sources = tool_snapshot["provenance"].get("source_files", [])
    matching = [f for f in schema_sources if f["path"] == provider["path"]]
    if len(matching) > 1:
        raise ValueError("ambiguous schema source provenance")
    provider_status = ("not_recorded" if not matching else "same_source_file_hash"
                       if matching[0]["sha256"] == provider["sha256"] else "source_file_hash_differs")
    missing = tool_snapshot["missing_tools"]
    blocked = bool(missing) or provider_status == "source_file_hash_differs"
    payloads = {"database_snapshot.json": deepcopy(database), "tool_schema_snapshot.json": deepcopy(tool_snapshot), "clock.json": clock}
    manifest = {
        "schema_version": SCHEMA, "source_plan_set_fingerprint": plans["plan_set_fingerprint"],
        "source_assembly_fingerprint": assembly_fingerprint,
        "source_database_receipt_fingerprint": database_receipt["binding_set_fingerprint"],
        "source_files": deepcopy(source_files),
        "payloads": {name: {"sha256": sha256(encode(data)).hexdigest(), "content_fingerprint": content_sha256(data)} for name, data in payloads.items()},
        "required_tool_names": requested,
        "integrity": {"database_receipt": "matched", "schema_snapshot": "validated", "reference_time": "explicit_preserved"},
        "compatibility": {"clock_provider_vs_schema_source": provider_status,
            "database_model_version": "not_recorded_in_historical_receipt",
            "deployed_endpoint_schema": "not_verified",
            "whole_environment_compatibility": "not_established",
            "clock_timezone": timezone_status},
        "status": "blocked_source_gap" if blocked else "prepared_with_compatibility_limits",
        "missing_tools": deepcopy(missing),
        "summary": {"table_counts": {t: len(database[t]) for t in ("users", "reservations", "flights")},
            "requested_tool_count": len(requested), "available_tool_count": len(tool_snapshot["tools"]),
            "missing_tool_count": len(missing), "external_calls": 0, "business_tool_calls": 0},
        "scope": {"candidate_selection": False, "predicate_evaluation": False, "fixture_mutation": False,
                  "user_prompt_generation": False, "runtime_execution": False},
        "consumer_obligations": ["resolve compatibility limits before claiming executable test readiness",
            "verify identities, relations and relevant conditions separately",
            "do not infer operational tools solely from Oracle observation tools",
            "extend schema coverage explicitly if later binding needs additional tools"],
    }
    manifest["bundle_fingerprint"] = content_sha256(manifest)
    return manifest, payloads


def write_bundle(output, manifest, payloads):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in PAYLOADS:
        (output / name).write_bytes(encode(payloads[name]))
    (output / "sources.json").write_bytes(encode(manifest))


def load_bundle(directory):
    """Verify a saved package with no fallback to live tau2 or original files."""
    directory = Path(directory)
    manifest = read_json(directory / "sources.json")
    verify_fingerprint(manifest, "bundle_fingerprint")
    if manifest["schema_version"] != SCHEMA or set(manifest["payloads"]) != set(PAYLOADS):
        raise ValueError("unsupported data-source bundle")
    payloads = {}
    for name in PAYLOADS:
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("bundle payload must stay inside bundle directory")
        if sha256(path.read_bytes()).hexdigest() != manifest["payloads"][name]["sha256"]:
            raise ValueError(f"payload byte hash mismatch: {name}")
        data = read_json(path)
        if content_sha256(data) != manifest["payloads"][name]["content_fingerprint"]:
            raise ValueError(f"payload content fingerprint mismatch: {name}")
        payloads[name] = data
    _check_database(payloads["database_snapshot.json"])
    _check_schemas(payloads["tool_schema_snapshot.json"], manifest["required_tool_names"])
    _check_clock(payloads["clock.json"]["value"])
    if (manifest["status"] != "prepared_with_compatibility_limits"
            or payloads["tool_schema_snapshot.json"]["missing_tools"]
            or manifest["compatibility"]["clock_provider_vs_schema_source"] == "source_file_hash_differs"):
        raise ValueError("data-source bundle has blocking source gaps")
    return {"manifest": manifest, "database": payloads["database_snapshot.json"],
            "tool_snapshot": payloads["tool_schema_snapshot.json"], "clock": payloads["clock.json"]}
