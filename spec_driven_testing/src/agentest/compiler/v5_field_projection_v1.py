"""Technical field extraction only: neither semantic grounding nor Oracle verdicts."""

from copy import deepcopy

from .v5_given_when_preparation_v1 import verify_fingerprint
from .v5_source_assertion_v1 import validate_assertions
from .v5_step3_intake_v1 import seal


SNAPSHOT_SCHEMA = "agentspectesting.v5-tool-schema-snapshot/v0.1"
SCHEMA = "agentspectesting.v5-field-projection-preparation/v0.1"


def requested_tools(assertions):
    return sorted({h["tool_name"] for row in assertions["branches"]
                   for a in row["assertions"] for h in a["observation_mapping_hints"]
                   if isinstance(h.get("tool_name"), str) and h["tool_name"]})


def make_snapshot(tools, provenance, requested):
    """Schema snapshot supplied by an adapter; hashes are integrity, not trust proofs."""
    if not isinstance(requested, list) or any(not isinstance(n, str) or not n for n in requested):
        raise ValueError("requested tool names must be nonempty strings")
    if not isinstance(tools, dict) or any(not isinstance(k, str) or not isinstance(v, dict) for k, v in tools.items()):
        raise ValueError("tools must map names to schema records")
    if set(tools) - set(requested):
        raise ValueError("snapshot contains unrequested tools")
    for record in tools.values():
        schema = record.get("parameters")
        if not isinstance(schema, dict) or not isinstance(schema.get("properties", {}), dict):
            raise ValueError("tool parameters/properties must be schema objects")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(n, str) for n in required):
            raise ValueError("required must be an array of field names")
    return seal({"schema_version": SNAPSHOT_SCHEMA, "requested_tools": sorted(set(requested)),
                 "tools": deepcopy(tools), "missing_tools": sorted(set(requested) - set(tools)),
                 "provenance": deepcopy(provenance)}, "snapshot_fingerprint")


def validate_snapshot(snapshot, requested):
    verify_fingerprint(snapshot, "snapshot_fingerprint")
    if snapshot != make_snapshot(snapshot["tools"], snapshot["provenance"], requested):
        raise ValueError("snapshot shape or requested tool set differs")


def prepare_projections(assertions, contracts, evidence, snapshot):
    validate_assertions(assertions, contracts, evidence)
    validate_snapshot(snapshot, requested_tools(assertions))
    rows = []
    for branch in assertions["branches"]:
        for assertion in branch["assertions"]:
            field = assertion["subject"]["name"]
            candidates = []
            by_tool = {}
            for hint in assertion["observation_mapping_hints"]:
                by_tool.setdefault(hint.get("tool_name"), []).append(hint["candidate_id"])
            for tool_name, ids in sorted(by_tool.items(), key=lambda pair: str(pair[0])):
                record = snapshot["tools"].get(tool_name)
                schema = (record or {}).get("parameters", {})
                endpoint = schema.get("properties", {}).get(field)
                status = "direct_field_type_matches"
                if record is None:
                    status = "tool_not_in_snapshot"
                elif schema.get("type") != "object" or any(k in schema for k in ("$ref", "allOf", "anyOf", "oneOf", "if")):
                    status = "unsupported_parameter_schema"
                elif not isinstance(endpoint, dict):
                    status = "field_not_in_schema"
                elif not isinstance(endpoint.get("type"), str) or any(k in endpoint for k in ("$ref", "allOf", "anyOf", "oneOf", "if")):
                    status = "requires_schema_resolution"
                elif endpoint["type"] != assertion["predicate"]["type"]:
                    status = "source_schema_type_conflict"
                candidates.append({"tool_name": tool_name, "source_candidate_ids": sorted(set(ids)),
                                   "status": status, "tool_description": (record or {}).get("description"),
                                   "field_schema": deepcopy(endpoint),
                                   "schema_field_required": field in schema.get("required", []),
                                   "semantic_correspondence": "not_established_by_schema_check"})
            # Do not choose a convenient matching candidate when another mapping is unresolved.
            ready = len(candidates) == 1 and candidates[0]["status"] == "direct_field_type_matches"
            projection = None
            if ready:
                projection = {"event_stream": "canonical_tau_events",
                              "event_filter": {"event_kind": "assistant_tool_call", "tool_name": candidates[0]["tool_name"]},
                              "value_path": ["arguments", field],
                              "selection": "all_matching_events_in_supplied_trace"}
            rows.append({"branch_id": branch["branch_id"], "assertion_id": assertion["assertion_id"],
                         "candidates": candidates, "projection": projection,
                         "technical_projection_ready": ready,
                         "status": "technical_projection_only" if ready else "mapping_unresolved",
                         "source_scope": deepcopy(assertion["scope"]),
                         "source_predicate": deepcopy(assertion["predicate"]),
                         "spec_scope_binding_accepted": False, "runtime_ready": False,
                         "remaining_requirements": ["semantic_field_and_operation_correspondence",
                                                    "spec_scope_and_applicability", "trace_completeness_and_provenance"],
                         "no_matching_event": "no_observation_not_a_spec_verdict"})
    return seal({"schema_version": SCHEMA,
                 "source_assertion_set_fingerprint": assertions["assertion_set_fingerprint"],
                 "tool_snapshot_fingerprint": snapshot["snapshot_fingerprint"],
                 "projections": rows,
                 "uncompiled_branches": [b["branch_id"] for b in assertions["branches"] if not b["assertions"]],
                 "allowed_use": "offline_observation_extraction_not_oracle_execution",
                 "runtime_lowering_allowed": False,
                 "summary": {"assertion_count": len(rows), "technical_projection_count": sum(r["technical_projection_ready"] for r in rows),
                             "runtime_ready": 0, "external_llm_calls": 0, "target_agent_calls": 0}},
                "preparation_fingerprint")


def validate_preparation(result, assertions, contracts, evidence, snapshot):
    if result != prepare_projections(assertions, contracts, evidence, snapshot):
        raise ValueError("projection preparation differs from source and schema reconstruction")


def extract_projection(events, projection):
    """Project already-normalized evidence; never coerce values or apply spec predicates.

    Input is a single caller-selected trace, not proof of a complete/scoped trajectory.
    This utility does not authorize a projection as the correct spec observation.
    """
    if not isinstance(projection, dict):
        raise ValueError("a technical projection is required")
    event_filter = projection.get("event_filter")
    path = projection.get("value_path")
    if (not isinstance(event_filter, dict) or set(event_filter) != {"event_kind", "tool_name"}
            or event_filter["event_kind"] != "assistant_tool_call"
            or not isinstance(event_filter["tool_name"], str) or not event_filter["tool_name"]
            or not isinstance(path, list) or len(path) != 2 or path[0] != "arguments"
            or not isinstance(path[1], str) or not path[1]
            or projection != {"event_stream": "canonical_tau_events", "event_filter": event_filter,
                              "value_path": path, "selection": "all_matching_events_in_supplied_trace"}):
        raise ValueError("unsupported technical projection")
    if not isinstance(events, list):
        raise ValueError("canonical events must be an array")
    observations = []
    last_index = -1
    for event in events:
        if not isinstance(event, dict) or type(event.get("event_index")) is not int or event["event_index"] <= last_index:
            raise ValueError("canonical events need unique increasing integer indices")
        last_index = event["event_index"]
        if event.get("event_kind") not in {"assistant_tool_call", "assistant_message", "user_message", "tool_result"}:
            raise ValueError("unknown canonical event kind")
        if event["event_kind"] != "assistant_tool_call":
            continue
        if not isinstance(event.get("tool_name"), str) or not event["tool_name"]:
            raise ValueError("tool call has no tool name")
        if event["tool_name"] != event_filter["tool_name"]:
            continue
        item = {"event_index": event["event_index"], "source_message_index": event.get("source_message_index"),
                "tool_call_id": event.get("tool_call_id")}
        arguments = event.get("arguments")
        if not isinstance(arguments, dict):
            item["status"] = "malformed_arguments"
        elif path[1] not in arguments:
            item["status"] = "field_missing"
        else:
            item.update(status="value_observed", value=deepcopy(arguments[path[1]]))
        observations.append(item)
    return {"status": "observations_extracted" if observations else "no_matching_event",
            "observations": observations, "matching_event_count": len(observations),
            "scope_and_applicability": "not_assessed", "trace_completeness": "not_assessed",
            "spec_verdict": None, "task_completion_verdict": None}
