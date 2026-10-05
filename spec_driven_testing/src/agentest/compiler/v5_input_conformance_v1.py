"""Ground selected input-schema checks; keep GWT coverage and task results separate."""

from copy import deepcopy
import json

from jsonschema import Draft202012Validator

from .artifacts import content_sha256
from .v5_field_projection_v1 import validate_preparation, extract_projection
from .v5_step3_intake_v1 import index, seal


SCHEMA = "agentspectesting.v5-selected-input-conformance/v0.1"


def compile_input_checks(projections, assertions, contracts, evidence, snapshot, legacy_specs, catalog):
    validate_preparation(projections, assertions, contracts, evidence, snapshot)
    old = index(legacy_specs["specs"], "spec_id")
    tools = index(catalog["tools"], "tool_name")
    sources = index(assertions["branches"], "branch_id")
    rows = []
    for projection in projections["projections"]:
        source = sources[projection["branch_id"]]
        context = source["source_context"]
        assertion = index(source["assertions"], "assertion_id")[projection["assertion_id"]]
        previous = old.get(context["spec_id"])
        issues = []
        if context["origin"] != "schema" or context["kind"] != "ARG":
            issues.append("not_explicit_schema_argument_source")
        core_equal = previous is not None and all(previous.get(k) == context[k] for k in ("spec_id", "origin", "kind", "rule_text"))
        if not core_equal:
            issues.append("historical_core_rule_differs_or_missing")
        bindings, endpoints = [], []
        spec_position = next((i for i, s in enumerate(legacy_specs["specs"]) if s["spec_id"] == context["spec_id"]), None)
        program = None
        if not projection["technical_projection_ready"]:
            issues.append("technical_projection_unresolved")
        else:
            p = projection["projection"]
            tool_name, field = p["event_filter"]["tool_name"], p["value_path"][1]
            bindings = [{"json_pointer": f"/specs/{spec_position}/bindings/{i}", "value": deepcopy(b)}
                        for i, b in enumerate((previous or {}).get("bindings", []))
                        if b.get("tool") == tool_name and field in (b.get("params") or [])]
            if len(bindings) != 1 or (previous or {}).get("target_action") != tool_name:
                issues.append("historical_explicit_binding_not_unique_or_action_differs")
            tool = tools.get(tool_name)
            tool_position = next((i for i, t in enumerate(catalog["tools"]) if t["tool_name"] == tool_name), None)
            endpoints = [{"json_pointer": f"/tools/{tool_position}/observable_endpoints/{i}", "value": deepcopy(e)}
                         for i, e in enumerate((tool or {}).get("observable_endpoints", []))
                         if e.get("source_kind") == "tool_argument" and e.get("tool_name") == tool_name and e.get("source_path") == field]
            if len(endpoints) != 1:
                issues.append("input_endpoint_missing_or_ambiguous")
            else:
                endpoint = endpoints[0]["value"]
                current_schema = snapshot["tools"][tool_name]["parameters"]
                current_field = current_schema["properties"][field]
                if not (endpoint.get("type") == current_field.get("type") == assertion["predicate"]["type"]
                        and endpoint.get("required") is True and field in current_schema.get("required", [])
                        and assertion["field_presence_required"] is True):
                    issues.append("source_historical_and_local_presence_type_not_equal")
            if not issues:
                program = {"projection": deepcopy(p), "required": True,
                           "value_schema": {"type": assertion["predicate"]["type"]},
                           "evaluation_time": "assistant_argument_submission",
                           "tool_result_is_gate": False,
                           "no_matching_event": "not_observed",
                           "reported_scope": "selected_field_presence_and_type_in_supplied_trace"}
        rows.append({"branch_id": source["branch_id"], "assertion_id": assertion["assertion_id"],
                     "source_context": deepcopy(context),
                     "historical_core_rule_equal": core_equal,
                     "historical_bindings": bindings, "historical_input_endpoints": endpoints,
                     "historical_source_evidence": deepcopy((previous or {}).get("evidence", [])),
                     "grounding_limit": "historical_explicit_binding_rechecked_against_current_rule_and_local_schema_not_original_extraction_provenance_proof",
                     "program": program, "input_check_ready": program is not None, "issues": issues,
                     "gwt_scope_equivalence": "not_assessed", "gwt_oracle_ready": False,
                     "task_completion_policy": "not_selected"})
    return seal({"schema_version": SCHEMA,
                 "source_preparation_fingerprint": projections["preparation_fingerprint"],
                 "legacy_specs_fingerprint": content_sha256(legacy_specs),
                 "legacy_catalog_fingerprint": content_sha256(catalog),
                 "checks": rows, "uncompiled_branches": deepcopy(projections["uncompiled_branches"]),
                 "summary": {"selected_assertion_count": len(rows), "input_checks_ready": sum(r["input_check_ready"] for r in rows),
                             "gwt_oracles_ready": 0, "external_llm_calls": 0, "target_agent_calls": 0}}, "input_check_set_fingerprint")


def validate_input_checks(result, *sources):
    if result != compile_input_checks(*sources):
        raise ValueError("input checks differ from source reconstruction")


def evaluate_input_check(result, assertion_id, events, *, sources):
    """Read-only evaluation of supplied evidence, never dispatching a tool or Agent.

    Rebuild against the supplied verified source objects before using a program.
    The caller must load trusted frozen sources, as the production CLI does.
    """
    validate_input_checks(result, *sources)
    row = index(result["checks"], "assertion_id").get(assertion_id)
    if row is None or not row["input_check_ready"]:
        raise ValueError("no ready input check for this assertion")
    program = row["program"]
    extracted = extract_projection(events, program["projection"])
    observations = []
    validator = Draft202012Validator(program["value_schema"])
    for item in extracted["observations"]:
        check = deepcopy(item)
        if item["status"] == "malformed_arguments":
            verdict, reason = "evidence_error", "arguments_not_normalized_object"
        elif item["status"] == "field_missing":
            verdict, reason = "fail", "required_field_missing"
        else:
            try:
                # NaN/Infinity and Python-only objects are not JSON evidence.
                if json.loads(json.dumps(item["value"], allow_nan=False)) != item["value"]:
                    raise ValueError("value changes under JSON round trip")
            except (TypeError, ValueError):
                verdict, reason = "evidence_error", "value_not_json_compatible"
                check.pop("value", None)
                check["value_omitted"] = "non_json_compatible_evidence"
            else:
                passed = validator.is_valid(item["value"])
                verdict, reason = ("pass", "type_matches") if passed else ("fail", "type_mismatch")
        check.update(input_verdict=verdict, reason=reason)
        observations.append(check)
    counts = {v: sum(r["input_verdict"] == v for r in observations) for v in ("pass", "fail", "evidence_error")}
    # A witnessed mismatch remains visible alongside incomplete/malformed evidence.
    verdict = "fail" if counts["fail"] else "evidence_error" if counts["evidence_error"] else "pass" if observations else "not_observed"
    return {"assertion_id": assertion_id, "input_check_set_fingerprint": result["input_check_set_fingerprint"],
            "input_conformance_verdict": verdict, "counts": counts, "observations": observations,
            "verdict_scope": program["reported_scope"], "evidence_complete": "not_assessed",
            "gwt_coverage": "not_assessed", "gwt_verdict": None, "task_completion_verdict": None}
