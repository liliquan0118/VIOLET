"""Audit legacy references and prepare a bounded Then-membership pilot."""

from collections import Counter
from copy import deepcopy
import json

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import verify_fingerprint
from .v5_step3_intake_v1 import SCHEMA as INTAKE_SCHEMA, index, seal
from .oracle_requirement_pipeline_v7 import PACKET_SET_VERSION, PACKET_VERSION, TASK_NAME
from .oracle_decision_pipeline_v8 import build_oracle_decision_packets


SCHEMA = "agentspectesting.v5-step3-mapping-audit/v0.1"


def tool_index(catalog):
    tools = index(catalog["tools"], "tool_name")
    for tool in tools.values():
        index(tool["observable_endpoints"], "endpoint_id")
        args = [e["source_path"] for e in tool["observable_endpoints"] if e["source_kind"] == "tool_argument"]
        if len(args) != len(set(args)):
            raise ValueError("duplicate tool argument paths")
        if any(e.get("tool_name") != tool["tool_name"] for e in tool["observable_endpoints"]):
            raise ValueError("catalog endpoint/tool identity mismatch")
    return tools


def check_reference(tool_name, parameters, tools):
    if not isinstance(tool_name, str) or not tool_name:
        return {"status": "no_direct_tool_reference", "tool_name": tool_name, "parameters": parameters,
                "meaning": "other observation evidence may apply; not rejected or verified here"}
    tool = tools.get(tool_name)
    result = {"tool_name": tool_name, "parameters": deepcopy(parameters), "tool_description": None,
              "parameter_endpoints": [], "missing_parameters": []}
    if tool is None:
        return {**result, "status": "tool_not_in_frozen_catalog"}
    result["tool_description"] = tool["description"]
    endpoints = {e["source_path"]: e for e in tool["observable_endpoints"] if e["source_kind"] == "tool_argument"}
    for param in parameters:
        if param in endpoints:
            result["parameter_endpoints"].append(deepcopy(endpoints[param]))
        else:
            result["missing_parameters"].append(param)
    return {**result, "status": "parameter_not_in_frozen_catalog" if result["missing_parameters"] else "references_resolved_in_frozen_catalog"}


def requirement_parameters(contract):
    items = [*contract.get("scope_parameters", []), *contract.get("parameters", [])]
    if contract.get("parameter") is not None:
        items.append(contract["parameter"])
    if any(not isinstance(p, str) or not p for p in items):
        raise ValueError("invalid requirement parameter reference")
    return list(dict.fromkeys(items))


def audit_mapping(intake, catalog):
    verify_fingerprint(intake, "step3_intake_fingerprint")
    if intake.get("schema_version") != INTAKE_SCHEMA:
        raise ValueError("expected Step 3A intake")
    index(intake["branches"], "branch_id")
    tools = tool_index(catalog)
    rows = []
    for source in intake["branches"]:
        verify_fingerprint(source, "step3_input_fingerprint")
        hint = source["mapping_inventory"]["legacy_same_spec_hint"] or {}
        checks = []
        for position, binding in enumerate(hint.get("bindings") or []):
            params = binding.get("params") or []
            if not isinstance(params, list) or any(not isinstance(p, str) or not p for p in params):
                raise ValueError("invalid legacy mapping parameter list")
            checks.append({"source_binding_index": position, **check_reference(binding.get("tool"), params, tools)})
        li = source["legacy_inventory"]
        unchanged = li["changed_fields"] == []
        requirements = []
        for r in (li["accepted_branch"] or {}).get("requirements", []):
            c = r["observation_contract"]
            reference = check_reference(c.get("tool_name"), requirement_parameters(c), tools)
            requirements.append({"requirement_id": r["requirement_id"], "legacy_requirement_fingerprint": r["requirement_fingerprint"],
                                 "requirement_type": r["requirement_type"], "reference_check": reference,
                                 "source_context_status": "seven_fields_unchanged" if unchanged else "source_context_changed",
                                 "applicability": "not_decided", "reuse_authorized": False})
        rows.append(seal({"branch_id": source["branch_id"], "source_step3_input_fingerprint": source["step3_input_fingerprint"],
                          "legacy_binding_checks": checks, "legacy_requirement_checks": requirements,
                          "source_context_status": "seven_fields_unchanged" if unchanged else "no_same_id_legacy_branch" if li["changed_fields"] is None else "source_context_changed",
                          "mapping_semantic_applicability": "not_decided", "new_oracle_requirements": [], "new_expectations": []}, "mapping_audit_fingerprint"))
    checks = [c for r in rows for c in r["legacy_binding_checks"]]
    reqs = [c for r in rows for c in r["legacy_requirement_checks"]]
    return seal({"schema_version": SCHEMA, "source_step3_intake_fingerprint": intake["step3_intake_fingerprint"],
                 "frozen_tool_catalog_fingerprint": content_sha256(catalog), "branches": rows,
                 "summary": {"branch_count": len(rows), "legacy_binding_count": len(checks),
                             "binding_reference_status_counts": dict(Counter(c["status"] for c in checks)),
                             "branches_with_all_supplied_bindings_resolved": sum(bool(r["legacy_binding_checks"]) and all(c["status"] == "references_resolved_in_frozen_catalog" for c in r["legacy_binding_checks"]) for r in rows),
                             "branches_without_explicit_legacy_bindings": sum(not r["legacy_binding_checks"] for r in rows),
                             "source_context_status_counts": dict(Counter(r["source_context_status"] for r in rows)),
                             "legacy_requirement_count_in_scope": len(reqs),
                             "requirement_reference_status_counts": dict(Counter(r["reference_check"]["status"] for r in reqs)),
                             "semantic_applicability_decisions": 0, "oracle_reuse_authorized": 0,
                             "runtime_tool_schema_verified": False, "external_llm_calls": 0, "target_agent_calls": 0},
                 "scope": "historical frozen catalog reference checks, not current runtime parity, Then applicability, or Oracle correctness"}, "mapping_audit_set_fingerprint")


def prepare_membership_pilot(intake, audit):
    verify_fingerprint(intake, "step3_intake_fingerprint")
    verify_fingerprint(audit, "mapping_audit_set_fingerprint")
    if audit["source_step3_intake_fingerprint"] != intake["step3_intake_fingerprint"]:
        raise ValueError("audit/intake mismatch")
    audits = index(audit["branches"], "branch_id")
    # Deliberately bounded pilot covering three distinct existing observation
    # forms. Selection is deterministic and not a claim of representativeness.
    chosen = {}
    for b in intake["branches"]:
        context = b["then_input"]["source_context"]
        for r in (b["legacy_inventory"]["accepted_branch"] or {}).get("requirements", []):
            checked = next(c for c in audits[b["branch_id"]]["legacy_requirement_checks"] if c["requirement_id"] == r["requirement_id"])
            kind = r["requirement_type"]
            category = "argument" if kind == "tool_argument" else "ordering" if kind == "temporal_relation" else "permitted_action" if kind == "tool_call" and context["gwt"]["then"].startswith("The agent may ") else None
            if category is None or category in chosen:
                continue
            if checked["reference_check"]["status"] not in ("references_resolved_in_frozen_catalog", "no_direct_tool_reference"):
                continue
            candidate = {"candidate_id": r["requirement_id"], "candidate_kind": kind,
                         "requirement_text": r["requirement_text"], "observation_contract": deepcopy(r["observation_contract"]),
                         "source_refs": {"legacy_requirement_fingerprint": r["requirement_fingerprint"],
                                         "step3_input_fingerprint": b["step3_input_fingerprint"]}}
            candidate["observation_contract"].setdefault("action_description", checked["reference_check"].get("tool_description"))
            packet = {"schema_version": PACKET_VERSION, "task_name": TASK_NAME,
                      "branch_id": b["branch_id"], "candidate_id": r["requirement_id"],
                      "task_input": {"branch_context": deepcopy(context), "observable_channels": [], "proposed_observation": candidate}}
            chosen[category] = seal(packet, "packet_fingerprint")
    if set(chosen) != {"argument", "ordering", "permitted_action"}:
        raise ValueError("pilot diversity cannot be met; do not silently substitute")
    packets = list(chosen.values())
    candidates = seal({"schema_version": PACKET_SET_VERSION, "selected_branch_ids": [p["branch_id"] for p in packets],
                       "packets": packets, "summary": {"expected_model_calls": len(packets)}}, "packet_set_fingerprint")
    pilot = build_oracle_decision_packets(candidates)
    return candidates, pilot


def render_mapping(audit):
    lines = ["# Step 3B: legacy mapping reference check", "", "This only checks against the frozen tool catalog; it does not mean the mapping semantics hold or that the current runtime schema matches.", "",
             "```json", json.dumps(audit["summary"], ensure_ascii=False, indent=2), "```", "",
             "| Branch | Legacy binding reference checks | Legacy requirement check count |", "| --- | --- | --- |"]
    for b in audit["branches"]:
        checks = b["legacy_binding_checks"]
        desc = "; ".join(str(c["tool_name"]) + ": " + c["status"] for c in checks) or "no explicit bindings; other observations remain possible"
        lines.append(f"| {b['branch_id']} | {desc} | {len(b['legacy_requirement_checks'])} |")
    return "\n".join(lines) + "\n"
