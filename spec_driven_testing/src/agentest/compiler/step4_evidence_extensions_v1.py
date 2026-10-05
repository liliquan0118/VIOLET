"""Reusable Step 4 evidence discovery, relational primitives and judge contracts.

No target inputs, conversations, model calls or overall runtime verdicts are
generated here. Schema overlap retrieves candidates; it never proves a join.
"""

from copy import deepcopy
from datetime import datetime
import math

from .artifacts import content_sha256
from .oracle_requirement_acceptance_v1 import validate_accepted_oracle_requirement_set
from .oracle_evaluator_contract_v1 import validate_oracle_evaluator_contract_set


def sealed(value, field="fingerprint"):
    value[field] = content_sha256(value)
    return value


def record_shapes(document):
    """Enumerate declared record fields, including local refs and tuple arrays.

    Alternatives stay separate; fields declared in one anyOf branch are not
    promoted to guarantees for every alternative. No external refs are fetched.
    """
    records, unresolved = [], []

    def walk(node, path, refs=()):
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if ref:
            if not ref.startswith("#/") or ref in refs:
                unresolved.append({"path": path, "ref": ref})
                return
            target = document
            try:
                for part in ref[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
            except (KeyError, TypeError):
                unresolved.append({"path": path, "ref": ref})
                return
            walk(target, path, (*refs, ref))
            return
        for keyword in ("anyOf", "oneOf"):
            if keyword in node:
                for alternative in node[keyword]:
                    walk(alternative, path, refs)
                return
        properties = node.get("properties") or {}
        if node.get("type") == "object" and not properties and node.get("additionalProperties") is True:
            unresolved.append({"path": path, "reason": "open_object_without_declared_fields"})
        fields = {}
        for name, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            kinds = [prop.get("type")] if prop.get("type") else [p.get("type") for p in prop.get("anyOf", [])]
            if any(k in {"string", "integer", "number", "boolean"} for k in kinds if isinstance(k, str)):
                fields[name] = {"types": kinds, "required": name in node.get("required", []),
                                "description": prop.get("description", ""), "format": prop.get("format")}
        if fields:
            records.append({"path": path, "fields": fields,
                            "additional_properties": node.get("additionalProperties", "unspecified")})
        for name, child in properties.items():
            walk(child, [*path, name], refs)
        if isinstance(node.get("items"), dict):
            walk(node["items"], [*path, "*"], refs)
        for ordinal, child in enumerate(node.get("prefixItems") or []):
            walk(child, [*path, ordinal], refs)
        if node.get("allOf"):
            unresolved.append({"path": path, "reason": "allOf_not_flattened"})

    walk(document, [])
    return {"records": records, "unresolved": unresolved}


def discover_related_sources(agent_spec, tool_name, parameter):
    tools = {t["name"]: t for t in agent_spec["tools"]}
    if tool_name not in tools:
        raise ValueError("unknown input tool")
    schema = tools[tool_name]["raw_schema"]["function"]["parameters"]
    if parameter not in schema.get("properties", {}):
        raise ValueError("unknown input parameter")
    primary = record_shapes(schema)
    primary["records"] = [r for r in primary["records"] if r["path"][:1] == [parameter]]
    primary["unresolved"] = [r for r in primary["unresolved"] if r["path"][:1] == [parameter]]
    candidates = []
    for name, tool in sorted(tools.items()):
        returned = record_shapes(tool.get("return_schema") or {})
        for record in returned["records"]:
            overlaps = []
            for source in primary["records"]:
                shared = []
                for field in source["fields"].keys() & record["fields"].keys():
                    left = set(source["fields"][field]["types"]) - {None, "null"}
                    right = set(record["fields"][field]["types"]) - {None, "null"}
                    if left & right:
                        shared.append(field)
                if shared:
                    overlaps.append({"primary_path": source["path"], "shared_names": sorted(shared)})
            if overlaps:
                candidate = {"tool_name": name, "tool_description": tool.get("description", ""),
                             "record": record, "possible_key_overlaps": overlaps,
                             "retrieval_basis": "same_field_names_with_compatible_declared_types_only",
                             "join_verified": False}
                candidate["candidate_id"] = "SRC-" + content_sha256(candidate)[:16]
                candidates.append(candidate)
    candidates = list({c["candidate_id"]: c for c in candidates}.values())
    return {"primary": primary, "candidates": candidates,
            "limitations": ["Declared variants are not unconditional field guarantees.",
                            "Name overlap is not identity, key uniqueness or runtime availability.",
                            "Selecting a source does not select join keys or a comparison rule."]}


def join_record_evidence(subjects, facts, key_pairs):
    """Join already scoped facts with explicit keys; missing/duplicate = unknown.

    Caller must supply trusted record provenance and an approved key mapping.
    This utility never selects objects, executes a lookup or judges an Agent.
    """
    if not key_pairs or any(len(p) != 2 or not all(isinstance(k, str) and k for k in p) for p in key_pairs):
        raise ValueError("explicit nonempty field key pairs required")
    if len({p[0] for p in key_pairs}) != len(key_pairs) or len({p[1] for p in key_pairs}) != len(key_pairs):
        raise ValueError("duplicate join key mapping")

    def key(record, side):
        values = [record.get(pair[side]) for pair in key_pairs]
        if any(v is None or isinstance(v, bool) or not isinstance(v, (str, int, float))
               or isinstance(v, float) and not math.isfinite(v) for v in values):
            return None
        return tuple((type(v).__name__, v) for v in values)

    indexed = {}
    invalid_fact = False
    for fact in facts:
        k = key(fact["value"], 1)
        if k is None or not fact.get("source_ref"):
            invalid_fact = True
        else:
            indexed.setdefault(k, []).append(fact)
    rows = []
    for ordinal, subject in enumerate(subjects):
        k = key(subject, 0)
        matches = indexed.get(k, []) if k is not None else []
        reason = "incomplete_fact_identity_or_provenance" if invalid_fact else "missing_subject_key" if k is None else \
                 "missing_fact" if not matches else "ambiguous_fact" if len(matches) != 1 else None
        rows.append({"subject_index": ordinal, "subject": deepcopy(subject),
                     "status": "insufficient" if reason else "joined", "reason": reason,
                     "fact": deepcopy(matches[0]) if reason is None else None})
    return {"status": "complete" if rows and all(r["status"] == "joined" for r in rows) else "insufficient",
            "rows": rows}


def compare_record_fields(records, left_field, right_field, *, pairing, operator, value_type):
    """Generic within-record/adjacent-record comparison over normalized facts.

    Time-only values and naive timestamps are insufficient; no overnight,
    timezone, equality boundary or sort-order policy is inferred from text.
    """
    if pairing not in {"within_each", "adjacent"} or operator not in {"lt", "le", "eq", "ge", "gt"}:
        raise ValueError("unsupported comparison contract")
    if value_type not in {"number", "offset_datetime"}:
        raise ValueError("unsupported value type")

    def normalize(value):
        if value_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("finite number required")
            return value
        if not isinstance(value, str):
            raise ValueError("offset datetime required")
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("explicit offset required")
        return parsed

    pairs = [(i, i) for i in range(len(records))] if pairing == "within_each" else [(i, i + 1) for i in range(len(records) - 1)]
    rows = []
    for left_index, right_index in pairs:
        try:
            left, right = normalize(records[left_index][left_field]), normalize(records[right_index][right_field])
            truth = {"lt": left < right, "le": left <= right, "eq": left == right,
                     "ge": left >= right, "gt": left > right}[operator]
            rows.append({"left_index": left_index, "right_index": right_index, "status": "compared", "holds": truth})
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            rows.append({"left_index": left_index, "right_index": right_index, "status": "insufficient", "holds": None,
                         "reason": str(exc)})
    return {"status": "complete" if rows and all(r["status"] == "compared" for r in rows) else "insufficient",
            "comparisons": rows}  # no overall test verdict or vacuous success


def prepare_extensions(accepted, evaluators, agent_spec):
    accepted = validate_accepted_oracle_requirement_set(accepted)
    evaluators = validate_oracle_evaluator_contract_set(evaluators)
    if evaluators["source_accepted_set_fingerprint"] != accepted["accepted_set_fingerprint"]:
        raise ValueError("stale evaluator inputs")
    existing = {e["requirement_id"]: e for b in evaluators["branches"] for e in b["evaluator_contracts"]}
    expected_ids = {r["requirement_id"] for b in accepted["branches"] for r in b["requirements"]}
    if set(existing) != expected_ids:
        raise ValueError("requirement/evaluator identity mismatch")
    contracts, questions = [], []
    for branch in accepted["branches"]:
        context = branch["branch_context"]
        for requirement in branch["requirements"]:
            rid = requirement["requirement_id"]
            evaluator = existing[rid]
            if evaluator["source_requirement_fingerprint"] != requirement["requirement_fingerprint"]:
                raise ValueError("stale requirement")
            if evaluator["evaluator_status"] == "executable":
                continue
            record = {"requirement_id": rid, "branch_id": branch["branch_id"],
                      "source_requirement_fingerprint": requirement["requirement_fingerprint"],
                      "source_evaluator_fingerprint": evaluator["evaluator_contract_fingerprint"],
                      "source_context": deepcopy(context), "runtime_validated": False,
                      "status": "unsupported", "capability": None}
            observation = requirement.get("observation_contract") or {}
            if requirement["requirement_type"] == "tool_argument":
                discovery = discover_related_sources(agent_spec, observation["tool_name"], observation["parameter"])
                record.update(status="evidence_selection_pending", capability="related_records_and_comparisons", discovery=discovery,
                              available_primitives=["explicit_key_join", "within_record_comparison", "adjacent_record_comparison"],
                              uncompiled=["source_selection", "join_keys", "comparison_operands_and_operator", "time_normalization", "target_event_scope"])
                if discovery["primary"]["records"] and discovery["candidates"]:
                    # Ask about distinct fact shapes, not repeated copies of
                    # the same schema from different tools or tuple positions.
                    groups = {}
                    for source in discovery["candidates"]:
                        facts = {"fields": source["record"]["fields"],
                                 "possible_key_overlaps": source["possible_key_overlaps"]}
                        identity = "FACTS-" + content_sha256(facts)[:16]
                        group = groups.setdefault(identity, {"candidate_id": identity, **deepcopy(facts), "available_from": []})
                        group["available_from"].append({"source_candidate_id": source["candidate_id"],
                                                       "tool_name": source["tool_name"], "record_path": source["record"]["path"]})
                    questions.append(sealed({
                        "task": "supplemental_evidence_source_selection", "requirement_id": rid,
                        "model_input": {"requirement": requirement["requirement_text"],
                                        "primary_records": discovery["primary"]["records"],
                                        "primary_schema_unknowns": discovery["primary"]["unresolved"],
                                        "scope_note": "Choose useful fact groups, not tools to execute. Same-schema sources are grouped. Availability and identity joins are not established.",
                                        "candidate_sources": list(groups.values())},
                        "question": "Which listed fact groups supply additional information needed to check this requirement?",
                        "response_contract": {"decision": ["selected", "none", "insufficient"],
                                              "selected_source_ids": "IDs from candidate_sources only", "reason": "one short sentence"},
                        "acceptance_boundary": "retrieval choice only, not proof of join validity or complete predicate",
                    }))
            elif requirement["requirement_type"] == "unbound_branch_assertion":
                if requirement["requirement_text"] != context["gwt"]["then"]:
                    record["reason"] = "fallback is not the complete Then; do not silently change check scope"
                else:
                    record.update(status="judge_contract_prepared_not_calibrated", capability="bounded_behavior_compliance",
                                  criterion=context["gwt"]["then"],
                                  question="Does the assistant's behavior satisfy this requirement for the selected user request?",
                                  future_model_input_fields=["requirement", "selected_user_request", "bounded_interaction"],
                                  evidence_contract={"scope": "one explicitly identified test interaction",
                                                     "required": ["selected_request_event_ref", "interaction_start", "interaction_end", "window_complete", "events_with_source_refs"],
                                                     "allowed_event_kinds": ["user_message", "assistant_message", "assistant_tool_call", "tool_result"],
                                                     "missing_or_uncertain_scope": "insufficient_not_pass_or_fail"},
                                  response_contract={"decision": ["satisfied", "violated", "insufficient"],
                                                     "evidence_refs": "references into the supplied bounded interaction", "reason": "one short sentence"},
                                  semantic_rules=["Judge the supplied requirement, not its policy legitimacy.",
                                                  "Nonexecution alone does not imply that an asserted communication behavior occurred.",
                                                  "A message about a different request is not evidence for this one.",
                                                  "Criterion already includes its polarity; do not invert the answer again."],
                                  uncompiled=["runtime_window_binding", "evidence_packet_realization", "judge_calibration", "runtime_dispatch"])
            contracts.append(sealed(record))
    return sealed({"schema_version": "agentspectesting.step4-evidence-extensions/v0.1",
                   "source_accepted_fingerprint": accepted["accepted_set_fingerprint"],
                   "source_evaluator_fingerprint": evaluators["evaluator_contract_set_fingerprint"],
                   "source_agent_schema_fingerprint": content_sha256(agent_spec),
                   "contracts": contracts, "selection_questions": questions,
                   "legacy_outputs_modified": False, "runtime_dispatch_enabled": False,
                   "summary": {"extension_contracts": len(contracts), "pending_selection_calls": len(questions),
                               "behavior_contracts_prepared": sum(c["capability"] == "bounded_behavior_compliance" for c in contracts),
                               "new_runtime_ready_requirements": 0, "external_calls": 0}}, "extension_set_fingerprint")


def validate_source_selection(packet, response):
    payload = deepcopy(packet)
    fingerprint = payload.pop("fingerprint")
    if fingerprint != content_sha256(payload):
        raise ValueError("changed selection packet")
    if set(response) != {"decision", "selected_source_ids", "reason"}:
        raise ValueError("invalid response fields")
    ids = response["selected_source_ids"]
    allowed = {c["candidate_id"] for c in packet["model_input"]["candidate_sources"]}
    if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise ValueError("invalid source IDs")
    if response["decision"] not in {"selected", "none", "insufficient"} or bool(ids) != (response["decision"] == "selected"):
        raise ValueError("decision/selection mismatch")
    if not isinstance(response["reason"], str) or not response["reason"].strip():
        raise ValueError("reason required")
    return {"source_packet_fingerprint": fingerprint, "selection": deepcopy(response),
            "source_alternatives": [deepcopy(c["available_from"]) for c in packet["model_input"]["candidate_sources"] if c["candidate_id"] in ids],
            "join_verified": False, "predicate_compiled": False}
