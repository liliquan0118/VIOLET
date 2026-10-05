"""Compile a bounded, whole-sentence source grammar into symbolic field-type assertions.

Unrecognized text is deferred, never partially interpreted using a type substring.
This compiles a value requirement, not a tool binding, reachability check or task goal.
"""

from copy import deepcopy
import re

from .v5_oracle_review_contract_v1 import validate_contracts
from .v5_step3_intake_v1 import seal


SCHEMA = "agentspectesting.v5-source-assertion-set/v0.1"
GRAMMAR = "required-field-json-type/v0.1"
_PATTERN = re.compile(
    r"The agent must provide (?:a|an) (?P<field>[A-Za-z_][A-Za-z0-9_]*) "
    r"that is (?:a|an) (?P<type>string|number|integer|boolean|array|object)"
    r"(?P<example> such as (?:'[^'\n]*'|\"[^\"\n]*\"|true|false|-?\d+(?:\.\d+)?))?"
    r" in (?P<quantifier>each(?: single)?|every) (?P<scope>[A-Za-z][A-Za-z0-9_-]*(?: [A-Za-z][A-Za-z0-9_-]*)*)\.",
    re.IGNORECASE,
)
# This grammar accepts a simple noun phrase, not an arbitrary condition or relation.
_SCOPE_OPERATORS = frozenset("if unless except and or before after when where that which whose with without provided only not".split())


def parse_type_requirement(text):
    if not isinstance(text, str):
        raise ValueError("Then must be text")
    match = _PATTERN.fullmatch(text)
    if match is None:
        return None
    if set(match["scope"].lower().split()) & _SCOPE_OPERATORS:
        return None
    spans = {name: {"start": match.start(name), "end": match.end(name), "text": match[name]}
             for name in ("field", "type", "quantifier", "scope")}
    if match["example"]:
        spans["example"] = {"start": match.start("example"), "end": match.end("example"), "text": match["example"]}
    return {"grammar_id": GRAMMAR, "source_spans": spans,
            "field": match["field"], "json_type": match["type"].lower(),
            "scope_text": match["scope"], "example_is_constraint": False}


def compile_assertions(contracts, evidence):
    validate_contracts(contracts, evidence)
    rows = []
    for row in contracts["branches"]:
        context = row["spec_requirement"]["source_context"]
        text = context["gwt"]["then"]
        parsed = parse_type_requirement(text)
        assertions = []
        diagnostics = []
        if parsed is None:
            diagnostics.append({"code": "source_sentence_outside_supported_grammar",
                                "meaning": "requires another compiler or semantic interpretation; not untestable"})
        else:
            hints = []
            for c in row["observation_review"]["candidates"]:
                p = c["proposal"]
                observation = p.get("observation_contract") or {}
                if p["candidate_kind"] == "tool_argument" and observation.get("parameter") == parsed["field"]:
                    endpoint_type = (observation.get("endpoint") or {}).get("type")
                    hints.append({"candidate_id": c["candidate_id"],
                                  "tool_name": observation.get("tool_name"),
                                  "historical_endpoint_type": endpoint_type,
                                  "type_comparison": "unavailable" if endpoint_type is None else "equal" if endpoint_type == parsed["json_type"] else "conflict",
                                  "binding_authorized": False})
            assertions.append({
                "assertion_id": row["branch_id"] + "::SA01",
                "status": "symbolic_value_requirement_compiled",
                "grammar_id": parsed["grammar_id"],
                "normative_mode": "required",
                "subject": {"kind": "source_named_field", "name": parsed["field"]},
                "predicate": {"predicate_kind": "json_type", "type": parsed["json_type"]},
                "field_presence_required": True,
                "scope": {"quantifier": "each_source_scope_instance", "source_scope_text": parsed["scope_text"],
                          "source_given": context["gwt"]["given"], "source_when": context["gwt"]["when"],
                          "event_selector": None, "status": "requires_binding"},
                "source": {"then": text, "origin": context["origin"],
                           "source_contract_fingerprint": row["contract_fingerprint"],
                           "spans": parsed["source_spans"]},
                "illustrative_example": {"source_text": parsed["source_spans"].get("example", {}).get("text"),
                                         "compiled_as_equality_or_pattern": False},
                "observation_mapping_hints": hints,
                "no_matching_event": "applicability_undetermined_not_a_field_type_verdict",
                "runtime_ready": False,
            })
            diagnostics.append({"code": "scope_and_observation_binding_pending"})
            if any(h["type_comparison"] == "conflict" for h in hints):
                diagnostics.append({"code": "historical_endpoint_type_conflict", "meaning": "do not override source type or authorize this mapping"})
        rows.append(seal({"branch_id": row["branch_id"], "source_contract_fingerprint": row["contract_fingerprint"],
                          "source_context": deepcopy(context), "assertions": assertions,
                          "status": "symbolic_compiled_binding_pending" if assertions else "needs_semantic_compilation",
                          "diagnostics": diagnostics,
                          "task_completion": deepcopy(row["task_completion"]),
                          "reachability_contract": None, "runtime_lowering_allowed": False}, "assertion_branch_fingerprint"))
    return seal({"schema_version": SCHEMA, "grammar_id": GRAMMAR,
                 "source_contract_set_fingerprint": contracts["contract_set_fingerprint"],
                 "branches": rows,
                 "summary": {"branch_count": len(rows), "symbolic_assertion_count": sum(len(r["assertions"]) for r in rows),
                             "branches_needing_semantic_compilation": sum(not r["assertions"] for r in rows),
                             "runtime_ready": 0, "external_llm_calls": 0, "target_agent_calls": 0}},
                "assertion_set_fingerprint")


def validate_assertions(result, contracts, evidence):
    if result != compile_assertions(contracts, evidence):
        raise ValueError("assertions differ from source-derived compilation")
    return result
