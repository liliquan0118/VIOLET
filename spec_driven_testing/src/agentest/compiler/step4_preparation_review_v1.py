"""Integrate saved source choices into a Step 4 preparation review, offline.

This is not a test contract, a new evaluator, or a runtime readiness gate.
Unresolved relation semantics remain visible instead of being guessed.
"""

from collections import Counter
from copy import deepcopy

from .artifacts import content_sha256
from .step4_evidence_extensions_v1 import sealed, validate_source_selection


# These describe realization work, not missing meanings in the Step 4 contract.
LATER_WORK = {
    "runtime_window_binding": "steps7_8_concrete_interaction_binding_and_capture",
    "evidence_packet_realization": "step9_evidence_extraction",
    "runtime_dispatch": "step9_evaluator_dispatch",
}


def _check_seal(document, field):
    payload = deepcopy(document)
    if payload.pop(field, None) != content_sha256(payload):
        raise ValueError("changed artifact: " + field)


def _unique(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError("duplicate identity: " + key)
        result[row[key]] = row
    return result


def _selected_evidence(packet, checked):
    selected = set(checked["selection"]["selected_source_ids"])
    primary = packet["model_input"]["primary_records"]
    groups = []
    for group in packet["model_input"]["candidate_sources"]:
        if group["candidate_id"] not in selected:
            continue
        candidates = []
        for overlap in group["possible_key_overlaps"]:
            # Keep schema alternatives separate, even at the same record path.
            for variant_index, record in enumerate(primary):
                if record["path"] != overlap["primary_path"]:
                    continue
                names = [name for name in overlap["shared_names"]
                         if name in record["fields"] and name in group["fields"]
                         and (set(record["fields"][name]["types"]) - {None, "null"})
                         & (set(group["fields"][name]["types"]) - {None, "null"})]
                if not names:
                    continue
                candidates.append({
                    "primary_variant_index": variant_index,
                    "primary_path": deepcopy(record["path"]),
                    "candidate_key_pairs": [[name, name] for name in names],
                    "basis": "schema_name_overlap_only_not_identity_proof",
                    "key_approved": False,
                    "fields_requiring_presence_checks": [
                        {"side": side, "field": name}
                        for side, fields in (("primary", record["fields"]), ("supplement", group["fields"]))
                        for name in names if not fields[name]["required"] or "null" in fields[name]["types"]
                    ],
                })
        groups.append({"fact_group_id": group["candidate_id"],
                       "fields": deepcopy(group["fields"]),
                       "source_alternatives": deepcopy(group["available_from"]),
                       "join_candidates": candidates})
    return {
        "source_packet_fingerprint": packet["fingerprint"],
        "selection": deepcopy(checked["selection"]),
        "groups": groups,
        "primary_schema_unknowns": deepcopy(packet["model_input"]["primary_schema_unknowns"]),
        "join_verified": False, "predicate_compiled": False,
        "join_execution_requirements": [
            "approved_identity_mapping", "explicit_subject_and_source_event_scope",
            "nonmissing_keys_with_compatible_value_types", "unique_match_per_subject",
            "source_refs_for_each_fact", "preserve_subject_order_without_overwriting_values",
        ],
        "comparison_contract_required": [
            "operands_and_record_pairing", "comparison_operator_and_equality_boundary",
            "value_normalization_with_source_basis", "scope_and_quantifier",
        ],
        "on_missing_contract_or_evidence": "insufficient_not_pass_or_fail",
    }


def review_preparation(baseline, extensions, selections):
    """Account for every requirement without upgrading a source choice to a rule."""
    _check_seal(extensions, "extension_set_fingerprint")
    evaluators = baseline["evaluators"]
    _check_seal(evaluators, "evaluator_contract_set_fingerprint")
    if baseline["report"]["output_fingerprints"]["evaluators"] != evaluators["evaluator_contract_set_fingerprint"]:
        raise ValueError("stale evaluator report")
    if (extensions["source_evaluator_fingerprint"] != evaluators["evaluator_contract_set_fingerprint"]
            or extensions["source_accepted_fingerprint"] != evaluators["source_accepted_set_fingerprint"]
            or extensions["source_agent_schema_fingerprint"] != baseline["report"]["support_fingerprints"]["agent_spec"]):
        raise ValueError("stale extension inputs")
    for name, field in (("bindings", "binding_set_fingerprint"), ("semantic", "semantic_judge_contract_set_fingerprint")):
        _check_seal(baseline[name], field)
        if baseline["report"]["output_fingerprints"][name] != baseline[name][field]:
            raise ValueError("stale baseline report")
    contracts = _unique(extensions["contracts"], "requirement_id")
    packets = _unique(extensions["selection_questions"], "fingerprint")
    checked = {}
    for selection in selections:
        fp = selection["source_packet_fingerprint"]
        if fp not in packets or fp in checked:
            raise ValueError("unknown or duplicate selection packet")
        expected = validate_source_selection(packets[fp], selection["selection"])
        if expected != selection:
            raise ValueError("selection metadata does not match its validated answer")
        checked[fp] = expected
    questions = _unique(packets.values(), "requirement_id")
    rows = _unique(baseline["report"]["rows"], "requirement_id")
    evaluator_rows = _unique((r for b in evaluators["branches"] for r in b["evaluator_contracts"]), "requirement_id")
    if set(rows) != set(evaluator_rows) or not set(contracts) <= set(rows) or not set(questions) <= set(contracts):
        raise ValueError("requirement set mismatch")
    result = []
    for rid, source_row in rows.items():
        row = deepcopy(source_row)
        row.update(step4_open_items=[], later_step_dependencies=[], supplemental_contract=None, runtime_validated=False)
        extension = contracts.get(rid)
        if extension:
            _check_seal(extension, "fingerprint")
            if (extension["source_evaluator_fingerprint"] != evaluator_rows[rid]["evaluator_contract_fingerprint"]
                    or extension["source_requirement_fingerprint"] != evaluator_rows[rid]["source_requirement_fingerprint"]
                    or extension["branch_id"] != row["branch_id"]):
                raise ValueError("extension requirement mismatch")
            if evaluator_rows[rid]["evaluator_status"] == "executable":
                raise ValueError("extension cannot replace existing executable evaluator")
            row["supplemental_contract"] = deepcopy(extension)
            row["step4_open_items"] = deepcopy(extension.get("uncompiled", []))
            row["preparation_status"] = extension["status"]
            packet = questions.get(rid)
            selection = checked.get(packet["fingerprint"]) if packet else None
            if selection:
                row["selected_evidence"] = _selected_evidence(packet, selection)
                decision = selection["selection"]["decision"]
                row["preparation_status"] = "source_selected_rule_pending" if decision == "selected" else "source_selection_" + decision
                if decision == "selected":
                    row["step4_open_items"] = [x for x in row["step4_open_items"] if x != "source_selection"]
                # Keep the original extension under supplemental_contract as
                # provenance; the row status is the current preparation status.
        row["later_step_dependencies"] = [
            {"item": item, "owner": LATER_WORK[item]}
            for item in row["step4_open_items"] if item in LATER_WORK
        ]
        row["step4_open_items"] = [item for item in row["step4_open_items"] if item not in LATER_WORK]
        result.append(row)
    return sealed({
        "schema_version": "agentspectesting.step4-preparation-review/v0.1",
        "source_baseline_fingerprint": content_sha256(baseline),
        "source_extension_set_fingerprint": extensions["extension_set_fingerprint"],
        "source_selection_fingerprints": [content_sha256(s) for s in selections],
        "rows": result,
        "legacy_runtime_preconditions": deepcopy(baseline["report"]["runtime_preconditions"]),
        "summary": {"requirement_count": len(result), "branch_count": len({r["branch_id"] for r in result}),
                    "status_counts": dict(Counter(r["preparation_status"] for r in result)),
                    "saved_selections_consumed": len(checked), "unanswered_source_questions": len(packets) - len(checked),
                    "new_executable_evaluators": 0, "external_calls": 0, "target_agent_calls": 0},
        "runtime_dispatch_enabled": False, "whole_step4_complete": False,
        "scope": "Step 4 preparation review only; not Step 5 test contract or runtime validation",
    }, "review_fingerprint")
