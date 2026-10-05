"""Ordinary creation-request identity binding; no transaction or scenario synthesis."""
from collections import Counter
from copy import deepcopy

from ..compiler.artifacts import content_sha256
from ..compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .v5_object_references_v1 import SnapshotObjects, inspect_candidates, seal
from .v5_role_handoff_v1 import check_target_origin, project_source_requirements


SCHEMA = "agentspectesting.v5-requester-binding-set/v0.1"
POLICY = {
    "version": "tau-airline-ordinary-creation-identity/v0.1",
    "selection": "first_supplied_pair_with_resolved_identity_and_verified_owner_relation",
    "candidate_pair_is_atomic": True,
    "global_database_fallback": False,
    "copy_context_transaction_preferences": False,
    "user_fact_fields": ["user_id"],
    "user_fact_timing": "on_agent_request_only",
}


def _index(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate branch")
    return result


def _scope_hold(contract, role):
    decision = role["lifecycle"]["decision"]
    if decision == "not_determined":
        return "request_lifecycle_not_determined"
    if decision != "create_new":
        return "existing_object_binding_not_implemented_in_this_component"
    given = contract["given_contract"]
    # Only a literal unconditional input is supported, never infer True from [] alone.
    if (given["text"] != "True" or given["logic"]["text"] != "True"
            or given["database_conditions"] or given["non_database_conditions"]):
        return "conditional_given_requires_separate_verification"
    fixture = contract["fixture_contract"]
    if fixture["root"] != "reservations" or fixture["lookup"]["lookup_status"] != "anchored":
        return "unsupported_context_role_mapping"
    assertions = fixture["relation_assertions"]
    if any(assertions[source] != {"present": True, "value": "owner"}
           for source in ("user_requirements", "condition_matches")):
        return "owner_context_basis_missing_or_conflicting"
    return None


def bind_creation_requesters(contracts, references, handoff, database):
    for value, field in ((contracts, "contract_set_fingerprint"),
                         (references, "reference_set_fingerprint"), (handoff, "handoff_fingerprint")):
        verify_fingerprint(value, field)
    if handoff["schema_version"] != "agentspectesting.v5-role-handoff/v0.1":
        raise ValueError("unsupported role handoff")
    if references["source_contract_set_fingerprint"] != contracts["contract_set_fingerprint"]:
        raise ValueError("reference/contract lineage mismatch")
    if handoff["source_reference_set_fingerprint"] != references["reference_set_fingerprint"]:
        raise ValueError("handoff/reference lineage mismatch")
    ref_index = _index(references["rows"], "branch_id")
    roles = _index(handoff["rows"], "branch_id")
    contract_index = _index(contracts["contracts"], "branch_id")
    if not (ref_index.keys() == roles.keys() == contract_index.keys()):
        raise ValueError("source branch membership mismatch")
    store = SnapshotObjects(database, references["database_fingerprint"])
    rows = []
    for bid, contract in contract_index.items():
        reference, role = ref_index[bid], roles[bid]
        verify_fingerprint(contract, "branch_test_contract_fingerprint")
        verify_fingerprint(reference, "reference_row_fingerprint")
        verify_fingerprint(role, "role_row_fingerprint")
        if (role["source_contract_fingerprint"] != contract["branch_test_contract_fingerprint"]
                or reference["source_contract_fingerprint"] != contract["branch_test_contract_fingerprint"]
                or role["source_reference_row_fingerprint"] != reference["reference_row_fingerprint"]):
            raise ValueError("branch lineage mismatch")
        count = contract["fixture_contract"]["lookup"]["n_matches"]
        if not (count == len(reference["candidates"]) == reference["summary"]["supplied_pairs"]
                == role["candidate_pool_reference"]["candidate_count"]):
            raise ValueError("candidate pool count differs across sources")
        source_view = project_source_requirements(role)
        if source_view != {"request": contract["user_input_contract"]["request_description"],
                           "when": contract["when_contract"]["spec_when"]}:
            raise ValueError("role handoff changed source requirements")
        row = {
            "branch_id": bid, "source_contract_fingerprint": contract["branch_test_contract_fingerprint"],
            "source_role_row_fingerprint": role["role_row_fingerprint"],
            "source_reference_row_fingerprint": reference["reference_row_fingerprint"],
            "database_fingerprint": store.fingerprint,
            "source_user_requirements": source_view,
            "full_user_requirement_contract": deepcopy(contract["user_input_contract"]),
            "given_contract": deepcopy(contract["given_contract"]),
            "given_verification": "not_performed", "when_realized": False,
            "status": "not_bound", "hold_reason": _scope_hold(contract, role),
            "selected_candidate_index": None, "selected_pair_fingerprint": None,
            "candidate_checks": [],
            "role_bindings": {"requester": None, "context_anchor": None, "operation_target": None},
            "target_origin_guard": None, "driver_bindings": {}, "user_fact_grants": [],
            "pending_driver_binding_names": deepcopy(role["pending_driver_binding_names"]),
            "request_facts_complete": False, "runtime_ready": False,
            "remaining_work": ["new_object_request_facts_and_preferences", "when_and_conversation_realization",
                               "full_fixture_and_runtime_readiness"],
        }
        if row["hold_reason"]:
            row["status"] = "deferred"
            row["remaining_work"] = [row["hold_reason"], "full_fixture_and_runtime_readiness"]
        else:
            guard = check_target_origin(role, "future_result")
            if guard["status"] != "compatible":
                raise ValueError("creation target origin is incompatible")
            row["target_origin_guard"] = guard
            row["given_verification"] = "literal_true_no_state_predicates"
            # Recompute handles and ownership; do not trust recorded status booleans.
            candidates = inspect_candidates(contract["fixture_contract"], store)
            if candidates != reference["candidates"]:
                raise ValueError("candidate references differ from current snapshot reconstruction")
            for candidate in candidates:
                eligible = (candidate["reference_status"] == "resolved"
                            and candidate["relation_observation"]["status"] == "matched")
                row["candidate_checks"].append({
                    "candidate_index": candidate["candidate_index"],
                    "candidate_fingerprint": candidate["source_candidate_fingerprint"],
                    "identity_and_owner_verified": eligible,
                    "reference_errors": deepcopy(candidate["reference_errors"]),
                    "relation_status": candidate["relation_observation"]["status"],
                    "selected": False})
                if not eligible or row["selected_candidate_index"] is not None:
                    continue
                requester = candidate["user_reference"]["handle"]
                context = candidate["root_reference"]["handle"]
                user = store.read(requester)
                # The supplied pair nominates a test actor and context, not a historical author.
                row.update({"status": "requester_and_context_bound",
                    "selected_candidate_index": candidate["candidate_index"],
                    "selected_pair_fingerprint": candidate["source_candidate_fingerprint"],
                    "role_bindings": {"requester": deepcopy(requester), "context_anchor": deepcopy(context),
                                      "operation_target": None},
                    "driver_bindings": {"user_id": user["user_id"]}})
                row["candidate_checks"][-1]["selected"] = True
                if contract["user_input_contract"]["trigger_record"].get("supplies_queried") is True:
                    row["user_fact_grants"] = [{"role": "requester", "field": "user_id",
                        "timing": "on_agent_request_only", "value": user["user_id"],
                        "source_handle_fingerprint": requester["handle_fingerprint"],
                        "basis": "verified_test_actor_identity_and_upstream_supplies_queried"}]
            if row["status"] == "not_bound":
                row["status"] = "deferred"
                row["hold_reason"] = "no_supplied_pair_with_verified_identity_and_owner"
            row["pending_driver_binding_names"] = [name for name in row["pending_driver_binding_names"]
                                                     if name not in row["driver_bindings"]]
        rows.append(seal(row, "requester_binding_fingerprint"))
    return seal({"schema_version": SCHEMA, "source_handoff_fingerprint": handoff["handoff_fingerprint"],
        "source_contract_set_fingerprint": contracts["contract_set_fingerprint"],
        "source_reference_set_fingerprint": references["reference_set_fingerprint"],
        "database_fingerprint": store.fingerprint, "selection_policy": deepcopy(POLICY),
        "source_compatibility_limits": deepcopy(handoff["source_compatibility_limits"]),
        "rows": rows, "summary": {"branches": len(rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "selected_pairs": sum(r["selected_candidate_index"] is not None for r in rows),
            "requesters_bound": sum(r["role_bindings"]["requester"] is not None for r in rows),
            "context_anchors_bound": sum(r["role_bindings"]["context_anchor"] is not None for r in rows),
            "operation_targets_bound": 0, "user_fact_grants": sum(len(r["user_fact_grants"]) for r in rows),
            "runtime_ready_branches": 0, "external_calls": 0, "target_agent_calls": 0},
        "scope": "ordinary creation identity/context binding only; no policy-bypass inputs, transaction construction or execution"},
        "requester_binding_set_fingerprint")


def project_requester_facts(row, store):
    """Verify the sole allowed grant against the snapshot before exposing its value.

    This returns an availability envelope, not an initial message or auto-responder.
    The runtime must still enforce the on-request timing before sending a fact.
    """
    verify_fingerprint(row, "requester_binding_fingerprint")
    if row["database_fingerprint"] != store.fingerprint:
        raise ValueError("fact projection snapshot mismatch")
    grants = row["user_fact_grants"]
    facts = {}
    if grants:
        if len(grants) != 1 or row["status"] != "requester_and_context_bound":
            raise ValueError("unexpected identity fact grants")
        grant, handle = grants[0], row["role_bindings"]["requester"]
        if (grant["role"] != "requester" or grant["field"] != "user_id"
                or grant["timing"] != "on_agent_request_only" or handle["table"] != "users"
                or grant["source_handle_fingerprint"] != handle["handle_fingerprint"]
                or row["full_user_requirement_contract"]["trigger_record"].get("supplies_queried") is not True):
            raise ValueError("identity disclosure is outside supported grant")
        actual = store.read(handle)["user_id"]
        if grant["value"] != actual or row["driver_bindings"].get("user_id") != actual:
            raise ValueError("identity fact differs from bound user")
        facts["user_id"] = actual
    return {"source_requirements": {k: deepcopy(row["source_user_requirements"][k]) for k in ("request", "when")},
            "facts_available_on_agent_request": facts,
            "initial_user_message": None, "complete_user_input": False}


def render_bindings(result):
    lines = ["# Step 7E requester and context binding", "", "Identity preparation for ordinary new-creation requests only; not a complete fixture or a runnable test.", "",
             "| Branch | Status | Selected candidate index (0-based) | User fact count | Hold reason |",
             "| --- | --- | ---: | ---: | --- |"]
    for row in result["rows"]:
        index = row["selected_candidate_index"]
        lines.append(f"| {row['branch_id']} | {row['status']} | {index if index is not None else 'not selected'} | {len(row['user_fact_grants'])} | {row['hold_reason'] or 'none (complete request still not prepared)'} |")
    return "\n".join(lines + ["", "Existing orders are private context only; no itinerary, payment or passenger information was copied. The target object has not been created yet.", ""])
