"""Step 7B v2: object references against v5-branch-test-contract-set/v0.2 and
v5-driver-plan-set/v0.2 (155-branch). v1 (v5_object_references_v1.py) is untouched.

compile_references's only real v1-specific coupling is its hardcoded source
schema_version guard -- SnapshotObjects/inspect_candidates operate purely on
fixture_contract/the database and carry no contract/plan-schema-version
awareness at all, and fixture_binding_plan's shape is identical between v1 and
v2 plans (both built by the same frozen driver_plan_lowering_v1.py helpers), so
those are imported and reused directly rather than copied.
"""

from collections import Counter
from copy import deepcopy

from ..compiler.artifacts import content_sha256
from ..compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .v5_object_references_v1 import SCHEMA, SnapshotObjects, inspect_candidates, seal


def compile_references_v2(contracts, plans, bundle):
    verify_fingerprint(contracts, "contract_set_fingerprint")
    verify_fingerprint(plans, "plan_set_fingerprint")
    manifest = bundle["manifest"]
    verify_fingerprint(manifest, "bundle_fingerprint")
    if contracts["schema_version"] != "agentspectesting.v5-branch-test-contract-set/v0.2" or plans["schema_version"] != "agentspectesting.v5-driver-plan-set/v0.2":
        raise ValueError("unsupported source schema")
    if plans["source_contract_set_fingerprint"] != contracts["contract_set_fingerprint"] or manifest["source_plan_set_fingerprint"] != plans["plan_set_fingerprint"]:
        raise ValueError("contract/plan/data-source lineage mismatch")
    if manifest["source_assembly_fingerprint"] != contracts["source_fingerprints"]["assembly"]:
        raise ValueError("assembly lineage mismatch")
    if manifest["status"] != "prepared_with_compatibility_limits":
        raise ValueError("source bundle is blocked")
    def index(rows):
        out = {}
        for row in rows:
            if row["branch_id"] in out:
                raise ValueError("duplicate branch")
            out[row["branch_id"]] = row
        return out
    contract_index, plan_index = index(contracts["contracts"]), index(plans["plans"])
    if set(contract_index) != set(plan_index):
        raise ValueError("branch membership mismatch")
    store = SnapshotObjects(bundle["database"], manifest["payloads"]["database_snapshot.json"]["content_fingerprint"])
    rows = []
    for bid, contract in contract_index.items():
        plan = plan_index[bid]
        verify_fingerprint(contract, "branch_test_contract_fingerprint")
        verify_fingerprint(plan, "driver_plan_fingerprint")
        if plan["source_contract_fingerprint"] != contract["branch_test_contract_fingerprint"]:
            raise ValueError("plan branch source mismatch")
        fixture = contract["fixture_contract"]
        if plan["fixture_binding_plan"]["candidate_source"]["candidate_count"] != fixture["lookup"]["n_matches"]:
            raise ValueError("plan candidate count mismatch")
        candidates = inspect_candidates(fixture, store)
        lookup_status = fixture["lookup"]["lookup_status"]
        role_hint = ("context_anchor_not_operation_target" if lookup_status == "anchored" else
                     "upstream_target_candidate_not_operation_binding" if lookup_status == "matched" else
                     "generic_context_pool_no_target_assignment" if lookup_status == "no_conditions" else
                     "role_undetermined")
        row = {"branch_id": bid, "source_contract_fingerprint": contract["branch_test_contract_fingerprint"],
            "source_plan_fingerprint": plan["driver_plan_fingerprint"], "lookup_status": lookup_status,
            "source_lookup_ref": deepcopy(fixture["lookup_ref"]), "role_hint": role_hint,
            "candidates": candidates,
            "summary": {"supplied_pairs": len(candidates),
                "reference_status_counts": dict(Counter(c["reference_status"] for c in candidates)),
                "relation_status_counts": dict(Counter(c["relation_observation"]["status"] for c in candidates)),
                "duplicate_pair_occurrences": sum(c["duplicate_of_index"] is not None for c in candidates)},
            "role_bindings": {"requester": None, "operation_target": None, "context_anchor": None},
            "pending_driver_binding_names": deepcopy(plan["fixture_binding_plan"]["required_driver_binding_names"]),
            "condition_status": "not_evaluated", "user_information_disclosure": "not_decided",
            "runtime_ready": False}
        rows.append(seal(row, "reference_row_fingerprint"))
    all_candidates = [c for row in rows for c in row["candidates"]]
    return seal({"schema_version": SCHEMA,
        "source_contract_set_fingerprint": contracts["contract_set_fingerprint"],
        "source_plan_set_fingerprint": plans["plan_set_fingerprint"],
        "source_bundle_fingerprint": manifest["bundle_fingerprint"],
        "database_fingerprint": store.fingerprint,
        "source_compatibility_limits": deepcopy(manifest["compatibility"]),
        "rows": rows,
        "summary": {"branch_count": len(rows), "supplied_pair_occurrences": len(all_candidates),
            "resolved_pair_occurrences": sum(c["reference_status"] == "resolved" for c in all_candidates),
            "unresolved_pair_occurrences": sum(c["reference_status"] == "unresolved" for c in all_candidates),
            "relation_status_counts": dict(Counter(c["relation_observation"]["status"] for c in all_candidates)),
            "selected_candidates": 0, "runtime_ready_branches": 0, "external_calls": 0},
        "scope": "referential integrity only; no candidate ranking, scenario selection, eligibility validation or prompt generation",
        "consumer_policy": {"do_not_promote_handles_to_role_bindings": True,
            "do_not_disclose_private_ids_automatically": True, "preserve_given_and_when": True,
            "unknown_or_mismatched_relation_is_not_verified": True}}, "reference_set_fingerprint")
