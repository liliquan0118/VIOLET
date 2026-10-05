"""Snapshot-scoped object references, not scenario selection or eligibility checks."""

from collections import Counter
from copy import deepcopy

from ..compiler.artifacts import content_sha256
from ..compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .v5_candidate_binding_v1 import _object, _relation


SCHEMA = "agentspectesting.v5-object-reference-set/v0.1"


def seal(value, field):
    value[field] = content_sha256(value)
    return value


class SnapshotObjects:
    """Immutable private snapshot; reads return copies and never search other rows."""

    def __init__(self, database, expected_fingerprint):
        if content_sha256(database) != expected_fingerprint:
            raise ValueError("object store database fingerprint mismatch")
        self._database = deepcopy(database)
        self.fingerprint = expected_fingerprint

    def resolve(self, table, key):
        if not isinstance(table, str) or not isinstance(key, str) or not key:
            return {"status": "unresolved", "reason": "invalid_object_reference", "handle": None}
        value, error = _object(self._database, table, key)
        if error:
            return {"status": "unresolved", "reason": error, "handle": None}
        handle = seal({"schema_version": "agentspectesting.snapshot-object-handle/v0.1",
            "database_fingerprint": self.fingerprint, "table": table, "key": key,
            "object_fingerprint": content_sha256(value)}, "handle_fingerprint")
        return {"status": "resolved", "reason": None, "handle": handle}

    def read(self, handle):
        verify_fingerprint(handle, "handle_fingerprint")
        if handle["schema_version"] != "agentspectesting.snapshot-object-handle/v0.1" or handle["database_fingerprint"] != self.fingerprint:
            raise ValueError("object handle belongs to another snapshot")
        resolved = self.resolve(handle["table"], handle["key"])
        if resolved["status"] != "resolved" or resolved["handle"] != handle:
            raise ValueError("object handle no longer matches its record")
        return deepcopy(self._database[handle["table"]][handle["key"]])


def inspect_candidates(fixture, store):
    """Retain every supplied pair in order, including unresolved and duplicate pairs."""
    lookup = fixture["lookup"]
    candidates = lookup["matches"]
    if lookup["n_matches"] != len(candidates):
        raise ValueError("candidate count mismatch")
    results, first_seen = [], {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be an object")
        candidate_fp = content_sha256(candidate)
        user = store.resolve("users", candidate.get("user_id"))
        root_table = fixture["root"]
        if root_table is None:
            root = {"status": "not_applicable", "reason": None, "handle": None}
            if candidate.get("root_id") is not None:
                root = {"status": "unresolved", "reason": "root_id_supplied_without_root_table", "handle": None}
        else:
            root = store.resolve(root_table, candidate.get("root_id"))
        errors = [f"{role}:{r['reason']}" for role, r in (("user", user), ("root", root)) if r["status"] == "unresolved"]
        relation = {"status": "not_checked", "basis": "object_reference_unresolved"}
        if not errors:
            # Existing owner/not_owner comparison only, not business eligibility.
            obj = store.read(root["handle"]) if root["handle"] else None
            relation = _relation({"fixture_requirements": fixture}, candidate, obj)
        results.append({"candidate_index": index, "source_candidate": deepcopy(candidate),
            "source_candidate_fingerprint": candidate_fp, "duplicate_of_index": first_seen.get(candidate_fp),
            "user_reference": user, "root_reference": root,
            "reference_status": "unresolved" if errors else "resolved", "reference_errors": errors,
            "relation_observation": relation,
            "selected": False, "business_eligibility": "not_evaluated"})
        first_seen.setdefault(candidate_fp, index)
    return results


def compile_references(contracts, plans, bundle):
    verify_fingerprint(contracts, "contract_set_fingerprint")
    verify_fingerprint(plans, "plan_set_fingerprint")
    manifest = bundle["manifest"]
    verify_fingerprint(manifest, "bundle_fingerprint")
    if contracts["schema_version"] != "agentspectesting.v5-branch-test-contract-set/v0.1" or plans["schema_version"] != "agentspectesting.v5-driver-plan-set/v0.1":
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


def render_references(result):
    lines = ["# Step 7B object reference check", "", "A resolvable reference does not mean the object has been selected, nor that the business condition holds.", "",
        "| Branch | Candidate occurrences | Resolved references | Unresolved | Role hint |", "| --- | ---: | ---: | ---: | --- |"]
    for row in result["rows"]:
        s = row["summary"]
        lines.append(f"| {row['branch_id']} | {s['supplied_pairs']} | {s['reference_status_counts'].get('resolved', 0)} | {s['reference_status_counts'].get('unresolved', 0)} | {row['role_hint']} |")
    return "\n".join(lines + ["", "No candidate has been selected; requester/operation_target/context_anchor are all unbound.",
        "The relation comparison only checks existing records and is not a business eligibility judgment; no ID has been written into any user message.", ""])
