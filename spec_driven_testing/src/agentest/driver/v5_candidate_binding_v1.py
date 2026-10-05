"""Read-only candidate binding, not a ready-to-run Driver or prompt generator."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta

from ..compiler.artifacts import content_sha256
from ..compiler.v5_given_when_assembly_v1 import SCHEMA as ASSEMBLY_SCHEMA, assemble_given_when
from ..compiler.v5_given_when_preparation_v1 import verify_fingerprint


SCHEMA = "agentspectesting.v5-candidate-binding/v0.1"
IDENTITY_FIELDS = {"users": "user_id", "reservations": "reservation_id", "flights": "flight_number"}


def seal(value, field):
    value[field] = content_sha256(value)
    return value


def validate_source_assembly(assembly, intake, scope):
    verify_fingerprint(assembly, "assembly_set_fingerprint")
    if assembly["schema_version"] != ASSEMBLY_SCHEMA:
        raise ValueError("expected v5 upstream assembly")
    expected = assemble_given_when(intake, scope)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("source_release", "assembly_set_fingerprint")}
    if strip(expected) != strip(assembly):
        raise ValueError("assembly differs from frozen source reconstruction")


def _object(database, table, key):
    if table not in IDENTITY_FIELDS:
        return None, "unsupported_root_table"
    item = database.get(table, {}).get(key)
    if not isinstance(item, dict):
        return None, "object_not_found"
    if item.get(IDENTITY_FIELDS[table]) != key:
        return None, "object_identity_mismatch"
    return item, None


def _relation(branch, candidate, root_object):
    root = branch["fixture_requirements"]["root"]
    assertion = branch["fixture_requirements"]["relation_assertions"]["user_requirements"]
    if root in (None, "flights"):
        return {"status": "not_applicable", "basis": "no_owned_root_object_in_airline_adapter", "upstream_assertion": deepcopy(assertion)}
    if not assertion["present"] or assertion.get("value") not in ("owner", "not_owner"):
        return {"status": "not_checked", "basis": "no_supported_relation_assertion", "upstream_assertion": deepcopy(assertion)}
    owner = root_object.get("user_id")
    if not isinstance(owner, str):
        return {"status": "not_checked", "basis": "owner_identity_missing", "upstream_assertion": deepcopy(assertion)}
    equal = owner == candidate["user_id"]
    passed = equal if assertion["value"] == "owner" else not equal
    return {"status": "matched" if passed else "mismatched", "basis": "root_user_id_compared_to_candidate_user_id",
            "root_user_id": owner, "candidate_user_id": candidate["user_id"], "upstream_assertion": deepcopy(assertion)}


def _predicate_observation(condition, branch, objects, clock):
    raw = condition["source_condition"]
    base = {"condition_id": condition["condition_id"], "source_condition": deepcopy(raw), "truth": None,
            "status": "not_evaluated", "basis": None}
    encoding = condition["optional_encoding"]
    if encoding["status"] != "available":
        return {**base, "basis": "unsupported_condition_encoding"}
    expr = encoding["expression"]
    steps = expr["subject"]["path_steps"]
    if any(s["traversal"] != "field" for s in steps):
        return {**base, "basis": "collection_and_date_scope_not_bound"}
    table, root = raw["table"], branch["fixture_requirements"]["root"]
    if table == root:
        obj, role = objects["root"], "root"
    elif table == "users" and root == "reservations":
        obj, role = objects["user"], "requester"
    else:
        return {**base, "basis": "condition_subject_not_bound"}
    value = obj
    for step in steps:
        if not isinstance(value, dict) or step["field"] not in value:
            return {**base, "basis": "field_missing", "subject_role": role}
        value = value[step["field"]]
    expected = expr["right"].get("value")
    comparable = value
    if expr["right"]["kind"] == "relative_test_time":
        try:
            comparable = datetime.fromisoformat(value.replace("Z", "+00:00"))
            expected = datetime.fromisoformat(clock.replace("Z", "+00:00")) + timedelta(**expr["right"]["offset"])
        except (ValueError, TypeError, AttributeError, OverflowError):
            return {**base, "basis": "time_value_not_comparable", "actual": value, "subject_role": role}
    op = raw["op"]
    numeric = lambda v: type(v) in (int, float)
    compatible = type(comparable) is type(expected) or (numeric(comparable) and numeric(expected))
    try:
        if op in ("eq", "ne"):
            equal = comparable == expected if compatible else False
            truth = equal if op == "eq" else not equal
        elif op in ("gt", "ge", "lt", "le") and compatible and not isinstance(comparable, (bool, dict, list, type(None))):
            truth = {"gt": lambda: comparable > expected, "ge": lambda: comparable >= expected,
                     "lt": lambda: comparable < expected, "le": lambda: comparable <= expected}[op]()
        elif op in ("in", "not_in") and isinstance(expected, list):
            found = any(content_sha256(value) == content_sha256(x) for x in expected)
            truth = found if op == "in" else not found
        else:
            return {**base, "basis": "operator_or_value_type_not_supported", "actual": value, "subject_role": role}
    except (ValueError, TypeError, OverflowError):
        return {**base, "basis": "values_not_comparable", "actual": value, "subject_role": role}
    return {**base, "status": "evaluated", "truth": truth, "basis": "bound_snapshot_scalar_comparison",
            "actual": value, "expected": expected.isoformat() if isinstance(expected, datetime) else expected, "subject_role": role}


def _bind_branch(branch, database, clock):
    result = {"branch_id": branch["branch_id"], "source_assembly_fingerprint": branch["assembly_fingerprint"],
              "object_binding_status": "not_bound", "selected_candidate": None, "candidate_checks": [],
              "test_readiness": "not_assessed", "overall_given_truth": None,
              "runtime_reachability": "not_assessed", "oracle_status": "not_connected"}
    if branch["assembly_status"] != "ready_for_binding":
        return seal({**result, "object_binding_status": "blocked_by_input"}, "binding_fingerprint")
    fixture = branch["fixture_requirements"]
    result["given_requirements"] = deepcopy(branch["given_requirements"])
    result["fixture_requirement_context"] = {
        "root": deepcopy(fixture["root"]), "relation_assertions": deepcopy(fixture["relation_assertions"]),
        "lookup_ref": deepcopy(fixture["lookup_ref"]), "provenance": deepcopy(fixture["provenance"])}
    result["database_condition_observations"] = [
        {"condition_id": c["condition_id"], "source_condition": deepcopy(c["source_condition"]),
         "truth": None, "status": "not_evaluated", "basis": "candidate_not_bound"}
        for c in branch["given_requirements"]["database_conditions"]]
    result["user_input_preparation"] = {
        "source_context": deepcopy(branch["source_context"]),
        "requirements": deepcopy(branch["user_input_requirements"]),
        "original_trigger_requirement": deepcopy(branch["trigger_requirements"]),
        "status": "requirements_carried_values_not_realized",
        "initial_user_message": None,
        "private_context_is_not_user_supplied_information": True,
        "automatic_identifier_insertion": False,
    }
    result["non_database_requirements"] = deepcopy(branch["given_requirements"]["non_database_conditions"])
    result["lookup_status"] = fixture["lookup"]["lookup_status"]
    if result["lookup_status"] == "makeable":
        for observation in result["database_condition_observations"]:
            observation["basis"] = "requires_fixture_construction"
        return seal({**result, "object_binding_status": "requires_fixture_construction"}, "binding_fingerprint")
    if result["lookup_status"] not in ("matched", "anchored", "no_conditions"):
        return seal({**result, "object_binding_status": "unsupported_lookup_status"}, "binding_fingerprint")
    for index, candidate in enumerate(fixture["lookup"]["matches"]):
        user, user_error = _object(database, "users", candidate["user_id"])
        root = fixture["root"]
        obj, root_error = (None, None) if root is None else _object(database, root, candidate["root_id"])
        check = {"candidate_index": index, "candidate": deepcopy(candidate), "errors": [e for e in (user_error, root_error) if e]}
        if check["errors"]:
            result["candidate_checks"].append(check)
            continue
        relation = _relation(branch, candidate, obj)
        check["relation"] = relation
        result["candidate_checks"].append(check)
        if relation["status"] in ("mismatched", "not_checked"):
            continue
        result.update({"object_binding_status": "objects_bound", "selected_candidate": deepcopy(candidate),
                       "selected_candidate_index": index, "relation_observation": relation,
                       "candidate_role": "context_anchor_not_automatically_requested_object" if result["lookup_status"] != "matched" else "upstream_target_candidate_not_fully_verified",
                       "private_fixture_context": {"root_table": root, "user": deepcopy(user), "root": deepcopy(obj)},
                       "database_condition_observations": [_predicate_observation(c, branch, {"user": user, "root": obj}, clock)
                                                           for c in branch["given_requirements"]["database_conditions"]]})
        break
    if result["object_binding_status"] == "not_bound":
        result["object_binding_status"] = "no_resolvable_supplied_candidate"
    result["selection_policy"] = "first_supplied_pair_with_resolved_identity_and_applicable_relation_no_predicate_ranking"
    # Per-condition observations do not determine the Boolean meaning of Given.
    # Even all scalar observations true is not a proof of a complete test setup.
    return seal(result, "binding_fingerprint")


def bind_candidates(assembly, database, reference_time):
    verify_fingerprint(assembly, "assembly_set_fingerprint")
    if assembly.get("schema_version") != ASSEMBLY_SCHEMA:
        raise ValueError("expected v5 assembly schema")
    datetime.fromisoformat(reference_time.replace("Z", "+00:00"))
    if any(not isinstance(database.get(table), dict) for table in IDENTITY_FIELDS):
        raise ValueError("expected airline database tables")
    rows = assembly["branches"]
    if len({b["branch_id"] for b in rows}) != len(rows):
        raise ValueError("duplicate assembly branch")
    for branch in rows:
        verify_fingerprint(branch, "assembly_fingerprint")
    branches = [_bind_branch(b, database, reference_time) for b in rows]
    observations = [o for b in branches for o in b.get("database_condition_observations", [])]
    return seal({"schema_version": SCHEMA, "source_assembly_set_fingerprint": assembly["assembly_set_fingerprint"],
                 "database_fingerprint": content_sha256(database), "reference_time": reference_time,
                 "scope": "offline object binding and user-input requirements handoff; not generated executable test cases",
                 "branches": branches, "deferred_branch_ids": deepcopy(assembly["deferred_branch_ids"]),
                 "summary": {"branch_count": len(branches), "object_binding_status_counts": dict(Counter(b["object_binding_status"] for b in branches)),
                             "database_condition_observation_counts": dict(Counter("true" if o["truth"] is True else "false" if o["truth"] is False else "not_evaluated" for o in observations)),
                             "observation_hold_reasons": dict(Counter(o["basis"] for o in observations if o["status"] == "not_evaluated")),
                             "user_messages_generated": 0, "database_writes": 0, "external_llm_calls": 0,
                             "target_agent_calls": 0, "test_ready_branch_count": 0}}, "binding_set_fingerprint")
