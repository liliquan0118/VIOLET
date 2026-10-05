"""Step 5: lossless assembly of frozen v5 preparation, never test execution."""

from collections import Counter
from copy import deepcopy

from .artifacts import content_sha256
from .oracle_requirement_acceptance_v1 import validate_accepted_oracle_requirement_set
from .oracle_expectation_compiler_v2 import validate_oracle_expectation_set
from .v5_given_when_assembly_v1 import assemble_given_when
from .v5_given_when_preparation_v1 import verify_fingerprint


SCHEMA = "agentspectesting.v5-branch-test-contract-set/v0.1"


def _index(rows, key="branch_id"):
    result = {}
    for row in rows:
        identity = row.get(key)
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"missing or duplicate {key}")
        result[identity] = row
    return result


def _equal(left, right, label):
    if left != right:
        raise ValueError(f"{label} mismatch")


def _seal(value, key):
    value[key] = content_sha256(value)
    return value


def compile_contracts(intake, scope, assembly, accepted, expectations, preparation):
    """Join only the Step 3/4 pilot; do not fabricate Oracle for other branches."""
    verify_fingerprint(assembly, "assembly_set_fingerprint")
    # Reuse the frozen Step 2 mechanical assembler, not a new semantic review.
    rebuilt = assemble_given_when(intake, scope)
    actual = deepcopy(assembly)
    actual.pop("source_release", None)
    actual.pop("assembly_set_fingerprint")
    rebuilt.pop("assembly_set_fingerprint")
    _equal(actual, rebuilt, "Step 1/2 assembly")
    accepted = validate_accepted_oracle_requirement_set(accepted)
    expectations = validate_oracle_expectation_set(expectations)
    verify_fingerprint(preparation, "preparation_fingerprint")
    _equal(preparation["schema_version"], "agentspectesting.step4-unified-preparation/v0.1", "Step 4 schema")
    _equal(expectations["source_accepted_set_fingerprint"], accepted["accepted_set_fingerprint"], "expectation source")
    _equal(preparation["source_accepted_set_fingerprint"], accepted["accepted_set_fingerprint"], "preparation requirements")
    _equal(preparation["source_expectation_set_fingerprint"], expectations["expectation_set_fingerprint"], "preparation expectations")
    _equal(preparation["runtime_dispatch_enabled"], False, "preparation dispatch")
    source_by_id = _index(intake["branches"])
    gw_by_id = _index(assembly["branches"])
    accepted_by_id = _index(accepted["branches"])
    expected_by_id = _index(expectations["branches"])
    _equal(set(accepted_by_id), set(expected_by_id), "Oracle branch membership")
    if not accepted_by_id or not set(accepted_by_id) <= set(gw_by_id):
        raise ValueError("pilot is empty or outside admitted assembly")
    requirements = _index([r for b in accepted["branches"] for r in b["requirements"]], "requirement_id")
    expected = _index([r for b in expectations["branches"] for r in b["expectations"]], "requirement_id")
    prepared = _index(preparation["rows"], "requirement_id")
    _equal(set(requirements), set(expected), "requirement/expectation membership")
    _equal(set(requirements), set(prepared), "requirement/preparation membership")

    contracts = []
    for branch_id, branch in accepted_by_id.items():
        gw = gw_by_id[branch_id]
        if gw["assembly_status"] != "ready_for_binding":
            raise ValueError("pilot assembly has blocked input")
        context = branch["branch_context"]
        _equal(context, expected_by_id[branch_id]["branch_context"], "Oracle context")
        _equal(context["v5_source"]["source_assembly_fingerprint"], gw["assembly_fingerprint"], "branch assembly source")
        _equal(context["v5_source"]["source_refs"], gw["source_refs"], "branch source references")
        for key in ("given", "when", "then"):
            _equal(context["gwt"][key], gw["source_context"]["gwt"][key], f"branch {key}")
        for key in ("spec_id", "kind", "origin", "rule_text"):
            _equal(context[key], source_by_id[branch_id]["source_fields"][key], f"branch {key}")
        checks, gaps = [], []
        for requirement in branch["requirements"]:
            rid = requirement["requirement_id"]
            expectation, row = expected[rid], prepared[rid]
            for item in (requirement, expectation, row):
                _equal(item["branch_id"], branch_id, "requirement branch")
            _equal(row["source_requirement_fingerprint"], requirement["requirement_fingerprint"], "prepared requirement")
            _equal(expectation["source_requirement_fingerprint"], requirement["requirement_fingerprint"], "expected requirement")
            _equal(row["requirement_text"], requirement["requirement_text"], "requirement text")
            _equal(row["expectation_id"], expectation["expectation_id"], "prepared expectation ID")
            _equal(row["expected_observation"], expectation["expected_observation"], "prepared expectation")
            _equal(row["runtime_ready"], False, "preparation readiness")
            _equal(row["runtime_validated"], False, "preparation validation")
            route = row["effective_route"]
            if route["kind"] not in {"mechanical", "behavior", "record_relation"}:
                raise ValueError("unsupported prepared route interface; explicit adapter required")
            if route["kind"] == "record_relation":
                if not route.get("mandatory_guard") or not route["record_question_preparer"].endswith(".prepare_guarded_question"):
                    raise ValueError("record preparation lost mandatory guard")
            # Preserve the entire frozen row, including old provenance and current route.
            checks.append({"requirement": deepcopy(requirement), "expectation": deepcopy(expectation),
                           "preparation": deepcopy(row), "dispatch_authority": "preparation.effective_route"})
            gaps.extend({"requirement_id": rid, **deepcopy(gap)} for gap in row["known_gaps"])
        fixture = gw["fixture_requirements"]
        lookup = fixture["lookup"]
        _equal(lookup["n_matches"], len(lookup["matches"]), "candidate count")
        if fixture["selected_candidate"] is not None:
            raise ValueError("Step 5 does not accept preselected fixture as preparation input")
        work = [
            {"code": "prepare_abstract_driver_plan", "owner_step": 6},
            {"code": "bind_or_construct_and_verify_fixture", "owner_step": 7},
            {"code": "bind_user_requirements_and_symbolic_scope", "owner_step": 7},
            {"code": "verify_actual_given_when_and_collect_evidence", "owner_step": 8},
            {"code": "integrate_effective_routes_and_evaluate_trace", "owner_step": 9},
        ]
        if gaps:
            work.append({"code": "preserve_rule_gap_guard", "owner_step": 9,
                         "instruction": "apply the frozen preparation guard before any judge question; never infer a missing rule"})
        contract = {
            "branch_test_contract_id": f"{branch_id}::V5BTC01", "branch_id": branch_id,
            "spec_id": gw["spec_id"], "gwt": deepcopy(gw["source_context"]["gwt"]),
            "source_contract": {"fields": deepcopy(source_by_id[branch_id]["source_fields"]),
                                "references": deepcopy(gw["source_refs"])},
            "given_contract": deepcopy(gw["given_requirements"]),
            "when_contract": deepcopy(gw["trigger_requirements"]),
            "user_input_contract": deepcopy(gw["user_input_requirements"]),
            "fixture_contract": deepcopy(fixture),
            "oracle_contract": {"checks": checks, "known_gaps": gaps,
                                "legacy_runtime_preconditions": deepcopy(preparation["legacy_runtime_preconditions"]),
                                "downstream_obligations": deepcopy(preparation["downstream_obligations"])},
            "source_assembly": deepcopy(gw),
            "preparation_gates": {
                "given_when": "assembled_not_runtime_verified",
                "fixture": "candidate_pool_unverified" if lookup["matches"] else "binding_or_construction_required",
                "oracle": "prepared_with_rule_gaps" if gaps else "prepared",
            },
            "assembly_status": "assembled_with_rule_gaps" if gaps else "assembled",
            "planning_handoff": "eligible_with_explicit_obligations",
            "required_work": work,
            "runtime_ready": False, "runtime_validated": False, "agent_verdict": None,
        }
        contracts.append(_seal(contract, "branch_test_contract_fingerprint"))
    selected = set(accepted_by_id)
    result = {
        "schema_version": SCHEMA,
        "scope": "frozen Step 3/4 pilot only; assembly, not runnable cases or full-spec coverage",
        "source_fingerprints": {"intake": intake["intake_fingerprint"], "scope": scope["scope_fingerprint"],
            "assembly": assembly["assembly_set_fingerprint"], "accepted": accepted["accepted_set_fingerprint"],
            "expectations": expectations["expectation_set_fingerprint"], "preparation": preparation["preparation_fingerprint"]},
        "contracts": contracts,
        "out_of_pilot_branch_ids": [b["branch_id"] for b in assembly["branches"] if b["branch_id"] not in selected],
        "source_deferred_branch_ids": deepcopy(assembly["deferred_branch_ids"]),
        "summary": {"branch_count": len(contracts), "requirement_count": len(requirements),
            "assembly_status_counts": dict(Counter(c["assembly_status"] for c in contracts)),
            "route_counts": dict(Counter(r["effective_route"]["kind"] for r in prepared.values())),
            "known_rule_gap_count": sum(len(c["oracle_contract"]["known_gaps"]) for c in contracts),
            "out_of_pilot_count": len(gw_by_id) - len(selected),
            "source_deferred_count": len(assembly["deferred_branch_ids"]),
            "runtime_ready_count": 0, "external_calls": 0, "target_agent_calls": 0},
        "downstream_interface": {"requires_step6_adapter": True,
            "legacy_driver_accepts_this_schema": False,
            "reason": "v5 preserves unverified Given/When and new effective routes; never relabel as legacy ready_for_driver"},
        "runtime_dispatch_enabled": False,
    }
    return _seal(result, "contract_set_fingerprint")


def render_contracts(result):
    lines = ["# Step 5 Test Contract Summary", "", "This only assembles preparation requirements; it does not mean data has been bound, conditions hold, or the Agent has passed the test.", "",
             "| Branch | Checks | Data candidates | Judgment route | Missing rules |", "| --- | ---: | ---: | --- | ---: |"]
    for c in result["contracts"]:
        oracle = c["oracle_contract"]
        routes = ", ".join(x["preparation"]["effective_route"]["kind"] for x in oracle["checks"])
        lines.append(f"| {c['branch_id']} | {len(oracle['checks'])} | {c['fixture_contract']['lookup']['n_matches']} | {routes} | {len(oracle['known_gaps'])} |")
    lines += ["", f"Out of scope: {result['summary']['out_of_pilot_count']} integrated branches do not yet have a complete Oracle in this round; another {result['summary']['source_deferred_count']} branches remain deferred by Step 1.",
              "", "All branches still need subsequent planning, data and user-input binding, Given/When verification, and trajectory evaluation.",
              "The equal-time boundary continues to be blocked by the original guard; a missing rule does not mean pass or violation.",
              "The old Driver does not consume this schema directly; the next step is the Step 6 interface adaptation.", ""]
    return "\n".join(lines)
