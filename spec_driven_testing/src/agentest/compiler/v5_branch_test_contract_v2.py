"""Step 5, v2: lossless assembly of frozen Step 1/2 preparation plus the
CURRENT LIVE Step 3/4 oracle pipeline, across all 155 branches.

v1 (v5_branch_test_contract_v1.py, never edited, kept as-is for its own
6-branch/7-requirement frozen pilot scope) required a single frozen
"preparation" artifact (v5-evidence-step) that only ever covered 6 branches
and cannot be mechanically re-run at full scale (its judge-route calibration
is hand-authored per case, see docs/oracle_requirement_pipeline_v0_7.md
section 36). v2 does not use that artifact at all. It reads Step 1/2 from
the SAME frozen releases v1 used (those already cover the full 155 branches
-- verified against releases/v5-given-when-step-v0.1.0's own manifest
summary), and reads Step 3/4 from the CURRENT LIVE outputs/ produced by this
project's oracle_requirement_pipeline_v0_7.md work: accepted_requirements.json,
expectations.json, runtime_observation_bindings (extended), oracle_evaluator_contracts
(extended), and semantic_judge_contracts (extended). Nothing here overwrites
or reinterprets v1; this is a parallel, live-data path.

Each requirement's judging mechanism is classified into exactly one of four
route kinds, honestly reflecting what oracle_evaluator_contracts_v0_1_no_legacy_hint/contracts.json
already says about it -- never fabricated or inferred:
  - "mechanical": evaluator_status == "executable". The compiled program IS
    the checker; embedded verbatim.
  - "semantic_judge": evaluator_status == "deferred" AND a "ready"
    semantic_judge contract exists for this requirement_id (7 real cases as
    of this round). The judge program is real and calibrated at the profile
    level, but has never been run against a real execution (see
    docs/oracle_requirement_pipeline_v0_7.md section 33.5) -- this route
    kind does not claim otherwise.
  - "non_decisive": evaluator_status == "deferred" with no semantic_judge
    match. Confirmed in section 32 to be a correct terminal state (a
    permissive Then, or a bare tool_call kept alive only for a sibling
    temporal_relation's binding), not a gap.
  - "known_gap": evaluator_status == "needs_adjudication". No route exists
    yet; reported honestly, never silently dropped or defaulted to a pass.
"""

from collections import Counter
from copy import deepcopy

from .artifacts import content_sha256
from .oracle_requirement_acceptance_v1 import validate_accepted_oracle_requirement_set
from .oracle_expectation_compiler_v2 import validate_oracle_expectation_set
from .runtime_observation_binding_extension_v1 import validate_runtime_observation_binding_set_extended
from .oracle_evaluator_contract_extension_v1 import validate_oracle_evaluator_contract_set_extended
from .semantic_judge_contract_v1 import validate_semantic_judge_contract_set
from .v5_given_when_assembly_v1 import assemble_given_when
from .v5_given_when_preparation_v1 import verify_fingerprint


SCHEMA = "agentspectesting.v5-branch-test-contract-set/v0.2"
ROUTE_KINDS = frozenset({"mechanical", "semantic_judge", "non_decisive", "known_gap"})


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


def _route_for(requirement_id, evaluator_contract, binding, semantic_judge_by_id):
    status = evaluator_contract["evaluator_status"]
    # Every route except known_gap has a real runtime_binding worth carrying
    # forward for Step 6's observation planning (see
    # driver_plan_lowering_v1.py::_oracle_event_plan, which needs
    # runtime_observation_binding.runtime_binding as its event anchor source
    # -- confirmed against real data that "non_decisive" items are still
    # genuinely BOUND (e.g. airline_037_arg#b0::OR02's send_certificate
    # tool_call has a real event_filter, it just produces no pass/fail
    # verdict), only "known_gap" items are truly extractor_kind=="unbound"
    # with no target event anchor at all. See
    # docs/v5_step6_plans_v0_2.md.
    runtime_observation_binding = {
        "binding_id": binding["binding_id"],
        "runtime_binding": deepcopy(binding["runtime_binding"]),
        "expected_observation": deepcopy(binding["expected_observation"]),
    }
    if status == "executable":
        return {
            "kind": "mechanical",
            "program": deepcopy(evaluator_contract["program"]),
            "evaluator_contract_id": evaluator_contract["evaluator_contract_id"],
            "runtime_observation_binding": runtime_observation_binding,
        }
    if status == "needs_adjudication":
        return {
            "kind": "known_gap",
            "diagnostics": deepcopy(evaluator_contract.get("diagnostics") or []),
            "reason": "No mechanical program and no calibrated judge route are wired in for this requirement yet.",
        }
    # status == "deferred"
    judge = semantic_judge_by_id.get(requirement_id)
    if judge is not None and judge["contract_status"] == "ready":
        return {
            "kind": "semantic_judge",
            "semantic_judge_contract_id": judge["semantic_judge_contract_id"],
            "matched_profile_rule_id": judge["matched_profile_rule_id"],
            "program": deepcopy(judge["program"]),
            "runtime_verified": False,
            "runtime_observation_binding": runtime_observation_binding,
        }
    # A judge record exists whenever the underlying requirement really is a
    # semantic_requirement with a semantic_deferred binding and a deferred
    # evaluator (compile_semantic_judge_contracts_extended's own gate) --
    # that population includes BOTH the genuinely "confirmed correct"
    # permissive-Then candidates (their expected_observation.operator is
    # itself "non_decisive") AND real, open semantic-judgment gaps (operator
    # is "present"/"absent" but no ready judge could be produced -- see this
    # route's own diagnostics for the real reason: no unique profile rule
    # match, a target_event_kind mismatch, an operator the judge mechanism
    # does not support at all, or (since v0.2, section 48) a present-operator
    # claim whose judge_kind isn't claim_source_support yet. See
    # docs/agentcoveragetesting_reuse_log.md sections 45/48). The operator
    # value is the real, precise signal for which of these two this route
    # actually is -- not judge-record presence alone.
    operator = runtime_observation_binding["expected_observation"].get("operator")
    if judge is not None and operator != "non_decisive":
        reason = (
            f"This is a real semantic_requirement needing semantic judgment "
            f"(expected_observation.operator={operator!r}), not a confirmed correct "
            "terminal state -- but no ready semantic_judge_contract could be produced for "
            "it (see this route's own diagnostics for the real reason -- most commonly no "
            "unique semantic_judge_profile rule matched this requirement's text, or (for "
            "operator=='present') the matched rule's judge_kind is not yet supported by the "
            "v0.2 present-claim aggregation). This is a real, open gap in judge coverage."
        )
    else:
        reason = "Confirmed correct terminal state, not a gap -- see docs/oracle_requirement_pipeline_v0_7.md section 32."
    return {
        "kind": "non_decisive",
        "diagnostics": deepcopy(evaluator_contract.get("diagnostics") or [])
            + (deepcopy(judge.get("diagnostics") or []) if judge is not None and operator != "non_decisive" else []),
        "reason": reason,
        "runtime_observation_binding": runtime_observation_binding,
    }


def compile_contracts_v2(intake, scope, assembly, accepted, expectations, bindings, evaluator_contracts, semantic_judge_contracts):
    """Join the full live Step3/4 oracle pipeline against frozen Step1/2, over all 155 branches."""
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
    bindings = validate_runtime_observation_binding_set_extended(bindings)
    evaluator_contracts = validate_oracle_evaluator_contract_set_extended(evaluator_contracts)
    semantic_judge_contracts = validate_semantic_judge_contract_set(semantic_judge_contracts)

    _equal(expectations["source_accepted_set_fingerprint"], accepted["accepted_set_fingerprint"], "expectation source")
    _equal(semantic_judge_contracts["source_accepted_set_fingerprint"], accepted["accepted_set_fingerprint"], "semantic judge source (accepted)")
    _equal(
        semantic_judge_contracts["source_evaluator_contract_set_fingerprint"],
        evaluator_contracts["evaluator_contract_set_fingerprint"],
        "semantic judge source (evaluator contracts)",
    )

    source_by_id = _index(intake["branches"])
    gw_by_id = _index(assembly["branches"])
    accepted_by_id = _index(accepted["branches"])
    expected_by_id = _index(expectations["branches"])
    binding_by_id = _index(bindings["branches"])
    contract_by_id = _index(evaluator_contracts["branches"])
    _equal(set(accepted_by_id), set(expected_by_id), "Oracle branch membership")
    _equal(set(accepted_by_id), set(binding_by_id), "binding branch membership")
    _equal(set(accepted_by_id), set(contract_by_id), "evaluator contract branch membership")
    if not accepted_by_id or not set(accepted_by_id) <= set(gw_by_id):
        raise ValueError("accepted set is empty or outside admitted assembly")

    requirements = _index([r for b in accepted["branches"] for r in b["requirements"]], "requirement_id")
    expected = _index([r for b in expectations["branches"] for r in b["expectations"]], "requirement_id")
    evaluator_by_requirement = _index(
        [c for b in evaluator_contracts["branches"] for c in b["evaluator_contracts"]], "requirement_id"
    )
    binding_by_requirement = _index(
        [bb for b in bindings["branches"] for bb in b["bindings"]], "requirement_id"
    )
    semantic_judge_by_id = _index(semantic_judge_contracts["contracts"], "requirement_id")
    _equal(set(requirements), set(expected), "requirement/expectation membership")
    _equal(set(requirements), set(evaluator_by_requirement), "requirement/evaluator-contract membership")
    _equal(set(requirements), set(binding_by_requirement), "requirement/binding membership")

    contracts = []
    route_counts: Counter = Counter()
    gap_count = 0
    for branch_id, branch in accepted_by_id.items():
        gw = gw_by_id[branch_id]
        if gw["assembly_status"] != "ready_for_binding":
            raise ValueError("assembly has blocked input")
        context = branch["branch_context"]
        _equal(context, expected_by_id[branch_id]["branch_context"], "Oracle context")
        _equal(context["v5_source"]["source_assembly_fingerprint"], gw["assembly_fingerprint"], "branch assembly source")
        _equal(context["v5_source"]["source_refs"], gw["source_refs"], "branch source references")
        for key in ("given", "when", "then"):
            _equal(context["gwt"][key], gw["source_context"]["gwt"][key], f"branch {key}")
        for key in ("spec_id", "kind", "origin", "rule_text"):
            _equal(context[key], source_by_id[branch_id]["source_fields"][key], f"branch {key}")

        checks, gaps = [], []
        if not branch["requirements"]:
            # Real, confirmed data (see docs/oracle_requirement_pipeline_v0_7.md
            # section 37): airline_099_norm#b0 has acceptance_status=="accepted"
            # with an EMPTY requirements list -- every candidate generated for
            # this Then was independently judged "no" at Step3, leaving zero
            # oracle content for a real, admitted branch. This is invisible in
            # every earlier Step3/4-only completeness check (they all operate
            # per-requirement; a branch that contributes zero requirements
            # never shows up as "deferred" or "needs_adjudication", it just
            # silently contributes 0 to every count). Never silently pass this
            # through as a clean "assembled" branch -- report it as an honest,
            # branch-level known_gap distinct from any per-requirement gap.
            gap_count += 1
            gaps.append({
                "requirement_id": None,
                "kind": "branch_has_no_oracle_content",
                "reason": "This branch's accepted requirement set is empty -- every Step3 candidate for "
                          "this Then was judged not relevant. No mechanical, semantic_judge, or gap route "
                          "exists because there is no requirement to route at all.",
            })
        for requirement in branch["requirements"]:
            rid = requirement["requirement_id"]
            expectation = expected[rid]
            evaluator_contract = evaluator_by_requirement[rid]
            binding = binding_by_requirement[rid]
            for item in (requirement, expectation, evaluator_contract, binding):
                _equal(item["branch_id"], branch_id, "requirement branch")
            _equal(evaluator_contract["source_requirement_fingerprint"], requirement["requirement_fingerprint"], "evaluator contract requirement")
            _equal(expectation["source_requirement_fingerprint"], requirement["requirement_fingerprint"], "expected requirement")
            _equal(binding["source_requirement_fingerprint"], requirement["requirement_fingerprint"], "binding requirement")
            route = _route_for(rid, evaluator_contract, binding, semantic_judge_by_id)
            route_counts[route["kind"]] += 1
            checks.append({
                "requirement": deepcopy(requirement),
                "expectation": deepcopy(expectation),
                "effective_route": route,
                "runtime_ready": False,
                "runtime_validated": False,
            })
            if route["kind"] == "known_gap":
                gap_count += 1
                gaps.append({"requirement_id": rid, **deepcopy(route)})

        # Section 123: a branch whose accepted requirement set is NOT empty, and
        # whose every requirement routed to "non_decisive" (a confirmed-correct
        # terminal state, not a gap -- see _route_for's reason text above and
        # docs/oracle_requirement_pipeline_v0_7.md section 32), still compiles to
        # zero real executable oracle content: no route here is ever "mechanical"
        # or "semantic_judge", so generic_tau_v2_bound_plan_adapter_v1.py's checks
        # list (docs/agentcoveragetesting_reuse_log.md section 117) is empty for
        # this branch downstream -- exactly the same "nothing a real runtime
        # execution could ever check" fact branch_has_no_oracle_content reports
        # above, but this shape was previously invisible here (gaps stayed [],
        # assembly_status stayed "assembled") because no single route in it is
        # individually "known_gap". Distinct kind from branch_has_no_oracle_content:
        # this is NOT "Step2 produced nothing" (branch["requirements"] is real and
        # non-empty) -- it is "every real route is a confirmed-correct terminal
        # state that happens to carry zero executable content". Never fires
        # alongside a real known_gap route (that already makes `gaps` non-empty on
        # its own, above) or when `checks` is empty (branch_has_no_oracle_content
        # above already covers that case).
        if checks and all(c["effective_route"]["kind"] == "non_decisive" for c in checks):
            gap_count += 1
            gaps.append({
                "requirement_id": None,
                "kind": "branch_has_only_non_decisive_content",
                "reason": "This branch's accepted requirement set is not empty, and every one of its "
                          "requirements' effective_route is \"non_decisive\" -- a confirmed-correct "
                          "terminal state, not an open gap (see docs/oracle_requirement_pipeline_v0_7.md "
                          "section 32). But that also means none of them is \"mechanical\" or "
                          "\"semantic_judge\", so this branch compiles to zero real executable oracle "
                          "content downstream (see docs/agentcoveragetesting_reuse_log.md section 117's "
                          "not-scriptable marker, and section 123). This is not a defect to close and no "
                          "route here should be reclassified -- it is an honest signal, at this Step 5 "
                          "layer, that this branch currently has nothing a real runtime execution could "
                          "ever check.",
            })

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
        # Section 123: gate this specific instruction on a genuinely closable gap
        # (branch_has_no_oracle_content, or a real per-requirement known_gap route)
        # being present -- NOT merely on `gaps` being non-empty. Its wording ("no
        # route exists for this requirement; never infer a missing rule") would be
        # false for a branch whose only gaps entry is the new
        # "branch_has_only_non_decisive_content" marker above: every one of those
        # routes genuinely DOES exist and is already confirmed correct, there is no
        # missing rule to infer and nothing to "close" -- attaching this instruction
        # to that case would fabricate a false "unresolved" signal. Every branch
        # whose gaps population predates this section (branch_has_no_oracle_content
        # and/or a real known_gap route) is completely unaffected: this condition is
        # true for them exactly when `gaps` was already non-empty before, so this is
        # a pure narrowing for the one new kind, not a behavior change for anything
        # that existed before section 123.
        if any(g["kind"] != "branch_has_only_non_decisive_content" for g in gaps):
            work.append({"code": "close_known_gap_before_evaluation", "owner_step": 9,
                        "instruction": "no route exists for this requirement; never infer a missing rule or default to pass"})
        if any(c["effective_route"]["kind"] == "semantic_judge" for c in checks):
            work.append({"code": "run_semantic_judge_against_real_execution", "owner_step": 8,
                        "instruction": "profile-level calibration exists (see docs/oracle_requirement_pipeline_v0_7.md section 33), "
                                       "this specific requirement has never been judged against a real trajectory"})

        contract = {
            "branch_test_contract_id": f"{branch_id}::V5BTC02", "branch_id": branch_id,
            "spec_id": gw["spec_id"], "gwt": deepcopy(gw["source_context"]["gwt"]),
            "source_contract": {"fields": deepcopy(source_by_id[branch_id]["source_fields"]),
                                "references": deepcopy(gw["source_refs"])},
            "given_contract": deepcopy(gw["given_requirements"]),
            "when_contract": deepcopy(gw["trigger_requirements"]),
            "user_input_contract": deepcopy(gw["user_input_requirements"]),
            "fixture_contract": deepcopy(fixture),
            "oracle_contract": {"checks": checks, "known_gaps": gaps},
            "source_assembly": deepcopy(gw),
            "preparation_gates": {
                "given_when": "assembled_not_runtime_verified",
                "fixture": "candidate_pool_unverified" if lookup["matches"] else "binding_or_construction_required",
                "oracle": "prepared_with_known_gaps" if gaps else "prepared",
            },
            "assembly_status": "assembled_with_known_gaps" if gaps else "assembled",
            "planning_handoff": "eligible_with_explicit_obligations",
            "required_work": work,
            "runtime_ready": False, "runtime_validated": False, "agent_verdict": None,
        }
        contracts.append(_seal(contract, "branch_test_contract_fingerprint"))

    result = {
        "schema_version": SCHEMA,
        "scope": "full live Step3/4 oracle pipeline (155 branches); assembly, not runnable cases or completed evaluation",
        "source_fingerprints": {
            "intake": intake["intake_fingerprint"], "scope": scope["scope_fingerprint"],
            "assembly": assembly["assembly_set_fingerprint"], "accepted": accepted["accepted_set_fingerprint"],
            "expectations": expectations["expectation_set_fingerprint"],
            "bindings": bindings["binding_set_fingerprint"],
            "evaluator_contracts": evaluator_contracts["evaluator_contract_set_fingerprint"],
            "semantic_judge_contracts": semantic_judge_contracts["semantic_judge_contract_set_fingerprint"],
        },
        "contracts": contracts,
        "source_deferred_branch_ids": deepcopy(assembly["deferred_branch_ids"]),
        # v1's Step7 package compile (compile_package, v5_step7_package_v1.py)
        # reads this field as a pure pass-through into its own result summary.
        # v1's contracts were a subset admitted from a wider Step5 pool, with
        # some branches deliberately excluded from the 6-branch pilot; v2 IS
        # the full 155-branch corpus with no wider pool and no exclusions, so
        # this is always empty here -- added for schema compatibility with
        # the reused compile_package, not because v2 has any excluded scope.
        "out_of_pilot_branch_ids": [],
        "summary": {
            "branch_count": len(contracts), "requirement_count": len(requirements),
            "assembly_status_counts": dict(Counter(c["assembly_status"] for c in contracts)),
            "route_counts": dict(route_counts),
            "known_gap_count": gap_count,
            "source_deferred_count": len(assembly["deferred_branch_ids"]),
            "runtime_ready_count": 0, "external_calls": 0, "target_agent_calls": 0,
        },
        "downstream_interface": {"requires_step6_adapter": True,
            "legacy_driver_accepts_this_schema": False,
            "reason": "v2 preserves unverified Given/When and live effective routes; never relabel as legacy ready_for_driver"},
        "runtime_dispatch_enabled": False,
    }
    return _seal(result, "contract_set_fingerprint")


def render_contracts(result):
    lines = ["# Step 5 Test Contract Summary (v2, full live Step3/4)", "",
             "This only assembles preparation requirements; it does not mean data has been bound, conditions hold, or the Agent has passed the test.", "",
             "| Branch | Checks | Data candidates | Judgment route | Gaps |", "| --- | ---: | ---: | --- | ---: |"]
    for c in result["contracts"]:
        oracle = c["oracle_contract"]
        routes = ", ".join(x["effective_route"]["kind"] for x in oracle["checks"])
        lines.append(f"| {c['branch_id']} | {len(oracle['checks'])} | {c['fixture_contract']['lookup']['n_matches']} | {routes} | {len(oracle['known_gaps'])} |")
    lines += ["", f"Total: {result['summary']['branch_count']} branches, {result['summary']['requirement_count']} requirements; "
              f"judgment route distribution: {result['summary']['route_counts']}; known gaps: {result['summary']['known_gap_count']}.",
              "", "All branches still need subsequent planning, data and user-input binding, Given/When verification, and trajectory evaluation.",
              "The `semantic_judge` route has completed spec compilation and profile-level calibration, but has never been asked for an answer on a real execution.",
              "The old Driver does not consume this schema directly; the next step is the Step 6 interface adaptation.", ""]
    return "\n".join(lines)
