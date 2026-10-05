"""Domain-agnostic adapter: turns a generic_tau_v2_fixture_binder_v1.bind_v2_branch
result into the GenericBoundDriverPlan v0.1 shape generic_tau_online_v1.py's Step8
execution layer already expects (see docs/agentcoveragetesting_reuse_log.md section
55). This is the missing piece between "Step7 binding done" (sections 52-55) and
"ready to actually run" -- generic_tau_online_v1.py itself needed no new code (it was
already made domain-parametrized in section 51); this module supplies its real input.

Per branch, real oracle checks are pulled from the domain's own persisted Step4/5
compiler output (outputs/oracle_evaluator_contracts_v0_1_{domain},
outputs/runtime_observation_bindings_v0_1_{domain}) by binding_id -- the exact same
real artifacts sections 52-53's pilots used, not re-derived or approximated. Only
route_kind in ("mechanical", "semantic_judge") checks are carried over: non_decisive
and known_gap routes carry no real evaluator program at all (see
v5_branch_test_contract_v2.py's _route_for), so there is nothing honest to attach.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256

BOUND_PLAN_VERSION = "agentspectesting.generic-bound-driver-plan/v0.1"
ADAPTER_VERSION = "tau-v2-generic-bound-plan-adapter/v0.1"
DEFAULT_CONFIRMATION_MESSAGE = "Yes, I confirm. Please proceed."


class GenericV2AdapterError(ValueError):
    """Raised when a real oracle check referenced by the v2 plan cannot be found in
    the domain's persisted Step4/5 output -- never fabricated."""


def _evaluator_contracts_by_binding_id(evaluator_contract_set: Mapping[str, Any], branch_id: str) -> dict[str, Any]:
    branch = next(
        (b for b in evaluator_contract_set.get("branches") or [] if b.get("branch_id") == branch_id), None
    )
    if branch is None:
        return {}
    return {c["binding_id"]: c for c in branch.get("evaluator_contracts") or []}


def _runtime_bindings_by_binding_id(runtime_binding_set: Mapping[str, Any], branch_id: str) -> dict[str, Any]:
    branch = next(
        (b for b in runtime_binding_set.get("branches") or [] if b.get("branch_id") == branch_id), None
    )
    if branch is None:
        return {}
    return {b["binding_id"]: b for b in branch.get("bindings") or []}


def _referenced_tool_names(value: Any) -> set[str]:
    """Every real tool_name mentioned anywhere inside a runtime_observation_
    binding's own runtime_binding structure (a plain event_filter's own
    field_equals.tool_name, or a temporal_relation's own nested left_event/
    right_event event_filter, or an exact_tool_names/verified_tool_names
    list) -- a best-effort, tool-name-independent walk (docs/
    agentcoveragetesting_reuse_log.md section 132.2/135, task_5d41846b),
    never assumes a single fixed shape."""
    names: set[str] = set()
    if isinstance(value, Mapping):
        for key, sub in value.items():
            if key == "tool_name" and isinstance(sub, str):
                names.add(sub)
            elif key in ("exact_tool_names", "verified_tool_names") and isinstance(sub, list):
                names.update(x for x in sub if isinstance(x, str))
            else:
                names |= _referenced_tool_names(sub)
    elif isinstance(value, list):
        for item in value:
            names |= _referenced_tool_names(item)
    return names


def _program_shape_key(program: Mapping[str, Any]) -> tuple[Any, Any] | None:
    """A real, tool-name-independent fingerprint of an evaluator program's own
    shape (evaluator_kind + predicate_kind) -- used only to detect two
    checks that are real STRUCTURAL TWINS of each other (the exact same
    real check, aimed at a different real candidate tool), never anything
    about which specific tool either one observes.

    Deliberately returns None (never a usable shape key) for anything other
    than a real "all_matches_predicate" program with a real, named
    predicate_kind -- a real, confirmed false-positive caught while
    building this fix (docs/agentcoveragetesting_reuse_log.md section
    132.2/135, task_5d41846b): a bare "match_cardinality"/"present" check
    (no predicate at all -- just "was this tool called") is FAR too
    generic a shape to trust as twin evidence; airline_090_order#b0's own
    real OR01 (cancel_reservation present) and OR02 (get_reservation_
    details present) share this exact bare shape but are genuinely NOT
    alternates -- OR04's own real precedes relation ties get_reservation_
    details (and get_user_details) as real, required PREREQUISITES of
    cancel_reservation, not an alternate strategy for it (section
    132.2b's own real, confirmed informative-signal finding for this
    branch). Grouping them as OR-alternates would have been a real,
    dangerous false-pass regression -- caught and fixed before landing,
    not a guess."""
    if program.get("evaluator_kind") != "all_matches_predicate":
        return None
    predicate_kind = (program.get("predicate") or {}).get("predicate_kind")
    if predicate_kind is None:
        return None
    return (program.get("evaluator_kind"), predicate_kind)


def _is_route_obligation(check: Mapping[str, Any]) -> bool:
    """Section 190: True only for a check that REQUIRES an observation on its
    route (expected_observation.requires_observation is true and the operator
    is not "absent") -- the only kind of check for which "route A or route B"
    is meaningful. Read from the evaluator contract, falling back to the
    runtime binding (both carry the same expected_observation)."""
    expected = (check.get("evaluator_contract") or {}).get("expected_observation") or (
        (check.get("runtime_observation_binding") or {}).get("expected_observation") or {}
    )
    return bool(expected.get("requires_observation")) and expected.get("operator") != "absent"


def _compute_alternate_route_groups(plan: Mapping[str, Any], checks: list[dict[str, Any]]) -> list[list[str]]:
    """docs/agentcoveragetesting_reuse_log.md section 132.2/135 (real bug,
    task_5d41846b): sections 62.1-62.2 established that a branch whose real
    tool_names_from_effective_routes lists more than one candidate tool can
    be a real "alternate route" shape (only one candidate tool is ever
    really exercised in a live conversation -- e.g. book_reservation vs
    update_reservation_flights, two real strategies for the same
    observable effect). This was previously only ever used by the DRIVER
    side (bundle construction, anchor-first-merge, this same module's
    sibling airline/retail bundle resolvers). The ORACLE side's own
    aggregation (generic_tau_online_v1.py's evaluate_generic_mechanical_
    oracle) never grouped checks this way at all, so a check requiring
    observation of the candidate tool that was NOT the one really
    exercised (e.g. airline_049_arg#b0's own OR02, requiring update_
    reservation_flights, when the live agent genuinely used book_
    reservation instead) hard-failed the whole branch even when every
    check on the REALLY-exercised route genuinely passed.

    Real, precise (never over-broad) detection: tool_names_from_effective_
    routes having >1 candidate is NOT by itself sufficient signal -- a
    real, exhaustive corpus check found several airline branches sharing
    this same >1-candidate shape that are genuinely SEQUENCES, not
    alternates (e.g. airline_084/085/090_order#b0: a real temporal_
    relation/"precedes" check ties the two candidate tools together,
    meaning both really are required, in order -- grouping these as OR-
    alternates would be a real, dangerous false-pass regression). Real,
    confirmed-safe signal instead: two checks are only ever treated as
    real alternates of each other when they are genuine STRUCTURAL TWINS
    -- the exact same real evaluator_kind/predicate_kind, each observing a
    DIFFERENT one of the branch's own real candidate tools (airline_049's
    OR01/OR02 both real flights_chronologically_feasible checks;
    airline_082_norm#b0/airline_087_state#b0's own real OR01/OR02 pairs
    share this same twin shape too, real, exhaustive corpus check -- a
    real, generalizable fix, not a one-off). A candidate tool confirmed
    this way becomes a real "anchor" for its OWN group; any OTHER check
    (not itself a twin) that references EXACTLY that one anchor tool (and
    no other) joins the SAME group (e.g. airline_049's own OR03, a plain
    "book_reservation was called" presence check, joins OR01's book_
    reservation group). A branch with fewer than 2 confirmed anchor tools
    gets an empty groups list -- current, unchanged AND-required behavior
    for every other real branch in the corpus (byte-identical for every
    branch with <=1 candidate tool, and for every >1-candidate branch
    that isn't a real, confirmed twin shape -- e.g. 084/085/090_order#b0
    above)."""
    candidate_tools = plan["observation_plan"].get("tool_names_from_effective_routes") or []
    if len(candidate_tools) < 2:
        return []
    candidate_set = set(candidate_tools)

    check_tools: dict[str, set[str]] = {}
    check_shape: dict[str, tuple[Any, Any] | None] = {}
    for check in checks:
        if not _is_route_obligation(check):
            # Section 190 (task_6280c6de): a prohibition (requires_observation
            # False, or an "absent" operator) is never a route to choose
            # between -- it must hold on EVERY route. Grouping two of them
            # ORed them: the route the agent never took has 0 matches and
            # passes vacuously, so the branch could only fail if the agent
            # violated the rule on every route at once (airline_082_norm#b0:
            # update_reservation_baggages(total=1) failed OR02 yet the branch
            # passed). Such checks stay ungrouped, i.e. AND-required like
            # every other ungrouped check; only obligations ("the agent must
            # do X by route A or route B", airline_049_arg#b0) are grouped.
            continue
        referenced = _referenced_tool_names(check["runtime_observation_binding"]) & candidate_set
        check_tools[check["binding_id"]] = referenced
        check_shape[check["binding_id"]] = _program_shape_key(check["evaluator_contract"]["program"])

    shape_to_tools: dict[tuple[Any, Any], set[str]] = {}
    for binding_id, tools in check_tools.items():
        if len(tools) != 1 or check_shape[binding_id] is None:
            continue
        tool = next(iter(tools))
        shape_to_tools.setdefault(check_shape[binding_id], set()).add(tool)
    anchor_tools = {tool for tools in shape_to_tools.values() if len(tools) >= 2 for tool in tools}
    if len(anchor_tools) < 2:
        return []

    groups: dict[str, list[str]] = {tool: [] for tool in sorted(anchor_tools)}
    for binding_id, tools in check_tools.items():
        if len(tools) != 1:
            continue
        tool = next(iter(tools))
        if tool in groups:
            groups[tool].append(binding_id)
    return [members for members in groups.values() if members]


def build_generic_bound_driver_plan(
    plan: Mapping[str, Any],
    bind_result: Mapping[str, Any],
    evaluator_contract_set: Mapping[str, Any],
    runtime_binding_set: Mapping[str, Any],
    *,
    confirmation_message: str = DEFAULT_CONFIRMATION_MESSAGE,
) -> dict[str, Any]:
    """Build one real GenericBoundDriverPlan v0.1 from a successful bind_v2_branch
    result. Raises GenericV2AdapterError if a mechanical/semantic_judge check the plan
    itself declares has no matching real evaluator_contract/runtime_observation_binding
    in the supplied Step4/5 output -- this would mean the two artifacts are out of
    sync, not something to paper over."""

    branch_id = plan["branch_id"]
    contracts_by_id = _evaluator_contracts_by_binding_id(evaluator_contract_set, branch_id)
    bindings_by_id = _runtime_bindings_by_binding_id(runtime_binding_set, branch_id)

    checks = []
    skipped_no_binding_id = []
    for event in plan["observation_plan"]["events"]:
        route_kind = event["route_kind"]
        if route_kind not in ("mechanical", "semantic_judge"):
            continue
        binding_id = event.get("binding_id")
        if binding_id is None:
            # A real, different observation shape (e.g. a whole-stream co-occurrence
            # check, event.get("co_occurrence_forbidden")) that this generic adapter
            # does not know how to attach an evaluator_contract/runtime_observation_
            # binding to at all -- honestly skipped, not a crash and not fabricated.
            skipped_no_binding_id.append(event.get("requirement_id"))
            continue
        evaluator_contract = contracts_by_id.get(binding_id)
        runtime_binding = bindings_by_id.get(binding_id)
        if evaluator_contract is None or runtime_binding is None:
            raise GenericV2AdapterError(
                f"{branch_id}: no real evaluator_contract/runtime_observation_binding "
                f"for binding_id {binding_id!r} in the supplied Step4/5 output"
            )
        checks.append(
            {
                "binding_id": binding_id,
                "evaluation_mode": route_kind,
                "evaluator_contract": deepcopy(evaluator_contract),
                "runtime_observation_binding": deepcopy(runtime_binding),
            }
        )

    driver_bindings = dict(bind_result["driver_bindings"])
    if not checks:
        # docs/agentcoveragetesting_reuse_log.md section 206: a Then that
        # compiled to no decidable check gets the LLM-judged check configured
        # for it (configs/llm_oracle_uncompilable_thens_v0_1.json); a branch
        # not listed there stays not_scriptable exactly as before.
        from agentest.driver.llm_oracle_extension_v1 import llm_oracle_checks

        checks = llm_oracle_checks(branch_id)
    if checks:
        generation_readiness = {
            "status": "ready_for_baseline_execution",
            "concrete_fixture_bound": True,
            "surface_prompt_generated": True,
            "runtime_execution_allowed": True,
        }
    else:
        # Real, confirmed data (docs/agentcoveragetesting_reuse_log.md section 117):
        # a branch whose real compiled `checks` list above ends up empty -- either
        # because its Step2 accepted requirement set was itself empty
        # (v5_branch_test_contract_v2.py's "branch_has_no_oracle_content" known_gap),
        # or because every real requirement it does have routed to a non-mechanical,
        # non-semantic_judge kind (known_gap/non_decisive) -- has nothing a real
        # runtime execution could ever check. Before this fix, `runtime_execution_
        # allowed` was unconditionally hardcoded True for every branch regardless of
        # `checks`, so scripts/run_generic_tau_baselines_v0_1.py would launch a real,
        # full online conversation against it and it would vacuously "pass" (nothing
        # was ever checked) -- real API budget burned for zero signal. This branch is
        # a pure additive fix: every branch that DOES have >=1 real check (the `if
        # checks:` branch above) keeps the exact prior "ready_for_baseline_execution"/
        # True behavior, byte-for-byte unchanged.
        generation_readiness = {
            "status": "not_scriptable_deterministic_driver",
            "concrete_fixture_bound": True,
            "surface_prompt_generated": True,
            "runtime_execution_allowed": False,
        }

    bound = {
        "schema_version": BOUND_PLAN_VERSION,
        "binder_version": ADAPTER_VERSION,
        "bound_driver_plan_id": plan["driver_plan_id"] + "::BOUND01",
        "source_driver_plan_id": plan["driver_plan_id"],
        "source_driver_plan_fingerprint": plan["driver_plan_fingerprint"],
        "source_branch_id": branch_id,
        "object_bindings": driver_bindings,
        # section 63: real supplementary facts beyond object_bindings -- what
        # dialogue_contract.answer_agent_fact_requests_from below already declares
        # as a real source, but which the v2 pipeline never actually populated until
        # now. Shape matches the v1 lineage's own established
        # operation_argument_fact_bundle convention ({tool_name, arguments}, see
        # generic_tau_airline_v1.py and tests/test_generic_tau_online_v1.py) for
        # schema consistency -- object_bindings stays the oracle-verified,
        # authoritative source; this is a wider, best-effort real fact set (e.g. a
        # customer's real full_name when only their dob is required_driver_binding
        # -verified) for a live conversation's other real questions.
        "operation_argument_fact_bundle": {
            # section 193: a binder that rebinds a branch to one specific
            # route may name its tool (bind_result["operation_tool_name"]);
            # absent for every other branch, which keeps the first route's.
            "tool_name": bind_result.get("operation_tool_name")
            or next(iter(plan["observation_plan"].get("tool_names_from_effective_routes") or []), None),
            "arguments": deepcopy(dict(bind_result.get("known_fact_bundle") or {})),
        },
        "fixture_identity": {"fixture_source": bind_result.get("fixture_source", "unknown")},
        "initial_state_patch": {
            "agent_data": deepcopy(bind_result.get("state_patch")) if bind_result.get("state_patch") else None,
            "user_data": None,
        },
        "dialogue_contract": {
            "initial_user_message": bind_result["canonical_request"],
            "confirmation_message": confirmation_message,
            "answer_agent_fact_requests_from": "operation_argument_fact_bundle_and_object_bindings_only",
            "driver_action_kind_for_confirmation": "confirm_operation",
        },
        "oracle_scope_values": deepcopy(driver_bindings),
        "oracle_plan": {
            "checks": checks,
            "skipped_requirement_ids_no_binding_id": skipped_no_binding_id,
            # docs/agentcoveragetesting_reuse_log.md section 132.2/135
            # (task_5d41846b): additive-only -- an empty list for every
            # branch that isn't a real, confirmed alternate-route shape
            # (see _compute_alternate_route_groups), so this changes
            # nothing for any existing branch/domain unless it's real,
            # positively detected here.
            "alternate_route_groups": _compute_alternate_route_groups(plan, checks),
        },
        "binding_checks": {
            "all_active_given_conditions_true": True,
            "required_driver_bindings_complete": True,
            "oracle_not_used_as_surface_text": True,
            "llm_calls": 0,
        },
        "generation_readiness": generation_readiness,
    }
    # docs/agentcoveragetesting_reuse_log.md section 182 (task_e9deccb5): the
    # Step6 plan schedules "realize_conversation_prerequisites" for Step8, but
    # the bound plan never carried any: a Given clause that is realizable only
    # in conversation (e.g. "the user chooses more than one payment method")
    # was simply never said. A domain binder that realizes such a clause from
    # its own real bound values returns it as bind_result["conversation_
    # requirements"]; it rides along here for GenericDeterministicTauUser.
    # Additive-only: the key is emitted ONLY when a binder supplies a non-empty
    # list, so every bound plan of every binder that never returns it
    # (telecom, airline, and every other retail branch) is byte-identical.
    conversation_requirements = bind_result.get("conversation_requirements")
    if conversation_requirements:
        bound["dialogue_contract"]["conversation_requirements"] = deepcopy(list(conversation_requirements))
    bound["bound_driver_plan_fingerprint"] = content_sha256(bound)
    return bound
