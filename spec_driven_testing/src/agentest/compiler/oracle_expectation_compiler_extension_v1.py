"""Extends oracle_expectation_compiler_v1 (not frozen -- not in
scripts/prepare_v5_step4_v0_1.py's COMPILERS drift-check tuple, unlike
runtime_observation_binding_v1.py/oracle_evaluator_contract_v1.py -- but kept
as a separate extension module anyway, for the same auditability reason every
other override in this project lives in its own _extension_v1 module rather
than an in-place edit) with a targeted override for a real bug found in this
round (see docs/oracle_requirement_pipeline_v0_7.md section 25, found
independently by 3 parallel review agents and cross-verified against the real
compiled data for all 155 branches).

The bug: `_observation_expectation`'s catch-all for any requirement_type not
in {tool_argument, tool_argument_constraint, temporal_relation} (i.e.
tool_call) always derives operator="present"/"absent" from the branch's own
normative mode, with no regard for whether this specific tool_call requirement
is the Then's actual polarity subject or just a bare, ancillary observation of
a tool a MORE SPECIFIC sibling requirement (tool_argument/
tool_argument_constraint/temporal_relation/turn_shape_constraint/etc. -- any
non-tool_call kind) already targets on that same tool_name.

Scanning the real, current 155-branch corpus found exactly 33 such
forbidden-mode instances (operator=="absent") where the tool_call requirement
shares its tool_name with a more specific sibling. In every one of the 33
(inspected individually, not sampled), the branch's own WHEN explicitly
requests the exact action this tool performs (e.g. "The user requests to book
a reservation" alongside "The agent must not book a reservation with more
than one travel certificate...") -- a compliant trajectory legitimately calls
this tool, just with compliant arguments (which the sibling requirement is
what actually judges); forcing operator="absent" here is never correct.

This is deliberately narrower than "any tool_call requirement with a same-tool
sibling": required-mode's operator=="present" for the identical structural
shape (83 - 33 = 50 real instances) is NOT touched, since it is not obviously
wrong the same way -- the tool needing to be called for its sibling's
argument-level check to even apply, and requiring that call to actually
happen, is a coherent, likely-correct expectation, not a forced-polarity bug.

Separately: this override does not by itself fix every practical consequence
of such a tool_call requirement having full-tool-parameter scope_constraints
(a real, driver_bindings only ever supplying user_id/reservation_id in
production -- see generic_tau_airline_v1.py -- so evaluate_oracle_contract's
_scope_matches almost always reports missing_driver_bindings for these,
independent of forbidden/required polarity; that produces an "unavailable"
verdict rather than "fail", which runtime_evaluation_v2.py's branch
aggregation turns into branch_verdict "incomplete" rather than "fail"). That
is a distinct, still-open gap this round did not attempt to fix; downgrading
these 33 records to non_decisive sidesteps it for exactly these instances
(a non_decisive expectation compiles to nothing at Step4 at all, so the
scope-matching question never arises for them), but does not address the
gap for any other bound tool_call/tool_argument requirement.

A second, independent override was added later (see
docs/oracle_requirement_pipeline_v0_7.md section 27,
_upgrade_conditional_temporal_relation below): permitted-mode branches whose
When never explicitly requests the permitted action have EVERY requirement
collapsed to operator="non_decisive" by the frozen compiler, including
genuinely checkable temporal_relation "confirm X before doing Y" conditional
obligations -- this upgrades exactly that shape to operator="relation_true"
with requires_observation=False (vacuous pass when the target action never
happens), the temporal_relation analogue of how this module's sibling
tool_argument predicate tables (_COMPENSATION_FORMULA_RULES etc.) already
bypass the same non_decisive collapse.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .oracle_expectation_compiler_v1 import (
    COMPILATION_STATUSES,
    EXPECTATION_VERSION,
    NORMATIVE_MODES,
    OBSERVATION_OPERATORS,
    QUANTIFIERS,
    compile_oracle_expectations,
)
from .oracle_expectation_compiler_v2 import compile_oracle_expectations as compile_oracle_expectations_v5_request
from .then_atomization import ThenAtomizationError


PROGRAM_SOURCE = "extension_v1"


def _upgrade_conditional_temporal_relation(
    expectation: Mapping[str, Any], requirement: Mapping[str, Any]
) -> dict[str, Any] | None:
    """airline_037_arg#b0 / airline_038_arg#b0 (see
    docs/oracle_requirement_pipeline_v0_7.md section 27): both are
    `permitted`-mode branches ("may offer a certificate ... after confirming
    the facts") whose own When never explicitly requests offering
    compensation, so the frozen compiler's permission-resolution logic (the
    "otherwise" branch of oracle_expectation_compiler_v1.py's
    `_observation_expectation`) collapses EVERY requirement in these branches
    to operator="non_decisive" -- including a new `temporal_relation`
    "confirm the facts before offering compensation" candidate added this
    round (see _TEMPORAL_RELATION_FALLBACK_TARGETS in
    oracle_requirement_pipeline_v7.py), which is a genuine, checkable,
    conditional obligation: certificate offers are optional, but WHEN one is
    made, it must be preceded by confirming the facts, exactly the same
    "if X happens, Y must also hold" shape this module already carves out an
    exception for via _COMPENSATION_FORMULA_RULES/_BAGGAGE_REDUCTION_RULES
    (tool_argument requirements bypass non_decisive and get a real predicate
    that vacuously passes when the tool is never called). temporal_relation
    requirements have no equivalent bypass at the CONTRACT layer (the frozen
    evaluator compiler's operator=="relation_true" branch is what routes to
    `all_target_events_have_predecessor`; "non_decisive" hits its unconditional
    "unsupported_expected_operator" catch-all instead), so the fix has to
    happen here, at the EXPECTATION layer, before that operator is ever set.

    Upgrades operator "non_decisive" -> "relation_true" for exactly this
    shape (permitted mode, temporal_relation kind, non_decisive by way of the
    "otherwise" permission-resolution rule) with requires_observation=False --
    critically NOT True: with target_count==0 (the certificate never
    offered), all_target_events_have_predecessor's own passed = (not
    missing_required) and (target_count == matched_count) is True only when
    missing_required is False, i.e. only when requires_observation is False.
    Setting True here would wrongly fail every trajectory that never offers
    compensation at all, which permitted mode explicitly allows.

    Deliberately narrow: only touches temporal_relation requirements whose
    baseline operator is non_decisive via the permission_rule=="otherwise"
    path (never required-mode's "relation_true", already correct; never
    permitted-mode's "aligned_explicit_request" path, which is a different,
    already-decisive resolution)."""
    if requirement.get("requirement_type") != "temporal_relation":
        return None
    if expectation.get("normative_mode") != "permitted":
        return None
    if (expectation.get("derivation") or {}).get("permission_rule") != "otherwise":
        return None
    if expectation["expected_observation"].get("operator") != "non_decisive":
        return None
    record = deepcopy(dict(expectation))
    record.pop("expectation_fingerprint", None)
    record["compilation_status"] = "compiled"
    record["expected_observation"] = {
        "operator": "relation_true",
        "quantifier": "each_target_event",
        "requires_observation": False,
    }
    record["derivation"] = {
        **record["derivation"],
        "reason": (
            "Upgraded by extension_v1: this temporal_relation requirement is "
            "a conditional obligation ('if the target action happens, this "
            "ordering must hold') that permitted mode's own "
            "operator=non_decisive collapse would otherwise discard "
            "entirely -- a vacuous-pass-when-never-triggered relation check "
            "is compiled instead, mirroring how this module's "
            "_COMPENSATION_FORMULA_RULES already handles the analogous "
            "tool_argument case for the same branches. See "
            "docs/oracle_requirement_pipeline_v0_7.md section 27."
        ),
        "program_source": PROGRAM_SOURCE,
    }
    record["expectation_fingerprint"] = content_sha256(record)
    return record


def _non_tool_call_tool_names(requirements: list[Mapping[str, Any]]) -> set[str]:
    names = set()
    for requirement in requirements:
        if requirement.get("requirement_type") == "tool_call":
            continue
        tool_name = (requirement.get("observation_contract") or {}).get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            names.add(tool_name)
    return names


# docs/agentcoveragetesting_reuse_log.md section 194 (task_fd3f3d3b): the
# persisted retail Step3 expectations carried these hand-applied
# expected_observation values since sections 77/78/80/86, and a from-scratch
# recompile silently reverted them. Keyed by requirement_id (each value was
# verified individually in its own section); only expected_observation is
# replaced -- everything else, derivation included, is the baseline's.
_EXPECTED_OBSERVATION_OVERRIDES: dict[str, dict[str, Any]] = {
    # sections 78/80: When is "the agent uses find_user_id_by_name_zip" -- an
    # OPTIONAL action. The Then (argument format) only applies when the agent
    # takes that route; an agent that authenticates by email is compliant.
    # (Same ids as oracle_evaluator_contract_extension_v1's
    # _REQUIRES_OBSERVATION_FALSE_OVERRIDE, which no longer fires for them
    # because this Step3 value is already false.)
    **{
        rid: {"operator": "predicate_true", "quantifier": "all_matching_events", "requires_observation": False}
        for rid in (
            "retail_008_arg#b0::OR01", "retail_009_arg#b0::OR01",
            "retail_010_arg#b0::OR01", "retail_033_arg#b0::OR01",
        )
    },
    # section 77: "must not call X more than once for the same order" is an
    # at-most-N bound (AT_MOST_N_CALL_RULES / call_count_le), not "never call
    # X": the frozen modality mapping's operator=absent forbids even the one
    # legitimate call the branch's own When requests.
    **{
        rid: {"operator": "predicate_true", "quantifier": "all_matching_events", "requires_observation": False}
        for rid in (
            "retail_060_order_modify_pending_order_address#b0v0::OR01",
            "retail_060_order_modify_pending_order_items#b0v0::OR01",
            "retail_060_order_modify_pending_order_payment#b0v0::OR01",
        )
    },
    # section 80 (reuse log section 197): When "The agent calls the
    # get_details_by_id tool", Then "must provide the id parameter as a
    # string when calling get_details_by_id" -- conditional on that route;
    # get_customer_by_id is a functionally identical alternative (tau2
    # telecom tools.py dispatches C-ids to it). Same id as the evaluator
    # layer's _REQUIRES_OBSERVATION_FALSE_OVERRIDE.
    "telecom_004_arg#b0::OR01": {
        "operator": "predicate_true", "quantifier": "all_matching_events", "requires_observation": False,
    },
}

# Reuse log section 197: section 66 recompiled exactly these 17 telecom
# branches (Pattern #2 turn-shape: the 11 telecom_036_order_*; Pattern #3
# alias removal: 041/046/054/080/081; Pattern #4 IFF: 038) with the v5 Step3
# CLI (scripts/compile_v5_oracle_expectations_v0_1.py ->
# oracle_expectation_compiler_v2), which analyses the request against
# branch_context.user_request instead of the When and records that in the
# derivation (request_source/user_request). expected_observation is identical
# to the v1 baseline for every telecom record; only the derivation differs.
# Their persisted Step3 records are that compiler's output, byte for byte.
_V5_USER_REQUEST_BASELINE_BRANCHES = frozenset({
    "telecom_036_order_get_bills_for_customer#b0", "telecom_036_order_get_customer_by_id#b0",
    "telecom_036_order_get_customer_by_name#b0", "telecom_036_order_get_customer_by_phone#b0",
    "telecom_036_order_get_data_usage#b0", "telecom_036_order_get_details_by_id#b0",
    "telecom_036_order_refuel_data#b0", "telecom_036_order_send_payment_request#b0v0",
    "telecom_036_order_send_payment_request#b0v1", "telecom_036_order_suspend_line#b0",
    "telecom_036_order_transfer_to_human_agents#b0",
    "telecom_038_norm#b0", "telecom_041_order#b0", "telecom_046_order#b0", "telecom_054_order#b0",
    "telecom_080_order#b0", "telecom_081_order#b0",
})


def _with_v5_request_baseline_branches(
    baseline: Mapping[str, Any],
    accepted_requirement_set: Mapping[str, Any],
    expectation_policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    """baseline with the _V5_USER_REQUEST_BASELINE_BRANCHES branches taken
    from the v2 compiler (and request_status_counts recounted); unchanged
    when the set contains none of them."""
    if not any(b["branch_id"] in _V5_USER_REQUEST_BASELINE_BRANCHES for b in baseline["branches"]):
        return baseline
    v2 = {b["branch_id"]: b for b in compile_oracle_expectations_v5_request(accepted_requirement_set, expectation_policy)["branches"]}
    result = deepcopy(dict(baseline))
    result["branches"] = [
        deepcopy(v2[b["branch_id"]]) if b["branch_id"] in _V5_USER_REQUEST_BASELINE_BRANCHES else b
        for b in result["branches"]
    ]
    counts = Counter(e["derivation"]["request_analysis"]["status"] for b in result["branches"] for e in b["expectations"])
    result["summary"] = {
        **result["summary"],
        "request_status_counts": dict(sorted(counts.items())),  # as oracle_expectation_compiler_v1 builds it
    }
    return result


def _apply_expected_observation_override(expectation: Mapping[str, Any]) -> dict[str, Any] | None:
    override = _EXPECTED_OBSERVATION_OVERRIDES.get(expectation.get("requirement_id"))
    if override is None or expectation.get("expected_observation") == override:
        return None
    record = deepcopy(dict(expectation))
    record.pop("expectation_fingerprint", None)
    record["expected_observation"] = dict(override)
    record["expectation_fingerprint"] = content_sha256(record)
    return record


def compile_oracle_expectations_extended(
    accepted_requirement_set: Mapping[str, Any],
    expectation_policy: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = _with_v5_request_baseline_branches(
        compile_oracle_expectations(accepted_requirement_set, expectation_policy),
        accepted_requirement_set, expectation_policy,
    )
    requirements_by_id = {
        requirement["requirement_id"]: requirement
        for branch in accepted_requirement_set["branches"]
        for requirement in branch["requirements"]
    }

    status_counts: Counter[str] = Counter()
    normative_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    adjudication_branches: list[str] = []
    branches = []
    for source_branch in baseline["branches"]:
        requirements = [
            requirements_by_id[e["requirement_id"]] for e in source_branch["expectations"]
        ]
        same_tool_siblings = _non_tool_call_tool_names(requirements)
        expectations = []
        for expectation, requirement in zip(source_branch["expectations"], requirements):
            tool_name = (requirement.get("observation_contract") or {}).get("tool_name")
            if (
                requirement.get("requirement_type") == "tool_call"
                and expectation["expected_observation"].get("operator") == "absent"
                and tool_name in same_tool_siblings
            ):
                record = deepcopy(expectation)
                record.pop("expectation_fingerprint", None)
                record["compilation_status"] = "non_decisive"
                record["expected_observation"] = {
                    "operator": "non_decisive",
                    "quantifier": "not_applicable",
                    "requires_observation": False,
                }
                record["derivation"] = {
                    **record["derivation"],
                    "reason": (
                        "Downgraded by extension_v1: this bare tool_call "
                        f"requirement shares its tool_name ({tool_name!r}) with "
                        "a more specific sibling requirement in the same "
                        "branch, and the branch's own When explicitly requests "
                        "this tool's action, so a compliant trajectory "
                        "legitimately calls it -- forcing operator=absent "
                        "('must never be called') is wrong; the sibling "
                        "requirement is what actually judges this call. See "
                        "docs/oracle_requirement_pipeline_v0_7.md section 25."
                    ),
                    "program_source": PROGRAM_SOURCE,
                }
                record["expectation_fingerprint"] = content_sha256(record)
                expectation = record
            upgraded = _upgrade_conditional_temporal_relation(expectation, requirement)
            if upgraded is not None:
                expectation = upgraded
            overridden = _apply_expected_observation_override(expectation)
            if overridden is not None:
                expectation = overridden
            expectations.append(expectation)
            status_counts[expectation["compilation_status"]] += 1
            if expectation.get("normative_mode"):
                normative_counts[expectation["normative_mode"]] += 1
            operator_counts[expectation["expected_observation"]["operator"]] += 1
        branch_needs_adjudication = any(
            item["compilation_status"] == "needs_adjudication" for item in expectations
        )
        if branch_needs_adjudication:
            adjudication_branches.append(source_branch["branch_id"])
        branches.append(
            {
                "branch_id": source_branch["branch_id"],
                "branch_context": deepcopy(source_branch["branch_context"]),
                "expectations": expectations,
                "compilation_status": (
                    # Explicit empty-collection case (docs/agentcoveragetesting_
                    # reuse_log.md section 110), same shape as the analogous fix
                    # in oracle_requirement_acceptance_v1.py: `any()` over an
                    # empty `expectations` list is vacuously False, so a branch
                    # whose accepted requirement set (and therefore its
                    # expectations, which are compiled 1:1 from it) is empty
                    # would otherwise fall through to "compiled" -- a falsely-
                    # healthy status for a branch with zero real expectations.
                    # Checked first on purpose: branch_needs_adjudication and
                    # the non_decisive any() are both derived from
                    # `expectations`, so when it is empty they are always
                    # False and this is a pure addition, not a behavior change,
                    # for every branch with at least one real expectation.
                    "branch_has_no_oracle_content"
                    if not expectations
                    else (
                        "needs_adjudication"
                        if branch_needs_adjudication
                        else (
                            "contains_non_decisive"
                            if any(item["compilation_status"] == "non_decisive" for item in expectations)
                            else "compiled"
                        )
                    )
                ),
            }
        )

    expectation_count = sum(len(branch["expectations"]) for branch in branches)
    result = {
        "schema_version": baseline["schema_version"],
        "source_accepted_set_fingerprint": baseline["source_accepted_set_fingerprint"],
        "expectation_policy_fingerprint": baseline["expectation_policy_fingerprint"],
        "expectation_policy_id": baseline["expectation_policy_id"],
        "branches": branches,
        "summary": {
            "branch_count": len(branches),
            "expectation_count": expectation_count,
            "normative_mode_counts": {
                mode: normative_counts[mode] for mode in sorted(NORMATIVE_MODES)
            },
            "compilation_status_counts": {
                status: status_counts[status] for status in sorted(COMPILATION_STATUSES)
            },
            "observation_operator_counts": {
                operator: operator_counts[operator] for operator in sorted(OBSERVATION_OPERATORS)
            },
            "request_status_counts": baseline["summary"]["request_status_counts"],
            "compiled_branch_count": sum(
                branch["compilation_status"] == "compiled" for branch in branches
            ),
            "adjudication_branch_count": len(adjudication_branches),
        },
        "adjudication_branches": adjudication_branches,
        "next_stage": "runtime_observation_binding",
    }
    result["expectation_set_fingerprint"] = content_sha256(result)
    return result


def validate_oracle_expectation_set_extended(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    fingerprint = result.pop("expectation_set_fingerprint", None)
    if fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid extended oracle expectation set")
    identities = []
    statuses: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    for branch in result.get("branches") or []:
        for raw in branch.get("expectations") or []:
            record = deepcopy(dict(raw))
            record_fingerprint = record.pop("expectation_fingerprint", None)
            if (
                record.get("schema_version") != EXPECTATION_VERSION
                or record_fingerprint != content_sha256(record)
            ):
                raise ThenAtomizationError("invalid extended oracle expectation record")
            if record.get("compilation_status") not in COMPILATION_STATUSES:
                raise ThenAtomizationError("invalid expectation compilation status")
            observation = record.get("expected_observation") or {}
            if observation.get("operator") not in OBSERVATION_OPERATORS:
                raise ThenAtomizationError("invalid expected observation operator")
            if observation.get("quantifier") not in QUANTIFIERS:
                raise ThenAtomizationError("invalid expected observation quantifier")
            identities.append(record.get("expectation_id"))
            statuses[record["compilation_status"]] += 1
            operators[observation["operator"]] += 1
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("extended oracle expectation IDs must be unique")
    summary = result.get("summary") or {}
    if summary.get("expectation_count") != len(identities):
        raise ThenAtomizationError("extended oracle expectation count mismatch")
    result["expectation_set_fingerprint"] = fingerprint
    return result


def compile_oracle_expectations_extended_file(
    *,
    accepted_requirements_path,
    expectation_policy_path,
    output_path,
) -> dict[str, Any]:
    import json
    from pathlib import Path

    accepted = json.loads(Path(accepted_requirements_path).read_text(encoding="utf-8"))
    policy = json.loads(Path(expectation_policy_path).read_text(encoding="utf-8"))
    result = compile_oracle_expectations_extended(accepted, policy)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
