"""Extends the frozen semantic_judge_contract_v1 so it can run against the
extension layer's runtime observation bindings / Oracle evaluator contracts
(runtime_observation_binding_extension_v1.py / oracle_evaluator_contract_extension_v1.py),
instead of only the small, frozen-only, 2-requirement proof-of-concept scope
it was originally calibrated and run against.

Never edits the frozen file. Reuses every frozen helper (rule matching,
program construction, contract-set validation) unchanged -- the only real
difference from compile_semantic_judge_contracts is which
binding-set/evaluator-contract-set VALIDATORS it calls and how it establishes
lineage closure between them, for the same reason
runtime_evaluation_extension_v1.py had to diverge from the frozen runtime
evaluator (see docs/oracle_requirement_pipeline_v0_7.md section 28.7): the
extension compilers' output carries EXTENDED_SET_VERSION schema_versions and
does not populate source_binding_set_fingerprint the way the frozen compiler
does, so the frozen validate_runtime_observation_binding_set /
validate_oracle_evaluator_contract_set reject it outright on schema_version,
and even a hand-patched schema_version would still fail the frozen
fingerprint cross-check. This module validates with the extension layer's own
validators and establishes lineage closure the same way
runtime_evaluation_extension_v1.py does: via the population check the frozen
function's per-branch loop already performs (every accepted branch_id must
appear in both the binding set and the evaluator contract set), not via a
cross-set fingerprint field the extension compilers never carry forward.
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .oracle_evaluator_contract_extension_v1 import (
    validate_oracle_evaluator_contract_set_extended,
)
from .oracle_requirement_acceptance_v1 import validate_accepted_oracle_requirement_set
from .runtime_observation_binding_extension_v1 import (
    validate_runtime_observation_binding_set_extended,
)
from .semantic_judge_contract_v1 import (
    CONTRACT_SET_VERSION,
    CONTRACT_STATUSES,
    _branch_index,
    _identity_index,
    _program,
    validate_semantic_judge_contract_set,
    validate_semantic_judge_profile,
)
from .then_atomization import ThenAtomizationError


def _program_for_present_claim(rule: Mapping[str, Any]) -> dict[str, Any] | None:
    """v0.2 (docs/agentcoveragetesting_reuse_log.md section 48): the frozen
    _program (semantic_judge_contract_v1.py) only ever builds an "absence of
    violation" aggregation -- scan every assistant message, fail if ANY
    claim is unsupported -- which is the right shape for an operator=="absent"
    claim ("the agent must not say X") but the wrong shape for
    operator=="present" ("the agent must say X"): a real telecom/retail
    requirement like "inform the user that a payment request has been sent"
    needs "at least one claim satisfies X and is supported", not "no claim is
    unsupported" (a trajectory with zero relevant messages at all would
    vacuously pass the frozen aggregation, which is exactly backwards for a
    "must say X" claim). Scoped to claim_source_support only -- there is no
    real telecom/retail candidate yet asking for a present-operator
    subjective_expression judgment, and that judge_kind's real semantics
    (is this phrasing marked as subjective) do not obviously generalize to a
    "present" reading the same way, so this deliberately returns None rather
    than guessing at one.
    """
    if rule["judge_kind"] != "claim_source_support":
        return None
    return {
        "judge_kind": "claim_source_support",
        "target_selector": {
            "event_kind": "assistant_message",
            "selection": "all_non_empty_messages",
        },
        "source_policy": {
            "channels": deepcopy(rule["allowed_source_channels"]),
            "event_cutoff": "strictly_before_target_message",
            "exclude_prior_assistant_messages": True,
        },
        "model_stages": ["claim_atomization", "claim_support"],
        "aggregation": {
            "violation_when": "no_claim_both_matches_the_requirement_and_is_supported",
            "incomplete_when": "any_model_result_is_invalid_or_insufficient",
            "pass_when": "at_least_one_claim_matches_the_requirement_and_is_supported",
        },
    }


def compile_semantic_judge_contracts_extended(
    accepted_requirement_set: Mapping[str, Any],
    runtime_binding_set: Mapping[str, Any],
    evaluator_contract_set: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = validate_accepted_oracle_requirement_set(accepted_requirement_set)
    bindings = validate_runtime_observation_binding_set_extended(runtime_binding_set)
    evaluators = validate_oracle_evaluator_contract_set_extended(evaluator_contract_set)
    profile = validate_semantic_judge_profile(profile)

    binding_branches = _branch_index(bindings)
    evaluator_branches = _branch_index(evaluators)
    records = []
    counts: Counter[str] = Counter()
    branch_ids = set()
    for accepted_branch in accepted.get("branches") or []:
        branch_id = accepted_branch["branch_id"]
        if branch_id not in binding_branches or branch_id not in evaluator_branches:
            raise ThenAtomizationError("semantic judge inputs have different branch sets")
        binding_index = _identity_index(
            binding_branches[branch_id], "bindings", "requirement_id"
        )
        evaluator_index = _identity_index(
            evaluator_branches[branch_id], "evaluator_contracts", "requirement_id"
        )
        for requirement in accepted_branch.get("requirements") or []:
            requirement_id = requirement["requirement_id"]
            binding = binding_index.get(requirement_id)
            evaluator = evaluator_index.get(requirement_id)
            if binding is None or evaluator is None:
                raise ThenAtomizationError("semantic judge requirement lineage is open")
            if not (
                requirement.get("requirement_type") == "semantic_requirement"
                and binding.get("binding_status") == "semantic_deferred"
                and evaluator.get("evaluator_status") == "deferred"
            ):
                continue
            text = str(requirement.get("requirement_text") or "")
            matches = [
                rule
                for rule in profile["rules"]
                if re.search(rule["requirement_pattern"], text, re.IGNORECASE)
            ]
            diagnostics = []
            status = "ready"
            program = {"judge_kind": "unavailable"}
            matched_rule_id = None
            if len(matches) != 1:
                status = "needs_adjudication"
                diagnostics.append(
                    {
                        "code": "semantic_judge_rule_match_not_unique",
                        "matched_rule_ids": [rule["rule_id"] for rule in matches],
                    }
                )
            elif matches[0]["target_event_kind"] not in (
                binding.get("runtime_binding", {}).get("candidate_event_kinds") or []
            ):
                status = "needs_adjudication"
                diagnostics.append(
                    {"code": "semantic_judge_target_event_not_allowed_by_binding"}
                )
            else:
                operator = binding.get("expected_observation", {}).get("operator")
                if operator == "absent":
                    matched_rule_id = matches[0]["rule_id"]
                    program = _program(matches[0])
                elif operator == "present":
                    # v0.2 (section 48): "must say X" needs "at least one
                    # supported claim", not the frozen "no unsupported claim"
                    # aggregation -- see _program_for_present_claim's own
                    # docstring. Falls back to needs_adjudication (not a
                    # crash, not a silent frozen-shape reuse) for any
                    # judge_kind that function doesn't cover yet.
                    present_program = _program_for_present_claim(matches[0])
                    if present_program is None:
                        status = "needs_adjudication"
                        diagnostics.append(
                            {"code": "semantic_judge_v0_2_present_requires_claim_source_support"}
                        )
                    else:
                        matched_rule_id = matches[0]["rule_id"]
                        program = present_program
                else:
                    status = "needs_adjudication"
                    diagnostics.append({"code": "semantic_judge_v0_1_requires_absence_or_presence"})
            record = {
                "schema_version": "agentspectesting.semantic-judge-contract/v0.1",
                "semantic_judge_contract_id": f"{evaluator['evaluator_contract_id']}::SJ01",
                "branch_id": branch_id,
                "requirement_id": requirement_id,
                "binding_id": binding["binding_id"],
                "evaluator_contract_id": evaluator["evaluator_contract_id"],
                "contract_status": status,
                "matched_profile_rule_id": matched_rule_id,
                "policy_requirement": text,
                "program": program,
                "diagnostics": diagnostics,
                "source_requirement_fingerprint": requirement["requirement_fingerprint"],
                "source_binding_fingerprint": binding["binding_fingerprint"],
                "source_evaluator_contract_fingerprint": evaluator[
                    "evaluator_contract_fingerprint"
                ],
            }
            record["semantic_judge_contract_fingerprint"] = content_sha256(record)
            records.append(record)
            counts[status] += 1
            branch_ids.add(branch_id)

    result = {
        "schema_version": CONTRACT_SET_VERSION,
        "profile_id": profile["profile_id"],
        "source_accepted_set_fingerprint": accepted["accepted_set_fingerprint"],
        "source_binding_set_fingerprint": bindings["binding_set_fingerprint"],
        "source_evaluator_contract_set_fingerprint": evaluators[
            "evaluator_contract_set_fingerprint"
        ],
        "contracts": records,
        "summary": {
            "semantic_deferred_requirement_count": len(records),
            "covered_branch_count": len(branch_ids),
            "contract_status_counts": {
                status: counts[status] for status in sorted(CONTRACT_STATUSES)
            },
        },
    }
    result["semantic_judge_contract_set_fingerprint"] = content_sha256(result)
    validate_semantic_judge_contract_set(result)
    return result
