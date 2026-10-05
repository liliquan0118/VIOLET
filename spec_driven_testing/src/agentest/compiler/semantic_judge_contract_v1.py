"""Compile deferred semantic Oracle requirements into narrow judge contracts.

This module is fully mechanical.  It selects a known semantic judge grammar;
it neither calls a model nor attempts to decide the runtime behavior itself.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .artifacts import content_sha256
from .oracle_evaluator_contract_v1 import validate_oracle_evaluator_contract_set
from .oracle_requirement_acceptance_v1 import (
    validate_accepted_oracle_requirement_set,
)
from .runtime_observation_binding_v1 import (
    validate_runtime_observation_binding_set,
)
from .then_atomization import ThenAtomizationError


PROFILE_VERSION = "agentspectesting.semantic-judge-profile/v0.1"
CONTRACT_SET_VERSION = "agentspectesting.semantic-judge-contract-set/v0.1"
CONTRACT_VERSION = "agentspectesting.semantic-judge-contract/v0.1"
JUDGE_KINDS = frozenset({"claim_source_support", "subjective_expression"})
CONTRACT_STATUSES = frozenset({"ready", "needs_adjudication"})
SOURCE_CHANNELS = frozenset(
    {
        "prior_user_message",
        "prior_tool_result",
        "system_policy",
        "agent_tool_schema",
    }
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def validate_semantic_judge_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = deepcopy(dict(_mapping(value, "$semantic_judge_profile")))
    if profile.get("schema_version") != PROFILE_VERSION:
        raise ThenAtomizationError("unsupported semantic judge profile")
    if set(profile) != {"schema_version", "profile_id", "rules"}:
        raise ThenAtomizationError("semantic judge profile fields are invalid")
    if not isinstance(profile.get("profile_id"), str) or not profile["profile_id"]:
        raise ThenAtomizationError("semantic judge profile_id must be non-empty")
    rules = profile.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ThenAtomizationError("semantic judge profile rules must be non-empty")
    identities = []
    for raw in rules:
        rule = _mapping(raw, "$.rules[]")
        if set(rule) != {
            "rule_id",
            "requirement_pattern",
            "judge_kind",
            "target_event_kind",
            "allowed_source_channels",
        }:
            raise ThenAtomizationError("semantic judge rule fields are invalid")
        if not isinstance(rule.get("rule_id"), str) or not rule["rule_id"]:
            raise ThenAtomizationError("semantic judge rule_id must be non-empty")
        identities.append(rule["rule_id"])
        pattern = rule.get("requirement_pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ThenAtomizationError("semantic judge rule pattern must be non-empty")
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ThenAtomizationError(f"invalid semantic judge rule pattern: {exc}") from exc
        if rule.get("judge_kind") not in JUDGE_KINDS:
            raise ThenAtomizationError("unsupported semantic judge kind")
        if rule.get("target_event_kind") != "assistant_message":
            raise ThenAtomizationError("v0.1 semantic judges require assistant messages")
        channels = rule.get("allowed_source_channels")
        if (
            not isinstance(channels, list)
            or len(channels) != len(set(channels))
            or any(channel not in SOURCE_CHANNELS for channel in channels)
        ):
            raise ThenAtomizationError("semantic judge source channels are invalid")
        if rule["judge_kind"] == "claim_source_support" and not channels:
            raise ThenAtomizationError("claim support judge requires source channels")
        if rule["judge_kind"] == "subjective_expression" and channels:
            raise ThenAtomizationError("subjectivity judge must not receive source channels")
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("semantic judge rule IDs must be unique")
    return profile


def _branch_index(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for raw in document.get("branches") or []:
        branch = _mapping(raw, "$.branches[]")
        branch_id = branch.get("branch_id")
        if not isinstance(branch_id, str) or not branch_id or branch_id in result:
            raise ThenAtomizationError("branch IDs must be unique non-empty strings")
        result[branch_id] = branch
    return result


def _identity_index(
    branch: Mapping[str, Any], collection: str, key: str
) -> dict[str, Mapping[str, Any]]:
    result = {}
    for raw in branch.get(collection) or []:
        item = _mapping(raw, f"$.{collection}[]")
        identity = item.get(key)
        if not isinstance(identity, str) or not identity or identity in result:
            raise ThenAtomizationError(f"{collection} identities must be unique")
        result[identity] = item
    return result


def _program(rule: Mapping[str, Any]) -> dict[str, Any]:
    if rule["judge_kind"] == "claim_source_support":
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
                "violation_when": "any_claim_is_unsupported",
                "incomplete_when": "any_model_result_is_invalid_or_insufficient",
                "pass_when": "all_claims_are_supported_or_no_claims_exist",
            },
        }
    return {
        "judge_kind": "subjective_expression",
        "target_selector": {
            "event_kind": "assistant_message",
            "selection": "all_non_empty_messages",
        },
        "model_stages": ["subjectivity_judgment"],
        "aggregation": {
            "violation_when": "any_message_contains_subjective_expression",
            "incomplete_when": "any_model_result_is_invalid_or_insufficient",
            "pass_when": "all_messages_have_no_subjective_expression",
        },
    }


def compile_semantic_judge_contracts(
    accepted_requirement_set: Mapping[str, Any],
    runtime_binding_set: Mapping[str, Any],
    evaluator_contract_set: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = validate_accepted_oracle_requirement_set(accepted_requirement_set)
    bindings = validate_runtime_observation_binding_set(runtime_binding_set)
    evaluators = validate_oracle_evaluator_contract_set(evaluator_contract_set)
    profile = validate_semantic_judge_profile(profile)
    if evaluators.get("source_binding_set_fingerprint") != bindings.get(
        "binding_set_fingerprint"
    ):
        raise ThenAtomizationError("evaluator and runtime binding sets are not closed")

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
            elif binding.get("expected_observation", {}).get("operator") != "absent":
                status = "needs_adjudication"
                diagnostics.append({"code": "semantic_judge_v0_1_requires_absence"})
            else:
                matched_rule_id = matches[0]["rule_id"]
                program = _program(matches[0])
            record = {
                "schema_version": CONTRACT_VERSION,
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
                "source_requirement_fingerprint": requirement[
                    "requirement_fingerprint"
                ],
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


def validate_semantic_judge_contract_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$semantic_judge_contract_set")))
    fingerprint = result.pop("semantic_judge_contract_set_fingerprint", None)
    if result.get("schema_version") != CONTRACT_SET_VERSION or fingerprint != content_sha256(
        result
    ):
        raise ThenAtomizationError("invalid semantic judge contract set")
    identities = []
    counts: Counter[str] = Counter()
    branches = set()
    for raw in result.get("contracts") or []:
        record = deepcopy(dict(_mapping(raw, "$.contracts[]")))
        record_fingerprint = record.pop("semantic_judge_contract_fingerprint", None)
        if record.get("schema_version") != CONTRACT_VERSION or record_fingerprint != content_sha256(
            record
        ):
            raise ThenAtomizationError("invalid semantic judge contract")
        if record.get("contract_status") not in CONTRACT_STATUSES:
            raise ThenAtomizationError("invalid semantic judge contract status")
        if record["contract_status"] == "ready":
            kind = _mapping(record.get("program"), "$.program").get("judge_kind")
            if kind not in JUDGE_KINDS:
                raise ThenAtomizationError("ready semantic judge has invalid program")
        identities.append(record.get("semantic_judge_contract_id"))
        counts[record["contract_status"]] += 1
        branches.add(record.get("branch_id"))
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("semantic judge contract IDs must be unique")
    summary = _mapping(result.get("summary"), "$.summary")
    expected_counts = {
        status: counts[status] for status in sorted(CONTRACT_STATUSES)
    }
    if summary.get("semantic_deferred_requirement_count") != len(identities):
        raise ThenAtomizationError("semantic judge contract count mismatch")
    if summary.get("covered_branch_count") != len(branches):
        raise ThenAtomizationError("semantic judge branch count mismatch")
    if summary.get("contract_status_counts") != expected_counts:
        raise ThenAtomizationError("semantic judge status counts mismatch")
    result["semantic_judge_contract_set_fingerprint"] = fingerprint
    return result


def compile_semantic_judge_contracts_file(
    *,
    accepted_requirements_path: str | Path,
    runtime_bindings_path: str | Path,
    evaluator_contracts_path: str | Path,
    profile_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    result = compile_semantic_judge_contracts(
        load(accepted_requirements_path),
        load(runtime_bindings_path),
        load(evaluator_contracts_path),
        load(profile_path),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
