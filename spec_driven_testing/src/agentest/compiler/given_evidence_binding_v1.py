"""Apply narrow binding decisions without pretending they are complete Driver contracts."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .given_binding_decisions_v1 import (
    validate_given_binding_decision_result_set,
)
from .given_evidence_requirements_v1 import (
    validate_given_evidence_requirement_set,
)
from .then_atomization import ThenAtomizationError


BINDING_SET_VERSION = "agentspectesting.given-evidence-binding-set/v0.1"
BINDING_VERSION = "agentspectesting.given-evidence-binding/v0.1"
BINDING_STATUSES = frozenset({"ready", "partially_bound", "blocked"})

_NEXT_TASK = {
    "fixture_locator_choice": "fixture_relation_binding",
    "evaluator_shape_choice": "evaluator_operand_binding",
    "capability_polarity_choice": "capability_scenario_realization",
    "source_component_choice": "component_requirement_binding",
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def apply_given_binding_decisions(
    requirement_set: Mapping[str, Any], decision_result_set: Mapping[str, Any]
) -> dict[str, Any]:
    requirements = validate_given_evidence_requirement_set(requirement_set)
    decisions = validate_given_binding_decision_result_set(decision_result_set)
    if decisions["source_requirement_set_fingerprint"] != requirements[
        "requirement_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given binding decision/requirement lineage mismatch")
    decision_index = {item["requirement_id"]: item for item in decisions["results"]}
    expected_decisions = {
        item["requirement_id"]
        for item in requirements["requirements"]
        if item["resolution_status"] == "needs_binding"
    }
    if set(decision_index) != expected_decisions:
        raise ThenAtomizationError("Given binding decisions do not cover the binding queue")

    bindings = []
    for requirement in requirements["requirements"]:
        original_status = requirement["resolution_status"]
        if original_status == "ready":
            binding_status = "ready"
            next_task = None
            decision = None
        elif original_status == "blocked":
            binding_status = "blocked"
            next_task = None
            decision = None
        else:
            result = decision_index[requirement["requirement_id"]]
            expected_type = {
                "fixture_locator_binding": "fixture_locator_choice",
                "evaluator_contract_binding": "evaluator_shape_choice",
                "capability_scenario_binding": "capability_polarity_choice",
                "source_component_decomposition": "source_component_choice",
            }[requirement["binding_task"]]
            if result["decision_type"] != expected_type:
                raise ThenAtomizationError(
                    f"Given binding decision type mismatch: {requirement['requirement_id']}"
                )
            answer = result["answer"]
            if (
                answer.get("decision") == "insufficient"
                or answer.get("selected_candidate_id") == "insufficient"
            ):
                binding_status = "blocked"
                next_task = None
            else:
                binding_status = "partially_bound"
                next_task = _NEXT_TASK[result["decision_type"]]
            decision = {
                "decision_type": result["decision_type"],
                "answer": deepcopy(answer),
                "resolution_basis": result["resolution_basis"],
                "source_packet_fingerprint": result[
                    "source_packet_fingerprint"
                ],
            }
        binding = {
            "schema_version": BINDING_VERSION,
            "binding_id": f"{requirement['requirement_id']}::BINDING",
            "requirement_id": requirement["requirement_id"],
            "branch_id": requirement["branch_id"],
            "condition_id": requirement["condition_id"],
            "source_kind": requirement["source_kind"],
            "binding_status": binding_status,
            "next_binding_task": next_task,
            "base_requirement": deepcopy(requirement["requirement"]),
            "binding_decision": decision,
            "when": requirement["when"],
            "source_requirement_fingerprint": requirement[
                "requirement_fingerprint"
            ],
        }
        binding["binding_fingerprint"] = content_sha256(binding)
        bindings.append(binding)

    binding_index = {item["requirement_id"]: item for item in bindings}
    branches = []
    for branch in requirements["branches"]:
        statuses = [
            binding_index[item]["binding_status"]
            for item in branch["requirement_ids"]
        ]
        status = (
            "blocked"
            if "blocked" in statuses
            else "partially_bound"
            if "partially_bound" in statuses
            else "ready"
        )
        record = {
            "branch_id": branch["branch_id"],
            "requirement_ids": deepcopy(branch["requirement_ids"]),
            "binding_status": status,
            "source_branch_requirement_fingerprint": branch[
                "branch_requirement_fingerprint"
            ],
        }
        record["branch_binding_fingerprint"] = content_sha256(record)
        branches.append(record)

    binding_counts = Counter(item["binding_status"] for item in bindings)
    task_counts = Counter(
        item["next_binding_task"]
        for item in bindings
        if item["next_binding_task"] is not None
    )
    branch_counts = Counter(item["binding_status"] for item in branches)
    result = {
        "schema_version": BINDING_SET_VERSION,
        "bindings": bindings,
        "branches": branches,
        "summary": {
            "binding_count": len(bindings),
            "binding_status_counts": dict(sorted(binding_counts.items())),
            "next_binding_task_counts": dict(sorted(task_counts.items())),
            "branch_count": len(branches),
            "branch_status_counts": dict(sorted(branch_counts.items())),
        },
        "source_requirement_set_fingerprint": requirements[
            "requirement_set_fingerprint"
        ],
        "source_decision_result_set_fingerprint": decisions[
            "result_set_fingerprint"
        ],
    }
    result["binding_set_fingerprint"] = content_sha256(result)
    return validate_given_evidence_binding_set(result)


def validate_given_evidence_binding_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_evidence_binding_set")))
    fingerprint = result.pop("binding_set_fingerprint", None)
    if (
        result.get("schema_version") != BINDING_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given evidence binding set")
    bindings = result.get("bindings")
    branches = result.get("branches")
    if not isinstance(bindings, list) or not isinstance(branches, list):
        raise ThenAtomizationError("Given evidence binding arrays are invalid")
    binding_ids = []
    requirement_ids = []
    for raw in bindings:
        item = deepcopy(dict(_mapping(raw, "$.bindings[]")))
        supplied = item.pop("binding_fingerprint", None)
        if (
            item.get("schema_version") != BINDING_VERSION
            or item.get("binding_status") not in BINDING_STATUSES
            or supplied != content_sha256(item)
        ):
            raise ThenAtomizationError("invalid Given evidence binding")
        if item["binding_status"] == "partially_bound" and not item.get(
            "next_binding_task"
        ):
            raise ThenAtomizationError("partial Given binding has no next task")
        if item["binding_status"] != "partially_bound" and item.get(
            "next_binding_task"
        ) is not None:
            raise ThenAtomizationError("non-partial Given binding has a next task")
        binding_ids.append(item.get("binding_id"))
        requirement_ids.append(item.get("requirement_id"))
    if (
        len(binding_ids) != len(set(binding_ids))
        or len(requirement_ids) != len(set(requirement_ids))
    ):
        raise ThenAtomizationError("Given evidence binding identities overlap")
    branch_ids = []
    listed_requirement_ids = []
    for raw in branches:
        branch = deepcopy(dict(_mapping(raw, "$.branches[]")))
        supplied = branch.pop("branch_binding_fingerprint", None)
        if (
            branch.get("binding_status") not in BINDING_STATUSES
            or supplied != content_sha256(branch)
        ):
            raise ThenAtomizationError("invalid Given branch binding")
        branch_ids.append(branch.get("branch_id"))
        listed_requirement_ids.extend(branch.get("requirement_ids") or [])
    if len(branch_ids) != len(set(branch_ids)):
        raise ThenAtomizationError("Given branch binding identities overlap")
    if sorted(listed_requirement_ids) != sorted(requirement_ids):
        raise ThenAtomizationError("Given branch binding membership is not closed")
    binding_counts = Counter(item["binding_status"] for item in bindings)
    task_counts = Counter(
        item["next_binding_task"]
        for item in bindings
        if item["next_binding_task"] is not None
    )
    branch_counts = Counter(item["binding_status"] for item in branches)
    expected = {
        "binding_count": len(bindings),
        "binding_status_counts": dict(sorted(binding_counts.items())),
        "next_binding_task_counts": dict(sorted(task_counts.items())),
        "branch_count": len(branches),
        "branch_status_counts": dict(sorted(branch_counts.items())),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given evidence binding summary mismatch")
    result["binding_set_fingerprint"] = fingerprint
    return result
