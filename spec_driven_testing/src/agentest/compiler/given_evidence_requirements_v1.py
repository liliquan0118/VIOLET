"""Compile Given source decisions into lossless Driver evidence requirements."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .driver_evidence_capabilities_v1 import (
    validate_driver_evidence_capability_profile,
)
from .given_checkability_v1 import validate_given_checkability_result_set
from .given_evidence_source_v2 import validate_given_evidence_source_packet_set_v2
from .given_logical_form_v1 import validate_given_logical_form_set
from .given_source_family_v1 import (
    validate_given_source_family_packet_set,
    validate_given_source_family_result_set,
)
from .then_atomization import ThenAtomizationError


REQUIREMENT_SET_VERSION = "agentspectesting.given-evidence-requirement-set/v0.1"
REQUIREMENT_VERSION = "agentspectesting.given-evidence-requirement/v0.1"
STATUSES = frozenset({"ready", "needs_binding", "blocked"})

_NATURAL_REQUIREMENT_TYPES = {
    "user_speech_act": ("user_speech_act_constraint", "ready", None),
    "user_supplied_fact": ("user_fact_constraint", "ready", None),
    "fixture_state": (
        "fixture_state_constraint",
        "needs_binding",
        "fixture_locator_binding",
    ),
    "derived_state": (
        "derived_state_constraint",
        "needs_binding",
        "evaluator_contract_binding",
    ),
    "prior_runtime_event": (
        "prior_runtime_event_constraint",
        "needs_binding",
        "trace_event_binding",
    ),
    "agent_capability": (
        "agent_capability_constraint",
        "needs_binding",
        "capability_scenario_binding",
    ),
    "multiple_sources": (
        "composite_evidence_constraint",
        "needs_binding",
        "source_component_decomposition",
    ),
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _references(value: Any) -> list[str]:
    found = set()
    if isinstance(value, Mapping):
        ref = value.get("ref")
        if isinstance(ref, str) and ref:
            found.add(ref)
        for child in value.values():
            found.update(_references(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_references(child))
    return sorted(found)


def _source_packet_index(source_packets: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        member["condition_id"]: packet
        for packet in source_packets["packets"]
        for member in packet["members"]
    }


def _natural_payload(
    source_kind: str, condition: str, packet: Mapping[str, Any]
) -> tuple[str, str, str | None, dict[str, Any]]:
    requirement_type, status, binding_task = _NATURAL_REQUIREMENT_TYPES[source_kind]
    evidence = packet["model_input"]["accepted_evidence"]
    payload = {
        "type": requirement_type,
        "condition_statement": condition,
        "timing": "must_hold_before_when",
    }
    if source_kind == "user_speech_act":
        payload["realization_constraint"] = (
            "the user interaction must make the speech act true without asserting "
            "independent fixture facts"
        )
    elif source_kind == "user_supplied_fact":
        payload["authority_constraint"] = (
            "use only when the selected policy treats the user as authoritative for "
            "this fact"
        )
    elif source_kind == "fixture_state":
        payload["candidate_observables"] = deepcopy(
            evidence["matching_state_observables"]
        )
    elif source_kind == "derived_state":
        payload["candidate_policy_lines"] = deepcopy(
            evidence["matching_system_policy_lines"]
        )
        payload["candidate_observables"] = deepcopy(
            evidence["matching_state_observables"]
        )
    elif source_kind == "prior_runtime_event":
        payload["trace_constraint"] = (
            "bind an earlier event and a trace-verifiable event matcher"
        )
    elif source_kind == "agent_capability":
        payload["candidate_policy_lines"] = deepcopy(
            evidence["matching_system_policy_lines"]
        )
        payload["available_agent_action_names"] = deepcopy(
            evidence["available_agent_action_names"]
        )
    elif source_kind == "multiple_sources":
        payload["component_constraint"] = (
            "decompose into two or more independently typed evidence requirements"
        )
        payload["candidate_policy_lines"] = deepcopy(
            evidence["matching_system_policy_lines"]
        )
        payload["candidate_observables"] = deepcopy(
            evidence["matching_state_observables"]
        )
    return requirement_type, status, binding_task, payload


def compile_given_evidence_requirements(
    logical_form_set: Mapping[str, Any],
    evidence_source_packet_set: Mapping[str, Any],
    checkability_result_set: Mapping[str, Any],
    source_family_packet_set: Mapping[str, Any],
    source_family_result_set: Mapping[str, Any],
    driver_capability_profile: Mapping[str, Any],
) -> dict[str, Any]:
    forms = validate_given_logical_form_set(logical_form_set)
    evidence_packets = validate_given_evidence_source_packet_set_v2(
        evidence_source_packet_set
    )
    checkability = validate_given_checkability_result_set(checkability_result_set)
    family_packets = validate_given_source_family_packet_set(source_family_packet_set)
    family_results = validate_given_source_family_result_set(source_family_result_set)
    profile = validate_driver_evidence_capability_profile(driver_capability_profile)

    if evidence_packets["source_logical_form_set_fingerprint"] != forms[
        "logical_form_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given requirement logical-form lineage mismatch")
    if evidence_packets["source_driver_profile_fingerprint"] != profile[
        "profile_fingerprint"
    ]:
        raise ThenAtomizationError("Given requirement Driver profile lineage mismatch")
    if checkability["source_evidence_packet_set_fingerprint"] != evidence_packets[
        "packet_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given requirement checkability lineage mismatch")
    if family_packets["source_evidence_packet_set_fingerprint"] != evidence_packets[
        "packet_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given requirement family packet lineage mismatch")
    if family_packets["source_checkability_result_set_fingerprint"] != checkability[
        "result_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given requirement family/checkability mismatch")
    if family_results["source_packet_set_fingerprint"] != family_packets[
        "packet_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given requirement family result lineage mismatch")

    mechanical = {
        item["condition_id"]: item
        for item in evidence_packets["mechanical_resolutions"]
    }
    source_packet_by_condition = _source_packet_index(evidence_packets)
    family = {item["condition_id"]: item for item in family_results["results"]}
    blocked = {
        item["condition_id"]: item
        for item in family_results["non_checkable_conditions"]
    }
    check_index = {item["condition_id"]: item for item in checkability["results"]}

    requirements = []
    branches = []
    for form in forms["forms"]:
        term_index = {item["condition_id"]: item for item in form["condition_terms"]}
        requirement_ids = []
        branch_statuses = []
        for condition_id in form["branch_alignment"]["active_condition_ids"]:
            term = term_index[condition_id]
            requirement_id = f"{condition_id}::EVIDENCE"
            requirement_ids.append(requirement_id)
            source_lineage = {}
            if condition_id in mechanical:
                resolution = mechanical[condition_id]
                source_kind = resolution["source_kind"]
                source_lineage["source_mechanical_resolution_fingerprint"] = resolution[
                    "resolution_fingerprint"
                ]
                if source_kind == "compiler_constant":
                    status = "ready"
                    binding_task = None
                    payload = {
                        "type": "compiler_constant",
                        "value": True,
                        "timing": "compile_time",
                    }
                else:
                    predicate = deepcopy(term["condition"])
                    status = "ready"
                    binding_task = None
                    payload = {
                        "type": (
                            "derived_predicate"
                            if source_kind == "derived_state"
                            else "fixture_predicate"
                        ),
                        "predicate": predicate,
                        "references": _references(predicate),
                        "timing": "must_hold_before_when",
                    }
            elif condition_id in family:
                decision = family[condition_id]
                source_kind = decision["source_kind"]
                source_packet = source_packet_by_condition[condition_id]
                _, status, binding_task, payload = _natural_payload(
                    source_kind, term["exact_span"].strip(), source_packet
                )
                source_lineage["source_family_packet_fingerprint"] = decision[
                    "source_packet_fingerprint"
                ]
            elif condition_id in blocked:
                decision = blocked[condition_id]
                source_kind = decision["checkability"]
                status = "blocked"
                binding_task = None
                payload = {
                    "type": "undefined_evidence_source",
                    "condition_statement": term["exact_span"].strip(),
                    "reason": decision["reason"],
                    "timing": "must_hold_before_when",
                }
                check = check_index[condition_id]
                source_lineage["source_checkability_evidence_fingerprint"] = check[
                    "evidence_packet_fingerprint"
                ]
            else:
                raise ThenAtomizationError(
                    f"active Given condition has no evidence decision: {condition_id}"
                )
            item = {
                "schema_version": REQUIREMENT_VERSION,
                "requirement_id": requirement_id,
                "branch_id": form["branch_id"],
                "condition_id": condition_id,
                "condition_kind": term["condition_kind"],
                "source_kind": source_kind,
                "resolution_status": status,
                "binding_task": binding_task,
                "requirement": payload,
                "when": form["gwt"]["when"],
                "source_logical_form_fingerprint": form[
                    "logical_form_fingerprint"
                ],
                **source_lineage,
            }
            item["requirement_fingerprint"] = content_sha256(item)
            requirements.append(item)
            branch_statuses.append(status)
        branch_status = (
            "blocked"
            if "blocked" in branch_statuses
            else "needs_binding"
            if "needs_binding" in branch_statuses
            else "ready"
        )
        branch = {
            "branch_id": form["branch_id"],
            "given_operator": form["operator"],
            "active_condition_ids": deepcopy(
                form["branch_alignment"]["active_condition_ids"]
            ),
            "requirement_ids": requirement_ids,
            "resolution_status": branch_status,
            "source_logical_form_fingerprint": form[
                "logical_form_fingerprint"
            ],
        }
        branch["branch_requirement_fingerprint"] = content_sha256(branch)
        branches.append(branch)

    status_counts = Counter(item["resolution_status"] for item in requirements)
    source_counts = Counter(item["source_kind"] for item in requirements)
    task_counts = Counter(
        item["binding_task"]
        for item in requirements
        if item["binding_task"] is not None
    )
    branch_counts = Counter(item["resolution_status"] for item in branches)
    result = {
        "schema_version": REQUIREMENT_SET_VERSION,
        "requirements": requirements,
        "branches": branches,
        "summary": {
            "requirement_count": len(requirements),
            "requirement_status_counts": dict(sorted(status_counts.items())),
            "source_kind_counts": dict(sorted(source_counts.items())),
            "binding_task_counts": dict(sorted(task_counts.items())),
            "branch_count": len(branches),
            "branch_status_counts": dict(sorted(branch_counts.items())),
        },
        "source_logical_form_set_fingerprint": forms[
            "logical_form_set_fingerprint"
        ],
        "source_evidence_packet_set_fingerprint": evidence_packets[
            "packet_set_fingerprint"
        ],
        "source_checkability_result_set_fingerprint": checkability[
            "result_set_fingerprint"
        ],
        "source_family_packet_set_fingerprint": family_packets[
            "packet_set_fingerprint"
        ],
        "source_family_result_set_fingerprint": family_results[
            "result_set_fingerprint"
        ],
        "source_driver_profile_fingerprint": profile["profile_fingerprint"],
    }
    result["requirement_set_fingerprint"] = content_sha256(result)
    return validate_given_evidence_requirement_set(result)


def validate_given_evidence_requirement_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_evidence_requirement_set")))
    fingerprint = result.pop("requirement_set_fingerprint", None)
    if (
        result.get("schema_version") != REQUIREMENT_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given evidence requirement set")
    requirements = result.get("requirements")
    branches = result.get("branches")
    if not isinstance(requirements, list) or not isinstance(branches, list):
        raise ThenAtomizationError("Given evidence requirement arrays are invalid")
    requirement_ids = []
    condition_ids = []
    for raw in requirements:
        item = deepcopy(dict(_mapping(raw, "$.requirements[]")))
        supplied = item.pop("requirement_fingerprint", None)
        if (
            item.get("schema_version") != REQUIREMENT_VERSION
            or item.get("resolution_status") not in STATUSES
            or supplied != content_sha256(item)
        ):
            raise ThenAtomizationError("invalid Given evidence requirement")
        if item["resolution_status"] == "needs_binding" and not item.get(
            "binding_task"
        ):
            raise ThenAtomizationError("needs-binding Given requirement has no task")
        if item["resolution_status"] != "needs_binding" and item.get(
            "binding_task"
        ) is not None:
            raise ThenAtomizationError("non-binding Given requirement has a task")
        requirement_ids.append(item.get("requirement_id"))
        condition_ids.append(item.get("condition_id"))
    if (
        len(requirement_ids) != len(set(requirement_ids))
        or len(condition_ids) != len(set(condition_ids))
    ):
        raise ThenAtomizationError("Given evidence requirement identities overlap")
    branch_ids = []
    listed_requirement_ids = []
    for raw in branches:
        branch = deepcopy(dict(_mapping(raw, "$.branches[]")))
        supplied = branch.pop("branch_requirement_fingerprint", None)
        if (
            branch.get("resolution_status") not in STATUSES
            or supplied != content_sha256(branch)
        ):
            raise ThenAtomizationError("invalid Given branch requirement")
        branch_ids.append(branch.get("branch_id"))
        listed_requirement_ids.extend(branch.get("requirement_ids") or [])
    if len(branch_ids) != len(set(branch_ids)):
        raise ThenAtomizationError("Given branch requirement identities overlap")
    if sorted(listed_requirement_ids) != sorted(requirement_ids):
        raise ThenAtomizationError("Given branch/requirement membership is not closed")
    status_counts = Counter(item["resolution_status"] for item in requirements)
    source_counts = Counter(item["source_kind"] for item in requirements)
    task_counts = Counter(
        item["binding_task"]
        for item in requirements
        if item["binding_task"] is not None
    )
    branch_counts = Counter(item["resolution_status"] for item in branches)
    expected = {
        "requirement_count": len(requirements),
        "requirement_status_counts": dict(sorted(status_counts.items())),
        "source_kind_counts": dict(sorted(source_counts.items())),
        "binding_task_counts": dict(sorted(task_counts.items())),
        "branch_count": len(branches),
        "branch_status_counts": dict(sorted(branch_counts.items())),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given evidence requirement summary mismatch")
    result["requirement_set_fingerprint"] = fingerprint
    return result
