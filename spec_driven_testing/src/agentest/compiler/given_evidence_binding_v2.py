"""Deterministically lower accepted detail decisions into Driver evidence bindings."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .given_binding_details_v1 import (
    validate_given_binding_detail_packet_set,
    validate_given_binding_detail_response,
)
from .given_evidence_binding_v1 import validate_given_evidence_binding_set
from .then_atomization import ThenAtomizationError


BINDING_SET_VERSION = "agentspectesting.given-evidence-binding-set/v0.2"
BINDING_VERSION = "agentspectesting.given-evidence-binding/v0.2"
ACCEPTANCE_VERSION = "agentspectesting.given-binding-detail-acceptance/v0.1"
STATUSES = frozenset({"ready", "blocked"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _candidate_by_id(items: list[Any], candidate_id: str, id_field: str) -> Mapping[str, Any]:
    for item in items:
        if isinstance(item, Mapping) and item.get(id_field) == candidate_id:
            return item
    raise ThenAtomizationError(f"accepted detail candidate is missing: {candidate_id}")


def _lower_detail(packet: Mapping[str, Any], answer: Mapping[str, Any]) -> tuple[str, Any]:
    model_input = _mapping(packet["model_input"], "$.model_input")
    detail_type = packet["detail_type"]
    if detail_type == "fixture_relation_detail":
        selected = answer["selected_candidate_id"]
        candidate = _candidate_by_id(
            model_input["candidate_contracts"], selected, "candidate_id"
        )
        return ("blocked", None) if selected == "insufficient" else (
            "ready", deepcopy(candidate["contract"])
        )
    if detail_type in {"binary_operand_detail", "collection_operand_detail"}:
        operands = model_input["operand_candidates"]
        left = _candidate_by_id(operands, answer["left_operand_id"], "operand_id")
        right = _candidate_by_id(operands, answer["right_operand_id"], "operand_id")
        return "ready", {
            "kind": "deterministic_comparison",
            "operator": model_input["fixed_operator"],
            "left_operand": deepcopy(left),
            "right_operand": deepcopy(right),
        }
    if detail_type == "membership_operand_detail":
        operands = model_input["operand_candidates"]
        values = [
            deepcopy(_candidate_by_id(operands, item, "operand_id"))
            for item in answer["value_operand_ids"]
        ]
        set_operand = deepcopy(
            _candidate_by_id(operands, answer["set_operand_id"], "operand_id")
        )
        return "ready", {
            "kind": "deterministic_membership",
            "operator": model_input["fixed_operator"],
            "value_operands": values,
            "set_operand": set_operand,
        }
    if detail_type == "policy_evaluator_detail":
        if answer["decision"] == "insufficient":
            return "blocked", None
        policies = {
            item["policy_line_id"]: item for item in model_input["policy_lines"]
        }
        evidence = {
            item["endpoint_id"]: item
            for item in model_input["evidence_candidates"]
        }
        return "ready", {
            "kind": "policy_eligibility_evaluator",
            "policy_lines": [deepcopy(policies[item]) for item in answer["policy_line_ids"]],
            "evidence_endpoints": [deepcopy(evidence[item]) for item in answer["evidence_ids"]],
        }
    if detail_type == "status_semantics_detail":
        evaluator = deepcopy(model_input["resulting_evaluator"])
        evaluator.pop("blocking_values_ref", None)
        evaluator["blocking_values"] = deepcopy(answer["blocking_status_values"])
        return "ready", {
            "kind": "policy_status_evaluator",
            "policy_line": deepcopy(model_input["blocking_policy_line"]),
            "status_endpoint": deepcopy(model_input["status_endpoint"]),
            "evaluator": evaluator,
        }
    if detail_type == "capability_scenario_detail":
        selected = answer["selected_candidate_id"]
        candidate = _candidate_by_id(
            model_input["candidate_scenarios"], selected, "candidate_id"
        )
        return ("blocked", None) if selected == "insufficient" else (
            "ready",
            {
                "kind": "capability_scenario",
                "required_polarity": model_input["required_polarity"],
                "scenario": deepcopy(candidate),
            },
        )
    if detail_type == "component_requirement_detail":
        selected = answer["selected_candidate_id"]
        candidate = _candidate_by_id(
            model_input["candidate_contracts"], selected, "candidate_id"
        )
        return ("blocked", None) if selected == "insufficient" else (
            "ready",
            {
                "kind": "composite_evidence",
                "required_source_kinds": deepcopy(model_input["required_source_kinds"]),
                **deepcopy(candidate["contract"]),
            },
        )
    raise ThenAtomizationError(f"unsupported detail lowering type: {detail_type}")


def apply_given_binding_detail_selections(
    binding_set: Mapping[str, Any],
    acceptance_manifest: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    bindings = validate_given_evidence_binding_set(binding_set)
    manifest = deepcopy(dict(_mapping(acceptance_manifest, "$acceptance_manifest")))
    if manifest.get("schema_version") != ACCEPTANCE_VERSION:
        raise ThenAtomizationError("invalid Given detail acceptance version")
    if manifest.get("source_binding_set_fingerprint") != bindings[
        "binding_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given detail acceptance/binding lineage mismatch")
    selections = manifest.get("selections")
    if not isinstance(selections, list):
        raise ThenAtomizationError("Given detail selections must be an array")
    selection_index = {}
    for raw in selections:
        item = _mapping(raw, "$.selections[]")
        condition_id = item.get("condition_id")
        if not isinstance(condition_id, str) or condition_id in selection_index:
            raise ThenAtomizationError("Given detail selection identities overlap")
        if item.get("source_id") not in sources:
            raise ThenAtomizationError("Given detail selection source is missing")
        if not isinstance(item.get("resolution_basis"), str) or not item[
            "resolution_basis"
        ]:
            raise ThenAtomizationError("Given detail resolution basis is missing")
        selection_index[condition_id] = item
    expected = {
        item["condition_id"]
        for item in bindings["bindings"]
        if item["binding_status"] == "partially_bound"
    }
    if set(selection_index) != expected:
        raise ThenAtomizationError("Given detail acceptance does not cover the partial queue")

    source_indexes = {}
    source_fingerprints = {}
    for source_id, raw_source in sources.items():
        source = _mapping(raw_source, f"$sources.{source_id}")
        packet_set = validate_given_binding_detail_packet_set(
            _mapping(source.get("packet_set"), f"$sources.{source_id}.packet_set")
        )
        if packet_set["source_binding_set_fingerprint"] != bindings[
            "binding_set_fingerprint"
        ]:
            raise ThenAtomizationError("Given detail source/binding lineage mismatch")
        responses = _mapping(source.get("responses"), f"$sources.{source_id}.responses")
        source_indexes[source_id] = {
            item["member"]["condition_id"]: (item, responses.get(item["packet_id"]))
            for item in packet_set["packets"]
        }
        source_fingerprints[source_id] = packet_set["packet_set_fingerprint"]

    final_bindings = []
    condition_to_binding = {item["condition_id"]: item for item in bindings["bindings"]}
    for binding in bindings["bindings"]:
        base = {
            "schema_version": BINDING_VERSION,
            "binding_id": binding["binding_id"],
            "requirement_id": binding["requirement_id"],
            "branch_id": binding["branch_id"],
            "condition_id": binding["condition_id"],
            "source_kind": binding["source_kind"],
            "base_requirement": deepcopy(binding["base_requirement"]),
            "when": binding["when"],
            "source_partial_binding_fingerprint": binding["binding_fingerprint"],
        }
        if binding["binding_status"] == "ready":
            base["binding_status"] = "ready"
            base["executable_binding"] = deepcopy(binding["base_requirement"])
            base["detail_resolution"] = None
        elif binding["binding_status"] == "blocked":
            base["binding_status"] = "blocked"
            base["executable_binding"] = None
            base["detail_resolution"] = None
        else:
            selection = selection_index[binding["condition_id"]]
            packet_response = source_indexes[selection["source_id"]].get(
                binding["condition_id"]
            )
            if packet_response is None or packet_response[1] is None:
                raise ThenAtomizationError("accepted Given detail response is missing")
            packet, raw_response = packet_response
            if packet["member"]["source_binding_fingerprint"] != binding[
                "binding_fingerprint"
            ]:
                raise ThenAtomizationError("accepted Given detail member lineage mismatch")
            answer = validate_given_binding_detail_response(packet, raw_response)
            status, executable = _lower_detail(packet, answer)
            base["binding_status"] = status
            base["executable_binding"] = executable
            base["detail_resolution"] = {
                "detail_type": packet["detail_type"],
                "answer": answer,
                "resolution_basis": selection["resolution_basis"],
                "source_id": selection["source_id"],
                "source_packet_fingerprint": packet["packet_fingerprint"],
            }
        base["binding_fingerprint"] = content_sha256(base)
        final_bindings.append(base)

    final_index = {item["requirement_id"]: item for item in final_bindings}
    branches = []
    for branch in bindings["branches"]:
        statuses = [
            final_index[item]["binding_status"] for item in branch["requirement_ids"]
        ]
        record = {
            "branch_id": branch["branch_id"],
            "requirement_ids": deepcopy(branch["requirement_ids"]),
            "binding_status": "blocked" if "blocked" in statuses else "ready",
            "source_partial_branch_binding_fingerprint": branch[
                "branch_binding_fingerprint"
            ],
        }
        record["branch_binding_fingerprint"] = content_sha256(record)
        branches.append(record)
    binding_counts = Counter(item["binding_status"] for item in final_bindings)
    branch_counts = Counter(item["binding_status"] for item in branches)
    basis_counts = Counter(
        item["detail_resolution"]["resolution_basis"]
        for item in final_bindings
        if item["detail_resolution"] is not None
    )
    result = {
        "schema_version": BINDING_SET_VERSION,
        "bindings": final_bindings,
        "branches": branches,
        "summary": {
            "binding_count": len(final_bindings),
            "binding_status_counts": dict(sorted(binding_counts.items())),
            "detail_resolution_basis_counts": dict(sorted(basis_counts.items())),
            "branch_count": len(branches),
            "branch_status_counts": dict(sorted(branch_counts.items())),
        },
        "source_partial_binding_set_fingerprint": bindings[
            "binding_set_fingerprint"
        ],
        "source_acceptance_manifest_fingerprint": content_sha256(manifest),
        "source_detail_packet_set_fingerprints": dict(sorted(source_fingerprints.items())),
    }
    result["binding_set_fingerprint"] = content_sha256(result)
    return validate_given_evidence_binding_set_v2(result)


def validate_given_evidence_binding_set_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_evidence_binding_set_v2")))
    supplied = result.pop("binding_set_fingerprint", None)
    if result.get("schema_version") != BINDING_SET_VERSION or supplied != content_sha256(result):
        raise ThenAtomizationError("invalid Given evidence binding v2 set")
    bindings = result.get("bindings")
    branches = result.get("branches")
    if not isinstance(bindings, list) or not isinstance(branches, list):
        raise ThenAtomizationError("Given evidence binding v2 arrays are invalid")
    requirement_ids = []
    counts = Counter()
    basis_counts = Counter()
    for raw in bindings:
        item = deepcopy(dict(_mapping(raw, "$.bindings[]")))
        fingerprint = item.pop("binding_fingerprint", None)
        if (
            item.get("schema_version") != BINDING_VERSION
            or item.get("binding_status") not in STATUSES
            or fingerprint != content_sha256(item)
        ):
            raise ThenAtomizationError("invalid Given evidence binding v2")
        if (item["binding_status"] == "ready") != (item.get("executable_binding") is not None):
            raise ThenAtomizationError("Given evidence binding v2 readiness is inconsistent")
        requirement_ids.append(item.get("requirement_id"))
        counts[item["binding_status"]] += 1
        if item.get("detail_resolution") is not None:
            basis_counts[item["detail_resolution"]["resolution_basis"]] += 1
    if len(requirement_ids) != len(set(requirement_ids)):
        raise ThenAtomizationError("Given evidence binding v2 identities overlap")
    listed = []
    branch_counts = Counter()
    for raw in branches:
        branch = deepcopy(dict(_mapping(raw, "$.branches[]")))
        fingerprint = branch.pop("branch_binding_fingerprint", None)
        if branch.get("binding_status") not in STATUSES or fingerprint != content_sha256(branch):
            raise ThenAtomizationError("invalid Given branch binding v2")
        listed.extend(branch.get("requirement_ids") or [])
        branch_counts[branch["binding_status"]] += 1
    if sorted(listed) != sorted(requirement_ids):
        raise ThenAtomizationError("Given binding v2 branch membership is not closed")
    expected = {
        "binding_count": len(bindings),
        "binding_status_counts": dict(sorted(counts.items())),
        "detail_resolution_basis_counts": dict(sorted(basis_counts.items())),
        "branch_count": len(branches),
        "branch_status_counts": dict(sorted(branch_counts.items())),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given evidence binding v2 summary mismatch")
    result["binding_set_fingerprint"] = supplied
    return result
