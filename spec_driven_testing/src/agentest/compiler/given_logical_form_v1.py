"""Close Given boolean relations while keeping branch alignment explicit."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .given_condition_terms_v1 import validate_given_condition_term_set
from .given_relation_resolution_v1 import (
    validate_given_relation_packet_set,
    validate_given_relation_response,
)
from .given_when_contract_v2 import validate_given_when_contract_set_v2
from .then_atomization import ThenAtomizationError


LOGICAL_FORM_SET_VERSION = "agentspectesting.given-logical-form-set/v0.1"
LOGICAL_FORM_VERSION = "agentspectesting.given-logical-form/v0.1"
OPERATORS = frozenset({"constant", "single", "all", "any", "unresolved"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def materialize_given_logical_forms(
    contract_set: Mapping[str, Any],
    term_set: Mapping[str, Any],
    relation_packet_set: Mapping[str, Any],
    relation_responses: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine mechanical and reviewed relation results for every Given branch."""

    contracts = validate_given_when_contract_set_v2(contract_set)
    terms = validate_given_condition_term_set(term_set)
    packets = validate_given_relation_packet_set(relation_packet_set)
    response_map = deepcopy(dict(_mapping(relation_responses, "$relation_responses")))
    if terms["source_contract_set_fingerprint"] != contracts[
        "given_when_contract_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given logical form contract lineage mismatch")
    if packets["source_condition_term_set_fingerprint"] != terms[
        "condition_term_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given logical form term lineage mismatch")
    packet_index = {item["packet_id"]: item for item in packets["packets"]}
    if set(response_map) != set(packet_index):
        raise ThenAtomizationError("Given relation responses must cover the closed batch")

    relation_by_branch: dict[str, dict[str, Any]] = {}
    for packet in packets["packets"]:
        checked = validate_given_relation_response(
            packet, _mapping(response_map[packet["packet_id"]], "$.response")
        )
        for member in packet["members"]:
            relation_by_branch[member["branch_id"]] = {
                "decision": checked["decision"],
                "reason": checked["reason"],
                "source_relation_packet_id": packet["packet_id"],
                "source_relation_packet_fingerprint": packet["packet_fingerprint"],
            }

    contract_index = {item["branch_id"]: item for item in contracts["contracts"]}
    mechanical_single = set(packets["mechanical_single_branch_ids"])
    forms = []
    status_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    alignment_counts: Counter[str] = Counter()
    for branch in terms["branches"]:
        branch_id = branch["branch_id"]
        contract = contract_index.get(branch_id)
        if contract is None:
            raise ThenAtomizationError(f"missing source contract for {branch_id}")
        source_logic = contract["given_logic"]
        relation_evidence = None
        if branch["materialization_status"] != "complete":
            logic_status = "unresolved"
            operator = "unresolved"
            basis = "condition_materialization_blocked"
        elif source_logic["logic_status"] == "resolved":
            logic_status = "resolved"
            operator = source_logic["operator"]
            basis = "source_contract_mechanical_logic"
        elif branch_id in mechanical_single:
            logic_status = "resolved"
            operator = "single"
            basis = "single_materialized_condition"
        else:
            relation_evidence = relation_by_branch.get(branch_id)
            if relation_evidence is None:
                raise ThenAtomizationError(f"missing relation decision for {branch_id}")
            decision = relation_evidence["decision"]
            if decision == "unclear":
                logic_status = "unresolved"
                operator = "unresolved"
                basis = "model_relation_unclear"
            else:
                logic_status = "resolved"
                operator = "all" if decision == "all_required" else "any"
                basis = "reviewed_model_relation"

        variant_context = branch.get("variant_context") or {}
        requires_alignment = bool(variant_context.get("requires_branch_alignment"))
        if logic_status != "resolved":
            alignment_status = "blocked_by_unresolved_logic"
            active_condition_ids = None
        elif not requires_alignment:
            alignment_status = "not_required"
            active_condition_ids = [
                item["condition_id"] for item in branch["condition_terms"]
            ]
        elif len(branch["condition_terms"]) == 1:
            alignment_status = "mechanically_resolved_single_condition"
            active_condition_ids = [branch["condition_terms"][0]["condition_id"]]
        elif operator != "any":
            # A conjunction does not contain branch-selecting alternatives: every
            # condition remains active for every Then variant.
            alignment_status = "mechanically_resolved_full_given"
            active_condition_ids = [
                item["condition_id"] for item in branch["condition_terms"]
            ]
        else:
            alignment_status = "requires_resolution"
            active_condition_ids = None

        form = {
            "schema_version": LOGICAL_FORM_VERSION,
            "branch_id": branch_id,
            "gwt": deepcopy(branch["gwt"]),
            "source_rule_context": deepcopy(branch["source_rule_context"]),
            "variant_context": deepcopy(branch["variant_context"]),
            "logic_status": logic_status,
            "operator": operator,
            "condition_terms": deepcopy(branch["condition_terms"]),
            "condition_ids": [
                item["condition_id"] for item in branch["condition_terms"]
            ],
            "resolution_basis": basis,
            "relation_evidence": deepcopy(relation_evidence),
            "branch_alignment": {
                "status": alignment_status,
                "active_condition_ids": active_condition_ids,
            },
            "source_given_logic_fingerprint": branch[
                "source_given_logic_fingerprint"
            ],
            "source_condition_branch_fingerprint": branch[
                "condition_branch_fingerprint"
            ],
        }
        form["logical_form_fingerprint"] = content_sha256(form)
        forms.append(form)
        status_counts[logic_status] += 1
        operator_counts[operator] += 1
        alignment_counts[alignment_status] += 1

    result = {
        "schema_version": LOGICAL_FORM_SET_VERSION,
        "forms": forms,
        "summary": {
            "branch_count": len(forms),
            "logic_status_counts": dict(sorted(status_counts.items())),
            "operator_counts": dict(sorted(operator_counts.items())),
            "branch_alignment_status_counts": dict(sorted(alignment_counts.items())),
        },
        "source_contract_set_fingerprint": contracts[
            "given_when_contract_set_fingerprint"
        ],
        "source_condition_term_set_fingerprint": terms[
            "condition_term_set_fingerprint"
        ],
        "source_relation_packet_set_fingerprint": packets[
            "packet_set_fingerprint"
        ],
    }
    result["logical_form_set_fingerprint"] = content_sha256(result)
    return validate_given_logical_form_set(result)


def validate_given_logical_form_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_logical_form_set")))
    fingerprint = result.pop("logical_form_set_fingerprint", None)
    if (
        result.get("schema_version") != LOGICAL_FORM_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given logical form set")
    forms = result.get("forms")
    if not isinstance(forms, list):
        raise ThenAtomizationError("Given logical forms must be an array")
    branch_ids = []
    status_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    alignment_counts: Counter[str] = Counter()
    for raw in forms:
        form = deepcopy(dict(_mapping(raw, "$.forms[]")))
        form_fingerprint = form.pop("logical_form_fingerprint", None)
        if (
            form.get("schema_version") != LOGICAL_FORM_VERSION
            or form_fingerprint != content_sha256(form)
        ):
            raise ThenAtomizationError("invalid Given logical form")
        operator = form.get("operator")
        if operator not in OPERATORS:
            raise ThenAtomizationError("invalid Given logical operator")
        condition_ids = form.get("condition_ids")
        actual_ids = [item.get("condition_id") for item in form.get("condition_terms") or []]
        if condition_ids != actual_ids or len(actual_ids) != len(set(actual_ids)):
            raise ThenAtomizationError("Given logical form condition identities mismatch")
        alignment = _mapping(form.get("branch_alignment"), "$.branch_alignment")
        active_ids = alignment.get("active_condition_ids")
        if active_ids is not None and (
            not isinstance(active_ids, list)
            or not set(active_ids).issubset(set(condition_ids))
        ):
            raise ThenAtomizationError("Given active condition identities are invalid")
        branch_ids.append(form.get("branch_id"))
        status_counts[form.get("logic_status")] += 1
        operator_counts[operator] += 1
        alignment_counts[alignment.get("status")] += 1
    if len(branch_ids) != len(set(branch_ids)):
        raise ThenAtomizationError("Given logical form branch IDs must be unique")
    expected_summary = {
        "branch_count": len(forms),
        "logic_status_counts": dict(sorted(status_counts.items())),
        "operator_counts": dict(sorted(operator_counts.items())),
        "branch_alignment_status_counts": dict(sorted(alignment_counts.items())),
    }
    if result.get("summary") != expected_summary:
        raise ThenAtomizationError("Given logical form summary mismatch")
    result["logical_form_set_fingerprint"] = fingerprint
    return result
