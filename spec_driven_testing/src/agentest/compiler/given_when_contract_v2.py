"""Compile sound Given contracts without assuming dropped clauses form an AND list."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .given_when_contract_v1 import compile_given_when_contracts
from .then_atomization import ThenAtomizationError


CONTRACT_SET_VERSION = "agentspectesting.given-when-contract-set/v0.2"
CONTRACT_VERSION = "agentspectesting.given-when-contract/v0.2"
LOGIC_VERSION = "agentspectesting.given-logic/v0.1"
LOGIC_STATUSES = frozenset({"resolved", "requires_clause_resolution"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _variant_groups(contracts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for contract in contracts:
        grouped[
            (
                contract["spec_id"],
                contract["gwt"]["given"],
                contract["gwt"]["when"],
            )
        ].append(contract)
    result: dict[str, dict[str, Any]] = {}
    for key, members in grouped.items():
        if len(members) < 2:
            continue
        member_ids = [item["branch_id"] for item in members]
        context = {
            "variant_group_id": f"{key[0]}::GW-VARIANTS::{content_sha256(key)[:12]}",
            "member_branch_ids": member_ids,
            "member_count": len(members),
            "distinct_then_count": len({item["gwt"]["then"] for item in members}),
            "same_given": True,
            "same_when": True,
        }
        context["requires_branch_alignment"] = (
            key[1].casefold() != "true" and context["distinct_then_count"] > 1
        )
        for branch_id in member_ids:
            result[branch_id] = deepcopy(context)
    return result


def _given_logic(contract: Mapping[str, Any]) -> dict[str, Any]:
    atoms = contract["given_atoms"]
    term_views = [
        {
            "term_id": atom["given_atom_id"],
            "source_text": atom["source_text"],
            "term_kind": atom["atom_kind"],
            "atomicity_status": (
                "compiler_constant"
                if atom["atom_kind"] == "unconditional"
                else "structured_condition"
                if atom["atom_kind"] == "database_condition"
                else "unverified_source_clause"
            ),
        }
        for atom in atoms
    ]
    kinds = {atom["atom_kind"] for atom in atoms}
    if kinds == {"unconditional"} and len(atoms) == 1:
        status = "resolved"
        operator = "constant"
        basis = "compiler_constant"
    elif kinds == {"database_condition"}:
        status = "resolved"
        operator = "all" if len(atoms) > 1 else "single"
        basis = "structured_filter_conjunction"
    else:
        status = "requires_clause_resolution"
        operator = "unresolved"
        basis = "upstream_dropped_or_unmapped_clause_has_no_boolean_structure"
    result = {
        "schema_version": LOGIC_VERSION,
        "logic_status": status,
        "operator": operator,
        "basis": basis,
        "terms": term_views,
    }
    result["given_logic_fingerprint"] = content_sha256(result)
    return result


def compile_given_when_contracts_v2(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_v1 = compile_given_when_contracts(records)
    source_contracts = source_v1["contracts"]
    source_records = {
        record["gwt"]["branch_id"]: record for record in records
    }
    variant_index = _variant_groups(source_contracts)
    contracts = []
    for source in source_contracts:
        contract = deepcopy(source)
        source_fingerprint = contract.pop("given_when_contract_fingerprint")
        contract["schema_version"] = CONTRACT_VERSION
        contract["given_logic"] = _given_logic(contract)
        contract["variant_context"] = variant_index.get(contract["branch_id"])
        source_record = source_records[contract["branch_id"]]
        contract["source_rule_context"] = {
            "rule_text": source_record.get("rule_text"),
            "origin": source_record.get("origin"),
            "kind": source_record.get("kind"),
        }
        contract["source_v1_contract_fingerprint"] = source_fingerprint
        contract["given_when_contract_fingerprint"] = content_sha256(contract)
        contracts.append(contract)

    logic_counts = Counter(
        contract["given_logic"]["logic_status"] for contract in contracts
    )
    variant_groups = {
        context["variant_group_id"]: context
        for context in variant_index.values()
    }
    result = {
        "schema_version": CONTRACT_SET_VERSION,
        "contracts": contracts,
        "summary": {
            **deepcopy(source_v1["summary"]),
            "given_logic_status_counts": {
                key: logic_counts[key] for key in sorted(LOGIC_STATUSES)
            },
            "variant_group_count": len(variant_groups),
            "branch_alignment_required_group_count": sum(
                context["requires_branch_alignment"]
                for context in variant_groups.values()
            ),
        },
        "source_record_set_fingerprint": source_v1[
            "source_record_set_fingerprint"
        ],
        "source_v1_contract_set_fingerprint": source_v1[
            "given_when_contract_set_fingerprint"
        ],
    }
    result["given_when_contract_set_fingerprint"] = content_sha256(result)
    return validate_given_when_contract_set_v2(result)


def validate_given_when_contract_set_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_when_contract_set_v2")))
    fingerprint = result.pop("given_when_contract_set_fingerprint", None)
    if (
        result.get("schema_version") != CONTRACT_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given/When v2 contract set")
    contracts = result.get("contracts")
    if not isinstance(contracts, list):
        raise ThenAtomizationError("Given/When v2 contracts must be an array")
    branch_ids = []
    logic_counts: Counter[str] = Counter()
    group_index: dict[str, dict[str, Any]] = {}
    for raw in contracts:
        contract = deepcopy(dict(_mapping(raw, "$.contracts[]")))
        contract_fingerprint = contract.pop("given_when_contract_fingerprint", None)
        if (
            contract.get("schema_version") != CONTRACT_VERSION
            or contract_fingerprint != content_sha256(contract)
        ):
            raise ThenAtomizationError("invalid Given/When v2 contract")
        branch_id = contract.get("branch_id")
        branch_ids.append(branch_id)
        logic = deepcopy(dict(_mapping(contract.get("given_logic"), "$.given_logic")))
        logic_fingerprint = logic.pop("given_logic_fingerprint", None)
        if (
            logic.get("schema_version") != LOGIC_VERSION
            or logic_fingerprint != content_sha256(logic)
            or logic.get("logic_status") not in LOGIC_STATUSES
        ):
            raise ThenAtomizationError("invalid Given logical form")
        term_ids = [item.get("term_id") for item in logic.get("terms") or []]
        atom_ids = [item.get("given_atom_id") for item in contract.get("given_atoms") or []]
        if term_ids != atom_ids:
            raise ThenAtomizationError("Given logical form term lineage mismatch")
        expected_operator = (
            "unresolved"
            if logic["logic_status"] == "requires_clause_resolution"
            else logic.get("operator")
        )
        if logic.get("operator") != expected_operator:
            raise ThenAtomizationError("Given logical form status/operator mismatch")
        if logic["logic_status"] == "resolved" and any(
            term.get("atomicity_status") == "unverified_source_clause"
            for term in logic["terms"]
        ):
            raise ThenAtomizationError("unverified Given clause cannot be resolved")
        logic_counts[logic["logic_status"]] += 1
        context = contract.get("variant_context")
        if context is not None:
            context = dict(_mapping(context, "$.variant_context"))
            if branch_id not in context.get("member_branch_ids", []):
                raise ThenAtomizationError("variant context branch lineage mismatch")
            group_id = context.get("variant_group_id")
            previous = group_index.setdefault(group_id, context)
            if previous != context:
                raise ThenAtomizationError("variant context differs within group")
        rule_context = _mapping(
            contract.get("source_rule_context"), "$.source_rule_context"
        )
        if not isinstance(rule_context.get("rule_text"), str) or not rule_context[
            "rule_text"
        ].strip():
            raise ThenAtomizationError("Given v2 source rule context is missing")
    if len(branch_ids) != len(set(branch_ids)):
        raise ThenAtomizationError("Given/When v2 branch IDs must be unique")
    source_summary = {
        key: value
        for key, value in result.get("summary", {}).items()
        if key
        not in {
            "given_logic_status_counts",
            "variant_group_count",
            "branch_alignment_required_group_count",
        }
    }
    expected_summary = {
        **source_summary,
        "given_logic_status_counts": {
            key: logic_counts[key] for key in sorted(LOGIC_STATUSES)
        },
        "variant_group_count": len(group_index),
        "branch_alignment_required_group_count": sum(
            context.get("requires_branch_alignment") is True
            for context in group_index.values()
        ),
    }
    if result.get("summary") != expected_summary:
        raise ThenAtomizationError("Given/When v2 summary mismatch")
    result["given_when_contract_set_fingerprint"] = fingerprint
    return result
