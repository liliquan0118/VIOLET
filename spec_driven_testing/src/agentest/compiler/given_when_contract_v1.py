"""Compile GWT Given/When records into source-grounded eligibility contracts.

This compiler deliberately does not infer the meaning of prose that the
upstream fixture lookup could not map.  Structured database conditions are
preserved exactly; explicitly dropped or absent mappings remain visible as
unresolved atoms for a later resolver.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .then_atomization import ThenAtomizationError


CONTRACT_SET_VERSION = "agentspectesting.given-when-contract-set/v0.1"
CONTRACT_VERSION = "agentspectesting.given-when-contract/v0.1"
ATOM_VERSION = "agentspectesting.given-atom/v0.1"
STATUSES = frozenset({"ready", "partial", "needs_adjudication"})
ATOM_KINDS = frozenset(
    {
        "unconditional",
        "database_condition",
        "unresolved_non_database_condition",
        "unmapped_given",
    }
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _nonempty_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ThenAtomizationError(f"{path} must be non-empty text")
    return value.strip()


def _atom(
    *,
    atom_id: str,
    branch_id: str,
    kind: str,
    status: str,
    source_text: str,
    evidence_source: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": ATOM_VERSION,
        "given_atom_id": atom_id,
        "branch_id": branch_id,
        "atom_kind": kind,
        "resolution_status": status,
        "source_text": source_text,
        "required_truth_value": True,
        "evidence_source": evidence_source,
        "payload": deepcopy(dict(payload)),
    }
    result["given_atom_fingerprint"] = content_sha256(result)
    return result


def _compile_record(raw: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    record = deepcopy(dict(_mapping(raw, f"$[{ordinal}]")))
    gwt = _mapping(record.get("gwt"), f"$[{ordinal}].gwt")
    branch_id = _nonempty_text(gwt.get("branch_id"), f"$[{ordinal}].gwt.branch_id")
    given = _nonempty_text(gwt.get("given"), f"$[{ordinal}].gwt.given")
    when = _nonempty_text(gwt.get("when"), f"$[{ordinal}].gwt.when")
    then = _nonempty_text(gwt.get("then"), f"$[{ordinal}].gwt.then")
    conditions = record.get("conditions")
    dropped = record.get("dropped")
    errors = record.get("errors")
    if not isinstance(conditions, list) or not isinstance(dropped, list) or not isinstance(errors, list):
        raise ThenAtomizationError("conditions, dropped, and errors must be arrays")

    atoms = []
    if given.casefold() == "true":
        atoms.append(
            _atom(
                atom_id=f"{branch_id}::GIVEN::A001",
                branch_id=branch_id,
                kind="unconditional",
                status="resolved",
                source_text=given,
                evidence_source="compiler_constant",
                payload={"constant": True},
            )
        )
    else:
        for index, raw_condition in enumerate(conditions, start=1):
            condition = deepcopy(
                dict(_mapping(raw_condition, f"$[{ordinal}].conditions[{index - 1}]"))
            )
            required = {"table", "path", "op", "value"}
            if not required.issubset(condition):
                raise ThenAtomizationError("database condition lacks table/path/op/value")
            atoms.append(
                _atom(
                    atom_id=f"{branch_id}::GIVEN::A{len(atoms) + 1:03d}",
                    branch_id=branch_id,
                    kind="database_condition",
                    status="resolved",
                    source_text=given,
                    evidence_source="fixture_database",
                    payload={"condition": condition},
                )
            )
        for index, raw_dropped in enumerate(dropped, start=1):
            item = _mapping(raw_dropped, f"$[{ordinal}].dropped[{index - 1}]")
            clause = _nonempty_text(item.get("clause"), "$.dropped[].clause")
            reason = _nonempty_text(item.get("reason"), "$.dropped[].reason")
            atoms.append(
                _atom(
                    atom_id=f"{branch_id}::GIVEN::A{len(atoms) + 1:03d}",
                    branch_id=branch_id,
                    kind="unresolved_non_database_condition",
                    status="unresolved",
                    source_text=clause,
                    evidence_source="unresolved",
                    payload={"upstream_drop_reason": reason},
                )
            )
        if not conditions and not dropped:
            atoms.append(
                _atom(
                    atom_id=f"{branch_id}::GIVEN::A001",
                    branch_id=branch_id,
                    kind="unmapped_given",
                    status="unresolved",
                    source_text=given,
                    evidence_source="unresolved",
                    payload={"reason": "nontrivial_given_has_no_upstream_mapping"},
                )
            )

    resolved_count = sum(atom["resolution_status"] == "resolved" for atom in atoms)
    unresolved_count = len(atoms) - resolved_count
    if errors or (unresolved_count and not resolved_count):
        status = "needs_adjudication"
    elif unresolved_count:
        status = "partial"
    else:
        status = "ready"
    trigger = {
        "trigger_id": f"{branch_id}::WHEN::T001",
        "branch_id": branch_id,
        "source_text": when,
        "evidence_source": "runtime_interaction_trace",
        "binding_status": "requires_driver_binding",
    }
    trigger["trigger_fingerprint"] = content_sha256(trigger)
    contract = {
        "schema_version": CONTRACT_VERSION,
        "given_when_contract_id": f"{branch_id}::GW01",
        "branch_id": branch_id,
        "spec_id": _nonempty_text(record.get("spec_id"), f"$[{ordinal}].spec_id"),
        "kind": record.get("kind"),
        "origin": record.get("origin"),
        "gwt_index": record.get("gwt_index"),
        "gwt": {"given": given, "when": when, "then": then},
        "fixture_root": record.get("root"),
        "given_atoms": atoms,
        "when_trigger": trigger,
        "compilation_status": status,
        "diagnostics": {
            "resolved_atom_count": resolved_count,
            "unresolved_atom_count": unresolved_count,
            "upstream_errors": deepcopy(errors),
            "fixture_lookup_status": record.get("lookup_status"),
            "fixture_match_count": record.get("n_matches"),
        },
        "source_record_fingerprint": content_sha256(record),
    }
    contract["given_when_contract_fingerprint"] = content_sha256(contract)
    return contract


def compile_given_when_contracts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ThenAtomizationError("Given/When source must be an array")
    contracts = [_compile_record(raw, index) for index, raw in enumerate(records)]
    branch_ids = [contract["branch_id"] for contract in contracts]
    if len(branch_ids) != len(set(branch_ids)):
        raise ThenAtomizationError("Given/When branch IDs must be unique")
    status_counts = Counter(contract["compilation_status"] for contract in contracts)
    atom_counts = Counter(
        atom["atom_kind"]
        for contract in contracts
        for atom in contract["given_atoms"]
    )
    result = {
        "schema_version": CONTRACT_SET_VERSION,
        "contracts": contracts,
        "summary": {
            "branch_count": len(contracts),
            "contract_status_counts": {
                key: status_counts[key] for key in sorted(STATUSES)
            },
            "given_atom_count": sum(atom_counts.values()),
            "given_atom_kind_counts": {
                key: atom_counts[key] for key in sorted(ATOM_KINDS)
            },
            "when_trigger_count": len(contracts),
            "llm_calls": 0,
        },
        "source_record_set_fingerprint": content_sha256(list(records)),
    }
    result["given_when_contract_set_fingerprint"] = content_sha256(result)
    return validate_given_when_contract_set(result)


def validate_given_when_contract_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_when_contract_set")))
    fingerprint = result.pop("given_when_contract_set_fingerprint", None)
    if result.get("schema_version") != CONTRACT_SET_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid Given/When contract set")
    contracts = result.get("contracts")
    if not isinstance(contracts, list):
        raise ThenAtomizationError("Given/When contracts must be an array")
    status_counts: Counter[str] = Counter()
    atom_counts: Counter[str] = Counter()
    branch_ids = []
    for raw in contracts:
        contract = deepcopy(dict(_mapping(raw, "$.contracts[]")))
        contract_fingerprint = contract.pop("given_when_contract_fingerprint", None)
        if contract.get("schema_version") != CONTRACT_VERSION or contract_fingerprint != content_sha256(contract):
            raise ThenAtomizationError("invalid Given/When contract")
        status = contract.get("compilation_status")
        if status not in STATUSES:
            raise ThenAtomizationError("invalid Given/When compilation status")
        branch_id = contract.get("branch_id")
        branch_ids.append(branch_id)
        resolved = 0
        unresolved = 0
        for raw_atom in contract.get("given_atoms") or []:
            atom = deepcopy(dict(_mapping(raw_atom, "$.given_atoms[]")))
            atom_fingerprint = atom.pop("given_atom_fingerprint", None)
            if atom.get("schema_version") != ATOM_VERSION or atom_fingerprint != content_sha256(atom):
                raise ThenAtomizationError("invalid Given atom")
            if atom.get("branch_id") != branch_id or atom.get("atom_kind") not in ATOM_KINDS:
                raise ThenAtomizationError("Given atom lineage or kind is invalid")
            atom_counts[atom["atom_kind"]] += 1
            if atom.get("resolution_status") == "resolved":
                resolved += 1
            elif atom.get("resolution_status") == "unresolved":
                unresolved += 1
            else:
                raise ThenAtomizationError("Given atom resolution status is invalid")
        expected_status = (
            "needs_adjudication"
            if contract.get("diagnostics", {}).get("upstream_errors")
            or (unresolved and not resolved)
            else "partial"
            if unresolved
            else "ready"
        )
        if status != expected_status:
            raise ThenAtomizationError("Given/When status contradicts its atoms")
        trigger = deepcopy(dict(_mapping(contract.get("when_trigger"), "$.when_trigger")))
        trigger_fingerprint = trigger.pop("trigger_fingerprint", None)
        if trigger.get("branch_id") != branch_id or trigger_fingerprint != content_sha256(trigger):
            raise ThenAtomizationError("invalid When trigger")
        status_counts[status] += 1
    if len(branch_ids) != len(set(branch_ids)):
        raise ThenAtomizationError("Given/When contract branch IDs must be unique")
    expected_summary = {
        "branch_count": len(contracts),
        "contract_status_counts": {
            key: status_counts[key] for key in sorted(STATUSES)
        },
        "given_atom_count": sum(atom_counts.values()),
        "given_atom_kind_counts": {
            key: atom_counts[key] for key in sorted(ATOM_KINDS)
        },
        "when_trigger_count": len(contracts),
        "llm_calls": 0,
    }
    if result.get("summary") != expected_summary:
        raise ThenAtomizationError("Given/When contract summary mismatch")
    result["given_when_contract_set_fingerprint"] = fingerprint
    return result
