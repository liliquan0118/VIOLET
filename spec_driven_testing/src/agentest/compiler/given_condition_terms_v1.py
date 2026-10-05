"""Materialize actual Given condition terms before boolean relation resolution."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .artifacts import content_sha256
from .given_clause_atomicity_v1 import (
    validate_given_clause_atomicity_packet_set,
    validate_given_clause_atomicity_response_set,
)
from .given_clause_split_v1 import validate_given_clause_split_result_set
from .given_when_contract_v2 import validate_given_when_contract_set_v2
from .then_atomization import ThenAtomizationError


TERM_SET_VERSION = "agentspectesting.given-condition-term-set/v0.1"
BRANCH_VERSION = "agentspectesting.given-condition-branch/v0.1"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def materialize_given_condition_terms(
    contract_set: Mapping[str, Any],
    atomicity_packet_set: Mapping[str, Any],
    atomicity_response_set: Mapping[str, Any],
    split_result_set: Mapping[str, Any],
) -> dict[str, Any]:
    contracts = validate_given_when_contract_set_v2(contract_set)
    packets = validate_given_clause_atomicity_packet_set(atomicity_packet_set)
    responses = validate_given_clause_atomicity_response_set(atomicity_response_set)
    splits = validate_given_clause_split_result_set(split_result_set)
    if packets["source_contract_set_fingerprint"] != contracts[
        "given_when_contract_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given condition term contract lineage mismatch")
    if responses["source_packet_set_fingerprint"] != packets[
        "packet_set_fingerprint"
    ]:
        raise ThenAtomizationError("Given condition term atomicity lineage mismatch")
    atomicity = {item["term_id"]: item for item in responses["responses"]}
    split_index = {item["term_id"]: item for item in splits["results"]}
    branches = []
    status_counts: Counter[str] = Counter()
    for contract in contracts["contracts"]:
        atom_index = {item["given_atom_id"]: item for item in contract["given_atoms"]}
        condition_terms = []
        blocked = []
        for term in contract["given_logic"]["terms"]:
            term_id = term["term_id"]
            atom = atom_index[term_id]
            if term["atomicity_status"] == "compiler_constant":
                condition_terms.append(
                    {
                        "condition_id": f"{term_id}::C001",
                        "source_term_id": term_id,
                        "condition_kind": "compiler_constant",
                        "constant": True,
                    }
                )
            elif term["atomicity_status"] == "structured_condition":
                condition_terms.append(
                    {
                        "condition_id": f"{term_id}::C001",
                        "source_term_id": term_id,
                        "condition_kind": "structured_condition",
                        "condition": deepcopy(atom["payload"]["condition"]),
                    }
                )
            else:
                decision = atomicity.get(term_id)
                if decision is None or decision["decision"] == "unclear":
                    blocked.append(term_id)
                    continue
                if decision["decision"] == "single_condition":
                    spans = [term["source_text"]]
                    kind = "source_clause"
                else:
                    split = split_index.get(term_id)
                    if split is None:
                        blocked.append(term_id)
                        continue
                    spans = split["condition_spans"]
                    kind = "split_source_clause"
                for ordinal, span in enumerate(spans, start=1):
                    condition_terms.append(
                        {
                            "condition_id": f"{term_id}::C{ordinal:03d}",
                            "source_term_id": term_id,
                            "condition_kind": kind,
                            "exact_span": span,
                            "source_clause": term["source_text"],
                        }
                    )
        status = "blocked" if blocked else "complete"
        branch = {
            "schema_version": BRANCH_VERSION,
            "branch_id": contract["branch_id"],
            "gwt": deepcopy(contract["gwt"]),
            "source_rule_context": deepcopy(contract["source_rule_context"]),
            "variant_context": deepcopy(contract["variant_context"]),
            "source_given_logic_status": contract["given_logic"]["logic_status"],
            "source_given_logic_fingerprint": contract["given_logic"][
                "given_logic_fingerprint"
            ],
            "condition_terms": condition_terms,
            "materialization_status": status,
            "blocked_source_term_ids": blocked,
            "source_contract_fingerprint": contract[
                "given_when_contract_fingerprint"
            ],
        }
        branch["condition_branch_fingerprint"] = content_sha256(branch)
        branches.append(branch)
        status_counts[status] += 1
    result = {
        "schema_version": TERM_SET_VERSION,
        "branches": branches,
        "summary": {
            "branch_count": len(branches),
            "materialization_status_counts": {
                key: status_counts[key] for key in ("blocked", "complete")
            },
            "condition_term_count": sum(
                len(item["condition_terms"]) for item in branches
            ),
        },
        "source_contract_set_fingerprint": contracts[
            "given_when_contract_set_fingerprint"
        ],
        "source_atomicity_response_set_fingerprint": responses[
            "response_set_fingerprint"
        ],
        "source_split_result_set_fingerprint": splits["result_set_fingerprint"],
    }
    result["condition_term_set_fingerprint"] = content_sha256(result)
    return validate_given_condition_term_set(result)


def validate_given_condition_term_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$given_condition_term_set")))
    fingerprint = result.pop("condition_term_set_fingerprint", None)
    if (
        result.get("schema_version") != TERM_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid Given condition term set")
    branches = result.get("branches")
    if not isinstance(branches, list):
        raise ThenAtomizationError("Given condition branches must be an array")
    branch_ids = [item.get("branch_id") for item in branches]
    condition_ids = [
        term.get("condition_id")
        for branch in branches
        for term in branch.get("condition_terms") or []
    ]
    if len(branch_ids) != len(set(branch_ids)) or len(condition_ids) != len(
        set(condition_ids)
    ):
        raise ThenAtomizationError("Given condition term identities are not unique")
    status_counts = Counter(item.get("materialization_status") for item in branches)
    expected = {
        "branch_count": len(branches),
        "materialization_status_counts": {
            key: status_counts[key] for key in ("blocked", "complete")
        },
        "condition_term_count": len(condition_ids),
    }
    if result.get("summary") != expected:
        raise ThenAtomizationError("Given condition term summary mismatch")
    result["condition_term_set_fingerprint"] = fingerprint
    return result
