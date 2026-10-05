"""Spec-level source allocation plus branch-level Then atomization.

Version 0.2 corrects the v0.1 alignment boundary: source fidelity is audited
once for the complete set of sibling GWT branches belonging to a spec.  The
partition and independent inventory tasks remain isolated per branch.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .then_atomization import (
    ALIGNMENT_FINDING_TYPES,
    ALIGNMENT_STATUSES,
    ThenAtomizationError,
    build_then_atomization_packets,
    reconcile_then_atomization,
    validate_then_atomization_packet,
)


PACKET_SET_SCHEMA_VERSION = "agentspectesting.then-atomization-packet-set/v0.2"
SPEC_PACKET_SCHEMA_VERSION = "agentspectesting.then-spec-alignment-packet/v0.2"
RESPONSE_SET_SCHEMA_VERSION = "agentspectesting.then-atomization-response-set/v0.2"
ATOM_SET_SCHEMA_VERSION = "agentspectesting.then-atom-set/v0.2"
ATOM_COLLECTION_SCHEMA_VERSION = "agentspectesting.then-atom-collection/v0.2"
SPEC_ALIGNMENT_STATUSES = frozenset({"aligned", "issues_found", "ambiguous"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ThenAtomizationError(f"{path} must be a non-empty string")
    return value.strip()


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise ThenAtomizationError(
            f"{path} fields are invalid: expected {sorted(expected)}, got {sorted(value)}"
        )


def _source_pool(packet: Mapping[str, Any]) -> list[str]:
    context = packet["source_context"]
    values = [context["rule_text"], context.get("source_rule", "")]
    values.extend(context.get("clauses", []))
    values.extend(
        evidence.get("quote", "")
        for evidence in context.get("evidence", [])
        if isinstance(evidence, Mapping)
    )
    return [value for value in values if isinstance(value, str) and value]


def _validate_source_quotes(value: Any, packet: Mapping[str, Any], path: str) -> list[str]:
    if not isinstance(value, list):
        raise ThenAtomizationError(f"{path} must be an array")
    result = []
    pool = _source_pool(packet)
    for index, quote in enumerate(value):
        text = _nonempty(quote, f"{path}[{index}]")
        if not any(text in source for source in pool):
            raise ThenAtomizationError(f"{path}[{index}] is absent from supplied source")
        result.append(text)
    if len(result) != len(set(result)):
        raise ThenAtomizationError(f"{path} must not contain duplicates")
    return result


def _spec_alignment_packet(spec: Mapping[str, Any], selected: set[str]) -> dict[str, Any]:
    spec_id = _nonempty(spec.get("spec_id"), "$.spec_id")
    branches = []
    for index, raw in enumerate(spec.get("gwt") or []):
        gwt = _mapping(raw, f"$.gwt[{index}]")
        branches.append(
            {
                "branch_id": _nonempty(gwt.get("branch_id"), f"$.gwt[{index}].branch_id"),
                "given": _nonempty(gwt.get("given"), f"$.gwt[{index}].given"),
                "when": _nonempty(gwt.get("when"), f"$.gwt[{index}].when"),
                "then": _nonempty(gwt.get("then"), f"$.gwt[{index}].then"),
            }
        )
    selected_for_spec = [item["branch_id"] for item in branches if item["branch_id"] in selected]
    context = {
        "origin": _nonempty(spec.get("origin"), "$.origin"),
        "confidence": _nonempty(spec.get("confidence"), "$.confidence"),
        "rule_text": _nonempty(spec.get("rule_text"), "$.rule_text"),
        "clauses": deepcopy(spec.get("clauses") or []),
        "source_rule": spec.get("source_rule", ""),
        "evidence": deepcopy(spec.get("evidence") or []),
    }
    packet = {
        "schema_version": SPEC_PACKET_SCHEMA_VERSION,
        "spec_id": spec_id,
        "selected_branch_ids": selected_for_spec,
        "source_context": context,
        "gwt_branches": branches,
        "task_input": {
            "spec_id": spec_id,
            "source_context": deepcopy(context),
            "gwt_branches": deepcopy(branches),
        },
    }
    packet["packet_fingerprint"] = content_sha256(packet)
    return packet


def build_then_atomization_v2_packets(
    spec_document: Mapping[str, Any],
    *,
    selected_branch_ids: Sequence[str],
    source_path: str | None = None,
) -> dict[str, Any]:
    if not selected_branch_ids:
        raise ThenAtomizationError("v0.2 requires an explicit non-empty branch selection")
    branch_set = build_then_atomization_packets(
        spec_document,
        selected_branch_ids=selected_branch_ids,
        source_path=source_path,
    )
    selected = set(selected_branch_ids)
    selected_spec_ids = {packet["spec_id"] for packet in branch_set["packets"]}
    specs = spec_document.get("specs")
    if not isinstance(specs, list):
        raise ThenAtomizationError("$.specs must be an array")
    spec_packets = [
        _spec_alignment_packet(_mapping(spec, "$.specs[]"), selected)
        for spec in specs
        if isinstance(spec, Mapping) and spec.get("spec_id") in selected_spec_ids
    ]
    payload = {
        "schema_version": PACKET_SET_SCHEMA_VERSION,
        "source": {"path": source_path},
        "selected_branch_ids": list(branch_set["selected_branch_ids"]),
        "spec_alignment_packets": spec_packets,
        "branch_packets": branch_set["packets"],
        "summary": {
            "selected_spec_count": len(spec_packets),
            "selected_branch_count": len(branch_set["packets"]),
            "expected_model_calls": len(spec_packets) + 2 * len(branch_set["packets"]),
            "calls_by_task": {
                "spec_source_alignment": len(spec_packets),
                "predicate_partition": len(branch_set["packets"]),
                "claim_inventory": len(branch_set["packets"]),
            },
            "coverage_model_calls": 0,
        },
    }
    payload["packet_set_fingerprint"] = content_sha256(payload)
    return payload


def build_then_atomization_v2_packets_file(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    selected_branch_ids: Sequence[str],
) -> dict[str, Any]:
    source = Path(spec_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    result = build_then_atomization_v2_packets(
        document,
        selected_branch_ids=selected_branch_ids,
        source_path=str(source.resolve()),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def validate_spec_alignment_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    packet = deepcopy(dict(_mapping(value, "$")))
    fingerprint = packet.pop("packet_fingerprint", None)
    if packet.get("schema_version") != SPEC_PACKET_SCHEMA_VERSION:
        raise ThenAtomizationError("unsupported spec-alignment packet schema")
    if not isinstance(fingerprint, str) or fingerprint != content_sha256(packet):
        raise ThenAtomizationError("spec-alignment packet fingerprint mismatch")
    branches = packet.get("gwt_branches")
    if not isinstance(branches, list) or not branches:
        raise ThenAtomizationError("spec-alignment packet requires sibling GWT branches")
    branch_ids = [_nonempty(item.get("branch_id"), "$.gwt_branches[].branch_id") for item in branches]
    if len(branch_ids) != len(set(branch_ids)):
        raise ThenAtomizationError("spec-alignment sibling branch IDs must be unique")
    selected = packet.get("selected_branch_ids")
    if not isinstance(selected, list) or not selected or not set(selected).issubset(branch_ids):
        raise ThenAtomizationError("selected branches must be a non-empty subset of sibling branches")
    packet["packet_fingerprint"] = fingerprint
    return packet


def validate_then_atomization_v2_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    document = deepcopy(dict(_mapping(value, "$")))
    fingerprint = document.pop("packet_set_fingerprint", None)
    if document.get("schema_version") != PACKET_SET_SCHEMA_VERSION:
        raise ThenAtomizationError("unsupported v0.2 packet-set schema")
    if not isinstance(fingerprint, str) or fingerprint != content_sha256(document):
        raise ThenAtomizationError("v0.2 packet-set fingerprint mismatch")
    spec_packets = document.get("spec_alignment_packets")
    branch_packets = document.get("branch_packets")
    if not isinstance(spec_packets, list) or not isinstance(branch_packets, list):
        raise ThenAtomizationError("v0.2 packet arrays are missing")
    checked_specs = [validate_spec_alignment_packet(item) for item in spec_packets]
    checked_branches = [validate_then_atomization_packet(item) for item in branch_packets]
    selected_ids = [item["branch_id"] for item in checked_branches]
    if document.get("selected_branch_ids") != selected_ids:
        raise ThenAtomizationError("selected branch index does not match branch packets")
    expected = len(checked_specs) + 2 * len(checked_branches)
    if document.get("summary", {}).get("expected_model_calls") != expected:
        raise ThenAtomizationError("v0.2 expected model-call count is inconsistent")
    document["spec_alignment_packets"] = checked_specs
    document["branch_packets"] = checked_branches
    document["packet_set_fingerprint"] = fingerprint
    return document


def render_spec_alignment_prompt(packet: Mapping[str, Any], template: str) -> str:
    checked = validate_spec_alignment_packet(packet)
    placeholder = "{spec_source_alignment_input}"
    if placeholder not in template:
        raise ThenAtomizationError(f"prompt template is missing {placeholder}")
    return template.replace(
        placeholder,
        json.dumps(checked["task_input"], ensure_ascii=False, indent=2),
    )


def canonicalize_spec_alignment_response(
    response: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Canonicalize a derived top-level status when all details say aligned."""

    result = deepcopy(dict(_mapping(response, "$.spec_alignment")))
    changes: list[dict[str, str]] = []
    branches = result.get("branch_results")
    unallocated = result.get("unallocated_requirements")
    if (
        result.get("overall_status") == "issues_found"
        and isinstance(branches, list)
        and branches
        and all(
            isinstance(branch, Mapping) and branch.get("alignment_status") == "aligned"
            for branch in branches
        )
        and unallocated == []
    ):
        result["overall_status"] = "aligned"
        changes.append(
            {
                "field": "overall_status",
                "from": "issues_found",
                "to": "aligned",
                "reason": "derived_alignment_status",
            }
        )
    return result, changes


def validate_spec_alignment_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    checked = validate_spec_alignment_packet(packet)
    result = deepcopy(dict(_mapping(response, "$.spec_alignment")))
    _exact_keys(
        result,
        {"spec_id", "overall_status", "branch_results", "unallocated_requirements", "reason"},
        "$.spec_alignment",
    )
    if result.get("spec_id") != checked["spec_id"]:
        raise ThenAtomizationError("spec-alignment response spec_id mismatch")
    if result.get("overall_status") not in SPEC_ALIGNMENT_STATUSES:
        raise ThenAtomizationError("spec-alignment overall_status is invalid")
    _nonempty(result.get("reason"), "$.spec_alignment.reason")
    branch_results = result.get("branch_results")
    if not isinstance(branch_results, list):
        raise ThenAtomizationError("spec-alignment branch_results must be an array")
    branch_map = {item["branch_id"]: item for item in checked["gwt_branches"]}
    seen: set[str] = set()
    for index, raw in enumerate(branch_results):
        path = f"$.spec_alignment.branch_results[{index}]"
        branch = _mapping(raw, path)
        _exact_keys(branch, {"branch_id", "alignment_status", "findings", "reason"}, path)
        branch_id = branch.get("branch_id")
        if branch_id not in branch_map or branch_id in seen:
            raise ThenAtomizationError(f"{path}.branch_id is unknown or duplicated")
        seen.add(branch_id)
        if branch.get("alignment_status") not in ALIGNMENT_STATUSES:
            raise ThenAtomizationError(f"{path}.alignment_status is invalid")
        _nonempty(branch.get("reason"), f"{path}.reason")
        findings = branch.get("findings")
        if not isinstance(findings, list):
            raise ThenAtomizationError(f"{path}.findings must be an array")
        then = branch_map[branch_id]["then"]
        for finding_index, raw_finding in enumerate(findings):
            finding_path = f"{path}.findings[{finding_index}]"
            finding = _mapping(raw_finding, finding_path)
            _exact_keys(
                finding,
                {"finding_type", "then_spans", "source_quotes", "description"},
                finding_path,
            )
            if finding.get("finding_type") not in ALIGNMENT_FINDING_TYPES:
                raise ThenAtomizationError(f"{finding_path}.finding_type is invalid")
            spans = finding.get("then_spans")
            if not isinstance(spans, list):
                raise ThenAtomizationError(f"{finding_path}.then_spans must be an array")
            for span in spans:
                if _nonempty(span, f"{finding_path}.then_spans[]") not in then:
                    raise ThenAtomizationError(f"{finding_path} cites text outside branch Then")
            finding["source_quotes"] = _validate_source_quotes(
                finding.get("source_quotes"), checked, f"{finding_path}.source_quotes"
            )
            _nonempty(finding.get("description"), f"{finding_path}.description")
        if branch["alignment_status"] == "aligned" and findings:
            raise ThenAtomizationError(f"{path}: aligned branch must have no findings")
        if branch["alignment_status"] != "aligned" and not findings:
            raise ThenAtomizationError(f"{path}: non-aligned branch requires findings")
    if seen != set(branch_map):
        raise ThenAtomizationError("spec-alignment response must audit every sibling branch")

    unallocated = result.get("unallocated_requirements")
    if not isinstance(unallocated, list):
        raise ThenAtomizationError("unallocated_requirements must be an array")
    for index, raw in enumerate(unallocated):
        path = f"$.spec_alignment.unallocated_requirements[{index}]"
        item = _mapping(raw, path)
        _exact_keys(item, {"source_quotes", "description"}, path)
        item["source_quotes"] = _validate_source_quotes(
            item.get("source_quotes"), checked, f"{path}.source_quotes"
        )
        if not item["source_quotes"]:
            raise ThenAtomizationError(f"{path}.source_quotes must not be empty")
        _nonempty(item.get("description"), f"{path}.description")
    if result["overall_status"] == "aligned":
        if unallocated or any(item["alignment_status"] != "aligned" for item in branch_results):
            raise ThenAtomizationError("aligned spec response contains branch/allocation issues")
    if result["overall_status"] == "issues_found":
        if not unallocated and all(item["alignment_status"] == "aligned" for item in branch_results):
            raise ThenAtomizationError("issues_found spec response contains no branch/allocation issue")
    return result


def reconcile_then_atomization_v2(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_then_atomization_v2_packet_set(packet_set)
    responses = _mapping(response_set, "$.response_set")
    if responses.get("schema_version") != RESPONSE_SET_SCHEMA_VERSION:
        raise ThenAtomizationError("unsupported v0.2 response-set schema")
    spec_responses = responses.get("spec_alignments")
    branch_responses = responses.get("branch_responses")
    if not isinstance(spec_responses, list) or not isinstance(branch_responses, list):
        raise ThenAtomizationError("v0.2 response arrays are missing")

    spec_packet_map = {item["spec_id"]: item for item in packets["spec_alignment_packets"]}
    checked_spec_responses: dict[str, dict[str, Any]] = {}
    for raw in spec_responses:
        spec_id = _nonempty(raw.get("spec_id"), "$.spec_alignments[].spec_id")
        if spec_id not in spec_packet_map or spec_id in checked_spec_responses:
            raise ThenAtomizationError(f"unknown or duplicate spec alignment: {spec_id}")
        checked_spec_responses[spec_id] = validate_spec_alignment_response(
            spec_packet_map[spec_id], raw
        )
    if set(checked_spec_responses) != set(spec_packet_map):
        raise ThenAtomizationError("v0.2 response set does not cover selected specs")

    branch_response_map: dict[str, Mapping[str, Any]] = {}
    for raw in branch_responses:
        branch_id = _nonempty(raw.get("branch_id"), "$.branch_responses[].branch_id")
        if branch_id in branch_response_map:
            raise ThenAtomizationError(f"duplicate branch response: {branch_id}")
        branch_response_map[branch_id] = raw

    atom_sets = []
    for packet in packets["branch_packets"]:
        branch_id = packet["branch_id"]
        if branch_id not in branch_response_map:
            raise ThenAtomizationError(f"missing branch response: {branch_id}")
        branch_response = branch_response_map[branch_id]
        _exact_keys(
            branch_response,
            {"branch_id", "predicate_partition", "claim_inventory"},
            f"$.branch_responses[{branch_id}]",
        )
        spec_response = checked_spec_responses[packet["spec_id"]]
        branch_alignment = next(
            item for item in spec_response["branch_results"] if item["branch_id"] == branch_id
        )
        legacy_responses = {
            "branch_id": branch_id,
            "source_alignment": {
                "alignment_status": branch_alignment["alignment_status"],
                "findings": deepcopy(branch_alignment["findings"]),
                "reason": branch_alignment["reason"],
            },
            "predicate_partition": branch_response["predicate_partition"],
            "claim_inventory": branch_response["claim_inventory"],
        }
        atom_set = reconcile_then_atomization(packet, legacy_responses)
        atom_set.pop("then_atom_set_fingerprint")
        atom_set["schema_version"] = ATOM_SET_SCHEMA_VERSION
        atom_set["spec_allocation_audit"] = {
            "overall_status": spec_response["overall_status"],
            "unallocated_requirements": deepcopy(spec_response["unallocated_requirements"]),
            "all_sibling_branch_ids": [
                item["branch_id"] for item in spec_response["branch_results"]
            ],
        }
        atom_set["model_task_accounting"] = {
            "shared_spec_source_alignment_calls": 1,
            "predicate_partition_calls": 1,
            "claim_inventory_calls": 1,
            "amortized_calls_for_this_branch": 2,
        }
        atom_set["then_atom_set_fingerprint"] = content_sha256(atom_set)
        atom_sets.append(atom_set)

    status_counts: dict[str, int] = {}
    for item in atom_sets:
        status = item["decomposition_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    result = {
        "schema_version": ATOM_COLLECTION_SCHEMA_VERSION,
        "packet_set_fingerprint": packets["packet_set_fingerprint"],
        "atom_sets": atom_sets,
        "summary": {
            "spec_count": len(spec_packet_map),
            "branch_count": len(atom_sets),
            "atom_count": sum(len(item["atoms"]) for item in atom_sets),
            "relation_count": sum(len(item["relations"]) for item in atom_sets),
            "status_counts": dict(sorted(status_counts.items())),
            "model_calls_consumed": len(spec_packet_map) + 2 * len(atom_sets),
            "coverage_model_calls": 0,
        },
    }
    result["collection_fingerprint"] = content_sha256(result)
    return result


def reconcile_available_then_atomization_v2(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconcile the closed subset of a partially successful no-retry run."""

    packets = validate_then_atomization_v2_packet_set(packet_set)
    responses = _mapping(response_set, "$.response_set")
    if responses.get("schema_version") != RESPONSE_SET_SCHEMA_VERSION:
        raise ThenAtomizationError("unsupported v0.2 response-set schema")
    spec_responses = responses.get("spec_alignments")
    branch_responses = responses.get("branch_responses")
    if not isinstance(spec_responses, list) or not isinstance(branch_responses, list):
        raise ThenAtomizationError("v0.2 response arrays are missing")
    available_specs = {
        _nonempty(item.get("spec_id"), "$.spec_alignments[].spec_id")
        for item in spec_responses
    }
    available_branches = {
        _nonempty(item.get("branch_id"), "$.branch_responses[].branch_id")
        for item in branch_responses
    }
    branch_packet_map = {item["branch_id"]: item for item in packets["branch_packets"]}
    eligible_branches = [
        branch_id
        for branch_id in packets["selected_branch_ids"]
        if branch_id in available_branches
        and branch_packet_map[branch_id]["spec_id"] in available_specs
    ]
    if not eligible_branches:
        raise ThenAtomizationError("partial response set has no closed spec/branch subset")
    eligible_spec_ids = {
        branch_packet_map[branch_id]["spec_id"] for branch_id in eligible_branches
    }
    subset_spec_packets = []
    for original in packets["spec_alignment_packets"]:
        if original["spec_id"] not in eligible_spec_ids:
            continue
        item = deepcopy(original)
        item.pop("packet_fingerprint")
        item["selected_branch_ids"] = [
            branch_id
            for branch_id in item["selected_branch_ids"]
            if branch_id in eligible_branches
        ]
        item["packet_fingerprint"] = content_sha256(item)
        subset_spec_packets.append(item)
    subset_branch_packets = [branch_packet_map[branch_id] for branch_id in eligible_branches]
    subset_packets = {
        "schema_version": PACKET_SET_SCHEMA_VERSION,
        "source": deepcopy(packets["source"]),
        "selected_branch_ids": eligible_branches,
        "spec_alignment_packets": subset_spec_packets,
        "branch_packets": subset_branch_packets,
        "summary": {
            "selected_spec_count": len(subset_spec_packets),
            "selected_branch_count": len(subset_branch_packets),
            "expected_model_calls": len(subset_spec_packets) + 2 * len(subset_branch_packets),
            "calls_by_task": {
                "spec_source_alignment": len(subset_spec_packets),
                "predicate_partition": len(subset_branch_packets),
                "claim_inventory": len(subset_branch_packets),
            },
            "coverage_model_calls": 0,
        },
    }
    subset_packets["packet_set_fingerprint"] = content_sha256(subset_packets)
    subset_responses = {
        "schema_version": RESPONSE_SET_SCHEMA_VERSION,
        "spec_alignments": [
            item for item in spec_responses if item.get("spec_id") in eligible_spec_ids
        ],
        "branch_responses": [
            item for item in branch_responses if item.get("branch_id") in eligible_branches
        ],
    }
    result = reconcile_then_atomization_v2(subset_packets, subset_responses)
    result.pop("collection_fingerprint")
    result["availability"] = {
        "source_packet_set_fingerprint": packets["packet_set_fingerprint"],
        "closed_subset": True,
        "eligible_branch_ids": eligible_branches,
        "missing_spec_alignment_ids": sorted(
            {item["spec_id"] for item in packets["spec_alignment_packets"]} - available_specs
        ),
        "missing_branch_response_ids": sorted(
            set(packets["selected_branch_ids"]) - available_branches
        ),
    }
    result["collection_fingerprint"] = content_sha256(result)
    return result


def reconcile_then_atomization_v2_files(
    packet_set_path: str | Path,
    response_set_path: str | Path,
    output_path: str | Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    packets = json.loads(Path(packet_set_path).read_text(encoding="utf-8"))
    responses = json.loads(Path(response_set_path).read_text(encoding="utf-8"))
    result = (
        reconcile_available_then_atomization_v2(packets, responses)
        if allow_partial
        else reconcile_then_atomization_v2(packets, responses)
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
