"""Prepare and reconcile source-grounded atomic claims from GWT ``then`` text.

The module deliberately stops before oracle-type classification, tool binding,
or Coverage Model lookup.  Model-facing tasks decide only source alignment and
semantic claim boundaries.  IDs, lineage, validation, reconciliation, and
fingerprints are deterministic.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256


PACKET_SCHEMA_VERSION = "agentspectesting.then-atomization-packet/v0.1"
PACKET_SET_SCHEMA_VERSION = "agentspectesting.then-atomization-packet-set/v0.1"
RESPONSE_SET_SCHEMA_VERSION = "agentspectesting.then-atomization-response-set/v0.1"
ATOM_SET_SCHEMA_VERSION = "agentspectesting.then-atom-set/v0.1"
ATOM_COLLECTION_SCHEMA_VERSION = "agentspectesting.then-atom-collection/v0.1"

ALIGNMENT_STATUSES = frozenset({"aligned", "partial", "conflict", "ambiguous"})
RESOLUTION_STATUSES = frozenset({"resolved", "ambiguous"})
MODALITIES = frozenset({"required", "prohibited", "permitted"})
RELATIONS = frozenset(
    {"before", "after", "same_turn", "mutually_exclusive", "requires"}
)
ALIGNMENT_FINDING_TYPES = frozenset(
    {
        "omitted_requirement",
        "unsupported_detail",
        "incorrect_modality",
        "incorrect_polarity",
        "incorrect_scope",
        "unresolved_context",
    }
)


class ThenAtomizationError(ValueError):
    """Raised when a Then atomization artifact violates its contract."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ThenAtomizationError(f"{path} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, path: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise ThenAtomizationError(f"{path} must be an array")
    result = [_nonempty_string(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if not allow_empty and not result:
        raise ThenAtomizationError(f"{path} must not be empty")
    if len(result) != len(set(result)):
        raise ThenAtomizationError(f"{path} must not contain duplicates")
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ThenAtomizationError(
            f"{path} fields are invalid: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _validate_exact_spans(spans: Any, source: str, path: str) -> list[str]:
    values = _string_list(spans, path, allow_empty=False)
    for span in values:
        if span not in source:
            raise ThenAtomizationError(f"{path} span is not an exact substring of then: {span!r}")
    return values


def build_then_atomization_packets(
    spec_document: Mapping[str, Any],
    *,
    selected_branch_ids: Sequence[str] | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    """Build one model-task packet per GWT branch from the new JSON format."""

    document = _mapping(spec_document, "$")
    specs = document.get("specs")
    if not isinstance(specs, list):
        raise ThenAtomizationError("$.specs must be an array")
    selected = set(selected_branch_ids or [])
    if len(selected) != len(selected_branch_ids or []):
        raise ThenAtomizationError("selected_branch_ids must not contain duplicates")

    packets: list[dict[str, Any]] = []
    seen_branches: set[str] = set()
    for spec_index, raw_spec in enumerate(specs):
        spec = _mapping(raw_spec, f"$.specs[{spec_index}]")
        spec_id = _nonempty_string(spec.get("spec_id"), f"$.specs[{spec_index}].spec_id")
        rule_text = _nonempty_string(
            spec.get("rule_text"), f"$.specs[{spec_index}].rule_text"
        )
        origin = _nonempty_string(spec.get("origin"), f"$.specs[{spec_index}].origin")
        confidence = _nonempty_string(
            spec.get("confidence"), f"$.specs[{spec_index}].confidence"
        )
        clauses = _string_list(spec.get("clauses", []), f"$.specs[{spec_index}].clauses")
        source_rule = spec.get("source_rule", "")
        if not isinstance(source_rule, str):
            raise ThenAtomizationError(f"$.specs[{spec_index}].source_rule must be a string")
        evidence = spec.get("evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(item, Mapping) for item in evidence):
            raise ThenAtomizationError(f"$.specs[{spec_index}].evidence must be an array of objects")
        gwt_items = spec.get("gwt")
        if not isinstance(gwt_items, list) or not gwt_items:
            raise ThenAtomizationError(f"$.specs[{spec_index}].gwt must be a non-empty array")

        for gwt_index, raw_gwt in enumerate(gwt_items):
            gwt = _mapping(raw_gwt, f"$.specs[{spec_index}].gwt[{gwt_index}]")
            branch_id = _nonempty_string(
                gwt.get("branch_id"), f"$.specs[{spec_index}].gwt[{gwt_index}].branch_id"
            )
            if branch_id in seen_branches:
                raise ThenAtomizationError(f"duplicate branch_id: {branch_id}")
            seen_branches.add(branch_id)
            if selected and branch_id not in selected:
                continue
            given = _nonempty_string(gwt.get("given"), f"{branch_id}.given")
            when = _nonempty_string(gwt.get("when"), f"{branch_id}.when")
            then = _nonempty_string(gwt.get("then"), f"{branch_id}.then")
            packet = {
                "schema_version": PACKET_SCHEMA_VERSION,
                "spec_id": spec_id,
                "branch_id": branch_id,
                "source_context": {
                    "origin": origin,
                    "confidence": confidence,
                    "rule_text": rule_text,
                    "clauses": clauses,
                    "source_rule": source_rule,
                    "evidence": deepcopy(evidence),
                },
                "gwt_context": {"given": given, "when": when, "then": then},
                "task_inputs": {
                    "source_alignment": {
                        "spec_id": spec_id,
                        "branch_id": branch_id,
                        "given": given,
                        "when": when,
                        "then": then,
                        "rule_text": rule_text,
                        "source_rule": source_rule,
                        "clauses": clauses,
                        "evidence": deepcopy(evidence),
                    },
                    "predicate_partition": {
                        "spec_id": spec_id,
                        "branch_id": branch_id,
                        "given_context": given,
                        "when_context": when,
                        "then": then,
                    },
                    "claim_inventory": {
                        "spec_id": spec_id,
                        "branch_id": branch_id,
                        "given_context": given,
                        "when_context": when,
                        "then": then,
                    },
                },
            }
            packet["packet_fingerprint"] = content_sha256(packet)
            packets.append(packet)

    missing = sorted(selected - {packet["branch_id"] for packet in packets})
    if missing:
        raise ThenAtomizationError(f"selected branches were not found: {missing}")
    payload = {
        "schema_version": PACKET_SET_SCHEMA_VERSION,
        "source": {"path": source_path},
        "selected_branch_ids": [packet["branch_id"] for packet in packets],
        "packets": packets,
        "summary": {
            "branch_count": len(packets),
            "expected_model_calls": len(packets) * 3,
            "calls_per_branch": {
                "source_alignment": 1,
                "predicate_partition": 1,
                "claim_inventory": 1,
            },
            "coverage_model_calls": 0,
        },
    }
    payload["packet_set_fingerprint"] = content_sha256(payload)
    return payload


def build_then_atomization_packets_file(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    selected_branch_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = Path(spec_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    result = build_then_atomization_packets(
        document,
        selected_branch_ids=selected_branch_ids,
        source_path=str(source.resolve()),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def render_then_task_prompt(
    packet: Mapping[str, Any], task_name: str, template: str
) -> str:
    """Render one checked packet into one of the three narrow prompt templates."""

    checked = validate_then_atomization_packet(packet)
    field_by_task = {
        "source_alignment": "{source_alignment_input}",
        "predicate_partition": "{predicate_partition_input}",
        "claim_inventory": "{claim_inventory_input}",
    }
    if task_name not in field_by_task:
        raise ThenAtomizationError(f"unknown task_name: {task_name}")
    placeholder = field_by_task[task_name]
    if placeholder not in template:
        raise ThenAtomizationError(f"prompt template is missing {placeholder}")
    task_input = checked["task_inputs"][task_name]
    return template.replace(placeholder, json.dumps(task_input, ensure_ascii=False, indent=2))


def validate_then_atomization_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    packet = deepcopy(dict(_mapping(value, "$")))
    fingerprint = packet.pop("packet_fingerprint", None)
    if packet.get("schema_version") != PACKET_SCHEMA_VERSION:
        raise ThenAtomizationError("unsupported Then atomization packet schema")
    if not isinstance(fingerprint, str) or fingerprint != content_sha256(packet):
        raise ThenAtomizationError("Then atomization packet fingerprint mismatch")
    packet["packet_fingerprint"] = fingerprint
    return packet


def validate_then_atomization_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    document = deepcopy(dict(_mapping(value, "$")))
    fingerprint = document.pop("packet_set_fingerprint", None)
    if document.get("schema_version") != PACKET_SET_SCHEMA_VERSION:
        raise ThenAtomizationError("unsupported packet-set schema")
    if not isinstance(fingerprint, str) or fingerprint != content_sha256(document):
        raise ThenAtomizationError("Then atomization packet-set fingerprint mismatch")
    packets = document.get("packets")
    branch_ids = document.get("selected_branch_ids")
    if not isinstance(packets, list) or not isinstance(branch_ids, list):
        raise ThenAtomizationError("packet-set packets and selected_branch_ids must be arrays")
    checked_packets = [
        validate_then_atomization_packet(_mapping(packet, f"$.packets[{index}]"))
        for index, packet in enumerate(packets)
    ]
    actual_ids = [packet["branch_id"] for packet in checked_packets]
    if branch_ids != actual_ids or len(actual_ids) != len(set(actual_ids)):
        raise ThenAtomizationError("packet-set selected_branch_ids do not exactly index packets")
    summary = _mapping(document.get("summary"), "$.summary")
    if summary.get("branch_count") != len(checked_packets):
        raise ThenAtomizationError("packet-set summary branch_count is inconsistent")
    if summary.get("expected_model_calls") != len(checked_packets) * 3:
        raise ThenAtomizationError("packet-set expected_model_calls is inconsistent")
    document["packets"] = checked_packets
    document["packet_set_fingerprint"] = fingerprint
    return document


def _source_quote_pool(packet: Mapping[str, Any]) -> list[str]:
    context = packet["source_context"]
    values = [context["rule_text"], context.get("source_rule", "")]
    values.extend(context.get("clauses", []))
    values.extend(
        item.get("quote", "")
        for item in context.get("evidence", [])
        if isinstance(item, Mapping)
    )
    return [value for value in values if isinstance(value, str) and value]


def _validate_alignment(value: Any, packet: Mapping[str, Any]) -> dict[str, Any]:
    then = packet["gwt_context"]["then"]
    source_pool = _source_quote_pool(packet)
    result = deepcopy(dict(_mapping(value, "$.source_alignment")))
    _exact_keys(result, {"alignment_status", "findings", "reason"}, "$.source_alignment")
    status = result.get("alignment_status")
    if status not in ALIGNMENT_STATUSES:
        raise ThenAtomizationError("$.source_alignment.alignment_status is invalid")
    _nonempty_string(result.get("reason"), "$.source_alignment.reason")
    findings = result.get("findings")
    if not isinstance(findings, list):
        raise ThenAtomizationError("$.source_alignment.findings must be an array")
    normalized_findings = []
    for index, raw in enumerate(findings):
        path = f"$.source_alignment.findings[{index}]"
        finding = deepcopy(dict(_mapping(raw, path)))
        _exact_keys(
            finding,
            {"finding_type", "then_spans", "source_quotes", "description"},
            path,
        )
        if finding.get("finding_type") not in ALIGNMENT_FINDING_TYPES:
            raise ThenAtomizationError(f"{path}.finding_type is invalid")
        finding["then_spans"] = (
            _validate_exact_spans(finding["then_spans"], then, f"{path}.then_spans")
            if finding["then_spans"]
            else []
        )
        finding["source_quotes"] = _string_list(
            finding["source_quotes"], f"{path}.source_quotes"
        )
        for quote in finding["source_quotes"]:
            if not any(quote in source for source in source_pool):
                raise ThenAtomizationError(
                    f"{path}.source_quotes contains text absent from supplied source: {quote!r}"
                )
        _nonempty_string(finding.get("description"), f"{path}.description")
        normalized_findings.append(finding)
    if status == "aligned" and normalized_findings:
        raise ThenAtomizationError("aligned source response must not contain findings")
    if status != "aligned" and not normalized_findings:
        raise ThenAtomizationError("non-aligned source response requires findings")
    result["findings"] = normalized_findings
    return result


def _validate_partition(value: Any, then: str) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$.predicate_partition")))
    _exact_keys(result, {"resolution_status", "atoms", "relations", "reason"}, "$.predicate_partition")
    status = result.get("resolution_status")
    if status not in RESOLUTION_STATUSES:
        raise ThenAtomizationError("$.predicate_partition.resolution_status is invalid")
    _nonempty_string(result.get("reason"), "$.predicate_partition.reason")
    atoms = result.get("atoms")
    relations = result.get("relations")
    if not isinstance(atoms, list) or not isinstance(relations, list):
        raise ThenAtomizationError("partition atoms and relations must be arrays")
    if status == "ambiguous" and (atoms or relations):
        raise ThenAtomizationError("ambiguous partition must not select atoms or relations")
    if status == "resolved" and not atoms:
        raise ThenAtomizationError("resolved partition requires at least one atom")
    atom_ids: set[str] = set()
    for index, raw in enumerate(atoms, start=1):
        path = f"$.predicate_partition.atoms[{index - 1}]"
        atom = _mapping(raw, path)
        _exact_keys(atom, {"atom_id", "modality", "claim", "evidence_spans"}, path)
        if atom.get("atom_id") != f"T{index:02d}":
            raise ThenAtomizationError("partition atom IDs must be consecutive T01, T02, ...")
        if atom.get("modality") not in MODALITIES:
            raise ThenAtomizationError(f"{path}.modality is invalid")
        _nonempty_string(atom.get("claim"), f"{path}.claim")
        _validate_exact_spans(atom.get("evidence_spans"), then, f"{path}.evidence_spans")
        atom_ids.add(atom["atom_id"])
    relation_ids: set[str] = set()
    for index, raw in enumerate(relations, start=1):
        path = f"$.predicate_partition.relations[{index - 1}]"
        relation = _mapping(raw, path)
        _exact_keys(
            relation,
            {"relation_id", "relation", "left_atom_id", "right_atom_id", "evidence_spans"},
            path,
        )
        if relation.get("relation_id") != f"R{index:02d}":
            raise ThenAtomizationError("partition relation IDs must be consecutive R01, R02, ...")
        if relation.get("relation") not in RELATIONS:
            raise ThenAtomizationError(f"{path}.relation is invalid")
        if relation.get("left_atom_id") not in atom_ids or relation.get("right_atom_id") not in atom_ids:
            raise ThenAtomizationError(f"{path} references an unknown atom")
        if relation["left_atom_id"] == relation["right_atom_id"]:
            raise ThenAtomizationError(f"{path} must connect different atoms")
        _validate_exact_spans(relation.get("evidence_spans"), then, f"{path}.evidence_spans")
        relation_ids.add(relation["relation_id"])
    return result


def _validate_inventory(value: Any, then: str) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$.claim_inventory")))
    _exact_keys(result, {"resolution_status", "claims", "relations", "reason"}, "$.claim_inventory")
    status = result.get("resolution_status")
    if status not in RESOLUTION_STATUSES:
        raise ThenAtomizationError("$.claim_inventory.resolution_status is invalid")
    _nonempty_string(result.get("reason"), "$.claim_inventory.reason")
    claims = result.get("claims")
    relations = result.get("relations")
    if not isinstance(claims, list) or not isinstance(relations, list):
        raise ThenAtomizationError("inventory claims and relations must be arrays")
    if status == "ambiguous" and (claims or relations):
        raise ThenAtomizationError("ambiguous inventory must not select claims or relations")
    if status == "resolved" and not claims:
        raise ThenAtomizationError("resolved inventory requires at least one claim")
    claim_ids: set[str] = set()
    for index, raw in enumerate(claims, start=1):
        path = f"$.claim_inventory.claims[{index - 1}]"
        claim = _mapping(raw, path)
        _exact_keys(claim, {"claim_id", "claim", "evidence_spans"}, path)
        if claim.get("claim_id") != f"C{index:02d}":
            raise ThenAtomizationError("inventory claim IDs must be consecutive C01, C02, ...")
        _nonempty_string(claim.get("claim"), f"{path}.claim")
        _validate_exact_spans(claim.get("evidence_spans"), then, f"{path}.evidence_spans")
        claim_ids.add(claim["claim_id"])
    for index, raw in enumerate(relations, start=1):
        path = f"$.claim_inventory.relations[{index - 1}]"
        relation = _mapping(raw, path)
        _exact_keys(
            relation,
            {"relation_id", "relation", "left_claim_id", "right_claim_id", "evidence_spans"},
            path,
        )
        if relation.get("relation_id") != f"Q{index:02d}":
            raise ThenAtomizationError("inventory relation IDs must be consecutive Q01, Q02, ...")
        if relation.get("relation") not in RELATIONS:
            raise ThenAtomizationError(f"{path}.relation is invalid")
        if relation.get("left_claim_id") not in claim_ids or relation.get("right_claim_id") not in claim_ids:
            raise ThenAtomizationError(f"{path} references an unknown claim")
        if relation["left_claim_id"] == relation["right_claim_id"]:
            raise ThenAtomizationError(f"{path} must connect different claims")
        _validate_exact_spans(relation.get("evidence_spans"), then, f"{path}.evidence_spans")
    return result


def validate_then_task_response(
    packet: Mapping[str, Any], task_name: str, response: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one saved model response without requiring the other tasks."""

    checked = validate_then_atomization_packet(packet)
    if task_name == "source_alignment":
        return _validate_alignment(response, checked)
    if task_name == "predicate_partition":
        return _validate_partition(response, checked["gwt_context"]["then"])
    if task_name == "claim_inventory":
        return _validate_inventory(response, checked["gwt_context"]["then"])
    raise ThenAtomizationError(f"unknown task_name: {task_name}")


def canonicalize_then_task_response(
    task_name: str, response: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Canonicalize narrowly defined, semantics-free local-ID variants.

    Partition relations use ``Rxx`` while the independent inventory uses
    ``Qxx``.  Some model responses copy the partition prefix into an otherwise
    valid inventory.  Inventory relation IDs are not referenced by any other
    response field, so an exact consecutive ``R01, R02, ...`` sequence can be
    safely renamed without changing a claim or relation.  No other malformed
    IDs are repaired here; validation remains strict for them.
    """

    result = deepcopy(dict(_mapping(response, "$.response")))
    changes: list[dict[str, str]] = []
    if task_name != "claim_inventory":
        return result, changes
    relations = result.get("relations")
    if not isinstance(relations, list) or not relations:
        return result, changes
    observed = [
        relation.get("relation_id") if isinstance(relation, Mapping) else None
        for relation in relations
    ]
    copied_partition_ids = [f"R{index:02d}" for index in range(1, len(relations) + 1)]
    if observed != copied_partition_ids:
        return result, changes
    for index, relation in enumerate(relations, start=1):
        old_id = relation["relation_id"]
        new_id = f"Q{index:02d}"
        relation["relation_id"] = new_id
        changes.append(
            {
                "field": f"relations[{index - 1}].relation_id",
                "from": old_id,
                "to": new_id,
                "reason": "inventory_relation_namespace",
            }
        )
    return result, changes


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[\w$]+|<=|>=|!=|==|<|>", value.casefold()))


def _claim_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_text = _normalized(str(left.get("claim", "")))
    right_text = _normalized(str(right.get("claim", "")))
    if left_text and right_text and (left_text in right_text or right_text in left_text):
        return True
    for left_span in left.get("evidence_spans", []):
        for right_span in right.get("evidence_spans", []):
            lhs = _normalized(left_span)
            rhs = _normalized(right_span)
            if lhs and rhs and (lhs in rhs or rhs in lhs):
                return True
    return False


def _audit_partition(
    partition: Mapping[str, Any], inventory: Mapping[str, Any]
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    atoms = partition["atoms"]
    claims = inventory["claims"]
    atom_to_claims = {
        atom["atom_id"]: [claim["claim_id"] for claim in claims if _claim_matches(atom, claim)]
        for atom in atoms
    }
    claim_to_atoms = {
        claim["claim_id"]: [atom["atom_id"] for atom in atoms if _claim_matches(atom, claim)]
        for claim in claims
    }
    findings: list[dict[str, Any]] = []
    for claim_id, atom_ids in claim_to_atoms.items():
        if not atom_ids:
            findings.append({"finding_type": "omitted_claim", "claim_ids": [claim_id], "atom_ids": []})
        elif len(atom_ids) > 1:
            findings.append({"finding_type": "duplicated_claim", "claim_ids": [claim_id], "atom_ids": atom_ids})
    for atom_id, claim_ids in atom_to_claims.items():
        if not claim_ids:
            findings.append({"finding_type": "unsupported_atom", "claim_ids": [], "atom_ids": [atom_id]})
        elif len(claim_ids) > 1:
            findings.append({"finding_type": "fused_claims", "claim_ids": claim_ids, "atom_ids": [atom_id]})

    unique_map = {
        claim_id: atom_ids[0]
        for claim_id, atom_ids in claim_to_atoms.items()
        if len(atom_ids) == 1
    }
    partition_relations = {
        (value["relation"], value["left_atom_id"], value["right_atom_id"]): value["relation_id"]
        for value in partition["relations"]
    }
    matched_partition_relation_ids: set[str] = set()
    for relation in inventory["relations"]:
        left = unique_map.get(relation["left_claim_id"])
        right = unique_map.get(relation["right_claim_id"])
        key = (relation["relation"], left, right)
        if left is None or right is None or key not in partition_relations:
            findings.append(
                {
                    "finding_type": "omitted_relation",
                    "claim_ids": [relation["left_claim_id"], relation["right_claim_id"]],
                    "atom_ids": [value for value in (left, right) if value is not None],
                }
            )
        else:
            matched_partition_relation_ids.add(partition_relations[key])
    for relation in partition["relations"]:
        if relation["relation_id"] not in matched_partition_relation_ids:
            findings.append(
                {
                    "finding_type": "unsupported_relation",
                    "claim_ids": [],
                    "atom_ids": [relation["left_atom_id"], relation["right_atom_id"]],
                }
            )
    return unique_map, findings


def reconcile_then_atomization(
    packet: Mapping[str, Any], responses: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate three independent responses and materialize one audited atom set."""

    checked_packet = validate_then_atomization_packet(packet)
    response = _mapping(responses, "$.responses")
    _exact_keys(
        response,
        {"branch_id", "source_alignment", "predicate_partition", "claim_inventory"},
        "$.responses",
    )
    if response.get("branch_id") != checked_packet["branch_id"]:
        raise ThenAtomizationError("response branch_id does not match packet")
    then = checked_packet["gwt_context"]["then"]
    alignment = _validate_alignment(response["source_alignment"], checked_packet)
    partition = _validate_partition(response["predicate_partition"], then)
    inventory = _validate_inventory(response["claim_inventory"], then)

    reconciliation_findings: list[dict[str, Any]] = []
    claim_to_atom: dict[str, str] = {}
    if partition["resolution_status"] == "resolved" and inventory["resolution_status"] == "resolved":
        claim_to_atom, reconciliation_findings = _audit_partition(partition, inventory)

    if alignment["alignment_status"] in {"conflict", "ambiguous"}:
        status = "blocked_source_alignment"
    elif partition["resolution_status"] == "ambiguous" or inventory["resolution_status"] == "ambiguous":
        status = "needs_review"
    elif reconciliation_findings:
        status = "needs_review"
    elif alignment["alignment_status"] == "partial":
        status = "source_partial"
    else:
        status = "resolved"

    atoms = []
    relations = []
    if partition["resolution_status"] == "resolved":
        atom_id_map = {
            atom["atom_id"]: f"{checked_packet['branch_id']}::{atom['atom_id']}"
            for atom in partition["atoms"]
        }
        for atom in partition["atoms"]:
            atoms.append(
                {
                    "atom_id": atom_id_map[atom["atom_id"]],
                    "source_atom_id": atom["atom_id"],
                    "modality": atom["modality"],
                    "claim": atom["claim"],
                    "evidence_spans": list(atom["evidence_spans"]),
                }
            )
        for relation in partition["relations"]:
            relations.append(
                {
                    "relation_id": f"{checked_packet['branch_id']}::{relation['relation_id']}",
                    "source_relation_id": relation["relation_id"],
                    "relation": relation["relation"],
                    "left_atom_id": atom_id_map[relation["left_atom_id"]],
                    "right_atom_id": atom_id_map[relation["right_atom_id"]],
                    "evidence_spans": list(relation["evidence_spans"]),
                }
            )

    result = {
        "schema_version": ATOM_SET_SCHEMA_VERSION,
        "spec_id": checked_packet["spec_id"],
        "branch_id": checked_packet["branch_id"],
        "packet_fingerprint": checked_packet["packet_fingerprint"],
        "source_then": then,
        "source_alignment": alignment,
        "decomposition_status": status,
        "atoms": atoms,
        "relations": relations,
        "audit": {
            "partition_status": partition["resolution_status"],
            "inventory_status": inventory["resolution_status"],
            "claim_to_atom": claim_to_atom,
            "findings": reconciliation_findings,
        },
        "oracle_classification_performed": False,
        "tool_binding_performed": False,
        "coverage_model_used": False,
        "model_task_accounting": {
            "source_alignment_calls": 1,
            "predicate_partition_calls": 1,
            "claim_inventory_calls": 1,
            "total_calls": 3,
        },
    }
    result["then_atom_set_fingerprint"] = content_sha256(result)
    return result


def reconcile_then_atomization_collection(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets_document = validate_then_atomization_packet_set(
        _mapping(packet_set, "$.packet_set")
    )
    responses_document = _mapping(response_set, "$.response_set")
    if responses_document.get("schema_version") != RESPONSE_SET_SCHEMA_VERSION:
        raise ThenAtomizationError("unsupported response-set schema")
    packets = packets_document.get("packets")
    responses = responses_document.get("responses")
    if not isinstance(packets, list) or not isinstance(responses, list):
        raise ThenAtomizationError("packet-set packets and response-set responses must be arrays")
    response_map: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(responses):
        response = _mapping(raw, f"$.responses[{index}]")
        branch_id = _nonempty_string(response.get("branch_id"), f"$.responses[{index}].branch_id")
        if branch_id in response_map:
            raise ThenAtomizationError(f"duplicate response branch_id: {branch_id}")
        response_map[branch_id] = response
    atom_sets = []
    for raw_packet in packets:
        packet = validate_then_atomization_packet(_mapping(raw_packet, "$.packets[]"))
        branch_id = packet["branch_id"]
        if branch_id not in response_map:
            raise ThenAtomizationError(f"missing responses for branch: {branch_id}")
        atom_sets.append(reconcile_then_atomization(packet, response_map.pop(branch_id)))
    if response_map:
        raise ThenAtomizationError(f"responses reference unknown branches: {sorted(response_map)}")
    status_counts: dict[str, int] = {}
    for result in atom_sets:
        status = result["decomposition_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    result = {
        "schema_version": ATOM_COLLECTION_SCHEMA_VERSION,
        "packet_set_fingerprint": packets_document.get("packet_set_fingerprint"),
        "atom_sets": atom_sets,
        "summary": {
            "branch_count": len(atom_sets),
            "atom_count": sum(len(item["atoms"]) for item in atom_sets),
            "relation_count": sum(len(item["relations"]) for item in atom_sets),
            "status_counts": dict(sorted(status_counts.items())),
            "model_calls_consumed": len(atom_sets) * 3,
            "coverage_model_calls": 0,
        },
    }
    result["collection_fingerprint"] = content_sha256(result)
    return result


def reconcile_then_atomization_files(
    packet_set_path: str | Path,
    response_set_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    packet_set = json.loads(Path(packet_set_path).read_text(encoding="utf-8"))
    response_set = json.loads(Path(response_set_path).read_text(encoding="utf-8"))
    result = reconcile_then_atomization_collection(packet_set, response_set)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
