"""Staged, input-adequate semantic compilation for GWT specifications.

Version 0.3 deliberately separates three questions:

1. What normative requirements are stated by one source anchor?
2. What outcomes are literally asserted by one GWT Then?
3. How does that branch map the asserted outcomes to the source requirements?

The first two model tasks are independent and can run in parallel.  The third
is not prepared until both validated answers exist.  Sibling completeness is
then computed mechanically from explicit IDs rather than asked of a model.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .then_atomization import MODALITIES, RELATIONS, RESOLUTION_STATUSES, ThenAtomizationError


STAGE1_PACKET_SET_VERSION = "agentspectesting.gwt-semantic-stage1-packets/v0.3"
SOURCE_PACKET_VERSION = "agentspectesting.source-semantics-packet/v0.3"
THEN_PACKET_VERSION = "agentspectesting.then-semantics-packet/v0.3"
STAGE1_RESPONSE_SET_VERSION = "agentspectesting.gwt-semantic-stage1-responses/v0.3"
MAPPING_PACKET_SET_VERSION = "agentspectesting.gwt-semantic-mapping-packets/v0.3"
MAPPING_PACKET_VERSION = "agentspectesting.gwt-semantic-mapping-packet/v0.3"
MAPPING_RESPONSE_SET_VERSION = "agentspectesting.gwt-semantic-mapping-responses/v0.3"
SEMANTIC_RESULT_VERSION = "agentspectesting.gwt-semantic-result/v0.3"

SOURCE_TASK = "source_requirement_extraction"
THEN_TASK = "then_assertion_extraction"
MAPPING_TASK = "branch_requirement_mapping"

REFERENCE_ROLES = frozenset({"condition", "temporal_anchor", "scope_anchor"})
BRANCH_RELATIONS = frozenset({"direct_case", "complement_case", "irrelevant", "ambiguous"})
COVERAGE_STATUSES = frozenset({"covered", "omitted", "not_expected", "ambiguous"})
SUPPORT_STATUSES = frozenset({"supported", "contradicted", "unsupported", "ambiguous"})
MAPPING_OVERALL_STATUSES = frozenset({"aligned", "issues_found", "ambiguous"})


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


def _exact_spans(value: Any, source: str, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ThenAtomizationError(f"{path} must be a non-empty array")
    result = []
    for index, raw in enumerate(value):
        span = _nonempty(raw, f"{path}[{index}]")
        if span not in source:
            raise ThenAtomizationError(f"{path}[{index}] is not an exact source substring")
        result.append(span)
    if len(result) != len(set(result)):
        raise ThenAtomizationError(f"{path} must not contain duplicates")
    return result


def _source_pool(context: Mapping[str, Any]) -> list[str]:
    values = [context.get("rule_text", ""), context.get("source_rule", "")]
    values.extend(context.get("clauses") or [])
    values.extend(
        evidence.get("quote", "")
        for evidence in context.get("evidence") or []
        if isinstance(evidence, Mapping)
    )
    return [value for value in values if isinstance(value, str) and value]


def _source_spans(value: Any, context: Mapping[str, Any], path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ThenAtomizationError(f"{path} must be a non-empty array")
    pool = _source_pool(context)
    result = []
    for index, raw in enumerate(value):
        span = _nonempty(raw, f"{path}[{index}]")
        if not any(span in source for source in pool):
            raise ThenAtomizationError(f"{path}[{index}] is absent from the source anchor")
        result.append(span)
    if len(result) != len(set(result)):
        raise ThenAtomizationError(f"{path} must not contain duplicates")
    return result


def _source_context(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "origin": _nonempty(spec.get("origin"), "$.spec.origin"),
        "confidence": _nonempty(spec.get("confidence"), "$.spec.confidence"),
        "rule_text": _nonempty(spec.get("rule_text"), "$.spec.rule_text"),
        "clauses": deepcopy(spec.get("clauses") or []),
        "source_rule": spec.get("source_rule", ""),
        "evidence": deepcopy(spec.get("evidence") or []),
    }


def _fingerprinted(payload: dict[str, Any]) -> dict[str, Any]:
    payload["packet_fingerprint"] = content_sha256(payload)
    return payload


def build_gwt_semantic_stage1_packets(
    spec_document: Mapping[str, Any],
    *,
    selected_branch_ids: Sequence[str],
    source_path: str | None = None,
) -> dict[str, Any]:
    """Prepare the two independent first-stage task families.

    All sibling branches of a selected spec are included so later completeness
    checks do not confuse sibling allocation with omission.
    """

    if not selected_branch_ids:
        raise ThenAtomizationError("v0.3 requires a non-empty explicit branch selection")
    selected = list(selected_branch_ids)
    if len(selected) != len(set(selected)):
        raise ThenAtomizationError("selected branch IDs must be unique")
    specs = spec_document.get("specs")
    if not isinstance(specs, list):
        raise ThenAtomizationError("$.specs must be an array")
    branch_owner: dict[str, str] = {}
    for spec_index, raw_spec in enumerate(specs):
        spec = _mapping(raw_spec, f"$.specs[{spec_index}]")
        spec_id = _nonempty(spec.get("spec_id"), f"$.specs[{spec_index}].spec_id")
        for branch_index, raw_gwt in enumerate(spec.get("gwt") or []):
            gwt = _mapping(raw_gwt, f"$.specs[{spec_index}].gwt[{branch_index}]")
            branch_id = _nonempty(gwt.get("branch_id"), "$.gwt[].branch_id")
            if branch_id in branch_owner:
                raise ThenAtomizationError(f"duplicate branch ID: {branch_id}")
            branch_owner[branch_id] = spec_id
    missing = [branch_id for branch_id in selected if branch_id not in branch_owner]
    if missing:
        raise ThenAtomizationError(f"selected branches are missing: {missing}")
    selected_spec_ids = {branch_owner[branch_id] for branch_id in selected}

    source_packets: list[dict[str, Any]] = []
    then_packets: list[dict[str, Any]] = []
    sibling_ids: list[str] = []
    for raw_spec in specs:
        spec = _mapping(raw_spec, "$.specs[]")
        spec_id = spec.get("spec_id")
        if spec_id not in selected_spec_ids:
            continue
        context = _source_context(spec)
        source_packets.append(
            _fingerprinted(
                {
                    "schema_version": SOURCE_PACKET_VERSION,
                    "task_name": SOURCE_TASK,
                    "spec_id": spec_id,
                    "task_contract": {
                        "question": "What normative requirements are explicitly stated by this source anchor?",
                        "provided_inputs": ["source_context"],
                        "non_goals": [
                            "judge any GWT branch",
                            "infer tool behavior",
                            "search other specs",
                            "decide test observability",
                        ],
                    },
                    "source_context": context,
                    "task_input": {"spec_id": spec_id, "source_context": deepcopy(context)},
                }
            )
        )
        for raw_gwt in spec.get("gwt") or []:
            gwt = _mapping(raw_gwt, "$.gwt[]")
            branch_id = _nonempty(gwt.get("branch_id"), "$.gwt[].branch_id")
            sibling_ids.append(branch_id)
            context_gwt = {
                "given": _nonempty(gwt.get("given"), f"$.gwt[{branch_id}].given"),
                "when": _nonempty(gwt.get("when"), f"$.gwt[{branch_id}].when"),
                "then": _nonempty(gwt.get("then"), f"$.gwt[{branch_id}].then"),
            }
            then_packets.append(
                _fingerprinted(
                    {
                        "schema_version": THEN_PACKET_VERSION,
                        "task_name": THEN_TASK,
                        "spec_id": spec_id,
                        "branch_id": branch_id,
                        "selected_for_test": branch_id in selected,
                        "task_contract": {
                            "question": "What outcomes are literally asserted by this Then, and which mentioned events are only references?",
                            "provided_inputs": ["gwt_context"],
                            "non_goals": [
                                "judge support from the source spec",
                                "bind tools",
                                "decide fixture reachability",
                                "decide test observability",
                            ],
                        },
                        "gwt_context": context_gwt,
                        "task_input": {
                            "branch_id": branch_id,
                            "gwt_context": deepcopy(context_gwt),
                        },
                    }
                )
            )

    payload = {
        "schema_version": STAGE1_PACKET_SET_VERSION,
        "source": {"path": source_path},
        "selected_branch_ids": selected,
        "included_sibling_branch_ids": sibling_ids,
        "source_packets": source_packets,
        "then_packets": then_packets,
        "summary": {
            "selected_spec_count": len(source_packets),
            "selected_branch_count": len(selected),
            "included_sibling_branch_count": len(then_packets),
            "expected_stage1_model_calls": len(source_packets) + len(then_packets),
            "calls_by_task": {
                SOURCE_TASK: len(source_packets),
                THEN_TASK: len(then_packets),
            },
            "coverage_model_calls": 0,
        },
    }
    payload["packet_set_fingerprint"] = content_sha256(payload)
    return payload


def validate_gwt_semantic_stage1_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != STAGE1_PACKET_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.3 stage1 packet-set schema")
    if fingerprint != content_sha256(result):
        raise ThenAtomizationError("v0.3 stage1 packet-set fingerprint mismatch")
    source_packets = result.get("source_packets")
    then_packets = result.get("then_packets")
    if not isinstance(source_packets, list) or not isinstance(then_packets, list):
        raise ThenAtomizationError("v0.3 stage1 packet arrays are missing")
    for expected_version, expected_task, packets in (
        (SOURCE_PACKET_VERSION, SOURCE_TASK, source_packets),
        (THEN_PACKET_VERSION, THEN_TASK, then_packets),
    ):
        for packet in packets:
            checked = dict(_mapping(packet, "$.packets[]"))
            packet_fingerprint = checked.pop("packet_fingerprint", None)
            if checked.get("schema_version") != expected_version:
                raise ThenAtomizationError("v0.3 task packet schema mismatch")
            if checked.get("task_name") != expected_task:
                raise ThenAtomizationError("v0.3 task name mismatch")
            if packet_fingerprint != content_sha256(checked):
                raise ThenAtomizationError("v0.3 task packet fingerprint mismatch")
    expected = len(source_packets) + len(then_packets)
    if result.get("summary", {}).get("expected_stage1_model_calls") != expected:
        raise ThenAtomizationError("v0.3 stage1 call count mismatch")
    if result.get("included_sibling_branch_ids") != [item["branch_id"] for item in then_packets]:
        raise ThenAtomizationError("v0.3 sibling branch index mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def render_gwt_semantic_stage1_prompt(packet: Mapping[str, Any], template: str) -> str:
    task = packet.get("task_name")
    placeholder = {
        SOURCE_TASK: "{source_requirement_input}",
        THEN_TASK: "{then_assertion_input}",
    }.get(task)
    if placeholder is None or placeholder not in template:
        raise ThenAtomizationError("stage1 prompt template/task mismatch")
    return template.replace(
        placeholder,
        json.dumps(packet["task_input"], ensure_ascii=False, indent=2),
    )


def validate_source_semantics_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.source_semantics")))
    _exact_keys(result, {"spec_id", "resolution_status", "requirements", "reason"}, "$.source_semantics")
    if result.get("spec_id") != packet.get("spec_id"):
        raise ThenAtomizationError("source-semantics spec_id mismatch")
    status = result.get("resolution_status")
    if status not in RESOLUTION_STATUSES:
        raise ThenAtomizationError("source-semantics resolution_status is invalid")
    _nonempty(result.get("reason"), "$.source_semantics.reason")
    requirements = result.get("requirements")
    if not isinstance(requirements, list):
        raise ThenAtomizationError("source-semantics requirements must be an array")
    if status == "resolved" and not requirements:
        raise ThenAtomizationError("resolved source semantics requires requirements")
    if status == "ambiguous" and requirements:
        raise ThenAtomizationError("ambiguous source semantics must not select requirements")
    context = packet["source_context"]
    for index, raw in enumerate(requirements, start=1):
        path = f"$.source_semantics.requirements[{index - 1}]"
        requirement = _mapping(raw, path)
        _exact_keys(
            requirement,
            {"requirement_id", "modality", "claim", "evidence_spans", "applicability"},
            path,
        )
        if requirement.get("requirement_id") != f"S{index:02d}":
            raise ThenAtomizationError("source requirement IDs must be consecutive S01, S02, ...")
        if requirement.get("modality") not in MODALITIES:
            raise ThenAtomizationError(f"{path}.modality is invalid")
        _nonempty(requirement.get("claim"), f"{path}.claim")
        requirement["evidence_spans"] = _source_spans(
            requirement.get("evidence_spans"), context, f"{path}.evidence_spans"
        )
        applicability = _mapping(requirement.get("applicability"), f"{path}.applicability")
        _exact_keys(applicability, {"mode", "alternatives"}, f"{path}.applicability")
        mode = applicability.get("mode")
        alternatives = applicability.get("alternatives")
        if mode not in {"unconditional", "any_of"} or not isinstance(alternatives, list):
            raise ThenAtomizationError(f"{path}.applicability is invalid")
        if mode == "unconditional" and alternatives:
            raise ThenAtomizationError("unconditional applicability must have no alternatives")
        if mode == "any_of" and not alternatives:
            raise ThenAtomizationError("any_of applicability requires alternatives")
        for alternative_index, raw_alternative in enumerate(alternatives, start=1):
            alternative_path = f"{path}.applicability.alternatives[{alternative_index - 1}]"
            alternative = _mapping(raw_alternative, alternative_path)
            _exact_keys(alternative, {"alternative_id", "condition", "evidence_spans"}, alternative_path)
            if alternative.get("alternative_id") != f"A{alternative_index:02d}":
                raise ThenAtomizationError("applicability alternative IDs must be consecutive A01, A02, ...")
            _nonempty(alternative.get("condition"), f"{alternative_path}.condition")
            alternative["evidence_spans"] = _source_spans(
                alternative.get("evidence_spans"), context, f"{alternative_path}.evidence_spans"
            )
    return result


def validate_then_semantics_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.then_semantics")))
    _exact_keys(
        result,
        {"branch_id", "resolution_status", "assertions", "reference_events", "relations", "reason"},
        "$.then_semantics",
    )
    if result.get("branch_id") != packet.get("branch_id"):
        raise ThenAtomizationError("then-semantics branch_id mismatch")
    status = result.get("resolution_status")
    if status not in RESOLUTION_STATUSES:
        raise ThenAtomizationError("then-semantics resolution_status is invalid")
    _nonempty(result.get("reason"), "$.then_semantics.reason")
    assertions = result.get("assertions")
    references = result.get("reference_events")
    relations = result.get("relations")
    if not all(isinstance(value, list) for value in (assertions, references, relations)):
        raise ThenAtomizationError("then semantic collections must be arrays")
    if status == "resolved" and not assertions:
        raise ThenAtomizationError("resolved Then semantics requires at least one assertion")
    if status == "ambiguous" and (assertions or references or relations):
        raise ThenAtomizationError("ambiguous Then semantics must not select a parse")
    then = packet["gwt_context"]["then"]
    entity_ids: set[str] = set()
    for index, raw in enumerate(assertions, start=1):
        path = f"$.then_semantics.assertions[{index - 1}]"
        assertion = _mapping(raw, path)
        _exact_keys(assertion, {"assertion_id", "modality", "claim", "evidence_spans"}, path)
        if assertion.get("assertion_id") != f"T{index:02d}":
            raise ThenAtomizationError("Then assertion IDs must be consecutive T01, T02, ...")
        if assertion.get("modality") not in MODALITIES:
            raise ThenAtomizationError(f"{path}.modality is invalid")
        _nonempty(assertion.get("claim"), f"{path}.claim")
        assertion["evidence_spans"] = _exact_spans(
            assertion.get("evidence_spans"), then, f"{path}.evidence_spans"
        )
        entity_ids.add(assertion["assertion_id"])
    for index, raw in enumerate(references, start=1):
        path = f"$.then_semantics.reference_events[{index - 1}]"
        reference = _mapping(raw, path)
        _exact_keys(reference, {"reference_id", "role", "event", "evidence_spans"}, path)
        if reference.get("reference_id") != f"E{index:02d}":
            raise ThenAtomizationError("Then reference IDs must be consecutive E01, E02, ...")
        if reference.get("role") not in REFERENCE_ROLES:
            raise ThenAtomizationError(f"{path}.role is invalid")
        _nonempty(reference.get("event"), f"{path}.event")
        reference["evidence_spans"] = _exact_spans(
            reference.get("evidence_spans"), then, f"{path}.evidence_spans"
        )
        entity_ids.add(reference["reference_id"])
    for index, raw in enumerate(relations, start=1):
        path = f"$.then_semantics.relations[{index - 1}]"
        relation = _mapping(raw, path)
        _exact_keys(
            relation,
            {"relation_id", "relation", "left_entity_id", "right_entity_id", "evidence_spans"},
            path,
        )
        if relation.get("relation_id") != f"R{index:02d}":
            raise ThenAtomizationError("Then relation IDs must be consecutive R01, R02, ...")
        if relation.get("relation") not in RELATIONS:
            raise ThenAtomizationError(f"{path}.relation is invalid")
        if relation.get("left_entity_id") not in entity_ids or relation.get("right_entity_id") not in entity_ids:
            raise ThenAtomizationError(f"{path} references an unknown assertion/reference")
        if relation["left_entity_id"] == relation["right_entity_id"]:
            raise ThenAtomizationError(f"{path} must connect different entities")
        relation["evidence_spans"] = _exact_spans(
            relation.get("evidence_spans"), then, f"{path}.evidence_spans"
        )
    return result


def canonicalize_then_semantics_response(
    response: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Canonicalize only unambiguous local entity-ID spelling variants.

    Models naturally use A for "assertion" and sometimes omit zero padding.
    When IDs exactly follow list order, they can be renamed to the contract's
    T/E/R namespaces and every relation endpoint can be updated mechanically.
    Arbitrary, duplicated, or out-of-order IDs are left unchanged for strict
    validation to reject.
    """

    result = deepcopy(dict(_mapping(response, "$.then_semantics")))
    assertions = result.get("assertions")
    references = result.get("reference_events")
    relations = result.get("relations")
    if not all(isinstance(value, list) for value in (assertions, references, relations)):
        return result, []
    changes: list[dict[str, str]] = []
    entity_map: dict[str, str] = {}

    def canonicalize_sequence(
        values: list[Any], field: str, accepted_prefixes: set[str], canonical_prefix: str
    ) -> bool:
        observed = []
        for index, item in enumerate(values, start=1):
            if not isinstance(item, Mapping):
                return False
            raw = item.get(field)
            if not isinstance(raw, str):
                return False
            match = re.fullmatch(r"([A-Za-z])(\d+)", raw)
            if (
                match is None
                or match.group(1).upper() not in accepted_prefixes
                or int(match.group(2)) != index
            ):
                return False
            observed.append(raw)
        if len(observed) != len(set(observed)):
            return False
        for index, (item, old_id) in enumerate(zip(values, observed), start=1):
            new_id = f"{canonical_prefix}{index:02d}"
            if old_id != new_id:
                item[field] = new_id
                entity_map[old_id] = new_id
                changes.append(
                    {
                        "field": f"{field}[{index - 1}]",
                        "from": old_id,
                        "to": new_id,
                        "reason": "local_entity_id_namespace",
                    }
                )
            else:
                entity_map[old_id] = old_id
        return True

    if not canonicalize_sequence(assertions, "assertion_id", {"A", "T"}, "T"):
        return result, []
    if not canonicalize_sequence(references, "reference_id", {"E"}, "E"):
        return result, []
    if not canonicalize_sequence(relations, "relation_id", {"R"}, "R"):
        return result, []
    for index, relation in enumerate(relations):
        for field in ("left_entity_id", "right_entity_id"):
            old_id = relation.get(field)
            if old_id in entity_map and entity_map[old_id] != old_id:
                relation[field] = entity_map[old_id]
                changes.append(
                    {
                        "field": f"relations[{index}].{field}",
                        "from": old_id,
                        "to": entity_map[old_id],
                        "reason": "local_entity_id_reference",
                    }
                )
    return result, changes


def validate_gwt_semantic_stage1_response_set(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_gwt_semantic_stage1_packet_set(packet_set)
    responses = deepcopy(dict(_mapping(response_set, "$.responses")))
    _exact_keys(responses, {"schema_version", "source_responses", "then_responses"}, "$.responses")
    if responses.get("schema_version") != STAGE1_RESPONSE_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.3 stage1 response-set schema")
    source_packet_map = {item["spec_id"]: item for item in packets["source_packets"]}
    then_packet_map = {item["branch_id"]: item for item in packets["then_packets"]}
    source_responses = responses.get("source_responses")
    then_responses = responses.get("then_responses")
    if not isinstance(source_responses, list) or not isinstance(then_responses, list):
        raise ThenAtomizationError("v0.3 stage1 response arrays are missing")
    checked_sources: dict[str, dict[str, Any]] = {}
    for raw in source_responses:
        spec_id = raw.get("spec_id") if isinstance(raw, Mapping) else None
        if spec_id not in source_packet_map or spec_id in checked_sources:
            raise ThenAtomizationError(f"unknown or duplicate source response: {spec_id}")
        checked_sources[spec_id] = validate_source_semantics_response(source_packet_map[spec_id], raw)
    checked_thens: dict[str, dict[str, Any]] = {}
    for raw in then_responses:
        branch_id = raw.get("branch_id") if isinstance(raw, Mapping) else None
        if branch_id not in then_packet_map or branch_id in checked_thens:
            raise ThenAtomizationError(f"unknown or duplicate Then response: {branch_id}")
        checked_thens[branch_id] = validate_then_semantics_response(then_packet_map[branch_id], raw)
    if set(checked_sources) != set(source_packet_map):
        raise ThenAtomizationError("stage1 responses do not cover all selected specs")
    if set(checked_thens) != set(then_packet_map):
        raise ThenAtomizationError("stage1 responses do not cover all included siblings")
    return {
        "schema_version": STAGE1_RESPONSE_SET_VERSION,
        "source_responses": [checked_sources[item["spec_id"]] for item in packets["source_packets"]],
        "then_responses": [checked_thens[item["branch_id"]] for item in packets["then_packets"]],
    }


def build_gwt_semantic_mapping_packets(
    stage1_packet_set: Mapping[str, Any], stage1_response_set: Mapping[str, Any]
) -> dict[str, Any]:
    packets = validate_gwt_semantic_stage1_packet_set(stage1_packet_set)
    responses = validate_gwt_semantic_stage1_response_set(packets, stage1_response_set)
    source_map = {item["spec_id"]: item for item in responses["source_responses"]}
    then_map = {item["branch_id"]: item for item in responses["then_responses"]}
    mapping_packets = []
    for then_packet in packets["then_packets"]:
        spec_id = then_packet["spec_id"]
        branch_id = then_packet["branch_id"]
        task_input = {
            "spec_id": spec_id,
            "branch_id": branch_id,
            "source_requirements": deepcopy(source_map[spec_id]["requirements"]),
            "gwt_context": deepcopy(then_packet["gwt_context"]),
            "then_semantics": deepcopy(then_map[branch_id]),
        }
        mapping_packets.append(
            _fingerprinted(
                {
                    "schema_version": MAPPING_PACKET_VERSION,
                    "task_name": MAPPING_TASK,
                    "spec_id": spec_id,
                    "branch_id": branch_id,
                    "selected_for_test": then_packet["selected_for_test"],
                    "task_contract": {
                        "question": "How does this one branch map its parsed Then assertions to the supplied source requirements under its Given/When?",
                        "provided_inputs": [
                            "source_requirements",
                            "gwt_context",
                            "then_semantics",
                        ],
                        "non_goals": [
                            "reparse the source",
                            "reparse the Then",
                            "judge sibling completeness",
                            "search other specs",
                            "decide test observability",
                        ],
                    },
                    "task_input": task_input,
                }
            )
        )
    payload = {
        "schema_version": MAPPING_PACKET_SET_VERSION,
        "stage1_packet_set_fingerprint": packets["packet_set_fingerprint"],
        "selected_branch_ids": deepcopy(packets["selected_branch_ids"]),
        "included_sibling_branch_ids": deepcopy(packets["included_sibling_branch_ids"]),
        "mapping_packets": mapping_packets,
        "summary": {
            "expected_mapping_model_calls": len(mapping_packets),
            "coverage_model_calls": 0,
        },
    }
    payload["packet_set_fingerprint"] = content_sha256(payload)
    return payload


def validate_gwt_semantic_mapping_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != MAPPING_PACKET_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.3 mapping packet-set schema")
    if fingerprint != content_sha256(result):
        raise ThenAtomizationError("v0.3 mapping packet-set fingerprint mismatch")
    packets = result.get("mapping_packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("v0.3 mapping packets are missing")
    for packet in packets:
        checked = dict(_mapping(packet, "$.mapping_packets[]"))
        packet_fingerprint = checked.pop("packet_fingerprint", None)
        if checked.get("schema_version") != MAPPING_PACKET_VERSION or checked.get("task_name") != MAPPING_TASK:
            raise ThenAtomizationError("v0.3 mapping packet schema/task mismatch")
        if packet_fingerprint != content_sha256(checked):
            raise ThenAtomizationError("v0.3 mapping packet fingerprint mismatch")
    if result.get("summary", {}).get("expected_mapping_model_calls") != len(packets):
        raise ThenAtomizationError("v0.3 mapping call count mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def render_gwt_semantic_mapping_prompt(packet: Mapping[str, Any], template: str) -> str:
    if packet.get("task_name") != MAPPING_TASK or "{branch_mapping_input}" not in template:
        raise ThenAtomizationError("mapping prompt template/task mismatch")
    return template.replace(
        "{branch_mapping_input}",
        json.dumps(packet["task_input"], ensure_ascii=False, indent=2),
    )


def validate_branch_mapping_response(
    packet: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.branch_mapping")))
    _exact_keys(
        result,
        {"branch_id", "overall_status", "requirement_mappings", "assertion_mappings", "reason"},
        "$.branch_mapping",
    )
    if result.get("branch_id") != packet.get("branch_id"):
        raise ThenAtomizationError("branch-mapping branch_id mismatch")
    if result.get("overall_status") not in MAPPING_OVERALL_STATUSES:
        raise ThenAtomizationError("branch-mapping overall_status is invalid")
    _nonempty(result.get("reason"), "$.branch_mapping.reason")
    source_requirements = packet["task_input"]["source_requirements"]
    then_semantics = packet["task_input"]["then_semantics"]
    requirement_ids = [item["requirement_id"] for item in source_requirements]
    assertion_ids = [item["assertion_id"] for item in then_semantics["assertions"]]
    alternative_ids = {
        item["requirement_id"]: {
            alternative["alternative_id"]
            for alternative in item["applicability"]["alternatives"]
        }
        for item in source_requirements
    }
    requirement_mappings = result.get("requirement_mappings")
    assertion_mappings = result.get("assertion_mappings")
    if not isinstance(requirement_mappings, list) or not isinstance(assertion_mappings, list):
        raise ThenAtomizationError("branch-mapping mapping fields must be arrays")
    if [item.get("requirement_id") for item in requirement_mappings] != requirement_ids:
        raise ThenAtomizationError("branch mapping must cover requirements once in input order")
    if [item.get("assertion_id") for item in assertion_mappings] != assertion_ids:
        raise ThenAtomizationError("branch mapping must cover assertions once in input order")
    for index, raw in enumerate(requirement_mappings):
        path = f"$.branch_mapping.requirement_mappings[{index}]"
        item = _mapping(raw, path)
        _exact_keys(
            item,
            {
                "requirement_id",
                "branch_relation",
                "matched_alternative_ids",
                "assertion_ids",
                "coverage_status",
                "reason",
            },
            path,
        )
        if item.get("branch_relation") not in BRANCH_RELATIONS:
            raise ThenAtomizationError(f"{path}.branch_relation is invalid")
        if item.get("coverage_status") not in COVERAGE_STATUSES:
            raise ThenAtomizationError(f"{path}.coverage_status is invalid")
        matched = item.get("matched_alternative_ids")
        linked = item.get("assertion_ids")
        if not isinstance(matched, list) or not set(matched).issubset(alternative_ids[item["requirement_id"]]):
            raise ThenAtomizationError(f"{path}.matched_alternative_ids is invalid")
        if not isinstance(linked, list) or not set(linked).issubset(assertion_ids):
            raise ThenAtomizationError(f"{path}.assertion_ids is invalid")
        relation = item["branch_relation"]
        coverage = item["coverage_status"]
        if relation == "irrelevant" and (
            coverage != "not_expected" or linked or matched
        ):
            raise ThenAtomizationError(
                f"{path}: irrelevant requirement must be not_expected with no links"
            )
        if relation in {"direct_case", "complement_case"} and coverage == "not_expected":
            raise ThenAtomizationError(
                f"{path}: governing requirement cannot be not_expected"
            )
        if coverage == "covered" and not linked:
            raise ThenAtomizationError(f"{path}: covered requirement requires assertions")
        if coverage == "omitted" and linked:
            raise ThenAtomizationError(f"{path}: omitted requirement cannot link assertions")
        _nonempty(item.get("reason"), f"{path}.reason")
    for index, raw in enumerate(assertion_mappings):
        path = f"$.branch_mapping.assertion_mappings[{index}]"
        item = _mapping(raw, path)
        _exact_keys(item, {"assertion_id", "support_status", "requirement_ids", "reason"}, path)
        if item.get("support_status") not in SUPPORT_STATUSES:
            raise ThenAtomizationError(f"{path}.support_status is invalid")
        linked = item.get("requirement_ids")
        if not isinstance(linked, list) or not set(linked).issubset(requirement_ids):
            raise ThenAtomizationError(f"{path}.requirement_ids is invalid")
        if item["support_status"] in {"supported", "contradicted"} and not linked:
            raise ThenAtomizationError(f"{path} requires a source requirement reference")
        if item["support_status"] == "unsupported" and linked:
            raise ThenAtomizationError(f"{path}: unsupported assertion must not claim source support")
        _nonempty(item.get("reason"), f"{path}.reason")
    has_issue = any(
        item["coverage_status"] in {"omitted", "ambiguous"} for item in requirement_mappings
    ) or any(
        item["support_status"] != "supported" for item in assertion_mappings
    )
    if result["overall_status"] == "aligned" and has_issue:
        raise ThenAtomizationError("aligned branch mapping contains an issue")
    if result["overall_status"] == "issues_found" and not has_issue:
        raise ThenAtomizationError("issues_found branch mapping contains no issue")
    if result["overall_status"] == "ambiguous" and not any(
        item["coverage_status"] == "ambiguous" for item in requirement_mappings
    ) and not any(item["support_status"] == "ambiguous" for item in assertion_mappings):
        raise ThenAtomizationError("ambiguous branch mapping contains no ambiguous row")
    return result


def reconcile_gwt_semantic_mappings(
    stage1_packet_set: Mapping[str, Any],
    stage1_response_set: Mapping[str, Any],
    mapping_packet_set: Mapping[str, Any],
    mapping_response_set: Mapping[str, Any],
) -> dict[str, Any]:
    stage1_packets = validate_gwt_semantic_stage1_packet_set(stage1_packet_set)
    stage1_responses = validate_gwt_semantic_stage1_response_set(stage1_packets, stage1_response_set)
    mapping_packets = validate_gwt_semantic_mapping_packet_set(mapping_packet_set)
    responses = deepcopy(dict(_mapping(mapping_response_set, "$.mapping_responses")))
    _exact_keys(responses, {"schema_version", "branch_mappings"}, "$.mapping_responses")
    if responses.get("schema_version") != MAPPING_RESPONSE_SET_VERSION:
        raise ThenAtomizationError("unsupported v0.3 mapping response-set schema")
    raw_mappings = responses.get("branch_mappings")
    if not isinstance(raw_mappings, list):
        raise ThenAtomizationError("branch_mappings must be an array")
    packet_map = {item["branch_id"]: item for item in mapping_packets["mapping_packets"]}
    checked_map: dict[str, dict[str, Any]] = {}
    for raw in raw_mappings:
        branch_id = raw.get("branch_id") if isinstance(raw, Mapping) else None
        if branch_id not in packet_map or branch_id in checked_map:
            raise ThenAtomizationError(f"unknown or duplicate branch mapping: {branch_id}")
        checked_map[branch_id] = validate_branch_mapping_response(packet_map[branch_id], raw)
    if set(checked_map) != set(packet_map):
        raise ThenAtomizationError("mapping responses do not cover every included sibling")

    source_map = {item["spec_id"]: item for item in stage1_responses["source_responses"]}
    then_map = {item["branch_id"]: item for item in stage1_responses["then_responses"]}
    spec_allocations = []
    for spec_id, source in source_map.items():
        sibling_packets = [item for item in mapping_packets["mapping_packets"] if item["spec_id"] == spec_id]
        allocations = []
        for requirement in source["requirements"]:
            requirement_id = requirement["requirement_id"]
            branch_entries = [
                next(
                    item
                    for item in checked_map[packet["branch_id"]]["requirement_mappings"]
                    if item["requirement_id"] == requirement_id
                )
                for packet in sibling_packets
            ]
            allocated_branches = [
                packet["branch_id"]
                for packet, entry in zip(sibling_packets, branch_entries)
                if entry["branch_relation"] in {"direct_case", "complement_case"}
            ]
            alternatives = {
                alternative_id
                for entry in branch_entries
                for alternative_id in entry["matched_alternative_ids"]
            }
            expected_alternatives = {
                item["alternative_id"] for item in requirement["applicability"]["alternatives"]
            }
            allocations.append(
                {
                    "requirement_id": requirement_id,
                    "allocated_branch_ids": allocated_branches,
                    "uncovered_alternative_ids": sorted(expected_alternatives - alternatives),
                    "allocated": bool(allocated_branches),
                }
            )
        spec_allocations.append({"spec_id": spec_id, "requirements": allocations})

    selected = set(stage1_packets["selected_branch_ids"])
    branches = []
    for packet in mapping_packets["mapping_packets"]:
        branch_id = packet["branch_id"]
        semantics = then_map[branch_id]
        branches.append(
            {
                "spec_id": packet["spec_id"],
                "branch_id": branch_id,
                "selected_for_test": branch_id in selected,
                "mapping_status": checked_map[branch_id]["overall_status"],
                "assertions": deepcopy(semantics["assertions"]),
                "reference_events": deepcopy(semantics["reference_events"]),
                "relations": deepcopy(semantics["relations"]),
                "mapping": checked_map[branch_id],
                "mechanical_testability_flags": {
                    "permission_only": bool(semantics["assertions"])
                    and all(item["modality"] == "permitted" for item in semantics["assertions"]),
                    "has_reference_events": bool(semantics["reference_events"]),
                },
            }
        )
    result = {
        "schema_version": SEMANTIC_RESULT_VERSION,
        "stage1_packet_set_fingerprint": stage1_packets["packet_set_fingerprint"],
        "mapping_packet_set_fingerprint": mapping_packets["packet_set_fingerprint"],
        "spec_allocations": spec_allocations,
        "branches": branches,
        "summary": {
            "spec_count": len(source_map),
            "included_sibling_branch_count": len(branches),
            "selected_branch_count": len(selected),
            "stage1_model_calls": len(stage1_packets["source_packets"]) + len(stage1_packets["then_packets"]),
            "mapping_model_calls": len(mapping_packets["mapping_packets"]),
            "coverage_model_calls": 0,
        },
    }
    result["result_fingerprint"] = content_sha256(result)
    return result


def build_gwt_semantic_stage1_packets_file(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    selected_branch_ids: Sequence[str],
) -> dict[str, Any]:
    source = Path(spec_path)
    result = build_gwt_semantic_stage1_packets(
        json.loads(source.read_text(encoding="utf-8")),
        selected_branch_ids=selected_branch_ids,
        source_path=str(source.resolve()),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
