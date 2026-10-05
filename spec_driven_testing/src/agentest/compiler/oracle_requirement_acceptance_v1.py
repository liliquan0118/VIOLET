"""Deterministic acceptance and runtime-bindability inventory for selected oracles."""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .artifacts import content_sha256
from .oracle_requirement_pipeline_v7 import REQUIREMENT_SET_VERSION
from .profiles import validate_sut_adapter_profile
from .then_atomization import ThenAtomizationError


ACCEPTED_SET_VERSION = "agentspectesting.accepted-oracle-requirement-set/v0.1"
ACCEPTED_REQUIREMENT_VERSION = "agentspectesting.accepted-oracle-requirement/v0.1"
BINDING_STATUSES = frozenset(
    {"bound", "resolvable", "semantic_only", "needs_adjudication"}
)

_TOOL_CALL = re.compile(
    r"\b(?:make\s+)?(?:a\s+)?tool\s+call\s+to\s+([A-Za-z_][A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
_SEND_MESSAGE = re.compile(
    r"\bsend\s+the\s+message\s+(['\"])(.+?)\1(?:\s+to\s+the\s+user)?\b",
    re.IGNORECASE,
)
_NORMAL_TOKEN = re.compile(r"[a-z0-9_]+")
_OVERLAP_STOP = frozenset(
    {"a", "an", "and", "are", "be", "being", "of", "or", "the", "to", "with"}
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _norm(text: Any) -> str:
    return " ".join(_NORMAL_TOKEN.findall(str(text or "").casefold()))


def _informative_tokens(text: Any) -> set[str]:
    return {
        token
        for token in _NORMAL_TOKEN.findall(str(text or "").casefold())
        if token not in _OVERLAP_STOP
    }


def _validate_requirement_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$requirements")))
    fingerprint = result.pop("requirement_set_fingerprint", None)
    if (
        result.get("schema_version") != REQUIREMENT_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid v0.7 oracle requirement set")
    branches = result.get("branches")
    if not isinstance(branches, list) or not branches:
        raise ThenAtomizationError("oracle requirement branches must be a non-empty array")
    branch_ids = []
    selected_count = 0
    identities = []
    for raw_branch in branches:
        branch = _mapping(raw_branch, "$.branches[]")
        branch_id = branch.get("branch_id")
        if not isinstance(branch_id, str) or not branch_id:
            raise ThenAtomizationError("oracle requirement branch_id is invalid")
        branch_ids.append(branch_id)
        selected = branch.get("selected_requirements")
        if not isinstance(selected, list):
            raise ThenAtomizationError("selected_requirements must be an array")
        for raw_selected in selected:
            selected_record = _mapping(raw_selected, "$.selected_requirements[]")
            candidate = _mapping(selected_record.get("candidate"), "$.candidate")
            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ThenAtomizationError("selected candidate_id is invalid")
            identities.append((branch_id, candidate_id))
            selected_count += 1
    if len(branch_ids) != len(set(branch_ids)):
        raise ThenAtomizationError("oracle requirement branch IDs must be unique")
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("selected oracle candidate identities must be unique")
    if result.get("summary", {}).get("selected_count") != selected_count:
        raise ThenAtomizationError("oracle selected requirement count mismatch")
    result["requirement_set_fingerprint"] = fingerprint
    return result


def _tool_index(tool_catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    tools = tool_catalog.get("tools")
    if not isinstance(tools, list):
        raise ThenAtomizationError("tool catalog tools must be an array")
    result = {}
    for raw in tools:
        tool = _mapping(raw, "$.tool_catalog.tools[]")
        name = tool.get("tool_name")
        if not isinstance(name, str) or not name or name in result:
            raise ThenAtomizationError("tool catalog contains invalid or duplicate tool_name")
        result[name] = tool
    return result


def _semantic_unit_ids(document: Mapping[str, Any]) -> set[str]:
    units = document.get("semantic_units")
    if not isinstance(units, list):
        raise ThenAtomizationError("semantic unit model semantic_units must be an array")
    result = set()
    for raw in units:
        unit = _mapping(raw, "$.semantic_units[]")
        unit_id = unit.get("semantic_unit_id")
        if not isinstance(unit_id, str) or not unit_id or unit_id in result:
            raise ThenAtomizationError("semantic unit model contains invalid or duplicate IDs")
        result.add(unit_id)
    return result


def _tool_endpoint_paths(tool: Mapping[str, Any]) -> set[str]:
    return {
        endpoint.get("source_path")
        for endpoint in tool.get("observable_endpoints") or []
        if isinstance(endpoint, Mapping)
        and endpoint.get("source_kind") == "tool_argument"
        and isinstance(endpoint.get("source_path"), str)
    }


def _scope_signature(contract: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(contract.get("scope_parameters") or []))


def _classify(
    candidate: Mapping[str, Any],
    *,
    tools: Mapping[str, Mapping[str, Any]],
    semantic_ids: set[str],
    channels: set[str],
) -> tuple[str, str, dict[str, Any] | None, str]:
    kind = str(candidate.get("candidate_kind") or "")
    text = str(candidate.get("requirement_text") or "")
    contract = _mapping(candidate.get("observation_contract") or {}, "$.observation_contract")
    refs = _mapping(candidate.get("source_refs") or {}, "$.source_refs")

    def tool_problem(tool_name: Any) -> str | None:
        if "tool_call" not in channels:
            return "SUT profile does not expose the tool_call channel."
        if not isinstance(tool_name, str) or tool_name not in tools:
            return f"Tool {tool_name!r} is absent from the accepted tool catalog."
        return None

    def endpoint_problem(tool_name: str, parameters: list[str]) -> str | None:
        available = _tool_endpoint_paths(tools[tool_name])
        missing = [parameter for parameter in parameters if parameter not in available]
        if missing:
            return f"Tool argument endpoints are missing for: {', '.join(missing)}."
        return None

    if kind == "tool_call":
        tool_name = contract.get("tool_name")
        problem = tool_problem(tool_name)
        if problem:
            return "needs_adjudication", problem, None, f"tool_call::{tool_name}"
        scope = list(contract.get("scope_parameters") or [])
        problem = endpoint_problem(str(tool_name), scope)
        key = f"tool_call::{tool_name}::scope::{','.join(sorted(scope))}"
        if problem:
            return "needs_adjudication", problem, None, key
        return (
            "bound",
            "Tool call and every scope parameter have accepted runtime endpoints.",
            None,
            key,
        )

    if kind == "tool_argument":
        tool_name = contract.get("tool_name")
        parameter = contract.get("parameter")
        problem = tool_problem(tool_name)
        if not problem and isinstance(parameter, str):
            problem = endpoint_problem(str(tool_name), [parameter, *list(_scope_signature(contract))])
        if problem or not isinstance(parameter, str) or not parameter:
            return (
                "needs_adjudication",
                problem or "Tool argument parameter is missing.",
                None,
                f"tool_argument::{tool_name}::{parameter}",
            )
        key = (
            f"tool_argument::{tool_name}::{parameter}::scope::"
            f"{','.join(_scope_signature(contract))}"
        )
        return "bound", "Tool argument has an accepted runtime endpoint.", None, key

    if kind == "tool_argument_constraint":
        tool_name = contract.get("tool_name")
        parameters = contract.get("parameters")
        problem = tool_problem(tool_name)
        if not isinstance(parameters, list) or not parameters or not all(
            isinstance(item, str) and item for item in parameters
        ):
            problem = problem or "Constraint parameters are missing or invalid."
            parameters = []
        if not problem:
            problem = endpoint_problem(
                str(tool_name), [*parameters, *list(_scope_signature(contract))]
            )
        key = (
            f"tool_argument_constraint::{tool_name}::{','.join(parameters)}::"
            f"{_norm(contract.get('constraint_text'))}::scope::"
            f"{','.join(_scope_signature(contract))}"
        )
        if problem:
            return "needs_adjudication", problem, None, key
        return (
            "bound",
            "All arguments in the constraint have accepted runtime endpoints.",
            None,
            key,
        )

    if kind == "assistant_literal":
        literal = contract.get("literal")
        key = f"assistant_literal::{json.dumps(literal, ensure_ascii=False)}"
        if "assistant_message" not in channels:
            return (
                "needs_adjudication",
                "SUT profile does not expose the assistant_message channel.",
                None,
                key,
            )
        if not isinstance(literal, str) or not literal:
            return "needs_adjudication", "Assistant literal is empty.", None, key
        return "bound", "Exact assistant-message literal is directly observable.", None, key

    if kind == "turn_shape_constraint":
        # No tool/endpoint to validate -- this checks whether a single turn's
        # events (grouped by source_message_index, see
        # runtime_observation_binding_v1.py::normalize_tau_messages) exhibit a
        # forbidden shape (e.g. both a tool call and a message at once), so
        # "bound" only requires both channels to be observable, like
        # assistant_literal above but for two channels instead of one.
        key = f"turn_shape_constraint::{_norm(text)}"
        missing = {"assistant_message", "tool_call"} - channels
        if missing:
            return (
                "needs_adjudication",
                f"SUT profile does not expose: {', '.join(sorted(missing))}.",
                None,
                key,
            )
        return (
            "bound",
            "Both assistant_message and tool_call channels are observable; "
            "message-shape co-occurrence is directly observable from the "
            "event stream.",
            None,
            key,
        )

    if kind == "temporal_relation":
        unit_id = contract.get("left_event_semantic_unit_id")
        relation = contract.get("relation")
        right_event = contract.get("right_event")
        key = f"temporal::{unit_id}::{relation}::{right_event}"
        if unit_id not in semantic_ids:
            return (
                "needs_adjudication",
                "Temporal left-event semantic unit is absent from the accepted model.",
                None,
                key,
            )
        if relation != "precedes" or not isinstance(right_event, str) or not right_event:
            return "needs_adjudication", "Temporal relation is incomplete.", None, key
        if not contract.get("runtime_contract_refs"):
            return "needs_adjudication", "Temporal runtime lineage is empty.", None, key
        return (
            "resolvable",
            "Relation and lineage are known; the two runtime event bindings remain to be compiled.",
            {
                "resolver_kind": "temporal_event_binding",
                "left_event_semantic_unit_id": unit_id,
                "relation": relation,
                "right_event": right_event,
            },
            key,
        )

    if kind == "semantic_requirement":
        unit_id = refs.get("semantic_unit_id")
        key = f"semantic::{_norm(text)}"
        if not isinstance(unit_id, str) or unit_id not in semantic_ids:
            return (
                "needs_adjudication",
                "Semantic requirement lineage is absent from the accepted model.",
                None,
                key,
            )
        tool_match = _TOOL_CALL.search(text)
        if tool_match:
            tool_name = tool_match.group(1)
            problem = tool_problem(tool_name)
            tool_key = f"tool_call::{tool_name}::scope::"
            if problem:
                return "needs_adjudication", problem, None, tool_key
            return (
                "resolvable",
                "The semantic text explicitly names a tool call.",
                {"resolver_kind": "explicit_tool_call", "tool_name": tool_name},
                tool_key,
            )
        message_match = _SEND_MESSAGE.search(text)
        if message_match:
            literal = message_match.group(2)
            message_key = f"assistant_literal::{json.dumps(literal, ensure_ascii=False)}"
            if "assistant_message" not in channels:
                return (
                    "needs_adjudication",
                    "SUT profile does not expose the assistant_message channel.",
                    None,
                    message_key,
                )
            return (
                "resolvable",
                "The semantic text contains an explicit assistant-message literal.",
                {"resolver_kind": "explicit_assistant_literal", "literal": literal},
                message_key,
            )
        return (
            "semantic_only",
            "No exact tool, argument, message-literal, or temporal runtime anchor is present.",
            None,
            key,
        )

    return (
        "needs_adjudication",
        f"Candidate kind {kind!r} has no accepted binding rule.",
        None,
        f"unsupported::{kind}::{_norm(text)}",
    )


def accept_oracle_requirements(
    requirement_set: Mapping[str, Any],
    tool_catalog: Mapping[str, Any],
    semantic_unit_model: Mapping[str, Any],
    sut_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Inventory, safely deduplicate, and classify selected oracle requirements."""

    requirements = _validate_requirement_set(requirement_set)
    tools = _tool_index(tool_catalog)
    semantic_ids = _semantic_unit_ids(semantic_unit_model)
    profile = validate_sut_adapter_profile(sut_profile)
    channels = set(profile["observable_channels"])

    accepted_branches = []
    all_statuses: Counter[str] = Counter()
    input_selected_count = 0
    accepted_count = 0
    merged_count = 0
    potential_overlap_count = 0
    adjudication_branches = []

    status_rank = {
        "bound": 4,
        "resolvable": 3,
        "semantic_only": 2,
        "needs_adjudication": 1,
    }
    for raw_branch in requirements["branches"]:
        branch = _mapping(raw_branch, "$.branches[]")
        grouped: dict[str, dict[str, Any]] = {}
        for selected in branch["selected_requirements"]:
            selected_record = _mapping(selected, "$.selected_requirements[]")
            candidate = _mapping(selected_record["candidate"], "$.candidate")
            input_selected_count += 1
            status, reason, resolution_hint, equivalence_key = _classify(
                candidate,
                tools=tools,
                semantic_ids=semantic_ids,
                channels=channels,
            )
            source = {
                "candidate_id": candidate["candidate_id"],
                "candidate_kind": candidate.get("candidate_kind"),
                "candidate_origin": candidate.get("candidate_origin"),
                "requirement_text": candidate.get("requirement_text"),
                "source_refs": deepcopy(candidate.get("source_refs") or {}),
                "selection_reason": selected_record.get("selection_reason"),
            }
            existing = grouped.get(equivalence_key)
            if existing is None:
                grouped[equivalence_key] = {
                    "requirement_text": candidate.get("requirement_text"),
                    "requirement_type": candidate.get("candidate_kind"),
                    "runtime_binding_status": status,
                    "binding_reason": reason,
                    "resolution_hint": deepcopy(resolution_hint),
                    "observation_contract": deepcopy(
                        candidate.get("observation_contract") or {}
                    ),
                    "equivalence_key": equivalence_key,
                    "source_candidates": [source],
                }
                continue
            merged_count += 1
            existing["source_candidates"].append(source)
            if status_rank[status] > status_rank[existing["runtime_binding_status"]]:
                for field, value in {
                    "requirement_text": candidate.get("requirement_text"),
                    "requirement_type": candidate.get("candidate_kind"),
                    "runtime_binding_status": status,
                    "binding_reason": reason,
                    "resolution_hint": deepcopy(resolution_hint),
                    "observation_contract": deepcopy(
                        candidate.get("observation_contract") or {}
                    ),
                }.items():
                    existing[field] = value

        accepted = []
        for index, item in enumerate(grouped.values(), start=1):
            record = {
                "schema_version": ACCEPTED_REQUIREMENT_VERSION,
                "requirement_id": f"{branch['branch_id']}::OR{index:02d}",
                "branch_id": branch["branch_id"],
                **item,
            }
            record["requirement_fingerprint"] = content_sha256(record)
            accepted.append(record)
            all_statuses[record["runtime_binding_status"]] += 1
        overlap_hints = []
        semantic_requirements = [
            item
            for item in accepted
            if item["runtime_binding_status"] == "semantic_only"
        ]
        concrete_tool_calls = [
            item
            for item in accepted
            if item["requirement_type"] == "tool_call"
            and item["runtime_binding_status"] == "bound"
        ]
        concrete_tool_constraints = [
            item
            for item in accepted
            if item["requirement_type"] in ("tool_argument", "tool_argument_constraint")
            and item["runtime_binding_status"] == "bound"
        ]
        for semantic in semantic_requirements:
            semantic_tokens = _informative_tokens(semantic["requirement_text"])
            if len(semantic_tokens) < 3:
                continue
            for concrete in concrete_tool_calls:
                contract = concrete["observation_contract"]
                anchor_tokens = _informative_tokens(
                    f"{contract.get('tool_name', '')} "
                    f"{contract.get('action_description', '')}"
                )
                if semantic_tokens <= anchor_tokens:
                    overlap_hints.append(
                        {
                            "semantic_requirement_id": semantic["requirement_id"],
                            "concrete_requirement_id": concrete["requirement_id"],
                            "signal": "semantic_tokens_contained_in_tool_action_anchor",
                            "disposition": "not_merged_without_equivalence_evidence",
                        }
                    )
            for concrete in concrete_tool_constraints:
                anchor_tokens = _informative_tokens(concrete["requirement_text"])
                if semantic_tokens <= anchor_tokens:
                    overlap_hints.append(
                        {
                            "semantic_requirement_id": semantic["requirement_id"],
                            "concrete_requirement_id": concrete["requirement_id"],
                            "signal": "semantic_tokens_contained_in_bound_constraint_text",
                            "disposition": "not_merged_without_equivalence_evidence",
                        }
                    )
        potential_overlap_count += len(overlap_hints)
        branch_needs_adjudication = any(
            item["runtime_binding_status"] == "needs_adjudication" for item in accepted
        )
        if branch_needs_adjudication:
            adjudication_branches.append(branch["branch_id"])
        accepted_count += len(accepted)
        accepted_branches.append(
            {
                "branch_id": branch["branch_id"],
                "branch_context": deepcopy(branch.get("branch_context") or {}),
                "requirements": accepted,
                "acceptance_status": (
                    # Explicit empty-collection case (docs/agentcoveragetesting_
                    # reuse_log.md section 110): `any()` over an empty `accepted`
                    # list is vacuously False, so without this branch a branch
                    # whose accepted-requirement set is genuinely empty (every
                    # Step3 candidate for this Then was judged not relevant)
                    # would fall through to "accepted" -- a falsely-healthy
                    # status for a branch that in fact carries zero oracle
                    # content. Checked before branch_needs_adjudication/
                    # semantic_requirements on purpose: both are themselves
                    # derived from `accepted`, so when `accepted` is empty they
                    # are always False/[] and would never have produced
                    # anything other than "accepted" anyway -- this is a pure
                    # addition, not a behavior change, for every branch with a
                    # non-empty accepted set.
                    "branch_has_no_oracle_content"
                    if not accepted
                    else (
                        "needs_adjudication"
                        if branch_needs_adjudication
                        else (
                            "accepted_with_semantic_deferred"
                            if semantic_requirements
                            else "accepted"
                        )
                    )
                ),
                "potential_overlap_hints": overlap_hints,
                "summary": {
                    "input_selected_count": len(branch["selected_requirements"]),
                    "accepted_requirement_count": len(accepted),
                    "merged_source_count": len(branch["selected_requirements"])
                    - len(accepted),
                },
            }
        )

    result = {
        "schema_version": ACCEPTED_SET_VERSION,
        "source_requirement_set_fingerprint": requirements[
            "requirement_set_fingerprint"
        ],
        "tool_catalog_fingerprint": content_sha256(tool_catalog),
        "semantic_unit_model_fingerprint": content_sha256(semantic_unit_model),
        "sut_profile_fingerprint": content_sha256(profile),
        "sut_profile_id": profile["profile_id"],
        "branches": accepted_branches,
        "summary": {
            "branch_count": len(accepted_branches),
            "input_selected_count": input_selected_count,
            "accepted_requirement_count": accepted_count,
            "merged_source_count": merged_count,
            "binding_status_counts": {
                status: all_statuses[status] for status in sorted(BINDING_STATUSES)
            },
            "accepted_branch_count": len(accepted_branches)
            - len(adjudication_branches),
            "adjudication_branch_count": len(adjudication_branches),
            "semantic_deferred_branch_count": sum(
                branch["acceptance_status"] == "accepted_with_semantic_deferred"
                for branch in accepted_branches
            ),
            "potential_overlap_count": potential_overlap_count,
        },
        "adjudication_branches": adjudication_branches,
        "semantic_deferred_branches": [
            branch["branch_id"]
            for branch in accepted_branches
            if branch["acceptance_status"] == "accepted_with_semantic_deferred"
        ],
        "next_stage": "expected_outcome_compilation",
    }
    result["accepted_set_fingerprint"] = content_sha256(result)
    return result


def validate_accepted_oracle_requirement_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$accepted_requirements")))
    fingerprint = result.pop("accepted_set_fingerprint", None)
    if (
        result.get("schema_version") != ACCEPTED_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid accepted oracle requirement set")
    identities = []
    statuses: Counter[str] = Counter()
    source_count = 0
    for raw_branch in result.get("branches") or []:
        branch = _mapping(raw_branch, "$.branches[]")
        for raw in branch.get("requirements") or []:
            requirement = deepcopy(dict(_mapping(raw, "$.requirements[]")))
            requirement_fingerprint = requirement.pop("requirement_fingerprint", None)
            if (
                requirement.get("schema_version") != ACCEPTED_REQUIREMENT_VERSION
                or requirement_fingerprint != content_sha256(requirement)
            ):
                raise ThenAtomizationError("invalid accepted oracle requirement")
            if requirement.get("branch_id") != branch.get("branch_id"):
                raise ThenAtomizationError("accepted requirement branch mismatch")
            status = requirement.get("runtime_binding_status")
            if status not in BINDING_STATUSES:
                raise ThenAtomizationError("invalid runtime binding status")
            identities.append(requirement.get("requirement_id"))
            statuses[status] += 1
            sources = requirement.get("source_candidates")
            if not isinstance(sources, list) or not sources:
                raise ThenAtomizationError("accepted requirement has no source candidates")
            source_count += len(sources)
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("accepted requirement IDs must be unique")
    summary = _mapping(result.get("summary"), "$.summary")
    if summary.get("accepted_requirement_count") != len(identities):
        raise ThenAtomizationError("accepted requirement count mismatch")
    if summary.get("input_selected_count") != source_count:
        raise ThenAtomizationError("accepted source candidate count mismatch")
    expected_counts = {status: statuses[status] for status in sorted(BINDING_STATUSES)}
    if summary.get("binding_status_counts") != expected_counts:
        raise ThenAtomizationError("accepted binding status counts mismatch")
    result["accepted_set_fingerprint"] = fingerprint
    return result


def accept_oracle_requirements_file(
    *,
    requirements_path: str | Path,
    tool_catalog_path: str | Path,
    semantic_unit_model_path: str | Path,
    sut_profile_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    result = accept_oracle_requirements(
        load(requirements_path),
        load(tool_catalog_path),
        load(semantic_unit_model_path),
        load(sut_profile_path),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
