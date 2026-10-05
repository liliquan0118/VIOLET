"""v1.1.0 expectation compiler plus optional separate user-request context.

Versioned copy preserves Step 1/2 frozen dependencies. Policies and output
schemas are unchanged; input without user_request reproduces v1 exactly.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .artifacts import content_sha256
from .oracle_requirement_acceptance_v1 import (
    validate_accepted_oracle_requirement_set,
)
from .then_atomization import ThenAtomizationError


POLICY_VERSION = "agentspectesting.oracle-expectation-policy/v0.1"
EXPECTATION_SET_VERSION = "agentspectesting.oracle-expectation-set/v0.1"
EXPECTATION_VERSION = "agentspectesting.oracle-expectation/v0.1"

NORMATIVE_MODES = frozenset({"required", "forbidden", "permitted"})
COMPILATION_STATUSES = frozenset(
    {"compiled", "non_decisive", "needs_adjudication"}
)
OBSERVATION_OPERATORS = frozenset(
    {"present", "absent", "predicate_true", "relation_true", "non_decisive"}
)
QUANTIFIERS = frozenset(
    {"exists", "none", "all_matching_events", "each_target_event", "not_applicable"}
)

_TOKEN = re.compile(r"[a-z0-9]+")
_TOKEN_STOP = frozenset({"a", "an", "agent", "the", "to", "tool"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ThenAtomizationError(f"{path} must be a non-empty array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ThenAtomizationError(f"{path} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ThenAtomizationError(f"{path} must not contain duplicates")
    return list(value)


def validate_expectation_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    policy = deepcopy(dict(_mapping(value, "$expectation_policy")))
    if policy.get("schema_version") != POLICY_VERSION:
        raise ThenAtomizationError("unsupported oracle expectation policy")
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"]:
        raise ThenAtomizationError("expectation policy_id must be non-empty")
    prefixes = _mapping(policy.get("modality_prefixes"), "$.modality_prefixes")
    if set(prefixes) != NORMATIVE_MODES:
        raise ThenAtomizationError("expectation modality prefixes are incomplete")
    all_prefixes = []
    for mode in sorted(NORMATIVE_MODES):
        all_prefixes.extend(_string_list(prefixes.get(mode), f"$.modality_prefixes.{mode}"))
    lowered = [prefix.casefold() for prefix in all_prefixes]
    if len(lowered) != len(set(lowered)):
        raise ThenAtomizationError("expectation modality prefixes overlap exactly")
    request = _mapping(policy.get("request_detection"), "$.request_detection")
    if set(request) != {"explicit_markers", "action_alignment"}:
        raise ThenAtomizationError("expectation request_detection fields are invalid")
    _string_list(request.get("explicit_markers"), "$.request_detection.explicit_markers")
    if request.get("action_alignment") != "tool_name_tokens_subset_of_when":
        raise ThenAtomizationError("unsupported request action-alignment policy")
    permission = _mapping(
        policy.get("permission_resolution"), "$.permission_resolution"
    )
    if set(permission) != {"aligned_explicit_request", "otherwise"}:
        raise ThenAtomizationError("expectation permission_resolution fields are invalid")
    if permission.get("aligned_explicit_request") != "required":
        raise ThenAtomizationError("aligned explicit requests must resolve to required")
    if permission.get("otherwise") != "non_decisive":
        raise ThenAtomizationError("unaligned permissions must remain non_decisive")
    return policy


_BICONDITIONAL_MARKERS = ("if and only if", " iff ", " iff,")


def _is_biconditional_then(then: str) -> bool:
    """A Then phrased as an IFF (docs/agentcoveragetesting_reuse_log.md
    section 66, Pattern #4 -- e.g. telecom_038_norm#b0: "must transfer...
    if and only if the request cannot be handled...") states two real,
    opposite-polarity obligations at once (transfer when out of scope; do
    NOT transfer when in scope). One concrete conversation branch's single
    When can realize at most one side, so compiling it via the normal single-
    polarity modality-prefix match silently drops the "only if" half and
    turns the whole rule into an unconditional "must transfer" obligation --
    a real, observed miscompilation, not a hypothetical. Detecting the
    literal "if and only if"/"iff" phrasing is a mechanical, high-precision
    signal (this project has no example of it appearing incidentally); which
    concrete request scenario would realize a "not-in-scope" or "in-scope"
    half is real domain-semantic judgment this function does not attempt --
    it only avoids emitting a false single-polarity obligation, deferring to
    needs_adjudication instead."""

    lowered = f" {then.casefold()} "
    return any(marker in lowered for marker in _BICONDITIONAL_MARKERS)


_CONTRASTIVE_QUOTE_MARKER = re.compile(
    r"(?:different from|differs from|other than|instead of)\s*['\"]", re.IGNORECASE
)


def _has_contrastive_quoted_reference(then: str) -> bool:
    """Kept in sync with oracle_expectation_compiler_v1's own copy (docs/
    agentcoveragetesting_reuse_log.md section 86) -- a Then phrased as
    "must not state ... is different from 'X'" quotes X as the CORRECT,
    compliant reference value being contrasted against, not literal text to
    forbid, so this branch's own "must not state" modality cannot be
    mechanically applied to it without asserting the opposite of correct
    policy. Deferred to needs_adjudication instead."""

    return bool(_CONTRASTIVE_QUOTE_MARKER.search(then))


def _modality(then: str, policy: Mapping[str, Any]) -> tuple[str | None, str | None]:
    matches = []
    for mode, prefixes in policy["modality_prefixes"].items():
        for prefix in prefixes:
            if then.casefold().startswith(prefix.casefold()):
                matches.append((len(prefix), mode, prefix))
    if not matches:
        return None, None
    _, mode, prefix = max(matches)
    return mode, prefix


def _tokens(text: Any) -> set[str]:
    result = set()
    for raw in _TOKEN.findall(str(text or "").casefold().replace("_", " ")):
        if raw in _TOKEN_STOP:
            continue
        token = raw[:-1] if raw.endswith("s") and len(raw) > 3 else raw
        result.add(token)
    return result


def _request_analysis(
    when: str, requirement: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    lowered = when.casefold()
    markers = [
        marker
        for marker in policy["request_detection"]["explicit_markers"]
        if marker.casefold() in lowered
    ]
    if not markers:
        return {
            "status": "not_explicit",
            "matched_marker": None,
            "action_anchor": None,
            "alignment_evidence": None,
        }
    contract = _mapping(requirement.get("observation_contract") or {}, "$.observation_contract")
    tool_name = contract.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        return {
            "status": "explicit_unaligned",
            "matched_marker": markers[0],
            "action_anchor": None,
            "alignment_evidence": "requirement_has_no_exact_tool_anchor",
        }
    anchor_tokens = _tokens(tool_name)
    when_tokens = _tokens(when)
    aligned = bool(anchor_tokens) and anchor_tokens <= when_tokens
    return {
        "status": "explicit_aligned" if aligned else "explicit_unaligned",
        "matched_marker": markers[0],
        "action_anchor": {"kind": "tool_name", "value": tool_name},
        "alignment_evidence": {
            "method": policy["request_detection"]["action_alignment"],
            "anchor_tokens": sorted(anchor_tokens),
            "when_tokens": sorted(when_tokens),
        },
    }


def _observation_expectation(
    requirement: Mapping[str, Any],
    effective_mode: str,
) -> dict[str, Any]:
    kind = requirement.get("requirement_type")
    if effective_mode == "non_decisive":
        return {
            "operator": "non_decisive",
            "quantifier": "not_applicable",
            "requires_observation": False,
        }
    if kind in {"tool_argument", "tool_argument_constraint"}:
        return {
            "operator": "predicate_true",
            "quantifier": "all_matching_events",
            "requires_observation": effective_mode == "required",
        }
    if kind == "temporal_relation":
        return {
            "operator": "relation_true",
            "quantifier": "each_target_event",
            "requires_observation": effective_mode == "required",
        }
    return {
        "operator": "present" if effective_mode == "required" else "absent",
        "quantifier": "exists" if effective_mode == "required" else "none",
        "requires_observation": effective_mode == "required",
    }


def compile_oracle_expectations(
    accepted_requirement_set: Mapping[str, Any],
    expectation_policy: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = validate_accepted_oracle_requirement_set(accepted_requirement_set)
    policy = validate_expectation_policy(expectation_policy)
    branches = []
    normative_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    request_counts: Counter[str] = Counter()
    adjudication_branches = []

    for source_branch in accepted["branches"]:
        branch = _mapping(source_branch, "$.branches[]")
        context = _mapping(branch.get("branch_context"), "$.branch_context")
        gwt = _mapping(context.get("gwt"), "$.branch_context.gwt")
        when = str(gwt.get("when") or "")
        then = str(gwt.get("then") or "")
        # v5 supplies the user request separately from the Agent-side trigger.
        # Absence keeps the exact v1.1.0 behavior; an explicitly empty request
        # must not fall back to a different request inferred from When.
        supplied_request = context.get("user_request")
        request_text = when
        if "user_request" in context:
            supplied_request = _mapping(supplied_request, "$.branch_context.user_request")
            if not isinstance(supplied_request.get("text"), str):
                raise ThenAtomizationError("user_request.text must be a string")
            request_text = supplied_request["text"]
        biconditional_then = _is_biconditional_then(then)
        contrastive_reference_then = (
            not biconditional_then and _has_contrastive_quoted_reference(then)
        )
        default_normative_mode, default_matched_prefix = (
            (None, None)
            if biconditional_then or contrastive_reference_then
            else _modality(then, policy)
        )
        compiled = []
        for requirement in branch["requirements"]:
            request = _request_analysis(request_text, requirement, policy)
            request_counts[request["status"]] += 1
            # Section 127 (docs/agentcoveragetesting_reuse_log.md, kept in
            # sync with the real production fix in
            # oracle_expectation_compiler_v1.py -- see that file's own,
            # fuller comment for the complete rationale): a contrastive-
            # quoted-reference Then only risks a false absent-operator
            # obligation for a requirement whose runtime binding can
            # literally string-match the Then's own quoted reference value.
            # A requirement_type=="semantic_requirement" candidate's real
            # runtime binding is never literal-matched -- it is either bound
            # to a same-branch tool_call by alias, or deferred to the
            # holistic, non-literal semantic_event_matcher_deferred judge
            # lane -- so the deferral is unnecessarily conservative for it.
            contrastive_reference_applies = contrastive_reference_then and (
                requirement.get("requirement_type") != "semantic_requirement"
            )
            if contrastive_reference_then and not contrastive_reference_applies:
                normative_mode, matched_prefix = _modality(then, policy)
            else:
                normative_mode, matched_prefix = default_normative_mode, default_matched_prefix
            effective_mode = normative_mode
            status = "compiled"
            reason = "Then modality directly determines the expected observation."
            permission_rule = None
            if biconditional_then:
                status = "needs_adjudication"
                effective_mode = "non_decisive"
                reason = (
                    "Then is phrased as an if-and-only-if: it states two "
                    "opposite-polarity obligations at once, which this "
                    "branch's single When cannot realize both halves of. "
                    "Deferred rather than compiled as a false single-polarity "
                    "obligation (see docs/agentcoveragetesting_reuse_log.md "
                    "section 66, Pattern #4)."
                )
            elif contrastive_reference_applies:
                status = "needs_adjudication"
                effective_mode = "non_decisive"
                reason = (
                    "Then quotes a reference value inside a 'different "
                    "from'/'other than'/'instead of' contrast -- the quote "
                    "is the CORRECT, compliant value being contrasted "
                    "against, not literal text to forbid. Deferred rather "
                    "than compiled as a false absent-operator obligation "
                    "(see docs/agentcoveragetesting_reuse_log.md section 86)."
                )
            elif normative_mode is None:
                status = "needs_adjudication"
                effective_mode = "non_decisive"
                reason = "Then does not start with any configured modality prefix."
            elif normative_mode == "permitted":
                if request["status"] == "explicit_aligned":
                    effective_mode = policy["permission_resolution"][
                        "aligned_explicit_request"
                    ]
                    permission_rule = "aligned_explicit_request"
                    reason = (
                        "The operation is permitted and the When explicitly requests "
                        "the same tool-anchored action."
                    )
                else:
                    effective_mode = policy["permission_resolution"]["otherwise"]
                    permission_rule = "otherwise"
                    status = "non_decisive"
                    reason = (
                        "Permission without an aligned explicit action request does not "
                        "require presence or absence."
                    )
            expectation = _observation_expectation(requirement, str(effective_mode))
            record = {
                "schema_version": EXPECTATION_VERSION,
                "expectation_id": f"{requirement['requirement_id']}::E01",
                "requirement_id": requirement["requirement_id"],
                "branch_id": branch["branch_id"],
                "compilation_status": status,
                "normative_mode": normative_mode,
                "effective_mode": effective_mode,
                "expected_observation": expectation,
                "derivation": {
                    "then": then,
                    "matched_modality_prefix": matched_prefix,
                    "when": when,
                    "request_analysis": request,
                    "permission_rule": permission_rule,
                    "expectation_policy_id": policy["policy_id"],
                    "reason": reason,
                },
                "source_requirement_fingerprint": requirement[
                    "requirement_fingerprint"
                ],
                "runtime_binding_status": requirement["runtime_binding_status"],
            }
            if "user_request" in context:
                record["derivation"]["request_source"] = "branch_context.user_request"
                record["derivation"]["user_request"] = deepcopy(supplied_request)
                if permission_rule == "aligned_explicit_request":
                    record["derivation"]["reason"] = (
                        "The operation is permitted and the supplied user request "
                        "explicitly requests the same tool-anchored action."
                    )
            record["expectation_fingerprint"] = content_sha256(record)
            compiled.append(record)
            if normative_mode:
                normative_counts[normative_mode] += 1
            status_counts[status] += 1
            operator_counts[expectation["operator"]] += 1
        branch_needs_adjudication = any(
            item["compilation_status"] == "needs_adjudication" for item in compiled
        )
        if branch_needs_adjudication:
            adjudication_branches.append(branch["branch_id"])
        branches.append(
            {
                "branch_id": branch["branch_id"],
                "branch_context": deepcopy(context),
                "expectations": compiled,
                "compilation_status": (
                    "needs_adjudication"
                    if branch_needs_adjudication
                    else (
                        "contains_non_decisive"
                        if any(
                            item["compilation_status"] == "non_decisive"
                            for item in compiled
                        )
                        else "compiled"
                    )
                ),
            }
        )

    expectation_count = sum(len(branch["expectations"]) for branch in branches)
    result = {
        "schema_version": EXPECTATION_SET_VERSION,
        "source_accepted_set_fingerprint": accepted["accepted_set_fingerprint"],
        "expectation_policy_fingerprint": content_sha256(policy),
        "expectation_policy_id": policy["policy_id"],
        "branches": branches,
        "summary": {
            "branch_count": len(branches),
            "expectation_count": expectation_count,
            "normative_mode_counts": {
                mode: normative_counts[mode] for mode in sorted(NORMATIVE_MODES)
            },
            "compilation_status_counts": {
                status: status_counts[status]
                for status in sorted(COMPILATION_STATUSES)
            },
            "observation_operator_counts": {
                operator: operator_counts[operator]
                for operator in sorted(OBSERVATION_OPERATORS)
            },
            "request_status_counts": dict(sorted(request_counts.items())),
            "compiled_branch_count": sum(
                branch["compilation_status"] == "compiled" for branch in branches
            ),
            "adjudication_branch_count": len(adjudication_branches),
        },
        "adjudication_branches": adjudication_branches,
        "next_stage": "runtime_observation_binding",
    }
    result["expectation_set_fingerprint"] = content_sha256(result)
    return result


def validate_oracle_expectation_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$expectations")))
    fingerprint = result.pop("expectation_set_fingerprint", None)
    if (
        result.get("schema_version") != EXPECTATION_SET_VERSION
        or fingerprint != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid oracle expectation set")
    identities = []
    statuses: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    for branch in result.get("branches") or []:
        for raw in branch.get("expectations") or []:
            record = deepcopy(dict(_mapping(raw, "$.expectations[]")))
            record_fingerprint = record.pop("expectation_fingerprint", None)
            if (
                record.get("schema_version") != EXPECTATION_VERSION
                or record_fingerprint != content_sha256(record)
            ):
                raise ThenAtomizationError("invalid oracle expectation record")
            if record.get("branch_id") != branch.get("branch_id"):
                raise ThenAtomizationError("oracle expectation branch mismatch")
            if record.get("compilation_status") not in COMPILATION_STATUSES:
                raise ThenAtomizationError("invalid expectation compilation status")
            if record.get("normative_mode") not in NORMATIVE_MODES | {None}:
                raise ThenAtomizationError("invalid expectation normative mode")
            observation = _mapping(
                record.get("expected_observation"), "$.expected_observation"
            )
            if observation.get("operator") not in OBSERVATION_OPERATORS:
                raise ThenAtomizationError("invalid expected observation operator")
            if observation.get("quantifier") not in QUANTIFIERS:
                raise ThenAtomizationError("invalid expected observation quantifier")
            if not isinstance(observation.get("requires_observation"), bool):
                raise ThenAtomizationError("requires_observation must be boolean")
            identities.append(record.get("expectation_id"))
            statuses[record["compilation_status"]] += 1
            operators[observation["operator"]] += 1
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("oracle expectation IDs must be unique")
    summary = _mapping(result.get("summary"), "$.summary")
    if summary.get("expectation_count") != len(identities):
        raise ThenAtomizationError("oracle expectation count mismatch")
    expected_statuses = {
        status: statuses[status] for status in sorted(COMPILATION_STATUSES)
    }
    if summary.get("compilation_status_counts") != expected_statuses:
        raise ThenAtomizationError("expectation status counts mismatch")
    expected_operators = {
        operator: operators[operator] for operator in sorted(OBSERVATION_OPERATORS)
    }
    if summary.get("observation_operator_counts") != expected_operators:
        raise ThenAtomizationError("expected observation operator counts mismatch")
    result["expectation_set_fingerprint"] = fingerprint
    return result


def compile_oracle_expectations_file(
    *,
    accepted_requirements_path: str | Path,
    expectation_policy_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    accepted = json.loads(Path(accepted_requirements_path).read_text(encoding="utf-8"))
    policy = json.loads(Path(expectation_policy_path).read_text(encoding="utf-8"))
    result = compile_oracle_expectations(accepted, policy)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
