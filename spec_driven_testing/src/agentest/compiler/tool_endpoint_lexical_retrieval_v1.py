"""Token-overlap retrieval of tool catalog endpoints/tools relevant to a span of
natural-language text.

Extracted, byte-for-byte behavior preserved, from `given_evidence_source_v2.py`
(originally private `_tokens`/`_endpoint_view`/`_relevant_endpoints`, built for
scoring Given-condition text against the tool catalog). Made public here so a
second caller (`oracle_requirement_pipeline_v7.py`, scoring Then text instead of
Given-condition text, to discover candidate tools without a pre-narrowed
`relevant_tools` list) can depend on the same, independently-tested function
instead of either reaching into another module's private helper or duplicating
the tokenization/stemming/scoring logic a second time. See
docs/oracle_requirement_pipeline_v0_7.md for why this was extracted.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP_WORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "being",
    "by", "can", "cannot", "does", "for", "from", "has", "have", "in",
    "is", "it", "its", "must", "no", "not", "of", "on", "or", "otherwise",
    "that", "the", "their", "this", "to", "was", "were", "will", "with",
    "agent", "airline", "current", "flight", "least", "one", "reservation",
    "specified", "user",
}


def tokens(text: Any) -> set[str]:
    if not isinstance(text, str):
        return set()
    result = set()
    for item in _TOKEN_RE.findall(text.lower().replace("_", " ")):
        variants = {item}
        if len(item) > 4 and item.endswith("s") and not item.endswith("ss"):
            variants.add(item[:-1])
        if len(item) > 5 and item.endswith("ed"):
            variants.update({item[:-2], item[:-1]})
        if len(item) > 6 and item.endswith("ing"):
            variants.update({item[:-3], item[:-3] + "e"})
        result.update(value for value in variants if value not in _STOP_WORDS)
    return result


def endpoint_view(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "endpoint_id": endpoint.get("endpoint_id"),
        "source_kind": endpoint.get("source_kind"),
        "source_path": endpoint.get("source_path"),
        "tool_name": endpoint.get("tool_name"),
        "description": endpoint.get("description") or "",
        "allowed_values": deepcopy(endpoint.get("allowed_values") or []),
    }


def relevant_endpoints(
    tool_catalog: Mapping[str, Any], condition_text: str, limit: int = 8
) -> tuple[list[dict[str, Any]], int, list[str]]:
    query = tokens(condition_text)
    scored = []
    total = 0
    action_names = []
    for tool in tool_catalog.get("tools") or []:
        action_names.append(tool.get("tool_name"))
        for endpoint in tool.get("observable_endpoints") or []:
            total += 1
            haystack = " ".join(
                str(endpoint.get(field) or "")
                for field in ("endpoint_id", "tool_name", "source_path", "description")
            )
            haystack += " " + str(tool.get("description") or "")
            overlap = query & tokens(haystack)
            if overlap:
                scored.append((-len(overlap), str(endpoint.get("endpoint_id")), endpoint))
    matches = []
    seen = set()
    for _, _, item in sorted(scored):
        signature = (
            item.get("source_kind"),
            item.get("description"),
            tuple(item.get("allowed_values") or []),
        )
        if signature in seen:
            continue
        seen.add(signature)
        matches.append(endpoint_view(item))
        if len(matches) == limit:
            break
    return matches, total, sorted(item for item in action_names if isinstance(item, str))
