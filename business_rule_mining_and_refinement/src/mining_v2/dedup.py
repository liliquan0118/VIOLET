"""Spec de-duplication.

Both the schema (struct_miner) and the system prompt (prompt_miner) can now emit
ARG specs, so the same argument constraint may appear twice — once as a schema
template ("Argument 'origin' ... must be uppercase") and once in natural language
from the prompt ("origin must be a valid uppercase IATA code"). String matching
is unreliable across these forms, so we reconcile them with a small LLM pass that
clusters equivalent specs and keeps one representative per cluster.

Only ARG specs are de-duplicated (that is where struct/prompt overlap). STATE /
ORDER / NORM come only from the prompt and are left untouched. Within a cluster,
the structural spec is preferred (precise, hallucination-free); otherwise the
grounded ("stated") spec over a "hypothesized" one.
"""

from __future__ import annotations

import json
import logging

from src.mining_v2.spec import Spec

logger = logging.getLogger(__name__)

_CONF_RANK = {"structural": 0, "stated": 1, "hypothesized": 2}  # lower = keep


def _merge_evidence(dst: Spec, src: Spec) -> None:
    """Fold a dropped duplicate's evidence into the kept spec, so dropping a
    spec never loses sentence-level quote coverage."""
    seen = {(e.quote, e.anchor) for e in dst.evidence}
    for e in src.evidence:
        if (e.quote, e.anchor) not in seen:
            dst.evidence.append(e)
            seen.add((e.quote, e.anchor))

_DEDUP_SYSTEM = """\
You are given a list of ARGUMENT-level constraints on tool calls, each with an
index, the tool it applies to, and its text. Some are duplicates: they express
the SAME constraint on the SAME argument of the SAME tool, just worded
differently (e.g. a schema form vs a natural-language form).

Group indices that are duplicates of each other. Two constraints are duplicates
ONLY if they constrain the same argument of the same tool to the same thing.
Different arguments, different tools, or genuinely different limits are NOT
duplicates. When unsure, do NOT group them.

Output ONLY a JSON object:
{"duplicate_groups": [[<idx>, <idx>, ...], ...]}
List only groups with 2+ members; singletons are implied."""


def _dedup_arg_specs(specs: list[Spec], model: str, temperature: float) -> list[Spec]:
    arg_specs = [s for s in specs if s.kind == "ARG"]
    other = [s for s in specs if s.kind != "ARG"]
    if len(arg_specs) < 2:
        return specs

    lines = []
    for i, s in enumerate(arg_specs):
        tools = ",".join(s.relevant_tools) or "?"
        lines.append(f"{i}. [{tools}] ({s.confidence}) {s.rule_text}")
    user = "ARG constraints:\n" + "\n".join(lines)

    from src.core.llm_client import get_llm_client
    client = get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=2000,
            messages=[
                {"role": "system", "content": _DEDUP_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        logger.warning("ARG dedup failed, keeping all: %s", e)
        return specs

    # Build kept set: for each duplicate group keep the best-ranked spec; drop rest
    dropped: set[int] = set()
    n_groups = 0
    for grp in data.get("duplicate_groups", []):
        idxs = [i for i in grp if isinstance(i, int) and 0 <= i < len(arg_specs)]
        if len(idxs) < 2:
            continue
        n_groups += 1
        keep = min(idxs, key=lambda i: _CONF_RANK.get(arg_specs[i].confidence, 3))
        for i in idxs:
            if i != keep:
                dropped.add(i)
                _merge_evidence(arg_specs[keep], arg_specs[i])

    kept_arg = [s for i, s in enumerate(arg_specs) if i not in dropped]
    logger.info("ARG dedup: %d ARG specs → %d (%d dups removed in %d groups)",
                len(arg_specs), len(kept_arg), len(dropped), n_groups)
    return other + kept_arg


# ── Redundant-hypothesis removal (hypothesized vs grounded) ──────────────────

_HYP_SYSTEM = """\
You are given GROUNDED business rules (stated in the agent's prompt) and
HYPOTHESIZED rules (guessed from domain knowledge). Mark a hypothesized rule
REDUNDANT only if a grounded rule imposes the SAME OBLIGATION on the SAME tool
scope — the same tool(s), and where the rules bind specific parameters, the
same parameters. The same obligation on a DIFFERENT tool is NOT redundant.

Judge by the OBLIGATION, not by shared topic words. Two rules about the same
tool are NOT redundant if they demand different things. Be precise:

- "must obtain the reservation id" (collect an input) is a DIFFERENT obligation
  from "the reservation must belong to the user" (ownership check) — NOT redundant.
- "confirm the facts before compensation" is DIFFERENT from "obtain explicit user
  confirmation before an irreversible write" — NOT redundant.
- "if any flight already flown, transfer" already states the cancellable-state
  precondition — a hypothesis re-asserting "reservation must be in a cancellable
  state" IS redundant.

Default to KEEPING a hypothesis. Mark redundant only when a grounded rule clearly
demands the very same thing. Output ONLY a JSON object: {"redundant": [<idx>, ...]}"""


def _drop_redundant_hypotheses(specs: list[Spec], model: str, temperature: float) -> list[Spec]:
    # Only Pass-3 (STATE/ORDER/NORM) hypotheses are reconciled here; ARG
    # hypotheses were already handled by the ARG dedup pass.
    hyp = [s for s in specs if s.confidence == "hypothesized" and s.kind in ("STATE", "ORDER", "NORM")]
    rest = [s for s in specs if s not in hyp]
    grounded = [s for s in rest if s.confidence != "hypothesized"]
    if not hyp or not grounded:
        return specs

    # Compare against grounded STATE/ORDER/NORM (Pass-3 hypotheses are these kinds)
    def _scope(s: Spec) -> str:
        if s.bindings:
            return ",".join(
                f"{b['tool']}({','.join(b.get('params', []))})" if b.get("params")
                else b["tool"] for b in s.bindings)
        return ",".join(s.relevant_tools) or "agent"

    g_relevant = [s for s in grounded if s.kind in ("STATE", "ORDER", "NORM")]
    g_lines = "\n".join(
        f"- [{s.kind}] ({_scope(s)}) {s.rule_text}" for s in g_relevant) or "(none)"
    h_lines = "\n".join(
        f"{i}. [{s.kind}] ({_scope(s)}) {s.rule_text}" for i, s in enumerate(hyp))
    user = f"GROUNDED rules:\n{g_lines}\n\nHYPOTHESIZED rules:\n{h_lines}"

    from src.core.llm_client import get_llm_client
    client = get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=1000,
            messages=[
                {"role": "system", "content": _HYP_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        logger.warning("Hypothesis dedup failed, keeping all: %s", e)
        return specs

    redundant = {i for i in data.get("redundant", []) if isinstance(i, int) and 0 <= i < len(hyp)}
    kept_hyp = [s for i, s in enumerate(hyp) if i not in redundant]
    logger.info("Hypothesis dedup: %d STATE/ORDER/NORM hypotheses → %d (%d redundant with grounded)",
                len(hyp), len(kept_hyp), len(redundant))
    return rest + kept_hyp


# ── Grounded internal dedup (same obligation AND same tool scope) ────────────

_GROUNDED_SYSTEM = """\
You are given GROUNDED business rules (STATE/ORDER/NORM) recovered from an agent's
system prompt. The same rule can appear twice because it was stated in two places
(e.g. once in a general section and once under a specific tool).

Return groups of indices that are TRUE DUPLICATES. Two rules are duplicates ONLY
when they impose the SAME obligation AND cover the SAME tool scope.

Be STRICT and CONSERVATIVE:
- Different tools are DIFFERENT rules, even if the obligation is the same kind.
  "reservation must belong to the user before cancel" and "... before update" are
  DIFFERENT (different target tools) — NOT duplicates.
- Different obligations are different rules even on the same tool.
- When unsure, do NOT group. Only group near-verbatim restatements of one rule.

Output ONLY a JSON object: {"duplicate_groups": [[<idx>, <idx>, ...], ...]}"""


def _dedup_grounded(specs: list[Spec], model: str, temperature: float) -> list[Spec]:
    grounded = [s for s in specs
                if s.confidence != "hypothesized" and s.kind in ("STATE", "ORDER", "NORM")]
    other = [s for s in specs if s not in grounded]
    if len(grounded) < 2:
        return specs

    lines = []
    for i, s in enumerate(grounded):
        tools = ",".join(s.relevant_tools) or "agent"
        lines.append(f"{i}. [{s.kind}] ({tools}) {s.rule_text}")
    user = "GROUNDED rules:\n" + "\n".join(lines)

    from src.core.llm_client import get_llm_client
    client = get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=1500,
            messages=[{"role": "system", "content": _GROUNDED_SYSTEM},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        logger.warning("Grounded dedup failed, keeping all: %s", e)
        return specs

    dropped: set[int] = set()
    n_groups = 0
    for grp in data.get("duplicate_groups", []):
        idxs = [i for i in grp if isinstance(i, int) and 0 <= i < len(grounded)]
        if len(idxs) < 2:
            continue
        n_groups += 1
        keep = min(idxs)  # keep the earliest (all grounded, equal confidence)
        for i in idxs:
            if i != keep:
                dropped.add(i)
                _merge_evidence(grounded[keep], grounded[i])

    kept = [s for i, s in enumerate(grounded) if i not in dropped]
    logger.info("Grounded dedup: %d grounded STATE/ORDER/NORM → %d (%d dups in %d groups)",
                len(grounded), len(kept), len(dropped), n_groups)
    return other + kept


def dedup(specs: list[Spec], *, model: str, temperature: float = 0.0) -> list[Spec]:
    """Central de-duplication across the whole spec set:
      1. ARG cross-source (schema / prompt / domain-knowledge)
      2. grounded internal (same obligation AND same tool scope)
      3. hypothesized STATE/ORDER/NORM redundant with a grounded rule
    """
    specs = _dedup_arg_specs(specs, model, temperature)
    specs = _dedup_grounded(specs, model, temperature)
    specs = _drop_redundant_hypotheses(specs, model, temperature)
    return specs
