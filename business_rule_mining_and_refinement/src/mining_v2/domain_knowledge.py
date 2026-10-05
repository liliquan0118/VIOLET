"""Domain-knowledge miner (the third source).

Some business logic is never written in the prompt because it is domain common
sense (e.g. "the reservation you cancel must belong to the caller"), yet the
agent must still honor it and its violation is a real bug. This module uses the
model's DOMAIN PRIOR to induce such implicit contracts, driven by tools:

  Stage A — per tool: what implicit constraints must THIS tool satisfy (every
            tool is examined; proposes ALL four kinds incl. ARG).
  Stage B — cross-tool / cumulative: constraints spanning tools or accumulating
            across calls (STATE/ORDER/NORM; ARG is per-call, not cross-tool).

Output: hypothesized specs (confidence="hypothesized", origin="domain_knowledge").
They are reconciled with schema/prompt specs downstream by dedup.py. Coverage is
best-effort (LLM prior), not guaranteed.
"""

from __future__ import annotations

import logging

from src.core.types import AgentUnderTest
from src.mining_v2.common import call_json, specialize, tool_context
from src.mining_v2.spec import Evidence, KINDS, Spec

logger = logging.getLogger(__name__)

_CROSS_KINDS = ("STATE", "ORDER", "NORM")   # cross-tool contracts are never ARG

_PERTOOL_SYSTEM = """\
You are a {domain} domain expert. Using your knowledge of how {domain} systems
must behave, infer the IMPLICIT constraints that THIS ONE tool must satisfy —
rules a correct {domain} agent must honor that the prompt leaves UNSTATED (domain
common sense), whose violation is a genuine business-logic bug.

Classify each by WHAT it constrains:
- ARG   : this tool's OWN argument value(s) — a value format/range, or a relation
          between its parameters, that domain sense implies but the schema does
          not already declare (e.g. an airport code is 3 uppercase letters; a
          future date is not in the past; origin != destination; used <= total).
- STATE : a resource this tool touches must be in some state — it belongs to the
          user (ownership), it exists, it is in an eligible/valid status.
- ORDER : something must be looked up / verified / confirmed / collected before
          this tool may be called.
- NORM  : a behavioral obligation tied to this tool (what the agent must or must
          not say/do around it).

Rules:
- Infer from THIS tool's semantics (name, params, read/write nature). A public
  lookup/search touching no user-owned resource often needs only ARG (or nothing)
  — then keep it short or return an empty list.
- Do NOT restate a rule already in KNOWN RULES; do NOT restate what the schema's
  declared enum/type already guarantees.
- Each proposal MUST state how_violated (a concrete way the agent would break it)
  and which param(s)/resource of this tool it is about.
- Prefer a few well-grounded constraints over many weak ones. Output ONLY JSON."""

_PERTOOL_USER = """\
DOMAIN: {domain}
TOOL UNDER AUDIT: {tool_line}

ALL TOOLS (context):
{tool_list}

KNOWN RULES (do not restate):
{found}

Return JSON:
{{
  "contracts": [
    {{
      "rule_text": "<one sentence>",
      "kind": "ARG|STATE|ORDER|NORM",
      "params": ["<param of this tool it concerns, if any>"],
      "how_violated": "<concrete violation>",
      "rationale": "<the domain-knowledge reason>"
    }}
  ]
}}"""

_CROSS_SYSTEM = """\
You are a {domain} domain expert. Infer CROSS-TOOL and CUMULATIVE implicit
constraints the agent must honor — where a single violation spans MULTIPLE tools
or ACCUMULATES across calls — that the prompt leaves UNSTATED and whose violation
is a genuine bug. Shapes (not content): one tool's effect forbids a later call
on the same resource; a quantity summed across calls has a cap; a resource, once
released/closed/cancelled, must not be reused or modified.

Only STATE / ORDER / NORM. Do not restate a KNOWN rule. Each proposal MUST state
how_violated and the tools involved. Output ONLY JSON."""

_CROSS_USER = """\
DOMAIN: {domain}

TOOLS ([rw] name(params): description):
{tool_list}

KNOWN RULES (do not restate):
{found}

Return JSON:
{{
  "contracts": [
    {{
      "rule_text": "<one sentence>",
      "kind": "STATE|ORDER|NORM",
      "relevant_tools": ["<tool>", ...],
      "how_violated": "<concrete cross-tool/cumulative violation>",
      "rationale": "<the domain-knowledge reason>"
    }}
  ]
}}"""


def mine_domain_knowledge(
    agent: AgentUnderTest,
    known: list[Spec],
    *,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> list[Spec]:
    """Induce implicit contracts from the model's domain prior, driven by tools.

    `known` = specs already found (prompt + schema) so this pass does not restate
    them. Returns hypothesized specs, per-tool-specialized.
    """
    tool_names = {t.name for t in agent.tools}
    tl = tool_context(agent)
    valid_params = {t.name: {p.name for p in t.parameters} for t in agent.tools}
    tool_lines = {ln.strip().split("] ", 1)[-1].split("(", 1)[0]: ln.strip()
                  for ln in tl.split("\n")}

    specs: list[Spec] = []
    counter = 0

    def _known_str() -> str:
        base = [f"  - [{s.kind}] ({','.join(s.relevant_tools) or 'agent'}) {s.rule_text}"
                for s in known]
        base += [f"  - [{s.kind}] ({','.join(s.relevant_tools) or 'agent'}) {s.rule_text}"
                 for s in specs]
        return "\n".join(base) or "  (none)"

    def _emit(rule_text, kind, tools, params_by_tool, cross, how, rationale):
        nonlocal counter
        counter += 1
        bindings = [{"tool": t, "params": params_by_tool.get(t, [])} for t in tools]
        note = f"how_violated: {how}. {rationale}".strip()
        specs.append(Spec(
            spec_id=f"h{counter:03d}",
            target_action=(tools[0] if tools else "agent"),
            rule_text=rule_text, kind=kind, clauses=[rule_text],
            relevant_tools=tools, bindings=bindings, cross_tool=cross,
            source_rule=f"{agent.domain or 'x'}_h{counter:03d}",
            confidence="hypothesized", origin="domain_knowledge",
            evidence=[Evidence(source="domain_knowledge", quote=note)],
        ))

    # ── Stage A: per tool ─────────────────────────────────────────────────────
    for t in agent.tools:
        system = _PERTOOL_SYSTEM.format(domain=agent.domain or "general")
        user = _PERTOOL_USER.format(
            domain=agent.domain or "general",
            tool_line=tool_lines.get(t.name, t.name),
            tool_list=tl, found=_known_str(),
        )
        data = call_json(system, user, model, temperature, max_tokens=2000)
        for h in data.get("contracts", []):
            kind = str(h.get("kind", "")).upper()
            rule_text = (h.get("rule_text") or "").strip()
            if kind not in KINDS or not rule_text:
                continue
            params = [p for p in h.get("params", []) if p in valid_params.get(t.name, set())]
            _emit(rule_text, kind, [t.name], {t.name: params}, False,
                  h.get("how_violated", ""), h.get("rationale", ""))

    n_pertool = counter

    # ── Stage B: cross-tool / cumulative ──────────────────────────────────────
    system = _CROSS_SYSTEM.format(domain=agent.domain or "general")
    user = _CROSS_USER.format(domain=agent.domain or "general", tool_list=tl, found=_known_str())
    data = call_json(system, user, model, temperature, max_tokens=2500)
    for h in data.get("contracts", []):
        kind = str(h.get("kind", "")).upper()
        rule_text = (h.get("rule_text") or "").strip()
        if kind not in _CROSS_KINDS or not rule_text:
            continue
        tools = [t for t in h.get("relevant_tools", []) if t in tool_names]
        if not tools:
            continue
        _emit(rule_text, kind, tools, {}, True,
              h.get("how_violated", ""), h.get("rationale", ""))

    logger.info("Domain knowledge: %d per-tool + %d cross-tool = %d hypotheses",
                n_pertool, counter - n_pertool, len(specs))
    return specialize(specs)
