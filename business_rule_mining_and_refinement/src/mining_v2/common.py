"""Shared helpers for the mining_v2 extractors.

Used by both the prompt miner (Pass 1) and the domain-knowledge miner (Pass 3):
an OpenAI-style JSON call, compact per-tool context, read/write-type inference,
and the per-tool "specialize" split.
"""

from __future__ import annotations

import json
import logging
import re

from src.core.types import AgentUnderTest
from src.mining_v2.spec import Spec

logger = logging.getLogger(__name__)


# ── LLM JSON call ─────────────────────────────────────────────────────────────

# OpenAI reasoning models rename max_tokens -> max_completion_tokens and reject a
# non-default temperature. DeepSeek reasoning models (v4-pro, reasoner) keep the
# max_tokens name and accept temperature, but like all reasoners spend hidden
# reasoning tokens against the SAME budget — so they need extra headroom or the
# JSON gets truncated (finish_reason="length" -> empty extraction).
_OPENAI_REASONING_RE = re.compile(r"^(gpt-5|o[1-9])", re.IGNORECASE)
_DEEPSEEK_REASONING_RE = re.compile(r"deepseek-(reasoner|v[4-9])", re.IGNORECASE)


# Per-stage token accounting, so "which stage costs what" is measured rather
# than guessed. Keyed by the label each call site passes.
USAGE: dict[str, dict] = {}


def _record_usage(label: str, model: str, resp) -> None:
    usage = getattr(resp, "usage", None)
    entry = USAGE.setdefault(label, {"calls": 0, "prompt": 0, "completion": 0,
                                     "models": set()})
    entry["calls"] += 1
    entry["models"].add(model)
    if usage is not None:
        entry["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        entry["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)


def usage_report() -> dict:
    """JSON-serialisable snapshot of USAGE (sets become sorted lists)."""
    return {
        label: {**{k: v for k, v in counts.items() if k != "models"},
                "models": sorted(counts["models"])}
        for label, counts in sorted(USAGE.items())
    }


def call_json(system: str, user: str, model: str, temperature: float,
              max_tokens: int, label: str = "") -> dict:
    from src.core.llm_client import get_llm_client
    client = get_llm_client()
    m = model or ""
    openai_reasoning = bool(_OPENAI_REASONING_RE.match(m))
    reasoning = openai_reasoning or bool(_DEEPSEEK_REASONING_RE.search(m))
    # Reasoners spend hidden reasoning tokens against this budget (v4-pro alone
    # burns ~16k on the whole-prompt extraction), so give a large floor.
    budget = max(max_tokens * 6, 32000) if reasoning else max_tokens
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    if openai_reasoning:
        kwargs["max_completion_tokens"] = budget
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = budget
    resp = client.chat.completions.create(**kwargs)
    _record_usage(label or "unlabelled", model, resp)
    choice = resp.choices[0]
    raw = choice.message.content or "{}"
    if choice.finish_reason == "length":
        logger.warning("LLM output hit the token limit (model=%s, budget=%d); "
                       "JSON likely truncated.", model, budget)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON (first 200 chars): %s", raw[:200])
        return {}


# ── Tool read/write type + compact context ────────────────────────────────────

_WRITE_NAME_RE = re.compile(
    r"^(create|update|modify|cancel|delete|add|remove|set|book|reserve|send|"
    r"submit|register|change|transfer|return|exchange|enable|disable|suspend|"
    r"resume|make|reset|reboot|toggle|refund|charge|pay)\b", re.IGNORECASE)
_READ_NAME_RE = re.compile(
    r"^(get|find|list|search|fetch|retrieve|check|look|show|calculate|view)\b",
    re.IGNORECASE)


def tool_rw_type(tool) -> str:
    """WRITE | READ | THINK | OTHER — from the source ToolType marker if present,
    else inferred from the tool-name verb. Source is preferred but not required."""
    src = tool.source_code or ""
    if "ToolType.WRITE" in src:
        return "WRITE"
    if "ToolType.READ" in src:
        return "READ"
    if "ToolType.THINK" in src:
        return "THINK"
    if _WRITE_NAME_RE.match(tool.name):
        return "WRITE"
    if _READ_NAME_RE.match(tool.name):
        return "READ"
    return "OTHER"


def tool_context(agent: AgentUnderTest, param_desc_len: int = 55) -> str:
    """Compact per-tool context: name, rw-type, ALL params (with short
    descriptions), and a one-line tool description.

    Short per-parameter descriptions let the LLM map a policy's business terms to
    the RIGHT parameter (e.g. "date of birth" -> dob) so bindings are accurate,
    without the bulk of a full JSON schema (types / nesting are struct_miner's
    job, not the prompt miner's). Every param is listed — dropping any risks
    hiding a constraint-relevant argument (e.g. total_baggages / insurance) so the
    LLM never binds a rule to it.
    """
    lines: list[str] = []
    for t in agent.tools:
        params = []
        for p in t.parameters:
            s = p.name
            if getattr(p, "enum_values", None):
                s += f"∈{p.enum_values}"
            pdesc = (getattr(p, "description", "") or "").strip().split("\n")[0]
            if pdesc:
                s += f" ({pdesc[:param_desc_len]})"
            params.append(s)
        desc = (t.description or "").strip().split("\n")[0][:110]
        lines.append(f"  - [{tool_rw_type(t)}] {t.name}({', '.join(params)}): {desc}")
    return "\n".join(lines)


# ── Binding coverage check ───────────────────────────────────────────────────
#
# The normalize pass is TOLD to walk the full tool list ("a rule naming a
# category applies to EVERY tool in that category"), and still binds the
# confirm-before-write rule to a single tool run after run — batch listing is
# exactly the task shape the model fails at. So coverage is enforced as a
# funnel of tasks the model IS reliable at:
#
#   screen      one call per rule: does the rule's own wording implicate more
#               tools than its bindings hold (category quantification, an
#               enumeration of actions, or no binding at all)? A semantic
#               judgment — no regex over phrasings, no curated verb lists, so
#               nothing to re-curate per domain.
#   adjudicate  for a screened rule, one yes/no call per unbound tool; a yes
#               counts only when it QUOTES the fragment of the rule/policy
#               that covers this tool (the same quote-grounding that keeps
#               the prompt miner honest).
#
# Nothing is added without a verified quote; nothing rests on the model
# volunteering a complete list.

_SCREEN_SYSTEM = """\
You screen ONE business rule of a {domain} customer-service agent: do the
rule's own words place an obligation on MORE tools than it is currently
bound to?

Answer true ONLY when the rule's wording ranges over ACTIONS:
- it quantifies over a category of actions ("before any action that updates
  the database ...", "all account changes ...") and the tool list plausibly
  contains several tools of that category; or
- it enumerates several actions ("cancel, modify, return, exchange") whose
  tools are not all bound; or
- it is bound to NO tool but names an action some tool performs.

Answer false otherwise — in particular:
- a rule stating a format/validity/lookup constraint on a PARAMETER ("the
  order id must be a string starting with #") stays where it was mined;
  other tools sharing that parameter name are covered by their own specs,
  and expanding it would only duplicate them;
- a state/ownership/eligibility condition phrased for one action stays with
  that action for the same reason, even though sibling actions have the
  same condition;
- a rule describing the EFFECT or outcome of one specific action ("after
  user confirmation the order status becomes 'return requested'") belongs
  to that action alone — the outcome would be false for any other tool;
- the rule concerns one specific action and its tool is already bound;
- the rule governs only the agent's language;
- the extra actions it mentions are preconditions/context, not actions it
  constrains.

Return only JSON: {{"broader": true|false, "reason": "<one line>"}}"""

_SCREEN_USER = """\
RULE: {rule}

POLICY QUOTES the rule was extracted from:
{evidence}

CURRENTLY BOUND TOOLS: {bound}

ALL TOOLS of the agent: {tool_names}

Return the JSON object."""

_COVER_SYSTEM = """\
You check ONE business rule of a {domain} customer-service agent against ONE
tool: is a call to this tool governed by this rule — could a call to it
violate the rule?

Judge from the rule's own wording (the policy quotes give its context):
- a rule quantifying over a category ("before any action that updates the
  database ...") governs EVERY tool that performs such an action;
- a rule enumerating actions ("cancel, modify, return, exchange") governs the
  tools performing each enumerated action;
- a rule about one specific action does NOT govern tools for other actions,
  however similar; an action the rule mentions only as a precondition or as
  context is NOT thereby constrained;
- a rule constraining a PARAMETER's format/validity/lookup, or stating a
  state/ownership condition phrased for a different action, does NOT extend
  to this tool merely because it shares the parameter or the condition —
  such tools are covered by their own specs, and a duplicate binding here
  is wrong;
- a rule describing the EFFECT or outcome of one specific action ("after
  confirmation the status becomes 'return requested'") does NOT govern this
  tool unless this tool performs exactly that action — the stated outcome
  would be false here. When in doubt, answer false.

A true answer MUST carry "quote": the fragment of RULE or the policy quotes,
copied verbatim, whose wording covers this tool. No qualifying fragment — no
true answer. If the rule constrains specific arguments of THIS tool, name
them in "params"; otherwise return an empty params list.

Return only JSON:
{{"applies": true|false, "quote": "<verbatim fragment>",
  "params": ["<param>", ...], "reason": "<one line>"}}"""

_COVER_USER = """\
RULE: {rule}

POLICY QUOTES the rule was extracted from:
{evidence}

TOOL: {tool_line}

Return the JSON object."""


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def verify_binding_coverage(agent: AgentUnderTest, specs: list[Spec], *,
                            domain: str, model: str, temperature: float = 0.0
                            ) -> dict:
    """Screen every rule for broader-than-bound wording; adjudicate screened
    rules against every unbound tool, quote-verified. Mutates specs in place;
    returns a small report."""
    tool_by_name = {t.name: t for t in agent.tools}
    all_names = sorted(tool_by_name)
    # A rule specialize() already split binds its tools ACROSS the sibling
    # specs sharing its source_rule. Screening one sibling against only its
    # own binding re-discovers the enumeration and duplicates the others, so
    # "bound" means the union over the source_rule group.
    group_bound: dict[str, set[str]] = {}
    for s in specs:
        key = s.source_rule or s.spec_id
        group_bound.setdefault(key, set()).update(
            b["tool"] for b in (s.bindings or []))
    n_screened = n_broader = n_checked = n_added = n_quote_rejected = 0
    added_log: list[dict] = []
    for s in specs:
        # Only rules with textual authority: the under-binding failure is a
        # policy sentence quantifying over a category of actions. Schema and
        # domain-knowledge rules are generated per tool by construction —
        # "expanding" one merely duplicates (or fabricates) its sibling
        # tools' own specs, and their quotes are self-referential.
        if s.cross_tool or s.origin != "prompt":
            continue
        bound = set(group_bound.get(s.source_rule or s.spec_id) or
                    {b["tool"] for b in (s.bindings or [])})
        ev = "\n".join(f"  - {e.quote}" for e in s.evidence) or "  (none)"
        n_screened += 1
        sv = call_json(
            _SCREEN_SYSTEM.format(domain=domain),
            _SCREEN_USER.format(rule=s.rule_text, evidence=ev,
                                bound=", ".join(sorted(bound)) or "(none)",
                                tool_names=", ".join(all_names)),
            model, temperature, max_tokens=250, label="binding_screen")
        if not (isinstance(sv, dict) and sv.get("broader") is True):
            continue
        n_broader += 1
        haystack = _norm_ws(s.rule_text) + " || " + " || ".join(
            _norm_ws(e.quote) for e in s.evidence)
        for tool_name in all_names:
            if tool_name in bound:
                continue
            tool = tool_by_name[tool_name]
            params = [p.name for p in tool.parameters]
            desc = (tool.description or "").strip().split("\n")[0][:110]
            n_checked += 1
            v = call_json(
                _COVER_SYSTEM.format(domain=domain),
                _COVER_USER.format(rule=s.rule_text, evidence=ev,
                                   tool_line=f"{tool_name}({', '.join(params)}): {desc}"),
                model, temperature, max_tokens=350, label="binding_coverage")
            if not (isinstance(v, dict) and v.get("applies") is True):
                continue
            quote = v.get("quote")
            if not isinstance(quote, str) or not quote.strip() \
                    or _norm_ws(quote) not in haystack:
                n_quote_rejected += 1
                logger.debug("coverage: %s -> %s rejected — quote %r does not "
                             "verify", s.spec_id, tool_name, quote)
                continue
            good_params = [p for p in (v.get("params") or [])
                           if isinstance(p, str) and p in params]
            s.bindings = (s.bindings or []) + [{"tool": tool_name,
                                                "params": good_params}]
            s.relevant_tools = [b["tool"] for b in s.bindings]
            group_bound.setdefault(s.source_rule or s.spec_id,
                                   set()).add(tool_name)
            n_added += 1
            added_log.append({"spec_id": s.spec_id, "tool": tool_name,
                              "quote": quote.strip(),
                              "reason": (v.get("reason") or "").strip()})
    logger.info("Binding coverage: %d rules screened, %d broader, %d tool "
                "checks, %d bindings added, %d rejected on quote",
                n_screened, n_broader, n_checked, n_added, n_quote_rejected)
    return {"n_screened": n_screened, "n_broader": n_broader,
            "n_checked": n_checked, "n_added": n_added,
            "n_quote_rejected": n_quote_rejected, "added": added_log}


# ── Per-tool specialization ──────────────────────────────────────────────────

def specialize(specs: list[Spec]) -> list[Spec]:
    """Split a rule that applies to several tools INDEPENDENTLY into one spec per
    tool, so each becomes its own test target; keep a shared source_rule so the
    common origin is not lost. Cross-tool / cumulative rules (cross_tool=True) and
    single-tool rules are left unchanged.
    """
    out: list[Spec] = []
    n_split = 0
    for s in specs:
        bindings = s.bindings or [{"tool": t, "params": []} for t in s.relevant_tools]
        tools = [b["tool"] for b in bindings]
        if s.cross_tool or len(tools) <= 1:
            out.append(s)
            continue
        n_split += 1
        src = s.source_rule or s.spec_id
        for b in bindings:
            out.append(Spec(
                spec_id=f"{s.spec_id}_{b['tool']}",
                target_action=b["tool"],
                rule_text=s.rule_text,
                kind=s.kind,
                clauses=list(s.clauses),
                relevant_tools=[b["tool"]],
                bindings=[b],
                cross_tool=False,
                source_rule=src,
                confidence=s.confidence,
                origin=s.origin,
                evidence=list(s.evidence),
            ))
    if n_split:
        logger.info("Specialize: split %d multi-tool rules → %d specs total",
                    n_split, len(out))
    return out


def build_coverage_grid(
    agent: AgentUnderTest,
    specs: list[Spec],
    kinds: tuple[str, ...] = ("STATE", "ORDER"),
) -> list[tuple[str, str]]:
    """Return (tool, kind) cells with no spec yet (STATE/ORDER only by default)."""
    filled: set[tuple[str, str]] = set()
    for s in specs:
        for tool in s.relevant_tools:
            filled.add((tool, s.kind))
    empty: list[tuple[str, str]] = []
    for tool in (t.name for t in agent.tools):
        for kind in kinds:
            if (tool, kind) not in filled:
                empty.append((tool, kind))
    return empty
