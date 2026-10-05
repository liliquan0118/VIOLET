"""Schema miner (the first source): grounded extraction from the tool schemas.

This side reads what the SCHEMA itself declares/describes — it does NOT guess
domain rules (that is domain_knowledge's job). Two LLM passes:

  1. ARG — the model READS each tool's parameters (name, type, description,
     enum, example values) and extracts every argument constraint the schema
     describes or its meaning implies: formats (incl. example-implied), ranges,
     enum value-domains, and relations between parameters (e.g. origin must
     differ from destination). Extraction is broad — value is judged later.
  2. ORDER (data-flow) — from each tool's inputs vs. outputs, a genuine
     call-sequence constraint: tool B has an input that can ONLY be obtained
     from tool A's return value (not from the user). A second adversarial pass
     drops spurious ones (inputs the user could supply directly).

Scope is ARG + ORDER — what the tool CONTRACT grounds. STATE (ownership /
eligibility) and NORM live in the prompt/domain, not the schema, so they are
mined by prompt_miner and domain_knowledge, not here.

All specs are origin="schema", confidence="structural". Overlaps with prompt /
domain-knowledge are reconciled downstream by dedup.py.
"""

from __future__ import annotations

import logging

from src.core.types import AgentUnderTest
from src.mining_v2.common import call_json, specialize, tool_rw_type
from src.mining_v2.spec import Evidence, Spec

logger = logging.getLogger(__name__)


# ── Tool input/output contract text (grounding for the LLM) ───────────────────

def _distill_source(source: str) -> str:
    """Keep the signature + docstring (inputs/outputs), drop the body — the body
    is implementation, not contract. If no docstring is found, keep the signature
    line(s) only."""
    if not source:
        return ""
    lines = source.splitlines()
    # find the def line
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("def ")), 0)
    out = lines[start:]
    text = "\n".join(out)
    # cut at the end of the docstring if present, else at the first non-doc body line
    if '"""' in text:
        first = text.find('"""')
        second = text.find('"""', first + 3)
        if second != -1:
            return text[:second + 3].strip()
    return "\n".join(out[:1]).strip()


def _tool_contracts(agent: AgentUnderTest) -> str:
    """One block per tool: rw-type, signature+docstring (inputs & return), and the
    declared parameter list. This is the grounding the LLM reasons over."""
    blocks: list[str] = []
    for t in agent.tools:
        rw = tool_rw_type(t)
        params = []
        for p in t.parameters:
            req = "required" if p.required else "optional"
            enum = f" ∈{p.enum_values}" if p.enum_values else ""
            desc = (p.description or "").strip().split("\n")[0]
            params.append(f"      - {p.name} ({p.type}, {req}){enum}: {desc}")
        params_str = "\n".join(params) if params else "      (no parameters)"
        sig = _distill_source(t.source_code or "")
        sig_block = f"\n    signature/docstring:\n{_indent(sig, 6)}" if sig else ""
        blocks.append(
            f"  [{rw}] {t.name}\n"
            f"    inputs:\n{params_str}{sig_block}"
        )
    return "\n\n".join(blocks)


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + ln for ln in text.splitlines())


# ── Pass 1: LLM ARG (grounded in the schema text) ─────────────────────────────

_ARG_SYSTEM = """\
You extract argument constraints for a {domain} agent's tools by READING and
UNDERSTANDING each tool's parameters (name, type, description) and signature.
This is the EXTRACTION stage — capture EVERY constraint the schema describes or
its meaning implies. Do NOT skip a constraint because it seems obvious or trivial
— extract it anyway.

Extract, for each parameter, every constraint it carries:
  - FORMAT — stated or shown by an example value (e.g. "IATA code such as 'SFO'"
    -> 3-letter uppercase; 'YYYY-MM-DD' -> that date format). An example value IS a constraint, not a throwaway.
  - RANGE / bound / length the description states.
  - ENUM value-domain ("one of A, B, C").
  - RELATION between parameters implied by their MEANING — origin must differ from
    destination, a non-free count must not exceed a total. List BOTH params.

Only argument-VALUE constraints — not ownership, eligibility, call order, or
behaviour (those are other kinds, handled elsewhere).

For each constraint give:
- rule_text: one clear sentence.
- tool, params: the tool and the argument name(s) it binds (both, for a relation).
- quote: the words or example the constraint is drawn from.
Output ONLY a JSON object."""

_ARG_USER = """\
DOMAIN: {domain}

TOOL CONTRACTS:
{contracts}

Return JSON:
{{
  "constraints": [
    {{
      "rule_text": "<one sentence>",
      "tool": "<tool name>",
      "params": ["<param>", ...],
      "quote": "<verbatim words from the schema text>"
    }}
  ]
}}"""


# ── Layer 3: LLM ORDER (data-flow sequencing) + verification ──────────────────

_ORDER_SYSTEM = """\
You find genuine CALL-SEQUENCE constraints between a {domain} agent's tools,
from DATA FLOW only.

A constraint "tool A must be called before tool B" is GENUINE only when:
  tool B has an input parameter whose value can ONLY be obtained from tool A's
  RETURN value — the agent cannot fill it from the user or the conversation.

It is SPURIOUS (do NOT report) when:
  - the input is something the user can provide directly (an id they already know);
  - tool B validates the dependency internally;
  - the ordering is only "good practice", not a hard data dependency.

Reason from the inputs and returns shown. Output ONLY a JSON object."""

_ORDER_USER = """\
DOMAIN: {domain}

TOOL CONTRACTS (inputs and returns):
{contracts}

Return JSON (only genuine data-flow sequencing constraints):
{{
  "order": [
    {{
      "predecessor": "<tool A that must run first>",
      "successor": "<tool B with the dependent input>",
      "input_param": "<B's input that can only come from A's return>",
      "rule_text": "<one sentence, e.g. 'B must not be called before A, because B's <param> can only come from A's return'>",
      "quote": "<the A-returns-X / B-requires-X evidence from the contracts>"
    }}
  ]
}}"""

_VERIFY_SYSTEM = """\
You verify whether proposed data-flow sequencing constraints are GENUINE or
SPURIOUS for a {domain} agent.

GENUINE only when tool B has an input whose value can ONLY be obtained from tool
A's RETURN — an opaque value the user could NEVER know or state without A first
producing it.

SPURIOUS whenever the dependent input is an ENTITY ID or field the USER can
provide directly from their own knowledge — a reservation_id, user_id, order_id,
confirmation code, etc. that the customer already has. In these agents the user
typically states such ids themselves, so "must create/book before
cancel/get/update" is SPURIOUS: the user already holds the id of an existing
record and no predecessor call is required to obtain it. Also spurious if the
dependency is only best-practice ordering or is enforced internally by B.

Be strict; default to SPURIOUS when in doubt. Output ONLY a JSON object."""

_VERIFY_USER = """\
DOMAIN: {domain}

TOOL CONTRACTS:
{contracts}

CANDIDATES:
{candidates}

Return JSON:
{{
  "verdicts": [
    {{"predecessor": "...", "successor": "...", "verdict": "genuine" | "spurious"}}
  ]
}}"""


def mine_schema(
    agent: AgentUnderTest,
    *,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> list[Spec]:
    """LLM schema mining: ARG (grounded in the tool contract) + ORDER (data-flow)."""
    domain = agent.domain or "general"
    contracts = _tool_contracts(agent)
    tool_names = {t.name for t in agent.tools}
    valid_params = {t.name: {p.name for p in t.parameters} for t in agent.tools}

    specs: list[Spec] = []

    # Pass 1 — LLM ARG (grounded)
    data = call_json(
        _ARG_SYSTEM.format(domain=domain),
        _ARG_USER.format(domain=domain, contracts=contracts),
        model, temperature, max_tokens=4000,
    )
    for c in data.get("constraints", []):
        rule_text = (c.get("rule_text") or "").strip()
        tool = c.get("tool", "")
        if not rule_text or tool not in tool_names:
            continue
        params = [p for p in c.get("params", []) if p in valid_params.get(tool, set())]
        quote = (c.get("quote") or "").strip()
        specs.append(Spec(
            spec_id=f"sa{len(specs):03d}", target_action=tool, rule_text=rule_text,
            kind="ARG", clauses=[rule_text], relevant_tools=[tool],
            bindings=[{"tool": tool, "params": params}], cross_tool=False,
            confidence="structural", origin="schema",
            evidence=[Evidence(source="schema", quote=quote)] if quote else [],
        ))
    n_arg = len(specs)

    # Pass 2 — LLM ORDER (data-flow) + adversarial verification
    order_specs = _mine_order(agent, contracts, domain, model, temperature)
    specs.extend(order_specs)

    logger.info("Schema miner: %d specs (%d LLM-ARG + %d ORDER)",
                len(specs), n_arg, len(order_specs))
    return specialize(specs)


def _mine_order(agent, contracts, domain, model, temperature) -> list[Spec]:
    tool_names = {t.name for t in agent.tools}
    valid_params = {t.name: {p.name for p in t.parameters} for t in agent.tools}

    data = call_json(
        _ORDER_SYSTEM.format(domain=domain),
        _ORDER_USER.format(domain=domain, contracts=contracts),
        model, temperature, max_tokens=3000,
    )
    cands = []
    for o in data.get("order", []):
        a, b = o.get("predecessor", ""), o.get("successor", "")
        if a in tool_names and b in tool_names and a != b:
            cands.append(o)
    if not cands:
        return []

    # Verify: drop spurious (inputs the user could supply directly).
    cand_str = "\n".join(
        f"  - {o['predecessor']} must precede {o['successor']} "
        f"(via {o.get('input_param', '?')})" for o in cands
    )
    v = call_json(
        _VERIFY_SYSTEM.format(domain=domain),
        _VERIFY_USER.format(domain=domain, contracts=contracts, candidates=cand_str),
        model, temperature, max_tokens=1500,
    )
    genuine = {(x.get("predecessor"), x.get("successor"))
               for x in v.get("verdicts", []) if x.get("verdict") == "genuine"}
    # If the verifier returned nothing usable, keep candidates (fail-open, logged).
    if not v.get("verdicts"):
        logger.warning("ORDER verifier returned no verdicts; keeping %d candidates", len(cands))
        genuine = {(o["predecessor"], o["successor"]) for o in cands}

    specs: list[Spec] = []
    for o in cands:
        a, b = o["predecessor"], o["successor"]
        if (a, b) not in genuine:
            continue
        param = o.get("input_param", "")
        params = [param] if param in valid_params.get(b, set()) else []
        rule_text = (o.get("rule_text") or
                     f"{b} must not be called before {a}, because its "
                     f"{param or 'input'} can only come from {a}'s return value.").strip()
        quote = (o.get("quote") or "").strip()
        specs.append(Spec(
            spec_id=f"so{len(specs):03d}", target_action=b, rule_text=rule_text,
            kind="ORDER", clauses=[rule_text], relevant_tools=[b],
            bindings=[{"tool": b, "params": params}], cross_tool=False,
            confidence="structural", origin="schema",
            evidence=[Evidence(source="schema", quote=quote)] if quote else [],
        ))
    return specs
