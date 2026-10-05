"""Prompt miner with a coverage audit whose completeness is guaranteed by code.

Same extraction, adjudication and normalization as prompt_miner.py (their
prompts are copied verbatim below; nothing is imported from that file). What changes is the coverage
audit between extraction and adjudication:

  prompt_miner.py   ONE LLM call re-emits every sentence of the policy with
                    anchor/occurrence and decides `quoted` itself. Output is
                    ~4x the policy; on telecom's 23k-char policy the model
                    stopped at 30% with valid JSON, so nothing noticed and 70%
                    of the policy was never audited.

  here              1. code segments the policy into OCCURRENCES — bookkeeping
                       only, never shown to the model, so completeness is by
                       construction and the model still reads the unmodified
                       policy;
                    2. ONE small call per spec: given the full policy and that
                       spec alone, which sentence(s) does each of its quotes
                       cover — verbatim text plus the immediately preceding
                       sentence, the same anchor mechanism as before, so a
                       sentence that appears twice is disambiguated with the
                       spec's own context (target_action, bindings);
                    3. code maps each claim to an occurrence and VERIFIES the
                       text; a claim naming text the policy does not contain at
                       that place is rejected and logged;
                    4. unquoted occurrences go to the per-sentence adjudication
                       exactly as before.

The earlier objection to "splitting the prompt" — that numbering or fragmenting
what the model reads distorts the text and loses the context needed to tell
two occurrences apart — is honoured: the model never sees the segmentation.
"""

from __future__ import annotations

import logging
import re

from src.core.types import AgentUnderTest
from src.mining_v2.common import (
    call_json,
    specialize,
    tool_context,
    verify_binding_coverage,
)
from src.mining_v2.spec import Evidence, KINDS, Spec

logger = logging.getLogger(__name__)


# ── prompts copied verbatim from prompt_miner.py (this module is self-contained; the original file is not imported or modified) ──

_PASS1_SYSTEM = """\
You extract business-rule SPECS from a {domain} agent's system prompt.

You are given the FULL system prompt (read it whole; use its section headers and
structure as context) plus the tools the agent can call with their parameters.

A SPEC is one business rule (one proposition) the agent must obey — an
obligation, prohibition, condition, limit, or required procedure. A rule may be
an explicit instruction to the agent (must / must not / should / need) OR a business
FACT, VALUE, LIMIT, or TABLE the agent must apply or enforce even though it is
stated as a plain fact rather than a command. Do NOT skip a sentence just because
it does not tell the agent to do something: a stated fact, value, or table that
determines an outcome IS a rule. Emit specs for:
  (a) constraints tied to a tool call (who/when/with-what a tool may be called), and
  (b) conversational obligations with NO specific tool — required troubleshooting
      steps, info the agent must collect/verify, things it must or must not say,
      when to escalate.
Do NOT skip a procedural/behavioral rule just because it names no tool. Aim for
COMPLETE coverage — extract every obligation, even obvious ones.

GROUPING — the unit is ONE PROPOSITION (one rule). Judge by MEANING, using ONE test:

  Two obligations are DISTINCT specs iff one could hold while the other is
  violated.

- CANNOT be violated independently → ONE spec: branches of the same decision, a
  rule and its exception, or a RELATION binding several params (e.g.
  origin != destination, used_amount <= total_amount — do NOT split by number of
  parameters).
- CAN be violated independently → SEPARATE specs, even within one sentence:
  e.g. "cannot remove bags" vs "cannot add insurance"; "make only one tool call
  at a time" vs "do not respond while calling a tool".
- If a rule mandates an ORDER of steps, the ordering itself is part of the rule.

For each spec give:
- rule_text: ONE clear sentence stating the rule (join disjunction branches with "or").
- kind: WHAT the rule CONSTRAINS (its target, not how you would check it):
    ARG   : constrains a tool call's ARGUMENT(s) — the value/format/range a
            parameter may take, a relation between parameters (origin !=
            destination), or the value a parameter must equal, INCLUDING a
            computed value (e.g. amount == 50 x passenger_count — the rule
            constrains the `amount` argument; that its inputs may be looked up is
            a checking detail, not what it constrains). An enumerated value-domain
            ("X is one of A, B, C") is an ARG enum ONLY IF X is a scalar argument
            the agent itself passes; if X is a property referred to by id (a
            payment-method type, a membership level — the agent passes an id, not
            the value), skip it as glossary.
    STATE : constrains DB/account STATE that a resource must be in — ownership
            ("the reservation must belong to the user"), eligibility ("only if
            business / within 24h / silver member"), or a stored record's status.
    ORDER : constrains the CALL SEQUENCE — what must be looked up / confirmed /
            collected before the call, required ordering, must-not-repeat.
    NORM  : constrains the agent's LANGUAGE OUTPUT / behavior — "do not
            proactively offer compensation", "no subjective recommendations",
            "deny requests against the rules", "transfer to a human when X".
- bindings: which tool parameters the rule binds, as a list of
  {{"tool": <tool>, "params": [<param>, ...]}}. Include one entry per tool the
  rule applies to; params are that tool's argument names it constrains (empty list
  if the rule is about the tool but no specific arg, e.g. ordering/ownership).
  For an agent-wide NORM with no specific tool, use an empty bindings list.
- cross_tool: true if the rule RELATES calls to DIFFERENT tools, so it cannot
  be tested on any single tool alone — typically a required sequence between
  tools ("call get_user_details to verify the user BEFORE update_...", "first
  call transfer_to_human_agents, THEN send the hold message"). false if the
  rule imposes the same obligation on each of its tools SEPARATELY (e.g. "the
  reservation_id must belong to the user" holds independently for cancel,
  update, get) — such rules are split and tested per tool.
- target_action: a SHORT label (tool name when it maps to one tool; else a
  concept like "confirm_before_write", "compensation", "agent_behavior").
- quote: the exact sentence(s) from the system prompt this rule is drawn from,
  copied VERBATIM (used as evidence; do not paraphrase).

Rules: only reference provided tool names/params; do not invent rules the prompt
does not support. Output ONLY a JSON object."""

_PASS1_USER = """\
DOMAIN: {domain}

TOOLS (name(params): description — you may only reference these tool names/params):
{tool_list}

SYSTEM PROMPT:
{policy}

Return JSON:
{{
  "specs": [
    {{
      "rule_text": "<one sentence>",
      "kind": "ARG|STATE|ORDER|NORM",
      "bindings": [{{"tool": "<tool>", "params": ["<param>", ...]}}, ...],
      "cross_tool": true|false,
      "target_action": "<short canonical label>",
      "quote": "<verbatim sentence(s) from the prompt>"
    }}
  ]
}}"""

_ADJ_SYSTEM = """\
You adjudicate ONE sentence of a {domain} agent's system prompt that is NOT yet
quoted as evidence by any extracted spec. Locate the sentence in the FULL
prompt using its anchor (the sentence immediately preceding it) and occurrence
marker, read it in that context, and decide:

- "extends_spec": the sentence states (part of) an obligation an existing spec
  already captures — its text should be added to that spec's evidence. Give the
  spec_id. Use this for a restatement of a rule under another section, an
  emphasis/reinforcement of an existing rule, or a branch/condition line of a
  rule already extracted. Pick the spec whose tools/section match THIS
  occurrence of the sentence.
- "new_rule": the sentence states an obligation, constraint, limit, or
  fact-to-enforce that NO existing spec captures. Provide the full spec fields.
  kind is WHAT the rule CONSTRAINS:
    ARG   : a tool call's argument(s) — the value/format/range a parameter may
            take, or a relation between parameters
    STATE : DB/account state a resource must be in — ownership, eligibility,
            a stored record's status
    ORDER : the call sequence — what must be looked up / confirmed / collected
            before the call, required ordering
    NORM  : the agent's language output / behavior
- "not_a_rule": the sentence imposes nothing the agent could violate —
  removing it from the prompt would not change what a compliant agent may,
  must, or must not do. Give a one-clause reason. This covers, besides
  headings and data-model listings: a description of a tool or of an action
  the USER can take on their device ("check_sim_status - Checks if your SIM
  card is working"); an enumeration of the values a reading can take
  ("signal strength can be poor, fair, good"); an explanation of how
  something works or why it fails ("MMS relies on cellular service", "APN
  settings might be incorrect resulting in loss of service"); a list of
  possible causes; a lead-in ("The user has 2 options:"). A rule tells the
  AGENT what it must, may or must not do — a fact the agent might USE while
  troubleshooting is not a rule unless the sentence itself directs the
  agent's action ("you must first ...", "guide the user to ...", "do not
  ...").

Output ONLY a JSON object."""

_ADJ_USER = """\
DOMAIN: {domain}

TOOLS (name(params): description — you may only reference these tool names/params):
{tool_list}

EXTRACTED SPECS (with their verbatim evidence quotes):
{specs}

SYSTEM PROMPT:
{policy}

CANDIDATE SENTENCE (not quoted by any spec):
  text: {text}
  anchor: {anchor}
  occurrence: {occ}

Return JSON:
{{
  "verdict": "extends_spec|new_rule|not_a_rule",
  "spec_id": "<spec id, for extends_spec only>",
  "spec": {{
    "rule_text": "<one sentence>",
    "kind": "ARG|STATE|ORDER|NORM",
    "bindings": [{{"tool": "<tool>", "params": ["<param>", ...]}}, ...],
    "cross_tool": true|false,
    "target_action": "<short canonical label>",
    "quote": "<the candidate sentence, verbatim>"
  }},
  "reason": "<one clause, for not_a_rule only>"
}}"""

_BIND_SYSTEM = """\
You normalize ONE business rule of a {domain} agent: re-derive its kind, the
tools it binds, and its cross_tool flag.

You are given the rule (with the policy sentences it was extracted from), the
full system prompt for context, and the complete tool list.

kind — WHAT the rule CONSTRAINS:
  ARG   : a tool call's argument(s) — the value/format/range a parameter may
          take, or a relation between parameters
  STATE : DB/account state a resource must be in — ownership, eligibility,
          a stored record's status
  ORDER : the call sequence — what must be looked up / confirmed / collected
          before the call, required ordering
  NORM  : the agent's language output / behavior

bindings — go through the tool list and decide, for EACH tool: must a call to
this tool obey this rule — i.e. could a call to this tool violate it? Include
every such tool. Do not drop a tool because another one is more typical (a
rule naming a category, e.g. "any action that updates the booking database",
applies to EVERY tool in that category); do not add tools the rule cannot
apply to. For each included tool list the parameters the rule constrains
(empty list if the rule governs the call but no specific argument; an ARG rule
must name the constrained params). For an agent-wide behavioral rule tied to
no tool, return an empty bindings list.

cross_tool — true if the rule RELATES calls to DIFFERENT tools (a required
sequence between tools) so it cannot be tested on one tool alone; false if it
imposes the same obligation on each of its tools separately.

Output ONLY a JSON object."""

_BIND_USER = """\
DOMAIN: {domain}

TOOLS (name(params): description — you may only reference these tool names/params):
{tool_list}

SYSTEM PROMPT:
{policy}

RULE ([{kind}]): {rule}
EVIDENCE:
{evidence}

Return JSON:
{{
  "kind": "ARG|STATE|ORDER|NORM",
  "bindings": [{{"tool": "<tool>", "params": ["<param>", ...]}}, ...],
  "cross_tool": true|false
}}"""


# ── 1. segmentation (code, bookkeeping only) ─────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def segment_policy(policy: str) -> list[dict]:
    """The policy as a list of sentence OCCURRENCES, in document order.

    A non-empty line is one unit; a line holding several prose sentences is
    split at sentence ends followed by a capital. Each occurrence carries the
    immediately preceding occurrence's text as `anchor` and "k/n" as `occ`
    when the same normalized text appears n>1 times. Nothing here is ever
    shown to the model; it only fixes what "complete" means.
    """
    units: list[str] = []
    for line in policy.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if len(stripped) > 160 and _SENT_SPLIT.search(stripped):
            units.extend(s for s in _SENT_SPLIT.split(stripped) if s.strip())
        else:
            units.append(stripped)
    counts: dict[str, int] = {}
    for u in units:
        counts[_norm(u)] = counts.get(_norm(u), 0) + 1
    seen: dict[str, int] = {}
    out: list[dict] = []
    for i, u in enumerate(units):
        key = _norm(u)
        seen[key] = seen.get(key, 0) + 1
        out.append({"idx": i, "text": u, "anchor": units[i - 1] if i else "",
                    "occ": f"{seen[key]}/{counts[key]}" if counts[key] > 1 else ""})
    return out


# ── 2. per-spec claims (LLM, full policy, one spec) ──────────────────────────

_CLAIM_SYSTEM = """\
You locate the evidence quotes of ONE business-rule spec inside a {domain}
agent's system prompt. Each quote was copied from the prompt when the rule was
extracted, possibly with light copying edits (formatting, punctuation, a
dropped word).

For EACH quote, list the prompt sentence(s) it covers — one entry per
sentence (a list item, a heading line or a standalone line is one sentence; a
quote spanning several sentences yields several entries). For every entry give:
- text: that sentence copied VERBATIM from the prompt (keep formatting as-is)
- anchor: the sentence IMMEDIATELY PRECEDING it in the prompt, verbatim ("" if
  it is the very first line)
- occ: "k/n" if the same sentence text appears n>1 times in the prompt and this
  is the k-th occurrence; "" if unique. When a sentence appears in several
  sections, pick the occurrence that matches THIS spec's tools and subject —
  a quote belongs to one place.

List only sentences the quote actually reproduces. Do not add sentences that
merely express the same obligation in other words. Output ONLY a JSON object."""

_CLAIM_USER = """\
DOMAIN: {domain}

SPEC:
  rule_text: {rule}
  kind: {kind}   target_action: {target}   bindings: {bindings}
  quotes:
{quotes}

SYSTEM PROMPT:
{policy}

Return JSON:
{{"covered": [{{"quote_index": <i>, "text": "<verbatim sentence>",
               "anchor": "<preceding sentence>", "occ": "<k/n or empty>"}}, ...]}}"""


def _locate(claim: dict, occurrences: list[dict]) -> list[int]:
    """Map one claimed (text, anchor, occ) to occurrence indexes ([] = rejected).

    Exact text wins: a unique exact match needs no anchor at all. Only when
    the same text occurs several times do anchor and occurrence marker
    disambiguate — and when they cannot (a whole block repeated under two
    sections shares its anchors, and the model gave no marker) EVERY
    anchor-consistent occurrence is returned rather than none: losing
    coverage sends an already-quoted sentence to adjudication, which then
    mints a duplicate spec. Attribution may be broad; completeness must not
    be lost. Containment is the fallback for quotes with light copying edits.
    """
    t = _norm(claim.get("text"))
    if len(t) < 6:
        return []
    hits = [o for o in occurrences if _norm(o["text"]) == t]
    if not hits and len(t) >= 15:
        hits = [o for o in occurrences
                if t in _norm(o["text"]) or _norm(o["text"]) in t]
    if not hits:
        return []
    if len(hits) == 1:
        return [hits[0]["idx"]]
    a = _norm(claim.get("anchor"))
    if a:
        by_anchor = [o for o in hits if _norm(o["anchor"]) == a
                     or (len(a) >= 15 and (a in _norm(o["anchor"]) or _norm(o["anchor"]) in a))]
        if by_anchor:
            hits = by_anchor
    if len(hits) == 1:
        return [hits[0]["idx"]]
    occ = str(claim.get("occ") or "").strip()
    if occ:
        by_occ = [o for o in hits if o["occ"] == occ]
        if len(by_occ) == 1:
            return [by_occ[0]["idx"]]
    return [o["idx"] for o in hits]


def claim_occurrences(spec: Spec, occurrences: list[dict], policy: str, *,
                      domain: str, model: str, temperature: float
                      ) -> tuple[set[int], list[dict]]:
    """-> (occurrence indexes this spec's quotes cover, rejected claims)."""
    quotes = [e.quote for e in spec.evidence if e.quote]
    if not quotes:
        return set(), []
    user = _CLAIM_USER.format(
        domain=domain, rule=spec.rule_text, kind=spec.kind, target=spec.target_action,
        bindings=spec.bindings, policy=policy,
        quotes="\n".join(f"    [{i}] {q}" for i, q in enumerate(quotes)))
    data = call_json(_CLAIM_SYSTEM.format(domain=domain), user, model, temperature,
                     max_tokens=1500, label="coverage_claim")
    covered: set[int] = set()
    rejected: list[dict] = []
    for claim in data.get("covered", []) or []:
        if not isinstance(claim, dict):
            continue
        idxs = _locate(claim, occurrences)
        if not idxs:
            rejected.append({"spec_id": spec.spec_id, **{k: claim.get(k) for k in ("text", "anchor", "occ")}})
        else:
            covered.update(idxs)
    # Deterministic floor: a unique sentence reproduced verbatim inside a quote
    # is covered whether or not the model listed it.
    for o in occurrences:
        if any(len(_norm(o["text"])) >= 15 and _norm(o["text"]) in _norm(q) for q in quotes):
            covered.add(o["idx"])
    return covered, rejected


def audit_coverage(agent: AgentUnderTest, specs: list[Spec], *, model: str,
                   temperature: float = 0.0) -> tuple[list[dict], list[dict]]:
    """-> (occurrences with quoted/quoted_by, rejected claims). Pure: no spec
    is modified. Completeness holds by construction of `occurrences`."""
    policy = agent.system_prompt or ""
    domain = agent.domain or "general"
    occurrences = segment_policy(policy)
    quoted_by: dict[int, list[str]] = {}
    rejected: list[dict] = []
    for s in specs:
        covered, rej = claim_occurrences(s, occurrences, policy, domain=domain,
                                         model=model, temperature=temperature)
        rejected.extend(rej)
        for idx in covered:
            quoted_by.setdefault(idx, []).append(s.spec_id)
    for o in occurrences:
        o["quoted"] = o["idx"] in quoted_by
        o["quoted_by"] = quoted_by.get(o["idx"], [])
    return occurrences, rejected


# ── 3. mining: extraction -> v2 audit -> adjudication -> normalization ───────

def mine_prompt(agent: AgentUnderTest, *, model: str = "gpt-4.1",
                temperature: float = 0.0) -> tuple[list[Spec], dict]:
    """Drop-in for prompt_miner.mine_prompt with the v2 coverage audit."""
    tl = tool_context(agent)
    valid_tools = {t.name for t in agent.tools}
    valid_params = {t.name: {p.name for p in t.parameters} for t in agent.tools}
    policy = agent.system_prompt or ""
    domain = agent.domain or "general"

    def _parse_bindings(raw) -> list[dict]:
        out = []
        for b in raw or []:
            tool = b.get("tool", "")
            if tool not in valid_tools:
                continue
            params = [p for p in b.get("params", []) if p in valid_params.get(tool, set())]
            out.append({"tool": tool, "params": params})
        return out

    specs: list[Spec] = []

    def _absorb(data: dict, anchor: str = "") -> int:
        n = 0
        for sp in data.get("specs", []):
            kind = str(sp.get("kind", "")).upper()
            rule_text = (sp.get("rule_text") or "").strip()
            if kind not in KINDS or not rule_text:
                continue
            bindings = _parse_bindings(sp.get("bindings"))
            quote = (sp.get("quote") or "").strip()
            counter = len(specs) + 1
            specs.append(Spec(
                spec_id=f"p{counter:03d}",
                target_action=(sp.get("target_action") or "unknown").strip(),
                rule_text=rule_text, kind=kind, clauses=[rule_text],
                relevant_tools=[b["tool"] for b in bindings], bindings=bindings,
                cross_tool=bool(sp.get("cross_tool", False)),
                source_rule=f"{domain}_r{counter:03d}",
                confidence="stated", origin="prompt",
                evidence=[Evidence(source="prompt", quote=quote, anchor=anchor)] if quote else [],
            ))
            n += 1
        return n

    def _spec_block() -> str:
        import json
        return json.dumps([
            {"spec_id": s.spec_id, "kind": s.kind, "rule_text": s.rule_text,
             "target_action": s.target_action, "bindings": s.bindings,
             "quotes": [e.quote for e in s.evidence]} for s in specs],
            ensure_ascii=False, indent=1)

    # ── extraction (unchanged) ──
    n_extract = _absorb(call_json(_PASS1_SYSTEM.format(domain=domain),
                                  _PASS1_USER.format(domain=domain, tool_list=tl, policy=policy),
                                  model, temperature, max_tokens=6000))

    # ── v2 coverage audit ──
    sentences, rejected = audit_coverage(agent, specs, model=model, temperature=temperature)
    candidates = [s for s in sentences if not s["quoted"]]

    # ── adjudication (unchanged) ──
    n_extended = n_new = n_not_rule = 0
    verdict_log: list[dict] = []
    a_system = _ADJ_SYSTEM.format(domain=domain)
    for cand in candidates:
        text, anchor, occ = cand["text"], cand["anchor"], cand["occ"]
        a_user = _ADJ_USER.format(domain=domain, tool_list=tl, specs=_spec_block(),
                                  policy=policy, text=text, anchor=anchor, occ=occ or "unique")
        v = call_json(a_system, a_user, model, temperature, max_tokens=1500)
        verdict = (v.get("verdict") or "").strip()
        loc = f"{anchor} | occ {occ}" if occ else anchor
        entry = {"idx": cand["idx"], "text": text, "anchor": anchor, "occ": occ, "verdict": verdict or "?"}
        if verdict == "extends_spec":
            sid = (v.get("spec_id") or "").strip()
            entry["spec_id"] = sid
            sp = next((s for s in specs if s.spec_id == sid), None)
            if sp is None:
                entry["applied"] = False
            else:
                if all(not (text == e.quote and loc == e.anchor) for e in sp.evidence):
                    sp.evidence.append(Evidence(source="prompt", quote=text, anchor=loc))
                    n_extended += 1
                entry["applied"] = True
        elif verdict == "new_rule":
            spd = dict(v.get("spec") or {})
            spd.setdefault("quote", text)
            added = _absorb({"specs": [spd]}, anchor=loc)
            n_new += added
            entry["applied"] = bool(added)
            if added:
                entry["spec_id"] = specs[-1].spec_id
        elif verdict == "not_a_rule":
            n_not_rule += 1
            entry["reason"] = (v.get("reason") or "").strip()
            entry["applied"] = True
        else:
            entry["applied"] = False
        verdict_log.append(entry)

    # ── normalization (unchanged) ──
    n_rebound = 0
    b_system = _BIND_SYSTEM.format(domain=domain)
    for s in specs:
        ev = "\n".join(f"  - {e.quote}" for e in s.evidence) or "  (none)"
        v = call_json(b_system, _BIND_USER.format(domain=domain, tool_list=tl, policy=policy,
                                                  kind=s.kind, rule=s.rule_text, evidence=ev),
                      model, temperature, max_tokens=800)
        if "bindings" not in v:
            continue
        new_b = _parse_bindings(v.get("bindings"))
        new_ct = bool(v.get("cross_tool", s.cross_tool))
        new_kind = str(v.get("kind", s.kind)).upper()
        if new_kind not in KINDS:
            new_kind = s.kind
        if ({(b["tool"], tuple(b["params"])) for b in new_b} != {(b["tool"], tuple(b["params"])) for b in s.bindings}
                or new_ct != s.cross_tool or new_kind != s.kind):
            n_rebound += 1
        s.kind, s.bindings, s.cross_tool = new_kind, new_b, new_ct
        s.relevant_tools = [b["tool"] for b in new_b]

    # ── binding coverage: catch the under-binding normalize keeps making ──
    # (category rules bound to one salient verb; see common.verify_binding_coverage)
    coverage = verify_binding_coverage(agent, specs, domain=domain,
                                       model=model, temperature=temperature)

    report = {"audit": "v2", "n_extract": n_extract, "n_enum_sentences": len(sentences),
              "n_unquoted": len(candidates), "n_claims_rejected": len(rejected),
              "n_extended": n_extended, "n_new_rules": n_new, "n_not_rule": n_not_rule,
              "n_rebound": n_rebound, "binding_coverage": coverage,
              "sentences": sentences, "verdicts": verdict_log,
              "rejected_claims": rejected}
    logger.info("Prompt mining (audit v2): %d specs (%d extraction + %d adjudicated new); "
                "%d/%d occurrences unquoted → %d extend, %d not-a-rule; %d claims rejected; "
                "%d rebound", len(specs), n_extract, n_new, len(candidates), len(sentences),
                n_extended, n_not_rule, len(rejected), n_rebound)
    return specialize(specs), report
