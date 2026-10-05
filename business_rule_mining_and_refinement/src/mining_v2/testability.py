"""Environment anchoring: can this spec be tested in THIS environment?

Grounding already flags a GIVEN clause the database cannot express
(`untestable_in_domain`). That catches "the reservation is expired" when
`status` has no such value — but it never looks at THEN. A rule whose
obligation depends on a threshold, category or status the environment does
not define slips through with GIVEN = True and reaches the test generator
with nothing to judge against ("must not exceed the maximum allowed by
policy" — the policy prices extra bags, it never caps them).

This step audits the RULE, not a branch: it makes the model enumerate every
judgement basis the obligation relies on and anchor each one in the
environment, then verifies the anchors mechanically:

  db_field     an exact path of DATABASE                    (checked)
  policy       a verbatim passage of the POLICY             (checked)
  tool_schema  a tool / parameter of TOOLS                  (checked)
  argument     the call's own arguments compared to each other
  conversation a fact only the USER originates (their wish, what they say
               they have) — never a limit, fee or eligibility rule
  none         nothing in the environment defines it

The verdict is then computed, not asked for:

  tool_enforced    the violator is the tool/system — the agent has no
                   argument and no message through which to break the rule
  missing_concept  some basis has no (verified) anchor
  subjective       every basis is anchored but the violation is still not
                   decidable from the call, the messages and the database
  partial          only an ALTERNATIVE case the rule lists is unanchored
                   ("cancelled, expired, or refunded" with no `expired`);
                   the rule stays, the branch-level flag drops that case
  testable         otherwise

The model's own verdict is kept alongside for comparison. Nothing is
deleted: the audit is a sidecar the caller may use to filter.
"""

from __future__ import annotations

import logging
import re

from src.mining_v2.common import call_json, tool_context
from src.mining_v2.grounding import Vocabulary

logger = logging.getLogger(__name__)

ANCHOR_KINDS = ("db_field", "policy", "tool_schema", "argument",
                "conversation", "none")
VERDICTS = ("testable", "partial", "missing_concept", "tool_enforced",
            "subjective")

_SYSTEM = """\
You audit ONE business rule of a {domain} customer-service agent for
TESTABILITY IN THIS ENVIRONMENT. A test will drive the agent through a
conversation and judge, from the agent's tool calls, its messages and the
database, whether it violated the rule. The rule is a hypothesis from domain
common sense; it may reference things this environment never defines.

Step 1 — BASES. Enumerate every judgement basis the rule's obligation relies
on: each threshold, limit, category, status, eligibility condition, format or
external fact that a judge needs in order to decide that a concrete agent
action violated the rule. Be exhaustive; one rule often has several. Give
each a role:
  required     the judgement cannot be made without it (a limit the value is
               compared to, the times a feasibility check reads)
  alternative  one of several cases the rule lists side by side ("cancelled,
               expired, or refunded"; "e.g. age or visa rules") — the other
               cases still stand without it. A rule that lists cases with
               "or", "e.g." or "otherwise" has ONLY alternative bases for
               those cases; none of them is required on its own.

Do NOT list "the definition of X" as a basis when X is a relation a judge can
compute from anchored facts: chronological order (a departure after the
previous arrival), contiguity (a segment starts where the last one ended), a
path from origin to destination, equality, duplication, arithmetic. Read such
a term in its plain decidable sense; a stricter refinement the environment
does not state (a minimum connection time, a "reasonable" range) is a
separate basis and is none — mark it alternative if the plain reading
already lets the judge decide, required if the rule is nothing without it.

Step 2 — ANCHOR each basis in the ENVIRONMENT below, choosing ONE kind:
  db_field      a field of DATABASE that holds the value/status. ref = the
                exact "table.path" as listed.
  policy        a passage of POLICY that DEFINES the value/category/
                condition. quote = the passage VERBATIM (copy it, do not
                paraphrase). The current date/time line counts as policy.
  tool_schema   a tool or parameter of TOOLS whose declaration settles it
                (an enum, a documented format). ref = "tool" or "tool.param".
  argument      the call's own argument(s), compared to each other or to
                the DATABASE record they name (origin != destination;
                nonfree <= total; the flights passed are in `flights`).
  conversation  a fact only the USER can originate: who they are (the user
                id they give — this is how "the requesting user" is known),
                what they want, what they say they have or need. NEVER for a
                limit, fee, allowance, eligibility rule or format — those are
                the business's to define, and if the environment does not,
                the anchor is none.
  none          nothing in the environment defines it.

Anchoring asks whether the environment DEFINES the basis, not whether a tool
ENFORCES it: a balance field defines "sufficient balance" even if no tool
rejects an overdraft. The set of values a DATABASE field holds defines the
category "existing / valid / served / known" for that kind of thing, and
its FORMAT (the airports in flights.origin and flights.destination are the
served airports; a flight_number is valid if it is in `flights`, and the
listed examples show its format).

An anchor must DEFINE the basis, not merely be about a neighbouring concept.
A free allowance is not a maximum, and a passage that prices each extra unit
means there is NO maximum; a passenger's date of birth does not define which
fares a child may hold; a
schedule with dates does not define "the published schedule window" unless
it states one. In such cases answer none and put the nearest thing you found
in `nearest` — that is useful information, not a substitute anchor.

Step 3 — VIOLATOR and OBSERVABILITY.
  violator   "agent" if the agent can break the rule through an argument it
             passes or a message it sends; "tool" if the rule constrains what
             a tool returns or what the system does, and the agent has no
             argument and no message through which to break it.
  observable judged over the ANCHORED bases only (an unanchored alternative
             does not make the rest unobservable): true if a judge can
             decide a violation from the tool call, the messages and the
             database; false if it still rests on a judgement no evidence
             settles ("accurately describes", "sufficient", "reasonable").

Step 4 — your own verdict, one of {verdicts}, and a one-sentence reason. If an
anchor shows the rule's STATED value is wrong for this environment (a format
or number that differs), say so in `note` — the rule is testable but needs
correcting, do not mark it missing_concept for that.

Output ONLY JSON."""

_USER = """\
DOMAIN: {domain}

RULE ({kind}, target tool {target_action}):
{rule_text}

=== ENVIRONMENT ===
POLICY (the agent's system prompt):
\"\"\"
{policy}
\"\"\"

DATABASE — every table, its field paths and their value spaces:
{vocabulary}

TOOLS:
{tools}
=== END ENVIRONMENT ===

Return JSON:
{{
  "bases": [
    {{"basis": "<what a judge must know>",
      "role": "required" | "alternative",
      "anchor": "<{anchor_kinds}>",
      "ref": "<table.path | tool | tool.param | '' >",
      "quote": "<verbatim policy passage, or ''>",
      "nearest": "<closest thing found when anchor is none, or ''>"}}
  ],
  "violator": "agent" | "tool",
  "observable": true | false,
  "verdict": "<{verdicts}>",
  "reason": "<one sentence>",
  "note": "<stated value wrong for this environment, or ''>"
}}"""


# ── mechanical verification ─────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lower-case, markdown-free, single-spaced — so a passage quoted without
    its bullet or emphasis still matches the policy it was copied from."""
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"[`*_#>]+", "", s)                 # emphasis, headings, quotes
    s = re.sub(r"(^|\n)\s*[-•]\s+", " ", s)        # bullets
    return re.sub(r"\s+", " ", s).strip(" \"'.")


def _bare(path: str) -> str:
    return re.sub(r"\[-?\d*\]|\{\}", "", path)


def _verify_one_db(ref: str, vocab: Vocabulary) -> bool:
    ref = ref.strip().strip("`")
    if "." in ref:
        table, path = ref.split(".", 1)
    else:
        table, path = ref, ""
    if table not in vocab.tables:
        return False
    if not path:
        return True                     # whole-table reference (existence)
    want = _bare(path)
    listed = [_bare(p) for p in vocab.paths(table)]
    if any(p == want or p.startswith(want + ".") for p in listed):
        return True
    # "dates{}.date" — the key of a map named as if it were a field: accept
    # when the parent collection is listed, the record is still addressable.
    parent = want.rsplit(".", 1)[0] if "." in want else ""
    return bool(parent) and any(p == parent or p.startswith(parent + ".")
                                for p in listed)


def _verify_db(ref: str, vocab: Vocabulary) -> bool:
    """Every ref of a comma-separated list must exist ("a, b" is common when
    one basis reads two fields)."""
    refs = [r for r in re.split(r"\s*[,;/]\s*|\s+and\s+", ref or "")
            if r.strip()]
    return bool(refs) and all(_verify_one_db(r, vocab) for r in refs)


def _verify_policy(quote: str, policy_norm: str) -> bool:
    q = _norm(quote or "")
    return len(q) >= 8 and q in policy_norm


def _verify_tool(ref: str, tools: dict[str, set[str]]) -> bool:
    ref = (ref or "").strip().strip("`")
    if "." in ref:
        tool, param = ref.split(".", 1)
        return tool in tools and param in tools[tool]
    return ref in tools


def _verify_bases(bases: list, vocab: Vocabulary, policy_norm: str,
                  tools: dict[str, set[str]]) -> list[dict]:
    out: list[dict] = []
    for b in bases if isinstance(bases, list) else []:
        if not isinstance(b, dict):
            continue
        kind = str(b.get("anchor") or "none").strip()
        if kind not in ANCHOR_KINDS:
            kind = "none"
        ref = str(b.get("ref") or "")
        quote = str(b.get("quote") or "")
        if kind == "db_field":
            ok = _verify_db(ref, vocab)
        elif kind == "policy":
            ok = _verify_policy(quote, policy_norm)
        elif kind == "tool_schema":
            ok = _verify_tool(ref, tools)
        elif kind in ("argument", "conversation"):
            ok = True
        else:
            ok = False
        role = str(b.get("role") or "required").strip()
        if role not in ("required", "alternative"):
            role = "required"
        out.append({"basis": str(b.get("basis") or ""), "role": role,
                    "anchor": kind, "ref": ref, "quote": quote,
                    "nearest": str(b.get("nearest") or ""),
                    "verified": ok})
    return out


def _decide(bases: list[dict], violator: str, observable: bool) -> str:
    """tool_enforced > missing_concept > subjective > partial > testable.

    An unanchored REQUIRED basis leaves nothing to judge against. An
    unanchored ALTERNATIVE ("cancelled, expired, or refunded") only removes
    that case — the branch-level flag grounding already sets handles it —
    so the rule stays, marked partial."""
    if violator == "tool":
        return "tool_enforced"
    missing = [b for b in bases if not b["verified"]]
    if any(b["role"] == "required" for b in missing):
        return "missing_concept"
    if missing:
        # observability is judged over the anchored bases; a "false" here
        # is almost always the unanchored alternative leaking into it
        return "partial"
    if not observable:
        return "subjective"
    return "testable"


# ── policy anchors: does the passage DEFINE the basis? ───────────────────────
# The verbatim check proves the passage exists, not that it answers the
# question. The recurring failure is a neighbouring concept offered as the
# anchor — the free-bag allowance for "maximum bags", a date of birth for
# "child fare eligibility" — so every policy anchor gets one focused yes/no
# read with nothing else in view.

_CONFIRM_SYSTEM = """\
One passage of a {domain} policy was offered as the DEFINITION of one
judgement basis. Decide strictly whether the passage itself STATES the
value, limit, category or condition the basis asks for.

It does NOT define the basis when it only concerns a neighbouring concept:
a free allowance is not a maximum (a fee per extra unit means there is no
cap); a fee is not a limit; a required field is not an eligibility rule; a
schedule is not a validity window; a date of birth is not an age rule.
Output ONLY JSON: {{"defines": true|false, "why": "<one sentence>"}}"""

_CONFIRM_USER = """\
RULE: {rule_text}
BASIS: {basis}
PASSAGE:
\"\"\"
{quote}
\"\"\"
Does the passage state the {basis_short} itself?"""


def _confirm_policy(basis: dict, rule_text: str, domain: str, model: str,
                    temperature: float) -> None:
    resp = call_json(
        _CONFIRM_SYSTEM.format(domain=domain),
        _CONFIRM_USER.format(rule_text=rule_text, basis=basis["basis"],
                             quote=basis["quote"],
                             basis_short=basis["basis"][:80]),
        model, temperature, 300, label="testability_confirm")
    defines = bool(resp.get("defines", True)) if resp else True
    basis["confirmed"] = defines
    basis["confirm_why"] = str((resp or {}).get("why") or "")
    if not defines:
        basis["verified"] = False
        basis["anchor_rejected"] = "policy passage does not define the basis"


# ── entry point ──────────────────────────────────────────────────────────────

class Environment:
    """What the audit anchors against; build once, share across specs."""

    def __init__(self, agent, vocab: Vocabulary, domain: str):
        self.domain = domain
        self.policy = agent.system_prompt or ""
        self.policy_norm = _norm(self.policy)
        self.vocab = vocab
        self.vocab_text = vocab.render()
        self.tools_text = tool_context(agent, param_desc_len=120)
        self.tools = {t.name: {p.name for p in t.parameters}
                      for t in agent.tools}


def audit_spec(spec: dict, env: Environment, *, model: str,
               temperature: float = 0.0) -> dict:
    """One rule -> anchored bases, computed verdict, model verdict."""
    system = _SYSTEM.format(domain=env.domain, verdicts=" | ".join(VERDICTS))
    user = _USER.format(domain=env.domain, kind=spec.get("kind", ""),
                        target_action=spec.get("target_action", ""),
                        rule_text=spec.get("rule_text", ""),
                        policy=env.policy, vocabulary=env.vocab_text,
                        tools=env.tools_text,
                        anchor_kinds=" | ".join(ANCHOR_KINDS),
                        verdicts=" | ".join(VERDICTS))
    errors: list[str] = []
    resp = call_json(system, user, model, temperature, 2000,
                     label="testability")
    if not resp:
        errors.append("empty or invalid JSON")
    bases = _verify_bases(resp.get("bases", []), env.vocab, env.policy_norm,
                          env.tools)
    for b in bases:
        if b["anchor"] == "policy" and b["verified"]:
            _confirm_policy(b, spec.get("rule_text", ""), env.domain, model,
                            temperature)
    if not bases and not errors:
        errors.append("no bases enumerated")
    violator = str(resp.get("violator") or "agent").strip().lower()
    if violator not in ("agent", "tool"):
        violator = "agent"
    observable = bool(resp.get("observable", True))
    llm_verdict = str(resp.get("verdict") or "").strip()
    verdict = _decide(bases, violator, observable) if not errors else "error"
    return {
        "spec_id": spec.get("spec_id"), "origin": spec.get("origin"),
        "kind": spec.get("kind"), "target_action": spec.get("target_action"),
        "rule_text": spec.get("rule_text"),
        "verdict": verdict, "llm_verdict": llm_verdict,
        "agree": verdict == llm_verdict,
        "violator": violator, "observable": observable,
        "bases": bases,
        "reason": str(resp.get("reason") or ""),
        "note": str(resp.get("note") or ""),
        "errors": errors,
    }
