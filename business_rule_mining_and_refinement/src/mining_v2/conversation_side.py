"""The conversation side of one GWT scenario, derived from what grounding
could NOT plant.

Grounding partitions GIVEN: every atomic clause becomes a database condition
or lands in `nonDBconditions` (the third slot grounding returns) with a reason and a `realizable` tag saying who can
still make it true. This step takes the clauses tagged "conversation" as its
INPUT and restates each from the user's point of view — it does not re-read
GIVEN and decide for itself what the database left out. Two derivations of
the same text from two prompts disagreed on 42 of 78 entries (a price
premise lost, "no reservation exists" dropped, DB-planted facts repeated);
one derivation feeding the other cannot.

It also lifts the QUALIFIERS out of WHEN: what the request must contain
beyond the bare action ("...and requests to change or cancel" — two acts;
"...specifying total_baggages and nonfree_baggages" — two values). The bare
action is the trigger and is stored separately; a "qualifier" whose quote
is really the whole of WHEN is removed mechanically, which is what keeps
this extractor from padding. (A token-subset test against the trigger cannot
do this job: the trigger usually IS the whole WHEN, so every genuine
fragment of it is a subset and would be removed along with the restatements.)

Clauses tagged "fixture" (ownership) and "none" (no field or value exists;
nobody can establish them) are not handed to the user: the first is the
lookup stage's job, the second makes the branch untestable in this domain
and the caller flags it.
"""

from __future__ import annotations

import re

from src.mining_v2.common import call_json
from src.mining_v2.user_requirements import _content

_SYSTEM = """\
You write the CONVERSATION side of ONE test scenario for a {domain}
customer-service agent. The database side is settled: some of GIVEN is
planted as records, and the CLAUSES listed below are the part no record can
carry — true only if the user makes them true by what they say or do.

Two outputs.

"preconditions" — one entry per listed clause, in order, tagged with its
clause_index: the same fact restated as something the USER says or does.
  "the reason for cancellation is not covered by insurance"
    -> "the user gives a reason for cancelling that the insurance does not
        cover"
  "the requested date is beyond the published schedule"
    -> "the user asks about a date years in the future"
Keep the fact; change only the point of view. Never add, merge or drop a
clause. When the clause is already user-side, keep it nearly verbatim.

"when_qualifiers" — what WHEN requires of the request BEYOND the bare
action: additional acts, values or circumstances the user must include for
the request to be this scenario ("complains about the delay and asks to
change or cancel" has two acts; "specifying total_baggages and
nonfree_baggages" names two values the user must give). Each entry carries
"source_quote", a fragment copied VERBATIM from WHEN. The bare action ("the
user requests to book a reservation") is not a qualifier — a WHEN that is
only the action yields an empty list.

Rules:
- only the USER's words or actions; nothing the agent does, no tool names;
- nothing from THEN, and no fact the scenario does not state.
Return only JSON:
{{"preconditions": [{{"clause_index": <i>, "condition": "<text>"}}, ...],
  "when_qualifiers": [{{"condition": "<text>", "source_quote": "<verbatim>"}}, ...]}}"""

_USER = """\
RULE (context): {rule}

SCENARIO
  GIVEN: {given}
  WHEN : {when}
  THEN (context only — nothing here is the user's to supply): {then}

CLAUSES (restate each from the user's side; keep the numbering):
{clauses}

Return the JSON object."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _is_whole_when(quote: str, when: str) -> bool:
    """A qualifier standing on (nearly) all of WHEN is WHEN restated, not a
    demand beyond the bare action."""
    w = _content(when)
    return bool(w) and len(_content(quote) & w) >= 0.8 * len(w)


def _check(result, n_clauses: int, when: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["result must be a JSON object"]
    pre = result.get("preconditions")
    if not isinstance(pre, list):
        errors.append("preconditions must be a list")
        pre = []
    seen: list[int] = []
    for i, item in enumerate(pre):
        if not isinstance(item, dict):
            errors.append(f"preconditions[{i}] is not an object")
            continue
        idx = item.get("clause_index")
        if not isinstance(idx, int) or not 0 <= idx < n_clauses:
            errors.append(f"preconditions[{i}].clause_index must be in "
                          f"[0, {n_clauses - 1}]")
        else:
            seen.append(idx)
        text = item.get("condition")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"preconditions[{i}].condition must be a non-empty string")
        elif text.strip().lower().startswith("the agent"):
            errors.append(f"preconditions[{i}] has the agent as subject; "
                          "restate it as the user's words or actions")
    missing = [i for i in range(n_clauses) if i not in seen]
    dupes = sorted({i for i in seen if seen.count(i) > 1})
    if missing:
        errors.append(f"clauses {missing} have no precondition — restate every clause")
    if dupes:
        errors.append(f"clauses {dupes} were restated more than once — exactly one each")
    quals = result.get("when_qualifiers")
    if not isinstance(quals, list):
        errors.append("when_qualifiers must be a list (possibly empty)")
        quals = []
    hay = _norm(when)
    for i, item in enumerate(quals):
        if not isinstance(item, dict) or not isinstance(item.get("condition"), str) \
                or not item["condition"].strip():
            errors.append(f"when_qualifiers[{i}].condition must be a non-empty string")
            continue
        quote = item.get("source_quote")
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f"when_qualifiers[{i}].source_quote is missing")
        elif _norm(quote) not in hay:
            errors.append(f"when_qualifiers[{i}].source_quote {quote!r} is not a "
                          "verbatim fragment of WHEN")
    return errors


def extract_conversation_side(gwt: dict, dropped: list[dict], rule_text: str,
                              *, trigger: str = "", domain: str = "general",
                              model: str = "gpt-4.1", temperature: float = 0.0
                              ) -> tuple[dict, list[str]]:
    """-> ({"conversation_preconditions", "when_qualifiers", "meta"}, errors).

    conversation_preconditions
                    the "conversation"-realizable dropped clauses, restated
                    user-side, each carrying its source clause;
    when_qualifiers WHEN's demands beyond the bare action, with verbatim
                    provenance, minus any that merely restate the trigger.
    """
    when = str(gwt.get("when", "")).strip()
    clauses = [d for d in dropped if d.get("realizable", "conversation") == "conversation"]
    meta = {"clauses_in": len(clauses), "qualifiers_restating_trigger": 0}
    block = "\n".join(f"{i}. {c.get('clause', '')}" for i, c in enumerate(clauses)) \
        or "(none)"
    user = _USER.format(rule=rule_text or "(not provided)",
                        given=gwt.get("given", ""), when=when,
                        then=gwt.get("then", ""), clauses=block)
    result: dict = {}
    errors: list[str] = []
    for attempt in range(2):
        result = call_json(_SYSTEM.format(domain=domain), user, model, temperature,
                           max_tokens=900, label="conversation_side")
        errors = _check(result, len(clauses), when)
        if not errors:
            break
        if attempt == 0:
            user += ("\n\nRETRY: the previous answer failed validation:\n- "
                     + "\n- ".join(errors) + "\nReturn a corrected JSON object.")
    if errors:
        # Fall back to the clauses themselves: provenance-perfect, merely not
        # rephrased. Losing them would be the one unacceptable outcome.
        pre = [{"condition": c.get("clause", ""), "source_clause": c.get("clause", ""),
                "reason": c.get("reason", "")} for c in clauses]
        return {"conversation_preconditions": pre, "when_qualifiers": [], "meta": meta}, errors

    pre = []
    for item in sorted(result["preconditions"], key=lambda x: x["clause_index"]):
        src = clauses[item["clause_index"]]
        pre.append({"condition": item["condition"].strip(),
                    "source_clause": src.get("clause", ""),
                    "reason": src.get("reason", "")})
    quals = []
    for item in result["when_qualifiers"]:
        text, quote = item["condition"].strip(), item["source_quote"].strip()
        if _is_whole_when(quote, when):
            meta["qualifiers_restating_trigger"] += 1
            continue
        quals.append({"condition": text, "source_quote": quote})
    return {"conversation_preconditions": pre, "when_qualifiers": quals, "meta": meta}, []
