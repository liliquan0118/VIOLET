"""Two-step Given grounding: route clauses to tables, then encode them.

The single-call compiler in grounding.py asks one completion to segment the
Given, route every fact to a table, and encode field/op/value/quant all at
once. Measured on the long negated conjunctions of the De Morgan branches,
that call drops a clause roughly once in eight runs — silently, because
nothing records what the clause set was supposed to be.

This module splits the task so coverage becomes checkable:

  Step 1  segment_clauses   split GIVEN into atomic factual clauses and route
                            each to the table whose FIELD stores the fact;
                            non-database clauses are routed to null with a
                            reason. Also names the root (the entity WHEN
                            acts on).
  Step 2  encode_clauses    one batched call encodes EVERY routed clause into
                            {table, path, op, value[, quant]}, tagged with its
                            clause index. Code — not the model — verifies the
                            1:1 coverage against Step 1's list and retries
                            with the missing clauses named. A clause DATABASE
                            cannot express is answered with "unencodable"
                            instead: it satisfies coverage and joins the
                            dropped list, so the check can never again be
                            satisfied by inventing a nearby field.

The public entry point `compile_condition_v2` keeps the same signature and
return shape as grounding.compile_condition, so callers can A/B the two.
"""

from __future__ import annotations

import json
import logging
import re

from src.mining_v2.grounding import Vocabulary, validate_condition

logger = logging.getLogger(__name__)


# ── Step 1: segment and route ────────────────────────────────────────────────

_SEGMENT_SYSTEM = """\
You prepare ONE precondition of a {domain} agent test for compilation into
database conditions. Do NOT write conditions — you only split and route.

The precondition is the GIVEN of a Given-When-Then scenario: the state the
database must already be in when the test conversation starts.

Split GIVEN into its atomic factual clauses and route each clause to the
table whose field stores that fact. Also name the root.

## root

The table of the entity WHEN acts on (the booking being cancelled, the order
being returned). When WHEN only involves the person, the root IS the
person's table. Use null only when WHEN acts on no entity of DATABASE — an
empty GIVEN empties the clauses, never the root.

## Clauses

- one entry per atomic fact GIVEN states — never merge two facts into one
  entry and never leave one out; a negated fact is one clause ("the flight
  is NOT cancelled by the airline");
- alternatives over ONE field are ONE clause ("the user is a silver or gold
  member"; "the booking is cancelled or refunded" — both name states of the
  booking's status) — never split such an "or": clauses are ANDed
  downstream, and splitting turns "or" into an impossible "and". Judge this
  by what the alternatives are ABOUT, not by which of them DATABASE happens
  to list: alternative states of one property stay one clause even when the
  value space is missing some of them, and Step 2 keeps the ones it can.
  An "or" joining alternatives about DIFFERENT properties ("the account does
  not exist or is deactivated") is not one field's value list: ANDing them is
  wrong and keeping one silently tests a different scenario. Route that whole
  clause to null, reason "disjunction over different properties";
- "table": the table whose FIELD stores the fact — pick it by field
  ownership in DATABASE, which may differ from the root ("some segment has
  already been flown" routes to the table holding the per-day flight
  status, not to the booking);
- a clause that no listed field stores — it exists only in the dialogue
  (the user's wants, claims, reasons) or the schema simply has no field for
  it — gets "table": null and a one-clause "reason".
- "the provided id matches no record" is dialogue too — the user simply
  says an unknown id; route it to null. But a record IN some state ("the
  reservation is cancelled") is an ordinary field condition — keep it even
  when no stored record currently satisfies it. So is "does NOT have an X
  with property P" ("no reservation with cancelled flights"): that is the
  NEGATION of P, not a non-existence claim — and it routes to the table
  whose field stores P itself ("cancelled flights" is the per-day flight
  status, NOT the reservation's own status field).
- a comparison between two STORED or computed values ("the new total is
  lower than the original price") cannot be expressed — a condition compares
  one field against a literal only; route it to null.
- a clause about WHO the record belongs to — whether the root is or is not
  the property of the person making the request ("the booking does not
  belong to the requesting user", "the line belongs to the caller") — is not
  a field condition either: no stored value names "the requester". It is a
  relation between the two things the lookup returns, so route it to null
  and add "relation": "owner" or "not_owner" to that clause. The lookup
  stage pairs a record with a non-owner when you say "not_owner", which is
  the only way an ownership scenario can be set up at all.
- so is a clause built on a value the CONVERSATION supplies rather than one
  you can write down here. The tell is a back-reference to something WHEN or
  the user names: "no flights are operated on THAT date", "the REQUESTED
  airport is not served". No stored record fixes
  those — the dialogue does. Route the clause to null and name the missing
  value in the reason. Never invent a placeholder for it, and never fall
  back on a claim about everything ("no flights AT ALL") in place of the one
  about the value the user names.

GIVEN saying "True" states no precondition: return an empty clause list.

Return only valid JSON:
{{"root": "<table>" | null,
  "clauses": [{{"clause": "<one fact>", "table": "<table>" | null,
               "reason": "<only when table is null>",
               "relation": "<only for an ownership clause: owner|not_owner>"}}, ...]}}"""

_SEGMENT_USER = """\
THE RULE BEING TESTED: {rule}

THE SCENARIO
  GIVEN (split THIS): {condition}
  WHEN  (context; names the entity the root comes from): {trigger}
  THEN  (context only): {then}

DATABASE — every table, its field paths and their value spaces:
{vocabulary}

Return the JSON object (root, clauses)."""


# ── Step 2: encode the routed clauses ────────────────────────────────────────

_ENCODE_SYSTEM = """\
You encode the routed clauses of ONE {domain} test precondition into
executable database conditions. The clauses are already split and routed to
tables — encode EVERY listed clause; do not re-split, re-route, merge or
drop any.

## Each condition

  {{"clause_index": <i>, "table": "<table>", "path": "<path>", "op": "<op>",
    "value": <value>}}
  plus, when needed: "quant": "all"

Emit at least one condition per listed clause, tagged with that clause's
index, on that clause's routed table.

  table   the table whose field the condition constrains — pick it by field
            ownership in DATABASE, which may differ from the root: "the user
            is a gold member" is a condition on `users.membership`; "some
            segment has already been flown" is one on `flights.dates{{}}.status`
            (the fact lives on the flight, not on the booking). How the root's
            records connect to that table is resolved later, at entity-query
            time.
    path    the field the condition constrains, copied EXACTLY from DATABASE
            below. The "[]" and "{{}}" markers are part of the path:
            "dates.status" is not "dates{{}}.status". A positional claim may
            replace "[]" with an index: "flights[0].date" is the FIRST element,
            "flights[-1].date" the LAST ("the first segment", "the final leg").
            Ordered lists only — "{{}}" collections have no order and take no
            index.
    op      eq ne lt le gt ge — compare a value the path reaches
            in — the field's value is one of the given list; an either/or
            clause lists every alternative ("silver or gold" -> op "in",
            value ["silver", "gold"]); contains — the field's value contains
            the given text (substring, strings only)
            count_eq count_gt count_lt count_ge count_le — compare HOW MANY
            values the path reaches against a non-negative integer. Only for
            clauses about HOW MANY ("more than one segment"); a clause about a
            property of some element compares the element's field instead. To
            count a collection's elements, address the collection itself
            ("flights[]"): the one case where the path is a listed path's
            PREFIX rather than a listed path.
            A count covers EVERY entry the collection holds, never the
            subset matching some key or description: `count_eq 0` says the
            record holds NO entries at all, which is almost never what a
            clause means. "No flight on the date the user names" is about ONE
            key of a "{{}}" mapping and no count expresses it — report such a
            clause unencodable instead of counting the whole mapping.
    value   the literal to compare against: "in" takes a list, counts take a
            non-negative integer
    quant   "any" (default): some value the path reaches satisfies the
            comparison; "all": every one must, and at least one exists. A
            universal clause ("none ...", "every ...", "all ...") needs
            quant "all", and so does a NEGATED fact on a path that reaches
            MANY values (a "[]" list, a "{{}}" mapping, or another table's
            records): "the flight is NOT cancelled" must hold of every
            flight the scenario reaches — the negation of "some value is X"
            is "no value is X". Plain `ne` without quant fits only
            single-valued paths; NEVER put "all" on a single-valued field —
            "all" also requires the value to exist, so a record whose
            optional field is absent (an uncancelled reservation's status)
            would wrongly fail. Never on count_*.

## When a clause cannot be encoded

A clause reaches you because Step 1 judged some table's fields to be the
right home for it. With DATABASE in front of you that judgement can turn out
wrong: no listed path expresses the clause (the property is simply not
stored, or the state it names is absent from the field's value space), or the
value to compare against is not a literal you can write (it is whatever the
user says during the conversation). Do NOT approximate it with the nearest
field. Return that clause as

  {{"clause_index": <i>, "unencodable": "<what DATABASE is missing>"}}

and encode the other clauses normally. An approximation is worse than an
omission: the planted state then differs from the one the scenario names, so
the run silently tests something else and still looks healthy. Use this only
when DATABASE really cannot express the clause — not to avoid a hard
encoding.

## Values are enums — cover every value that qualifies

When DATABASE lists a field's value space, a condition means those enum
values, not the everyday word. A condition about something having HAPPENED or
COMPLETED must include every listed value that implies it, including
in-progress states: with statuses [available, landed, cancelled, delayed,
flying, on time], "the flight already departed" is op "in",
value ["landed", "flying"] — not "landed" alone. The negation of an event
("not yet flown") includes the values where the event never happened AND the
ones where it never will (a cancelled entry was not flown) — pending states
are not the only "not yet". Never compare against a value the listing does
not contain.

When a clause OFFERS ALTERNATIVE states and the value space lists only some
of them, encode those ("in" over the listed ones) and leave the rest out:
the scenario is then narrower than the clause but is still one the clause
names. That licence needs alternatives in the clause itself and covers
nothing else. A clause naming ONE state the value space does not list ("the
booking is expired", "the account is deactivated") has no alternative to
fall back on: it is unencodable. Never substitute the closest listed value —
"expired" is not "cancelled", and a scenario that plants the wrong state
tests the wrong rule while looking perfectly healthy.

## Time values are symbolic

Never compute a concrete timestamp for a temporal bound — emit it
symbolically:
  today / now:      "value": {{"ref":"$TODAY"}}  /  {{"ref":"$NOW"}}
  relative bound:   "value": {{"ref":"$NOW","offset":{{"hours":-24}}}}
                    "value": {{"ref":"$TODAY","offset":{{"days":7}}}}
`ref` is `$NOW` or `$TODAY`; offset units are `hours`, `days`, `years`
(negative = past). An absolute date written in GIVEN stays a literal string
("2024-05-20").

Never invent a path. Return only valid JSON:
{{"conditions": [{{"clause_index": ..., "table": "...", "path": "...",
                  "op": "...", "value": ...}},
                 {{"clause_index": ..., "unencodable": "..."}}, ...]}}"""

_ENCODE_USER = """\
GIVEN (context for reading the clauses): {condition}

CLAUSES TO ENCODE (encode every one, on its routed table):
{clauses_block}

DATABASE — every table, its field paths and their value spaces:
{vocabulary}

Return the JSON object (conditions)."""


# ── Step 3: who can establish what no condition could ────────────────────────

_CLASSIFY_SYSTEM = """\
For ONE {domain} test scenario, judge each GIVEN clause the grounding step
found no database field for: who can make it true?

  "conversation"  the user, by what they say or do — their wants, claims and
                  reasons ("the reason for cancelling is covered by
                  insurance": they give such a reason; "the reason is NOT
                  covered": they give one that is not); an id, date or
                  amount they name ("the id matches no record", "the account
                  does not exist": they simply say an unknown one); and anything
                  that follows from the request they choose to make ("the
                  new total is higher than the original": they ask for a
                  pricier cabin; "the passenger count changes": they ask to
                  add one).
  "none"          nobody. DATABASE has no field or value for the state and
                  no request brings it about. The user SAYING it does not
                  make it so — the agent would look and see an ordinary
                  record.
  "field"         the grounding step was wrong: a listed field DOES carry
                  the fact. Say so and the clause goes back to be encoded.
                  A record's existence is its id field being non-empty; a
                  payment that was made is in the payment history; a state
                  an enum value expresses is a field condition.

"none" and "field" must name what you looked for, as "missing":
  a field           "<table>.<path>"           copied from DATABASE below,
                                               or the path you expected
  a value           "<table>.<path> = <value>"  when the field exists but
                                               its value space lacks the
                                               state
The name is checked against DATABASE. Look before you answer "none": if the
field you name turns out to exist, the clause is encoded after all.

Return only JSON:
{{"verdicts": [{{"clause_index": <i>,
               "realizable": "conversation" | "none" | "field",
               "missing": "<required unless conversation>",
               "why": "<one sentence>"}}, ...]}}"""

_CLASSIFY_USER = """\
SCENARIO (context)
  GIVEN: {condition}
  WHEN : {trigger}
  THEN : {then}

CLAUSES WITHOUT A CONDITION (judge every one; keep the numbering):
{clauses_block}

DATABASE — every table, its field paths and their value spaces:
{vocabulary}

Return the JSON object (verdicts)."""

_MISSING_RE = re.compile(r"^\s*([A-Za-z_]\w*)\.([\w.\[\]{}-]+)\s*(?:=\s*(.+?))?\s*$")


def _bare(path: str) -> str:
    """A path with its collection markers removed, so "dates" meets
    "dates{}" and "payment_methods.id" meets "payment_methods{}.id"."""
    return re.sub(r"\[-?\d*\]|\{\}", "", path)


def _lookup_missing(missing: str, vocab: Vocabulary) -> tuple[bool, str]:
    """Does what the model calls missing actually exist? -> (exists, note).

    "table.path" exists when the path is listed for that table;
    "table.path = value" exists when the field has a value space and the
    value is in it. Anything unparseable is treated as not existing — the
    verdict then stands, but the note records that it could not be checked.
    """
    m = _MISSING_RE.match(missing or "")
    if not m:
        return False, "could not be checked (not of the form table.path)"
    table, path, value = m.group(1), m.group(2), m.group(3)
    if table not in vocab.tables:
        return False, f"table {table!r} is not in the database"
    # Match on the marker-free form, and accept a prefix of a listed path:
    # "dates" names the dates{} mapping even though only its leaves are
    # listed. Anything less strict than the vocabulary's own notation and a
    # wrong "none" slips through as verified.
    listed = {p: _bare(p) for p in vocab.paths(table)}
    want = _bare(path)
    hits = [p for p, b in listed.items() if b == want or b.startswith(want + ".")]
    if not hits:
        return False, f"{table}.{path} is not a field"
    normalized = next((p for p in hits if listed[p] == want), hits[0])
    if value is None:
        return True, f"{table}.{normalized} exists"
    field = vocab.field(table, normalized)
    values = getattr(field, "values", None) if field else None
    if not values:
        return True, f"{table}.{path} exists and has no fixed value space"
    plain = value.strip().strip("'\"")
    if plain in {str(v) for v in values}:
        return True, f"{plain!r} is in the value space of {table}.{path}"
    return False, f"{plain!r} is not in the value space of {table}.{path}"


def _eq_conflicts(conditions: list) -> list[str]:
    """Two `eq` on one single-valued path with different values.

    ANDed, `status eq "delivered"` and `status eq "exchange requested"` can
    never both hold of one scalar field — the clause meant "either", which is
    `in` over both values. A path through "[]" or "{}" reaches many values,
    where two eq CAN hold of different elements, so only marker-free paths
    are checked.
    """
    seen: dict[tuple, set] = {}
    for cond in conditions:
        if not isinstance(cond, dict) or "unencodable" in cond or cond.get("op") != "eq":
            continue
        path = str(cond.get("path", ""))
        if "[]" in path or "{}" in path or "[" in path:
            continue
        seen.setdefault((cond.get("table"), path), set()).add(json.dumps(cond.get("value")))
    return [f"`{table}.{path}` is required to equal {sorted(vals)} at once — a "
            "single-valued field cannot; if the clause offers alternatives use "
            "op \"in\" with all of them"
            for (table, path), vals in seen.items() if len(vals) > 1]


_OR_CLAUSE_RE = re.compile(r"\bor\b", re.IGNORECASE)


def _or_split(conditions: list, routed: list[dict]) -> list[str]:
    """An "or" clause encoded as conditions on DIFFERENT fields.

    Conditions are ANDed, so "the first or last name is 'Test'" written as
    `first_name eq Test` AND `last_name eq Test` demands both — the opposite
    of the clause. Alternatives over one field are one `in`; alternatives
    over different fields are not expressible and must be reported
    unencodable (Step 1 should have routed them to null).
    """
    by_clause: dict[int, set] = {}
    for cond in conditions:
        if not isinstance(cond, dict) or "unencodable" in cond:
            continue
        idx = cond.get("clause_index")
        if isinstance(idx, int) and 0 <= idx < len(routed):
            by_clause.setdefault(idx, set()).add((cond.get("table"), cond.get("path")))
    return [f"clause {idx} ({routed[idx]['clause']!r}) says \"or\" but was "
            f"encoded as ANDed conditions on {sorted(p for _, p in paths)} — "
            "alternatives over one field are one \"in\"; over different "
            "fields, report the clause unencodable"
            for idx, paths in by_clause.items()
            if len(paths) > 1 and _OR_CLAUSE_RE.search(routed[idx]["clause"])]


def _check_verdicts(result, n_clauses: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["result must be a JSON object"]
    verdicts = result.get("verdicts")
    if not isinstance(verdicts, list):
        return ["verdicts must be a list"]
    seen: list[int] = []
    for i, v in enumerate(verdicts):
        if not isinstance(v, dict):
            errors.append(f"verdicts[{i}] is not an object")
            continue
        idx = v.get("clause_index")
        if not isinstance(idx, int) or not 0 <= idx < n_clauses:
            errors.append(f"verdicts[{i}].clause_index must be in [0, {n_clauses - 1}]")
        else:
            seen.append(idx)
        tag = v.get("realizable")
        if tag not in ("conversation", "none", "field"):
            errors.append(f"verdicts[{i}].realizable must be conversation|none|field")
        elif tag != "conversation":
            missing = str(v.get("missing") or "").strip()
            if not _MISSING_RE.match(missing):
                errors.append(
                    f"verdicts[{i}]: {tag!r} must name ONE field as "
                    "\"table.path\" or \"table.path = value\" (got "
                    f"{missing!r}) — no lists, no \"or\", no placeholders")
    missing = [i for i in range(n_clauses) if i not in seen]
    if missing:
        errors.append(f"clauses {missing} have no verdict — judge every clause")
    return errors


def _encode(condition: str, routed: list[dict], vocab: Vocabulary, *,
            domain: str, model: str, temperature: float, call_json) -> tuple[dict, list[str]]:
    """Step 2 as a callable: encode `routed` clauses, retry once on errors."""
    clauses_block = "\n".join(
        f"{i}. {c['clause']}  ->  table `{c['table']}`" for i, c in enumerate(routed))
    system2 = _ENCODE_SYSTEM.format(domain=domain)
    user2 = _ENCODE_USER.format(condition=condition, clauses_block=clauses_block,
                                vocabulary=vocab.render())
    result: dict = {}
    errors: list[str] = []
    for attempt in range(2):
        result = call_json(system2, user2, model, temperature,
                           max_tokens=1400, label="ground_encode")
        errors = _check_encoding(result, routed, vocab)
        if not errors:
            break
        if attempt == 0:
            user2 += ("\n\nRETRY: the previous answer failed validation:\n- "
                      + "\n- ".join(errors) + "\nReturn a corrected JSON object.")
    return result, errors


# ── helpers ──────────────────────────────────────────────────────────────────

def _check_segmentation(result: dict, vocab: Vocabulary) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["result must be a JSON object"]
    root = result.get("root")
    if root is not None and root not in vocab.tables:
        errors.append(f"root {root!r} is not a table of this database")
    clauses = result.get("clauses")
    if not isinstance(clauses, list):
        return errors + ["clauses must be a list (possibly empty)"]
    for i, entry in enumerate(clauses):
        if not isinstance(entry, dict):
            errors.append(f"clauses[{i}] is not an object")
            continue
        text = entry.get("clause")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"clauses[{i}].clause must be a non-empty string")
        table = entry.get("table")
        if table is not None and table not in vocab.tables:
            errors.append(f"clauses[{i}].table {table!r} is not a table "
                          "of this database")
    return errors


def _count_conflicts(conditions: list) -> list[str]:
    """Counts on ONE path that cannot hold together.

    "at most one credit card" and "more than one certificate" count DIFFERENT
    SUBSETS of the same payment list; the DSL counts a whole collection, so
    written on one path they become `count_gt 1` AND `count_le 1` — an
    unsatisfiable precondition that matches nothing and reports no error. The
    subset is what cannot be expressed, and the clause has to say so.
    """
    bounds: dict[tuple, list] = {}
    for cond in conditions:
        if not isinstance(cond, dict) or "unencodable" in cond:
            continue
        op, value = cond.get("op"), cond.get("value")
        if not isinstance(op, str) or not op.startswith("count_"):
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        low, high = bounds.setdefault((cond.get("table"), cond.get("path")),
                                      [0, float("inf")])
        if op == "count_eq":
            low, high = max(low, value), min(high, value)
        elif op == "count_gt":
            low = max(low, value + 1)
        elif op == "count_ge":
            low = max(low, value)
        elif op == "count_lt":
            high = min(high, value - 1)
        elif op == "count_le":
            high = min(high, value)
        bounds[(cond.get("table"), cond.get("path"))] = [low, high]
    return [f"the counts on `{table}.{path}` cannot hold together (they "
            f"require between {low} and {high} values): they count different "
            "SUBSETS of one collection, which no count expresses — report "
            "that clause unencodable"
            for (table, path), (low, high) in bounds.items() if low > high]


def _check_encoding(result: dict, routed: list[dict],
                    vocab: Vocabulary) -> list[str]:
    """Every routed clause covered, on its routed table, and each condition
    valid against the vocabulary.

    A clause may also come back as {"clause_index", "unencodable": reason}:
    that counts as covered and carries no predicate to validate. Coverage is
    what this check exists for — without a legal way to say "DATABASE cannot
    express this", the only way past it is to invent a condition, which is
    how "expired" once became `status eq cancelled`.
    """
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["result must be a JSON object"]
    conditions = result.get("conditions")
    if not isinstance(conditions, list):
        return ["conditions must be a list"]
    covered: set[int] = set()
    for i, cond in enumerate(conditions):
        if not isinstance(cond, dict):
            errors.append(f"conditions[{i}] is not an object")
            continue
        index = cond.get("clause_index")
        known = isinstance(index, int) and 0 <= index < len(routed)
        if not known:
            errors.append(f"conditions[{i}].clause_index must be an integer "
                          f"in [0, {len(routed) - 1}]")
        else:
            covered.add(index)
        if "unencodable" in cond:
            reason = cond.get("unencodable")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"conditions[{i}].unencodable must be a "
                              "non-empty reason")
            continue
        if known and cond.get("table") != routed[index]["table"]:
            errors.append(
                f"conditions[{i}] is on table {cond.get('table')!r} but "
                f"clause {index} was routed to "
                f"{routed[index]['table']!r}")
        stripped = {k: v for k, v in cond.items() if k != "clause_index"}
        errors.extend(f"conditions[{i}]: {e}"
                      for e in validate_condition(stripped, vocab))
    errors.extend(_count_conflicts(conditions))
    errors.extend(_eq_conflicts(conditions))
    errors.extend(_or_split(conditions, routed))
    missing = [i for i in range(len(routed)) if i not in covered]
    if missing:
        errors.extend(
            f"clause {i} ({routed[i]['clause']!r}) has no condition — encode "
            "every listed clause" for i in missing)
    return errors


# ── entry point ──────────────────────────────────────────────────────────────

def compile_condition_v2(
    condition: str,
    vocab: Vocabulary,
    *,
    trigger: str = "",
    then: str = "",
    rule: str = "",
    ctx: dict | None = None,
    domain: str = "general",
    model: str = "gpt-4.1",
    temperature: float = 0.0,
) -> tuple[str | None, list[dict], list[dict], list[str]]:
    """Natural-language precondition -> (root, conditions, dropped, errors).

    Same signature and return shape as grounding.compile_condition; the third
    slot — written to the artifact as `nonDBconditions` — holds the clauses
    no condition was written for (each
    {"clause", "reason"}) — those Step 1 routed to null, plus those Step 2
    reported unencodable — instead of free-form ungrounded entries. They are
    the conversation's business: what no record can establish, the dialogue
    must. `ctx` is unused at compile time — temporal bounds come back
    symbolic.
    """
    from src.mining_v2.common import call_json

    system = _SEGMENT_SYSTEM.format(domain=domain)
    user = _SEGMENT_USER.format(
        rule=rule or "(not provided)", condition=condition,
        trigger=trigger or "(not provided)", then=then or "(not provided)",
        vocabulary=vocab.render())

    seg: dict = {}
    errors: list[str] = []
    for attempt in range(2):
        seg = call_json(system, user, model, temperature,
                        max_tokens=900, label="ground_segment")
        errors = _check_segmentation(seg, vocab)
        if not errors:
            break
        if attempt == 0:
            user += ("\n\nRETRY: the previous answer failed validation:\n- "
                     + "\n- ".join(errors) + "\nReturn a corrected JSON object.")
    if errors:
        return None, [], [], [f"segment: {e}" for e in errors]

    root = seg.get("root")
    clauses = [c for c in seg.get("clauses") or [] if isinstance(c, dict)]
    routed = [c for c in clauses if c.get("table")]
    dropped: list[dict] = []
    for c in clauses:
        if c.get("table"):
            continue
        entry = {"clause": str(c.get("clause", "")).strip(),
                 "reason": str(c.get("reason", "")).strip()}
        if c.get("relation") in ("owner", "not_owner"):
            entry["relation"] = c["relation"]
            entry["realizable"] = "fixture"
        dropped.append(entry)

    conditions: list[dict] = []
    if routed:
        result, errors = _encode(condition, routed, vocab, domain=domain,
                                 model=model, temperature=temperature,
                                 call_json=call_json)
        if errors:
            return None, [], dropped, [f"encode: {e}" for e in errors]
        encoded = [c for c in result["conditions"] if isinstance(c, dict)]
        conditions = [{k: v for k, v in c.items() if k != "clause_index"}
                      for c in encoded if "unencodable" not in c]
        for c in encoded:
            if "unencodable" not in c:
                continue
            index = c.get("clause_index")
            clause = (routed[index]["clause"]
                      if isinstance(index, int) and 0 <= index < len(routed) else "")
            dropped.append({"clause": clause,
                            "reason": str(c["unencodable"]).strip()})

    # Step 3: for every clause still without a condition (and not the
    # fixture's), who can establish it? A "none" or "field" verdict must name
    # the field it looked for; the name is checked against the vocabulary,
    # and a field that turns out to exist sends the clause back to Step 2.
    # The judgement is separated from segmentation on purpose: given a "none"
    # bucket while routing, the model reached for it instead of encoding
    # (042's cabin condition, 088's record existence both went missing).
    pending = [d for d in dropped if d.get("realizable") != "fixture"]
    warnings: list[str] = []
    if pending:
        block = "\n".join(f"{i}. {d['clause']}" for i, d in enumerate(pending))
        user3 = _CLASSIFY_USER.format(condition=condition,
                                      trigger=trigger or "(not provided)",
                                      then=then or "(not provided)",
                                      clauses_block=block,
                                      vocabulary=vocab.render())
        result3: dict = {}
        errors3: list[str] = []
        for attempt in range(2):
            result3 = call_json(_CLASSIFY_SYSTEM.format(domain=domain), user3,
                                model, temperature, max_tokens=1200,
                                label="ground_classify")
            errors3 = _check_verdicts(result3, len(pending))
            if not errors3:
                break
            if attempt == 0:
                user3 += ("\n\nRETRY: the previous answer failed validation:\n- "
                          + "\n- ".join(errors3) + "\nReturn a corrected JSON object.")
        verdicts = ({v["clause_index"]: v for v in result3.get("verdicts", [])
                     if isinstance(v, dict)} if not errors3 else {})
        if errors3:
            warnings.append("classify: " + "; ".join(errors3)
                            + " — every clause handed to the conversation")
        bounce: list[dict] = []
        for i, d in enumerate(pending):
            v = verdicts.get(i) or {}
            tag = v.get("realizable", "conversation")
            missing = str(v.get("missing") or "").strip()
            d["why"] = str(v.get("why") or "").strip()
            if tag == "conversation":
                d["realizable"] = "conversation"
                continue
            exists, note = _lookup_missing(missing, vocab)
            d["missing"] = missing
            d["check"] = note
            if exists:
                # The model says (or its own "missing" proves) the field is
                # there: re-encode rather than hand a storable fact to the
                # user or, worse, declare it untestable.
                d["realizable"] = "field"
                d["table"] = _MISSING_RE.match(missing).group(1)
                bounce.append(d)
                if tag == "none":
                    warnings.append(f"classify: {d['clause']!r} called none, but "
                                    f"{note} — re-encoded")
            elif tag == "field":
                d["realizable"] = "conversation"
                warnings.append(f"classify: {d['clause']!r} called a field, but "
                                f"{note} — handed to the conversation")
            else:
                d["realizable"] = "none"
                d["verified"] = True
        if bounce:
            routed2 = [{"clause": d["clause"], "table": d["table"]} for d in bounce]
            result2, errors2 = _encode(condition, routed2, vocab, domain=domain,
                                       model=model, temperature=temperature,
                                       call_json=call_json)
            encoded2 = ([c for c in result2["conditions"] if isinstance(c, dict)]
                        if not errors2 else [])
            # "count_eq 0" / "count_lt 1" from a bounce is the absence pattern
            # wearing a condition's clothes — always false, and exactly what
            # the record-absence rule keeps out of the database side.
            def _empty_set(c: dict) -> bool:
                op, val = c.get("op", ""), c.get("value")
                return (op == "count_eq" and val == 0) or (op == "count_lt" and val == 1) \
                    or (op == "count_le" and val == 0)
            got = {c["clause_index"] for c in encoded2
                   if "unencodable" not in c and not _empty_set(c)}
            encoded2 = [c for c in encoded2 if "unencodable" in c or not _empty_set(c)]
            for i, d in enumerate(bounce):
                if i in got:
                    dropped.remove(d)
                else:
                    d["realizable"] = "conversation"
                    warnings.append(f"classify: {d['clause']!r} was sent back to "
                                    "encoding and got no usable condition — "
                                    "handed to the conversation")
            conditions += [{k: v for k, v in c.items() if k != "clause_index"}
                           for c in encoded2 if "unencodable" not in c]
    for d in dropped:
        d.setdefault("realizable", "conversation")

    if conditions and root is None:
        root = conditions[0]["table"]
    return root, conditions, dropped, [f"warning: {w}" for w in warnings]
