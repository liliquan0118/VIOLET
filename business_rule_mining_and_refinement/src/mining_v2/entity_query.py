"""Rule-based entity lookup: from grounded conditions to (user, root record) pairs.

Downstream of grounding: a scenario's Given is a root table plus a flat list of
{table, path, op, value[, quant]} conditions. This module answers "which real
records can the test run against?" deterministically — no LLM:

  find_user_matches(db, vocab, root, conditions)
      -> [{"user_id": ..., "root_id": ...}, ...]

Evaluation reuses grounding.find_matching (conditions on the root table are
checked on the record itself; conditions naming another table are checked on
the records the root reaches by following foreign keys, with quant "all"
spanning the reached records). Every hit is then JOINED to the person who owns
it, because a τ²-bench test needs a user the simulator can identify as:

  root == users/customers      the record IS the person;
  root has a direct owner key  follow user_id / customer_id;
  otherwise                    walk the link graph BACKWARDS (a flight has no
                               owner field, but reservations reference flights
                               and reservations have owners), so a root like
                               `flights` yields one pair per booking user.
"""

from __future__ import annotations

from collections import defaultdict

from src.mining_v2.grounding import (
    Vocabulary,
    _dump,
    find_matching,
    infer_links,
    owner_field,
    resolve,
    tables_of,
)

_PERSON_TABLES = ("users", "customers")


def reverse_index(db, links: dict) -> dict[str, dict[str, set[str]]]:
    """{target_table: {target_id: {(src_table, src_id), ...}}} for every link.

    Built once per database; lets ownership walk against the arrow of the
    foreign keys (flights <- reservations, lines <- customers).
    """
    tables = tables_of(db)
    idx: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    for src_table, pairs in links.items():
        for path, target in pairs:
            for src_id, record in tables[src_table].items():
                for key in resolve(_dump(record), path):
                    idx[target][str(key)].add((src_table, src_id))
    return {t: dict(m) for t, m in idx.items()}


def owners_of(db, vocab: Vocabulary, links: dict, rev: dict,
              table: str, record_id: str, _seen: frozenset = frozenset()) -> set[str]:
    """User/customer ids that own one record, walking links either direction."""
    if table in _PERSON_TABLES:
        return {record_id}
    if (table, record_id) in _seen:
        return set()
    _seen = _seen | {(table, record_id)}
    tables = tables_of(db)
    record = (tables.get(table) or {}).get(record_id)
    if record is None:
        return set()
    out: set[str] = set()
    link = owner_field(vocab, table)
    if link:
        key, owner = link
        out |= {str(v) for v in resolve(_dump(record), key)
                if str(v) in (tables.get(owner) or {})}
    if not out:
        for src_table, src_id in rev.get(table, {}).get(str(record_id), ()):
            out |= owners_of(db, vocab, links, rev, src_table, src_id, _seen)
    return out


def find_user_matches(db, vocab: Vocabulary, root: str, conditions: list[dict],
                      *, links: dict | None = None, rev: dict | None = None,
                      ctx: dict | None = None, limit: int | None = None,
                      relation: str = "owner") -> list[dict]:
    """(user, root record) pairs whose root record satisfies every condition.

    `relation` decides who the user in the pair is:

      "owner"      the person the record belongs to — the ordinary case.
      "not_owner"  a person it does NOT belong to. The ownership rules ("must
                   not cancel a reservation that is not the requester's") can
                   only be set up this way: no field stores "the requester",
                   so the premise is a relation between the two halves of the
                   pair rather than a condition on either. Pairing a record
                   with its own owner, as "owner" does, makes the premise
                   false and the test vacuous.

    `limit` caps the number of PAIRS returned; matching itself is exhaustive.
    A root record reachable from several persons yields one pair per person;
    under "owner" a record with no reachable person is dropped (nothing to
    identify as), while under "not_owner" it is usable by anyone.
    """
    links = infer_links(db, vocab) if links is None else links
    rev = reverse_index(db, links) if rev is None else rev
    everyone = [p["user_id"] for p in all_persons(db)] if relation == "not_owner" else []
    pairs: list[dict] = []
    for root_id in find_matching(db, vocab, root, conditions, links=links, ctx=ctx):
        owners = owners_of(db, vocab, links, rev, root, root_id)
        if relation == "not_owner":
            candidates = [u for u in everyone if u not in owners]
        else:
            candidates = sorted(owners)
        for user_id in candidates:
            pairs.append({"user_id": user_id, "root_id": root_id})
            if limit and len(pairs) >= limit:
                return pairs
    return pairs


# ── zero-match triage: which states a tool can create before the test ────────
#
# Mechanism is domain-general; entries are per-domain data. A rule says: the
# condition "<table>.<path> == <target>" can be ESTABLISHED offline by calling
# <tool> on a record that satisfies the rule's own precondition (e.g. retail
# can only cancel a PENDING order). `owner_arg` names a tool argument that
# takes the record's owner id (telecom's suspend_line needs customer_id).

SETUP_RULES: dict[str, list[dict]] = {
    "airline": [dict(table="reservations", path="status", target="cancelled",
                     tool="cancel_reservation", root_arg="reservation_id",
                     extra_args={}, owner_arg=None, pre_target=None)],
    "retail": [dict(table="orders", path="status", target="cancelled",
                    tool="cancel_pending_order", root_arg="order_id",
                    extra_args={"reason": "no longer needed"},
                    owner_arg=None, pre_target="pending")],
    "telecom": [dict(table="lines", path="status", target="suspended",
                     tool="suspend_line", root_arg="line_id",
                     extra_args={"reason": "user requested suspension"},
                     owner_arg="customer_id", pre_target="active")],
}


def _enum_value(vocab: Vocabulary, table: str, path: str, word: str):
    """The declared enum value spelled the way the database spells it."""
    f = vocab.field(table, path)
    for v in (f.values if f else []):
        if str(v).lower() == str(word).lower():
            return v
    return word


def _rule_for(cond: dict, rules: list[dict], vocab: Vocabulary):
    if cond.get("negate") or cond.get("op") not in ("eq", "in"):
        return None
    values = cond.get("value") if isinstance(cond.get("value"), list) else [cond.get("value")]
    for r in rules:
        target = _enum_value(vocab, r["table"], r["path"], r["target"])
        if (cond.get("table") == r["table"] and cond.get("path") == r["path"]
                and any(str(v) == str(target) for v in values)):
            return r
    return None


def all_persons(db) -> list[dict]:
    """Every user/customer id, as bare candidate rows (no root record)."""
    tables = tables_of(db)
    for name in _PERSON_TABLES:
        if name in tables:
            return [{"user_id": str(u), "root_id": None} for u in tables[name]]
    return []


def _patchable_cond(cond: dict, vocab: Vocabulary) -> bool:
    """True when a fixture patch can legally ESTABLISH this condition: an
    eq/in on literal value(s) that are all members of the field's DECLARED
    domain. The declaration is what legitimises the state (a `BillStatus`
    lists "Awaiting Payment" even when no stored bill is in it — the db
    instance being sparse does not make the state illegal); date arithmetic
    ($NOW/$TODAY refs) and free-form fields stay out."""
    if cond.get("op") not in ("eq", "in") or "value" not in cond:
        return False
    values = cond["value"] if isinstance(cond["value"], list) else [cond["value"]]
    if not values or any(isinstance(v, dict) for v in values):
        return False
    field = vocab.field(cond.get("table", ""),
                        str(cond.get("path", "")).replace("[0]", "[]"))
    domain = [str(v) for v in (getattr(field, "values", None) or [])]
    return bool(domain) and all(str(v) in domain for v in values)


def find_or_make(db, vocab: Vocabulary, domain: str, root: str,
                 conditions: list[dict], *, links: dict | None = None,
                 rev: dict | None = None, ctx: dict | None = None,
                 limit: int | None = None, relation: str = "owner") -> dict:
    """Lookup with the simplified zero-match policy.

    matched     -> (user, root record) pairs whose record satisfies the
                   conditions
    anchored    -> no conditions at all; the pairs are arbitrary records of
                   the root table, there only so the user can name one
    makeable    -> zero matches, but a setup rule covers a blocking condition
                   on the root table: the conditions are passed through
                   unchanged for a later stage to establish the state
    patchable   -> zero matches, but every blocking condition that fails is
                   an eq/in on DECLARED domain values: a fixture patch can
                   legally set them. `matches` are the (user, record) pairs
                   satisfying the REMAINING conditions — the records to
                   patch — and `patch` lists the conditions to establish
    unmakeable  -> zero matches and no setup rule or patch applies: every
                   user is returned as a candidate identity instead
    """
    links = infer_links(db, vocab) if links is None else links
    rev = reverse_index(db, links) if rev is None else rev
    pairs = find_user_matches(db, vocab, root, conditions, links=links,
                              rev=rev, ctx=ctx, limit=limit, relation=relation)
    if pairs:
        # With no condition every record of the root table "matches": the
        # pair is an ANCHOR (a concrete record for the user to refer to), not
        # a record selected for satisfying anything. Say so, or the count of
        # satisfied premises is inflated by every Given-True scenario.
        status = "matched" if conditions else "anchored"
        return {"lookup_status": status, "matches": pairs, "relation": relation}
    rules = SETUP_RULES.get(domain, [])
    if any(_rule_for(c, rules, vocab) and _rule_for(c, rules, vocab)["table"] == root
           for c in conditions):
        return {"lookup_status": "makeable", "matches": [],
                "DBconditions": conditions}
    patch = [c for c in conditions if _patchable_cond(c, vocab)]
    if patch:
        relaxed = [c for c in conditions if c not in patch]
        relaxed_pairs = find_user_matches(db, vocab, root, relaxed,
                                          links=links, rev=rev, ctx=ctx,
                                          limit=limit, relation=relation)
        if relaxed_pairs:
            return {"lookup_status": "patchable", "matches": relaxed_pairs,
                    "patch": patch, "relation": relation}
    return {"lookup_status": "unmakeable", "matches": all_persons(db)}
