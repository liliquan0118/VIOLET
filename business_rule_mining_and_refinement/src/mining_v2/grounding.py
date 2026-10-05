"""Ground a scenario's Given onto the domain database.

A mined Given is natural language ("a reservation in economy cabin with a
flight that has already been flown"). Everything downstream — planting the
fixture, checking it took, deriving the violating state — needs that sentence
as something executable. Two earlier answers were unsatisfying: v1's
ConditionMatcher took structured conditions but resolved a dotted path by its
last segment only, so it could not reach inside a list; testgen's
`_find_candidates` has an LLM write free Python that is `eval`-ed, which is
expressive but non-deterministic, unverifiable, and cannot be negated.

This module supplies the missing layer between them:

  build_vocabulary(db)      every field path the domain actually has, with its
                            type and declared value domain — read off the
                            pydantic models, so it is exact and costs nothing.
  validate(pred, vocab)     a predicate may only mention paths that exist; an
                            unknown path is reported here rather than surfacing
                            later as a fixture that cannot be planted.
  evaluate(pred, record)    deterministic execution, no eval, no LLM.
  find_matching / assignments
                            select real records, or read the predicate as the
                            patch that would make a record satisfy it.

The predicate is a small closed grammar rather than code, which is what makes
validation and re-evaluation possible:

    [{"table": "reservations", "path": "cabin", "op": "eq",
      "value": "economy"},
     {"table": "reservations", "path": "flights[].date", "op": "lt",
      "value": "$TODAY"}]

PATH SEMANTICS. A path resolves to a LIST of values: "cabin" yields one,
"flights[].date" yields one per flight segment, "flights[0].date" (or any
"[i]" index) exactly the i-th element, an absent field yields none.
`quant` picks how many of them the comparison must hold of: "any" (the
default) — at least one, so "flights[].date < $TODAY" reads "some flight has
already been flown"; "all" — every value, and at least one must exist, so the
universal "no flight has flown" is `quant "all", op "ge"`. Under "any" the
order comparisons are NOT each other's opposites — a booking with one past
and one future segment satisfies both `lt` and `ge` — which is why universals
get an explicit quantifier instead of a swapped operator.
"""

from __future__ import annotations

from enum import Enum

import calendar
import re
import typing
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, get_args, get_origin


# ── L0: state vocabulary ─────────────────────────────────────────────────────

@dataclass
class Field:
    """One addressable field of a table.

    `values` is the DECLARED domain, read off a Literal annotation — it is
    authoritative: the API rejects anything outside it. `examples` are values
    observed in the stored records, kept strictly separate because data shows
    what happens to be there, not what is allowed.
    """
    path: str                                        # "flights[].date"
    type: str                                        # str | int | bool | enum | …
    values: list = field(default_factory=list)       # declared domain
    examples: list = field(default_factory=list)     # observed samples


@dataclass
class Table:
    name: str
    record_type: str
    n_records: int
    fields: list[Field] = field(default_factory=list)

    @property
    def paths(self) -> set[str]:
        return {f.path for f in self.fields}


@dataclass
class Vocabulary:
    """Every table, field, type and value domain of the agent's database."""
    domain: str
    tables: dict[str, Table] = field(default_factory=dict)

    def paths(self, table: str) -> set[str]:
        entry = self.tables.get(table)
        return entry.paths if entry else set()

    def field(self, table: str, path: str) -> Field | None:
        for f in (self.tables.get(table).fields if table in self.tables else []):
            if f.path == path:
                return f
        return None

    def render(self, tables: list[str] | None = None) -> str:
        """The prompt form: the paths a predicate is allowed to reference."""
        lines: list[str] = []
        for name, entry in self.tables.items():
            if tables and name not in tables:
                continue
            lines.append(f"table `{name}` ({entry.n_records} records):")
            for f in entry.fields:
                if f.values:
                    domain = f" ∈{f.values}"
                elif f.examples:
                    domain = f" e.g. {f.examples}"
                else:
                    domain = ""
                lines.append(f"    {f.path} ({f.type}{domain})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "tables": {
                name: {
                    "record_type": t.record_type,
                    "n_records": t.n_records,
                    "fields": [
                        {"path": f.path, "type": f.type,
                         "values": f.values, "examples": f.examples}
                        for f in t.fields
                    ],
                }
                for name, t in self.tables.items()
            },
        }


def _members(ann) -> list:
    """The concrete types an annotation can hold, with Optional/Union flattened.

    Always a list, so the walker has ONE shape to handle: a plain field yields
    one member, a discriminated union (a flight date that is available /
    delayed / landed) yields one per member and every member's fields are
    collected.
    """
    origin = get_origin(ann)
    if origin is typing.Union or str(origin) == "types.UnionType":
        return [a for a in get_args(ann) if a is not type(None)] or [ann]
    return [ann]


def _scalar(ann) -> tuple[str, list]:
    """A leaf annotation -> (type name, declared value domain).

    The domain comes from a Literal annotation OR an Enum class (possibly
    wrapped in Optional). telecom declares its statuses as Enums; with only
    Literal read, those fields had an empty value space, so "not in the value
    space" could never be checked there and the encoder saw two observed
    examples instead of the seven members.
    """
    if "Literal" in str(ann):
        return "enum", list(get_args(ann))
    candidates = [ann, *get_args(ann)]
    for cand in candidates:
        if isinstance(cand, type) and issubclass(cand, Enum):
            return cand.__name__, [m.value for m in cand]
    return getattr(ann, "__name__", str(ann)), []


def _fields_of(model, prefix: str, depth: int, max_depth: int,
               seen: frozenset) -> list[Field]:
    """Every field path reachable from a record model, depth-first."""
    declared = getattr(model, "model_fields", None)
    if not declared or depth > max_depth or model in seen:
        return []
    seen = seen | {model}
    out: list[Field] = []

    def descend(ann, path: str) -> None:
        for member in _members(ann):
            origin, args = get_origin(member), get_args(member)
            if origin is list and args:                     # list[X] -> "path[]"
                descend(args[0], f"{path}[]")
            elif origin is dict and len(args) == 2:         # dict[K,V] -> "path{}"
                descend(args[1], f"{path}{{}}")
            elif hasattr(member, "model_fields"):           # nested record
                out.extend(_fields_of(member, path, depth + 1, max_depth, seen))
            else:
                kind, values = _scalar(member)
                out.append(Field(path, kind, values))

    for name, meta in declared.items():
        descend(meta.annotation, f"{prefix}.{name}" if prefix else name)
    return out


def _merge(fields: list[Field]) -> list[Field]:
    """Collapse repeats of one path, UNIONing declared domains.

    Union members each pin the shared discriminator to their own literal
    (status: "available" / "delayed" / …); keeping only the first would hide
    every other state the field can be in.
    """
    merged: dict[str, Field] = {}
    for f in fields:
        prev = merged.get(f.path)
        if prev is None:
            merged[f.path] = Field(f.path, f.type, list(f.values))
            continue
        for value in f.values:
            if value not in prev.values:
                prev.values.append(value)
        if prev.type != f.type:
            prev.type = "mixed"
    return list(merged.values())


def tables_of(db) -> dict[str, dict]:
    """The domain's record collections, keyed by table name.

    Domains disagree on the shape: airline and retail hold `{id: record}`,
    telecom holds plain lists. Lists are keyed by whichever field looks like
    the primary id so the rest of the module can assume a mapping.
    """
    tables: dict[str, dict] = {}
    for name in dir(db):
        if name.startswith("_") or name.startswith("model_"):
            continue
        value = getattr(db, name, None)
        if isinstance(value, dict) and value:
            if hasattr(next(iter(value.values())), "model_dump"):
                tables[name] = value
        elif isinstance(value, list) and value and hasattr(value[0], "model_dump"):
            singular = name[:-1] if name.endswith("s") else name
            key = next((k for k in (f"{singular}_id", "id")
                        if k in getattr(type(value[0]), "model_fields", {})), None)
            tables[name] = {
                (str(getattr(r, key)) if key else str(i)): r
                for i, r in enumerate(value)
            }
    return tables


def _add_examples(fields: list[Field], records: list, max_values: int) -> None:
    """Attach observed values to the fields that declare no domain.

    Purely illustrative — it tells a reader (and the compile prompt) what an
    id or a date looks like. A field WITH a declared domain gets none: the
    declaration is already exact, and samples would only invite mistaking
    "what the data happens to contain" for "what is allowed".
    """
    for f in fields:
        if f.values:
            continue
        seen: list = []
        for record in records:
            for value in resolve(record, f.path):
                if isinstance(value, (str, int, float, bool)) and value not in seen:
                    seen.append(value)
                    if len(seen) >= max_values:
                        break
            if len(seen) >= max_values:
                break
        f.examples = seen

# L0 entry point
def build_vocabulary(db, *, domain: str = "", max_depth: int = 4,
                     sample: int = 60, max_examples: int = 3) -> Vocabulary:
    """The agent's database as tables -> fields -> type + value domain.

    INPUT   the domain's DB object (the pydantic container tau2 hands to the
            tools; `load_domain_db(root, domain)` returns it).
    OUTPUT  a Vocabulary: for every table, its record type, size, and every
            addressable field path with its type, declared domain and — where
            nothing is declared — a few observed example values.

    Paths come from the MODELS, so the result is exact and data-independent:
    an empty table still lists its full field set. Only `examples` reads
    records, and only as illustration.
    """
    vocab = Vocabulary(domain=domain)
    for name, table in tables_of(db).items():
        record_type = type(next(iter(table.values())))
        fields = _merge(_fields_of(record_type, "", 0, max_depth, frozenset()))
        # Slice BEFORE dumping: only `sample` records are read, and dumping a
        # whole table to look at 60 of them is pure waste.
        _add_examples(fields,
                      [_dump(r) for _, r in zip(range(sample), table.values())],
                      max_examples)
        vocab.tables[name] = Table(name=name, record_type=record_type.__name__,
                                   n_records=len(table), fields=fields)
    return vocab


# ── conditions ───────────────────────────────────────────────────────────────
# A Given is a CONJUNCTION of atoms. The GWT stage has already split every
# disjunction into its own scenario, so by the time a precondition reaches
# here there is nothing left to nest: measured over 106 real Givens, not one
# needed an OR across two fields, the few same-field ones are covered by `in`,
# and every universal ("no flight has flown") is a flat `quant: "all"` on one
# path — a flag on the atom, not a nested quantifier with its own scope.
#
# So a condition is FLAT — table, path, op, value — and a precondition is a
# list of them, ANDed. That is not only enough, it is what removed the whole
# class of bugs this module kept hitting: a nested predicate has an inner and
# an outer frame of reference, and paths from one kept being written into the
# other (a reservation's `cabin` inside a per-flight quantifier, which resolves
# to nothing, so the test passed vacuously). With no nesting, that cannot be
# expressed at all.

_COMPARISONS = {"eq", "ne", "lt", "le", "gt", "ge", "in", "contains"}
_COUNTS = {"count_eq", "count_gt", "count_lt", "count_ge", "count_le"}
OPS = _COMPARISONS | _COUNTS | {"exists", "missing"}


_PLACEHOLDER_RE = re.compile(r"^<.+>$|^\{\{.+\}\}$|^\$[A-Z_]{2,}$")
_REFS = ("$NOW", "$TODAY")
_OFFSET_UNITS = ("minutes", "hours", "days", "weeks", "months", "years")


def _ref_errors(value: Any) -> list[str]:
    """A symbolic value stands for the domain clock and nothing else.

    Anything else written as {"ref": ...} used to resolve to the current
    timestamp, so `user_id ne {"ref": "$REQUESTING_USER_ID"}` silently became
    a tautology and the entity query returned the records the Given excludes.
    A value that is neither a literal nor the clock — another record's field,
    something the user will only say during the conversation — is not
    expressible, and the clause has to be reported unencodable instead.
    """
    # A placeholder is a placeholder inside a list too: `in ["<line_id>"]`
    # slipped past the string check and matched nothing, silently.
    elements = value if isinstance(value, list) else [value]
    for element in elements:
        if isinstance(element, str) and _PLACEHOLDER_RE.match(element.strip()):
            return [f"value {element!r} is a placeholder, not a literal: a value "
                    "the conversation supplies cannot be compared against here"]
    if not isinstance(value, dict) or "ref" not in value:
        return []
    ref = value.get("ref")
    if ref not in _REFS:
        return [f"unknown symbolic value {ref!r}: the only symbols are "
                + " and ".join(_REFS) + " (the domain clock)"]
    offset = value.get("offset")
    if offset is None:
        return []
    if not isinstance(offset, dict):
        return ["offset must be an object of unit -> amount"]
    unknown = sorted(k for k in offset if k not in _OFFSET_UNITS)
    if unknown:
        return [f"unknown offset unit(s) {unknown}: use "
                + ", ".join(_OFFSET_UNITS)]
    return []


def validate_condition(cond: Any, vocab: Vocabulary) -> list[str]:
    """Check one atom: known table, known path, known op, value where needed."""
    errors: list[str] = []
    if not isinstance(cond, dict):
        return ["condition is not an object"]
    table = cond.get("table")
    if table not in vocab.tables:
        return [f"table {table!r} is not a table of this database"]
    path, op = cond.get("path"), cond.get("op")
    if op not in OPS:
        errors.append(f"unknown op {op!r}")
    quant = cond.get("quant", "any")
    if quant not in ("any", "all"):
        errors.append(f'unknown quant {quant!r} (use "any" or "all")')
    elif quant == "all" and op in OPS and op not in _COMPARISONS:
        errors.append(f'quant "all" only applies to comparison ops, not {op!r}')
    if not isinstance(path, str) or not path:
        errors.append("path is missing")
        return errors

    known = vocab.paths(table)
    # An index marker ("[0]", "[-1]") addresses one element of a list the
    # vocabulary lists as "[]" — normalize before checking existence.
    normalized = re.sub(r"\[-?\d+\]", "[]", path)
    # Counting and existence address the COLLECTION ("flights[]"), which is a
    # prefix of the leaf paths rather than one of them.
    collection_ok = (op in _COUNTS | {"exists", "missing"}
                     and any(k.startswith(normalized + ".") for k in known))
    if normalized not in known and not collection_ok:
        near = [k for k in known if k.split(".")[-1] == path.split(".")[-1]]
        hint = f" (did you mean {near[0]!r}?)" if near else ""
        errors.append(f"path {path!r} is not a field of `{table}`{hint}")
    if op in _COUNTS and "[" not in path and "{" not in path:
        errors.append(f"op {op!r} counts the values a path reaches, but "
                      f"`{table}.{path}` is single-valued (no [] or {{}} marker) "
                      "— it always reaches exactly one. A claim about how many "
                      "RECORDS share a value is a cross-record condition the "
                      "language cannot express; report it unencodable")
    if op in _COMPARISONS | _COUNTS and "value" not in cond:
        errors.append(f"value is missing for op {op!r}")
    elif op == "in" and not (isinstance(cond.get("value"), list)
                             and cond["value"]):
        errors.append('op "in" requires a non-empty list value')
    errors.extend(_ref_errors(cond.get("value")))
    errors.extend(_domain_errors(cond, vocab, table, normalized))
    return errors


def _domain_errors(cond: dict, vocab: Vocabulary, table: str, path: str) -> list[str]:
    """An eq/ne/in value must be a member of the field's declared domain.

    Without this an encoder could write `status eq "AWAITING PAYMENT"` for a
    field whose members are spelled "Awaiting Payment": valid path, valid op,
    and a condition that can never match. The comparison is exact, so the
    spelling has to be too.
    """
    if cond.get("op") not in ("eq", "ne", "in") or "value" not in cond:
        return []
    field = vocab.field(table, path)
    domain = list(getattr(field, "values", None) or [])
    if not domain:
        return []
    values = cond["value"] if isinstance(cond["value"], list) else [cond["value"]]
    allowed = {str(v) for v in domain}
    bad = [v for v in values if isinstance(v, (str, int, float)) and str(v) not in allowed]
    if not bad:
        return []
    return [f"value(s) {bad} not in the declared domain of `{table}.{path}` "
            f"{domain} — copy a member exactly, or report the clause unencodable"]


def validate(conditions: list, vocab: Vocabulary) -> list[str]:
    """Check a whole precondition, reporting the index of each bad atom."""
    if not isinstance(conditions, list):
        return ["conditions must be a list"]
    out: list[str] = []
    for i, cond in enumerate(conditions):
        out.extend(f"conditions[{i}]: {e}"
                   for e in validate_condition(cond, vocab))
    return out


# ── evaluation ───────────────────────────────────────────────────────────────

_SEGMENT = re.compile(r"([^.\[\{]+)(\[\]|\{\}|\[-?\d+\])?")


def resolve(record: Any, path: str) -> list:
    """Resolve a path to the LIST of values it reaches (0, 1 or many).

    "[]" expands every element, "{}" every value; an index marker ("[0]",
    "[-1]") picks one element of an ordered list — out-of-range indexes
    resolve to nothing, like any other absent field.
    """
    values: list = [record]
    for name, marker in _SEGMENT.findall(path):
        nxt: list = []
        for value in values:
            if isinstance(value, dict):
                item = value.get(name)
            else:
                item = getattr(value, name, None)
            if item is None:
                continue
            if marker == "[]" and isinstance(item, list):
                nxt.extend(item)
            elif marker == "{}" and isinstance(item, dict):
                nxt.extend(item.values())
            elif marker and marker != "[]" and marker != "{}":
                if isinstance(item, list):
                    index = int(marker[1:-1])
                    if -len(item) <= index < len(item):
                        nxt.append(item[index])
            else:
                nxt.append(item)
        values = nxt
        if not values:
            return []
    return values


def _as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _today(ctx: dict) -> str:
    return str(ctx.get("today") or date.today().isoformat())


_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ T]?(\d{2}:\d{2}(?::\d{2})?)?")


def _resolve_ref(value: dict, ctx: dict):
    """Resolve a symbolic time value {"ref": "$NOW"|"$TODAY", "offset": {...}}
    against the domain clock, to an ISO string that compares lexicographically
    with the stored dates ("YYYY-MM-DD") / timestamps ("...THH:MM:SS").

    $TODAY with a whole-day offset stays a date; everything else becomes a
    full timestamp. An unparseable clock falls back to the raw base string;
    an unknown ref raises rather than quietly standing in for the clock.
    """
    ref = str(value.get("ref"))
    if ref not in _REFS:
        raise ValueError(
            f"unknown symbolic value {ref!r}; the only symbols are "
            + " and ".join(_REFS))
    base = (_today(ctx) if ref == "$TODAY"
            else str(ctx.get("now") or datetime.now().isoformat()))
    match = _TIMESTAMP_RE.search(base)
    if not match:
        return base
    time_part = match.group(2) or "00:00:00"
    if len(time_part) == 5:
        time_part += ":00"
    moment = datetime.fromisoformat(f"{match.group(1)}T{time_part}")
    offset = value.get("offset") or {}
    months = int(offset.get("months") or 0) + 12 * int(offset.get("years") or 0)
    if months:
        total = moment.month - 1 + months
        year, month = moment.year + total // 12, total % 12 + 1
        day = min(moment.day, calendar.monthrange(year, month)[1])
        moment = moment.replace(year=year, month=month, day=day)
    moment += timedelta(minutes=int(offset.get("minutes") or 0),
                        hours=int(offset.get("hours") or 0),
                        days=int(offset.get("days") or 0),
                        weeks=int(offset.get("weeks") or 0))
    date_only = (ref == "$TODAY" and not match.group(2)
                 and not offset.get("minutes") and not offset.get("hours"))
    return moment.date().isoformat() if date_only else moment.isoformat()


def _literal(value, ctx: dict):
    """Resolve the symbolic time values a temporal condition needs."""
    if isinstance(value, dict) and "ref" in value:
        return _resolve_ref(value, ctx)
    if isinstance(value, str):
        if value == "$TODAY":
            return _today(ctx)
        if value == "$NOW":
            return str(ctx.get("now") or datetime.now().isoformat())
    return value


def _compare(op: str, left, right) -> bool:
    if op == "eq":
        return str(left) == str(right)
    if op == "ne":
        return str(left) != str(right)
    if op == "in":
        return str(left) in [str(x) for x in (right if isinstance(right, list)
                                              else [right])]
    if op == "contains":
        return str(right) in str(left)
    a, b = _as_number(left), _as_number(right)
    if a is None or b is None:
        # ISO date/time strings order correctly as text, which is what the
        # temporal conditions rely on.
        a, b = str(left), str(right)
    if op == "lt":
        return a < b
    if op == "le":
        return a <= b
    if op == "gt":
        return a > b
    if op == "ge":
        return a >= b
    return False


def evaluate(cond: dict, record: Any, ctx: dict | None = None) -> bool:
    """Does one atom hold of this record? Pure Python, no eval, no recursion.

    A path resolves to every value it reaches; `quant` picks the reading.
    "any" (the default) holds when SOME value satisfies the comparison —
    "flights[].date < today" is "some segment already flew". "all" holds when
    EVERY value satisfies it and at least one exists — the universal reading a
    fixture can actually exhibit. `ne` is universal on its own: no value
    equals the target, which stays true when the field is absent (a
    reservation with no status is certainly not cancelled).
    """
    ctx = ctx or {}
    if not isinstance(cond, dict):
        return False
    op = cond.get("op")
    values = resolve(record, cond.get("path") or "")

    if op == "exists":
        result = bool(values)
    elif op == "missing":
        result = not values
    elif op in _COUNTS:
        target = _as_number(_literal(cond.get("value"), ctx))
        n = len(values)
        result = False if target is None else {
            "count_eq": n == target, "count_gt": n > target,
            "count_lt": n < target, "count_ge": n >= target,
            "count_le": n <= target}[op]
    elif op in _COMPARISONS:
        target = _literal(cond.get("value"), ctx)
        if cond.get("quant") == "all":
            result = bool(values) and all(
                _compare(op, v, target) for v in values)
        elif op == "ne":
            result = not any(_compare("eq", v, target) for v in values)
        else:
            result = any(_compare(op, v, target) for v in values)
    else:
        return False
    return not result if cond.get("negate") else result


def holds(conditions: list[dict], record: Any,
          ctx: dict | None = None) -> bool:
    """Do ALL atoms hold of this record? (A precondition is a conjunction.)"""
    return all(evaluate(c, record, ctx) for c in conditions)


# ── selection and synthesis ──────────────────────────────────────────────────

def _dump(record) -> dict:
    if hasattr(record, "model_dump"):
        try:
            return record.model_dump(mode="json")
        except TypeError:
            return record.model_dump()
    return dict(record)


def infer_links(db, vocab: Vocabulary, *, sample: int = 200,
                threshold: float = 0.9) -> dict[str, list[tuple[str, str]]]:
    """{table: [(path, other_table), ...]} — the foreign keys, found by data.

    A string field whose values are keys of another table IS a reference to
    it, whatever it is named. Deriving this from the records rather than from
    naming conventions is what lets a condition be stated per table and linked
    up afterwards: the LLM says "the reservation is business class AND its
    user is a gold member", and the traversal that connects the two is a
    structural fact the code already knows.
    """
    tables = tables_of(db)
    keysets = {name: set(t) for name, t in tables.items()}
    links: dict[str, list[tuple[str, str]]] = {}
    for name, table in tables.items():
        records = [_dump(r) for _, r in zip(range(sample), table.values())]
        for f in vocab.tables[name].fields:
            if f.type != "str":
                continue
            values = [v for r in records for v in resolve(r, f.path)][:sample]
            if not values:
                continue
            for other, keys in keysets.items():
                if other == name:
                    continue
                hits = sum(1 for v in values if str(v) in keys)
                if hits / len(values) >= threshold:
                    links.setdefault(name, []).append((f.path, other))
    return links


def _key_pairs(dumped: dict, path: str) -> list[tuple[Any, set]]:
    """(foreign key value, the sibling values recorded beside it).

    When the key sits inside an array element — a booking's
    "flights[].flight_number" — the element's OTHER fields say WHICH part of
    the referenced record is meant. The segment's date is the one that
    matters: the flights table stores a status per calendar day, so without
    carrying that date over, "the flight is cancelled" would ask whether the
    aircraft was ever cancelled on ANY day of the month.
    """
    if "[]" not in path:
        return [(value, set()) for value in resolve(dumped, path)]
    head, tail = path.split("[]", 1)
    tail = tail.lstrip(".")
    pairs: list[tuple[Any, set]] = []
    for element in resolve(dumped, head + "[]"):
        if not isinstance(element, dict):
            # A scalar-element list ("users.reservations[]", telecom's
            # "line_ids[]") IS the key list: each element is the foreign key.
            if not tail:
                pairs.append((element, set()))
            continue
        siblings = {str(v) for k, v in element.items()
                    if k != tail and isinstance(v, (str, int, float))}
        for key in resolve(element, tail):
            pairs.append((key, siblings))
    return pairs


def _narrow(dump: dict, correlation: set) -> dict:
    """Keep only the entries of a keyed collection that the link points at.

    A flight's "dates" is keyed by calendar day; the booking said which day,
    so the rest of the month is not what the condition is about. Collections
    whose keys match nothing carried over are left whole — the correlation is
    a restriction where one applies, not a filter that empties the record.
    """
    if not correlation:
        return dump
    out = dict(dump)
    for name, value in dump.items():
        if isinstance(value, dict) and value:
            hits = {k: v for k, v in value.items() if str(k) in correlation}
            if hits:
                out[name] = hits
    return out


def _record_id(dumped: dict, table: str, tables: dict) -> str | None:
    """The id a dumped record is stored under in its table (by identity of
    the common id fields, falling back to a scan)."""
    for name in (f"{table[:-1]}_id", "id", f"{table}_id"):
        if name in dumped and dumped[name] is not None:
            return str(dumped[name])
    return None


def reverse_links(db, links: dict) -> dict:
    """{table: {record_id: {(src_table, src_id), ...}}} — who points AT a record.

    infer_links only sees keys stored on a record, so a table that holds no
    key to its owner (telecom's lines: customers.line_ids[] points at them,
    nothing points back) was unreachable from its own root. related_records
    walks these edges backwards so a condition on the owner still resolves.
    """
    tables = tables_of(db)
    rev: dict = {}
    for src_table, pairs in links.items():
        for src_id, record in (tables.get(src_table) or {}).items():
            dumped = _dump(record)
            for path, other in pairs:
                for key, _ in _key_pairs(dumped, path):
                    rev.setdefault(other, {}).setdefault(str(key), set()).add(
                        (src_table, str(src_id)))
    return rev


def related_records(db, links: dict, root: str, record_id: str,
                    target: str, *, max_hops: int = 3, rev: dict | None = None) -> list:
    """Records of `target` reachable from one root record by following keys.

    Breadth-first over the link graph, so a condition may sit two tables away
    (a telecom bill's customer's lines) without the caller spelling out the
    route. Each hop carries the correlation values from the row that held the
    key, so the record that comes back is already narrowed to the part the
    link designates — the booked DAY of a flight, not its whole schedule.
    """
    tables = tables_of(db)
    record = (tables.get(root) or {}).get(record_id)
    if record is None:
        return []
    frontier = [(root, _dump(record))]
    seen = {(root, record_id)}
    for _ in range(max_hops):
        if not frontier:
            break
        nxt = []
        for table, dumped in frontier:
            for path, other in links.get(table, []):
                for key, correlation in _key_pairs(dumped, path):
                    ident = (other, str(key))
                    if ident in seen:
                        continue
                    found = (tables.get(other) or {}).get(str(key))
                    if found is None:
                        continue
                    seen.add(ident)
                    nxt.append((other, _narrow(_dump(found), correlation)))
            # Backwards: records that hold a key to this one. There is no
            # correlation to narrow by from this side, so the whole record
            # comes back.
            this_id = _record_id(dumped, table, tables)
            if rev is not None and this_id is not None:
                for src_table, src_id in rev.get(table, {}).get(this_id, ()):
                    ident = (src_table, src_id)
                    if ident in seen:
                        continue
                    found = (tables.get(src_table) or {}).get(src_id)
                    if found is None:
                        continue
                    seen.add(ident)
                    nxt.append((src_table, _dump(found)))
        hits = [d for t, d in nxt if t == target]
        if hits:
            return hits
        frontier = nxt
    return []


_OWNER_TABLES = ("users", "customers")


def owner_field(vocab: Vocabulary, table: str) -> tuple[str, str] | None:
    """The (foreign key, table) that leads from `table` to its owning person.

    Every fixture has to name a user: the simulated customer must identify
    with a real id, or it invents a placeholder the agent cannot look up. So
    whatever entity a predicate is rooted at, the selection has to be able to
    answer "whose is it?".
    """
    if table in _OWNER_TABLES:
        return None                      # already the person
    for key, owner in (("user_id", "users"), ("customer_id", "customers")):
        if owner in vocab.tables and key in vocab.paths(table):
            return key, owner
    return None


def resolve_owner(db, vocab: Vocabulary, table: str,
                  record_id: str) -> str | None:
    """The id of the user this record belongs to (the record itself if it is
    already a user; None when the entity has no owner, e.g. a flight)."""
    if table in _OWNER_TABLES:
        return record_id
    link = owner_field(vocab, table)
    if not link:
        return None
    key, owner = link
    record = (tables_of(db).get(table) or {}).get(record_id)
    if record is None:
        return None
    values = resolve(_dump(record), key)
    for value in values:
        if str(value) in (tables_of(db).get(owner) or {}):
            return str(value)
    return None


def find_matching(db, vocab: Vocabulary, root: str, conditions: list[dict], *,
                  links: dict | None = None, ctx: dict | None = None,
                  limit: int | None = None) -> list[str]:
    """Ids of the root records satisfying every condition.

    Conditions on the root table are checked against the record itself; a
    condition naming another table constrains the records THIS root reaches —
    "the user is a gold member" means the user of this booking, not any gold
    user in the database. Selecting each table independently would pair a
    business-cabin booking with an unrelated gold member and quietly plant the
    wrong fixture: only a third of business bookings belong to gold members.
    """
    ctx = {**(ctx or {}), "tables": tables_of(db)}
    links = infer_links(db, vocab) if links is None else links
    rev = reverse_links(db, links)
    own = [c for c in conditions if c.get("table") == root]
    other = [c for c in conditions if c.get("table") != root]

    out: list[str] = []
    for record_id, record in (tables_of(db).get(root) or {}).items():
        if not holds(own, _dump(record), ctx):
            continue
        ok = True
        for cond in other:
            linked = related_records(db, links, root, record_id, cond["table"], rev=rev)
            # Nothing reachable cannot satisfy a condition about it. quant
            # "all" spans the reached records too: "none of the flights has
            # flown" must hold of EVERY flight the booking reaches — with the
            # existential reading a single unflown segment would satisfy it.
            if not linked:
                ok = False
            elif cond.get("quant") == "all":
                ok = all(evaluate(cond, r, ctx) for r in linked)
            else:
                ok = any(evaluate(cond, r, ctx) for r in linked)
            if not ok:
                break
        if ok:
            out.append(record_id)
            if limit and len(out) >= limit:
                break
    return out


def assignments(conditions: list[dict],
                ctx: dict | None = None) -> list[tuple[str, str, Any]]:
    """Read the conditions as the patch that would satisfy them.

    Only plain equalities on scalar paths name a single value; ranges, counts
    and negations do not, and are left to the caller.
    """
    ctx = ctx or {}
    out: list[tuple[str, str, Any]] = []
    for c in conditions:
        path = c.get("path") or ""
        if (c.get("op") == "eq" and not c.get("negate")
                and "[]" not in path and "{}" not in path):
            out.append((c.get("table"), path, _literal(c.get("value"), ctx)))
    return out


# ── L1: compile a Given into a predicate (the one LLM step) ──────────────────

_COMPILE_SYSTEM = """\
You turn ONE precondition of a {domain} agent test into database conditions.

The precondition is the GIVEN of a Given-When-Then scenario: the state the
database must already be in when the test conversation starts. Output a root
table plus a flat list of ATOMIC conditions, ANDed: each condition is one
{{table, path, op, value}} object, and each is checked on its own.

Method: split GIVEN into clauses; for each clause find the DATABASE field
that stores that fact and write one condition carrying the clause's FULL
meaning — its comparison and its quantifier (the table follows from the
field). A clause that no listed field stores is left out.

WHEN and THEN are shown so you can read GIVEN correctly (what "it" refers
to, which sense of an ambiguous phrase is meant), and WHEN names the entity
the root comes from. Neither adds conditions of its own: compile GIVEN, and
only GIVEN.

## root

The table of the entity WHEN acts on (the booking being cancelled, the order
being returned). When WHEN only involves the person, the root IS the
person's table. Use null only when WHEN acts on no entity of DATABASE — an
empty GIVEN empties the conditions, never the root.

## Each condition

  {{"table": "<table>", "path": "<field path>", "op": "<op>", "value": <v>}}
  plus, when needed: "quant": "all"

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
          in — the field's value is one of the given list; contains — the
          field's value contains the given text (substring, strings only)
          count_eq count_gt count_lt count_ge count_le — compare HOW MANY
          values the path reaches against a non-negative integer. Only for
          clauses about HOW MANY ("more than one segment"); a clause about a
          property of some element compares the element's field instead. To
          count a collection's elements, address the collection itself
          ("flights[]"): the one case where the path is a listed path's
          PREFIX rather than a listed path.
  value   the literal to compare against: "in" takes a list, counts take a
          non-negative integer
  quant   "any" (default): some value the path reaches satisfies the
          comparison; "all": every one must, and at least one exists. A
          universal clause ("none ...", "every ...", "all ...") needs
          quant "all". Never on count_*.

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

## Time values are symbolic

Never compute a concrete timestamp for a temporal bound — emit it
symbolically:
  today / now:      "value": {{"ref":"$TODAY"}}  /  {{"ref":"$NOW"}}
  relative bound:   "value": {{"ref":"$NOW","offset":{{"hours":-24}}}}
                    "value": {{"ref":"$TODAY","offset":{{"days":7}}}}
`ref` is `$NOW` or `$TODAY`; offset units are `hours`, `days`, `years`
(negative = past). An absolute date written in GIVEN stays a literal string
("2024-05-20").

## If part of it cannot be expressed

Do not invent a path and do not approximate with a different condition:
compile the clauses that map onto the listed paths and leave the rest out —
what exists only in the dialogue (the user's wants, claims, reasons) or has
no listed path is not a database precondition.

Return only valid JSON."""

_COMPILE_USER = """\
THE RULE BEING TESTED: {rule}

THE SCENARIO
  GIVEN (compile THIS): {condition}
  WHEN  (context only): {trigger}
  THEN  (context only): {then}

DATABASE — every table and the paths you may use:
{vocabulary}

Return:
{{
  "root": "<table the test record comes from>" | null,
  "conditions": [{{"table":"...","path":"...","op":"...","value":...}}, ...]
}}
A condition may additionally carry "quant": "all" (universal over a
multi-value path)."""


def compile_condition(
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
    """Natural-language precondition -> (root, conditions, ungrounded, errors).

    `conditions` is one predicate PER TABLE — [{"table":..., "predicate":...}]
    — and `root` names the table the test record comes from. Nothing in the
    predicates crosses tables: the link between a booking and its user is a
    structural fact `select()` follows, not something the model has to spell
    out. Asking it to write a join is what produced most of the failures the
    per-table form removes: paths from one table smuggled into another's
    predicate.

    Only `condition` (the scenario's GIVEN) is compiled. `trigger`, `then` and
    `rule` are supplied so the model can read the Given correctly — which
    entity a pronoun refers to, which sense of an ambiguous phrase is meant.
    The root is identified from GIVEN alone; nothing suggests one.

    `ctx` is not used at compile time: temporal bounds come back symbolic
    ({"ref": ..., "offset": ...}), and the domain clock only matters when
    the conditions are evaluated (`evaluate` / `find_matching` take it).

    Parts that no path can express are returned in `ungrounded` rather than
    sinking the whole condition — a precondition is often half database state
    and half something the user says.
    """
    from src.mining_v2.common import call_json

    system = _COMPILE_SYSTEM.format(domain=domain)
    user = _COMPILE_USER.format(
        vocabulary=vocab.render(),
        rule=rule or "(not provided)", condition=condition,
        trigger=trigger or "(not provided)", then=then or "(not provided)")

    errors: list[str] = []
    for attempt in range(2):
        result = call_json(system, user, model, temperature,
                           max_tokens=1200, label="ground")
        root = result.get("root")
        root = root if root in vocab.tables else None
        ungrounded = [u for u in (result.get("ungrounded") or [])
                      if isinstance(u, dict) and u.get("part")]

        raw = result.get("conditions") or []
        errors = validate(raw, vocab)
        conditions = [c for c in raw if isinstance(c, dict)] if not errors else []

        # A root is required whenever there is anything to select; without one
        # the conditions cannot be linked to a single record.
        if conditions and root is None:
            root = conditions[0]["table"]
        if not errors:
            return root, conditions, ungrounded, []
        if attempt == 0:
            user += ("\n\nRETRY: the previous answer failed validation:\n- "
                     + "\n- ".join(errors)
                     + "\nReturn a corrected JSON object.")
    return None, [], [], errors
