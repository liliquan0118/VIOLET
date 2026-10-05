"""Domain-agnostic Step7 fixture-binder core for the v2 pipeline (see
docs/agentcoveragetesting_reuse_log.md sections 52-55). Extracted from the telecom-
only generic_tau_telecom_v2_fixture_binder_v1.py after a real survey showed airline
and retail need the SAME mechanism (structured Given-condition evaluation + identity-
field/relation-based binding resolution) but different real schema details: telecom's
tables are JSON lists with array-membership ownership (customer.line_ids[]), while
airline/retail's tables are JSON dicts keyed by id with a direct foreign-key field on
the child row (reservation.user_id / order.user_id) -- both container shapes and both
relation mechanisms are handled here; each domain supplies its own small
DomainBindingConfig instead of forking this module three times.

Scope (confirmed with the user for telecom in section 54, applied uniformly here):
  - Given == "True": no constraint.
  - Single-table, unquantified, literal-valued, single-step-path database_conditions
    (comparators: eq, ne, in, lt, gt, ge, le), AND-combined within one table.
  - The "owner" / "not_owner" non_database_conditions relation, via one domain-
    supplied RelationConfig (array-membership or direct-FK).
  - required_driver_binding_names sourced from a real identity field navigable from
    the selected fixture via the domain's identity_fields/relation config, or a fixed,
    domain-supplied free-form placeholder value.

Deliberately OUT of scope, reported as an honest, categorized failure per branch, not
silently mishandled or fabricated: aggregate/quantified conditions, cross-table
conditions, multi-step/array-traversal predicate paths (e.g. a dict-of-objects field
or a list-projection path), non-literal (e.g. clock-relative) comparison values, and
any required_driver_binding_name with no configured resolution strategy -- most
notably the multi-field CREATE/UPDATE "bundle" bindings (a real flight to book, a real
list of order items to exchange, a full mailing address) that airline/retail branches
need far more of than telecom's mostly-scalar bindings. Building those bundles is
real, per-operation domain modeling (the same kind already flagged for
generic_tau_airline_v1.py's v1 lineage), not something a generic identity-lookup can
produce, and is not attempted here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence


class GenericV2BindingError(ValueError):
    """Raised when a branch's Given condition or a required driver binding falls
    outside this module's confirmed scope, or when no real fixture satisfies it."""


@dataclass(frozen=True)
class RelationConfig:
    """One real relation for a domain, covering the real container shapes observed:
    'array_membership' (parent row has an array of child ids, e.g. telecom
    customer.line_ids[]), 'direct_fk' (child row has a direct parent-id field, e.g.
    airline/retail reservation.user_id / order.user_id), and
    'array_of_objects_membership' (child row has an array of OBJECTS, each with a
    field matching the parent's own id -- e.g. airline reservation.flights[] entries
    each carrying their own flight_number, not a plain id list; parent=flights,
    child=reservations, since a flight belongs to no one but many reservations
    reference it -- navigation-only, forward direction (parent->child) only, since no
    real branch needs the reverse; see docs/agentcoveragetesting_reuse_log.md
    section 60)."""

    mechanism: str  # "array_membership" | "direct_fk" | "array_of_objects_membership"
    parent_table: str
    child_table: str
    parent_id_field: str
    child_id_field: str
    array_field: str | None = None  # array_membership / array_of_objects_membership
    fk_field: str | None = None  # direct_fk only
    array_item_key_field: str | None = None  # array_of_objects_membership only


@dataclass(frozen=True)
class DomainBindingConfig:
    """Everything this module needs to know about one domain's real schema that
    isn't already carried in the v2 plan's own structured condition/binding data.

    A domain can have more than one real relation worth knowing about (e.g. telecom
    has customer<->line ownership AND customer<->bill, a separate direct-FK relation)
    -- ownership_relation is the one real "owner"/"not_owner" Given conditions refer
    to; navigation_relations is every 1-hop edge this module may use to reach a
    required_driver_binding_name's target table from wherever the Given condition
    actually anchored the fixture (normally includes ownership_relation itself plus
    any extra direct-FK relations)."""

    domain: str
    # required_driver_binding_name -> (table, real_field_name)
    identity_fields: Mapping[str, tuple[str, str]]
    # required_driver_binding_name -> (table, dict-valued_field_name); resolved as the
    # first real key of that row's dict field, e.g. airline's payment_id from a real
    # user's payment_methods dict (a real key, not a fabricated one -- which specific
    # real key is picked is not modeled beyond "the first one", since which payment
    # method is "right" for a given branch is real domain judgment out of scope here)
    dict_key_fields: Mapping[str, tuple[str, str]]
    # required_driver_binding_name -> fixed placeholder value (schema-valid, not a
    # fact about any real fixture)
    free_form_placeholders: Mapping[str, Any]
    ownership_relation: RelationConfig
    navigation_relations: Sequence[RelationConfig]
    default_table: str
    # optional (field, value) preferred when picking an unconstrained default row,
    # e.g. telecom prefers a real Active line over a Suspended one
    preferred_row_predicate: tuple[str, Any] | None = None
    # real target tool name -> (table, field, value) preferred when picking a row of
    # that table FOR A BRANCH TARGETING THAT TOOL -- applies both to the
    # unconstrained default-row pick (overriding preferred_row_predicate for that
    # field) AND to navigation-resolved rows of that same table (e.g. a real bill
    # reached from an unrelated anchor line via the customer relation). An
    # unconstrained Given ("True") never states the target operation's own real
    # precondition explicitly (e.g. resume_line needs a real Suspended line, the
    # opposite of the domain-wide Active preference; send_payment_request needs a
    # real non-Paid bill) -- this is a tie-break among equally Given-valid/reachable
    # rows so a live conversation's real tool call actually does something instead
    # of silently landing on a no-op or already-settled fixture (section 63). Never
    # relaxes or changes Given-condition correctness itself.
    operation_row_preferences: Mapping[str, tuple[str, str, Any]] = field(default_factory=dict)
    # real target tool name -> predicate(default_table row, database) -> bool, tried
    # BEFORE operation_row_preferences when picking an unconstrained default row.
    # operation_row_preferences can only express "one field equals one value" --
    # some real operations need a genuine cross-table, multi-condition eligibility
    # check (e.g. airline's send_certificate: the reservation must reference a real
    # cancelled/delayed flight-date instance AND the user must be silver/gold or the
    # reservation has real insurance or flies business, section 65) that a single
    # (table, field, value) tuple cannot express. The domain's own config supplies
    # the real predicate; this module only calls it.
    operation_row_selectors: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = field(
        default_factory=dict
    )
    # real condition_id (e.g. "retail_067_state_modify_pending_order_address#e0::
    # DB::1") -> a partial dict shallow-merged into that ONE condition's own
    # optional_encoding.expression before select_fixture/select_database_row ever
    # see it. Exists for a real, narrow class of bug: Step1/2's parse_database_
    # condition (v5_given_when_preparation_v1.py) is a lossless pass-through of the
    # frozen upstream corpus's own "quant" tag, and at least one real branch
    # (docs/agentcoveragetesting_reuse_log.md section 98.7/101) has that tag on a
    # genuine single-row scalar condition that should never have carried one --
    # re-deriving it correctly at the source would touch Step1/2's whole-corpus
    # assembly_fingerprint for every branch in the domain (section 99/100's already-
    # documented cascade cost), while this dict corrects the LAST point before real
    # runtime consumption, condition_id-scoped, touching no persisted JSON artifact
    # and no other branch's behavior at all. Never used to relax/change a condition's
    # real field/op/value -- only to correct a structurally wrong quantifier tag.
    database_condition_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # real (plan["branch_id"], required_driver_binding_name) -> a literal, real value
    # that short-circuits BOTH resolve_driver_bindings' and resolve_known_fact_
    # bundle's normal identity_fields/dict_key_fields/free_form_placeholders
    # resolution for that one name, on that one branch only. Exists for a real,
    # narrow structural gap (docs/agentcoveragetesting_reuse_log.md section 128):
    # a domain's navigation_relations graph may have no edge at all between two rows
    # of the SAME table (e.g. retail's only relation is users->orders -- there is no
    # users<->users edge), so _navigate_to_table/select_fixture can only ever anchor
    # ONE real row of that table per branch, and every identity_fields name mapped to
    # that table necessarily resolves off that SAME row. A branch whose real claim
    # needs a SECOND, genuinely distinct real row of that same table (e.g. "the
    # user_id argument the agent is asked to act on does NOT match the authenticated
    # user establishing the conversation") has no other way to get one without a new
    # relation or a new table-self-join mechanism, neither of which the shipped
    # corpus otherwise needs. This hook supplies that second value directly and
    # explicitly, scoped to (branch_id, name) so every other branch in every domain
    # -- including this domain's other branches sharing the same navigation graph --
    # resolves exactly as before, byte-identical (verified by a full-corpus diff,
    # section 128). Never used to change what the AUTHENTICATED/anchor row's own
    # fields resolve to -- those keep coming from the normal fixture navigation.
    identity_binding_overrides: Mapping[tuple[str, str], Any] = field(default_factory=dict)
    # real plan["branch_id"] -> a real tool_name, consulted by _pick_unconstrained_
    # row ONLY when that branch's own real observation_plan.tool_names_from_
    # effective_routes is empty (docs/agentcoveragetesting_reuse_log.md section 129,
    # task_ce5cb893). A real structural gap: operation_row_selectors/operation_row_
    # preferences (both keyed by tool_name) are consulted by iterating tool_names_
    # from_effective_routes -- but a branch whose Then compiles to a semantic_judge
    # check over assistant_message text (not a literal tool_call observation) has an
    # empty route list even when its own real When/rule_text names a specific real
    # target tool in prose (e.g. retail_069_state#b0: "the user confirms modification
    # of the payment method for a pending order" -- modify_pending_order_payment --
    # but no tool_call route, since the check is "the agent must not SAY the status
    # changes", not "the agent must call X"). Without this hint, such a branch's
    # unconstrained fixture silently falls through every real per-tool selector to
    # config.preferred_row_predicate/rows[0] -- deterministic, not random, and blind
    # to any tool-specific real precondition (e.g. status=="pending"). Scoped to
    # (branch_id -> tool_name), so a branch without an entry here is completely
    # unaffected, and a branch WITH real tool_names_from_effective_routes already
    # (the normal case) never consults this at all -- verified by a full-corpus
    # diff, section 129.
    unconstrained_row_tool_hints: Mapping[str, str] = field(default_factory=dict)
    # opt-in (default False, byte-identical to before for every domain that doesn't
    # set it): when True, _pick_unconstrained_row becomes aware of the DIRECTION a
    # branch's own observation_plan requires for each of its tool_names_from_
    # effective_routes -- "present" (the tool must genuinely be called; the existing,
    # only behavior before this flag existed) vs "absent" (the tool must NEVER be
    # called). operation_row_selectors was designed and is still correctly used for
    # the "present" direction: it picks a row for which the predicate holds, so a
    # live, policy-compliant agent has a genuine reason to call the target tool (e.g.
    # airline's send_certificate needs a genuinely disrupted+eligible reservation).
    # Applied unchanged to an "absent" branch, that exact same predicate is
    # backwards: deliberately picking a row the predicate affirms guarantees a
    # policy-compliant agent will independently discover and act on that same real,
    # unrelated fact (e.g. a genuinely cancelled segment) and legitimately call the
    # very tool the branch's own check requires to never be called -- making an
    # absent-operator check structurally unwinnable regardless of agent behavior
    # (docs/agentcoveragetesting_reuse_log.md section 146, task_e74ae55e:
    # airline_099_norm#b0/airline_095_norm#b0/airline_037_arg#e0/airline_038_arg#
    # e0/e1/e2 all share this exact shape -- fixture_source=="unconstrained", no
    # database_conditions or realizable!="conversation" non_database_conditions at
    # all to otherwise anchor the fixture, and a compiled OR01 observation_plan
    # event whose own expected_observation.operator=="absent" names send_certificate
    # as the scoped tool; _pick_unconstrained_row nonetheless applied AIRLINE_
    # CONFIG.operation_row_selectors["send_certificate"] -- an eligibility-AFFIRMING
    # predicate meant for the opposite, "present" direction -- and always landed on
    # the same shared genuinely-eligible reservation QDGWHB). When True, a tool_name
    # whose own observation_plan event(s) are ALL operator=="absent" is treated as
    # requiring AVOIDANCE: _pick_unconstrained_row prefers a row the selector/
    # preference does NOT affirm, falling through to the normal preferred_row_
    # predicate/rows[0] default (never raising) when every row affirms it. A
    # tool_name with no "absent" observation event (the normal case, including
    # every branch of any domain that never sets this flag) is completely
    # unaffected -- verified by a full-corpus diff (docs/agentcoveragetesting_
    # reuse_log.md section 146) for every domain, byte-identical.
    unconstrained_row_absent_operation_aware: bool = False
    # opt-in (default empty, byte-identical to before for every domain/branch without
    # an entry): real plan["branch_id"] -> predicate(row, database) -> bool, consulted
    # ONLY by select_database_row's plain single-table path (a branch whose Given has
    # real database_conditions), as a tie-break among the rows that ALREADY satisfy
    # every one of those Given conditions: the first Given-satisfying row the predicate
    # also affirms is preferred over the plain first match; when none does, the plain
    # first match is returned exactly as before (never raises, never relaxes or changes
    # the Given, never constructs a row). Exists for a real, narrow gap
    # (docs/agentcoveragetesting_reuse_log.md section 177, task_c31f4cc1):
    # operation_row_selectors/operation_row_preferences only ever run on the
    # unconstrained/relation paths, so a branch whose Given is e.g. "the order is
    # pending" always landed on the fixture db's literal first pending order -- for
    # retail's modify_pending_order_payment branches that is #W5918442, whose owner
    # has exactly ONE real payment method (the order's own current one), so the bound
    # "new" payment_method_id equals the current one and the tool's own real "The new
    # payment method should be different from the current one" precondition makes the
    # positive path impossible. Keyed by branch_id (not tool_name) on purpose: sibling
    # branches of the same tool whose own Given/Then deliberately require the SAME
    # payment method (e.g. retail_023_arg#e2) must keep their current fixture.
    database_row_tiebreak_selectors: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = field(
        default_factory=dict
    )


def _get_nested(row: Mapping[str, Any], dotted_field: str) -> Any:
    """A dotted field name (e.g. 'name.first_name') walks nested dicts; a plain name
    is a direct field access, unchanged from before this was added. Real domain
    fields sometimes nest this way (retail's users[].name.first_name, .address.zip)
    -- not a new access mechanism, just letting identity_fields/dict_key_fields name
    a nested real field instead of only a top-level one."""
    value: Any = row
    for step in dotted_field.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(step)
    return value


def _rows(database: Mapping[str, Any], table: str) -> list[Mapping[str, Any]]:
    value = database.get(table)
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, list):
        return value
    return []


_COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda actual, expected: actual == expected,
    "ne": lambda actual, expected: actual != expected,
    "in": lambda actual, expected: actual in expected,
    "lt": lambda actual, expected: actual < expected,
    "gt": lambda actual, expected: actual > expected,
    "ge": lambda actual, expected: actual >= expected,
    "le": lambda actual, expected: actual <= expected,
}


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _resolve_relative_test_time(right: Mapping[str, Any], reference_time: str) -> Any:
    """Resolve a real right.kind=='relative_test_time' value (e.g. airline's
    {"ref": "$NOW", "offset": {"hours": -24}}, telecom's {"ref": "$TODAY"}) into a
    real comparable datetime ($NOW) or date ($TODAY, date-only -- telecom's own real
    fields like contract_end_date are plain dates, not datetimes) using the domain's
    own real reference_time (verified per-domain: airline's environment.tools
    ._get_datetime(), telecom's real fixed tau2.domains.telecom.utils.get_now())."""

    ref = right.get("ref")
    offset = right.get("offset") or {}
    target = _parse_iso(reference_time) + timedelta(
        days=offset.get("days", 0), hours=offset.get("hours", 0),
        minutes=offset.get("minutes", 0), seconds=offset.get("seconds", 0),
    )
    if ref == "$TODAY":
        return target.date()
    if ref == "$NOW":
        return target
    raise GenericV2BindingError(f"unsupported relative_test_time ref {ref!r}")


def _parse_temporal_row_value(value: Any, like: Any) -> Any:
    if not isinstance(value, str):
        return value
    parsed = _parse_iso(value)
    return parsed.date() if not isinstance(like, datetime) else parsed


def _condition_holds(
    row: Mapping[str, Any], encoding: Mapping[str, Any], *, reference_time: str | None = None
) -> bool:
    op = encoding.get("operator")
    path_steps = encoding["subject"]["path_steps"]
    right = encoding["right"]
    if right.get("kind") == "relative_test_time":
        if reference_time is None:
            raise GenericV2BindingError(
                f"non-literal Given comparison values are not supported: right.kind={right.get('kind')!r}"
            )
        if len(path_steps) != 1 or path_steps[0].get("traversal") != "field":
            raise GenericV2BindingError("multi-step/non-field database predicate paths are not supported")
        comparator = _COMPARATORS.get(op)
        if comparator is None:
            raise GenericV2BindingError(f"unsupported database predicate operator {op!r}")
        target = _resolve_relative_test_time(right, reference_time)
        actual = row.get(path_steps[0]["field"])
        if actual is None:
            return comparator(None, target)
        return comparator(_parse_temporal_row_value(actual, target), target)
    if right.get("kind") != "literal":
        raise GenericV2BindingError(
            f"non-literal Given comparison values are not supported: right.kind={right.get('kind')!r}"
        )
    if (
        len(path_steps) == 2
        and path_steps[0].get("traversal") == "array_items"
        and path_steps[1].get("traversal") == "field"
    ):
        # A list-of-objects field projected across all its elements (e.g. retail's
        # orders.payment_history[].payment_method_id) -- "count_gt"/"contains" are
        # real predicate_kinds this shape alone uses (see docs/
        # agentcoveragetesting_reuse_log.md section 60); any other op means
        # existential match across the projected values, same convention as
        # object_values.
        container = row.get(path_steps[0]["field"])
        projected = (
            [item.get(path_steps[1]["field"]) if isinstance(item, Mapping) else None for item in container]
            if isinstance(container, list)
            else []
        )
        if op == "count_gt":
            if not isinstance(right["value"], (int, float)) or isinstance(right["value"], bool):
                raise GenericV2BindingError("count_gt requires a numeric comparison value")
            return sum(1 for value in projected if value is not None) > right["value"]
        if op == "contains":
            needle = right["value"]
            return any(
                isinstance(value, str) and isinstance(needle, str) and needle in value for value in projected
            )
        comparator = _COMPARATORS.get(op)
        if comparator is None:
            raise GenericV2BindingError(f"unsupported database predicate operator {op!r}")
        return any(comparator(value, right["value"]) for value in projected)
    comparator = _COMPARATORS.get(op)
    if comparator is None:
        raise GenericV2BindingError(f"unsupported database predicate operator {op!r}")
    if all(step.get("traversal") == "field" for step in path_steps):
        # Real nested dict access (e.g. retail's users.address.address1) -- any
        # length, not just a single field, is the same real access _get_nested
        # already does for binding resolution.
        value: Any = row
        for step in path_steps:
            if not isinstance(value, Mapping):
                return comparator(None, right["value"])
            value = value.get(step["field"])
        return comparator(value, right["value"])
    if (
        len(path_steps) == 2
        and path_steps[0].get("traversal") == "object_values"
        and path_steps[1].get("traversal") == "field"
    ):
        # A dict-of-objects field projected across all its values (e.g. airline's
        # flights.dates{}.status across every real date instance) -- this
        # aggregation happens WITHIN a single row's own nested dict, so it needs no
        # cross-row grouping at all (select_database_row lets a quantified condition
        # shaped like this reach here directly, see section 61); a real, explicit
        # quantifier (present=True, e.g. airline_093's "ALL flights not cancelled")
        # is honored as given, and the unquantified shape (present=False, the only
        # one sections 58-59 had real branches for) defaults to "at least one"
        # (existential) -- matching how this same idea is already handled elsewhere
        # in this project (generic_tau_airline_v1.py's _flight_statuses + "in").
        container = row.get(path_steps[0]["field"])
        if not isinstance(container, Mapping):
            return comparator(None, right["value"])
        projected = [
            item.get(path_steps[1]["field"]) if isinstance(item, Mapping) else None
            for item in container.values()
        ]
        quantifier = encoding.get("quantifier") or {}
        if quantifier.get("present") and quantifier.get("value") == "all":
            return all(comparator(value, right["value"]) for value in projected)
        return any(comparator(value, right["value"]) for value in projected)
    raise GenericV2BindingError("multi-step/non-field database predicate paths are not supported")


def _construct_temporal_value(op: str, target: Any) -> Any:
    """A real value that satisfies `field {op} target` for a relative_test_time
    comparison (e.g. contract_end_date lt $TODAY) -- le/ge are trivially satisfied by
    the boundary value itself, lt/gt need a value strictly past/before it. `target` is
    always what _resolve_relative_test_time returns: a plain date ($TODAY) or a full
    datetime ($NOW) -- never a naive guess, always derived from the domain's own real
    reference clock (section 61)."""
    delta = timedelta(seconds=1) if isinstance(target, datetime) else timedelta(days=1)
    if op == "lt":
        return target - delta
    if op == "gt":
        return target + delta
    if op in ("le", "ge"):
        return target
    raise GenericV2BindingError(f"unsupported temporal construction operator {op!r}")


def _construct_missing_row(
    database: Mapping[str, Any],
    table: str,
    database_conditions: Sequence[Mapping[str, Any]],
    *,
    reference_time: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Best-effort, generic construction of a row satisfying the branch's
    database_conditions when no real row already does, by taking a real row from the
    same table and patching only the field(s) the conditions name -- the row's own id
    and every other real field (including whatever makes it navigable to other real
    rows, e.g. a constructed line's line_id still really belongs to its real owning
    customer) stay untouched. Supports eq/in/ne on a literal, unquantified,
    single-step field (sections 56 and 60), plus lt/gt/le/ge on a relative_test_time
    right value when a real reference_time is available (section 61, e.g. telecom_058's
    "contract_end_date is in the past" -- constructs a real date strictly before/after
    the domain's own real $NOW/$TODAY, or reuses the boundary value itself for
    le/ge). Returns None for anything else (a gt/lt condition against a literal, or a
    relative_test_time condition with no reference_time) so the caller falls back to
    the normal, honest out-of-scope error instead of guessing a value. "ne" never
    picks a value (there is no single "not equal to X" construction); it only requires
    the eventual base row to already satisfy it for real, trying successive real rows
    until one does (or none do).

    This is a real, accepted risk (see section 56): the patched field is not verified
    consistent with any other real business rule the live system might enforce beyond
    Pydantic's own type validation (checked when the resulting patch is actually
    applied via environment.tools.update_db -- see generic_tau_v2_bound_plan_adapter
    _v1.py). It never invents a table row that doesn't otherwise exist for real, and
    never touches any field the branch's own conditions didn't name.
    """

    # Two passes so an "eq" pins a field's value regardless of encounter order --
    # an "in" processed first (e.g. telecom_106_state#b2's status IN [Active,
    # Suspended, Pending Activation, Closed] alongside status == Closed) must not
    # lock in its own first-list-item default before a same-field "eq" is seen.
    parsed: list[tuple[str, str, Any]] = []
    for condition in database_conditions:
        encoding = condition["optional_encoding"]["expression"]
        op = encoding["operator"]
        path_steps = encoding["subject"]["path_steps"]
        if len(path_steps) != 1 or path_steps[0].get("traversal") != "field":
            return None
        if encoding["quantifier"]["present"]:
            return None
        right = encoding["right"]
        field = path_steps[0]["field"]
        if right.get("kind") == "relative_test_time":
            if op not in ("lt", "gt", "le", "ge") or reference_time is None:
                return None
            target = _resolve_relative_test_time(right, reference_time)
            # Folded into the "eq" pass below -- a resolved temporal value is just as
            # definite a value to set as a literal eq's, and gets the same
            # same-field-contradiction check for free.
            parsed.append((field, "eq", _construct_temporal_value(op, target).isoformat()))
            continue
        if right.get("kind") != "literal":
            return None
        if op not in ("eq", "in", "ne"):
            return None
        parsed.append((field, op, right["value"]))

    field_values: dict[str, Any] = {}
    for field, op, value in parsed:
        if op != "eq":
            continue
        if field in field_values and field_values[field] != value:
            raise GenericV2BindingError(
                f"self-contradictory Given condition: {table}.{field} must equal both "
                f"{field_values[field]!r} and {value!r}"
            )
        field_values[field] = value
    for field, op, value in parsed:
        if op != "in":
            continue
        if not value:
            return None
        if field in field_values:
            if field_values[field] not in value:
                raise GenericV2BindingError(
                    f"self-contradictory Given condition: {table}.{field}={field_values[field]!r} "
                    f"is not in the required set {value!r}"
                )
        else:
            field_values[field] = value[0]
    # "ne" is never used to pick a value (there is no single "not equal to X" value to
    # construct) -- it only needs to hold on whatever real, untouched value the
    # eventual base row already has for that field (e.g. telecom_110_state#b2's real
    # "line_id != ''", true of every real line already), so it is checked against
    # each candidate base row below rather than folded into field_values.
    ne_conditions = [(field, value) for field, op, value in parsed if op == "ne"]

    for base_row in _rows(database, table):
        patched_row = {**deepcopy(dict(base_row)), **field_values}
        if any(patched_row.get(field) == value for field, value in ne_conditions):
            continue
        container = database.get(table)
        if isinstance(container, Mapping):
            row_id = next((key for key, value in container.items() if value is base_row), None)
            if row_id is None:
                continue
            patch = {table: {row_id: deepcopy(field_values)}}
        else:
            full_list = []
            replaced = False
            for row in container or []:
                if row is base_row and not replaced:
                    full_list.append(deepcopy(patched_row))
                    replaced = True
                else:
                    full_list.append(deepcopy(dict(row)))
            patch = {table: full_list}
        return patched_row, patch
    return None


def _is_within_row_quantified_shape(condition: Mapping[str, Any]) -> bool:
    """True when a quantified condition's aggregation happens within one row's own
    nested dict/list field (object_values/array_items), not across separate rows --
    _condition_holds already evaluates these directly and honors the real quantifier
    value, so they need no relation-based row-grouping at all (section 61)."""
    path_steps = condition["optional_encoding"]["expression"]["subject"]["path_steps"]
    return len(path_steps) == 2 and path_steps[0].get("traversal") in ("object_values", "array_items")


def _select_cross_table_relation_row(
    database: Mapping[str, Any],
    database_conditions: Sequence[Mapping[str, Any]],
    relation: RelationConfig,
    *,
    reference_time: str | None = None,
) -> tuple[str, Mapping[str, Any]] | None:
    """If database_conditions span exactly the domain's ownership_relation's
    parent_table and child_table (e.g. airline's "user is a regular member AND the
    reservation is economy cabin", spanning users+reservations), find a real
    (parent, child) pair genuinely related to each other where each condition holds
    against whichever table it names, and anchor the fixture on the child row (the
    more specific entity, matching what a plain single-table database_conditions
    branch on that same child table would anchor on). Returns None if the
    conditions' tables don't match this relation -- the caller falls back to the
    ordinary cross-table rejection."""

    tables = {c["optional_encoding"]["expression"]["subject"]["table"] for c in database_conditions}
    if tables != {relation.parent_table, relation.child_table}:
        return None
    for condition in database_conditions:
        if condition["optional_encoding"]["expression"]["quantifier"]["present"] and not _is_within_row_quantified_shape(
            condition
        ):
            raise GenericV2BindingError("aggregate/quantified Given conditions are not supported")
    for parent in _rows(database, relation.parent_table):
        for child in _rows(database, relation.child_table):
            if not _relation_holds(relation, parent, child):
                continue
            if _row_satisfies_extra_conditions(
                parent, child, relation, database_conditions, reference_time=reference_time
            ):
                return relation.child_table, child
    raise GenericV2BindingError(
        f"no real ({relation.parent_table}, {relation.child_table}) pair satisfies "
        "the combined cross-table Given condition"
    )


def _quantified_condition_holds_over_group(
    children: Sequence[Mapping[str, Any]], condition: Mapping[str, Any], *, reference_time: str | None = None
) -> bool:
    encoding = condition["optional_encoding"]["expression"]
    quant = condition["optional_encoding"]["expression"]["quantifier"].get("value")
    if not children:
        # Vacuous groups: "all" is vacuously true (no real bill violates it), "any"
        # is false (no real bill satisfies it) -- standard quantifier convention.
        return quant == "all"
    if quant == "all":
        return all(_condition_holds(child, encoding, reference_time=reference_time) for child in children)
    if quant == "any":
        return any(_condition_holds(child, encoding, reference_time=reference_time) for child in children)
    raise GenericV2BindingError(f"unsupported quantifier value {quant!r}")


def _select_quantified_group_row(
    database: Mapping[str, Any],
    database_conditions: Sequence[Mapping[str, Any]],
    relation: RelationConfig,
    *,
    reference_time: str | None = None,
) -> tuple[str, Mapping[str, Any]] | None:
    """Find a real parent row (relation.parent_table) whose full real group of child
    rows (relation.child_table, via relation) collectively satisfies every one of
    database_conditions -- each condition evaluated with its own real quantifier
    (all/any) over that real group. Only supports every condition sharing a single
    quantified child table (no real branch combines a quantified group condition
    with anything else, see section 61); returns None if the domain has no real
    group with real children to check against at all."""

    for parent in _rows(database, relation.parent_table):
        children = [
            child for child in _rows(database, relation.child_table) if _relation_holds(relation, parent, child)
        ]
        if all(
            _quantified_condition_holds_over_group(children, c, reference_time=reference_time)
            for c in database_conditions
        ):
            return relation.parent_table, parent
    return None


def select_database_row(
    database: Mapping[str, Any],
    database_conditions: Sequence[Mapping[str, Any]],
    *,
    allow_construction: bool = False,
    relations: Sequence[RelationConfig] = (),
    reference_time: str | None = None,
    row_preference: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
) -> tuple[str, Mapping[str, Any], Mapping[str, Any] | None]:
    """row_preference (optional, default None -- byte-identical original behavior):
    see DomainBindingConfig.database_row_tiebreak_selectors. Only a tie-break among
    rows that already satisfy every Given condition on the plain single-table path;
    the cross-table/quantified-group paths and row construction ignore it."""
    tables = {c["optional_encoding"]["expression"]["subject"]["table"] for c in database_conditions}
    if len(tables) > 1:
        # Try every real relation this domain knows about (not just the one
        # designated ownership_relation) -- a cross-table Given may legitimately
        # span two tables connected only by a navigation-only relation (e.g.
        # telecom's plans<->lines, added in section 60 for binding navigation, not
        # originally wired into this join at all -- see section 61).
        for relation in relations:
            joined = _select_cross_table_relation_row(
                database, database_conditions, relation, reference_time=reference_time
            )
            if joined is not None:
                return joined[0], joined[1], None
        raise GenericV2BindingError(f"cross-table Given conditions are not supported: {sorted(tables)}")
    table = next(iter(tables))
    quantified = [c for c in database_conditions if c["optional_encoding"]["expression"]["quantifier"]["present"]]
    cross_row_quantified = [c for c in quantified if not _is_within_row_quantified_shape(c)]
    if cross_row_quantified:
        # A quantified condition on a plain single-step field (e.g. telecom's "the
        # user has paid ALL their overdue bills") is inherently about a whole real
        # group of SEPARATE rows under some real parent (every real bill belonging
        # to that real customer) -- a single row only ever has one scalar value for
        # that field, so the quantifier can only make sense across a real relation's
        # grouped children. Try every real relation whose child_table matches (see
        # docs/agentcoveragetesting_reuse_log.md section 61).
        for relation in relations:
            if relation.child_table != table:
                continue
            grouped = _select_quantified_group_row(
                database, database_conditions, relation, reference_time=reference_time
            )
            if grouped is not None:
                return grouped[0], grouped[1], None
        raise GenericV2BindingError("aggregate/quantified Given conditions are not supported")
    # Any remaining quantified condition(s) are shaped like object_values/array_items
    # (e.g. airline's "ALL of this reservation's flights are not cancelled" --
    # dates{}.status aggregated WITHIN one real flight/reservation row's own nested
    # dict/list) -- no cross-row grouping needed at all; _condition_holds already
    # honors the real quantifier value for this shape, so these fall straight
    # through to the ordinary single-row filter below, same as any other condition.
    matches = [
        row
        for row in _rows(database, table)
        if all(
            _condition_holds(row, c["optional_encoding"]["expression"], reference_time=reference_time)
            for c in database_conditions
        )
    ]
    if matches:
        if row_preference is not None:
            preferred = next((row for row in matches if row_preference(row, database)), None)
            if preferred is not None:
                return table, preferred, None
        return table, matches[0], None
    if allow_construction:
        constructed = _construct_missing_row(database, table, database_conditions, reference_time=reference_time)
        if constructed is not None:
            patched_row, patch = constructed
            return table, patched_row, patch
    raise GenericV2BindingError(f"no real {table!r} row satisfies the Given condition")


def _relation_holds(relation: RelationConfig, parent: Mapping[str, Any], child: Mapping[str, Any]) -> bool:
    if relation.mechanism == "array_membership":
        child_id = child.get(relation.child_id_field)
        return child_id in (parent.get(relation.array_field) or [])
    if relation.mechanism == "direct_fk":
        # The child's OWN id (relation.child_id_field, e.g. order_id) is never equal
        # to the parent's own id (e.g. user_id) -- comparing those two directly (a
        # real bug found and fixed here, see docs/agentcoveragetesting_reuse_log.md
        # section 57) always evaluates False regardless of real ownership. The real
        # check is the child's OWN foreign-key field (relation.fk_field, e.g.
        # order.user_id) against the parent's own id.
        return child.get(relation.fk_field) == parent.get(relation.parent_id_field)
    if relation.mechanism == "array_of_objects_membership":
        parent_id = parent.get(relation.parent_id_field)
        items = child.get(relation.array_field) or []
        return any(
            isinstance(item, Mapping) and item.get(relation.array_item_key_field) == parent_id
            for item in items
        )
    raise GenericV2BindingError(f"unsupported relation mechanism {relation.mechanism!r}")


def _row_satisfies_extra_conditions(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    relation: RelationConfig,
    extra_conditions: Sequence[Mapping[str, Any]],
    *,
    reference_time: str | None = None,
) -> bool:
    for condition in extra_conditions:
        encoding = condition["optional_encoding"]["expression"]
        table = encoding["subject"]["table"]
        if table == relation.parent_table:
            row = parent
        elif table == relation.child_table:
            row = child
        else:
            raise GenericV2BindingError(
                f"combined database_conditions reference a table ({table!r}) unrelated to "
                f"the branch's own relation ({relation.parent_table!r}/{relation.child_table!r})"
            )
        if not _condition_holds(row, encoding, reference_time=reference_time):
            return False
    return True


def _relation_child_operation_predicate(
    plan: Mapping[str, Any] | None,
    config: "DomainBindingConfig | None",
    child_table: str,
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None:
    """Mirrors _pick_unconstrained_row's own two-tier operation-preference lookup
    (operation_row_selectors tried first, then operation_row_preferences) -- but
    select_relation_pair's own anchor is always relation.parent_table, so the rows
    an operation preference should filter are relation.child_table, not necessarily
    config.default_table (they coincide for retail's orders, but nothing guarantees
    that in general). A branch with no real Given database_condition on the child
    table (e.g. retail's "the order belongs to the authenticated user", with no
    stated status) previously reached _relation_holds with no awareness at all of
    what its own real target tool needs to be callable -- select_relation_pair's
    unconstrained first-match then silently returns whichever (parent, child) pair
    happens to be first in real fixture-db iteration order, independent of whether
    that child row is real usable by the branch's own tool (see docs/
    agentcoveragetesting_reuse_log.md section 116: retail_088/092/094_state#b0 all
    landed on noah_brown_6181's one real order, which is "delivered", while their
    real target tools -- modify_pending_order_address/items/payment -- all real-
    require "pending" to even be callable). Returns None (no filtering at all,
    byte-identical original behavior) when the branch's own tool_names_from_
    effective_routes have no configured selector/preference for child_table, exactly
    as before this was added."""
    if plan is None or config is None:
        return None
    tool_names = plan["observation_plan"].get("tool_names_from_effective_routes") or []
    for tool_name in tool_names:
        selector = config.operation_row_selectors.get(tool_name)
        if selector is not None:
            return selector
    for tool_name in tool_names:
        preference = config.operation_row_preferences.get(tool_name)
        if preference is None:
            continue
        pref_table, field_name, expected = preference
        if pref_table == child_table:
            return lambda row, database, field_name=field_name, expected=expected: row.get(field_name) == expected
    return None


def select_relation_pair(
    database: Mapping[str, Any],
    non_database_conditions: Sequence[Mapping[str, Any]],
    relation: RelationConfig,
    *,
    extra_conditions: Sequence[Mapping[str, Any]] = (),
    reference_time: str | None = None,
    operation_preference: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
) -> tuple[Mapping[str, Any], Any] | None:
    """Returns (parent_row, child_id) for the branch's real owner/not_owner
    condition, or None if the branch has no such condition (e.g. Given=="True" or
    every non_database_condition is realizable=="conversation", which is genuinely
    Step8's job, not an unsupported shape). extra_conditions (real
    database_conditions from a branch that combines both condition types, e.g. "a
    PENDING order that ALSO belongs to the authenticated user") must additionally
    hold on whichever of parent/child their own table names -- reuses the same
    eq/ne/in/lt/gt/ge/le/literal/single-step-field scope _condition_holds already
    enforces, so an unsupported combined condition shape fails exactly as honestly as
    an unsupported plain database_conditions shape would. operation_preference
    (optional, see _relation_child_operation_predicate/section 116) is tried FIRST,
    over the exact same (parent, child) search -- a preferred pair, if any exists,
    is returned instead of the plain first match; when no preferred pair exists (or
    operation_preference is None, its default), this falls back to the exact
    original unconstrained first-match scan, so every branch with no configured
    operation preference for its own target tool(s) is byte-identical to before."""

    relevant = [
        c for c in non_database_conditions if c["source_condition"].get("realizable") != "conversation"
    ]
    unsupported = [c for c in relevant if c["source_condition"].get("relation") not in ("owner", "not_owner")]
    if unsupported:
        relations = sorted({c["source_condition"].get("relation") for c in unsupported})
        raise GenericV2BindingError(f"unsupported non_database_conditions relation(s): {relations}")
    if not relevant:
        return None
    if len(relevant) > 1:
        raise GenericV2BindingError("multiple relation conditions in one branch are not supported")
    wants_owner = relevant[0]["source_condition"]["relation"] == "owner"
    parents = _rows(database, relation.parent_table)
    children = _rows(database, relation.child_table)
    if operation_preference is not None:
        for parent in parents:
            for child in children:
                holds = _relation_holds(relation, parent, child)
                if holds != wants_owner:
                    continue
                if extra_conditions and not _row_satisfies_extra_conditions(
                    parent, child, relation, extra_conditions, reference_time=reference_time
                ):
                    continue
                if not operation_preference(child, database):
                    continue
                return parent, child.get(relation.child_id_field)
    for parent in parents:
        for child in children:
            holds = _relation_holds(relation, parent, child)
            if holds != wants_owner:
                continue
            if extra_conditions and not _row_satisfies_extra_conditions(
                parent, child, relation, extra_conditions, reference_time=reference_time
            ):
                continue
            return parent, child.get(relation.child_id_field)
    raise GenericV2BindingError(
        f"no real ({relation.parent_table}, {relation.child_table}) pair "
        f"{'satisfying' if wants_owner else 'violating'} the ownership relation was found"
    )


def _absent_operation_tool_names(plan: Mapping[str, Any]) -> set[str]:
    """The set of real tool_names for which EVERY real observation_plan event
    scoping that tool on this branch has expected_observation.operator=="absent"
    (the branch's own compiled check requires the tool to NEVER be called) --
    derived directly from the same real plan data _pick_unconstrained_row already
    receives, no new plan field needed. A tool_name with at least one non-"absent"
    event for it (e.g. "present"/"non_decisive"/"relation_true") is deliberately
    excluded -- only used when config.unconstrained_row_absent_operation_aware is
    True (see that field's own docstring, docs/agentcoveragetesting_reuse_log.md
    section 146)."""
    operators_by_tool: dict[str, set[str]] = {}
    for event in plan.get("observation_plan", {}).get("events") or []:
        tool_name = (
            ((event.get("target_event") or {}).get("event_filter") or {}).get("field_equals") or {}
        ).get("tool_name")
        if not tool_name:
            continue
        operator = (event.get("expected_observation") or {}).get("operator")
        operators_by_tool.setdefault(tool_name, set()).add(operator)
    return {tool_name for tool_name, ops in operators_by_tool.items() if ops == {"absent"}}


def _pick_unconstrained_row(
    plan: Mapping[str, Any], database: Mapping[str, Any], config: DomainBindingConfig
) -> Mapping[str, Any]:
    rows = _rows(database, config.default_table)
    tool_names = plan["observation_plan"].get("tool_names_from_effective_routes") or []
    if not tool_names:
        # section 129: a branch whose real route list is empty (e.g. a
        # semantic_judge-only check with no literal tool_call observation) may
        # still name a real target tool via this domain's own branch_id-scoped
        # hint -- see DomainBindingConfig.unconstrained_row_tool_hints' own
        # docstring. A real no-op for every branch without an entry.
        hinted_tool = config.unconstrained_row_tool_hints.get(plan.get("branch_id"))
        if hinted_tool is not None:
            tool_names = [hinted_tool]
    absent_tools = (
        _absent_operation_tool_names(plan) if config.unconstrained_row_absent_operation_aware else set()
    )
    for tool_name in tool_names:
        selector = config.operation_row_selectors.get(tool_name)
        if selector is not None:
            if tool_name in absent_tools:
                # See DomainBindingConfig.unconstrained_row_absent_operation_aware:
                # this branch's own check requires send_certificate-style tool_name
                # to NEVER be called, so a row the selector affirms (its normal,
                # "present"-direction meaning) is exactly the wrong row to pick --
                # prefer one it does NOT affirm instead.
                preferred = next((row for row in rows if not selector(row, database)), None)
            else:
                preferred = next((row for row in rows if selector(row, database)), None)
            if preferred is not None:
                return preferred
    for tool_name in tool_names:
        preference = config.operation_row_preferences.get(tool_name)
        if preference is None:
            continue
        pref_table, field_name, expected = preference
        if pref_table == config.default_table:
            if tool_name in absent_tools:
                preferred = next((row for row in rows if row.get(field_name) != expected), None)
            else:
                preferred = next((row for row in rows if row.get(field_name) == expected), None)
            if preferred is not None:
                return preferred
            continue
        # Cross-table preference (e.g. send_payment_request needs a real Overdue
        # bill, but the anchor table is "lines" -- a line's OWN fields can never
        # satisfy a bills-table preference). Probe each real default_table row's
        # own real navigation to pref_table and prefer one that actually reaches a
        # row satisfying it (e.g. a line whose real owning customer genuinely has a
        # real Overdue bill), not just any line whose customer happens to be
        # navigated to first (section 63).
        for row in rows:
            probe = Fixture(table=config.default_table, row=row, fixture_source="probe")
            target_row = _navigate_to_table(probe, pref_table, database, config, plan=plan)
            if target_row is not None and target_row.get(field_name) == expected:
                return row
    if config.preferred_row_predicate is not None:
        field_name, expected = config.preferred_row_predicate
        preferred = next((row for row in rows if row.get(field_name) == expected), None)
        if preferred is not None:
            return preferred
    if not rows:
        raise GenericV2BindingError(f"no real {config.default_table!r} row exists in the fixture db at all")
    return rows[0]


@dataclass(frozen=True)
class Fixture:
    table: str
    row: Mapping[str, Any]
    fixture_source: str
    state_patch: Mapping[str, Any] | None = None
    # The specific child id select_relation_pair found (e.g. the one reservation
    # genuinely confirmed NOT to belong to this fixture's user, for a not_owner
    # branch) -- navigation to the ownership relation's own child_table MUST reuse
    # this exact id, not re-derive "some" child generically (which, for a not_owner
    # branch, would silently find an OWNED child instead -- a real bug fixed in
    # section 57; see _navigate_via_relation's use of this field).
    relation_child_id: Any | None = None


def _apply_database_condition_overrides(
    database_conditions: Sequence[Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    """Shallow-merge config.database_condition_overrides into optional_encoding.
    expression for just the condition_id(s) named -- every other condition (this
    branch's own others, and every other branch's) passes through unchanged."""
    if not overrides:
        return database_conditions
    result = []
    for condition in database_conditions:
        override = overrides.get(condition.get("condition_id"))
        if override is None:
            result.append(condition)
            continue
        condition = deepcopy(dict(condition))
        condition["optional_encoding"] = deepcopy(dict(condition["optional_encoding"]))
        condition["optional_encoding"]["expression"] = {
            **condition["optional_encoding"]["expression"],
            **override,
        }
        result.append(condition)
    return result


def select_fixture(
    plan: Mapping[str, Any],
    database: Mapping[str, Any],
    config: DomainBindingConfig,
    *,
    allow_construction: bool = False,
    reference_time: str | None = None,
) -> Fixture:
    """Evaluate one real v2 driver plan's Given condition against the real database
    and return the real row selected. Raises GenericV2BindingError, with a real
    reason, for anything outside the confirmed scope -- never fabricates a fixture,
    unless allow_construction=True and no real row satisfies a (eq/in-only) condition,
    in which case a real row from the same table is patched to satisfy it (see
    _construct_missing_row) and Fixture.state_patch carries the real update_db-shaped
    patch a live Step8 run must apply before the conversation starts."""

    conditions = plan["fixture_binding_plan"]["conditions"]
    database_conditions = _apply_database_condition_overrides(
        conditions["database_conditions"], config.database_condition_overrides
    )
    non_database_conditions = conditions["non_database_conditions"]
    # section 177: None (original behavior) for every branch without an entry.
    row_preference = config.database_row_tiebreak_selectors.get(plan.get("branch_id"))
    if database_conditions and non_database_conditions:
        relevant_ndb = [
            c for c in non_database_conditions if c["source_condition"].get("realizable") != "conversation"
        ]
        if not relevant_ndb:
            # The non_database part is entirely realizable=="conversation" -- Step8's
            # job (realize_conversation_prerequisites), not an unsupported shape for
            # Step7 -- fall through to evaluating database_conditions alone, same as
            # if the branch never had a non_database_conditions entry at all.
            table, row, patch = select_database_row(
                database, database_conditions,
                allow_construction=allow_construction, relations=config.navigation_relations,
                reference_time=reference_time, row_preference=row_preference,
            )
            source = f"database_row:{table}" if patch is None else f"constructed_via_state_patch:{table}"
            return Fixture(table=table, row=row, fixture_source=source, state_patch=patch)
        pair = select_relation_pair(
            database, non_database_conditions, config.ownership_relation,
            extra_conditions=database_conditions, reference_time=reference_time,
        )
        if pair is None:
            raise GenericV2BindingError(
                "combined database_conditions with an unsupported non_database_conditions shape are not supported"
            )
        parent_row, child_id = pair
        return Fixture(
            table=config.ownership_relation.parent_table,
            row=parent_row,
            fixture_source="relation",
            relation_child_id=child_id,
        )
    if database_conditions:
        table, row, patch = select_database_row(
            database, database_conditions,
            allow_construction=allow_construction, relations=config.navigation_relations,
            reference_time=reference_time, row_preference=row_preference,
        )
        source = f"database_row:{table}" if patch is None else f"constructed_via_state_patch:{table}"
        return Fixture(table=table, row=row, fixture_source=source, state_patch=patch)
    if non_database_conditions:
        operation_preference = _relation_child_operation_predicate(
            plan, config, config.ownership_relation.child_table
        )
        pair = select_relation_pair(
            database, non_database_conditions, config.ownership_relation,
            operation_preference=operation_preference,
        )
        if pair is None:
            row = _pick_unconstrained_row(plan, database, config)
            return Fixture(table=config.default_table, row=row, fixture_source="unconstrained")
        parent_row, child_id = pair
        return Fixture(
            table=config.ownership_relation.parent_table,
            row=parent_row,
            fixture_source="relation",
            relation_child_id=child_id,
        )
    row = _pick_unconstrained_row(plan, database, config)
    return Fixture(table=config.default_table, row=row, fixture_source="unconstrained")


def navigate_to_table(
    fixture: Fixture,
    target_table: str,
    database: Mapping[str, Any],
    config: DomainBindingConfig,
    *,
    plan: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Public: navigate from the selected fixture row to a real row in target_table
    via the domain's one configured relation, or None if target_table is neither the
    fixture's own table nor reachable through that relation. Exposed so callers can
    build a full fixture-identity audit view beyond just the branch's own required
    bindings (see generic_tau_telecom_v2_fixture_binder_v1.py). plan is optional (see
    _navigate_to_table's own docstring, section 63)."""
    return _navigate_to_table(fixture, target_table, database, config, plan=plan)


def _navigate_via_relation(
    fixture: Fixture,
    target_table: str,
    database: Mapping[str, Any],
    relation: RelationConfig,
    *,
    preferred_child_id: Any = None,
    preferred_row: tuple[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    if fixture.table == relation.parent_table and target_table == relation.child_table:
        if preferred_child_id is not None:
            # select_relation_pair already found and verified this specific child
            # against the branch's real owner/not_owner condition (e.g. genuinely
            # NOT owned, for a not_owner branch) -- re-deriving "some" child here
            # generically would silently pick a DIFFERENT, unverified one (for
            # direct_fk this naturally finds an OWNED child, the opposite of what a
            # not_owner branch needs; see docs/agentcoveragetesting_reuse_log.md
            # section 57).
            for row in _rows(database, target_table):
                if row.get(relation.child_id_field) == preferred_child_id:
                    return row
            return None
        if relation.mechanism == "array_membership":
            child_ids = fixture.row.get(relation.array_field) or []
            if not child_ids:
                return None
            children_by_id = {row.get(relation.child_id_field): row for row in _rows(database, target_table)}
            if preferred_row is not None:
                field_name, expected = preferred_row
                preferred = next(
                    (children_by_id[cid] for cid in child_ids if children_by_id.get(cid, {}).get(field_name) == expected),
                    None,
                )
                if preferred is not None:
                    return preferred
            return children_by_id.get(child_ids[0])
        if relation.mechanism == "direct_fk":
            parent_id = fixture.row.get(relation.parent_id_field)
            candidates = [row for row in _rows(database, target_table) if row.get(relation.fk_field) == parent_id]
            if not candidates:
                return None
            if preferred_row is not None:
                field_name, expected = preferred_row
                preferred = next((row for row in candidates if row.get(field_name) == expected), None)
                if preferred is not None:
                    return preferred
            return candidates[0]
        if relation.mechanism == "array_of_objects_membership":
            parent_id = fixture.row.get(relation.parent_id_field)
            for row in _rows(database, target_table):
                items = row.get(relation.array_field) or []
                if any(
                    isinstance(item, Mapping) and item.get(relation.array_item_key_field) == parent_id
                    for item in items
                ):
                    return row
            return None
    if fixture.table == relation.child_table and target_table == relation.parent_table:
        if relation.mechanism == "array_membership":
            child_id = fixture.row.get(relation.child_id_field)
            for row in _rows(database, target_table):
                if child_id in (row.get(relation.array_field) or []):
                    return row
            return None
        if relation.mechanism == "direct_fk":
            fk_value = fixture.row.get(relation.fk_field)
            for row in _rows(database, target_table):
                if row.get(relation.parent_id_field) == fk_value:
                    return row
            return None
    return None


def _operation_row_preference(
    plan: Mapping[str, Any] | None, config: DomainBindingConfig, table: str
) -> tuple[str, Any] | None:
    if plan is None:
        return None
    for tool_name in plan["observation_plan"].get("tool_names_from_effective_routes") or []:
        preference = config.operation_row_preferences.get(tool_name)
        if preference is not None and preference[0] == table:
            return preference[1], preference[2]
    return None


def _navigate_to_table(
    fixture: Fixture,
    target_table: str,
    database: Mapping[str, Any],
    config: DomainBindingConfig,
    *,
    plan: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Breadth-first search over config.navigation_relations from the fixture's own
    table to target_table -- real schemas need more than one hop (e.g. telecom's
    lines -> customers -> bills: lines and bills share no direct relation, only both
    relate to customers). plan (optional) lets a real config.operation_row_preferences
    entry for one of the branch's own target tools prefer a real row of target_table
    that makes that tool's operation meaningful (e.g. a real non-Paid bill for
    send_payment_request) over whichever one this table's relation would otherwise
    reach first (section 63)."""

    if fixture.table == target_table:
        return fixture.row
    preferred_row = _operation_row_preference(plan, config, target_table)
    frontier: list[Fixture] = [fixture]
    visited_tables = {fixture.table}
    while frontier:
        next_frontier: list[Fixture] = []
        for current in frontier:
            for relation in config.navigation_relations:
                for candidate_table in (relation.parent_table, relation.child_table):
                    if candidate_table in visited_tables:
                        continue
                    preferred_child_id = (
                        current.relation_child_id if relation is config.ownership_relation else None
                    )
                    found = _navigate_via_relation(
                        current, candidate_table, database, relation,
                        preferred_child_id=preferred_child_id,
                        preferred_row=preferred_row if candidate_table == target_table else None,
                    )
                    if found is None:
                        continue
                    if candidate_table == target_table:
                        return found
                    visited_tables.add(candidate_table)
                    next_frontier.append(Fixture(table=candidate_table, row=found, fixture_source="navigated"))
        frontier = next_frontier
    return None


def resolve_driver_bindings(
    plan: Mapping[str, Any], fixture: Fixture, database: Mapping[str, Any], config: DomainBindingConfig
) -> dict[str, Any]:
    required = plan["fixture_binding_plan"]["required_driver_binding_names"]
    result: dict[str, Any] = {}
    for name in required:
        if name in config.identity_fields:
            table, field_name = config.identity_fields[name]
            row = _navigate_to_table(fixture, table, database, config, plan=plan)
            if row is None:
                raise GenericV2BindingError(
                    f"cannot resolve {name!r}: no real {table!r} row navigable from the selected fixture"
                )
            value = _get_nested(row, field_name)
            if value is None:
                raise GenericV2BindingError(f"selected {table!r} row has no real {field_name!r}")
            result[name] = value
        elif name in config.dict_key_fields:
            table, dict_field = config.dict_key_fields[name]
            row = _navigate_to_table(fixture, table, database, config, plan=plan)
            if row is None:
                raise GenericV2BindingError(
                    f"cannot resolve {name!r}: no real {table!r} row navigable from the selected fixture"
                )
            keys = sorted((row.get(dict_field) or {}).keys())
            if not keys:
                raise GenericV2BindingError(f"selected {table!r} row has no real {dict_field!r} entries")
            result[name] = keys[0]
        elif name in config.free_form_placeholders:
            result[name] = config.free_form_placeholders[name]
        else:
            raise GenericV2BindingError(f"no resolution strategy for required driver binding {name!r}")
    return result


def resolve_known_fact_bundle(
    fixture: Fixture, database: Mapping[str, Any], config: DomainBindingConfig, *, plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Every real identity/dict-key field this domain knows how to resolve from the
    selected fixture -- not just the branch's own required_driver_binding_names.
    Step6's required_driver_binding_names is deliberately narrower: it only lists
    names the compiled ORACLE needs a specific, checkable value for (see
    v5_driver_plan_v2.py's binding_names, derived from runtime_observation_binding
    driver_bindings.* references) -- it was never meant to be "every argument the
    live tool call needs to succeed". A real live conversation still needs to answer
    an agent's other real questions (e.g. a customer's real full_name when only
    their dob is required_driver_binding-verified) -- this is the supplementary,
    best-effort real source for that (see docs/agentcoveragetesting_reuse_log.md
    section 63). A name whose table isn't reachable from this particular fixture is
    silently skipped (not every domain field applies to every branch), never
    guessed."""
    result: dict[str, Any] = {}
    # The anchor row's own real scalar fields first (e.g. a line's real
    # phone_number/status/contract_end_date -- fields with no curated identity_
    # fields name at all, but still real, still potentially what an agent asks
    # about). identity_fields' curated semantic names are applied after and take
    # priority on any name collision.
    if fixture.row:
        for key, value in fixture.row.items():
            if value is None or isinstance(value, (Mapping, list)):
                continue
            result[key] = value
    for name, (table, field_name) in config.identity_fields.items():
        row = _navigate_to_table(fixture, table, database, config, plan=plan)
        if row is None:
            continue
        value = _get_nested(row, field_name)
        if value is not None:
            result[name] = value
    for name, (table, dict_field) in config.dict_key_fields.items():
        row = _navigate_to_table(fixture, table, database, config, plan=plan)
        if row is None:
            continue
        keys = sorted((row.get(dict_field) or {}).keys())
        if keys:
            result[name] = keys[0]
    # A branch's real oracle check can require a real argument's shape (e.g. "limit
    # must be a real integer") without that name ever being required_driver_binding
    # -verified for THIS branch (e.g. telecom_011's own oracle needs a real integer
    # arguments.limit, but "limit" isn't in this branch's own required_driver_
    # binding_names at all) -- the domain's own free_form_placeholders are real,
    # schema-valid default values (verified against live tool schemas, see section
    # 55) already used whenever a name IS required; offering them here too, when not
    # already resolved above, costs nothing and covers this real gap (section 63).
    for name, value in config.free_form_placeholders.items():
        result.setdefault(name, value)
    # identity_binding_overrides (see DomainBindingConfig) take priority over every
    # source above -- including the anchor row's own real scalar fields loop at the
    # top of this function, which would otherwise silently set e.g. "user_id" to the
    # AUTHENTICATED user's own id before this ever runs. Scoped to this exact
    # branch_id, so every branch without an entry is unaffected.
    branch_id = plan.get("branch_id") if plan is not None else None
    for (override_branch_id, name), value in config.identity_binding_overrides.items():
        if override_branch_id == branch_id:
            result[name] = value
    return result


def bind_v2_branch(
    plan: Mapping[str, Any],
    database: Mapping[str, Any],
    config: DomainBindingConfig,
    *,
    allow_construction: bool = False,
    reference_time: str | None = None,
) -> dict[str, Any]:
    """Bind one real v2 driver plan end to end (Given evaluation + required driver
    binding resolution + canonical request), within this module's confirmed scope.
    Raises GenericV2BindingError, with a real reason, for anything outside that
    scope -- callers sweeping a corpus should catch it per branch and report the
    reason, not treat a raise as a crash. allow_construction defaults to False, so
    every existing caller keeps its exact original (real-row-only) behavior.
    reference_time (a real ISO "now" string, domain-supplied) enables real
    right.kind=="relative_test_time" ($NOW/$TODAY) comparisons; omitting it keeps
    the original honest "non-literal...not supported" behavior for those branches."""

    fixture = select_fixture(
        plan, database, config, allow_construction=allow_construction, reference_time=reference_time
    )
    driver_bindings = resolve_driver_bindings(plan, fixture, database, config)
    return {
        "branch_id": plan["branch_id"],
        "driver_plan_id": plan["driver_plan_id"],
        "fixture_source": fixture.fixture_source,
        "fixture_table": fixture.table,
        "fixture_row": fixture.row,
        "state_patch": fixture.state_patch,
        "driver_bindings": driver_bindings,
        "known_fact_bundle": resolve_known_fact_bundle(fixture, database, config, plan=plan),
        "canonical_request": plan["test_point"]["when"]["supplied_user_request"],
    }


def sweep_domain(
    plans: Sequence[Mapping[str, Any]],
    database: Mapping[str, Any],
    config: DomainBindingConfig,
    *,
    allow_construction: bool = False,
    reference_time: str | None = None,
) -> dict[str, Any]:
    """Real, honest corpus-wide coverage report: how many real branches bind within
    scope, and a categorized count of why each real failure happened."""

    bound = []
    failures: dict[str, int] = {}
    for plan in plans:
        try:
            bound.append(bind_v2_branch(
                plan, database, config, allow_construction=allow_construction, reference_time=reference_time
            ))
        except GenericV2BindingError as exc:
            key = str(exc).split(":")[0]
            failures[key] = failures.get(key, 0) + 1
    return {
        "domain": config.domain,
        "total": len(plans),
        "bound_count": len(bound),
        "bound": bound,
        "failure_count": sum(failures.values()),
        "failures": failures,
    }
