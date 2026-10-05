"""Retail-specific CREATE/UPDATE bundle construction for the v2 pipeline (see
docs/agentcoveragetesting_reuse_log.md section 58) -- real per-operation domain
modeling flagged as out of scope since sections 51-55 (a real order's own item_ids, a
real available alternative product variant for new_item_ids, a real address). Unlike
airline, retail had no existing v1 binder to reuse -- this is genuinely new, but
mirrors the same fallback pattern: the generic v2 binder is tried first and its
result returned unchanged for every branch that already binds that way; only a
branch whose required_driver_binding_names include a real bundle field AND that
fails the generic path falls through to real bundle construction here.

Real tau2 constraints verified against tau2/domains/retail/tools.py before writing
this (not guessed):
  - modify_pending_order_items / modify_pending_order_address both need a real order
    with status=="pending".
  - exchange_delivered_order_items / return_delivered_order_items need a real order
    with status=="delivered".
  - a new_item_id must be a real, available variant of the SAME product as the item
    it replaces (self._get_variant(product_id, new_item_id).available), found via
    the real products[product_id]["variants"] catalog.
  - return_delivered_order_items requires payment_method_id to be the order's own
    original payment method (or a gift card) -- using the order's real
    payment_history[0].payment_method_id (rather than an arbitrary key from the
    user's payment_methods dict) satisfies this for every one of these tools, not
    just return.

Multi-tool branches (docs/agentcoveragetesting_reuse_log.md section 62): a branch's
tool_names_from_effective_routes can list more than one real candidate tool -- real
data shows these are OR01/OR02/... ALTERNATE routes (only one is ever really
exercised in a live conversation), not a sequential requirement to satisfy every tool
at once. bind_retail_v2_branch tries each candidate tool in order (reusing the exact
same per-tool resolvers above, nothing new), and the first one that really resolves
becomes the "anchor" -- its values win on any field-name collision. Any
still-missing required field is filled from the next candidate tool's own
resolution. When a later candidate tool shares the anchor's real root entity (here,
"orders"), it is resolved against that SAME real order (not an independently
re-searched one) so e.g. an order's item-modification fields and its address fields
never come from two different real orders -- if the anchor's own order doesn't
satisfy that later tool's own real precondition (e.g. anchor is a pending order but
the later tool needs delivered), that field is honestly left unresolved rather than
silently substituting an inconsistent order.
"""

from __future__ import annotations

from typing import Any, Mapping

from .generic_tau_retail_v2_fixture_binder_v1 import RETAIL_CONFIG, usable_different_payment_method_ids
from .generic_tau_v2_fixture_binder_v1 import GenericV2BindingError, bind_v2_branch, select_fixture

_ITEM_TOOLS_BY_REQUIRED_STATUS = {
    "modify_pending_order_items": "pending",
    "exchange_delivered_order_items": "delivered",
    "return_delivered_order_items": "delivered",
}
_ADDRESS_FIELDS = ("address1", "address2", "city", "state", "zip", "country")


def _find_order_with_status(database: Mapping[str, Any], status: str) -> Mapping[str, Any]:
    orders = database.get("orders") or {}
    order = next((o for o in orders.values() if o.get("status") == status), None)
    if order is None:
        raise GenericV2BindingError(f"no real order with status {status!r} exists in the fixture db")
    return order


def _find_available_alternative_variant(database: Mapping[str, Any], product_id: str, item_id: str) -> str:
    product = (database.get("products") or {}).get(product_id)
    if product is None:
        raise GenericV2BindingError(f"no real product {product_id!r} found in the fixture db")
    variants = product.get("variants") or {}
    alt = next(
        (vid for vid, variant in variants.items() if vid != item_id and variant.get("available")), None
    )
    if alt is None:
        raise GenericV2BindingError(f"no real available alternative variant for product {product_id!r}")
    return alt


def _resolve_order_items_bundle_for_order(
    order: Mapping[str, Any], database: Mapping[str, Any], required_names: set[str]
) -> dict[str, Any]:
    items = order.get("items") or []
    if not items:
        raise GenericV2BindingError(f"order {order['order_id']!r} has no real items")
    item = items[0]
    result: dict[str, Any] = {"order_id": order["order_id"]}
    if "item_ids" in required_names:
        result["item_ids"] = [item["item_id"]]
    if "new_item_ids" in required_names:
        result["new_item_ids"] = [
            _find_available_alternative_variant(database, item["product_id"], item["item_id"])
        ]
    if "payment_method_id" in required_names:
        history = order.get("payment_history") or []
        if not history:
            raise GenericV2BindingError(f"order {order['order_id']!r} has no real payment_history")
        result["payment_method_id"] = history[0]["payment_method_id"]
    return result


def _resolve_order_items_bundle(
    tool_name: str, database: Mapping[str, Any], required_names: set[str]
) -> dict[str, Any]:
    status = _ITEM_TOOLS_BY_REQUIRED_STATUS[tool_name]
    order = _find_order_with_status(database, status)
    return _resolve_order_items_bundle_for_order(order, database, required_names)


def _resolve_order_address_bundle_for_order(
    order: Mapping[str, Any], required_names: set[str]
) -> dict[str, Any]:
    address = order.get("address")
    if not address:
        raise GenericV2BindingError(f"order {order['order_id']!r} has no real address")
    result = {name: address[name] for name in _ADDRESS_FIELDS if name in required_names and name in address}
    if "order_id" in required_names:
        result["order_id"] = order["order_id"]
    return result


def _resolve_order_address_bundle(database: Mapping[str, Any], required_names: set[str]) -> dict[str, Any]:
    # modify_pending_order_address's own real tau2 precondition is status=="pending"
    # (raises "Non-pending order cannot be modified" otherwise) -- the search must
    # honor that, not just "any order with an address".
    order = _find_order_with_status(database, "pending")
    return _resolve_order_address_bundle_for_order(order, required_names)


def _resolve_new_order_address_bundle_for_order(
    order: Mapping[str, Any], database: Mapping[str, Any], required_names: set[str]
) -> dict[str, Any]:
    """A real, DIFFERENT address to move the order to -- modify_pending_order_address
    literally means changing the shipping address; _resolve_order_address_bundle_
    for_order (used elsewhere for supplementary context about the order's own real
    current state) returning that SAME address as the operation's target is a real
    no-op a live agent correctly refuses (section 63, found via a real retail smoke
    run: retail_015's real request landed on an address identical to the order's own
    current one). Real, not fabricated: picks another real order's own real address
    that actually differs in address1, rather than inventing new values."""
    current = order.get("address") or {}
    orders = database.get("orders") or {}
    candidate = next(
        (
            o["address"] for o in orders.values()
            if o.get("address") and o["address"].get("address1") != current.get("address1")
        ),
        None,
    )
    if candidate is None:
        raise GenericV2BindingError("no real order has a different real address to move to")
    result = {name: candidate[name] for name in _ADDRESS_FIELDS if name in required_names and name in candidate}
    if "order_id" in required_names:
        result["order_id"] = order["order_id"]
    return result


def _resolve_new_order_address_bundle(database: Mapping[str, Any], required_names: set[str]) -> dict[str, Any]:
    order = _find_order_with_status(database, "pending")
    return _resolve_new_order_address_bundle_for_order(order, database, required_names)


def _resolve_user_address_bundle(database: Mapping[str, Any], required_names: set[str]) -> dict[str, Any]:
    users = database.get("users") or {}
    user = next((u for u in users.values() if u.get("address")), None)
    if user is None:
        raise GenericV2BindingError("no real user with a real address exists in the fixture db")
    current_address = user["address"]
    # docs/agentcoveragetesting_reuse_log.md section 137.5.1 (task_23086e09):
    # modify_user_address means CHANGING the user's default address to
    # something new -- binding the operation's own TARGET values to this
    # SAME user's CURRENT on-file address (as this resolver originally did,
    # returning `address` unchanged below) is a real, structural no-op: a
    # live agent correctly, reasonably declines to make a write that changes
    # nothing (real retail_046_arg#b0 transcript: the bound target address
    # was byte-for-byte identical to noah_brown_6181's real current address,
    # and the agent explicitly, correctly refused across 3 turns), and the
    # branch's own oracle check (a `modify_user_address` call is
    # `requires_observation: true`) then has no legitimate pass path for
    # that correct behavior -- it can only pass if the agent makes a
    # pointless write. This mirrors _resolve_new_order_address_bundle_for_
    # order's already-established fix for the exact same shape of bug on the
    # ORDER address side (section 63, retail_015) -- picks another real
    # user's own real address instead of inventing new values, so the
    # target stays a real, valid fixture value, just genuinely different
    # from what's already on file for the user actually being modified.
    other_address = next(
        (
            u["address"] for u in users.values()
            if u.get("address") and u["address"].get("address1") != current_address.get("address1")
        ),
        None,
    )
    if other_address is None:
        raise GenericV2BindingError("no real user has a different real address to move to")
    # docs/agentcoveragetesting_reuse_log.md section 74: mirror
    # _enrich_order_fact_bundle's own convention -- offer every real field
    # this address can supply, not just whatever this branch's own
    # required_driver_binding_names happened to gate (a real branch like
    # retail_046_arg#b0 needs `country`, which the row genuinely has, but
    # the compiler's own required set didn't include it; filtering here
    # silently dropped it instead of letting it enrich the supplementary
    # fact bundle). Callers still filter to `required` for driver_bindings,
    # so this only widens the SUPPLEMENTARY known_fact_bundle.
    result = {name: other_address[name] for name in _ADDRESS_FIELDS if name in other_address}
    result["user_id"] = user["user_id"]
    return result


_ITEM_FIELD_NAMES = {"item_ids", "new_item_ids", "payment_method_id"}
_ADDRESS_FIELD_NAMES = set(_ADDRESS_FIELDS) | {"order_id"}


def _enrich_order_fact_bundle(order: Mapping[str, Any], database: Mapping[str, Any], fact_bundle: dict[str, Any]) -> None:
    """Best-effort: offer every real item-bundle and address field this order can
    supply, not just whatever this branch's own required_driver_binding_names
    happened to gate (section 63 -- e.g. retail_004/005's real required set is
    item_ids/order_id/payment_method_id, missing new_item_ids entirely, even though
    exchange_delivered_order_items needs it to actually succeed; retail_016's is
    missing city). Never lets a failure here (e.g. no real available alternative
    variant) break the branch's already-resolved required driver_bindings -- this
    only enriches the SUPPLEMENTARY fact bundle, silently skipped on failure."""
    try:
        items = _resolve_order_items_bundle_for_order(order, database, _ITEM_FIELD_NAMES)
    except GenericV2BindingError:
        items = {}
    try:
        address = _resolve_order_address_bundle_for_order(order, _ADDRESS_FIELD_NAMES)
    except GenericV2BindingError:
        address = {}
    for key, value in {**items, **address}.items():
        fact_bundle.setdefault(key, value)


_ORDER_SCOPED_TOOLS = set(_ITEM_TOOLS_BY_REQUIRED_STATUS) | {"modify_pending_order_address"}


def _resolve_single_tool_bundle(
    tool_name: str,
    database: Mapping[str, Any],
    required_names: set[str],
    *,
    anchor_order: Mapping[str, Any] | None,
    bypass_status_guard: bool = False,
) -> dict[str, Any]:
    if tool_name in _ITEM_TOOLS_BY_REQUIRED_STATUS:
        if anchor_order is not None:
            status = _ITEM_TOOLS_BY_REQUIRED_STATUS[tool_name]
            if not bypass_status_guard and anchor_order.get("status") != status:
                raise GenericV2BindingError(
                    f"anchor order {anchor_order.get('order_id')!r} does not satisfy "
                    f"{tool_name!r}'s own real status requirement ({status!r})"
                )
            return _resolve_order_items_bundle_for_order(anchor_order, database, required_names)
        return _resolve_order_items_bundle(tool_name, database, required_names)
    if tool_name == "modify_pending_order_address":
        if anchor_order is not None:
            if not bypass_status_guard and anchor_order.get("status") != "pending":
                raise GenericV2BindingError(
                    f"anchor order {anchor_order.get('order_id')!r} does not satisfy "
                    f"{tool_name!r}'s own real status requirement ('pending')"
                )
            return _resolve_new_order_address_bundle_for_order(anchor_order, database, required_names)
        return _resolve_new_order_address_bundle(database, required_names)
    if tool_name == "modify_user_address":
        return _resolve_user_address_bundle(database, required_names)
    raise GenericV2BindingError(f"no bundle resolution strategy for tool {tool_name!r}")


def _apply_identity_binding_overrides(plan: Mapping[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """docs/agentcoveragetesting_reuse_log.md section 128: apply RETAIL_CONFIG.
    identity_binding_overrides (see generic_tau_v2_fixture_binder_v1.py's
    DomainBindingConfig docstring) as the real LAST step before this function
    returns, regardless of which internal path (generic success or bundle
    construction) produced `result` -- a direct override (not setdefault),
    since bundle construction's own per-tool resolvers (e.g.
    _resolve_user_address_bundle, used for retail_096_state#b1's real target
    tool modify_user_address) already populate a real "user_id" key in the
    fact bundle themselves (the FIRST real user with a real address --
    coincidentally the correct AUTHENTICATED identity for this branch, but
    never the SECOND, adversarial one this branch's own known_fact_bundle
    needs to give the live dialogue driver something concrete to reference).
    Scoped to (branch_id, name) via RETAIL_CONFIG, so this is a real no-op for
    every branch without an entry -- verified by a full-corpus diff, section
    128."""
    overrides = RETAIL_CONFIG.identity_binding_overrides
    if not overrides:
        return result
    branch_id = plan.get("branch_id")
    fact_bundle = dict(result.get("known_fact_bundle") or {})
    driver_bindings = dict(result.get("driver_bindings") or {})
    changed = False
    for (override_branch_id, name), value in overrides.items():
        if override_branch_id != branch_id:
            continue
        fact_bundle[name] = value
        changed = True
        if name in driver_bindings:
            driver_bindings[name] = value
    if not changed:
        return result
    return {**result, "known_fact_bundle": fact_bundle, "driver_bindings": driver_bindings}


def _conversation_precondition(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """The branch's single real Step6 conversation precondition and the real
    Given non_database_condition it links to (realizable=="conversation").
    Raises (never guesses) when the plan does not have exactly that shape."""
    preconditions = ((plan.get("interaction_plan") or {}).get("user_requirements") or {}).get(
        "conversation_preconditions"
    ) or []
    if len(preconditions) != 1:
        raise GenericV2BindingError(
            f"{plan.get('branch_id')}: expected exactly one conversation precondition, found {len(preconditions)}"
        )
    precondition = preconditions[0]
    links = set(precondition.get("non_db_links") or [])
    clauses = [
        c for c in (plan["test_point"]["given"].get("non_database_conditions") or [])
        if c.get("condition_id") in links and c["source_condition"].get("realizable") == "conversation"
    ]
    if len(clauses) != 1:
        raise GenericV2BindingError(
            f"{plan.get('branch_id')}: conversation precondition does not link to exactly one "
            "realizable=='conversation' Given clause"
        )
    return precondition, clauses[0]


def _realize_split_across_different_payment_methods(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """docs/agentcoveragetesting_reuse_log.md section 182 (task_e9deccb5):
    retail_023_arg#e1's Given clause "the user chooses more than one payment
    method, all different from the original payment method" is
    realizable=="conversation" -- no db field encodes it, so the user has to
    SAY it. Grounded only in real bound data: the bound order (anchored by
    RETAIL_CONFIG's _order_owner_has_two_usable_different_payment_methods
    tie-break), its real original payment method, and the first two of the
    owner's real usable different methods -- the first of which must be the
    bound payment_method_id itself (the oracle's absent-check scope), so the
    single-method call a non-compliant agent would most likely make is exactly
    the one the check observes. Raises rather than fabricating when the bound
    order cannot support the clause."""
    precondition, clause = _conversation_precondition(plan)
    bindings = result.get("driver_bindings") or {}
    order = (database.get("orders") or {}).get(bindings.get("order_id"))
    if order is None:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound order_id is not a real order")
    original = ((order.get("payment_history") or [{}])[0]).get("payment_method_id")
    chosen = usable_different_payment_method_ids(order, database)[:2]
    if len(chosen) < 2 or chosen[0] != bindings.get("payment_method_id"):
        raise GenericV2BindingError(
            f"{plan.get('branch_id')}: bound order {order.get('order_id')} cannot realize "
            f"'{clause['source_condition']['clause']}' (usable different methods: {chosen})"
        )
    first, second = chosen
    statement = (
        f"Specifically, for order {order['order_id']} I'd like to split the payment across two of my "
        f"payment methods: {first} and {second} -- both are different from the payment method the order "
        f"was originally paid with ({original})."
    )
    return [{
        "source_ref": dict(precondition.get("source_ref") or {}),
        "non_db_links": list(precondition.get("non_db_links") or []),
        "condition": precondition["requirement"]["condition"],
        "given_clause": clause["source_condition"]["clause"],
        "realization": "user_statement",
        "statement": statement,
        "stated_values": {"payment_method_id": [first, second]},
    }]


# docs/agentcoveragetesting_reuse_log.md section 182 (task_e9deccb5): real
# plan["branch_id"] -> realizer for a Given clause the scripted user can only
# realize by SAYING it (realizable=="conversation" and not already carried by
# the request text or by a bound value the facts message states). A real,
# all-domain scan of the 42 branches with such clauses found this to be the
# only retail one (see the reuse log section 182.1 table); every other retail
# conversation clause is already expressed by the request itself or by a bound
# value. The realizer's output travels in bind_result["conversation_
# requirements"] -> the bound plan's dialogue_contract.conversation_requirements
# (generic_tau_v2_bound_plan_adapter_v1.py, emitted only when non-empty) ->
# GenericDeterministicTauUser (generic_tau_online_v1.py). Every branch without
# an entry is byte-identical to before.
_CONVERSATION_STATEMENT_REALIZERS = {
    "retail_023_arg#e1": _realize_split_across_different_payment_methods,
}


_ITEM_REQUEST_VERBS = {
    "exchange_delivered_order_items": "exchange item {old} for item {new}",
    "modify_pending_order_items": "change item {old} to item {new}",
}


def _rebind_requested_item_not_in_order(
    plan: Mapping[str, Any], result: Mapping[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    """docs/agentcoveragetesting_reuse_log.md section 186 (task_405df040):
    retail_078_state#b0 ("the item specified for exchange does not exist in the
    delivered order") and retail_089_state#b0 ("the item does not exist in the
    specified order") have a realizable=="conversation" Given clause that the
    bound data used to state BACKWARDS -- _enrich_order_fact_bundle always puts
    the anchor order's own first item into item_ids, i.e. an item that IS in
    the order, so the scripted user asked about an item it really has and the
    item-existence check passed vacuously on a successful call.

    Rebinds item_ids to a real catalog item that is genuinely NOT in the bound
    order: the first (sorted) real variant of the SAME product as the order's
    own bound item that no line of this order holds and that is not the bound
    replacement (new_item_ids, the oracle's scope value, is kept unchanged, so
    "swap item X for item Y" stays a same-product request). Also states the
    request as a user statement (section 182 channel) so the agent is asked
    about exactly that item. Raises rather than fabricating when the bound data
    cannot support the clause."""
    precondition, clause = _conversation_precondition(plan)
    bindings = dict(result.get("driver_bindings") or {})
    facts = dict(result.get("known_fact_bundle") or {})
    order_id = bindings.get("order_id") or facts.get("order_id")
    order = (database.get("orders") or {}).get(order_id)
    tools = plan["observation_plan"].get("tool_names_from_effective_routes") or []
    tool = next((name for name in tools if name in _ITEM_REQUEST_VERBS), None)
    if order is None or tool is None:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: no real order/item tool to realize {clause['source_condition']['clause']!r}")
    in_order = {item.get("item_id") for item in order.get("items") or []}
    bound_items = list(facts.get("item_ids") or [])
    new_items = list(bindings.get("new_item_ids") or facts.get("new_item_ids") or [])
    anchor_item = next((item for item in order.get("items") or [] if bound_items and item.get("item_id") == bound_items[0]), None)
    if anchor_item is None or len(new_items) != 1:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound item_ids/new_item_ids are not a real single-item request on {order_id}")
    variants = ((database.get("products") or {}).get(anchor_item.get("product_id")) or {}).get("variants") or {}
    if new_items[0] not in variants:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: bound new item is not a variant of the bound item's product")
    candidates = sorted(v for v in variants if v not in in_order and v not in new_items)
    if not candidates:
        raise GenericV2BindingError(f"{plan.get('branch_id')}: no real variant of {anchor_item.get('product_id')} is absent from {order_id}")
    missing_item = candidates[0]
    facts["item_ids"] = [missing_item]
    if "item_ids" in bindings:
        bindings["item_ids"] = [missing_item]
    statement = "Specifically, for order {order} I'd like to {request}.".format(
        order=order_id, request=_ITEM_REQUEST_VERBS[tool].format(old=missing_item, new=new_items[0])
    )
    return {
        **result,
        "driver_bindings": bindings,
        "known_fact_bundle": facts,
        "conversation_requirements": [{
            "source_ref": dict(precondition.get("source_ref") or {}),
            "non_db_links": list(precondition.get("non_db_links") or []),
            "condition": precondition["requirement"]["condition"],
            "given_clause": clause["source_condition"]["clause"],
            "realization": "user_statement",
            "statement": statement,
            "stated_values": {},
        }],
    }


# docs/agentcoveragetesting_reuse_log.md section 186: branch_id -> a realizer that
# must REBIND a value (not only add a statement) for a conversation-only Given
# clause to hold. Returns the whole updated bind result (incl. its own
# conversation_requirements). Every branch without an entry is unchanged.
_CONVERSATION_CLAUSE_REBINDERS = {
    "retail_078_state#b0": _rebind_requested_item_not_in_order,
    "retail_089_state#b0": _rebind_requested_item_not_in_order,
}


def _apply_conversation_statement_realizers(
    plan: Mapping[str, Any], result: dict[str, Any], database: Mapping[str, Any]
) -> dict[str, Any]:
    rebinder = _CONVERSATION_CLAUSE_REBINDERS.get(plan.get("branch_id"))
    if rebinder is not None:
        return rebinder(plan, result, database)
    realizer = _CONVERSATION_STATEMENT_REALIZERS.get(plan.get("branch_id"))
    if realizer is None:
        return result
    return {**result, "conversation_requirements": realizer(plan, result, database)}


def bind_retail_v2_branch(plan: Mapping[str, Any], database: Mapping[str, Any]) -> dict[str, Any]:
    """Real, thin wrapper: bind via _bind_retail_v2_branch_core, then apply this
    domain's real identity_binding_overrides (section 128) as the true last step
    before returning -- see _apply_identity_binding_overrides' own docstring.
    section 182: then realize any conversation-only Given clause this branch
    opts into (_CONVERSATION_STATEMENT_REALIZERS), from the final bindings."""
    result = _apply_identity_binding_overrides(plan, _bind_retail_v2_branch_core(plan, database))
    return _apply_conversation_statement_realizers(plan, result, database)


def _bind_retail_v2_branch_core(plan: Mapping[str, Any], database: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one real retail v2 branch: try the generic path first (unaffected, zero
    regression for every already-working branch), and only on a generic failure fall
    back to real bundle construction.

    A branch's real tool_names_from_effective_routes can list more than one
    candidate tool (section 62's real OR-route finding -- only one is ever really
    exercised at runtime). Try each in the given order; the first that really
    resolves is the "anchor" (its values win on a field-name collision, and its
    fixture_source names it). Any still-missing required field is filled from a
    later candidate tool's own resolution -- sharing the anchor's real order when
    that later tool is also order-scoped, so item/address fields for the same
    branch are never sourced from two different real orders."""

    try:
        result = bind_v2_branch(plan, database, RETAIL_CONFIG, allow_construction=True)
    except GenericV2BindingError as generic_error:
        generic_message = str(generic_error)
    else:
        # The generic path's own resolve_known_fact_bundle only knows RETAIL_
        # CONFIG's domain-agnostic identity_fields/dict_key_fields (order_id, user
        # identity, payment_method_id) -- never retail's own item-bundle/address
        # domain knowledge (a real available alternative variant, a real shipping
        # address), even when the branch's own real anchor IS an order (section 63,
        # found via a real retail smoke run: retail_005 took this generic path,
        # since order_id+payment_method_id alone are generically resolvable, but its
        # real fact bundle then had no item_ids/new_item_ids/address at all).
        order_id = result["driver_bindings"].get("order_id") or (result.get("known_fact_bundle") or {}).get("order_id")
        order = (database.get("orders") or {}).get(order_id) if order_id else None
        if order is not None:
            fact_bundle = dict(result.get("known_fact_bundle") or {})
            _enrich_order_fact_bundle(order, database, fact_bundle)
            # docs/agentcoveragetesting_reuse_log.md section 116: _enrich_order_
            # fact_bundle above always supplies the order's OWN CURRENT address
            # (real, supplementary context about the order's existing state -- see
            # its own docstring) -- but when the branch's real target tool is
            # modify_pending_order_address, that same address is also what the
            # fact bundle hands the agent as the "new" address to move the order
            # to, a real no-op the agent then correctly asks to be corrected
            # instead of ever calling the tool (the exact bug _resolve_new_order_
            # address_bundle_for_order was already written for -- section 63's
            # retail_015 smoke-test finding -- but only wired into the bundle-
            # CONSTRUCTION fallback below, never into this generic-path success
            # branch, which retail_088_state#b0 reaches once its own order is a
            # real pending one). Override with a genuinely different real address
            # (never a fabricated one -- see that helper's own docstring); best-
            # effort, silently skipped on failure like every other enrichment
            # here, never touches the already-resolved required driver_bindings.
            if "modify_pending_order_address" in (
                plan["observation_plan"].get("tool_names_from_effective_routes") or []
            ):
                try:
                    new_address = _resolve_new_order_address_bundle_for_order(
                        order, database, _ADDRESS_FIELD_NAMES
                    )
                except GenericV2BindingError:
                    new_address = {}
                for key, value in new_address.items():
                    if key != "order_id":
                        fact_bundle[key] = value
            result = {**result, "known_fact_bundle": fact_bundle}
        return result

    tool_names = plan["observation_plan"].get("tool_names_from_effective_routes") or []
    required = set(plan["fixture_binding_plan"]["required_driver_binding_names"])
    if not tool_names:
        raise GenericV2BindingError(generic_message)

    # docs/agentcoveragetesting_reuse_log.md section 72: a branch with a real,
    # explicit Given condition on orders.status (e.g. "the order is
    # cancelled") was previously ignored on this bundle-construction path --
    # _find_order_with_status below picks a real order by the TARGET TOOL's
    # own success precondition (e.g. "delivered" for exchange), which is the
    # opposite of what a real negative _state branch's Given asks for, and
    # any real state_patch select_fixture would have constructed was
    # discarded outright (hardcoded state_patch=None below). Only override
    # the tool-driven anchor when select_fixture found a genuine
    # condition-matched/constructed orders row (fixture_source
    # "database_row:orders"/"constructed_via_state_patch:orders") -- NOT its
    # "unconstrained" fallback, which would silently change the anchor for
    # every branch with no real Given condition at all.
    given_anchor_order: Mapping[str, Any] | None = None
    given_state_patch: Mapping[str, Any] | None = None
    # docs/agentcoveragetesting_reuse_log.md section 128: a branch whose Given
    # is a real owner/not_owner RELATION condition (fixture_source "relation")
    # was ALSO being silently discarded here, same real bug class as section
    # 72 above but for a different fixture_source -- found via
    # retail_081/088/092/099_state#b1 (real adversarial "must deny" siblings
    # whose Given is "the order does NOT belong to the authenticated user",
    # relation "not_owner"): their own real per-tool bundle resolvers below
    # re-derive an UNRELATED anchor order from scratch (whichever real order
    # happens to satisfy the target tool's own status precondition first),
    # discarding the real not_owner pair select_fixture/select_relation_pair
    # already found -- and since the identity block further below then reads
    # that unrelated anchor order's OWN real owner as "the authenticated
    # user", the resulting fixture was silently NOT adversarial at all (the
    # "authenticated user" ended up being the very same real owner as the
    # order being acted on). Unlike the section 72 case, a "relation" Fixture's
    # own .row is a real USERS row (the relation's parent), not an orders row
    # -- the real anchor order has to be looked up via its
    # .relation_child_id -- and the real identity for the fact bundle has to
    # come from that SAME parent row, not re-derived from the anchor order's
    # owner (see given_identity_user below), or the not_owner condition would
    # be silently re-satisfied.
    #
    # Real, found-not-assumed scoping note: honoring "relation" here
    # unconditionally is NOT a no-op for every existing branch -- section 128
    # verified via a real full-corpus diff that a real, pre-existing,
    # ALREADY-SHIPPED branch (retail_080_state#b2, whose own Given is a real
    # not_owner claim too, about payment_method_id rather than order
    # ownership, and which also reaches this bundle-construction fallback for
    # exchange_delivered_order_items) would change its own already-shipped
    # operation_argument_fact_bundle if this were applied to it too, and
    # explicitly scoped this allow-list to just that task's own 4 new
    # branch_ids (the 5th, retail_094_state#b1, binds via the plain generic
    # path, never reaches here at all), leaving retail_080_state#b2's own
    # real instance of the same bug for a separate, dedicated follow-up
    # (task_114c7458) rather than fixing it as an unreviewed side effect.
    #
    # docs/agentcoveragetesting_reuse_log.md section 129: that follow-up
    # re-derived retail_080_state#b2's real fixture_source via a live
    # select_fixture probe against production tau2 db.json and confirmed it
    # DOES hit this exact "relation" branch of code -- fixture_source ==
    # "relation", row.user_id == "noah_brown_6181" (the real not_owner
    # parent, via the SAME users->orders ownership edge the 4 branches above
    # use), relation_child_id == "#W4817420" -- and that, unmodified, this
    # bundle-construction fallback silently re-derives identity from
    # #W4817420's own real owner (ava_moore_2033), making the Given's
    # not_owner claim trivially false (payment_method_id belonged to the very
    # "authenticated user" the fixture manufactured) -- a real instance of
    # the same underlying bug.
    #
    # section 129 DID first try simply adding "retail_080_state#b2" here (the
    # same fix as the 4 branches above) and found, via a real online rerun,
    # that it is the WRONG fix for this specific branch's own claim shape:
    # unlike the 4 branches above (whose real Given claim IS "the order does
    # not belong to the authenticated user" -- the anchor order and the
    # not-owned entity are the SAME thing), retail_080_state#b2's real Given
    # claim is about payment_method_id, not the order being exchanged -- its
    # real When ("the user requests to exchange items in A delivered order")
    # never claims that order is unowned by the authenticated user; the
    # order should be a real, legitimately-owned one so the exchange itself
    # is actionable, and only the REFERENCED payment_method_id should be the
    # adversarial, not-owned value. Anchoring the exchanged order itself to
    # the users->orders not_owner pair (as this branch's own code below does
    # for the 4 real order-ownership branches) makes retail_080_state#b2's
    # order ALSO genuinely not the authenticated user's own -- confirmed via
    # a real online rerun: the agent correctly, immediately refused the
    # whole request ("order #W4817420 is associated with a different user
    # account"), never reaching the payment-method question this branch
    # exists to test at all (a real, different, newly-introduced vacuous-
    # pass/confound bug, not a fix). retail_080_state#b2 is deliberately kept
    # OUT of this allow-list; its real fix (a targeted identity_binding_
    # overrides entry for just its own payment_method_id, leaving its
    # legitimately-owned anchor order/identity alone) lives in
    # generic_tau_retail_v2_fixture_binder_v1.py's RETAIL_CONFIG instead --
    # see that file's own comment for the real reasoning.
    _RELATION_GIVEN_BUNDLE_BRANCH_IDS = frozenset({
        "retail_081_state#b1", "retail_088_state#b1",
        "retail_092_state#b1", "retail_099_state#b1",
    })
    given_identity_user: Mapping[str, Any] | None = None
    try:
        given_fixture = select_fixture(plan, database, RETAIL_CONFIG, allow_construction=True)
    except GenericV2BindingError:
        given_fixture = None
    if given_fixture is not None and given_fixture.fixture_source in (
        "database_row:orders", "constructed_via_state_patch:orders",
    ):
        given_anchor_order = given_fixture.row
        given_state_patch = given_fixture.state_patch
    elif (
        given_fixture is not None
        and given_fixture.fixture_source == "relation"
        and plan.get("branch_id") in _RELATION_GIVEN_BUNDLE_BRANCH_IDS
    ):
        given_anchor_order = (database.get("orders") or {}).get(given_fixture.relation_child_id)
        given_identity_user = given_fixture.row
        given_state_patch = given_fixture.state_patch

    orders = database.get("orders") or {}
    full: dict[str, Any] = {}
    anchor_tool: str | None = None
    anchor_order: Mapping[str, Any] | None = given_anchor_order
    for tool_name in tool_names:
        if required <= set(full):
            break
        share_anchor = anchor_order is not None and tool_name in _ORDER_SCOPED_TOOLS
        try:
            resolved = _resolve_single_tool_bundle(
                tool_name, database, required,
                anchor_order=anchor_order if share_anchor else None,
                bypass_status_guard=given_anchor_order is not None,
            )
        except GenericV2BindingError:
            continue
        if anchor_tool is None:
            anchor_tool = tool_name
            if tool_name in _ORDER_SCOPED_TOOLS and anchor_order is None:
                anchor_order = orders.get(resolved.get("order_id"))
        for name, value in resolved.items():
            full.setdefault(name, value)

    if anchor_tool is None:
        raise GenericV2BindingError(generic_message)
    missing = required - set(full)
    if missing:
        raise GenericV2BindingError(
            f"bundle resolution for {sorted(tool_names)} does not cover required binding(s) {sorted(missing)}"
        )
    driver_bindings = {name: full[name] for name in required}
    fact_bundle = dict(full)
    if anchor_order is not None:
        _enrich_order_fact_bundle(anchor_order, database, fact_bundle)
    # Real retail policy.md: every operation needs the user authenticated FIRST via
    # email or name+zip -- but none of the per-tool resolvers above ever look up the
    # order's own owning user at all (they only resolve the specific fields THIS
    # tool's own arguments need), so a bundle-constructed branch's fact bundle had
    # no identity/verification fields whatsoever (section 63, found via a real
    # retail smoke run where the agent got stuck asking for identity forever).
    # docs/agentcoveragetesting_reuse_log.md section 74: this was gated on
    # anchor_order (an ORDER_SCOPED tool), so a user-anchored branch (e.g.
    # modify_user_address, whose resolver already found a real user directly)
    # never got this enrichment even though the real user was already in
    # hand -- real branches retail_046/065/096 got stuck the same way for
    # the same underlying reason. Fall back to the user _resolve_user_
    # address_bundle already resolved (via its own real user_id in `full`)
    # when there is no order anchor. docs/agentcoveragetesting_reuse_log.md
    # section 128: given_identity_user (set above, real "relation" Given
    # only) takes priority over anchor_order's own owner -- see that block's
    # own comment for why re-deriving identity from the anchor order's owner
    # is wrong for a real not_owner Given.
    if given_identity_user is not None:
        identity_user_id = given_identity_user.get("user_id")
    else:
        identity_user_id = anchor_order.get("user_id") if anchor_order is not None else full.get("user_id")
    if identity_user_id is not None:
        user = (database.get("users") or {}).get(identity_user_id)
        if user is not None:
            identity = {
                "user_id": user.get("user_id"),
                "email": user.get("email"),
                "first_name": (user.get("name") or {}).get("first_name"),
                "last_name": (user.get("name") or {}).get("last_name"),
                "zip": (user.get("address") or {}).get("zip"),
            }
            for key, value in identity.items():
                if value is not None:
                    fact_bundle.setdefault(key, value)
    return {
        "branch_id": plan["branch_id"],
        "driver_plan_id": plan["driver_plan_id"],
        "fixture_source": f"bundle_constructed:{anchor_tool}",
        "fixture_table": "orders" if given_anchor_order is not None else None,
        "fixture_row": given_anchor_order,
        "state_patch": given_state_patch,
        "driver_bindings": driver_bindings,
        # section 63: real supplementary facts beyond required_driver_binding_names
        # -- includes the order's own real owning user's identity fields (above),
        # not just whatever the per-tool resolvers filtered to required_names.
        "known_fact_bundle": fact_bundle,
        "canonical_request": plan["test_point"]["when"]["supplied_user_request"],
    }
