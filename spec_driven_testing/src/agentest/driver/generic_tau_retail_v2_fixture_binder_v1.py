"""Retail's real DomainBindingConfig for the shared v2 Step7 fixture-binder core
(generic_tau_v2_fixture_binder_v1.py -- see docs/agentcoveragetesting_reuse_log.md
section 55). Retail's v2 pipeline had no Step7 binder at all before this.

Real schema facts this config encodes: like airline, retail's tables (users/orders)
are JSON dicts keyed by id, and order ownership is a direct foreign key
(orders[id].user_id). first_name/last_name/zip are real but NESTED fields
(users[id].name.first_name, users[id].address.zip), resolved via identity_fields'
dotted-path support. payment_method_id is a real key into a user's own
payment_methods dict, like airline's payment_id.

Real coverage is expected to be well below telecom's 83%, for the same reason as
airline: many required_driver_binding_names combos are real order-modification
bundles (item_ids/new_item_ids to exchange, a full mailing address to update) that
need real per-operation construction, not a generic identity lookup.
"""

from __future__ import annotations

from typing import Any, Mapping

from .generic_tau_v2_fixture_binder_v1 import DomainBindingConfig, RelationConfig

_OWNERSHIP = RelationConfig(
    mechanism="direct_fk",
    parent_table="users",
    child_table="orders",
    parent_id_field="user_id",
    child_id_field="order_id",
    fk_field="user_id",
)


def _payment_method_changeable_order(order: Mapping[str, Any], database: Mapping[str, Any]) -> bool:
    """Real tau2 tools.py (modify_pending_order_payment, section 100/101): raises
    unless order.status=="pending", has exactly one real payment_history entry of
    transaction_type=="payment", and the NEW payment_method_id differs from the
    order's own current one. This tool had no operation_row_preferences entry at
    all before -- the unconstrained path fell through to `rows[0]`, the fixture db's
    literal first order (real bug found in retail_055_order_modify_pending_order_
    payment#b0, section 98.3/100: status=="processed", not "pending").

    RETAIL_CONFIG.dict_key_fields["payment_method_id"] always resolves to the
    owning user's alphabetically-first payment_methods key (generic mechanism, no
    "must differ from current" awareness) -- so this predicate only needs to
    confirm that alphabetically-first key genuinely differs from the order's own
    current payment_method_id; the existing generic resolution then does the right
    thing with no further changes needed. Verified against the real shipped fixture
    db: real orders exist that satisfy this (e.g. #W6779827/ethan_lopez_6291, whose
    real second payment method is a real credit card, sidestepping the tool's gift-
    card-balance check too), not a data limitation."""
    if order.get("status") != "pending":
        return False
    history = order.get("payment_history") or []
    if len(history) != 1 or history[0].get("transaction_type") != "payment":
        return False
    user = (database.get("users") or {}).get(order.get("user_id"))
    methods = (user or {}).get("payment_methods") or {}
    if not methods:
        return False
    first_key = sorted(methods.keys())[0]
    return first_key != history[0].get("payment_method_id")


def _order_owner_has_usable_different_payment_method(
    order: Mapping[str, Any], database: Mapping[str, Any]
) -> bool:
    """docs/agentcoveragetesting_reuse_log.md section 177 (task_c31f4cc1): the
    database_row_tiebreak_selectors predicate for retail's modify_pending_order_
    payment branches whose Given is a real orders.status database condition. Unlike
    _payment_method_changeable_order (above, used on the unconstrained/relation
    paths), this does NOT require status=="pending" itself -- it only ever runs as a
    tie-break among rows that already satisfy the branch's OWN Given status
    condition, which for the negative branches is deliberately NOT pending
    (processed/cancelled/delivered). It confirms the payment-method half of the
    tool's real preconditions (tau2 tools.py modify_pending_order_payment):

      - the order has a real original payment (payment_history[0] is a "payment");
      - the owner's alphabetically-first payment_methods key -- exactly what
        RETAIL_CONFIG.dict_key_fields["payment_method_id"] resolves the bound target
        to -- genuinely differs from that original payment method, so the bound
        target is a real, different, owner-held method (never fabricated);
      - if that target is a real gift card, its real balance covers the order's
        real payment amount (tools.py's "Insufficient gift card balance" check);
      - if the order is pending, it has exactly one real payment_history entry
        (tools.py's "There should be exactly one payment for a pending order").

    Before this, every such branch landed on the fixture db's literal first row for
    its status (#W5918442/sofia_rossi_8776, #W2611340/james_li_5688, #W3220387/
    amelia_silva_5103), each of whose owners holds exactly ONE payment method -- the
    order's own current one -- so the bound target was a guaranteed "The new
    payment method should be different from the current one" no-op."""
    history = order.get("payment_history") or []
    if not history or history[0].get("transaction_type") != "payment":
        return False
    if order.get("status") == "pending" and len(history) != 1:
        return False
    user = (database.get("users") or {}).get(order.get("user_id"))
    methods = (user or {}).get("payment_methods") or {}
    if not methods:
        return False
    first_key = sorted(methods.keys())[0]
    if first_key == history[0].get("payment_method_id"):
        return False
    target = methods[first_key] or {}
    if target.get("source") == "gift_card" and (target.get("balance") or 0) < (history[0].get("amount") or 0):
        return False
    return True


def usable_different_payment_method_ids(order: Mapping[str, Any], database: Mapping[str, Any]) -> list[str]:
    """docs/agentcoveragetesting_reuse_log.md section 182 (task_e9deccb5): the
    order owner's real payment_methods keys, in the same alphabetical order
    RETAIL_CONFIG.dict_key_fields["payment_method_id"] resolves from, that are
    (a) genuinely different from the order's original payment method
    (payment_history[0]) and (b) able to pay for the order on their own (a real
    gift card must cover payment_history[0].amount -- tau2 tools.py's
    "Insufficient gift card balance" check). Empty when the order has no
    original payment. Never invents an id: every entry is a real key of the
    owner's own real payment_methods dict."""
    history = order.get("payment_history") or []
    if not history or history[0].get("transaction_type") != "payment":
        return []
    user = (database.get("users") or {}).get(order.get("user_id"))
    methods = (user or {}).get("payment_methods") or {}
    original = history[0].get("payment_method_id")
    amount = history[0].get("amount") or 0
    result = []
    for key in sorted(methods):
        method = methods[key] or {}
        if key == original:
            continue
        if method.get("source") == "gift_card" and (method.get("balance") or 0) < amount:
            continue
        result.append(key)
    return result


def _order_owner_has_two_usable_different_payment_methods(
    order: Mapping[str, Any], database: Mapping[str, Any]
) -> bool:
    """docs/agentcoveragetesting_reuse_log.md section 182 (task_e9deccb5): the
    database_row_tiebreak_selectors predicate for retail_023_arg#e1, whose Given
    is "the order status is pending, but the user chooses MORE THAN ONE payment
    method, all different from the original payment method". Everything
    _order_owner_has_usable_different_payment_method (section 177) requires --
    in particular that the bound payment_method_id (the owner's alphabetically
    first key) is itself a usable, different method -- plus a SECOND usable
    different method, so the scripted user can truthfully ask to split the
    payment across two real methods it actually owns. Before this, the branch
    landed on #W5918442/sofia_rossi_8776, whose owner holds exactly one method
    (the order's own current one), so the "more than one, all different"
    clause was unrealizable and a compliant refusal ("you have no other
    method") passed vacuously."""
    if not _order_owner_has_usable_different_payment_method(order, database):
        return False
    return len(usable_different_payment_method_ids(order, database)) >= 2


# docs/agentcoveragetesting_reuse_log.md section 177 (task_c31f4cc1): the retail
# modify_pending_order_payment branches (Given = a real orders.status database
# condition, required_driver_binding_names include payment_method_id) whose own
# Given/Then need the bound payment_method_id to be a REAL, DIFFERENT method from the
# anchor order's current one -- per-branch triage (real Given/When/Then + real
# checks + real db.json):
#   - retail_067_state_modify_pending_order_payment#b0: Given "pending", Then "may use
#     modify_pending_order_payment" -- the positive path itself (a real defect).
#   - retail_023_arg#e0: Given "NOT pending, but the user chooses a single payment
#     method DIFFERENT from the original" -- the conversation clause is realized only
#     through the bound payment_method_id the driver states, so it must differ.
#   - retail_067_state_modify_pending_order_payment#e0 / retail_102_state#b2 /
#     retail_104_state#b0v2: non-pending negatives whose OR01 absent check is scoped to
#     arguments.payment_method_id == the bound one; with the current method bound, a
#     compliant agent's refusal was confounded ("that is already your payment
#     method") rather than resting only on the status rule each branch tests.
# Deliberately NOT listed (keep their current fixture; see section 177.1):
# retail_023_arg#e2 (Given: the chosen method is "NOT different"), retail_023_arg#b0v1
# and retail_044_arg#b0 (Then/checks target the same-method request itself),
# retail_020_arg#b0v5 (no payment-change target; payment_method_id is only the item
# price-difference method), retail_067_state_modify_pending_order_payment#b1 (Given
# "delivered" -- the tool's "Non-pending order" check fires first, independent of the
# payment method). retail_023_arg#e1 (its "more than one method" clause) was also
# left out here in section 177 because the deterministic driver could not state it;
# section 182 (task_e9deccb5) gives it its own, stricter selector below
# (_TWO_DIFFERENT_PAYMENT_METHODS_BRANCH_IDS) together with a conversation-statement
# realizer in generic_tau_retail_v2_bundle_resolver_v1.py -- a lone different method
# would still turn a compliant single-method change into a false fail.
_PAYMENT_METHOD_DIFFERS_BRANCH_IDS = (
    "retail_067_state_modify_pending_order_payment#b0",
    "retail_023_arg#e0",
    "retail_067_state_modify_pending_order_payment#e0",
    "retail_102_state#b2",
    "retail_104_state#b0v2",
    # docs/agentcoveragetesting_reuse_log.md section 183: the corrected "delivered ->
    # must not modify the payment method" branch (retail_session183_standalone
    # extension). Same reason as #e0 above: the first delivered order in the db
    # (#W4817420/ava_moore_2033) is owned by a user whose only method IS the
    # order's current one, so a refusal would be confounded by "that is already
    # your payment method" instead of resting on the status rule alone.
    "retail_067_state_modify_pending_order_payment#e1",
)

# docs/agentcoveragetesting_reuse_log.md section 182 (task_e9deccb5): Given = pending
# AND the user chooses more than one payment method, all different from the
# original (a realizable=="conversation" clause). Needs an owner with >= 2 usable
# different methods -- see _order_owner_has_two_usable_different_payment_methods.
_TWO_DIFFERENT_PAYMENT_METHODS_BRANCH_IDS = ("retail_023_arg#e1",)


# docs/agentcoveragetesting_reuse_log.md section 98.7/101: retail_067_state_
# modify_pending_order_address#e0's real Given is a single-row scalar compound
# condition (this ONE order's status is neither "pending" nor "delivered") --
# but the frozen upstream corpus (specs_retail_gpt41_full_run3_rebound_final_
# gwt_v5_conditions_matches.json) tags both DBconditions entries with a spurious
# "quant": "all", which select_database_row's cross-row group-aggregation path
# (_select_quantified_group_row) misreads as "every order this user owns must
# satisfy both ne clauses" -- a genuinely different claim, verified to land on
# a user who owns ZERO orders (a vacuous "all" match) instead of the real order
# (#W2611340, user james_li_5688, status "processed") that actually satisfies
# the condition as a plain per-row filter. Re-deriving this correctly at the
# Step1/2 source would touch the whole retail corpus's assembly_fingerprint
# (section 99/100's documented cascade cost) for a fix that only concerns 2
# conditions on 1 branch -- this dict corrects the tag at the true last point
# before Step7 runtime consumption (generic_tau_v2_fixture_binder_v1.py's
# select_fixture), touching no persisted JSON artifact.
_QUANT_ENCODING_OVERRIDE = {
    "retail_067_state_modify_pending_order_address#e0::DB::1": {"quantifier": {"present": False}},
    "retail_067_state_modify_pending_order_address#e0::DB::2": {"quantifier": {"present": False}},
}

RETAIL_CONFIG = DomainBindingConfig(
    domain="retail",
    identity_fields={
        "user_id": ("users", "user_id"),
        "order_id": ("orders", "order_id"),
        "first_name": ("users", "name.first_name"),
        "last_name": ("users", "name.last_name"),
        "zip": ("users", "address.zip"),
        # Real tau2 policy.md: "you have to authenticate the user identity by
        # locating their user id via email, or via name + zip code" -- email is the
        # OTHER real alternative the policy names, not previously offered anywhere
        # (section 63).
        "email": ("users", "email"),
    },
    dict_key_fields={
        "payment_method_id": ("users", "payment_methods"),
    },
    free_form_placeholders={
        "reason": "no longer needed",
        "summary": "Customer needs help that requires a human agent.",
        # calculate's real tau2 precondition is a character whitelist
        # (^[0-9+\-*/(). ]*$, tools.py:154) -- any valid arithmetic expression
        # passes; not required_driver_binding-verified for a SPECIFIC value (the
        # real oracle only checks the shape via matches_regex, see section 63), but
        # the live conversation still needs a real, always-valid instance to offer
        # when the branch's own canonical_request never embeds one.
        "expression": "2 + 2",
        # get_product_details's real precondition is just "a real product_id that
        # exists" (tau2's own _get_product raises ValueError otherwise) -- not tied
        # to any specific order/user, so no identity_fields navigation can reach it;
        # a real, verified product id from the shipped fixture db (Electric
        # Kettle), not invented (section 63).
        "product_id": "1075968781",
    },
    ownership_relation=_OWNERSHIP,
    navigation_relations=(_OWNERSHIP,),
    default_table="orders",
    preferred_row_predicate=None,
    operation_row_selectors={"modify_pending_order_payment": _payment_method_changeable_order},
    database_condition_overrides=_QUANT_ENCODING_OVERRIDE,
    # Real, verified tau2 preconditions (tau2/domains/retail/tools.py, section 63):
    # cancel_pending_order/modify_pending_order_items/modify_pending_order_address
    # all raise unless status=="pending"; exchange_delivered_order_items/
    # return_delivered_order_items need "delivered". An unconstrained Given ("True")
    # branch whose required_driver_binding_names is just order_id/reason (no item-
    # bundle fields) goes through the plain generic path, which had no status
    # awareness at all -- found via a real retail smoke run landing
    # cancel_pending_order on a real "processed" order (a genuine no-op the agent
    # correctly refused).
    operation_row_preferences={
        "cancel_pending_order": ("orders", "status", "pending"),
        "modify_pending_order_items": ("orders", "status", "pending"),
        "modify_pending_order_address": ("orders", "status", "pending"),
        "exchange_delivered_order_items": ("orders", "status", "delivered"),
        "return_delivered_order_items": ("orders", "status", "delivered"),
    },
    # docs/agentcoveragetesting_reuse_log.md section 128: retail_087_state#b1/
    # retail_096_state#b1 real adversarial siblings of retail_087_state#b0/
    # retail_096_state#b0 -- their real claim is "the user_id argument the agent is
    # pushed to call get_user_details/modify_user_address for does NOT match the
    # authenticated user establishing the conversation". _OWNERSHIP is retail's only
    # real navigation_relations edge (users->orders) -- there is no users<->users
    # edge, so select_fixture/_navigate_to_table can only ever anchor ONE real
    # "users" row per branch, and identity_fields' "user_id" necessarily resolves
    # off that SAME row for both #b0 and #b1 without this override. Both #b1
    # branches keep the exact same DBcondition-less/non_database_conditions shape as
    # their #b0 siblings (so the AUTHENTICATED user is still resolved by the normal,
    # untouched navigation path -- confirmed real: retail_087_state#b0 anchors via
    # the unconstrained-orders->owner path onto james_li_5688, retail_096_state#b0
    # anchors via _resolve_user_address_bundle onto noah_brown_6181); this override
    # supplies only the SECOND, genuinely different real user_id the Then text
    # forbids the target tool call from using -- both real, distinct rows of
    # tau2-bench's own retail db.json (not invented ids), and each other's real
    # authenticated identity (noah_brown_6181/james_li_5688), so both branches are
    # real, symmetric, non-degenerate "two distinct real users" fixtures.
    # retail_096_state#b1 additionally needs its own real "new address"
    # values genuinely different from noah_brown_6181's (the authenticated
    # user's) own real current address (986 Sunset Drive, Denver, CO 80279)
    # -- a real, confirmed-via-online-rerun gap: _resolve_user_address_
    # bundle (generic_tau_retail_v2_bundle_resolver_v1.py) supplies the
    # SAME user's own real current address as the "new" address to set
    # (unlike _resolve_new_order_address_bundle_for_order, which already has
    # a "genuinely different real address" fix for orders -- section 63 --
    # this branch's real target tool, modify_user_address, has no
    # equivalent), making the write action a guaranteed real no-op
    # regardless of the real user_id mismatch this branch exists to test
    # (confirmed via 2 independent real online reruns, both genuinely
    # refusing for the unrelated "address already on file" reason, never
    # engaging the real ownership question at all -- see docs/
    # agentcoveragetesting_reuse_log.md section 128). Overriding with real,
    # different values (james_li_5688's own real, current address -- not
    # fabricated) fixes this without touching retail_096_state#b0 or any
    # other branch that shares the same resolver (this table is branch_id-
    # scoped, applied as the real last step regardless of which internal
    # path resolved the branch).
    # docs/agentcoveragetesting_reuse_log.md section 129 (task_114c7458):
    # retail_080_state#b2's real Given ("the payment method specified in
    # 'payment_method_id' does not belong to the authenticated user") is a
    # real not_owner claim about payment_method_id, NOT about the order being
    # exchanged -- unlike retail_081/088/092/099_state#b1 above, whose real
    # not_owner claim IS about the order itself (so anchoring that order to
    # the users->orders not_owner relation pair is correct for them). Trying
    # the SAME fix (adding retail_080_state#b2 to generic_tau_retail_v2_
    # bundle_resolver_v1.py's _RELATION_GIVEN_BUNDLE_BRANCH_IDS) was tried
    # first and found wrong via a real online rerun: it makes the exchanged
    # order ITSELF not the authenticated user's own too, so a real agent
    # correctly refuses the whole request for order-ownership reasons before
    # ever reaching the payment-method question this branch exists to test
    # (a real, different confound bug -- see that file's own comment for the
    # full account). The real, correct fix leaves retail_080_state#b2's
    # anchor order/identity resolution completely untouched (still the
    # normal, Given-ignorant bundle-construction default -- a real delivered
    # order, #W4817420, legitimately owned by its own real identity,
    # ava_moore_2033 -- so the exchange itself stays actionable) and only
    # overrides payment_method_id with a real, different, genuinely NOT-
    # owned-by-ava_moore_2033 value: noah_brown_6181's own real
    # "paypal_5727330" (the SAME deterministic default not-owner anchor user
    # every order-ownership branch above already uses, for consistency, not
    # picked arbitrarily) -- a real, distinct row of tau2-bench's own retail
    # db.json (not fabricated), confirmed via a real online rerun (section
    # 129) to make the branch's own prior_call_consistency/get_user_details
    # check genuinely non-vacuous: ava_moore_2033's own real payment_methods
    # dict contains only "gift_card_8168843", so "paypal_5727330" genuinely
    # fails membership, and a compliant agent must use ava's own gift card
    # instead of the referenced one.
    # docs/agentcoveragetesting_reuse_log.md section 133.1 (task_9b8f8322):
    # retail_024_arg#b0's real Given is only a database_conditions status
    # check ("orders.status in [pending, pending (item modified)]") -- it
    # says nothing about payment methods at all, so select_fixture's plain
    # database_conditions-only path (select_database_row, no operation_row_
    # selectors/preferences awareness, since those only run for the relation/
    # unconstrained paths -- see select_fixture's branch at
    # generic_tau_v2_fixture_binder_v1.py's `if database_conditions:` arm)
    # lands on db.json's own first pending order (#W5918442, real owner
    # sofia_rossi_8776), and dict_key_fields' generic payment_method_id
    # resolution then picks that user's own alphabetically-first (and only)
    # payment method -- a real credit card (credit_card_5051208), not a gift
    # card at all. The branch's own oracle check requires observing a real
    # "Insufficient gift card balance to pay for the order" tool_call_error
    # from modify_pending_order_payment (tau2 tools.py: raised only when the
    # NEW payment_method_id resolves to a real GiftCard whose balance is less
    # than the order's own real payment amount) -- structurally unreachable
    # for sofia_rossi_8776, who owns no gift card at all. Verified against
    # the real shipped fixture db: no eq/in-only database_conditions rewrite
    # can express "this order's owner also has an underfunded gift card", so
    # this needs the same real identity_binding_overrides mechanism as
    # retail_080_state#b2/retail_096_state#b1 above -- a real, different,
    # verified (order, user, gift_card) triple: raj_sanchez_2970's own real
    # order #W4566809 (status "pending", exactly one real payment_history
    # entry of transaction_type "payment", amount 826.70, current payment
    # method credit_card_3362387) and raj_sanchez_2970's own real gift card
    # (gift_card_2259499, real balance 30.00 -- genuinely less than 826.70,
    # and genuinely different from the order's own current payment method).
    # All 7 fields overridden keep the bound fixture internally consistent
    # (order_id/user_id/status pointing at the SAME real row, first_name/
    # last_name/zip/email/payment_method_id matching that same real user) --
    # not a fabricated user or order, both real, verified rows of tau2-
    # bench's own retail db.json.
    identity_binding_overrides={
        ("retail_024_arg#b0", "order_id"): "#W4566809",
        ("retail_024_arg#b0", "user_id"): "raj_sanchez_2970",
        ("retail_024_arg#b0", "status"): "pending",
        ("retail_024_arg#b0", "first_name"): "Raj",
        ("retail_024_arg#b0", "last_name"): "Sanchez",
        ("retail_024_arg#b0", "zip"): "92147",
        ("retail_024_arg#b0", "email"): "raj.sanchez2046@example.com",
        ("retail_024_arg#b0", "payment_method_id"): "gift_card_2259499",
        ("retail_087_state#b1", "user_id"): "noah_brown_6181",
        ("retail_096_state#b1", "user_id"): "james_li_5688",
        ("retail_096_state#b1", "address1"): "215 River Road",
        ("retail_096_state#b1", "address2"): "Suite 991",
        ("retail_096_state#b1", "city"): "New York",
        ("retail_096_state#b1", "state"): "NY",
        ("retail_096_state#b1", "country"): "USA",
        ("retail_096_state#b1", "zip"): "10083",
        ("retail_080_state#b2", "payment_method_id"): "paypal_5727330",
    },
    # docs/agentcoveragetesting_reuse_log.md section 129 (task_ce5cb893):
    # retail_069_state#b0's real Then ("the agent must not state that the
    # order status will change to anything other than 'pending' after
    # payment method modification") compiles to a semantic_judge check over
    # assistant_message text, not a literal tool_call observation -- so its
    # own real observation_plan.tool_names_from_effective_routes is empty,
    # even though its own real When names a specific real target tool in
    # prose ("the user confirms modification of the payment method for a
    # pending order" -- modify_pending_order_payment). Without this hint,
    # _pick_unconstrained_row's real per-tool selectors/preferences (both
    # keyed by tool_name, both iterated FROM tool_names_from_effective_
    # routes) never run for this branch at all, and it falls through,
    # deterministically (rows[0], no randomness anywhere in that function --
    # a real, verified finding, not assumed: re-running online verification
    # with a different --seed cannot change this fixture draw), to db.json's
    # own real first order (#W2611340, ava real status "processed"), so
    # every real online rerun genuinely only ever exercises the refusal path
    # this branch's own real database already had a real, unrelated selector
    # for -- _payment_method_changeable_order (status=="pending", a real
    # payment_history entry, a real differing new payment_method_id) --
    # originally built for retail_055_order_modify_pending_order_payment#b0,
    # which reaches it via its own real tool_names_from_effective_routes.
    # This hint lets retail_069_state#b0 reuse that SAME real selector.
    # docs/agentcoveragetesting_reuse_log.md section 133.3 (task_20504983):
    # retail_055_order_cancel_pending_order#b1's real Given is the literal
    # text "True" (no database_conditions, no non_database_conditions), so
    # select_fixture falls all the way through to _pick_unconstrained_row --
    # but its own real target is a 3-way disjunctive OR-route (modify_
    # pending_order_address/items/payment, from section 49), whose real
    # tool identity is carried entirely in target_event.matcher.tool_names
    # (a "tool_name_in" shape), never in event_filter.field_equals.tool_name.
    # driver_plan_lowering_v1.py's real _target_tool_names helper (the
    # function that actually populates observation_plan.tool_names_from_
    # effective_routes at Step6) only ever reads the single-tool field_equals
    # shape -- confirmed via direct code read, not assumed -- so this
    # branch's own real tool_names_from_effective_routes compiles to an
    # empty list, and operation_row_preferences (section 116 already
    # configured "prefer pending status" for all 3 of these real tools) is
    # never consulted at all, falling through further to rows[0]: db.json's
    # own first order, real owner james_li_5688, whose 4 real orders are ALL
    # processed/delivered (zero pending) -- so no compliant agent can ever
    # succeed. A real, full retail-corpus scan (185 branches) for this exact
    # shape (empty tool_names_from_effective_routes AND a real matcher.
    # tool_names event whose names overlap this domain's own configured
    # operation_row_selectors/operation_row_preferences) found exactly ONE
    # match: this branch. (telecom_055_order#b0 has the identical real
    # Step6-lowering shape -- confirmed via the same scan run against
    # telecom's own outputs/v5_step6_plans_v0_2_telecom/plans.json -- but
    # driver_plan_lowering_v1.py is shared, domain-generic compiler
    # infrastructure: fixing _target_tool_names itself there would silently
    # change telecom's own real compiled Step6 output too, out of this
    # retail-only task's scope and directly risking a concurrent-edit
    # collision with the parallel telecom task. Flagged via a separate
    # spawn_task instead of fixed here.) The real, principled, already-
    # established fix for this exact class of problem (a genuine Step6
    # tool-identity compilation gap that a Step7-level hint can safely work
    # around without touching frozen/shared compiler code) is exactly
    # unconstrained_row_tool_hints' own precedent, immediately below, from
    # retail_069_state#b0/section 129 -- same real mechanism, different real
    # reason the routes list came up empty (there: semantic_judge-only,
    # no literal tool_call anchor at all; here: a real disjunctive OR-route
    # whose tool identity Step6 lowering doesn't yet know how to read). The
    # hinted tool is "modify_pending_order_payment" specifically (not
    # "cancel_pending_order", despite the branch_id's own name) because this
    # branch's real oracle_plan checks target the modify_* disjunction, not
    # cancel_pending_order at all -- reusing the SAME real, already-verified
    # _payment_method_changeable_order selector (status=="pending" + exactly
    # one real payment entry + a genuinely different alternate payment
    # method) that retail_069_state#b0 above already reuses, which is a
    # strictly STRONGER guarantee than any of the 3 disjunctive tools'
    # own real "pending" precondition needs individually.
    unconstrained_row_tool_hints={
        "retail_069_state#b0": "modify_pending_order_payment",
        "retail_055_order_cancel_pending_order#b1": "modify_pending_order_payment",
    },
    # docs/agentcoveragetesting_reuse_log.md section 177 -- see
    # _PAYMENT_METHOD_DIFFERS_BRANCH_IDS above.
    database_row_tiebreak_selectors={
        **{
            branch_id: _order_owner_has_usable_different_payment_method
            for branch_id in _PAYMENT_METHOD_DIFFERS_BRANCH_IDS
        },
        # section 182 -- see _TWO_DIFFERENT_PAYMENT_METHODS_BRANCH_IDS above.
        **{
            branch_id: _order_owner_has_two_usable_different_payment_methods
            for branch_id in _TWO_DIFFERENT_PAYMENT_METHODS_BRANCH_IDS
        },
    },
)
