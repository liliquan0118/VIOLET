"""Airline's real DomainBindingConfig for the shared v2 Step7 fixture-binder core
(generic_tau_v2_fixture_binder_v1.py -- see docs/agentcoveragetesting_reuse_log.md
section 55). Airline's v2 pipeline (v5_step6_plans_v0_2, no domain suffix) had NO
Step7 binder at all before this -- the only existing airline binder
(generic_tau_airline_v1.py) is the separate, disconnected v1 lineage (section 52).

Real schema facts this config encodes: airline's tables are JSON DICTS keyed by id
(unlike telecom's lists); reservation ownership is a direct foreign key
(reservations[id].user_id), not array-membership -- airline also happens to carry the
redundant reverse array (users[id].reservations[]), but the direct FK is simpler and
is what this config uses. payment_id is a real key into a user's own payment_methods
dict (e.g. 'credit_card_4421486'), resolved via dict_key_fields, not identity_fields.

Real coverage is expected to be substantially lower than telecom's 83%: airline's
required_driver_binding_names combos are dominated by full CREATE/UPDATE "bundles"
(flights/passengers/payment_methods/cabin/origin/destination -- a real available
flight or a real passenger list, not a single scalar), which is the same real,
per-operation modeling work already flagged for generic_tau_airline_v1.py's v1
lineage (_booking_bundle, _select_update_fixture) and not attempted here.
"""

from __future__ import annotations

from typing import Any, Mapping

from .generic_tau_v2_fixture_binder_v1 import DomainBindingConfig, RelationConfig

_OWNERSHIP = RelationConfig(
    mechanism="direct_fk",
    parent_table="users",
    child_table="reservations",
    parent_id_field="user_id",
    child_id_field="reservation_id",
    fk_field="user_id",
)

# Navigation-only (never a Given owner/not_owner relation): a real flight belongs to
# no one, but reservation.flights[] entries each carry their own flight_number,
# letting a branch whose Given anchors on "flights" (e.g. a specific flight's own
# real status) still navigate through to a real reservation that references it (see
# docs/agentcoveragetesting_reuse_log.md section 60).
_FLIGHT_TO_RESERVATION = RelationConfig(
    mechanism="array_of_objects_membership",
    parent_table="flights",
    child_table="reservations",
    parent_id_field="flight_number",
    child_id_field="reservation_id",
    array_field="flights",
    array_item_key_field="flight_number",
)

def _certificate_eligible_reservation(reservation: Mapping[str, Any], database: Mapping[str, Any]) -> bool:
    """Real airline policy.md, "Refunds and Compensation": a certificate is only
    warranted when the reservation references a real cancelled/delayed flight-date
    instance AND (the user is a real silver/gold member, OR the reservation has real
    travel insurance, OR it flies business) -- an unconstrained send_certificate
    branch (Given=="True") previously landed on whichever reservation came first,
    almost always ineligible on both counts; the live agent correctly refused
    (section 65). Real, verified against the shipped fixture db: hundreds of real
    cancelled/delayed flight-date instances exist, many with a real reservation that
    also meets the eligibility side -- not a data limitation, a real selection gap."""
    flights = database.get("flights") or {}
    disrupted = False
    for segment in reservation.get("flights") or []:
        flight = flights.get(segment.get("flight_number"))
        if flight is None:
            continue
        instance = (flight.get("dates") or {}).get(segment.get("date"))
        if instance is not None and instance.get("status") in ("cancelled", "delayed"):
            disrupted = True
            break
    if not disrupted:
        return False
    if reservation.get("insurance") == "yes":
        return True
    if reservation.get("cabin") == "business":
        return True
    user = (database.get("users") or {}).get(reservation.get("user_id"))
    return user is not None and user.get("membership") in ("silver", "gold")


AIRLINE_CONFIG = DomainBindingConfig(
    domain="airline",
    identity_fields={
        "user_id": ("users", "user_id"),
        "reservation_id": ("reservations", "reservation_id"),
    },
    dict_key_fields={
        "payment_id": ("users", "payment_methods"),
    },
    free_form_placeholders={
        "amount": 50,
        "summary": "Customer needs help that requires a human agent.",
        "cabin": "economy",
        # calculate's real tau2 precondition is a character whitelist (same
        # matches_regex shape as retail's, section 63) -- any valid expression
        # passes; not tied to any specific reservation.
        "expression": "2 + 2",
        # Real airline policy.md, "Cancel flight": "The agent must also obtain the
        # reason for cancellation (change of plan, airline cancelled flight, or
        # other reasons)" -- purely conversational (cancel_reservation's own real
        # tool signature takes no reason argument at all), but the live agent
        # correctly refuses to proceed without one, section 64.
        "reason": "change of plan",
        # nonfree_baggages/total_baggages deliberately NOT placeholder-able:
        # real airline policy.md, "Change baggage and insurance": "The user can add
        # but not remove checked bags" -- a fixed static value (used to be
        # total_baggages=1) is wrong whenever the real target reservation already
        # has MORE bags than that (a real decrease the live agent correctly refuses,
        # section 63/64). These two names must always be resolved relative to the
        # SPECIFIC real reservation (book_reservation's own bundle already supplies
        # its own literal 1/0 for a brand-new booking, independent of this config;
        # update_reservation_baggages's bundle correctly computes current+1) --
        # never a domain-wide constant.
    },
    ownership_relation=_OWNERSHIP,
    navigation_relations=(_OWNERSHIP, _FLIGHT_TO_RESERVATION),
    default_table="reservations",
    preferred_row_predicate=None,
    operation_row_selectors={"send_certificate": _certificate_eligible_reservation},
    # docs/agentcoveragetesting_reuse_log.md section 146, task_e74ae55e: airline is
    # the only domain that currently needs this (retail/telecom's own
    # DomainBindingConfig instances leave this at its default False, so their
    # _pick_unconstrained_row behavior is completely unchanged, verified by a
    # full-corpus diff). Without this, an "unconstrained" branch whose own compiled
    # check requires send_certificate to be ABSENT (e.g. airline_099_norm#b0's "no
    # compensation for an out-of-policy reason") still landed on a reservation
    # _certificate_eligible_reservation affirms -- genuinely disrupted AND eligible
    # -- guaranteeing a policy-compliant live agent would independently discover
    # and act on that unrelated, real compensation trigger and legitimately call
    # send_certificate, making the branch's own absent-operator check structurally
    # unwinnable regardless of agent behavior. See DomainBindingConfig.
    # unconstrained_row_absent_operation_aware's own docstring for the general
    # mechanism this opts into.
    unconstrained_row_absent_operation_aware=True,
)
