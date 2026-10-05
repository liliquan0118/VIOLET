"""Telecom's real DomainBindingConfig for the shared v2 Step7 fixture-binder core
(generic_tau_v2_fixture_binder_v1.py -- see docs/agentcoveragetesting_reuse_log.md
sections 52-55). This module used to contain telecom-specific binding logic; that
logic is now the shared, domain-agnostic engine in generic_tau_v2_fixture_binder_v1
.py (proven to reproduce byte-identical real results for telecom after the refactor).
This module is now just telecom's config plus thin wrappers preserving the original
public names so existing callers/tests are unaffected.

Real telecom schema facts this config encodes: tables are JSON LISTS (not dicts);
line ownership is array-membership (customer.line_ids[] contains a line_id, no
direct field on the line row itself); required_driver_binding_name "dob" maps to the
real customer record's own "date_of_birth" field (a real name mismatch with
tau2.domains.telecom.tools.get_customer_by_name's own "dob" argument name).
"""

from __future__ import annotations

from typing import Any, Mapping

from .generic_tau_v2_fixture_binder_v1 import (
    DomainBindingConfig,
    Fixture,
    GenericV2BindingError,
    RelationConfig,
    bind_v2_branch,
    navigate_to_table,
    resolve_driver_bindings as _resolve_driver_bindings,
    resolve_known_fact_bundle as _resolve_known_fact_bundle,
    select_fixture,
)


class GenericTelecomV2BindingError(GenericV2BindingError):
    """Telecom-specific alias of GenericV2BindingError, kept for existing callers."""


TELECOM_CONFIG = DomainBindingConfig(
    domain="telecom",
    identity_fields={
        "customer_id": ("customers", "customer_id"),
        "line_id": ("lines", "line_id"),
        "bill_id": ("bills", "bill_id"),
        "dob": ("customers", "date_of_birth"),
        "full_name": ("customers", "full_name"),
    },
    dict_key_fields={},
    free_form_placeholders={
        "reason": "Customer requested this change.",
        "summary": "Customer needs help that requires a human agent.",
        "gb_amount": 1.0,
        "limit": 5,
    },
    ownership_relation=RelationConfig(
        mechanism="array_membership",
        parent_table="customers",
        child_table="lines",
        parent_id_field="customer_id",
        child_id_field="line_id",
        array_field="line_ids",
    ),
    navigation_relations=(
        RelationConfig(
            mechanism="array_membership",
            parent_table="customers",
            child_table="lines",
            parent_id_field="customer_id",
            child_id_field="line_id",
            array_field="line_ids",
        ),
        # bills carry a direct customer_id field (unlike lines) -- a real, separate
        # relation, not reachable via the ownership relation above.
        RelationConfig(
            mechanism="direct_fk",
            parent_table="customers",
            child_table="bills",
            parent_id_field="customer_id",
            child_id_field="bill_id",
            fk_field="customer_id",
        ),
        # plans are a shared catalog (no owner of their own); a real line references
        # its plan via a direct plan_id field -- lets a branch whose Given anchors on
        # "plans" (e.g. telecom_115_state#b0's "the plan supports data service")
        # still navigate through to a real owning customer for other bindings
        # (section 60).
        RelationConfig(
            mechanism="direct_fk",
            parent_table="plans",
            child_table="lines",
            parent_id_field="plan_id",
            child_id_field="line_id",
            fk_field="plan_id",
        ),
    ),
    default_table="lines",
    preferred_row_predicate=("status", "Active"),
    # Real, verified tau2 preconditions (tau2/domains/telecom/tools.py, section 63):
    # resume_line raises unless status is Suspended/Pending Activation (the OPPOSITE
    # of the domain-wide Active preference above); suspend_line raises unless
    # Active (redundant with the domain-wide preference, kept explicit for
    # robustness); enable_roaming/disable_roaming don't raise on an already-correct
    # state but silently no-op ("Roaming was already enabled/disabled") instead of
    # producing the real, observable state change a live conversation needs.
    # tau2's own real telecom policy (data/tau2/domains/telecom/main_policy.md,
    # "Overdue Bill Payment"): "Check the bill status to make sure it is overdue...
    # The send payment request tool will not check if the bill is overdue. You
    # should always check that the bill is overdue before sending a payment
    # request." -- send_payment_request's own tool code doesn't raise on a non-
    # overdue bill, but the real agent policy requires Overdue specifically (not
    # just non-Paid); applies to a NAVIGATED bill row (reached from an anchor line
    # via the customer relation), not just an anchor pick.
    operation_row_preferences={
        "resume_line": ("lines", "status", "Suspended"),
        "suspend_line": ("lines", "status", "Active"),
        "disable_roaming": ("lines", "roaming_enabled", True),
        "enable_roaming": ("lines", "roaming_enabled", False),
        "send_payment_request": ("bills", "status", "Overdue"),
    },
)


def select_fixture_context(
    plan: Mapping[str, Any], database: Mapping[str, Any], *, allow_construction: bool = False
) -> Fixture:
    try:
        return select_fixture(plan, database, TELECOM_CONFIG, allow_construction=allow_construction)
    except GenericV2BindingError as exc:
        raise GenericTelecomV2BindingError(str(exc)) from exc


def resolve_driver_bindings(plan: Mapping[str, Any], fixture: Fixture) -> dict[str, Any]:
    try:
        return _resolve_driver_bindings(plan, fixture, {}, TELECOM_CONFIG)
    except GenericV2BindingError as exc:
        raise GenericTelecomV2BindingError(str(exc)) from exc


def _telecom_reference_time() -> str:
    # Real, fixed reference clock tau2's own telecom domain uses (tau2.domains
    # .telecom.utils.get_now(): "assume now is 2025-02-25 12:08:00") -- not this
    # session's real-world date, the domain's own frozen test-time convention.
    from tau2.domains.telecom.utils import get_now

    return get_now().isoformat()


def bind_telecom_v2_branch(
    plan: Mapping[str, Any], database: Mapping[str, Any], *, allow_construction: bool = False
) -> dict[str, Any]:
    try:
        fixture = select_fixture(
            plan, database, TELECOM_CONFIG,
            allow_construction=allow_construction, reference_time=_telecom_reference_time(),
        )
        driver_bindings = _resolve_driver_bindings(plan, fixture, database, TELECOM_CONFIG)
    except GenericV2BindingError as exc:
        raise GenericTelecomV2BindingError(str(exc)) from exc
    customer_row = navigate_to_table(fixture, "customers", database, TELECOM_CONFIG, plan=plan)
    line_row = navigate_to_table(fixture, "lines", database, TELECOM_CONFIG, plan=plan)
    bill_row = navigate_to_table(fixture, "bills", database, TELECOM_CONFIG, plan=plan)
    return {
        "branch_id": plan["branch_id"],
        "driver_plan_id": plan["driver_plan_id"],
        "fixture_source": fixture.fixture_source,
        "fixture_identity": {
            "customer_id": (customer_row or {}).get("customer_id"),
            "line_id": (line_row or {}).get("line_id"),
            "bill_id": (bill_row or {}).get("bill_id"),
        },
        "state_patch": fixture.state_patch,
        "driver_bindings": driver_bindings,
        "known_fact_bundle": _resolve_known_fact_bundle(fixture, database, TELECOM_CONFIG, plan=plan),
        "canonical_request": plan["test_point"]["when"]["supplied_user_request"],
    }
