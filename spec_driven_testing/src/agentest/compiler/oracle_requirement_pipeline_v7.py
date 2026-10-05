"""Test-driven oracle requirement candidates and binary GWT selection."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .profiles import validate_sut_adapter_profile
from .then_atomization import ThenAtomizationError
from .tool_endpoint_lexical_retrieval_v1 import relevant_endpoints as _relevant_endpoints


PACKET_SET_VERSION = "agentspectesting.oracle-requirement-packets/v0.7"
PACKET_VERSION = "agentspectesting.oracle-requirement-packet/v0.7"
RESPONSE_SET_VERSION = "agentspectesting.oracle-requirement-responses/v0.7"
REQUIREMENT_SET_VERSION = "agentspectesting.oracle-requirement-set/v0.7"
TASK_NAME = "gwt_oracle_requirement_selection"
DECISIONS = frozenset({"yes", "no", "ambiguous"})

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a", "an", "and", "are", "by", "for", "from", "in", "is", "of",
        "on", "or", "the", "to", "with", "information", "details",
    }
)
_QUOTED = re.compile(r"(['\"])(.+?)\1")
_EXPLICIT_TOOL = re.compile(r"\btool\s+call\s+to\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE)
# _QUOTED above is the only existing mechanism that generates a message-content
# candidate, and it only fires on a literal quoted span. A Then that prohibits a
# message claim in plain prose (no quotes) -- e.g. "The agent must not state that
# the price for each extra baggage is not 50 dollars." -- never gets a
# message-content candidate at all under the existing rules; see
# docs/oracle_requirement_pipeline_v0_7.md for the real branches this affects and
# why the fallback below produces a semantic_requirement (there is no literal
# string to match exactly, so this cannot be assistant_literal).
_MESSAGE_CLAIM_PROHIBITION = re.compile(
    r"\bmust not (?:state|say|claim|tell|inform|mention)\b|\bmust not send (?:a|the) message\b",
    re.IGNORECASE,
)

# Exact-Then-text lineage for specific _MESSAGE_CLAIM_PROHIBITION fallback
# candidates -- without this, every branch matching that regex gets a
# semantic_requirement with EMPTY source_refs, which oracle_requirement_acceptance_v1.py's
# _classify unconditionally routes to needs_adjudication ("Semantic requirement
# lineage is absent from the accepted model.") regardless of how clean or
# unambiguous the claim itself is -- see
# docs/oracle_requirement_pipeline_v0_7.md section 34. Two real branches
# entered here, deliberately keeping the same "recovered real data" vs
# "hand-asserted" distinction _TEMPORAL_RELATION_FALLBACK_TARGETS documents
# for airline_083/airline_097:
#
# - airline_047_arg#b0 ("must not state that the price for each extra
#   baggage is not 50 dollars"): RECOVERED, not asserted. The real
#   policy_semantic_unit_model.json contains NPU0010::C01::R01::A01
#   (non_action_relation), whose focal_relation_quote is the exact sentence
#   "Each extra baggage costs 50 dollars." -- this data already exists and
#   describes precisely this claim, it was simply never linked to this
#   branch by the ordinary per-unit candidate emission loop above.
#
# - airline_092_state#b0v1 ("must not send a message to the user stating the
#   reservation is being cancelled"): ASSERTED, not recovered. No semantic
#   unit in the real model states this specific message-content claim on its
#   own (searched the full corpus for any unit combining "cancel" with
#   "message"/"tell"/"inform"/"say"/"state" -- none exists). The closest real
#   unit is CA0044::A01 ("if any portion has already been flown, the agent
#   cannot help and a transfer is needed"), which is the broader ACTION-level
#   policy this message-content prohibition is a natural consequence of (this
#   branch's sibling variants airline_092_state#b0v0/v2 already mechanically
#   cover the ACTION side via tool_call checks on cancel_reservation -- this
#   v1 variant is specifically about not misinforming the user in the
#   message, a distinct check neither sibling covers). Linking to CA0044::A01
#   is a judgment call, not a recovery of already-computed data -- flagged
#   here exactly as prominently as airline_083's own asserted entry above.
_MESSAGE_CLAIM_LINEAGE_TARGETS: dict[str, dict[str, Any]] = {
    "The agent must not state that the price for each extra baggage is not "
    "50 dollars.": {
        "semantic_unit_id": "NPU0010::C01::R01::A01",
        "policy_ref_id": "CP0017::P02",
        "semantic_unit_kind": "non_action_relation",
        "candidate_origin": "then_message_claim_lineage_recovery",
    },
    "The agent must not send a message to the user stating the reservation "
    "is being cancelled.": {
        "semantic_unit_id": "CA0044::A01",
        "policy_ref_id": "CP0025::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_message_claim_lineage_assertion",
    },
    # --- docs/agentcoveragetesting_reuse_log.md section 120: 13
    # telecom/retail branches from the item-3/Pattern-1 zero-check audit,
    # the same _MESSAGE_CLAIM_PROHIBITION / empty-source_refs /
    # needs_adjudication dead-end as the two entries above. All 13
    # branches have branch_context.origin == "prompt" (rule_text is
    # copied straight from the real system-prompt policy, not
    # domain_knowledge), and every entry below was independently
    # re-verified against the real
    # tau2-bench/data/tau2/domains/{telecom,retail}/{main_}policy.md /
    # tech_support_manual.md text and the real
    # outputs/v5_step3_support_{telecom,retail}_v0_1/
    # policy_semantic_unit_model.json before being added -- see section
    # 120 for full per-branch grounding quotes. 9 of 13 are RECOVERED (a
    # real non_action_relation unit's own source_requirement_text already
    # IS this claim, just never wired to this branch); 4 are ASSERTED (no
    # unit isolates just this clause -- the closest real unit is a
    # broader action_commitment this message-content prohibition is a
    # natural consequence of), same judgment-call pattern as
    # airline_092_state#b0v1 above, not a claim those 4 units verbatim
    # state just this sentence.

    # telecom_047_state#b0::OR01 -- RECOVERED. main_policy.md line 116
    # ("A user can only have one bill in the AWAITING PAYMENT status at a
    # time.") == NPU0005::C01::R01::A01's own source_requirement_text,
    # verbatim.
    "The agent must not state that a user can have more than one bill in "
    "the AWAITING PAYMENT status at a time.": {
        "semantic_unit_id": "NPU0005::C01::R01::A01",
        "policy_ref_id": "CP0016::P02",
        "semantic_unit_kind": "non_action_relation",
        "candidate_origin": "then_message_claim_lineage_recovery",
    },
    # telecom_048_state#b0::OR02 -- RECOVERED. main_policy.md lines
    # 121-123 ("A line can be suspended for the following reasons: - The
    # user has an overdue bill. - The line's contract end date is in the
    # past.") == NPU0007::C01::R01::A01's own source_requirement_text ("A
    # line may be suspended when the user has an overdue bill or when the
    # line's contract end date is in the past."), the same two-reason
    # exhaustive list stated positively; this branch's sibling OR01
    # (tool_argument, already bound to suspend_line's reason param)
    # covers the ACTION side of the same passage.
    "The agent must not state that a line can be suspended for a reason "
    "other than the user having an overdue bill or the line's contract "
    "end date being in the past.": {
        "semantic_unit_id": "NPU0007::C01::R01::A01",
        "policy_ref_id": "CP0017::P01",
        "semantic_unit_kind": "non_action_relation",
        "candidate_origin": "then_message_claim_lineage_recovery",
    },
    # telecom_067_state#b0::OR01 -- RECOVERED. tech_support_manual.md
    # line 171 ("For MMS to work, the user must have cellular service and
    # mobile data (any speed).") == NPU0023::C01::R01::A01's own
    # source_requirement_text, verbatim (this branch's own rule_text is
    # this exact sentence; policy_ref_id CP0049::P01 covers both the
    # cellular-service and mobile-data conjuncts via
    # NPU0023::C01::R01::A01/A02).
    "The agent must not state that MMS can work without cellular service "
    "or mobile data.": {
        "semantic_unit_id": "NPU0023::C01::R01::A01",
        "policy_ref_id": "CP0049::P01",
        "semantic_unit_kind": "non_action_relation",
        "candidate_origin": "then_message_claim_lineage_recovery",
    },
    # telecom_073_state#b0v0/v1::OR01 -- ASSERTED, not recovered. Real
    # main_policy.md lines 107-108 ("Send the user a payment request for
    # the overdue bill. - This will change the status of the bill to
    # AWAITING PAYMENT.") supports both phrasing variants' claim, but no
    # standalone non_action_relation unit isolates just this clause -- a
    # real, checked search confirms it only appears folded into the
    # broader CP0016::P01 action_commitment (CA0012::A01, whose own
    # policy_statement contains "...send the user a payment request for
    # the overdue bill (which changes the bill status to AWAITING
    # PAYMENT)..." verbatim) alongside the rest of the Overdue Bill
    # Payment procedure.
    "The agent must not state that sending a payment request for an "
    "overdue bill does not change the bill status to AWAITING PAYMENT.": {
        "semantic_unit_id": "CA0012::A01",
        "policy_ref_id": "CP0016::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_message_claim_lineage_assertion",
    },
    "The agent must not state that sending a payment request for an "
    "overdue bill changes the bill status to something other than "
    "AWAITING PAYMENT.": {
        "semantic_unit_id": "CA0012::A01",
        "policy_ref_id": "CP0016::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_message_claim_lineage_assertion",
    },
    # telecom_074_state#b0::OR01 -- RECOVERED. main_policy.md line 120
    # ("When a line is suspended, the user will not have service.") ==
    # NPU0008::C01::R01::A01's own source_requirement_text ("the user
    # will not have service."), conditioned on the same "a line is
    # suspended" source atom, verbatim.
    "The agent must not state that a suspended line has service or will "
    "have service while it remains suspended.": {
        "semantic_unit_id": "NPU0008::C01::R01::A01",
        "policy_ref_id": "CP0017::P01",
        "semantic_unit_kind": "non_action_relation",
        "candidate_origin": "then_message_claim_lineage_recovery",
    },
    # telecom_075_state#b0v0/v1/v2::OR01 -- ASSERTED, not recovered. Real
    # main_policy.md line 131 ("Each plan specify the maxium data usage
    # per month.") supports all three rephrasings of the same claim, but
    # a real, checked search of policy_index/passage_index confirms this
    # sentence was never materialized into its own semantic unit at all
    # -- CP0018::P01's own "statement" field (the Data Refueling passage)
    # starts only from "Under the Data Refueling feature, when a user's
    # data usage on a line exceeds the plan's monthly data limit..." and
    # never quotes the "Each plan specify..." sentence itself. The
    # closest real unit is CA0020::A01, the Data Refueling
    # action_commitment for which "the plan's monthly data limit" is a
    # load-bearing precondition -- same "broader policy this is a
    # natural consequence of" judgment call as airline_092_state#b0v1/
    # telecom_073 above, applied identically to all three phrasing
    # variants of this one underlying claim.
    "The agent must not state that a plan does not have a maximum data "
    "usage per month.": {
        "semantic_unit_id": "CA0020::A01",
        "policy_ref_id": "CP0018::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_message_claim_lineage_assertion",
    },
    "The agent must not state that there is no data usage limit for a "
    "plan.": {
        "semantic_unit_id": "CA0020::A01",
        "policy_ref_id": "CP0018::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_message_claim_lineage_assertion",
    },
    "The agent must not state that the maximum data usage per month is "
    "not specified for a plan.": {
        "semantic_unit_id": "CA0020::A01",
        "policy_ref_id": "CP0018::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_message_claim_lineage_assertion",
    },
    # retail_069_state#b0::OR01 -- RECOVERED. policy.md line 106 ("After
    # user confirmation, the order status will be kept as 'pending'.") ==
    # NPU0009::C01::R01::A01's own source_requirement_text ("the order
    # status remains 'pending'"), verbatim (CP0020::P02, "Modify
    # payment").
    "The agent must not state that the order status will change to "
    "anything other than 'pending' after payment method modification.": {
        "semantic_unit_id": "NPU0009::C01::R01::A01",
        "policy_ref_id": "CP0020::P02",
        "semantic_unit_kind": "non_action_relation",
        "candidate_origin": "then_message_claim_lineage_recovery",
    },
    # retail_070_norm#b0::OR01 -- RECOVERED. policy.md line 106 ("The
    # original payment method will be refunded immediately if it is a
    # gift card, otherwise it will be refunded within 5 to 7 business
    # days.") == NPU0009::C02::B01::R01::A01 ("the refund is immediate",
    # the gift-card branch) + NPU0009::C02::B02::R01::A01 ("it is
    # processed within 5 to 7 business days", the non-gift-card branch),
    # both CP0020::P02, both verbatim covering this claim's disjuncts.
    "The agent must not state that the original payment method will not "
    "be refunded immediately if it is a gift card, or that it will be "
    "refunded immediately if it is not a gift card, or provide any other "
    "refund timing that contradicts the policy.": {
        "semantic_unit_id": "NPU0009::C02::B01::R01::A01",
        "policy_ref_id": "CP0020::P02",
        "semantic_unit_kind": "non_action_relation",
        "candidate_origin": "then_message_claim_lineage_recovery",
    },
    # retail_072_order#b0::OR01 -- RECOVERED. policy.md line 126 ("After
    # user confirmation, the order status will be changed to 'return
    # requested', and the user will receive an email regarding how to
    # return items.") == NPU0017::C01::R01::A01 ("the order status is
    # changed to 'return requested'") + NPU0017::C02::R01::A01 ("the user
    # receives an email regarding how to return items"), both
    # CP0024::P01, both verbatim covering this claim's disjuncts.
    "The agent must not state that the order status will not be changed "
    "to 'return requested' or that the user will not receive an email "
    "regarding how to return items after confirmation.": {
        "semantic_unit_id": "NPU0017::C01::R01::A01",
        "policy_ref_id": "CP0024::P01",
        "semantic_unit_kind": "non_action_relation",
        "candidate_origin": "then_message_claim_lineage_recovery",
    },
    # retail_074_state#b0::OR01 -- RECOVERED. policy.md line 136 ("After
    # user confirmation, the order status will be changed to 'exchange
    # requested', and the user will receive an email regarding how to
    # return items.") == NPU0022::C01::R01::A01 ("the order status
    # changes to 'exchange requested'") + NPU0022::C02::R01::A01 ("the
    # user receives an email regarding how to return items"), both
    # CP0025::P05, both verbatim covering this claim's disjuncts.
    "The agent must not state that after user confirmation of an "
    "exchange, the order status will not be changed to 'exchange "
    "requested', or the user will not receive an email regarding how to "
    "return items.": {
        "semantic_unit_id": "NPU0022::C01::R01::A01",
        "policy_ref_id": "CP0025::P05",
        "semantic_unit_kind": "non_action_relation",
        "candidate_origin": "then_message_claim_lineage_recovery",
    },
}
_MESSAGE_CLAIM_LINEAGE_ORIGINS = frozenset(
    entry["candidate_origin"] for entry in _MESSAGE_CLAIM_LINEAGE_TARGETS.values()
)

# Added later (docs/agentcoveragetesting_reuse_log.md section 99): a positive
# analog of _MESSAGE_CLAIM_LINEAGE_TARGETS above -- that table only ever
# fires for a Then matching _MESSAGE_CLAIM_PROHIBITION ("must not state/say/
# claim..."), so a positive "the agent must guide/ask the user to do X"
# claim (not a prohibition) never gets a message-content candidate at all
# under the existing rules, for the same underlying reason the _QUOTED
# comment above gives: there is no literal quoted span to match exactly, so
# this cannot be assistant_literal, and X here (check_network_status,
# run_speed_test) is a real tau2 telecom USER tool, confirmed absent from
# tau2/domains/telecom/tools.py (the agent's own tool set), so this cannot
# be tool_call either -- the agent can only ever guide the user to use it,
# never call it itself.
#
# telecom_078_order#b1 / telecom_083_order#b1 (session-authored, docs/
# agentcoveragetesting_reuse_log.md section 99): both hand-authored this
# session to make telecom_078_order#b0/telecom_083_order#b0's existing,
# non_decisive, abstract Then ("must confirm...") concrete and checkable,
# grounded in the real tau2 telecom tech_support_workflow.md's first real
# diagnostic step (2.1.1 for 078, 3.1/3.2 for 083) and the real tau2
# telecom user_tools.py tool names. ASSERTED, not recovered: a real,
# targeted full-text search of policy_semantic_unit_model.json for "network
# status"/"check_network_status"/"run_speed_test" found no semantic unit
# genuinely matching "guide the user to check cellular service /run a speed
# test before proceeding" -- the model's real extraction never covered
# tech_support_workflow.md at all (only policy.md's CP-numbered passages),
# and the two real hits under those terms are CP0031 (airplane-mode
# diagnosis, a different real policy passage) and CP0039::P01 (only
# asserts run_speed_test's existence/purpose, not an obligation to use it).
# Both entries instead cite the closest REAL semantic unit that exists on
# each branch's OWN real policy passage (CP0037 for 078, CP0052 for 083) --
# the real "if the precondition is not met, refer users to the
# troubleshooting guide" action commitment their existing #b0 sibling's
# OR02/OR02 candidates already reference -- as the natural, concrete
# precursor this new Then's diagnostic-check obligation leads into, the
# same "broader policy this is a natural consequence of" judgment-call
# reasoning airline_092_state#b0v1's own asserted entry above documents,
# not a claim that either real unit is a verbatim match.
_POSITIVE_GUIDANCE_CLAIM_TARGETS: dict[str, dict[str, Any]] = {
    (
        "The agent must guide the user to use check_network_status() to "
        "verify they have cellular service, before taking any action "
        "related to mobile data."
    ): {
        "semantic_unit_id": "IA0016::A01",
        "policy_ref_id": "CP0037::P02",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    (
        "The agent must guide the user to use check_network_status() to "
        "verify they have cellular service and use run_speed_test() to "
        "verify their mobile data is working, before troubleshooting "
        "MMS issues."
    ): {
        "semantic_unit_id": "CA0057::A01",
        "policy_ref_id": "CP0052::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    # telecom_070_order#b0 (docs/agentcoveragetesting_reuse_log.md section
    # 111): unlike the two entries above, this one is RECOVERED, not
    # asserted -- the real policy_semantic_unit_model.json already contains
    # CA0060::A01 (policy_ref_id CP0055::P01) whose recorded
    # policy_statement is byte-for-byte the same claim as this branch's own
    # real gwt.then ("If check_wifi_calling_status() shows 'Wi-Fi Calling is
    # ON', the agent must guide the user to use toggle_wifi_calling() to
    # turn Wi-Fi Calling OFF."), sourced from a real, exact_unique policy-
    # lineage match against tech_support_workflow.md Step 3.4 ("If Wi-Fi
    # Calling is ON: Ask the user to turn Wi-Fi Calling OFF"). The ordinary
    # per-unit candidate-emission loop above (units_by_policy) DOES already
    # reach this exact unit for this branch -- but `_unit_text()`'s field
    # priority picks `direct_action_text` first, and for this unit that
    # field is only the fragment "guide the user" (the fuller
    # `modal_scope_text`, "guide the user to use `toggle_wifi_calling()` to
    # turn Wi-Fi Calling OFF.", is never reached) -- so the automatically
    # emitted candidate's requirement_text is a truncated, unusably vague
    # "guide the user", not this branch's real Then. Rather than changing
    # `_unit_text()`'s general field-priority order (used by every other
    # branch's per-unit emission, well outside this scoped fix), this entry
    # gives telecom_070 the same full-Then, exact-lineage candidate
    # directly, same as the two ASSERTED entries above. check_wifi_calling_
    # status()/toggle_wifi_calling() are real tau2 telecom USER tools
    # (tau2/domains/telecom/user_tools.py) -- confirmed absent from
    # tools.py, the agent's own tool set -- so this is guidance-only here
    # too, same reasoning as 078/083 above.
    (
        "The agent must guide the user to use toggle_wifi_calling() to "
        "turn Wi-Fi Calling OFF."
    ): {
        "semantic_unit_id": "CA0060::A01",
        "policy_ref_id": "CP0055::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_recovery",
    },
    # docs/agentcoveragetesting_reuse_log.md section 112: retail_071_norm#b0's
    # real Then ("must remind the customer to confirm they have provided all
    # the items they want to modify before modifying items in the pending
    # order") is a message-content reminder claim, same shape as telecom_070
    # above. Real policy_semantic_unit_model.json (retail) already contains
    # CA0033::A02 (policy_ref_id CP0021::P01) whose real
    # semantic_payload.direct_action_text is byte-for-byte "remind the
    # customer to confirm they have provided all the items they want to
    # modify" -- sourced from real policy.md (~line 110): "In particular,
    # remember to remind the customer to confirm they have provided all the
    # items they want to modify." Unlike telecom_070, this is ASSERTED, not
    # recovered: this branch's own policy_lineage_report does not link it to
    # CA0033::A02 (the ordinary per-unit candidate-emission loop never
    # reaches this unit for this branch), so the real, exact-text unit is
    # cited directly here, same judgment-call discipline as the
    # airline_092_state#b0v1 entry in _MESSAGE_CLAIM_LINEAGE_TARGETS above.
    (
        "The agent must remind the customer to confirm they have provided "
        "all the items they want to modify before modifying items in the "
        "pending order."
    ): {
        "semantic_unit_id": "CA0033::A02",
        "policy_ref_id": "CP0021::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    # docs/agentcoveragetesting_reuse_log.md section 112: retail_073_norm#b0,
    # same shape for the exchange-side reminder. Real
    # policy_semantic_unit_model.json (retail) already contains CA0048::A01
    # (policy_ref_id CP0025::P03) whose real semantic_payload.direct_
    # action_text/modal_scope_text/policy_statement are ALL byte-for-byte
    # "The agent must remind the customer to confirm they have provided all
    # items to be exchanged." -- sourced from real policy.md (~line 130):
    # "In particular, remember to remind the customer to confirm they have
    # provided all items to be exchanged." Same as retail_071 above: this
    # branch is not linked to the unit by policy_lineage_report, so ASSERTED
    # rather than recovered.
    (
        "The agent must remind the customer to confirm they have provided "
        "all items they want to exchange before proceeding with the "
        "exchange."
    ): {
        "semantic_unit_id": "CA0048::A01",
        "policy_ref_id": "CP0025::P03",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    # docs/agentcoveragetesting_reuse_log.md section 124: telecom_078_order#b2-
    # #b6 (session-authored, "whole-path coverage" -- full Path 2.1 coverage, following
    # up on section 99's #b1 which only covered tech_support_workflow.md Step
    # 2.1.1). Real, targeted search of policy_semantic_unit_model.json
    # (telecom) found real, RECOVERED semantic units for the individual
    # ATOMIC actions each of these Thens names (CA0049::A01/CP0042::P01 for
    # toggle_roaming(), CA0025::A01+CA0026::A01/CP0020::P02 for the line-level
    # roaming check+enable_roaming(), CA0048::A01/CP0041::P01 for toggle_
    # data(), CA0052::A01+A02/CP0045::P01 for refuel/plan-change) -- these are
    # already the basis of separate, already-shipped telecom_054_order#b0/
    # telecom_063_order#b0/telecom_062_order#b0/telecom_082_order#b0 branches
    # elsewhere in the 178-branch corpus (real, verified: grep of
    # specs_telecom_..._conditions_matches.json). What none of those isolated
    # atomic-action branches test is the SEQUENCING/gating property specific
    # to tech_support_workflow.md's Path 2.1 decision tree (Step 2.1.1 -> 2.1.2
    # -> 2.1.3 -> 2.1.4, each step's action gated on the outcome of the prior
    # one) -- that is what these 5 new telecom_078_order ORDER-kind siblings
    # add, same "before taking any action related to mobile data" ordering
    # shape #b1 already established for Step 2.1.1. Each entry below cites the
    # closest REAL recovered unit as the natural atomic action this ordering
    # constraint composes (ASSERTED linkage, not an exact-text recovery --
    # same discipline as the #b1/#b0 entries above, which the real semantic
    # unit's own policy_statement does not itself express any ordering
    # relative to the other Path 2.1 steps).
    #
    # v0.1.0 -> v0.1.1 real correction (same discipline as retail session
    # 119's own v0.1.0 -> v0.1.1 fix): the first drafted Then text for all 5
    # branches led with "After the agent has confirmed X, if Y, the agent
    # must ..." -- real inspection of oracle_expectation_compiler_v2.py's
    # `_modality()` (configs/oracle_expectation_policies/tau_bench_task_
    # completion_v0_1.json's modality_prefixes) showed it classifies
    # normative_mode purely by whether the Then STRING LITERALLY STARTS WITH
    # "The agent must " (or "The agent must not "/"The agent may ") --
    # leading with "After the agent has confirmed..." would never match any
    # prefix and silently fail to classify (verified with a direct probe
    # against the real policy file before compiling, not assumed). All 5
    # entries below were rewritten to lead literally with "The agent must "
    # and push the conditional/ordering clause after the main verb instead
    # -- verified this classifies as normative_mode="required" for all 5
    # before re-running Step1-4.
    (
        "The agent must guide the user to use toggle_roaming() to turn "
        "Data Roaming ON and ask them to rerun the speed test, if the "
        "user reports they are traveling outside their usual service "
        "area and Data Roaming is off after the agent has confirmed "
        "cellular service is available, before checking Mobile Data "
        "settings or Data Usage for the line."
    ): {
        "semantic_unit_id": "CA0049::A01",
        "policy_ref_id": "CP0042::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    (
        "The agent must check whether the line associated with the phone "
        "number the user provided is roaming enabled, and if it is not, "
        "call enable_roaming() to enable it at no cost for the user, if "
        "the user reports they are traveling outside their usual service "
        "area with Data Roaming already on but mobile data still not "
        "working, before checking Mobile Data settings or Data Usage for "
        "the line."
    ): {
        "semantic_unit_id": "CA0025::A01",
        "policy_ref_id": "CP0020::P02",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    (
        "The agent must guide the user to use toggle_data() to turn "
        "Mobile Data ON and ask them to rerun the speed test, if Mobile "
        "Data is off after the agent has confirmed cellular service is "
        "available and the user is not experiencing a roaming issue, "
        "before checking Data Usage for the line."
    ): {
        "semantic_unit_id": "CA0048::A01",
        "policy_ref_id": "CP0041::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    (
        "The agent must ask the user whether they want to refuel data or "
        "change to a plan with a higher data limit, if the line data "
        "usage has exceeded the plan data limit after the agent has "
        "confirmed cellular service, ruled out a roaming issue, and "
        "confirmed Mobile Data is on, before asking them to rerun the "
        "speed test or transferring to technical support."
    ): {
        "semantic_unit_id": "CA0052::A01",
        "policy_ref_id": "CP0045::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    (
        "The agent must ask the user to rerun the speed test and check "
        "data connectivity, and if there is still no connectivity, call "
        "transfer_to_human_agents(), if the line data usage has not "
        "exceeded the plan data limit after the agent has confirmed "
        "cellular service, ruled out a roaming issue, and confirmed "
        "Mobile Data is on."
    ): {
        "semantic_unit_id": "CA0009::A01",
        "policy_ref_id": "CP0005::P02",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    # docs/agentcoveragetesting_reuse_log.md section 125: telecom_083_order#b2-
    # #b5 (session-authored, item 8's second half -- Path 3 (MMS) full-path
    # coverage, following up on section 99's #b1 which only covered
    # tech_support_workflow.md Steps 3.1/3.2). Same real gap shape as section
    # 124's telecom_078_order#b2-#b6: real, targeted search of
    # policy_semantic_unit_model.json (telecom) found real, RECOVERED
    # semantic units for the individual ATOMIC actions each of these Thens
    # names (CA0058::A01/CP0053::P02 for set_network_mode_preference() ->
    # 3G/4G/5G, CA0060::A01/CP0055::P01 for toggle_wifi_calling() -- the SAME
    # real unit telecom_070_order#b0's own entry above already uses,
    # CA0061::A01/CP0056::P01 for grant_app_permission(), CA0059::A01/
    # CP0054::P02 for reset_apn_settings()) -- these are already the basis of
    # separate, already-shipped telecom_066_order#b0/#b1, telecom_068_order#b0,
    # telecom_070_order#b0, telecom_071_order#b0, telecom_059_order#b0/
    # telecom_069_order#b0 branches elsewhere in the 178-branch corpus (real,
    # verified: grep of specs_telecom_..._conditions_matches.json). What none
    # of those isolated atomic-action branches test is the SEQUENCING/gating
    # property specific to tech_support_workflow.md's Path 3 decision tree
    # (Step 3.1 -> 3.2 -> 3.3 -> 3.4 -> 3.5 -> 3.6, each step's action gated on
    # the outcome of the prior one) -- that is what these 4 new
    # telecom_083_order ORDER-kind siblings add, same "before troubleshooting
    # MMS issues"/"before checking X" ordering shape #b1 already established
    # for Steps 3.1/3.2. Each entry below cites the closest REAL recovered
    # unit as the natural atomic action this ordering constraint composes
    # (ASSERTED linkage, not an exact-text recovery -- same discipline as the
    # #b1/#b0 entries and section 124's #b2-#b6 entries above, none of whose
    # real semantic units themselves express any ordering relative to the
    # other Path 3 steps).
    #
    # Unlike section 124's Path 2.1 remaining steps (a mix of USER-only
    # toggle_roaming()/toggle_data() and real AGENT-callable enable_roaming()/
    # get_data_usage()/refuel_data()), every one of Path 3's remaining-step
    # check+fix tools (check_network_mode_preference/set_network_mode_
    # preference, check_wifi_calling_status/toggle_wifi_calling,
    # check_app_permissions/grant_app_permission, check_apn_settings/
    # reset_apn_settings) lives exclusively on tau2's TelecomUserTools
    # (tau2/domains/telecom/user_tools.py, USER-only -- confirmed absent from
    # tools.py, the agent's own tool set) and the underlying fields
    # (network_mode_preference/wifi_calling_enabled/messaging app permissions/
    # active_apn_settings.mmsc_url) live only on TelecomUserDB (user_db.toml),
    # never on the agent-side TelecomDB (db.toml). So all 4 of these new
    # entries are guidance-only (no bound tool_call target is possible for
    # any of them), and -- the real, honest upside of that same structural
    # fact -- none of them can hit section 124.11's vacuous-pass problem
    # (get_data_usage() there was a real AGENT-callable tool that could read
    # a real fixture value contradicting the injected scenario prose; no
    # AGENT-callable tool here can ever read network_mode_preference/
    # wifi_calling_enabled/app permissions/mmsc_url at all, so the driver's
    # injected device-facts prose is structurally the only, self-consistent
    # source of truth the agent ever has for these 4 branches, same as
    # section 124's own non-vacuous #b2/#b3/#b4).
    (
        "The agent must guide the user to use set_network_mode_preference() "
        "to change the network mode to include at least 3G, 4G, or 5G and "
        "ask them to try sending an MMS message again, if the user's phone "
        "is connected to a 2G network only after the agent has confirmed "
        "cellular service is available and mobile data connectivity is "
        "working, before checking Wi-Fi Calling status, messaging app "
        "permissions, or APN settings."
    ): {
        "semantic_unit_id": "CA0058::A01",
        "policy_ref_id": "CP0053::P02",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    (
        "The agent must guide the user to use toggle_wifi_calling() to turn "
        "Wi-Fi Calling OFF and ask them to try sending an MMS message "
        "again, if Wi-Fi Calling is on after the agent has confirmed "
        "cellular service is available, mobile data connectivity is "
        "working, and the phone is connected to a network of 3G or higher, "
        "before checking messaging app permissions or APN settings."
    ): {
        "semantic_unit_id": "CA0060::A01",
        "policy_ref_id": "CP0055::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    (
        "The agent must guide the user to use grant_app_permission() to "
        "grant the messaging app's storage permission and ask them to try "
        "sending an MMS message again, if the messaging app is missing the "
        "storage permission after the agent has confirmed cellular service "
        "is available, mobile data connectivity is working, the phone is "
        "connected to a network of 3G or higher, and Wi-Fi Calling is off "
        "or turning it off did not resolve the issue, before checking APN "
        "settings."
    ): {
        "semantic_unit_id": "CA0061::A01",
        "policy_ref_id": "CP0056::P01",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
    (
        "The agent must guide the user to use reset_apn_settings() to "
        "reset the APN settings to carrier defaults and ask them to try "
        "sending an MMS message again, if the MMSC URL is missing from the "
        "APN settings after the agent has confirmed cellular service is "
        "available, mobile data connectivity is working, the phone is "
        "connected to a network of 3G or higher, and the messaging app has "
        "both the storage and SMS permissions granted."
    ): {
        "semantic_unit_id": "CA0059::A01",
        "policy_ref_id": "CP0054::P02",
        "semantic_unit_kind": "action_commitment",
        "candidate_origin": "then_positive_guidance_claim_assertion",
    },
}
_POSITIVE_GUIDANCE_CLAIM_ORIGINS = frozenset(
    entry["candidate_origin"] for entry in _POSITIVE_GUIDANCE_CLAIM_TARGETS.values()
)

# airline_133_state#b0v0/#b0v1: legacy_same_spec_hint's bindings for this spec exist
# (cancel_reservation/book_reservation/update_reservation_flights/
# update_reservation_baggages) but every one has an EMPTY params list -- this is not
# a case of _parameter_is_focal failing to recognize a paraphrase (that requires the
# binding data to exist), the data itself never named a parameter. Keyed by exact
# requirement_text (real Then, transcribed from assembly.json), same discipline as
# every other exact-text table in this codebase (see _LOOKUP_TARGET_ACTIONS lesson).
_PAYMENT_REUSE_TARGETS: dict[str, list[tuple[str, str]]] = {
    "The agent must not use a payment method from the cancelled reservation for a "
    "new booking.": [("book_reservation", "payment_methods")],
    "The agent must not use a payment method from the cancelled reservation for a "
    "modification referencing the cancelled reservation.": [
        ("update_reservation_flights", "payment_id"),
        ("update_reservation_baggages", "payment_id"),
    ],
}

# airline_032_arg#b0/033_arg#b0/081_order#b0: the legacy binding data itself names
# the relevant parameter, but the branch's real Given/When/Then/rule_text contains
# no reliable textual anchor for it at all -- 032 only says "collect payment" (no
# "payment id"/"payment method"), 033 never mentions payment/refund-method wording
# anywhere, and 081's "passengers" only appears as singular "passenger" in prose
# (adding "passenger" as a _PARAMETER_PARAPHRASES entry for "passengers" was
# verified to also wrongly activate on the 11 airline_040-046 baggage-allowance
# branches, which already have their own correct mechanism -- see
# docs/oracle_requirement_pipeline_v0_7.md). Stretching _parameter_is_focal to match
# these would mean inventing an anchor that isn't really there, so these three use
# the same exact-Then-text fallback discipline as _PAYMENT_REUSE_TARGETS instead.
_EXPLICIT_ARGUMENT_FALLBACK_TARGETS: dict[str, list[tuple[str, str]]] = {
    "The agent must collect payment from the user for the price difference before "
    "updating the reservation's cabin class.": [
        ("update_reservation_flights", "payment_id")
    ],
    "The agent must refund the user the difference between the original price and "
    "the new, lower price after the cabin change in the reservation.": [
        ("update_reservation_flights", "payment_id")
    ],
    "The agent must collect the first name, last name, and date of birth for each "
    "passenger before booking the reservation.": [("book_reservation", "passengers")],
    # airline_082_norm#b0: "checked bags" here never contains the exact
    # _PARAMETER_PARAPHRASES phrasing ("total number of checked bags"/"non-free
    # checked bags"), so _parameter_is_focal never fires -- this is why the
    # full-catalog scan never produced a tool_argument candidate for this
    # branch despite total_baggages being a real parameter on both tools (see
    # docs/oracle_requirement_pipeline_v0_7.md §20). The WHEN covers both
    # booking and modifying, so both tools are listed as targets.
    "The agent must not add checked bags to the reservation.": [
        ("book_reservation", "total_baggages"),
        ("update_reservation_baggages", "total_baggages"),
    ],
    # airline_113_state#b1: this branch's Given/Then only ever exercise an
    # invalid DESTINATION ("The destination airport code does not correspond
    # to an airport currently served by the airline."). The spec-level
    # rule_text names both "origin and destination", so the full-catalog scan
    # (_full_catalog_bindings) produces one composite tool_argument_constraint
    # covering both fields on search_direct_flight -- broader than what this
    # specific branch's Given/Then ever commits to (there is no sibling
    # branch under this spec that tests an invalid origin). Adding this exact
    # single-field fallback gives the pipeline a properly-scoped alternative;
    # the pre-existing composite candidate's judgment is flipped to "no" in
    # favor of this one (see docs/oracle_requirement_pipeline_v0_7.md §21).
    # Neither version currently compiles to an executable Step4 predicate --
    # search_direct_flight's origin/destination endpoints have no
    # allowed_values in the (frozen) tool catalog, and airport-network
    # membership isn't in any evaluator-extension rule table -- so this is a
    # scope-correctness fix only, with no runtime-behavior change today.
    "The agent must not call search_direct_flight with a destination airport "
    "code that is not in the airline's current network.": [
        ("search_direct_flight", "destination")
    ],
    # docs/agentcoveragetesting_reuse_log.md section 111 -- 10 telecom
    # branches audited as branch_has_no_oracle_content ("genuine_gap": a real
    # policy/tool hazard exists but nothing checks for it). These 2 entries
    # are the ones whose gap is that the full-catalog scan/lexical retrieval
    # never anchors the real focal parameter at all, needing an explicit
    # fallback the same way the airline entries above do (a real, findable
    # anchor named nowhere the automatic scan can reach it, not a fabricated
    # requirement):
    #
    # telecom_025_arg#b0: real main_policy.md, "Data Refueling" section --
    # "The maximum amount of data that can be refueled is 2GB." -- and the
    # real refuel_data(customer_id, line_id, gb_amount) tool
    # (tau2/domains/telecom/tools.py) only ever checks `gb_amount <= 0`; it
    # has no upper-bound check at all. The branch's own real gwt.then
    # ("no more than 2GB ... in a single refuel_data operation") never says
    # "gb_amount" and the rule_text's "2GB" phrasing does not trip
    # _parameter_is_focal's paraphrase matching, so no tool_argument
    # candidate binding this parameter was ever generated. Independently
    # judged: a first proposal binding this to the cumulative_sum_le
    # predicate (Family N, shared with telecom_119_state#b0's session-level
    # sibling claim) was correctly rejected on independent review -- summing
    # gb_amount across every call sharing a line_id would wrongly fail two
    # separate, individually-compliant sub-2GB calls, which this branch's
    # own "single ... operation" wording does not forbid. Bound instead to
    # the new, genuinely single-call argument_le_constant predicate (Family
    # P) via ARGUMENT_UPPER_BOUND_RULES in
    # configs/oracle_judgment_families/telecom_v0_1.json.
    #
    # telecom_113_order#b0: real main_policy.md ("You should try your best
    # to resolve the issue for the user before transferring the user to a
    # human agent.") and tech_support_workflow.md line 7 ("Make sure you try
    # all the relevant resolution steps before transferring the user to a
    # human agent."). The branch's own then_relevant_endpoint_retrieval
    # candidate only ever produces a bare tool_call existence probe on
    # transfer_to_human_agents -- present/absent semantics cannot express
    # "attempted the relevant steps first", which is a transcript-level
    # behavioral claim, not a fact about any single call in isolation.
    # Anchored on the real `summary` argument of the actual
    # transfer_to_human_agents(summary) call -- the same real anchor
    # telecom_032_arg#b0/telecom_033_arg#b0 already use for other
    # dialogue-behavior semantic_transcript_judgment checks on this same
    # tool -- and bound via a new SEMANTIC_TRANSCRIPT_JUDGMENT_RULES entry
    # in configs/oracle_judgment_families/telecom_v0_1.json.
    "The agent must refuel no more than 2GB of data in a single "
    "refuel_data operation.": [("refuel_data", "gb_amount")],
    "The agent must attempt all required troubleshooting steps and tool "
    "actions within their scope before transferring the user to a human "
    "agent.": [("transfer_to_human_agents", "summary")],
    # docs/agentcoveragetesting_reuse_log.md section 112: retail's 35-branch
    # genuine_gap authoring pass. _parameter_is_focal never fires for any of
    # these because each branch's real rule_text/then describes the real
    # constraint in prose ("the order being cancelled", "a new item of a
    # different product type") without literally naming the real tau2
    # tools.py parameter (order_id/new_item_ids/payment_method_id/item_ids)
    # the constraint actually binds to -- same "real anchor exists in the
    # tool catalog, but no textual anchor in the branch's own prose" gap
    # _EXPLICIT_ARGUMENT_FALLBACK_TARGETS exists for (see the airline_032/033
    # comment above).
    #
    # retail_025_arg#b0: real tau2 tools.py modify_pending_order_items looks
    # up new_item_id via self._get_variant(product_id, new_item_id) where
    # product_id is the OLD item's own product_id (line ~504-505) -- a
    # cross-product-type swap therefore already raises real ValueError
    # "Variant not found" (Variant._get_variant, tools.py line ~110) today;
    # bound via a new TOOL_CALL_OUTCOME_RULES entry in retail_v0_1.json.
    "The agent must not modify an item in a single pending order to a new "
    "item of a different product type.": [
        ("modify_pending_order_items", "new_item_ids")
    ],
    # retail_020_arg#b0v3: same real "Variant not found" guard as retail_025
    # above -- this sibling branch is the "product type" axis of the shared
    # retail_020 "nothing else can be modified" claim, and (unlike its
    # b0v0/b0v1/b0v2 siblings, whose delivery-date/recipient-name/order-total
    # claims name Order fields that have NO real tau2 data_model.py
    # counterpart at all) a cross-product-type item swap IS a real, reachable
    # action via modify_pending_order_items -- so this one axis gets its own
    # real mechanical check instead of the vacuous structurally_unreachable_
    # action treatment the other 3 siblings (and their own already-accepted
    # retail_020_arg#b0v4/v6/v7/v8 siblings) share.
    "The agent must not modify the product type of a pending order.": [
        ("modify_pending_order_items", "new_item_ids")
    ],
    # retail_044_arg#b0: real tau2 tools.py modify_pending_order_payment
    # raises ValueError "The new payment method should be different from the
    # current one" (line ~583-585) when the requested payment_method_id
    # equals order.payment_history[0].payment_method_id. Bound via a new
    # TOOL_CALL_OUTCOME_RULES entry (distinct key from the pre-existing
    # 'only choose a single payment method...' row, which is retail_023's
    # own, narrower arity claim -- see retail_023's disposition in section
    # 112, reclassified rather than bound here).
    "The agent must not modify the payment method of a pending order to the "
    "same payment method already set on the order.": [
        ("modify_pending_order_payment", "payment_method_id")
    ],
    # retail_049_arg#b0: real tau2 tools.py return_delivered_order_items
    # raises ValueError "Some item not found" (line ~707) when
    # item_ids.count(x) exceeds all_item_ids.count(x) for any item -- the
    # exact real mechanism enforcing "quantity returned must not exceed
    # quantity delivered". Bound via a new TOOL_CALL_OUTCOME_RULES entry,
    # reusing the same real match_groups shape (["Some item not found"],
    # "Non-delivered order cannot be returned") already used by the existing
    # rows 40/41 of retail_v0_1.json (different requirement_text, same real
    # tool/error family).
    "The agent must not process a return for a quantity of an item that "
    "exceeds the quantity of that item originally delivered in the order.": [
        ("return_delivered_order_items", "item_ids")
    ],
    # retail_026_arg#b0/#b1: real policy.md (~line 114) "The user must
    # provide a payment method to pay or receive refund of the price
    # difference. If the user provides a gift card, it must have enough
    # balance to cover the price difference." Both branches are real
    # sequencing/collection claims (payment method must be COLLECTED before
    # the tool call, not merely present as a call argument -- the real tool
    # already requires payment_method_id as a mandatory argument, so the
    # incremental, genuinely-checkable claim is about the conversation
    # collecting/confirming it, not the call's schema) -- bound to a new
    # SEMANTIC_TRANSCRIPT_JUDGMENT_RULES criterion in retail_v0_1.json,
    # anchored on the real payment_method_id argument of the real
    # modify_pending_order_items call, same anchoring discipline as
    # telecom_113 above.
    "The agent must collect a payment method from the user to pay or "
    "receive refund of the price difference before modifying items in the "
    "order.": [("modify_pending_order_items", "payment_method_id")],
    "The agent must collect a gift card from the user with enough balance "
    "to cover the price difference before modifying items in the order.": [
        ("modify_pending_order_items", "payment_method_id")
    ],
    # retail_060_order_exchange_delivered_order_items#b0v1 / retail_060_
    # order_modify_pending_order_items#b0v1: real policy.md (~line 84)
    # "Exchange or modify order tools can only be called once per order. Be
    # sure that all items to be changed are collected into a list before
    # making the tool call!!!" -- the "collect all items first" half of this
    # sentence is only meaningfully testable for the two real tools that
    # actually take an item_ids list (exchange_delivered_order_items/
    # modify_pending_order_items); the once-only half is already covered by
    # each tool's own #b0v0 sibling (AT_MOST_N_CALL_RULES, already accepted
    # in production for modify_pending_order_address/items/payment#b0v0).
    # Bound via a new SEMANTIC_TRANSCRIPT_JUDGMENT_RULES criterion, anchored
    # on the real item_ids argument, mirroring the existing 'The user must
    # confirm the order id and the list of items to be returned...' row
    # already in retail_v0_1.json.
    "The agent must not make a call to the exchange or modify order tools "
    "for an order without collecting all items to be changed into a single "
    "list.": [("exchange_delivered_order_items", "item_ids")],
    "The agent must not call the exchange or modify order tool for an "
    "order without collecting all items to be changed into a single list.": [
        ("modify_pending_order_items", "item_ids")
    ],
    # retail_101_order#b0v0/#b0v1: real policy.md (~line 24) "You should
    # transfer the user to a human agent if and only if the request cannot
    # be handled within the scope of your actions" + real transfer_to_
    # human_agents docstring ("Only transfer if... given the policy and the
    # available tools, you cannot solve the user's issue."). Same real
    # process/sequencing shape as telecom_113 above -- present/absent
    # semantics on transfer_to_human_agents alone cannot express "only after
    # exhausting relevant automated options", so bound the same way, to the
    # real summary argument of the real transfer_to_human_agents(summary)
    # call, via two new SEMANTIC_TRANSCRIPT_JUDGMENT_RULES criteria (b0v0/
    # b0v1 are two distinct real Then phrasings of the same underlying
    # claim, so each keeps its own criterion rather than being deduplicated
    # into one).
    "The agent must not call the transfer_to_human_agents tool without "
    "first attempting all relevant automated actions.": [
        ("transfer_to_human_agents", "summary")
    ],
    "The agent must not call the transfer_to_human_agents tool without "
    "confirming that the issue is truly out of scope.": [
        ("transfer_to_human_agents", "summary")
    ],
    # retail_075/081/088/092/094/099_state#b0 (the real order-ownership
    # family): real policy.md (~line 14) "You can only help one user per
    # conversation..., and must deny any requests for tasks related to any
    # other user." Each branch's own real rule_text says "The order being
    # <cancelled/exchanged/modified/returned> must belong to the
    # authenticated user making the request." (prose, no literal "order_id"
    # anchor) rather than retail_083/087/096's real rule_text ("The order_id
    # provided must belong to..."), which IS why _parameter_is_focal already
    # fires for those 3 siblings but not these 6 -- same real underlying
    # claim family, same real, already-proven-in-production prior_call_
    # consistency mechanism (retail_083_state#b0 is already accepted and
    # bound this exact way for get_order_details; retail_087/096 likewise
    # for user_id/get_user_details), just never given a textual anchor for
    # these 6 tools (cancel_pending_order/exchange_delivered_order_items/
    # modify_pending_order_address/modify_pending_order_items/modify_
    # pending_order_payment/return_delivered_order_items). Bound via 6 new
    # PRIOR_CALL_CONSISTENCY_RULES entries in retail_v0_1.json, each reusing
    # retail_083's exact real params shape (source_tools:
    # ["get_user_details"], result_path: "orders", check: "member_of_list",
    # target_arg: "order_id").
    "The agent must only cancel the pending order if it belongs to the "
    "authenticated user making the request.": [
        ("cancel_pending_order", "order_id")
    ],
    "The agent must only exchange items in a delivered order that belongs "
    "to the authenticated user making the request.": [
        ("exchange_delivered_order_items", "order_id")
    ],
    "The agent must only modify the shipping address of the order if it "
    "belongs to the authenticated user.": [
        ("modify_pending_order_address", "order_id")
    ],
    "The agent must modify items only in orders that belong to the "
    "authenticated user.": [("modify_pending_order_items", "order_id")],
    "The agent must only modify the payment method for a pending order if "
    "the order belongs to the authenticated user making the request.": [
        ("modify_pending_order_payment", "order_id")
    ],
    "The agent must process the return only if the order belongs to the "
    "authenticated user making the request.": [
        ("return_delivered_order_items", "order_id")
    ],
}

# airline_073_order#b0v0/v1: "must not make a tool call and send a message to
# the user at the same time" -- not a tool-argument value, not a specific
# assistant literal, and not a precedes relation between two ordinary events;
# it's a same-turn co-occurrence constraint. normalize_tau_messages tags every
# event derived from one raw message with the same source_message_index (see
# runtime_observation_binding_v1.py), so this is genuinely observable -- just
# not through any of the other 5 candidate kinds. Exact-Then-text set (no
# per-tool targets to emit, unlike the fallback tables above), each entry
# produces one turn_shape_constraint candidate (see
# docs/oracle_requirement_pipeline_v0_7.md §22 for the full design).
_TURN_SHAPE_FALLBACK_TARGETS: frozenset[str] = frozenset(
    {
        "The agent must not make a tool call and send a message to the user "
        "at the same time.",
        "The agent must not send a message to the user and make a tool call "
        "at the same time.",
        # Real gwt.then variants found by real-corpus search across telecom/
        # retail/airline (docs/agentcoveragetesting_reuse_log.md section 66) --
        # the original 2 entries only ever matched the one airline_073 case;
        # telecom_036_order_suspend_line#b0's own real then ("must not make a
        # suspend_line tool call and respond... in the same turn") fell through
        # this exact-match gate entirely and got a plain per-tool
        # tool_call/"absent" candidate instead, wrongly forbidding suspend_line
        # outright rather than only its same-turn co-occurrence with a message.
        # Same exact-match convention as every other fallback table in this
        # module (real, hand-verified per branch, not a fuzzy/regex match) --
        # just a real, larger verified set now, not a semantic generalization.
        "The agent must only make one tool call or one user response per "
        "turn, never both simultaneously.",
        "The agent must perform only one action per turn: either make a "
        "tool call or send a message to the user, but not both in the same "
        "turn.",
        "The agent must, in each turn, either make a tool call or respond "
        "to the user, but never both in the same turn.",
        "The agent must either make a tool call or send a message to the "
        "user in a single turn, but not both simultaneously.",
        "The agent must only make one tool call or send one message to the "
        "user at a time, never both simultaneously.",
        "The agent must, in each turn, either make a tool call or send a "
        "message to the user, but never both at the same time.",
        "The agent must, in each turn, either make a tool call or send a "
        "message to the user, but not both at the same time.",
        "The agent must not make a tool call and send a user message in "
        "the same turn.",
        "The agent must not send a user message and make a tool call in "
        "the same turn.",
        "The agent must not make a suspend_line tool call and respond to "
        "the user in the same turn; each action must be performed in a "
        "separate turn.",
        "The agent must, in each turn, either make one tool call or send a "
        "message to the user, but not both in the same turn.",
        "The agent must not make a tool call and send a message to the "
        "user in the same turn.",
        "The agent must not send a message to the user and make a tool "
        "call in the same turn.",
    }
)

# Discovered when the legacy_same_spec_hint-derived relevant_tools list was
# replaced (see docs/oracle_requirement_pipeline_v0_7.md) by
# then_relevant_endpoint_retrieval, a token-overlap match against tool/endpoint
# DESCRIPTION text: regenerating the full 155-branch corpus and diffing against
# the pre-change baseline surfaced 12 branches that lost a previously-BOUND
# (real, accepted) tool_call requirement, with exactly two root causes, both
# verified against real branch text before adding entries here:
#
# 1. cancel_reservation's own description ("Cancel the whole reservation.")
#    only contains "cancel" -- but tool_endpoint_lexical_retrieval_v1's simple
#    suffix-stripping stemmer does not reduce "cancelling"/"cancelled"/
#    "cancellation" (a doubled-consonant conjugation) down to "cancel", so it
#    never scores against Then text using those forms. tool_catalog_lexical_
#    retrieval has the identical limitation (it also never produces the bare
#    stem "cancel"). This affected 8 real branches, all verified below.
# 2. airline_031_arg#b0v0/v1/v2's Then text is only about origin/destination/
#    trip_type -- it never mentions "flights" at all, so neither lexical
#    mechanism nor description-overlap retrieval ranks update_reservation_
#    flights highly enough, even though it is the tool this Then is entirely
#    about (verified: the branches' target_action was
#    no_change_origin_dest_triptype under the old legacy hint).
#
# Fixing the shared stemmer's general double-consonant handling was considered
# and rejected for this round: it is reused by given_evidence_source_v2.py's
# already-verified Given-condition matching, and a broader stemming change
# risks touching that unrelated, working code for the sake of one verb. A
# small, exact-Then-text table is the same discipline already used elsewhere
# in this file for a hand-verified, no-reliable-automatic-anchor gap.
_TOOL_CALL_FALLBACK_TARGETS: dict[str, list[str]] = {
    "The agent must not update the reservation with a different origin.": [
        "update_reservation_flights"
    ],
    "The agent must not update the reservation with a different destination.": [
        "update_reservation_flights"
    ],
    "The agent must not update the reservation with a different trip type.": [
        "update_reservation_flights"
    ],
    "The agent must obtain the reason for cancellation from the user before "
    "cancelling the reservation.": ["cancel_reservation"],
    "The agent must not initiate any cancellation process for the "
    "reservation.": ["cancel_reservation"],
    "The agent must not update the reservation flights using "
    "update_reservation_flights on a cancelled reservation.": ["cancel_reservation"],
    "The agent must not update the reservation baggages using "
    "update_reservation_baggages on a cancelled reservation.": ["cancel_reservation"],
    "The agent must not update the reservation passengers using "
    "update_reservation_passengers on a cancelled reservation.": ["cancel_reservation"],
    "The agent must not call get_reservation_details for the cancelled "
    "reservation.": ["cancel_reservation"],
    "The agent must not use a payment method from the cancelled reservation "
    "for a new booking.": ["cancel_reservation"],
    "The agent must not use a payment method from the cancelled reservation "
    "for a modification referencing the cancelled reservation.": [
        "cancel_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
    ],
    "The agent may offer a certificate as a gesture after confirming the "
    "facts, with the amount being $100 times the number of passengers.": [
        "send_certificate"
    ],
    "The agent may offer a certificate as a gesture after confirming the "
    "facts and changing or cancelling the reservation, with the amount "
    "being $50 times the number of passengers.": ["send_certificate"],
    "The agent must use a flight_number that matches the airline's standard "
    "format: two-letter airline code followed by 1 to 4 digits (e.g., "
    "'AA1234').": ["get_flight_status"],
    # airline_097_order#b0: this Then never says "certificate"/"send", so
    # neither then_relevant_endpoint_retrieval's description-overlap scoring
    # nor tool_catalog_lexical_retrieval's tool-name matching ever discovers
    # send_certificate -- same root cause class as the cancel_reservation
    # conjugation gaps above, verified against the real Then text (see
    # docs/oracle_requirement_pipeline_v0_7.md section 24). Without this
    # candidate the branch has no tool_call anchor for "offering
    # compensation" at all, unlike its ARG-family siblings (037/038_arg#b0)
    # which already have one.
    "The agent must confirm the facts before offering compensation.": [
        "send_certificate"
    ],
    # airline_095_norm#b0: same root cause as 097 above -- this Then never
    # says "certificate"/"send" either ("must not offer compensation to the
    # user without the user explicitly requesting it"), so no lexical or
    # description-overlap mechanism discovers send_certificate. Without this
    # candidate the branch has no tool_call anchor for "offering
    # compensation" at all -- all three of its accepted requirements are
    # semantic_requirement candidates restating the same "must not offer
    # compensation/a certificate" idea with no bound observation behind any
    # of them (see docs/oracle_requirement_pipeline_v0_7.md section 27).
    "The agent must not offer compensation to the user without the user "
    "explicitly requesting it.": ["send_certificate"],
    # airline_099_norm#b0: same root cause as 095/097 above -- this Then
    # never says "certificate"/"send" either ("must not offer compensation
    # for a reason not listed in the policy"), so no lexical or
    # description-overlap mechanism discovers send_certificate. Without this
    # candidate the branch had literally ZERO accepted requirements (all 4
    # candidates then_relevant_endpoint_retrieval did generate --
    # book_reservation/get_user_details/list_all_airports/search_onestop_flight
    # -- were correctly judged "no", none of them are about compensation at
    # all), only surfaced by Step5/6 v2's stricter "every branch needs >=1
    # check" completeness requirement (see docs/v5_step6_plans_v0_2.md). Real
    # policy lineage confirmed: NPU0021::C01::R01::A01's focal_relation_quote
    # is the exact sentence "No compensation may be offered for any other
    # reason." -- the same CA0047/CA0050/CA0051 compensation policy family
    # 037/038/095/097/098 already draw from, this is its negative/else
    # completion (every OTHER reason not already covered by the $100/$50
    # cancelled/delayed cases).
    "The agent must not offer compensation for a reason not listed in the "
    "policy.": ["send_certificate"],
    # docs/agentcoveragetesting_reuse_log.md section 112: 5 real retail
    # branches whose mechanical tool_catalog_lexical_retrieval/then_relevant_
    # endpoint_retrieval candidate locks onto the WRONG real tool because of
    # lexical overlap between retail's order-action tool names (cancel_
    # pending_order/exchange_delivered_order_items/return_delivered_order_
    # items all share "order"/"delivered"/"items" tokens with each other).
    # Verified against real tau2 tools.py for each: only return_delivered_
    # order_items/exchange_delivered_order_items actually implement the
    # status=="delivered" (retail_063_state#e0, retail_102_state#b3/b4,
    # retail_104_state#b0v3) or "called once per order" (retail_060_order_
    # exchange_delivered_order_items#b0v0's own Then, about exchange_
    # delivered_order_items itself, not modify_pending_order_items) guard the
    # branch's own real Then is actually about.
    "The agent must not initiate the return process for the order.": [
        "return_delivered_order_items"
    ],
    "The agent must not process a return for a cancelled order.": [
        "return_delivered_order_items"
    ],
    "The agent must not process an exchange for a cancelled order.": [
        "exchange_delivered_order_items"
    ],
    "The agent must not return items from the cancelled order.": [
        "return_delivered_order_items"
    ],
    "The agent must not make more than one call to the exchange or modify "
    "order tools for the same order.": ["exchange_delivered_order_items"],
    # docs/agentcoveragetesting_reuse_log.md section 128: retail_099_state#b1
    # (a real adversarial "must deny" sibling of retail_099_state#b0, whose
    # own target tool is return_delivered_order_items). Same real root cause
    # class as the section 112 entries just above -- verified directly with
    # _relevant_endpoints(tool_catalog, then, limit=8): this exact Then text
    # scores cancel_pending_order's endpoints (order_id/reason) higher than
    # return_delivered_order_items's own (order_id/item_ids/...) purely on
    # generic shared tokens ("order"/"the"/"request"), for every rewording
    # tried (including "must not return items from the order..."), so this is
    # not fixable by choosing different Then wording -- return_delivered_
    # order_items never even appears in the top-8 ranking. Verified against
    # real tau2 tools.py: return_delivered_order_items is the only tool this
    # branch's real claim (retail_099_state's rule_text, "The order being
    # returned must belong to the authenticated user making the request.")
    # is about.
    "The agent must not process the return, because the order does not "
    "belong to the authenticated user making the request.": [
        "return_delivered_order_items"
    ],
}

# airline_083_order#b0/airline_097_order#b0: both Thens describe "the agent
# does X before the requested operation" -- exactly the shape _prerequisite_units
# already turns into a temporal_relation candidate instead of a plain
# semantic_requirement -- but the real Step2 assessment data never supplies a
# "prerequisite_event" candidate for either branch's target semantic unit, for
# two DIFFERENT reasons (see docs/oracle_requirement_pipeline_v0_7.md section
# 24 for the full investigation against the real frozen
# execution_matching_contracts.json):
#
# - airline_097_order#b0 (CA0048::A01, "confirm the facts (observations of the
#   current case)"): the real config DOES have exactly the right
#   prerequisite_monitor (OPF0005.prerequisite_monitors[*] with
#   prerequisite_monitor_id "PM::DR::CPD0011", required_prior_events naming
#   action_semantic_unit_id "CA0048::A01", policy_ref_ids ["CP0027::P01"] --
#   verified verbatim against releases/agentspectesting-method-v1.1.0's
#   upstream_artifacts/airline/execution_matching_contracts.json). It is
#   silently dropped by an internal Step2-assessment scoring cutoff (candidates
#   scoring more than 0.5 below the best-scoring candidate are dropped with no
#   diagnostic), not missing from the source data. The runtime_contract_refs
#   entry below is a verbatim transcription of that real monitor.
# - airline_083_order#b0 (CA0021::A01, "ask the user if they want to purchase
#   travel insurance"): no prerequisite_monitor anywhere in the same real
#   config file has policy_ref_ids including CP0018::P01 (083's own policy) --
#   confirmed by checking every one of OPF0001 (book_reservation)'s five
#   prerequisite_monitors; none references CP0018. This is a genuine data gap,
#   not a scoring dropout. operation_policy_family_id is hand-asserted as
#   OPF0001 -- book_reservation's real, verified operation-family id in the
#   same file -- because that is the ONLY field
#   runtime_observation_binding_v1.py's temporal_relation branch actually
#   reads out of runtime_contract_refs (see lines ~405-419: it builds
#   family_id -> operation_families[family_id] and ignores every other key).
#   There is no real contract_kind/contract_id/event_requirement_id to cite for
#   083, so none is included -- inventing one would misrepresent this as
#   recovered data instead of the asserted judgment call it actually is.
#
# Consumed by the per-branch loop below, which merges this into the SAME
# `prerequisites` dict the real exact_prerequisites/spec_prerequisites data
# already populates -- reusing the existing single-emission-point logic that
# already decides temporal_relation vs semantic_requirement for one semantic
# unit, rather than adding a second, duplicate candidate for the same unit_id.
_TEMPORAL_RELATION_FALLBACK_TARGETS: dict[str, dict[str, Any]] = {
    "The agent must ask the user if they want to buy travel insurance before "
    "booking the reservation.": {
        "left_event_semantic_unit_id": "CA0021::A01",
        "runtime_contract_refs": [{"operation_policy_family_id": "OPF0001"}],
        "candidate_origin": "then_temporal_prerequisite_assertion",
    },
    "The agent must confirm the facts before offering compensation.": {
        "left_event_semantic_unit_id": "CA0048::A01",
        "runtime_contract_refs": [
            {
                "contract_kind": "prerequisite_monitor",
                "contract_id": "PM::DR::CPD0011",
                "event_requirement_id": "ER::PM::DR::CPD0011::CA0048::A01",
                "operation_policy_family_id": "OPF0005",
            }
        ],
        "candidate_origin": "then_temporal_prerequisite_recovery",
    },
    # airline_037_arg#b0 / airline_038_arg#b0: same real prerequisite_monitor
    # (PM::DR::CPD0011, OPF0005) as airline_097_order#b0 above -- verified
    # directly against execution_matching_contracts.json's OPF0005 operation
    # family contract, whose operation_selector.verified_tool_names is
    # ["send_certificate"] and whose scope statement ("handle refunds and
    # compensation") is generic to the whole family, not scoped to any one
    # branch. 037/038's own real candidate pool already generates a
    # semantic_requirement for this exact unit (source_refs.semantic_unit_id
    # "CA0048::A01", policy_ref_id "CP0027::P01" -- identical lineage to 097's),
    # it just never gets classified as a prerequisite because the same Step2
    # scoring cutoff that dropped it for 097 dropped it here too. Unlike 097
    # (a `required`-mode branch, so the resulting temporal_relation compiles
    # straight to operator="relation_true"), 037/038 are `permitted`-mode
    # ("may offer... after confirming the facts") with a When that never
    # explicitly requests offering compensation, so the frozen expectation
    # compiler's own permission-resolution logic would otherwise collapse
    # this candidate's operator to "non_decisive" like every other requirement
    # in these two branches -- see the operator override in
    # oracle_expectation_compiler_extension_v1.py's
    # _upgrade_conditional_temporal_relations for why that is wrong here and
    # how it is fixed.
    "The agent may offer a certificate as a gesture after confirming the "
    "facts, with the amount being $100 times the number of passengers.": {
        "left_event_semantic_unit_id": "CA0048::A01",
        "runtime_contract_refs": [
            {
                "contract_kind": "prerequisite_monitor",
                "contract_id": "PM::DR::CPD0011",
                "event_requirement_id": "ER::PM::DR::CPD0011::CA0048::A01",
                "operation_policy_family_id": "OPF0005",
            }
        ],
        "candidate_origin": "then_temporal_prerequisite_recovery",
    },
    "The agent may offer a certificate as a gesture after confirming the "
    "facts and changing or cancelling the reservation, with the amount "
    "being $50 times the number of passengers.": {
        "left_event_semantic_unit_id": "CA0048::A01",
        "runtime_contract_refs": [
            {
                "contract_kind": "prerequisite_monitor",
                "contract_id": "PM::DR::CPD0011",
                "event_requirement_id": "ER::PM::DR::CPD0011::CA0048::A01",
                "operation_policy_family_id": "OPF0005",
            }
        ],
        "candidate_origin": "then_temporal_prerequisite_recovery",
    },
}

# airline_077_order#b0: structurally different from the two entries above --
# the Then is "first call transfer_to_human_agents, THEN send this exact
# message", so right_event is a specific MESSAGE LITERAL, not "the requested
# operation" (there is no operation after the tool call; the tool call IS the
# left event, the message IS the right event). _prerequisite_units's existing
# per-unit emission always hardcodes observation_contract["right_event"] =
# "requested_operation" (see the loop below), so this can't reuse that
# injection trick the way _TEMPORAL_RELATION_FALLBACK_TARGETS does -- it needs
# its own directly-added candidate, exact-Then-text keyed like
# _TURN_SHAPE_FALLBACK_TARGETS. right_event is set to the sentinel string
# "assistant_message_literal" (any non-empty string satisfies
# _classify's truthiness check -- see oracle_requirement_acceptance_v1.py; the
# frozen binding compiler never reads this string's actual content, only
# runtime_contract_refs, so reusing this field for a second sentinel value is
# safe) with the real literal text carried separately in
# right_event_literal, for runtime_observation_binding_extension_v1.py's new
# message-literal override to read. left_event_semantic_unit_id (CA0009::A01,
# "make a tool call to transfer_to_human_agents") and the message literal
# itself are both transcribed verbatim from this branch's real candidates
# (OC06/OC09 in the real corpus -- see
# docs/oracle_requirement_pipeline_v0_7.md section 24).
# runtime_contract_refs still needs to be non-empty for _classify's gate, but
# its content is never actually consulted for this branch (the override
# below fully replaces right_event's binding instead of resolving it via
# operation_policy_family_id/family_candidates) -- OPF0006 is included anyway,
# as an honest, real, verified reference (transfer_to_human_agents's own
# operation-family id in the real execution_matching_contracts.json), not a
# fabricated one, even though it is functionally inert here.
_TEMPORAL_MESSAGE_LITERAL_FALLBACK_TARGETS: dict[str, dict[str, Any]] = {
    "The agent must first make a tool call to transfer_to_human_agents, and "
    "then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. "
    "PLEASE HOLD ON.' to the user.": {
        "left_event_semantic_unit_id": "CA0009::A01",
        "right_event_literal": (
            "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
        ),
        "runtime_contract_refs": [{"operation_policy_family_id": "OPF0006"}],
        "candidate_origin": "then_temporal_message_literal_fallback",
    },
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _fingerprinted(value: dict[str, Any]) -> dict[str, Any]:
    value["packet_fingerprint"] = content_sha256(value)
    return value


def _stem(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    # Tool names contain ordinary plurals such as ``baggages`` and
    # ``passengers``.  Removing only the final ``s`` keeps their lexical stem
    # aligned with singular prose (``baggage`` / ``passenger``); a generic
    # ``-es`` rule would incorrectly turn ``baggages`` into ``baggag``.
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {_stem(token) for token in _TOKEN.findall(text.casefold()) if token not in _STOP}


# _parameter_is_focal below only recognizes a parameter's literal schema spelling
# by design (see its docstring) -- this is a deliberately tiny, explicit exception
# list for parameters verified (against a real branch) to be described by a
# paraphrase instead of the schema name. airline_036_arg#b0's real Then is "must
# collect exactly one payment or refund method, which must be either a gift card
# or a credit card..." -- it never says "payment_id" or "payment id", so the
# existing exact match alone leaves payment_id an unfocused scope parameter and no
# tool_argument/tool_argument_constraint candidate is ever generated for it (see
# docs/oracle_requirement_pipeline_v0_7.md for the full incident). This is not a
# general natural-language relaxation -- broadening the match generically was
# exactly the mistake _LOOKUP_TARGET_ACTIONS made in then_structural_classifier_v1.py
# (a coarse proxy standing in for real verification); each entry here must be
# individually verified against a real branch before being added.
_PARAMETER_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "payment_id": ("payment method", "refund method"),
    # airline_034_arg#b0v0's then says "...reduce the total number of checked
    # bags."; #b0v1's then says "...reduce the number of non-free checked bags."
    # -- verified the two phrasings don't cross-match each other's branch (each
    # branch's own then only contains its own parameter's phrase), so this stays
    # narrow per-branch even though both parameters share one paraphrase table.
    "total_baggages": ("total number of checked bags",),
    "nonfree_baggages": ("non-free checked bags", "number of non-free checked bags"),
    # airline_080_order#b0's then says "...ask for the trip type, origin, and
    # destination..." -- "trip type" never matches the literal schema spelling
    # "flight type", so the composite constraint's parameters list was missing
    # flight_type entirely (only origin/destination), even though constraint_text
    # promised all three (found by an independent post-hoc audit of the real
    # accepted_requirements.json, see docs/oracle_requirement_pipeline_v0_7.md).
    # Verified across the full 155-branch corpus: only 4 branches mention "trip
    # type" at all (this one, and airline_031_arg#b0v0/v1/v2, whose target tool
    # update_reservation_flights has no flight_type argument at all -- tools.py
    # confirms it only accepts reservation_id/cabin/flights/payment_id -- so this
    # paraphrase cannot spuriously bind anything on those branches).
    "flight_type": ("trip type",),
}


_QUOTED_IDENTIFIER = re.compile(r"'([a-z][a-z0-9_]*)'")


def _parameter_is_focal(
    parameter: str, *texts: str, real_parameter_names: frozenset[str] = frozenset()
) -> bool:
    """Return whether a binding field is explicitly part of the tested predicate.

    An identifier such as ``reservation_id`` is not made focal merely because
    prose mentions a reservation.  The complete schema name (underscore or
    whitespace spelling) must occur in the rule or current Then.  Otherwise it
    remains an execution-scope binding for the enclosing tool action. A small,
    explicit set of verified paraphrases (_PARAMETER_PARAPHRASES) is also
    recognized for the same reason.

    When a text single-quotes one or more tokens that are themselves real
    catalog parameter names (the convention rule_text/then generation uses
    when a sentence's subject is an exact parameter, e.g. "Each element in
    'item_ids' must be an item id such as '1008292230'." -- 'item_ids' is a
    real parameter, '1008292230' is just an example value, not a schema
    name), that quoting is treated as authoritative for that text: only an
    exact quoted-parameter match counts, and other real parameters' loose
    humanized spellings are not considered for that text. Without this, the
    loose humanized-spelling fallback below can accidentally match an
    unrelated tool's parameter whose spelling coincides with ordinary prose
    elsewhere in the same sentence -- e.g. the singular "item id" inside
    "...must be an item id such as..." spells out a *different* real tool's
    singular item_id parameter, even though the sentence's actual, quoted
    subject is the plural 'item_ids'. `real_parameter_names` is deliberately
    required to make this real-parameter filter possible -- a naive "any
    quoted lowercase token" version is *not* safe: quoted example values
    like 'sara_doe_496' or 'AA1234' are common in real rule_text and are not
    parameter references at all (verified against the real airline
    155-branch corpus, where treating them as such wrongly stripped several
    already-correct bindings). Real telecom/retail candidates verified the
    false-binding pattern this narrows (see
    docs/agentcoveragetesting_reuse_log.md section 24); quoting only ever
    narrows a match, never widens one, so this cannot introduce a new
    binding that didn't already exist.
    """

    spellings = {parameter.casefold(), parameter.replace("_", " ").casefold()}
    spellings.update(p.casefold() for p in _PARAMETER_PARAPHRASES.get(parameter, ()))
    real_lower = {p.casefold() for p in real_parameter_names}
    for text in texts:
        lowered = str(text or "").casefold()
        quoted = set(_QUOTED_IDENTIFIER.findall(lowered)) & real_lower
        if quoted:
            if parameter.casefold() in quoted:
                return True
            continue
        for spelling in spellings:
            if re.search(rf"(?<![a-z0-9_]){re.escape(spelling)}(?![a-z0-9_])", lowered):
                return True
    return False


# Explicit, per-phrase-verified action-phrase -> tool scoping. A handful of
# real telecom domain_knowledge rule_texts (docs/agentcoveragetesting_reuse_log.md
# section 30) name exactly ONE real parameter (line_id) that many real tools
# share (disable_roaming/enable_roaming/get_data_usage/refuel_data/
# resume_line/suspend_line all really do take line_id), so neither
# _parameter_is_focal's exact-name match nor the section-24 multi-real-
# parameter disqualifier (which only fires when 2+ DIFFERENT real parameters
# are co-named) can narrow the full-catalog scan down -- but the sentence's
# own wording names a specific ACTION that only one of those tools performs
# (e.g. "before it can be suspended" is only ever about suspend_line; the
# real ValueError text for this check is real tau2 telecom source's
# suspend_line-only "Line must be active to suspend", not something
# disable_roaming/get_data_usage/etc. ever raise). Each phrase below was
# verified against real tau2 telecom tools.py to confirm only the named
# tool's own real code path matches it. Matching is substring-on-rule_text,
# so it only ever narrows an existing full-catalog scan down to fewer tools
# -- it cannot introduce a binding that didn't already exist, and it has no
# effect at all on any rule_text that doesn't contain one of these exact
# phrases (verified to be absent from airline's real 155-branch corpus).
_ACTION_PHRASE_TOOL_SCOPE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("before it can be suspended", ("suspend_line",)),
    ("before it can be resumed", ("resume_line",)),
    ("before roaming can be enabled", ("enable_roaming",)),
    ("before roaming can be disabled", ("disable_roaming",)),
    # retail_027_arg (docs/agentcoveragetesting_reuse_log.md section 32):
    # real rule_text scopes this to the return flow ("...when returning
    # delivered order items"), and the real GWT's when/then confirm the
    # subject is return_delivered_order_items specifically -- but the shared
    # real parameter order_id let it leak onto all 7 order_id-taking tools
    # (get_order_details, modify_pending_order_payment, etc., none of which
    # have anything to do with "returning" items).
    ("when returning delivered order items", ("return_delivered_order_items",)),
    # retail_020/045/068/069/070 (docs/agentcoveragetesting_reuse_log.md
    # section 34): five more real retail claims that never bind to any
    # tool that's actually correct for them -- each currently only produces
    # candidates on modify_pending_order_address/modify_user_address (which
    # don't even take a payment_method_id parameter, and have nothing to do
    # with "pending order" scope limits or cancellation/payment-modification
    # outcomes), with no real alternative candidate already being generated
    # anywhere else in the pool. None of these name a parameter their real,
    # semantically-correct tool (cancel_pending_order,
    # modify_pending_order_payment) could be tested on either, so an empty
    # tool tuple here suppresses the noise entirely rather than guessing a
    # redirect that likely wouldn't produce a real candidate anyway.
    (
        "you can only modify its shipping address, payment method, or product item options",
        (),
    ),
    ("the payment method must be eligible for the order's currency and country.", ()),
    ("the order status will be changed to 'cancelled'", ()),
    ("the order status will be kept as 'pending'", ()),
    (
        "the original payment method will be refunded immediately if it is a gift card, "
        "otherwise it will be refunded within",
        (),
    ),
)


def _action_phrase_scope_tools(rule_text: str) -> frozenset[str] | None:
    """Returns the real target tool(s) for a rule_text matching one of
    _ACTION_PHRASE_TOOL_SCOPE's phrases, or None if no phrase matches (no
    restriction). Applied identically everywhere a rule_text's parameters get
    projected onto a tool set -- both _full_catalog_bindings' own per-tool
    scan below, and build_oracle_requirement_packets' separate
    tool_catalog_parameter_projection path (driven by _tool_candidates'
    lexical retrieval, which independently re-discovers a tool like
    disable_roaming via token overlap with "roaming" even when this phrase's
    real subject is only ever enable_roaming) -- confirmed both paths need
    it: narrowing only _full_catalog_bindings left the lexical-retrieval path
    re-introducing the exact same wrong-tool candidates on its own. See
    docs/agentcoveragetesting_reuse_log.md section 30.
    """
    lowered = rule_text.casefold()
    for phrase, tool_names in _ACTION_PHRASE_TOOL_SCOPE:
        if phrase in lowered:
            return frozenset(tool_names)
    return None


def _full_catalog_bindings(
    tool_map: Mapping[str, Mapping[str, Any]], rule_text: str = ""
) -> list[dict[str, Any]]:
    """Default source of `(tool, params)` bindings once a spec supplies none of its
    own (`spec.get("bindings") is None`): one record per catalog tool, covering
    every tool_argument endpoint it exposes. This widens WHICH (tool, parameter)
    pairs get checked against `_parameter_is_focal` below -- it does not change
    HOW they get checked; the exact-text/paraphrase gate is exactly as narrow as
    before. Replaces a prior dependency on `legacy_same_spec_hint` (a pre-narrowed
    binding list borrowed from an older, unrelated method release) -- see
    docs/oracle_requirement_pipeline_v0_7.md for why that dependency doesn't
    generalize to a new agent/domain and the real numbers behind this change.

    `rule_text` is optional and, when it matches an entry in
    _ACTION_PHRASE_TOOL_SCOPE (via _action_phrase_scope_tools), narrows the
    scanned tool set to just that phrase's real target tool(s) instead of
    every catalog tool.
    """
    scope = _action_phrase_scope_tools(rule_text)
    if scope is not None:
        tool_map = {name: tool_map[name] for name in scope if name in tool_map}
    bindings = []
    for tool_name, tool in tool_map.items():
        params = [
            endpoint.get("source_path")
            for endpoint in tool.get("observable_endpoints") or []
            if endpoint.get("source_kind") == "tool_argument"
            and isinstance(endpoint.get("source_path"), str)
        ]
        if params:
            bindings.append({"tool": tool_name, "params": params})
    return bindings


def _all_real_parameter_names(tool_map: Mapping[str, Mapping[str, Any]]) -> frozenset[str]:
    """Every tool_argument parameter name that exists anywhere in the real
    catalog, across every tool -- used only to detect when a rule_text/then
    verbatim-names two or more *different* tools' real parameters at once
    (see _mentioned_real_parameters / its use in build_oracle_requirement_packets).
    """
    return frozenset(
        endpoint.get("source_path")
        for tool in tool_map.values()
        for endpoint in tool.get("observable_endpoints") or []
        if endpoint.get("source_kind") == "tool_argument"
        and isinstance(endpoint.get("source_path"), str)
    )


def _mentioned_real_parameters(text: str, all_real_parameters: frozenset[str]) -> frozenset[str]:
    """Which of the catalog's real parameter names appear verbatim (exact
    underscored schema spelling, not the humanized/paraphrase forms
    _parameter_is_focal also accepts) in a rule_text/then string.

    Deliberately exact-spelling-only and not per-tool: this is used to
    detect a specific real failure pattern where a rule_text verbatim-names
    two DIFFERENT real parameters that belong to two DIFFERENT tools (e.g.
    "The line_id must belong to the customer specified by customer_id." --
    both are real telecom parameters, but only tools owning *both* can be
    the real subject; a tool that owns just one, such as
    get_bills_for_customer owning only customer_id and not line_id, is not
    a real candidate even though it does own one of the two named
    identifiers). See docs/agentcoveragetesting_reuse_log.md section 24.
    """
    lowered = str(text or "").casefold()
    return frozenset(
        parameter for parameter in all_real_parameters
        if re.search(rf"(?<![a-z0-9_]){re.escape(parameter.casefold())}(?![a-z0-9_])", lowered)
    )


def _tool_type(tool: Mapping[str, Any]) -> str | None:
    source = str((tool.get("implementation") or {}).get("source_text") or "")
    match = re.search(r"ToolType\.([A-Z_]+)", source)
    return match.group(1).casefold() if match else None


def _assessment_index(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for raw in document.get("items") or []:
        item = _mapping(raw, "$.assessment.items[]")
        ref = _mapping(item.get("selected_spec_ref"), "$.selected_spec_ref")
        branch_id = ref.get("branch_id")
        if isinstance(branch_id, str) and branch_id:
            result[branch_id] = _mapping(item.get("assessment"), "$.assessment")
    return result


def _policy_refs_by_spec(
    assessments: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for branch_id, assessment in assessments.items():
        spec_id = branch_id.split("#", 1)[0]
        values = result.setdefault(spec_id, [])
        for policy_ref in _policy_refs(assessment):
            if policy_ref not in values:
                values.append(policy_ref)
    return result


def _prerequisite_units(
    assessment: Mapping[str, Any] | None,
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    if not assessment:
        return result
    report = (assessment.get("target_resolution_report") or {}).get(
        "runtime_projection_report"
    ) or {}
    for candidate in report.get("candidates") or []:
        if candidate.get("candidate_kind") != "prerequisite_event":
            continue
        refs = [
            deepcopy(dict(ref))
            for ref in candidate.get("runtime_contract_refs") or []
            if isinstance(ref, Mapping)
        ]
        for unit_id in candidate.get("semantic_unit_ids") or []:
            if isinstance(unit_id, str) and unit_id:
                result.setdefault(unit_id, []).extend(refs)
    return result


def _prerequisite_units_by_spec(
    assessments: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    result: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for branch_id, assessment in assessments.items():
        spec_id = branch_id.split("#", 1)[0]
        spec_units = result.setdefault(spec_id, {})
        for unit_id, refs in _prerequisite_units(assessment).items():
            known = spec_units.setdefault(unit_id, [])
            known_ids = {
                (item.get("contract_kind"), item.get("contract_id")) for item in known
            }
            for ref in refs:
                identity = (ref.get("contract_kind"), ref.get("contract_id"))
                if identity not in known_ids:
                    known.append(ref)
                    known_ids.add(identity)
    return result


def _policy_refs(assessment: Mapping[str, Any] | None) -> list[str]:
    if not assessment:
        return []
    report = assessment.get("target_resolution_report") or {}
    lineage = report.get("policy_lineage_report") or {}
    refs = []
    for candidate in lineage.get("candidates") or []:
        value = candidate.get("policy_ref_id")
        if isinstance(value, str) and value and value not in refs:
            refs.append(value)
    return refs


def _semantic_units_by_policy(document: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for raw in document.get("semantic_units") or []:
        unit = _mapping(raw, "$.semantic_units[]")
        policy_ref = unit.get("policy_ref_id")
        if isinstance(policy_ref, str) and policy_ref:
            result.setdefault(policy_ref, []).append(unit)
    return result


def _unit_text(unit: Mapping[str, Any]) -> str:
    payload = unit.get("semantic_payload") or {}
    for key in ("direct_action_text", "focal_relation_quote", "source_requirement_text", "policy_statement"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _unit_modality(unit: Mapping[str, Any]) -> str | None:
    value = (unit.get("semantic_payload") or {}).get("modality")
    return value if isinstance(value, str) and value else None


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    contract = candidate.get("observation_contract") or {}
    tool = contract.get("tool_name")
    parameter = contract.get("parameter")
    literal = contract.get("literal")
    if candidate.get("candidate_kind") == "tool_argument":
        # Must include requirement_text (like tool_argument_constraint below),
        # not just (tool, parameter) -- otherwise a generic full-catalog-scan
        # candidate (spec_binding, requirement_text=rule_text) and a targeted
        # exact-Then-text fallback candidate (e.g. _PAYMENT_REUSE_TARGETS,
        # requirement_text=then) that happen to point at the same (tool,
        # parameter) collide on this key, and whichever _add_candidate call
        # runs first silently wins -- the other is dropped with no trace. This
        # is exactly what ate airline_133_state#b0v0/#b0v1's
        # then_payment_reuse_fallback candidates once §15's full-catalog scan
        # started also matching their (tool, parameter) via the branch's own
        # rule_text (see docs/oracle_requirement_pipeline_v0_7.md §19).
        return f"tool_argument::{tool}::{parameter}::{candidate.get('requirement_text')}"
    if candidate.get("candidate_kind") == "tool_argument_constraint":
        parameters = ",".join(contract.get("parameters") or [])
        return f"tool_argument_constraint::{tool}::{parameters}::{candidate.get('requirement_text')}"
    if candidate.get("candidate_kind") == "tool_call":
        return f"tool_call::{tool}"
    if candidate.get("candidate_kind") == "assistant_literal":
        return f"assistant_literal::{literal}"
    unit_id = (candidate.get("source_refs") or {}).get("semantic_unit_id")
    if unit_id:
        if candidate.get("candidate_origin") in _MESSAGE_CLAIM_LINEAGE_ORIGINS | _POSITIVE_GUIDANCE_CLAIM_ORIGINS:
            # Same collision class the tool_argument/tool_argument_constraint
            # branches above already guard against, hit for real here: a
            # semantic unit can be the real lineage for TWO genuinely
            # different observable claims in the same branch (e.g.
            # CA0044::A01's broad "the agent cannot help" action-level
            # prohibition, already a real candidate via the ordinary per-unit
            # loop, vs. this _MESSAGE_CLAIM_LINEAGE_TARGETS-injected candidate
            # asserting the narrower, distinct "must not SAY the reservation
            # is being cancelled" message-content claim for
            # airline_092_state#b0v1). Keying on unit_id alone would silently
            # drop the second one at _add_candidate -- confirmed empirically
            # (candidate_count dropped 1103->1102 before this fix). Scoped to
            # only these two lineage-injection origins, not every
            # unit-id-keyed candidate corpus-wide, so the other ~80 branches
            # whose candidates also key off a bare unit_id are byte-for-byte
            # unaffected. See docs/oracle_requirement_pipeline_v0_7.md
            # section 34.
            #
            # _POSITIVE_GUIDANCE_CLAIM_ORIGINS (docs/agentcoveragetesting_
            # reuse_log.md section 99) is the exact same collision class hit
            # again for real: telecom_078_order#b1's asserted lineage cites
            # IA0016::A01 (CP0037's "refer to the guide" action commitment)
            # as the closest real unit, but the branch's own ordinary
            # per-unit loop ALSO independently proposes a candidate for that
            # same unit_id (the existing "refer to the guide" restatement,
            # requirement_text-distinct from this Then-text candidate) --
            # confirmed empirically (candidate silently dropped, pool count
            # unchanged at 18, before this fix; 19 after).
            return f"semantic_unit::{unit_id}::{candidate.get('requirement_text')}"
        return f"semantic_unit::{unit_id}"
    return f"{candidate.get('candidate_kind')}::{candidate.get('requirement_text')}"


def _add_candidate(pool: list[dict[str, Any]], seen: set[str], candidate: dict[str, Any]) -> None:
    key = _candidate_key(candidate)
    if key in seen:
        return
    seen.add(key)
    candidate["candidate_key"] = key
    pool.append(candidate)


def _tool_candidates(
    spec: Mapping[str, Any],
    gwt: Mapping[str, Any],
    tool_map: Mapping[str, Mapping[str, Any]],
    tool_catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    relevant_tools = spec.get("relevant_tools")
    names: list[tuple[str, str]] = []
    for name in relevant_tools or []:
        if isinstance(name, str) and name in tool_map:
            names.append((name, "spec_relevant_tool"))
    then = str(gwt.get("then") or "")
    for match in _EXPLICIT_TOOL.finditer(then):
        name = match.group(1)
        if name in tool_map:
            names.append((name, "explicit_then_tool"))
    then_tokens = _tokens(then)
    for name, tool in tool_map.items():
        name_tokens = _tokens(name.replace("_", " "))
        if name_tokens and name_tokens <= then_tokens:
            names.append((name, "tool_catalog_lexical_retrieval"))
    if relevant_tools is None:
        # Default source once a spec supplies no explicit relevant_tools list
        # (i.e. no longer sourced from legacy_same_spec_hint): token-overlap
        # retrieval against tool/endpoint DESCRIPTION text (not just the tool's
        # own name, unlike tool_catalog_lexical_retrieval above) -- the same,
        # already-tested mechanism given_evidence_source_v2.py uses for
        # Given-condition text, reused here for Then text. See
        # docs/oracle_requirement_pipeline_v0_7.md for why relevant_tools could
        # not simply be dropped without a replacement (88/155 real branches
        # would get zero tool_call candidates otherwise).
        endpoints, _total, _action_names = _relevant_endpoints(tool_catalog, then, limit=8)
        seen_names = {name for name, _ in names}
        for endpoint in endpoints:
            name = endpoint.get("tool_name")
            if isinstance(name, str) and name in tool_map and name not in seen_names:
                names.append((name, "then_relevant_endpoint_retrieval"))
                seen_names.add(name)
        for name in _TOOL_CALL_FALLBACK_TARGETS.get(then, ()):
            if name in tool_map and name not in seen_names:
                names.append((name, "then_tool_fallback"))
                seen_names.add(name)
    result = []
    for name, origin in names:
        tool = tool_map[name]
        result.append(
            {
                "candidate_kind": "tool_call",
                "candidate_origin": origin,
                "requirement_text": f"Observe whether the agent calls {name}.",
                "modality_hint": None,
                "observation_contract": {
                    "channel": "tool_call",
                    "tool_name": name,
                    "tool_type": _tool_type(tool),
                    "action_description": str(tool.get("description") or "").strip(),
                },
                "source_refs": {"tool_name": name},
            }
        )
    return result


def build_oracle_requirement_packets(
    spec_document: Mapping[str, Any],
    assessment_document: Mapping[str, Any],
    semantic_unit_document: Mapping[str, Any],
    tool_catalog: Mapping[str, Any],
    sut_profile: Mapping[str, Any],
    *,
    selected_branch_ids: Sequence[str],
) -> dict[str, Any]:
    profile = validate_sut_adapter_profile(sut_profile)
    assessments = _assessment_index(assessment_document)
    spec_policy_refs = _policy_refs_by_spec(assessments)
    spec_prerequisites = _prerequisite_units_by_spec(assessments)
    units_by_policy = _semantic_units_by_policy(semantic_unit_document)
    tool_map = {
        item["tool_name"]: item
        for item in tool_catalog.get("tools") or []
        if isinstance(item, Mapping) and isinstance(item.get("tool_name"), str)
    }
    all_real_parameters = _all_real_parameter_names(tool_map)
    branch_map: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for raw_spec in spec_document.get("specs") or []:
        spec = _mapping(raw_spec, "$.specs[]")
        for raw_gwt in spec.get("gwt") or []:
            gwt = _mapping(raw_gwt, "$.gwt[]")
            branch_id = gwt.get("branch_id")
            if isinstance(branch_id, str) and branch_id:
                if branch_id in branch_map:
                    raise ThenAtomizationError(f"duplicate GWT branch: {branch_id}")
                branch_map[branch_id] = (spec, gwt)
    selected = list(selected_branch_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ThenAtomizationError("v0.7 selected branch IDs must be non-empty and unique")
    missing = [branch_id for branch_id in selected if branch_id not in branch_map]
    if missing:
        raise ThenAtomizationError(f"v0.7 selected branches are missing: {missing}")

    packets = []
    branch_summaries = []
    for branch_id in selected:
        spec, gwt = branch_map[branch_id]
        assessment = assessments.get(branch_id)
        pool: list[dict[str, Any]] = []
        seen: set[str] = set()

        source_parameter_groups: list[list[str]] = []
        source_scope_parameters: set[str] = set()
        scope_parameters_by_tool: dict[str, list[str]] = {}
        bindings = spec.get("bindings")
        used_full_catalog_bindings = bindings is None
        if bindings is None:
            bindings = _full_catalog_bindings(tool_map, str(spec.get("rule_text") or ""))
        # Only the full-catalog scan (one record per catalog tool, regardless
        # of which tool a rule_text/then is really about) is exposed to this
        # cross-tool disqualifier -- an explicit spec.bindings list is already
        # a curated, narrow selection and is left exactly as before.
        mentioned_real_parameters: frozenset[str] = frozenset()
        if used_full_catalog_bindings:
            mentioned_real_parameters = (
                _mentioned_real_parameters(str(spec.get("rule_text") or ""), all_real_parameters)
                | _mentioned_real_parameters(str(gwt.get("then") or ""), all_real_parameters)
            )
        for binding in bindings:
            tool_name = binding.get("tool")
            if tool_name not in tool_map:
                continue
            endpoints = {
                endpoint.get("source_path"): endpoint
                for endpoint in tool_map[tool_name].get("observable_endpoints") or []
                if endpoint.get("source_kind") == "tool_argument"
            }
            bound_parameters = [
                parameter for parameter in binding.get("params") or []
                if isinstance(parameter, str) and parameter
            ]
            if len(mentioned_real_parameters) >= 2 and not mentioned_real_parameters <= set(bound_parameters):
                # This rule_text/then verbatim-names two or more real
                # parameters that (together) span more than one tool -- e.g.
                # "The line_id must belong to the customer specified by
                # customer_id." names both telecom's line_id and customer_id.
                # A tool that owns only one of them (e.g.
                # get_bills_for_customer, which owns customer_id but has no
                # line_id parameter at all) is not a real candidate for this
                # claim even though it owns one of the named identifiers --
                # see docs/agentcoveragetesting_reuse_log.md section 24 for
                # the real telecom/retail false positives this fixes.
                continue
            parameters = [
                parameter for parameter in bound_parameters
                if _parameter_is_focal(
                    parameter,
                    str(spec.get("rule_text") or ""),
                    str(gwt.get("then") or ""),
                    real_parameter_names=all_real_parameters,
                )
            ]
            scope_parameters = [
                parameter for parameter in bound_parameters if parameter not in parameters
            ]
            if scope_parameters:
                scope_parameters_by_tool.setdefault(tool_name, []).extend(scope_parameters)
                source_scope_parameters.update(scope_parameters)
            if parameters:
                source_parameter_groups.append(parameters)
            if len(parameters) > 1:
                _add_candidate(
                    pool,
                    seen,
                    {
                        "candidate_kind": "tool_argument_constraint",
                        "candidate_origin": "spec_binding_composite",
                        "requirement_text": str(spec.get("rule_text") or ""),
                        "modality_hint": None,
                        "observation_contract": {
                            "channel": "tool_call",
                            "tool_name": tool_name,
                            "parameters": parameters,
                            "action_description": str(
                                tool_map[tool_name].get("description") or ""
                            ).strip(),
                            "endpoints": {
                                parameter: deepcopy(endpoints.get(parameter))
                                for parameter in parameters
                            },
                            "constraint_text": str(spec.get("rule_text") or ""),
                            "scope_parameters": scope_parameters,
                            "scope_endpoints": {
                                parameter: deepcopy(endpoints.get(parameter))
                                for parameter in scope_parameters
                            },
                        },
                        "source_refs": {"binding": deepcopy(dict(binding))},
                    },
                )
                continue
            for parameter in parameters:
                endpoint = endpoints.get(parameter)
                _add_candidate(
                    pool,
                    seen,
                    {
                        "candidate_kind": "tool_argument",
                        "candidate_origin": "spec_binding",
                        "requirement_text": str(spec.get("rule_text") or ""),
                        "modality_hint": None,
                        "observation_contract": {
                            "channel": "tool_call",
                            "tool_name": tool_name,
                            "parameter": parameter,
                            "action_description": str(
                                tool_map[tool_name].get("description") or ""
                            ).strip(),
                            "endpoint": deepcopy(endpoint) if endpoint else None,
                            "scope_parameters": scope_parameters,
                            "scope_endpoints": {
                                item: deepcopy(endpoints.get(item))
                                for item in scope_parameters
                            },
                        },
                        "source_refs": {"binding": deepcopy(dict(binding))},
                    },
                )

        generated_tool_candidates = _tool_candidates(spec, gwt, tool_map, tool_catalog)
        for candidate in generated_tool_candidates:
            contract = candidate["observation_contract"]
            tool_name = contract["tool_name"]
            endpoints = {
                endpoint.get("source_path"): endpoint
                for endpoint in tool_map[tool_name].get("observable_endpoints") or []
                if endpoint.get("source_kind") == "tool_argument"
            }
            scope_parameters = list(scope_parameters_by_tool.get(tool_name, []))
            if candidate["candidate_origin"] in {
                "explicit_then_tool",
                "tool_catalog_lexical_retrieval",
                "then_relevant_endpoint_retrieval",
                "then_tool_fallback",
            }:
                scope_parameters.extend(
                    parameter
                    for parameter in sorted(source_scope_parameters)
                    if parameter in endpoints and parameter not in scope_parameters
                )
            if scope_parameters:
                contract["scope_parameters"] = scope_parameters
                contract["scope_endpoints"] = {
                    parameter: deepcopy(endpoints.get(parameter))
                    for parameter in scope_parameters
                }
            _add_candidate(pool, seen, candidate)

        # A current branch may correct or specialize a stale top-level tool
        # binding.  If a mechanically retrieved tool exposes the same named
        # parameters, propose those endpoints; the GWT binary task decides
        # whether they belong to this branch.
        source_parameters = {
            parameter for group in source_parameter_groups for parameter in group
        }
        projected_tools = {
            item["observation_contract"]["tool_name"]
            for item in generated_tool_candidates
            if item["candidate_origin"] in {
                "explicit_then_tool",
                "tool_catalog_lexical_retrieval",
                "then_relevant_endpoint_retrieval",
                "then_tool_fallback",
            }
        }
        action_phrase_scope = _action_phrase_scope_tools(str(spec.get("rule_text") or ""))
        if action_phrase_scope is not None:
            projected_tools &= action_phrase_scope
        for tool_name in sorted(projected_tools):
            endpoints = {
                endpoint.get("source_path"): endpoint
                for endpoint in tool_map[tool_name].get("observable_endpoints") or []
                if endpoint.get("source_kind") == "tool_argument"
            }
            for group in source_parameter_groups:
                if not set(group) <= set(endpoints):
                    continue
                if len(group) > 1:
                    _add_candidate(
                        pool,
                        seen,
                        {
                            "candidate_kind": "tool_argument_constraint",
                            "candidate_origin": "tool_catalog_parameter_projection_composite",
                            "requirement_text": str(spec.get("rule_text") or ""),
                            "modality_hint": None,
                            "observation_contract": {
                                "channel": "tool_call",
                                "tool_name": tool_name,
                                "parameters": group,
                                "action_description": str(
                                    tool_map[tool_name].get("description") or ""
                                ).strip(),
                                "endpoints": {
                                    parameter: deepcopy(endpoints[parameter])
                                    for parameter in group
                                },
                                "constraint_text": str(spec.get("rule_text") or ""),
                                "scope_parameters": [
                                    parameter
                                    for parameter in sorted(source_scope_parameters)
                                    if parameter in endpoints
                                ],
                                "scope_endpoints": {
                                    parameter: deepcopy(endpoints[parameter])
                                    for parameter in sorted(source_scope_parameters)
                                    if parameter in endpoints
                                },
                            },
                            "source_refs": {
                                "projected_from_spec_parameters": sorted(source_parameters)
                            },
                        },
                    )
                    continue
                parameter = group[0]
                _add_candidate(
                    pool,
                    seen,
                    {
                        "candidate_kind": "tool_argument",
                        "candidate_origin": "tool_catalog_parameter_projection",
                        "requirement_text": str(spec.get("rule_text") or ""),
                        "modality_hint": None,
                        "observation_contract": {
                            "channel": "tool_call",
                            "tool_name": tool_name,
                            "parameter": parameter,
                            "action_description": str(
                                tool_map[tool_name].get("description") or ""
                            ).strip(),
                            "endpoint": deepcopy(endpoints[parameter]),
                            "scope_parameters": [
                                item
                                for item in sorted(source_scope_parameters)
                                if item in endpoints
                            ],
                            "scope_endpoints": {
                                item: deepcopy(endpoints[item])
                                for item in sorted(source_scope_parameters)
                                if item in endpoints
                            },
                        },
                        "source_refs": {
                            "projected_from_spec_parameters": sorted(source_parameters)
                        },
                    },
                )

        exact_policy_refs = _policy_refs(assessment)
        policy_refs = exact_policy_refs or spec_policy_refs.get(str(spec.get("spec_id")), [])
        lineage_origin = (
            "coverage_semantic_unit"
            if exact_policy_refs
            else "coverage_semantic_unit_via_sibling_lineage"
        )
        exact_prerequisites = _prerequisite_units(assessment)
        prerequisites = exact_prerequisites or spec_prerequisites.get(
            str(spec.get("spec_id")), {}
        )
        temporal_relation_fallback = _TEMPORAL_RELATION_FALLBACK_TARGETS.get(
            str(gwt.get("then") or "")
        )
        fallback_prerequisite_unit_id = None
        if (
            temporal_relation_fallback is not None
            and temporal_relation_fallback["left_event_semantic_unit_id"]
            not in prerequisites
        ):
            fallback_prerequisite_unit_id = temporal_relation_fallback[
                "left_event_semantic_unit_id"
            ]
            prerequisites = {
                **prerequisites,
                fallback_prerequisite_unit_id: temporal_relation_fallback[
                    "runtime_contract_refs"
                ],
            }
        for policy_ref in policy_refs:
            for unit in units_by_policy.get(policy_ref, []):
                text = _unit_text(unit)
                if not text:
                    continue
                unit_id = unit.get("semantic_unit_id")
                is_prerequisite = unit_id in prerequisites
                candidate_kind = (
                    "temporal_relation" if is_prerequisite else "semantic_requirement"
                )
                candidate_origin = lineage_origin
                observation_contract: dict[str, Any] = {
                    "channel": None,
                    "channel_binding_status": "deferred",
                }
                if is_prerequisite:
                    text = (
                        f"Whether the agent performs this behavior before the requested "
                        f"operation: {text}."
                    )
                    if unit_id == fallback_prerequisite_unit_id:
                        candidate_origin = temporal_relation_fallback["candidate_origin"]
                    else:
                        candidate_origin = (
                            "coverage_runtime_prerequisite"
                            if exact_prerequisites
                            else "coverage_runtime_prerequisite_via_sibling_lineage"
                        )
                    observation_contract.update(
                        {
                            "relation": "precedes",
                            "left_event_semantic_unit_id": unit_id,
                            "right_event": "requested_operation",
                            "runtime_contract_refs": deepcopy(prerequisites[unit_id]),
                        }
                    )
                _add_candidate(
                    pool,
                    seen,
                    {
                        "candidate_kind": candidate_kind,
                        "candidate_origin": candidate_origin,
                        "requirement_text": text,
                        "modality_hint": _unit_modality(unit),
                        "observation_contract": observation_contract,
                        "source_refs": {
                            "policy_ref_id": policy_ref,
                            "semantic_unit_id": unit.get("semantic_unit_id"),
                            "semantic_unit_kind": unit.get("semantic_unit_kind"),
                        },
                    },
                )

        then = str(gwt.get("then") or "")
        for match in _QUOTED.finditer(then):
            literal = match.group(2)
            if len(literal.split()) < 3:
                continue
            _add_candidate(
                pool,
                seen,
                {
                    "candidate_kind": "assistant_literal",
                    "candidate_origin": "quoted_then_literal",
                    "requirement_text": f"The assistant message contains: {literal}",
                    "modality_hint": None,
                    "observation_contract": {
                        "channel": "assistant_message",
                        "literal": literal,
                    },
                    "source_refs": {"exact_then_span": match.group(0)},
                },
            )

        # A Then that reads "must not send a message ... and make a tool call
        # ... at/in the same turn" (retail_057_order#b0v1, airline_073_order#b0v1)
        # matches BOTH this regex (via "must not send a/the message") AND the
        # _TURN_SHAPE_FALLBACK_TARGETS exact-text set below -- without this
        # guard, both blocks fire independently and produce two candidates
        # with byte-identical requirement_text but different candidate_kind
        # (semantic_requirement vs turn_shape_constraint), which
        # oracle_requirement_acceptance_v1.py's equivalence-key grouping never
        # merges (different kind-prefixed keys), so nothing downstream ever
        # resolves the collision on its own -- see
        # docs/agentcoveragetesting_reuse_log.md section 109 for the real
        # retail_057_order#b0v1 case this fixes and the real
        # airline_073_order#b0v1 sibling verified to already be judged clean
        # (turn_shape only) under the pre-existing judgment data, so this
        # guard is a zero-behavior-change no-op for that branch. The
        # turn_shape_constraint candidate strictly subsumes what this
        # semantic_requirement fallback would claim here (same text, plus a
        # concrete, directly-observable same-turn co-occurrence mechanism vs.
        # an unbound, lineage-less semantic claim), so suppressing the
        # fallback candidate only for a Then that is ALSO an exact
        # _TURN_SHAPE_FALLBACK_TARGETS match never affects the two hand-
        # verified _MESSAGE_CLAIM_LINEAGE_TARGETS entries (airline_047_arg#b0,
        # airline_092_state#b0v1) -- neither of their Then texts is a
        # turn-shape Then.
        if _MESSAGE_CLAIM_PROHIBITION.search(then) and then not in _TURN_SHAPE_FALLBACK_TARGETS:
            lineage = _MESSAGE_CLAIM_LINEAGE_TARGETS.get(then)
            _add_candidate(
                pool,
                seen,
                {
                    "candidate_kind": "semantic_requirement",
                    "candidate_origin": (
                        lineage["candidate_origin"]
                        if lineage is not None
                        else "then_message_claim_fallback"
                    ),
                    "requirement_text": then,
                    "modality_hint": None,
                    "observation_contract": {
                        "channel": None,
                        "channel_binding_status": "deferred",
                    },
                    "source_refs": (
                        {
                            "policy_ref_id": lineage["policy_ref_id"],
                            "semantic_unit_id": lineage["semantic_unit_id"],
                            "semantic_unit_kind": lineage["semantic_unit_kind"],
                        }
                        if lineage is not None
                        else {}
                    ),
                },
            )

        positive_guidance = _POSITIVE_GUIDANCE_CLAIM_TARGETS.get(then)
        if positive_guidance is not None:
            _add_candidate(
                pool,
                seen,
                {
                    "candidate_kind": "semantic_requirement",
                    "candidate_origin": positive_guidance["candidate_origin"],
                    "requirement_text": then,
                    "modality_hint": None,
                    "observation_contract": {
                        "channel": None,
                        "channel_binding_status": "deferred",
                    },
                    "source_refs": {
                        "policy_ref_id": positive_guidance["policy_ref_id"],
                        "semantic_unit_id": positive_guidance["semantic_unit_id"],
                        "semantic_unit_kind": positive_guidance["semantic_unit_kind"],
                    },
                },
            )

        def _emit_fallback_argument_candidates(targets, origin):
            # Shared by every exact-Then-text fallback table in this module
            # (_PAYMENT_REUSE_TARGETS, _EXPLICIT_ARGUMENT_FALLBACK_TARGETS): reuses
            # the same endpoint lookup the legacy-binding path already uses. Two
            # targets for the same Then (e.g. "a modification" covering both
            # update_reservation_flights and update_reservation_baggages) would
            # otherwise share one requirement_text -- oracle_decision_reconciliation_v8's
            # same_requirement_duplicate dedup key is requirement_text alone, so two
            # genuinely different (different-tool) candidates would collide and one
            # would be silently dropped. Appending the tool name keeps them distinct
            # without changing what the judge is asked (that reads the branch's own
            # gwt.then, not this field).
            for tool_name, parameter in targets:
                if tool_name not in tool_map:
                    continue
                endpoints = {
                    endpoint.get("source_path"): endpoint
                    for endpoint in tool_map[tool_name].get("observable_endpoints") or []
                    if endpoint.get("source_kind") == "tool_argument"
                }
                endpoint = endpoints.get(parameter)
                requirement_text = (
                    then if len(targets) == 1 else f"{then} ({tool_name})"
                )
                _add_candidate(
                    pool,
                    seen,
                    {
                        "candidate_kind": "tool_argument",
                        "candidate_origin": origin,
                        "requirement_text": requirement_text,
                        "modality_hint": None,
                        "observation_contract": {
                            "channel": "tool_call",
                            "tool_name": tool_name,
                            "parameter": parameter,
                            "action_description": str(
                                tool_map[tool_name].get("description") or ""
                            ).strip(),
                            "endpoint": deepcopy(endpoint) if endpoint else None,
                            "scope_parameters": [],
                            "scope_endpoints": {},
                        },
                        "source_refs": {},
                    },
                )

        _emit_fallback_argument_candidates(
            _PAYMENT_REUSE_TARGETS.get(then, ()), "then_payment_reuse_fallback"
        )
        _emit_fallback_argument_candidates(
            _EXPLICIT_ARGUMENT_FALLBACK_TARGETS.get(then, ()),
            "then_explicit_argument_fallback",
        )

        if then in _TURN_SHAPE_FALLBACK_TARGETS:
            _add_candidate(
                pool,
                seen,
                {
                    "candidate_kind": "turn_shape_constraint",
                    "candidate_origin": "then_turn_shape_fallback",
                    "requirement_text": then,
                    "modality_hint": None,
                    "observation_contract": {"channel": None},
                    "source_refs": {},
                },
            )

        message_literal_fallback = _TEMPORAL_MESSAGE_LITERAL_FALLBACK_TARGETS.get(then)
        if message_literal_fallback is not None:
            _add_candidate(
                pool,
                seen,
                {
                    "candidate_kind": "temporal_relation",
                    "candidate_origin": message_literal_fallback["candidate_origin"],
                    "requirement_text": (
                        f"Whether the agent performs this behavior before sending the "
                        f"message: {message_literal_fallback['right_event_literal']}."
                    ),
                    "modality_hint": None,
                    "observation_contract": {
                        "channel": None,
                        "channel_binding_status": "deferred",
                        "relation": "precedes",
                        "left_event_semantic_unit_id": message_literal_fallback[
                            "left_event_semantic_unit_id"
                        ],
                        "right_event": "assistant_message_literal",
                        "right_event_literal": message_literal_fallback["right_event_literal"],
                        "runtime_contract_refs": deepcopy(
                            message_literal_fallback["runtime_contract_refs"]
                        ),
                    },
                    "source_refs": {},
                },
            )

        # Full-catalog scanning (see _full_catalog_bindings) can now find the
        # SAME focal parameter/parameter-set on more than one tool for one
        # branch (e.g. payment_id paraphrase-matching on both
        # update_reservation_flights and update_reservation_baggages), each
        # candidate sharing one requirement_text (spec.rule_text or the exact
        # Then). oracle_decision_reconciliation_v8's same_requirement_duplicate
        # dedup key is normalized requirement_text alone -- undisambiguated,
        # two genuinely different (different-tool) candidates collide and one
        # is silently dropped (this is exactly the bug the "(tool_name)" suffix
        # in _emit_fallback_argument_candidates above was built to avoid for
        # airline_133_state#b0v1, but that fix only fires for its own two
        # exact-Then-text fallback tables -- the generic full-catalog-derived
        # candidates need the same treatment, generally, whenever it actually
        # happens, not just for those two hand-picked tables). Skip a
        # candidate whose text already carries a "(tool_name)" suffix so an
        # already-disambiguated fallback candidate is not double-suffixed.
        by_requirement_text: dict[str, list[dict[str, Any]]] = {}
        for candidate in pool:
            if candidate["candidate_kind"] in ("tool_argument", "tool_argument_constraint"):
                by_requirement_text.setdefault(candidate["requirement_text"], []).append(candidate)
        for text, group in by_requirement_text.items():
            tool_names = {
                candidate["observation_contract"].get("tool_name") for candidate in group
            }
            if len(tool_names) <= 1:
                continue
            for candidate in group:
                tool_name = candidate["observation_contract"].get("tool_name")
                suffix = f" ({tool_name})"
                if candidate["requirement_text"].endswith(suffix):
                    continue
                candidate["requirement_text"] = f"{candidate['requirement_text']}{suffix}"
                candidate["candidate_key"] = _candidate_key(candidate)

        if not pool:
            _add_candidate(
                pool,
                seen,
                {
                    "candidate_kind": "unbound_branch_assertion",
                    "candidate_origin": "gwt_then_fallback",
                    "requirement_text": then,
                    "modality_hint": None,
                    "observation_contract": {
                        "channel": None,
                        "channel_binding_status": "unbound",
                    },
                    "source_refs": {},
                },
            )

        branch_context = {
            "branch_id": branch_id,
            "spec_id": spec.get("spec_id"),
            "kind": spec.get("kind"),
            "origin": spec.get("origin"),
            "rule_text": spec.get("rule_text"),
            "target_action": spec.get("target_action"),
            "gwt": {
                "given": gwt.get("given"),
                "when": gwt.get("when"),
                "then": gwt.get("then"),
            },
        }
        for index, candidate in enumerate(pool, start=1):
            candidate = {"candidate_id": f"OC{index:02d}", **candidate}
            task_input = {
                "branch_context": deepcopy(branch_context),
                "observable_channels": deepcopy(profile["observable_channels"]),
                "proposed_observation": deepcopy(candidate),
            }
            packets.append(
                _fingerprinted(
                    {
                        "schema_version": PACKET_VERSION,
                        "task_name": TASK_NAME,
                        "branch_id": branch_id,
                        "candidate_id": candidate["candidate_id"],
                        "task_contract": {
                            "question": "Is proposed_observation one of the things the oracle should check for this GWT Then?",
                            "output_decision": "yes | no | ambiguous",
                            "deferred_decisions": [
                                "observation channel binding when absent",
                                "runtime evaluator implementation",
                                "fixture construction",
                                "prompt generation",
                            ],
                        },
                        "task_input": task_input,
                    }
                )
            )
        branch_summaries.append(
            {
                "branch_id": branch_id,
                "candidate_count": len(pool),
                "assessment_available": assessment is not None,
                "coverage_policy_refs": policy_refs,
                "candidate_origins": sorted({item["candidate_origin"] for item in pool}),
            }
        )

    result = {
        "schema_version": PACKET_SET_VERSION,
        "selected_branch_ids": selected,
        "sut_profile_id": profile["profile_id"],
        "branch_summaries": branch_summaries,
        "packets": packets,
        "summary": {
            "branch_count": len(selected),
            "candidate_count": len(packets),
            "expected_model_calls": len(packets),
            "llm_generated_candidates": 0,
            "llm_decisions_per_call": 1,
        },
    }
    result["packet_set_fingerprint"] = content_sha256(result)
    return result


def validate_oracle_requirement_packet_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$")))
    fingerprint = result.pop("packet_set_fingerprint", None)
    if result.get("schema_version") != PACKET_SET_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid v0.7 oracle requirement packet set")
    packets = result.get("packets")
    if not isinstance(packets, list):
        raise ThenAtomizationError("v0.7 packets must be an array")
    identities = []
    for raw in packets:
        packet = dict(_mapping(raw, "$.packets[]"))
        packet_fingerprint = packet.pop("packet_fingerprint", None)
        if packet.get("schema_version") != PACKET_VERSION or packet.get("task_name") != TASK_NAME:
            raise ThenAtomizationError("v0.7 packet schema/task mismatch")
        if packet_fingerprint != content_sha256(packet):
            raise ThenAtomizationError("v0.7 packet fingerprint mismatch")
        task_input = _mapping(packet.get("task_input"), "$.packet.task_input")
        if set(task_input) != {"branch_context", "observable_channels", "proposed_observation"}:
            raise ThenAtomizationError("v0.7 task input is not the closed contract")
        candidate = _mapping(task_input.get("proposed_observation"), "$.proposed_observation")
        if candidate.get("candidate_id") != packet.get("candidate_id"):
            raise ThenAtomizationError("v0.7 candidate identity mismatch")
        identities.append((packet.get("branch_id"), packet.get("candidate_id")))
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("v0.7 packet identities must be unique")
    if result.get("summary", {}).get("expected_model_calls") != len(packets):
        raise ThenAtomizationError("v0.7 call count mismatch")
    result["packet_set_fingerprint"] = fingerprint
    return result


def render_oracle_requirement_prompt(packet: Mapping[str, Any], template: str) -> str:
    placeholder = "{oracle_requirement_input}"
    if packet.get("task_name") != TASK_NAME or placeholder not in template:
        raise ThenAtomizationError("v0.7 prompt template/task mismatch")
    task_input = _mapping(packet.get("task_input"), "$.packet.task_input")
    context = _mapping(task_input.get("branch_context"), "$.branch_context")
    candidate = _mapping(task_input.get("proposed_observation"), "$.proposed_observation")
    contract = deepcopy(dict(_mapping(candidate.get("observation_contract"), "$.observation_contract")))
    # Endpoint schemas and provenance are compiler evidence.  They are retained
    # in the packet but intentionally hidden from the relevance judge so that
    # labels such as "spec_binding" cannot bias a yes/no answer.
    contract.pop("endpoint", None)
    model_input = {
        "branch": {
            "branch_id": context.get("branch_id"),
            "given": (context.get("gwt") or {}).get("given"),
            "when": (context.get("gwt") or {}).get("when"),
            "then": (context.get("gwt") or {}).get("then"),
        },
        "proposed_observation": {
            "candidate_id": candidate.get("candidate_id"),
            "observation": candidate.get("requirement_text"),
            "observable_as": contract,
        },
    }
    return template.replace(placeholder, json.dumps(model_input, ensure_ascii=False, indent=2))


def validate_oracle_requirement_response(packet: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(response, "$.oracle_requirement_response")))
    if set(result) != {"branch_id", "candidate_id", "decision", "reason"}:
        raise ThenAtomizationError("v0.7 response fields are invalid")
    if result.get("branch_id") != packet.get("branch_id") or result.get("candidate_id") != packet.get("candidate_id"):
        raise ThenAtomizationError("v0.7 response identity mismatch")
    if result.get("decision") not in DECISIONS:
        raise ThenAtomizationError("v0.7 decision is invalid")
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        raise ThenAtomizationError("v0.7 reason must be non-empty")
    return result


def materialize_oracle_requirements(
    packet_set: Mapping[str, Any], response_set: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply one complete, closed response batch to the mechanical candidates.

    A ``yes`` response means only that the observation belongs in the oracle.
    Polarity, temporal ordering, channel binding for deferred semantic units,
    and evaluator code remain explicitly outside this stage.
    """

    packets = validate_oracle_requirement_packet_set(packet_set)
    responses = _mapping(response_set, "$.oracle_requirement_responses")
    if responses.get("schema_version") != RESPONSE_SET_VERSION:
        raise ThenAtomizationError("v0.7 response-set schema mismatch")
    raw_responses = responses.get("responses")
    if not isinstance(raw_responses, list):
        raise ThenAtomizationError("v0.7 responses must be an array")

    packet_index = {
        (packet["branch_id"], packet["candidate_id"]): packet
        for packet in packets["packets"]
    }
    response_index: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in raw_responses:
        value = _mapping(raw, "$.responses[]")
        identity = (value.get("branch_id"), value.get("candidate_id"))
        packet = packet_index.get(identity)
        if packet is None:
            raise ThenAtomizationError(f"v0.7 response has unknown identity: {identity}")
        if identity in response_index:
            raise ThenAtomizationError(f"v0.7 duplicate response identity: {identity}")
        response_index[identity] = validate_oracle_requirement_response(packet, value)
    missing = [identity for identity in packet_index if identity not in response_index]
    if missing:
        raise ThenAtomizationError(f"v0.7 response set is incomplete: {missing}")

    branch_results = []
    decision_counts = {decision: 0 for decision in sorted(DECISIONS)}
    for summary in packets["branch_summaries"]:
        branch_id = summary["branch_id"]
        branch_packets = [
            packet for packet in packets["packets"] if packet["branch_id"] == branch_id
        ]
        selected = []
        ambiguous = []
        rejected = []
        for packet in branch_packets:
            response = response_index[(branch_id, packet["candidate_id"])]
            decision_counts[response["decision"]] += 1
            record = {
                "candidate": deepcopy(packet["task_input"]["proposed_observation"]),
                "selection_reason": response["reason"],
            }
            if response["decision"] == "yes":
                selected.append(record)
            elif response["decision"] == "ambiguous":
                ambiguous.append(record)
            else:
                rejected.append(record)
        branch_results.append(
            {
                "branch_id": branch_id,
                "branch_context": deepcopy(branch_packets[0]["task_input"]["branch_context"]),
                "selected_requirements": selected,
                "ambiguous_candidates": ambiguous,
                "rejected_candidates": rejected,
                "binding_status": (
                    "selection_ambiguous"
                    if ambiguous
                    else "ready_for_observation_binding"
                ),
            }
        )

    result = {
        "schema_version": REQUIREMENT_SET_VERSION,
        "source_packet_set_fingerprint": packets["packet_set_fingerprint"],
        "branches": branch_results,
        "summary": {
            "branch_count": len(branch_results),
            "candidate_count": len(packet_index),
            "selected_count": decision_counts["yes"],
            "rejected_count": decision_counts["no"],
            "ambiguous_count": decision_counts["ambiguous"],
            "ready_branch_count": sum(
                item["binding_status"] == "ready_for_observation_binding"
                for item in branch_results
            ),
        },
        "deferred_to_next_stage": [
            "expected polarity and allowed behavior",
            "temporal relation assembly",
            "runtime channel binding for deferred semantic requirements",
            "evaluator implementation",
        ],
    }
    result["requirement_set_fingerprint"] = content_sha256(result)
    return result


def build_oracle_requirement_packets_file(
    *,
    specs_path: str | Path,
    assessments_path: str | Path,
    semantic_units_path: str | Path,
    tool_catalog_path: str | Path,
    sut_profile_path: str | Path,
    output_path: str | Path,
    selected_branch_ids: Sequence[str],
) -> dict[str, Any]:
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    result = build_oracle_requirement_packets(
        load(specs_path),
        load(assessments_path),
        load(semantic_units_path),
        load(tool_catalog_path),
        load(sut_profile_path),
        selected_branch_ids=selected_branch_ids,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
