"""Extends the frozen runtime_observation_binding_v1 (releases/agentspectesting-method-v1.1.0)
with additional "obtain/ask for X before Y" temporal_relation left-event grammars that
the frozen _compile_semantic_left_event leaves "semantic_event_deferred". Never edits
the frozen file -- calls its (importable, underscore-prefixed-but-not-private-in-any
enforced sense) functions to get a baseline, then only patches the specific bindings
the frozen compiler could not already resolve.

See docs/v5_step4_generality_audit_v0_1.md §8 for why this exists as a separate
module rather than an in-place edit: the frozen file's bytes are pinned by
scripts/prepare_v5_step4_v0_1.py's legacy-compiler-drift check.

Three grammars, in order:
1. Named-fact disclosure -- a fact the rule names (e.g. "the user id") that IS a real
   argument of the resolved target tool itself (e.g. book_reservation.user_id). Reuses
   the frozen assistant_argument_disclosure binding_kind, just with a fact-derived
   subset of required_argument_paths instead of the full required_parameters set.
2. Prerequisite tool call -- a fact the rule names that is NOT an argument of the
   target tool (e.g. cancel_reservation never receives user_id) but IS returned by a
   real, separate lookup tool (get_user_details, get_reservation_details). A rule
   naming multiple such facts compiles left_event as a *list*: all of them must have
   an independent witness before the target event for a match (conjunction). A single
   dict remains the original, frozen shape/semantics -- extract_bound_observations_extended
   only special-cases the list shape and delegates everything else to the frozen
   extract_bound_observations unchanged.
3. Dialogue act -- an action the rule names that the agent itself performs by SAYING
   something (asking a question, confirming a fact), not by calling a tool or
   disclosing a value the target call also carries. There is no structured
   dialogue-act tag on the assistant side to match against (normalize_tau_messages's
   driver_action_kind tagging exists only for user_message events), so this grammar
   matches free-text assistant message content directly against a small,
   hand-curated keyword set per semantic unit. Also compiles left_event as a
   *list* (of one), for the same reason as grammar 2: the frozen
   extract_bound_observations's witness matching only inspects a matcher's
   `kind` for "contains_right_call_argument_values" and otherwise accepts ANY
   preceding event matching event_filter unconditionally, which would make a
   keyword check silently vacuous if left in the single-dict (frozen-extractor)
   shape -- see docs/oracle_requirement_pipeline_v0_7.md section 24.

Separately (not one of the three left_event grammars above, since it replaces
BOTH sides of a temporal_relation binding, not just left_event): a
message-literal right_event override for airline_077_order#b0, whose Then is
"first call transfer_to_human_agents, THEN send this exact message" -- the
right_event is a specific MESSAGE, not "the requested operation" the frozen
compiler's temporal_relation branch unconditionally assumes (it hardcodes
right_event as a tool-operation-family lookup with no path for a message at
all). Gated directly on the accepted requirement's own observation_contract
(a right_event sentinel value of "assistant_message_literal", set by
oracle_requirement_pipeline_v7.py's _TEMPORAL_MESSAGE_LITERAL_FALLBACK_TARGETS)
rather than on the frozen baseline's binding_status, since that baseline's
partial computation for this shape is meaningless noise to be replaced
wholesale, not built upon. left_event reuses a new, genuinely general fourth
grammar ("named tool call": the left semantic unit's action text literally
names a tool call, e.g. "make a tool call to X") -- general because it isn't
specific to 077's tool, just to the "X must itself be a tool call" shape.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .then_atomization import ThenAtomizationError
from .runtime_observation_binding_v1 import (
    BINDING_STATUSES,
    BINDING_VERSION,
    _argument_disclosure_match,
    _event_matches_filter,
    _mapping,
    _path,
    _scalar_leaves,
    _scope_matches,
    _semantic_unit_index,
    _surface,
    _tool_catalog_index,
    compile_runtime_observation_bindings,
    extract_bound_observations,
)


EXTENDED_SET_VERSION = "agentspectesting.runtime-observation-binding-set-extended/v0.1"
PROGRAM_SOURCE = "extension_v1"

# Checked bag allowance table, transcribed from the real tau2 airline policy
# (<TAU2_BENCH_DIR>/data/tau2/domains/airline/policy.md,
# "Checked bag allowance:" section, lines 82-94) -- a fixed, reviewable 3x3 constant
# table, not something that needs a runtime lookup. See
# docs/v5_step4_then_structural_classifier_v0_1.md §8 for why this is a *compile-time*
# constant here: the branches this applies to (airline_040-046_arg) each have a Given
# database condition that fixes membership tier, not a live get_user_details read.
_FREE_BAGGAGE_ALLOWANCE = {
    ("regular", "basic_economy"): 0,
    ("regular", "economy"): 1,
    ("regular", "business"): 2,
    ("silver", "basic_economy"): 1,
    ("silver", "economy"): 2,
    ("silver", "business"): 3,
    ("gold", "basic_economy"): 2,
    ("gold", "economy"): 3,
    ("gold", "business"): 4,
}
# Real bug found and fixed in this round (see
# docs/oracle_requirement_pipeline_v0_7.md section 24): this used to be gated
# on branch_context["target_action"].startswith("free_baggage_"), but
# v5_oracle_input_adapter_v1.py intentionally stopped populating target_action
# at all (it is always JSON null in every real branch, confirmed against the
# full 155-branch corpus, not just airline_040-049) -- so
# `.get("target_action", "")` returned None (the key IS present, just with a
# null value, so the dict.get default never applies), and
# None.startswith(...) crashed compile_runtime_observation_bindings_extended
# outright the moment it ran over real data containing ANY semantic_deferred
# record in ANY branch (not just a baggage one) -- confirmed by reproducing
# it against the real, current outputs/oracle_acceptance_v0_1_no_legacy_hint/
# accepted_requirements.json. This was never caught before because no round
# of this project had ever actually run this function over the real,
# post-§15 full corpus -- the only persisted bindings.json artifact
# (outputs/runtime_observation_bindings_v0_1/bindings.json) predates §15 and
# only covers a 20-branch calibration set, and every unit test for this
# function used a hand-built accepted_requirement_set that (accidentally)
# always supplied a real target_action string.
#
# Fixed by keying off the record's own accepted requirement_text instead --
# the twelve real baggage-allowance-formula requirements (airline_040-046,
# confirmed via grep of the real corpus) all share the exact same generated
# phrasing ("<membership tier> passengers get N free bag(s)"), which nothing
# else in the corpus matches (checked: no other semantic_deferred requirement
# anywhere in the corpus contains this phrase).
_FREE_BAGGAGE_REQUIREMENT_TEXT = re.compile(r"passengers get \d+ free bags?", re.IGNORECASE)

# docs/agentcoveragetesting_reuse_log.md section 93: _compile_semantic_left_event
# (this module's frozen caller, above) mechanically requires an "action
# details" disclosure message to state EVERY one of the target tool's real
# schema-required arguments, including a payment argument (payment_id/
# payment_method_id) even when the real, bound transaction results in a
# genuine $0 charge/refund -- a real, policy-compliant agent has no real
# reason to state which payment method it's using when nothing is being
# charged, and doing so or not is pure LLM wording variance, not a real
# behavioral difference. Real-verified per branch, not a text-pattern guess:
# each entry here was confirmed by replicating the real tau2 tool's own real
# charge/refund formula against this branch's own real bound argument
# values (see the reuse log section for each real computation). Maps
# requirement_id -> the exact "arguments.X" path to drop from that binding's
# required_argument_paths.
_ZERO_CHARGE_PAYMENT_ARGUMENT_OVERRIDE: dict[str, str] = {
    "airline_074_order#b2::OR01": "arguments.payment_id",
    # retail_055_order_cancel_pending_order#b2/retail_055_order_return_
    # delivered_order_items#b0 are both really bound to return_delivered_
    # order_items (verified against the real bound plan's exact_tool_names --
    # the first branch_id's own name is misleading), whose real tool body
    # (tau2-bench src/tau2/domains/retail/tools.py return_delivered_order_
    # items) never computes a price or appends an OrderPayment at all -- the
    # real refund is an out-of-band, later process per the tool's own real
    # docstring ("The user will receive follow-up email..."), so
    # payment_method_id never has monetary consequence within the bound
    # tool call, for these or any other real invocation of this tool.
    "retail_055_order_cancel_pending_order#b2::OR01": "arguments.payment_method_id",
    "retail_055_order_return_delivered_order_items#b0::OR01": "arguments.payment_method_id",
}

# Section 134 bug 2 (telecom_080_order#b0::OR01, docs/agentcoveragetesting_
# reuse_log.md section 131.2/131.6/134.2): the frozen _binding_for_requirement
# resolves a bare "tool_call" requirement_type to a SINGLE literal tool_name
# taken from the requirement's own observation_contract (here
# "get_customer_by_phone", picked upstream because the rule text says "the
# line associated with the phone number the user provided") -- but real tau2
# telecom tools.py confirms get_customer_by_phone (line ~49) returns a
# Customer record (name/dob/email/address/phone_number) and never includes
# roaming_enabled at all, while get_details_by_id (line ~234, id starting
# with "L" -> _get_line_by_id) is the real tool that returns the LINE record
# including roaming_enabled -- the ONLY tool that actually evidences "checked
# that the line ... is roaming enabled". A real, confirmed-compliant online
# transcript (telecom_080_order#b0, batch_03/results) called
# get_customer_by_id -> get_details_by_id(id="L1001") (observed
# roaming_enabled: false) -> enable_roaming, never calling
# get_customer_by_phone at all, and was wrongly scored fail purely because
# the bound evidence tool_name doesn't match -- the same false-negative
# shape Family P (docs/agentcoveragetesting_reuse_log.md section 79,
# _extract_tool_call_any_of_observations / retail_053's real
# find_user_id_by_email-OR-find_user_id_by_name_zip precedent) already has a
# real evaluation-time mechanism for, just never wired to a real compile-time
# producer for this requirement. Maps requirement_id -> the real, verified
# set of tool_names that each independently evidence the SAME fact this
# requirement's Then is actually about (unlike Family P's retail_053, where
# the rule text itself names 2 alternative methods, here both the originally
# bound tool_name and this table's replacement must independently be able to
# satisfy the fact -- get_customer_by_phone is kept as an alternative, not
# dropped, since some real compliant traces do surface roaming status via a
# get_customer_by_phone-anchored path in the wider conversation and this
# table only ever WIDENS accepted evidence, never narrows it below what
# already shipped).
_TOOL_CALL_ANY_OF_OVERRIDE: dict[str, list[str]] = {
    "telecom_080_order#b0::OR01": ["get_customer_by_phone", "get_details_by_id"],
    # section 194 (task_fd3f3d3b): retail_053's two siblings share the rule
    # "authenticate ... via email, or via name and zip code" -- sections
    # 79/133.2 hand-spliced exactly this tool_call_any_of shape into the
    # persisted bindings; recorded here so a from-scratch compile reproduces it.
    "retail_053_order_find_user_id_by_email#b0::OR01": ["find_user_id_by_email", "find_user_id_by_name_zip"],
    "retail_053_order_find_user_id_by_name_zip#b0::OR01": ["find_user_id_by_email", "find_user_id_by_name_zip"],
}

# docs/agentcoveragetesting_reuse_log.md section 100: the frozen
# _scope_constraints (runtime_observation_binding_v1.py) unconditionally
# treats EVERY parameter named in observation_contract.scope_parameters as
# an identity-scoping value that must EXACTLY equal a fixed reference
# (driver_bindings.<parameter>) -- correct for a parameter like
# reservation_id/order_id, wrong for transfer_to_human_agents's own
# `summary` argument: confirmed by reading the real transfer_to_human_
# agents implementation in every real tau2 domain's tools.py (airline/
# retail/telecom/mock/banking_knowledge) -- every one just returns
# "Transfer successful" and never reads or branches on `summary` at all;
# it exists purely so a human agent can later read a free-text description
# of the issue, and the tool's own docstring says exactly that ("a summary
# of the user's issue"). transfer_to_human_agents has no other real
# argument, so whenever it's the ONLY member of scope_parameters, this
# check degenerates into "the agent's own natural-language description of
# the issue must literally equal a fixed placeholder string" -- discovered
# for real when telecom_037/retail_058/retail_059 (docs/agentcoveragetesting
# _reuse_log.md section 100) were exercised online for the first time
# (retail_059_order#b0's real transcript: agent correctly explained the
# limitation, correctly called transfer_to_human_agents with its own
# accurate summary, correctly sent the exact required literal message --
# and still failed OR01/OR02 purely because its real summary text differed
# from the fixed placeholder). Confirmed corpus-wide via a real grep of
# accepted_requirements.json across all 3 domains: this shape (transfer_to_
# human_agents + scope_parameters == ["summary"]) is exactly 7 real
# requirements (telecom_038/039, retail_059_order#b0/#b1v0, airline_076_
# norm#b0/#b1, airline_077_order#b0) -- fixed once, generally, for all of
# them, not a per-branch override table, since the underlying fact (summary
# is always free text for this one specific tool) is a real, universal
# property of the tool itself, not a per-branch judgment call.
_UNCONSTRAINED_FREE_TEXT_SCOPE_PARAMETERS = frozenset({
    ("transfer_to_human_agents", "summary"),
})

# Only usable when the mapped argument name is confirmed (via tool_catalog) to
# actually be a real argument of the resolved target tool -- see module docstring.
_NAMED_FACT_VOCABULARY = {
    "user id": "user_id",
    "trip type": "flight_type",
    "origin": "origin",
    "destination": "destination",
}

# Only usable when the mapped lookup tool is confirmed to actually exist in the
# tool catalog -- see module docstring.
#
# "locate" (airline_085_order#b0, CA0041::A01: "help locate it", referring to
# an unknown reservation id -- see the surrounding policy_statement) maps to
# get_user_details for the same real reason "reservation id" does: that tool's
# real return payload is "the details of a user, including their
# reservations", i.e. it IS the mechanism this branch's own already-accepted
# sibling requirement (airline_085_order#b0::OR01, a bound, executable
# get_user_details tool_call) already independently identified as covering
# this exact action -- confirmed by reading that sibling's
# observation_contract before adding this entry, not guessed. See
# docs/oracle_requirement_pipeline_v0_7.md section 32.
_PREREQUISITE_LOOKUP_VOCABULARY = {
    "user id": "get_user_details",
    "reservation id": "get_reservation_details",
    "locate": "get_user_details",
}


def _named_facts_before_operation(
    normalized_action_text: str,
    selected_tool: str,
    tool_catalog: Mapping[str, Mapping[str, Any]],
    named_fact_vocabulary: Mapping[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    tool_parameters = set((tool_catalog.get(selected_tool) or {}).get("parameter_names") or [])
    matched_paths = []
    matched_phrases = []
    for phrase, parameter in named_fact_vocabulary.items():
        if phrase not in normalized_action_text:
            continue
        if parameter not in tool_parameters:
            continue
        matched_phrases.append(phrase)
        matched_paths.append(f"arguments.{parameter}")
    if not matched_paths:
        return None, {"code": "semantic_action_not_in_extension_grammar"}
    return (
        {
            "binding_kind": "assistant_argument_disclosure",
            "event_filter": {"event_kind": "assistant_message"},
            "matcher": {
                "kind": "contains_right_call_argument_values",
                "required_argument_paths": matched_paths,
                "collection_policy": "all_scalar_leaves",
                "match_policy": "one_message_contains_all_values",
            },
        },
        {"code": "compiled_named_fact_disclosure_grammar", "matched_phrases": matched_phrases},
    )


def _prerequisite_lookups_before_operation(
    normalized_action_text: str,
    tool_catalog: Mapping[str, Mapping[str, Any]],
    prerequisite_lookup_vocabulary: Mapping[str, str | list[str]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    matched_phrases = [
        phrase for phrase in prerequisite_lookup_vocabulary if phrase in normalized_action_text
    ]
    if not matched_phrases:
        return None, {"code": "semantic_action_not_in_extension_grammar"}
    # A matched phrase names a real requirement of the rule -- if its lookup tool(s)
    # aren't in the catalog we cannot verify that part, so the whole thing must stay
    # unbound rather than silently compile a weaker check missing a required fact.
    missing_tools = sorted(
        {
            tool
            for phrase in matched_phrases
            for tool in _as_tool_list(prerequisite_lookup_vocabulary[phrase])
            if tool not in tool_catalog
        }
    )
    if missing_tools:
        return None, {
            "code": "prerequisite_lookup_tool_unavailable",
            "matched_phrases": matched_phrases,
            "missing_tools": missing_tools,
        }
    left_events = []
    for phrase in matched_phrases:
        tools = _as_tool_list(prerequisite_lookup_vocabulary[phrase])
        if len(tools) == 1:
            left_events.append(
                {
                    "binding_kind": "prerequisite_tool_call",
                    "event_filter": {
                        "event_kind": "assistant_tool_call",
                        "field_equals": {"tool_name": tools[0]},
                    },
                }
            )
        else:
            # A single real-world fact (e.g. "identify the customer") may be
            # obtainable via any one of several real, alternative lookup
            # tools (telecom's get_customer_by_id/get_customer_by_phone/
            # get_customer_by_name all independently establish identity) --
            # unlike the conjunctive list _extract_temporal_observations_
            # extended already supports (ALL entries must be witnessed),
            # this single entry is satisfied by ANY ONE of `tools` being
            # called, so it cannot be expressed via the frozen
            # _event_matches_filter's exact field_equals -- see
            # _witness_for_left_event's "prerequisite_tool_call_any_of"
            # handling below, an extension-only matcher kind, not a change
            # to the frozen module's matching semantics.
            left_events.append(
                {
                    "binding_kind": "prerequisite_tool_call_any_of",
                    "event_filter": {"event_kind": "assistant_tool_call"},
                    "matcher": {"kind": "tool_name_in", "tool_names": tools},
                }
            )
    return left_events, {
        "code": "compiled_prerequisite_lookup_grammar",
        "matched_phrases": matched_phrases,
    }


def _as_tool_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


# airline_083_order#b0 (CA0021::A01)/airline_097_order#b0 (CA0048::A01): neither
# _NAMED_FACT_VOCABULARY nor _PREREQUISITE_LOOKUP_VOCABULARY matches "ask the
# user if they want to purchase travel insurance" or "confirm the facts
# (observations of the current case)" -- confirmed via direct substring check.
# Keyed by semantic_unit_id directly (unlike the two vocabularies above, which
# match phrases inside the unit's own direct_action_text) because both keyword
# sets here were hand-picked for these two specific branches, not derived from
# a phrase that would generalize automatically to other units. Deliberately
# loose (ANY one keyword, case-insensitive substring, via the shared _surface
# tokenizer) rather than exact-literal matching, because a real agent
# paraphrases the policy statement instead of repeating it verbatim.
#
# "insurance" alone is a safe anchor for CA0021::A01: it is a single-purpose,
# domain-specific noun with no other meaning anywhere in the airline domain.
# "confirm" alone for CA0048::A01 is a deliberately narrower, imperfect
# approximation -- an agent could say "confirmed" about something unrelated
# (e.g. confirming a booking) and this grammar cannot currently distinguish
# that from confirming the facts of a compensation case. Tightening this
# further (e.g. requiring a second, case-related keyword) was considered and
# rejected for now: there is no real recorded transcript corpus available in
# this session to validate a tighter phrase against, and guessing a more
# specific phrase risks the opposite failure (a real confirmation that never
# says any of the guessed words). Documented here, not silently accepted, per
# docs/oracle_requirement_pipeline_v0_7.md section 24.
# CA0042::A01 (airline_091_order#b0, "obtain the reason for cancellation"):
# neither of the two grammars above applies -- confirmed the real
# cancel_reservation(reservation_id) signature carries no "reason" argument at
# all (tau2-bench tools.py), so this fact is never disclosed as a tool
# argument (grammar 1) and is never returned by a lookup tool either (grammar
# 2); it can only ever surface as something the agent asks the user for, in a
# message. "reason" alone shares the same documented imprecision as
# "confirm" above (an agent could mention "reason" in an unrelated sense) --
# same rationale for leaving it a single loose keyword: no real transcript
# corpus available in this session to calibrate a tighter phrase against, and
# guessing a more specific one risks the opposite failure. Documented here,
# not silently accepted, per docs/oracle_requirement_pipeline_v0_7.md section
# 32.
#
# docs/agentcoveragetesting_reuse_log.md section 173 (task_eba70b14): the
# "CA0048::A01": ("confirm",) entry below is kept only as the historical
# record (and because the JSON config mirrors this dict byte-for-byte). It is
# no longer what a CA0048::A01 left_event compiles to:
# _FACT_VERIFICATION_PREREQUISITE_VOCABULARY (below) is consulted FIRST by both
# dispatch chains, so this keyword is now unreachable for that unit. A replay
# of every recorded live run of the 3 CA0048 branches showed the literal-
# "confirm" proxy was wrong in BOTH directions -- see that vocabulary's own
# comment.
_DIALOGUE_ACT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "CA0021::A01": ("insurance",),
    "CA0048::A01": ("confirm",),
    "CA0042::A01": ("reason",),
}

# docs/agentcoveragetesting_reuse_log.md section 173 (task_eba70b14, open since
# section 132.2). CA0048::A01's real policy text is "Always confirms the facts
# before offering compensation" / "offer a certificate as a gesture after
# confirming the facts" (tau2 airline policy.md lines 159/163/165), unit
# direct_action_text "confirm the facts (observations of the current case)":
# CONFIRMING THE FACTS means VERIFYING the case facts, not asking the user a
# yes/no question and not saying the word "confirm". The dialogue-act grammar
# compiled it to "some earlier assistant message contains the substring
# 'confirm' (or a '(yes/no)' marker, section 147's fallback)", which a real
# replay of every live airline_037_arg#b0/airline_038_arg#b0/airline_097_order
# #b0 transcript showed is wrong both ways:
#   - false FAIL: pass6 037/097 -- the agent called get_reservation_details +
#     get_user_details + get_flight_status on every segment, listed the
#     verified facts ("I've verified the details: ... HAT058 was cancelled"),
#     then sent the certificate; failed only because it wrote "verified"
#     instead of "confirmed" (037 even asked "Would you like me to send this
#     certificate to your account?", which section 147's "(yes/no)" fallback
#     does not cover either).
#   - false PASS: pass5 037 / pass3 097 -- the agent never checked any flight
#     status, took the user's "the airline cancelled it" at face value, and
#     passed only because its message said "Just to confirm the facts ... is
#     it correct that the airline cancelled the flights?" (asking the USER to
#     vouch for the claim is not confirming the fact).
# The two lookups below are the real tau2 airline tools that return the facts
# every compensation rule in policy.md lines 155-167 is conditioned on:
# get_reservation_details (cabin, insurance, passenger count -- the $100/$50
# "times the number of passengers" multiplier) and get_flight_status (whether
# the flight was actually cancelled/delayed -- nothing else in the tool set
# returns flight status). Conjunctive (both must precede the target call), a
# NECESSARY-condition proxy exactly like section 39's grammar 2; it does not
# claim the lookups were for the right reservation/flight.
_FACT_VERIFICATION_PREREQUISITE_VOCABULARY: dict[str, tuple[str, ...]] = {
    "CA0048::A01": ("get_reservation_details", "get_flight_status"),
}

# The three vocabulary dicts above (_NAMED_FACT_VOCABULARY, _PREREQUISITE_
# LOOKUP_VOCABULARY, _DIALOGUE_ACT_KEYWORDS) are kept in place, with their
# original comments, as the historical/readable source of airline's content
# -- same treatment as oracle_evaluator_contract_extension_v1.py's orphaned
# family dicts (docs/agentcoveragetesting_reuse_log.md section 19). They are
# no longer read directly by the dispatch functions below; the live default
# is _AIRLINE_BINDING_VOCABULARY_CONFIG, loaded from external JSON so a new
# domain (telecom, retail) can supply its own real vocabulary without editing
# this module's code (see docs/agentcoveragetesting_reuse_log.md section 37
# for the real-data validation this generalization is based on).
_DEFAULT_BINDING_VOCABULARY_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs/runtime_observation_bindings/airline_v0_1.json"
)


def load_runtime_observation_binding_vocabulary_config(path: str | Path) -> dict[str, Any]:
    """Load one domain's Family I vocabulary (named-fact/prerequisite-lookup/
    dialogue-act) from external JSON. Pure content move, not a new dispatch
    mechanism -- see module docstring and section 37 of the reuse log.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "agentspectesting.runtime-observation-binding-vocabulary/v0.1":
        raise ValueError(
            "unsupported runtime observation binding vocabulary config schema: "
            f"{data.get('schema_version')!r}"
        )
    return {
        "named_fact_vocabulary": dict(data.get("named_fact_vocabulary") or {}),
        "prerequisite_lookup_vocabulary": dict(data.get("prerequisite_lookup_vocabulary") or {}),
        "dialogue_act_keywords": {
            key: tuple(value) for key, value in (data.get("dialogue_act_keywords") or {}).items()
        },
        # Added later (docs/agentcoveragetesting_reuse_log.md section 44):
        # requirement_id-keyed, not part of compile_runtime_observation_
        # bindings_extended's own per-record patch chain -- consumed by the
        # separate apply_semantic_event_alias_patches post-processing step.
        "semantic_event_alias_vocabulary": dict(data.get("semantic_event_alias_vocabulary") or {}),
        # Added later (docs/agentcoveragetesting_reuse_log.md section 50):
        # semantic_unit_id-keyed, like dialogue_act_keywords -- consumed by
        # _effort_prerequisite_before_operation via the separate
        # resolve_ambiguous_temporal_right_events post-processing step.
        "effort_prerequisite_vocabulary": {
            key: tuple(value) for key, value in (data.get("effort_prerequisite_vocabulary") or {}).items()
        },
        # Added later (docs/agentcoveragetesting_reuse_log.md section 106):
        # requirement_id-keyed (a one-off shape tied to one real requirement,
        # not a reusable semantic category) -- consumed by
        # _resolve_ambiguous_right_event's case C via resolve_ambiguous_
        # temporal_right_events.
        "generic_requested_operation_vocabulary": {
            key: tuple(value) for key, value in (data.get("generic_requested_operation_vocabulary") or {}).items()
        },
        # Added later (docs/agentcoveragetesting_reuse_log.md section 106):
        # semantic_unit_id-keyed, like dialogue_act_keywords/effort_
        # prerequisite_vocabulary -- consumed by
        # _prerequisite_lookup_by_semantic_unit_before_operation via the
        # same post-processing step.
        "prerequisite_lookup_by_semantic_unit_vocabulary": {
            key: tuple(value)
            for key, value in (data.get("prerequisite_lookup_by_semantic_unit_vocabulary") or {}).items()
        },
        # Added later (docs/agentcoveragetesting_reuse_log.md section 173):
        # semantic_unit_id-keyed; each value is a CONJUNCTIVE tool list (every
        # tool must be called before the target operation). Consulted before
        # dialogue_act_keywords by both dispatch chains -- see
        # _fact_verification_before_operation.
        "fact_verification_prerequisite_vocabulary": {
            key: tuple(value)
            for key, value in (data.get("fact_verification_prerequisite_vocabulary") or {}).items()
        },
    }


_AIRLINE_BINDING_VOCABULARY_CONFIG = load_runtime_observation_binding_vocabulary_config(
    _DEFAULT_BINDING_VOCABULARY_CONFIG_PATH
)


def _resolve_ambiguous_right_event(
    right_event: Mapping[str, Any],
    requirement_id: Any = None,
    generic_requested_operation_vocabulary: Mapping[Any, Sequence[str]] = MappingProxyType({}),
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Three real, safe, generalizable overrides for a right_event the frozen
    branch_when_tool_feature_overlap scorer left "partial" (see
    docs/agentcoveragetesting_reuse_log.md section 49/106) -- never overrides
    the frozen scorer itself (used by all 313 real airline/telecom/retail
    candidates), only reinterprets its own already-computed output for three
    narrow, genuinely unambiguous (or, for C, hand-verified) shapes:

    A) Sole candidate: operation_family_candidates has exactly one real
    entry, verified against exactly one real tool -- there is no real
    disambiguation question at all (a token-overlap score of 0 there, e.g.
    telecom_040_norm#b0's WHEN text sharing no vocabulary with
    transfer_to_human_agents's own description, is a false negative of the
    scorer, not evidence of real ambiguity). Confirmed distinct from a case
    like telecom_055_order#b0, where 10 of 11 declared candidates carry NO
    verified_tool_names at all -- "the only ONE that happens to be verified"
    is not the same real fact as "the only one that exists" (which is why
    telecom_055 needed C below instead, not this case).

    B) Tied top score: multiple real tool candidates share the single
    highest score, strictly above every other candidate (a genuine token-
    overlap signal exists, it just cannot pick ONE winner) -- e.g.
    retail_055_order_cancel_pending_order#b1, where modify_pending_order_
    address/items/payment all score 12 against real cancel/exchange/return
    tools scoring only 7. Resolved as a real disjunction (ANY of the tied
    tools satisfies the requirement), not an arbitrary pick -- the requirement
    text in both real cases this was built against ("list the action
    details and obtain explicit user confirmation before modifying the
    order") is genuinely agnostic to which specific order-modifying action
    is taken.

    C) Hand-verified generic-operation disjunction (section 106): for
    telecom_055_order#b0::OR01, the 11 operation_family_candidates are not
    11 independently-verified real tool candidates -- they are 11 IDs
    mechanically fanned out from ONE real policy statement (CP0021::P01,
    "Under Technical Support, the agent must first identify the customer.",
    confirmed: main_policy.md's own "## Technical Support" section is
    literally that one sentence, no tool enumeration), and the pipeline's
    own evidence notes explicitly flag only 1 of 11 (disable_roaming) as
    carrying ANY real tool evidence, the other 10 explicitly "left empty
    rather than guessing". Naively wrapping just that 1 verified tool in
    the disjunctive machinery would be cosmetic (mechanically identical to
    case A, the exact thing this function's docstring already declines to
    do for telecom_055). Instead, real verification against tech_support_
    workflow.md (tau2-bench's own real technical-support troubleshooting
    doc) identifies which further real telecom tools genuinely appear as
    a troubleshooting action in that document: enable_roaming (Step 2.1.2,
    enabling a line's roaming to fix data unavailability), get_data_usage
    (Step 2.1.4, checking if usage exceeded the plan limit), refuel_data
    (Step 2.1.4, the real remediation for exceeded usage), resume_line
    (Step 1.4, lifting a suspension blocking service) -- confirmed real,
    single-tool matches to named workflow steps, unlike get_details_by_id
    (too generic, no 1:1 step correspondence) or suspend_line (never
    appears in any troubleshooting path -- only resume_line, lifting a
    suspension, does) or "change plan" (Step 2.1.4 also allows this, but
    main_policy.md's own "Change Plan" section names no single real tool
    to call, only a multi-step manual process -- not included to avoid
    fabricating a tool name). requirement_id-keyed (not semantic_unit_id
    or requirement_text -- this fan-out is a one-off shape tied to this one
    real requirement, not a reusable semantic category).

    Returns (new_right_event, diagnostic) on a resolved case, (None, None)
    when no shape applies (the real, structural ambiguity is left alone).
    """
    candidates = right_event.get("operation_family_candidates") or []
    if len(candidates) == 1 and len(candidates[0].get("verified_tool_names") or []) == 1:
        tool_name = candidates[0]["verified_tool_names"][0]
        new_right = {
            **deepcopy(dict(right_event)),
            "binding_status": "bound",
            "exact_tool_names": [tool_name],
            "event_filter": {"event_kind": "assistant_tool_call", "field_equals": {"tool_name": tool_name}},
            "scope_constraints": [],
        }
        return new_right, {
            "code": "compiled_sole_operation_family_candidate_override",
            "tool_name": tool_name,
        }
    scores = (right_event.get("operation_resolution") or {}).get("candidate_scores") or []
    if scores:
        top_score = max(item.get("score", 0) for item in scores)
        top_tools = sorted({item["tool_name"] for item in scores if item.get("score") == top_score})
        if top_score > 0 and len(top_tools) >= 2:
            new_right = {
                **deepcopy(dict(right_event)),
                "binding_status": "bound",
                "exact_tool_names": top_tools,
                "event_filter": {"event_kind": "assistant_tool_call"},
                "matcher": {"kind": "tool_name_in", "tool_names": top_tools},
                "scope_constraints": [],
            }
            return new_right, {
                "code": "compiled_tied_top_score_disjunctive_override",
                "tool_names": top_tools,
                "top_score": top_score,
            }
    generic_tools = generic_requested_operation_vocabulary.get(requirement_id)
    if generic_tools:
        tool_names = sorted(set(generic_tools))
        new_right = {
            **deepcopy(dict(right_event)),
            "binding_status": "bound",
            "exact_tool_names": tool_names,
            "event_filter": {"event_kind": "assistant_tool_call"},
            "matcher": {"kind": "tool_name_in", "tool_names": tool_names},
            "scope_constraints": [],
        }
        return new_right, {
            "code": "compiled_generic_requested_operation_override",
            "tool_names": tool_names,
        }
    return None, None


def _dialogue_act_before_operation(
    semantic_unit_id: Any,
    dialogue_act_keywords: Mapping[Any, tuple[str, ...]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    keywords = dialogue_act_keywords.get(semantic_unit_id)
    if not keywords:
        return None, {"code": "semantic_action_not_in_extension_grammar"}
    return (
        [
            {
                "binding_kind": "assistant_dialogue_act",
                "event_filter": {"event_kind": "assistant_message"},
                "matcher": {"kind": "contains_any_phrase", "phrases": list(keywords)},
            }
        ],
        {
            "code": "compiled_dialogue_act_grammar",
            "matched_semantic_unit_id": semantic_unit_id,
        },
    )


def _fact_verification_before_operation(
    semantic_unit_id: Any,
    fact_verification_prerequisite_vocabulary: Mapping[Any, Sequence[str]],
    tool_catalog: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """"Confirm/verify the facts before X" (section 173, task_eba70b14): a
    conjunctive list of real fact-returning lookups that must each precede the
    target operation -- the same list-valued "prerequisite_tool_call" shape
    section 39's grammar 2 already produces (so _extract_temporal_observations_
    extended evaluates it with no new matcher kind). See
    _FACT_VERIFICATION_PREREQUISITE_VOCABULARY for the real-transcript evidence.

    Returns (None, None) when this unit has no entry (callers then try the next
    grammar). When the unit HAS an entry but a tool_catalog is supplied and
    lacks one of the lookup tools, returns (None, diagnostic) -- callers must
    NOT fall through to the dialogue-act grammar in that case, since that would
    silently reinstate the literal-keyword proxy this grammar replaces."""
    tool_names = fact_verification_prerequisite_vocabulary.get(semantic_unit_id)
    if not tool_names:
        return None, None
    if tool_catalog is not None:
        missing = sorted(tool for tool in tool_names if tool not in tool_catalog)
        if missing:
            return None, {
                "code": "fact_verification_tool_unavailable",
                "matched_semantic_unit_id": semantic_unit_id,
                "missing_tools": missing,
            }
    return (
        [
            {
                "binding_kind": "prerequisite_tool_call",
                "event_filter": {
                    "event_kind": "assistant_tool_call",
                    "field_equals": {"tool_name": tool_name},
                },
            }
            for tool_name in tool_names
        ],
        {
            "code": "compiled_fact_verification_prerequisite_grammar",
            "matched_semantic_unit_id": semantic_unit_id,
            "tool_names": list(tool_names),
            "necessary_condition_only": True,
        },
    )


def _effort_prerequisite_before_operation(
    semantic_unit_id: Any,
    effort_prerequisite_vocabulary: Mapping[Any, Sequence[str]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Mechanical, necessary-condition proxy for a subjective "tried their
    best" / "tried all possible ways before escalating" claim (see
    docs/agentcoveragetesting_reuse_log.md section 50): real telecom_040/056
    candidates whose left_event is a genuine EFFORT claim, not a spoken
    dialogue act -- neither _dialogue_act_before_operation (scans assistant
    MESSAGE content) nor _prerequisite_lookups_before_operation/
    _named_facts_before_operation (both need a real semantic_unit_model's
    direct_action_text, which telecom/retail do not have) apply.

    Deliberately narrow: reuses the exact same "prerequisite_tool_call_any_of"
    matcher section 39 built for "identify the customer" (ANY ONE of a real
    tool set counts as a witness) -- here the set is "every real agent tool
    other than the target operation itself". This checks only that the agent
    attempted AT LEAST ONE other real action before the target operation --
    a real, honest NECESSARY condition (an agent that transfers with zero
    prior attempts definitely violates "tried their best"), not a full
    realization of the claim (whether the attempts made were adequate or
    exhaustive is a real semantic judgment this mechanical check cannot
    make and does not claim to). Config-keyed by semantic_unit_id, exactly
    like dialogue_act_keywords -- needs no semantic_unit_model lookup.
    """
    tool_names = effort_prerequisite_vocabulary.get(semantic_unit_id)
    if not tool_names:
        return None, {"code": "semantic_action_not_in_extension_grammar"}
    return (
        [
            {
                "binding_kind": "prerequisite_tool_call_any_of",
                "event_filter": {"event_kind": "assistant_tool_call"},
                "matcher": {"kind": "tool_name_in", "tool_names": list(tool_names)},
            }
        ],
        {
            "code": "compiled_effort_prerequisite_grammar",
            "matched_semantic_unit_id": semantic_unit_id,
            "necessary_condition_only": True,
        },
    )


def _prerequisite_lookup_by_semantic_unit_before_operation(
    semantic_unit_id: Any,
    prerequisite_lookup_by_semantic_unit_vocabulary: Mapping[Any, Sequence[str]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Mechanical proxy for a real "identify the customer"-shaped left_event
    (section 106, telecom_055_order#b0::OR01) -- same real, working
    "prerequisite_tool_call_any_of" shape section 39 built (ANY ONE of a
    real tool set counts as a witness), but reached from the production
    post-processing-patch path used for telecom/retail (unlike section 39's
    own _prerequisite_lookups_before_operation, which needs a real
    tool_catalog/semantic_unit_model lookup neither domain has persisted --
    confirmed: no telecom tool_catalog.json exists anywhere in outputs/, so
    that grammar has never actually run against real telecom data despite
    being wired into the main compile path). Deliberately a separate
    function from _effort_prerequisite_before_operation even though the
    produced shape is mechanically identical -- that function's own real
    semantics is "tried AT LEAST ONE other real action" (necessary-
    condition-only, any tool besides the target counts); this one's real
    semantics is "looked the customer up via ANY ONE of these SPECIFIC
    identity-lookup tools" -- conflating the two would misdescribe what was
    actually verified in this record's own diagnostics. Config-keyed by
    semantic_unit_id, exactly like dialogue_act_keywords/effort_
    prerequisite_vocabulary -- needs no semantic_unit_model lookup."""
    tool_names = prerequisite_lookup_by_semantic_unit_vocabulary.get(semantic_unit_id)
    if not tool_names:
        return None, {"code": "semantic_action_not_in_extension_grammar"}
    return (
        [
            {
                "binding_kind": "prerequisite_tool_call_any_of",
                "event_filter": {"event_kind": "assistant_tool_call"},
                "matcher": {"kind": "tool_name_in", "tool_names": list(tool_names)},
            }
        ],
        {
            "code": "compiled_prerequisite_lookup_by_semantic_unit_grammar",
            "matched_semantic_unit_id": semantic_unit_id,
        },
    )


def _extended_left_event(
    deferred_left_event: Mapping[str, Any],
    selected_tool: str,
    tool_catalog: Mapping[str, Mapping[str, Any]],
    semantic_units: Mapping[str, Mapping[str, Any]],
    binding_vocabulary_config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    semantic_unit_id = deferred_left_event.get("semantic_unit_id")
    unit = semantic_units.get(semantic_unit_id)
    if unit is None:
        return None, {"code": "semantic_unit_not_available"}
    action_text = str((unit.get("semantic_payload") or {}).get("direct_action_text") or "")
    normalized = " ".join(action_text.casefold().split())
    named_fact_event, named_fact_diagnostic = _named_facts_before_operation(
        normalized, selected_tool, tool_catalog,
        binding_vocabulary_config["named_fact_vocabulary"],
    )
    if named_fact_event is not None:
        return named_fact_event, named_fact_diagnostic
    prerequisite_event, prerequisite_diagnostic = _prerequisite_lookups_before_operation(
        normalized, tool_catalog,
        binding_vocabulary_config["prerequisite_lookup_vocabulary"],
    )
    if prerequisite_event is not None:
        return prerequisite_event, prerequisite_diagnostic
    fact_event, fact_diagnostic = _fact_verification_before_operation(
        semantic_unit_id,
        binding_vocabulary_config.get("fact_verification_prerequisite_vocabulary") or {},
        tool_catalog,
    )
    if fact_diagnostic is not None:
        # Either compiled, or the unit is a fact-verification unit whose
        # lookup tools are missing -- never fall through to the literal
        # dialogue-act keyword proxy for such a unit (section 173).
        return fact_event, fact_diagnostic
    return _dialogue_act_before_operation(
        semantic_unit_id, binding_vocabulary_config["dialogue_act_keywords"]
    )


# Fourth left-event grammar, used only by the message-literal override below
# (not wired into _extended_left_event's dispatch chain, since that chain is
# keyed off an already-resolved right-event tool -- this grammar's caller has
# no such tool at all, right_event being a message). Genuinely general: any
# semantic unit whose action text literally names "a tool call to X" resolves
# to "X was called", regardless of which tool X is -- not specific to
# transfer_to_human_agents.
_NAMED_TOOL_CALL_ACTION = re.compile(r"\bmake a tool call to ([a-z][a-z0-9_]*)\b")


def _named_tool_call_left_event(
    normalized_action_text: str,
    tool_catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    match = _NAMED_TOOL_CALL_ACTION.search(normalized_action_text)
    if match is None:
        return None, {"code": "semantic_action_not_in_extension_grammar"}
    tool_name = match.group(1)
    if tool_name not in tool_catalog:
        return None, {"code": "named_tool_call_tool_unavailable", "tool_name": tool_name}
    return (
        [
            {
                "binding_kind": "named_tool_call",
                "event_filter": {
                    "event_kind": "assistant_tool_call",
                    "field_equals": {"tool_name": tool_name},
                },
            }
        ],
        {"code": "compiled_named_tool_call_grammar", "matched_tool_name": tool_name},
    )


def _message_literal_temporal_binding(
    observation_contract: Mapping[str, Any],
    semantic_units: Mapping[str, Mapping[str, Any]],
    tool_catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    literal = observation_contract.get("right_event_literal")
    if not isinstance(literal, str) or not literal:
        return None, {"code": "right_event_literal_missing"}
    unit_id = observation_contract.get("left_event_semantic_unit_id")
    unit = semantic_units.get(unit_id)
    if unit is None:
        return None, {"code": "semantic_unit_not_available"}
    action_text = str((unit.get("semantic_payload") or {}).get("direct_action_text") or "")
    normalized = " ".join(action_text.casefold().split())
    left_event, left_diagnostic = _named_tool_call_left_event(normalized, tool_catalog)
    if left_event is None:
        return None, left_diagnostic
    runtime = {
        "extractor_kind": "temporal_relation",
        "event_stream": "canonical_tau_events",
        "order_field": "event_index",
        "relation": "precedes",
        "left_event": left_event,
        "right_event": {
            "binding_kind": "assistant_message_literal",
            "binding_status": "bound",
            "event_filter": {"event_kind": "assistant_message"},
            "matcher": {"kind": "contains_literal", "literal": literal, "case_sensitive": True},
        },
    }
    return runtime, {"code": "compiled_message_literal_temporal_grammar", **left_diagnostic}


def _given_condition_value(
    given_conditions_document: Mapping[str, Any] | None,
    branch_id: str,
    table: str,
    path: str,
) -> Any:
    conditions = (given_conditions_document or {}).get(branch_id) or []
    for condition in conditions:
        if condition.get("table") == table and condition.get("path") == path and condition.get("op") == "eq":
            return condition.get("value")
    return None


def _baggage_allowance_binding(
    branch_id: str, given_conditions_document: Mapping[str, Any] | None
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    membership = _given_condition_value(given_conditions_document, branch_id, "users", "membership")
    cabin = _given_condition_value(given_conditions_document, branch_id, "reservations", "cabin")
    if membership is None or cabin is None:
        return None, {"code": "given_membership_or_cabin_not_fixed"}
    free_bags = _FREE_BAGGAGE_ALLOWANCE.get((membership, cabin))
    if free_bags is None:
        return None, {"code": "unrecognized_membership_cabin_combination", "membership": membership, "cabin": cabin}
    runtime = {
        "extractor_kind": "event_filter",
        "event_stream": "canonical_tau_events",
        "event_filter": {"event_kind": "assistant_tool_call", "field_equals": {"tool_name": "book_reservation"}},
        "scope_constraints": [],
        "projection": {
            "kind": "field_tuple",
            "paths": ["arguments.total_baggages", "arguments.nonfree_baggages", "arguments.passengers"],
        },
        "baggage_allowance": {"free_bags_per_passenger": free_bags, "membership": membership, "cabin": cabin},
    }
    return runtime, {"code": "compiled_baggage_allowance_grammar", "membership": membership, "cabin": cabin}


def _drop_unconstrained_free_text_scope_constraints(record: Mapping[str, Any]) -> dict[str, Any]:
    """Strip a scope_constraint entry for a (tool_name, parameter) pair this
    module knows is genuinely free text (see _UNCONSTRAINED_FREE_TEXT_SCOPE_
    PARAMETERS's own docstring), regardless of which path (frozen baseline or
    one of the elif branches above) produced this record -- applies
    unconditionally, after every other real classification, since it is not
    itself a classification decision, just removing a real over-constraint
    the frozen compiler's own _scope_constraints has no way to express.
    """
    runtime = record.get("runtime_binding")
    if not isinstance(runtime, Mapping):
        return dict(record)
    event_filter = runtime.get("event_filter")
    tool_name = (
        event_filter.get("field_equals", {}).get("tool_name")
        if isinstance(event_filter, Mapping)
        else None
    )
    scope_constraints = runtime.get("scope_constraints") or []
    kept = [
        c for c in scope_constraints
        if (tool_name, str(c.get("actual_path", "")).removeprefix("arguments."))
        not in _UNCONSTRAINED_FREE_TEXT_SCOPE_PARAMETERS
    ]
    if kept == scope_constraints:
        return dict(record)
    record = deepcopy(dict(record))
    record.pop("binding_fingerprint", None)
    new_runtime = dict(record["runtime_binding"])
    new_runtime["scope_constraints"] = kept
    new_runtime["program_source"] = PROGRAM_SOURCE
    record["runtime_binding"] = new_runtime
    record["diagnostics"] = list(record.get("diagnostics") or []) + [
        {"code": "dropped_unconstrained_free_text_scope_parameter", "tool_name": tool_name},
        {"code": "compiled_by_extension_v1"},
    ]
    record["binding_fingerprint"] = content_sha256(record)
    return record


def compile_runtime_observation_bindings_extended(
    accepted_requirement_set: Mapping[str, Any],
    expectation_set: Mapping[str, Any],
    runtime_observation_profile: Mapping[str, Any],
    execution_matching_contracts: Mapping[str, Any],
    tool_catalog_document: Mapping[str, Any] | None = None,
    semantic_unit_model: Mapping[str, Any] | None = None,
    given_conditions_document: Mapping[str, Any] | None = None,
    *,
    binding_vocabulary_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    vocabulary_config = (
        binding_vocabulary_config
        if binding_vocabulary_config is not None
        else _AIRLINE_BINDING_VOCABULARY_CONFIG
    )
    baseline = compile_runtime_observation_bindings(
        accepted_requirement_set,
        expectation_set,
        runtime_observation_profile,
        execution_matching_contracts,
        tool_catalog_document,
        semantic_unit_model,
    )
    tool_catalog = _tool_catalog_index(tool_catalog_document)
    semantic_units = _semantic_unit_index(semantic_unit_model)
    requirement_types = {
        requirement["requirement_id"]: requirement.get("requirement_type")
        for branch in accepted_requirement_set["branches"]
        for requirement in branch["requirements"]
    }
    requirement_contracts = {
        requirement["requirement_id"]: requirement.get("observation_contract") or {}
        for branch in accepted_requirement_set["branches"]
        for requirement in branch["requirements"]
    }
    requirement_texts = {
        requirement["requirement_id"]: requirement.get("requirement_text") or ""
        for branch in accepted_requirement_set["branches"]
        for requirement in branch["requirements"]
    }

    status_counts: Counter[str] = Counter()
    branches = []
    unresolved_branches = []
    for source_branch in baseline["branches"]:
        records = []
        branch_then = str(
            ((source_branch.get("branch_context") or {}).get("gwt") or {}).get("then") or ""
        )
        for record in source_branch["bindings"]:
            runtime = record["runtime_binding"]
            left_event = runtime.get("left_event") if isinstance(runtime, Mapping) else None
            right_event = runtime.get("right_event") if isinstance(runtime, Mapping) else None
            if (
                requirement_types.get(record["requirement_id"]) == "temporal_relation"
                and requirement_contracts.get(record["requirement_id"], {}).get("right_event")
                == "assistant_message_literal"
            ):
                # airline_077_order#b0: right_event is a specific MESSAGE
                # LITERAL, not an operation -- the frozen _binding_for_requirement
                # has no path for that at all (it unconditionally treats
                # right_event as a tool-operation family lookup), so regardless
                # of whatever binding_status/runtime_binding the frozen
                # baseline computed for this record (it may have spuriously
                # "bound" right_event to some unrelated tool via WHEN-scoring,
                # or left it partial), this fully replaces runtime_binding
                # rather than patching one side of it -- same pattern as the
                # baggage_allowance/turn_shape_constraint overrides below, just
                # gated on the requirement's own observation_contract instead
                # of the frozen baseline's status. See
                # docs/oracle_requirement_pipeline_v0_7.md section 24.
                new_runtime, extra_diagnostic = _message_literal_temporal_binding(
                    requirement_contracts[record["requirement_id"]], semantic_units, tool_catalog
                )
                if new_runtime is not None:
                    new_runtime["program_source"] = PROGRAM_SOURCE
                    record = deepcopy(record)
                    record.pop("binding_fingerprint", None)
                    record["runtime_binding"] = new_runtime
                    record["binding_status"] = "bound"
                    record["diagnostics"] = list(record["diagnostics"]) + [
                        extra_diagnostic,
                        {"code": "compiled_by_extension_v1"},
                    ]
                    record["binding_fingerprint"] = content_sha256(record)
            elif (
                record["binding_status"] == "partial"
                and isinstance(runtime, Mapping)
                and runtime.get("extractor_kind") == "temporal_relation"
                and isinstance(left_event, Mapping)
                and left_event.get("binding_kind") == "semantic_event_deferred"
                and isinstance(right_event, Mapping)
                and right_event.get("binding_status") == "bound"
                and len(right_event.get("exact_tool_names") or []) == 1
            ):
                selected_tool = right_event["exact_tool_names"][0]
                new_left, extra_diagnostic = _extended_left_event(
                    left_event, selected_tool, tool_catalog, semantic_units,
                    vocabulary_config,
                )
                if new_left is not None:
                    record = deepcopy(record)
                    record.pop("binding_fingerprint", None)
                    new_runtime = deepcopy(record["runtime_binding"])
                    new_runtime["left_event"] = new_left
                    new_runtime["program_source"] = PROGRAM_SOURCE
                    record["runtime_binding"] = new_runtime
                    record["binding_status"] = "bound"
                    record["diagnostics"] = list(record["diagnostics"]) + [
                        extra_diagnostic,
                        {"code": "compiled_by_extension_v1"},
                    ]
                    record["binding_fingerprint"] = content_sha256(record)
            elif (
                record["binding_status"] == "semantic_deferred"
                and _FREE_BAGGAGE_REQUIREMENT_TEXT.search(
                    requirement_texts.get(record["requirement_id"], "")
                )
            ):
                new_runtime, extra_diagnostic = _baggage_allowance_binding(
                    source_branch["branch_id"], given_conditions_document
                )
                if new_runtime is not None:
                    new_runtime["program_source"] = PROGRAM_SOURCE
                    record = deepcopy(record)
                    record.pop("binding_fingerprint", None)
                    record["runtime_binding"] = new_runtime
                    record["binding_status"] = "bound"
                    record["diagnostics"] = list(record["diagnostics"]) + [
                        extra_diagnostic,
                        {"code": "compiled_by_extension_v1"},
                    ]
                    record["binding_fingerprint"] = content_sha256(record)
            elif (
                record["binding_status"] == "needs_adjudication"
                and requirement_types.get(record["requirement_id"])
                == "turn_shape_constraint"
            ):
                # The frozen _binding_for_requirement doesn't recognize this
                # candidate_kind at all and falls to its own catch-all
                # ("needs_adjudication", {"extractor_kind": "unbound"}), even
                # though _classify already accepted it as "bound" -- gating on
                # both the status AND the requirement_type keeps this from
                # ever matching an unrelated needs_adjudication record.
                new_runtime = {
                    "extractor_kind": "turn_shape_constraint",
                    "event_stream": "canonical_tau_events",
                    "program_source": PROGRAM_SOURCE,
                }
                record = deepcopy(record)
                record.pop("binding_fingerprint", None)
                record["runtime_binding"] = new_runtime
                record["binding_status"] = "bound"
                record["diagnostics"] = list(record["diagnostics"]) + [
                    {"code": "compiled_turn_shape_constraint_grammar"},
                    {"code": "compiled_by_extension_v1"},
                ]
                record["binding_fingerprint"] = content_sha256(record)
            elif record["requirement_id"] in _ZERO_CHARGE_PAYMENT_ARGUMENT_OVERRIDE:
                runtime = record.get("runtime_binding") or {}
                left_event = runtime.get("left_event") if isinstance(runtime, Mapping) else None
                matcher = (left_event or {}).get("matcher") if isinstance(left_event, Mapping) else None
                target_path = _ZERO_CHARGE_PAYMENT_ARGUMENT_OVERRIDE[record["requirement_id"]]
                if (
                    isinstance(left_event, Mapping)
                    and left_event.get("binding_kind") == "assistant_argument_disclosure"
                    and isinstance(matcher, Mapping)
                    and target_path in (matcher.get("required_argument_paths") or [])
                ):
                    record = deepcopy(record)
                    record.pop("binding_fingerprint", None)
                    new_runtime = deepcopy(record["runtime_binding"])
                    new_paths = [
                        p for p in new_runtime["left_event"]["matcher"]["required_argument_paths"]
                        if p != target_path
                    ]
                    new_runtime["left_event"]["matcher"]["required_argument_paths"] = new_paths
                    new_runtime["program_source"] = PROGRAM_SOURCE
                    record["runtime_binding"] = new_runtime
                    record["diagnostics"] = list(record["diagnostics"]) + [
                        {"code": "removed_zero_charge_payment_argument_requirement", "path": target_path},
                        {"code": "compiled_by_extension_v1"},
                    ]
                    record["binding_fingerprint"] = content_sha256(record)
            elif (
                record["requirement_id"] in _TOOL_CALL_ANY_OF_OVERRIDE
                and record["binding_status"] == "bound"
                and isinstance(runtime, Mapping)
                and runtime.get("extractor_kind") == "event_filter"
            ):
                # See _TOOL_CALL_ANY_OF_OVERRIDE's own comment: the frozen
                # baseline already bound this requirement to ONE literal
                # tool_name (event_filter/field_equals), which real-fails a
                # compliant agent that established the same fact via a
                # different, equally valid real tool. Reuses the exact
                # "tool_call_any_of" runtime_binding shape already live in
                # production for retail_053_order_find_user_id_by_email#b0
                # (matcher: {kind: tool_name_in, tool_names: [...]}) --
                # event_filter itself stays bare (event_kind only, no
                # field_equals) since the real disjunction lives in matcher,
                # not event_filter (see _extract_tool_call_any_of_observations's
                # own docstring). compile_oracle_evaluator_contracts_extended's
                # own is_tool_call_any_of override (gated on this same
                # extractor_kind) then compiles the matching evaluator_contract
                # program automatically -- no further change needed there.
                tool_names = _TOOL_CALL_ANY_OF_OVERRIDE[record["requirement_id"]]
                record = deepcopy(record)
                record.pop("binding_fingerprint", None)
                new_runtime = deepcopy(record["runtime_binding"])
                new_runtime["event_filter"] = {"event_kind": "assistant_tool_call"}
                new_runtime["matcher"] = {"kind": "tool_name_in", "tool_names": list(tool_names)}
                new_runtime["extractor_kind"] = "tool_call_any_of"
                new_runtime["program_source"] = PROGRAM_SOURCE
                record["runtime_binding"] = new_runtime
                record["diagnostics"] = list(record["diagnostics"]) + [
                    {"code": "widened_tool_call_any_of_evidence", "tool_names": list(tool_names)},
                    {"code": "compiled_by_extension_v1"},
                ]
                record["binding_fingerprint"] = content_sha256(record)
            # Section 173 (Bug A): independent of every branch above (they only
            # touch non-semantic_deferred shapes or the baggage override, which
            # turns the record "bound" first) -- see
            # statement_prohibition_polarity_rewrite.
            polarity_rewritten = _polarity_rewritten_record(record, branch_then)
            if polarity_rewritten is not None:
                record = polarity_rewritten
            record = _drop_unconstrained_free_text_scope_constraints(record)
            records.append(record)
            status_counts[record["binding_status"]] += 1
        branch_statuses = {record["binding_status"] for record in records}
        if "needs_adjudication" in branch_statuses:
            branch_status = "needs_adjudication"
        elif branch_statuses <= {"bound"}:
            branch_status = "bound"
        else:
            branch_status = "partially_bound"
            unresolved_branches.append(source_branch["branch_id"])
        branches.append(
            {
                "branch_id": source_branch["branch_id"],
                "branch_context": deepcopy(source_branch["branch_context"]),
                "bindings": records,
                "binding_status": branch_status,
                "potential_overlap_hints": deepcopy(
                    source_branch.get("potential_overlap_hints") or []
                ),
            }
        )

    binding_count = sum(len(branch["bindings"]) for branch in branches)
    result = {
        "schema_version": EXTENDED_SET_VERSION,
        "source_binding_set_fingerprint": baseline["binding_set_fingerprint"],
        "runtime_observation_profile_id": baseline["runtime_observation_profile_id"],
        "canonical_event_contract": deepcopy(baseline["canonical_event_contract"]),
        "raw_message_adapter": deepcopy(baseline["raw_message_adapter"]),
        "branches": branches,
        "summary": {
            "branch_count": len(branches),
            "binding_count": binding_count,
            "binding_status_counts": {
                status: status_counts[status] for status in sorted(BINDING_STATUSES)
            },
            "fully_bound_branch_count": sum(
                branch["binding_status"] == "bound" for branch in branches
            ),
            "partially_bound_branch_count": sum(
                branch["binding_status"] == "partially_bound" for branch in branches
            ),
            "adjudication_branch_count": sum(
                branch["binding_status"] == "needs_adjudication" for branch in branches
            ),
        },
        "unresolved_branches": unresolved_branches,
        "next_stage": "evaluator_contract_generation",
    }
    result["binding_set_fingerprint"] = content_sha256(result)
    return result


def validate_runtime_observation_binding_set_extended(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$runtime_bindings_extended")))
    fingerprint = result.pop("binding_set_fingerprint", None)
    if result.get("schema_version") != EXTENDED_SET_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid extended runtime observation binding set")
    identities = []
    statuses: Counter[str] = Counter()
    for branch in result.get("branches") or []:
        for raw in branch.get("bindings") or []:
            record = deepcopy(dict(_mapping(raw, "$.bindings[]")))
            record_fingerprint = record.pop("binding_fingerprint", None)
            if record.get("schema_version") != BINDING_VERSION or record_fingerprint != content_sha256(record):
                raise ThenAtomizationError("invalid runtime observation binding")
            if record.get("branch_id") != branch.get("branch_id"):
                raise ThenAtomizationError("runtime observation binding branch mismatch")
            if record.get("binding_status") not in BINDING_STATUSES:
                raise ThenAtomizationError("invalid runtime observation binding status")
            identities.append(record.get("binding_id"))
            statuses[record["binding_status"]] += 1
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("runtime observation binding IDs must be unique")
    summary = _mapping(result.get("summary"), "$.summary")
    if summary.get("binding_count") != len(identities):
        raise ThenAtomizationError("runtime observation binding count mismatch")
    expected = {status: statuses[status] for status in sorted(BINDING_STATUSES)}
    if summary.get("binding_status_counts") != expected:
        raise ThenAtomizationError("runtime observation binding status counts mismatch")
    result["binding_set_fingerprint"] = fingerprint
    return result


def apply_semantic_event_alias_patches(
    extended_binding_set: Mapping[str, Any],
    alias_vocabulary: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Post-processing patch over an already-validated extended binding set,
    for real semantic_requirement candidates whose target IS a single, real,
    named tool_call, but which the frozen same-branch-alias heuristic
    (_binding_for_requirement's "semantic_requirement" branch) cannot match:
    that heuristic only accepts a sibling tool_call candidate whose TOOL'S
    OWN DESCRIPTION text overlaps the semantic requirement's wording -- these
    real claims are instead about the tool's internal business-logic error
    text (e.g. retail's real gift-card-balance/payment-method-mismatch
    ValueErrors) or the tool's mere occurrence (e.g. telecom's real
    resume_line/enable_roaming/transfer_to_human_agents calls), neither of
    which is reflected in the tool's own description (see
    docs/agentcoveragetesting_reuse_log.md section 44).

    Keyed by requirement_id (not requirement_text): each entry is one
    specific, individually real-code-verified candidate, not a broadly
    reusable phrase the way the other Family I vocabularies are.

    Applied as a POST-PROCESSING step over the already-compiled,
    already-persisted binding set rather than by re-running
    compile_runtime_observation_bindings_extended from raw inputs: this
    project has no independently reconstructable tool_catalog_document/
    semantic_unit_model for telecom/retail (those inputs were airline-only,
    extracted from a frozen v1.1.0 archive that has no telecom/retail
    counterpart -- see prepare_step4_support_files_v0_1.py's own docstring),
    so re-deriving them from scratch risked silently diverging from the
    real, already-validated bindings section 39-43's work produced for
    every OTHER requirement. Patching the persisted document directly in
    place keeps every other record byte-for-byte unchanged.

    alias_vocabulary[requirement_id] = {"tool_name": str, "resolve_as":
    "bound" | "outcome_alias"}. "bound": the real claim is fully satisfied
    by the target event's mere presence/absence (its
    expected_observation.operator is already present/absent) -- setting
    binding_status to "bound" lets the FROZEN base compiler's own
    match_cardinality evaluator handle the rest; no new predicate_kind is
    needed. "outcome_alias": the real claim needs the target event's
    RESULT content (a real tool error text) -- binding_status is
    deliberately left as "semantic_deferred" (anything other than "bound")
    so the frozen baseline compiler still reports evaluator_status ==
    "deferred", which is what lets
    compile_oracle_evaluator_contracts_extended's TOOL_CALL_OUTCOME_RULES
    dispatch (gated on evaluator_status == "deferred") apply.
    """
    status_counts: Counter[str] = Counter()
    branches = []
    for branch in extended_binding_set["branches"]:
        records = []
        for record in branch["bindings"]:
            patch = alias_vocabulary.get(record["requirement_id"])
            if patch is not None and record["binding_status"] == "semantic_deferred":
                record = deepcopy(dict(record))
                record.pop("binding_fingerprint", None)
                record["runtime_binding"] = {
                    "extractor_kind": "event_filter",
                    "event_stream": "canonical_tau_events",
                    "event_filter": {
                        "event_kind": "assistant_tool_call",
                        "field_equals": {"tool_name": patch["tool_name"]},
                    },
                    "scope_constraints": [],
                    "projection": {"kind": "event"},
                    "program_source": PROGRAM_SOURCE,
                }
                if patch["resolve_as"] == "bound":
                    record["binding_status"] = "bound"
                record["diagnostics"] = list(record["diagnostics"]) + [
                    {
                        "code": "compiled_semantic_event_alias_vocabulary",
                        "tool_name": patch["tool_name"],
                        "resolve_as": patch["resolve_as"],
                    },
                    {"code": "compiled_by_extension_v1"},
                ]
                record["binding_fingerprint"] = content_sha256(record)
            else:
                record = deepcopy(dict(record))
            records.append(record)
            status_counts[record["binding_status"]] += 1
        branch_statuses = {r["binding_status"] for r in records}
        if "needs_adjudication" in branch_statuses:
            branch_status = "needs_adjudication"
        elif branch_statuses <= {"bound"}:
            branch_status = "bound"
        else:
            branch_status = "partially_bound"
        branches.append({**deepcopy(dict(branch)), "bindings": records, "binding_status": branch_status})

    binding_count = sum(len(branch["bindings"]) for branch in branches)
    result = {
        key: value
        for key, value in extended_binding_set.items()
        if key not in ("branches", "summary", "unresolved_branches", "binding_set_fingerprint")
    }
    result["branches"] = branches
    result["summary"] = {
        "branch_count": len(branches),
        "binding_count": binding_count,
        "binding_status_counts": {
            status: status_counts[status] for status in sorted(BINDING_STATUSES)
        },
        "fully_bound_branch_count": sum(
            branch["binding_status"] == "bound" for branch in branches
        ),
        "partially_bound_branch_count": sum(
            branch["binding_status"] == "partially_bound" for branch in branches
        ),
        "adjudication_branch_count": sum(
            branch["binding_status"] == "needs_adjudication" for branch in branches
        ),
    }
    result["unresolved_branches"] = [
        branch["branch_id"] for branch in branches if branch["binding_status"] == "partially_bound"
    ]
    result["binding_set_fingerprint"] = content_sha256(result)
    return result


def _try_left_event_fixes(
    left_event: Mapping[str, Any],
    dialogue_act_keywords: Mapping[Any, tuple[str, ...]],
    effort_prerequisite_vocabulary: Mapping[Any, Sequence[str]],
    prerequisite_lookup_by_semantic_unit_vocabulary: Mapping[Any, Sequence[str]] = MappingProxyType({}),
    fact_verification_prerequisite_vocabulary: Mapping[Any, Sequence[str]] = MappingProxyType({}),
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Tries the left-event grammars that need no real semantic_unit_
    model lookup (unlike section 39's own named-fact/prerequisite-lookup,
    which needs a real direct_action_text telecom/retail do not have) --
    fact_verification first (section 173: a unit listed there must never
    reach the literal dialogue-act proxy), then dialogue_act (a spoken claim),
    then effort_prerequisite (a
    behavioral, necessary-condition-only proxy; see docs/
    agentcoveragetesting_reuse_log.md section 50), then prerequisite_lookup_
    by_semantic_unit (section 106 -- a specific, named identity-lookup tool
    set, not "any other tool"). Returns (None, None) if no vocabulary has
    this semantic_unit_id.
    """
    semantic_unit_id = left_event.get("semantic_unit_id")
    new_left, diagnostic = _fact_verification_before_operation(
        semantic_unit_id, fact_verification_prerequisite_vocabulary,
    )
    if new_left is not None:
        return new_left, diagnostic
    new_left, diagnostic = _dialogue_act_before_operation(semantic_unit_id, dialogue_act_keywords)
    if new_left is not None:
        return new_left, diagnostic
    new_left, diagnostic = _effort_prerequisite_before_operation(
        semantic_unit_id, effort_prerequisite_vocabulary,
    )
    if new_left is not None:
        return new_left, diagnostic
    new_left, diagnostic = _prerequisite_lookup_by_semantic_unit_before_operation(
        semantic_unit_id, prerequisite_lookup_by_semantic_unit_vocabulary,
    )
    if new_left is not None:
        return new_left, diagnostic
    return None, None


def resolve_ambiguous_temporal_right_events(
    extended_binding_set: Mapping[str, Any],
    dialogue_act_keywords: Mapping[Any, tuple[str, ...]],
    effort_prerequisite_vocabulary: Mapping[Any, Sequence[str]] = MappingProxyType({}),
    generic_requested_operation_vocabulary: Mapping[Any, Sequence[str]] = MappingProxyType({}),
    prerequisite_lookup_by_semantic_unit_vocabulary: Mapping[Any, Sequence[str]] = MappingProxyType({}),
    fact_verification_prerequisite_vocabulary: Mapping[Any, Sequence[str]] = MappingProxyType({}),
) -> dict[str, Any]:
    """Post-processing patch over an already-validated extended binding set
    (see docs/agentcoveragetesting_reuse_log.md section 49), same reason and
    same pattern as apply_semantic_event_alias_patches: this project has no
    independently reconstructable tool_catalog_document/semantic_unit_model
    for telecom/retail, so re-running compile_runtime_observation_bindings_
    extended from raw inputs risks silently diverging every OTHER already-
    correct binding in the corpus, which used whatever real tool_catalog
    produced the currently-persisted document. _resolve_ambiguous_right_event
    (used here) needs no tool_catalog/semantic_units at all -- it only
    reinterprets a right_event's own already-computed operation_resolution --
    so patching the persisted document directly is both safer and sufficient.

    Two real shapes, both real candidates in this corpus:

    1. right_event still "partial": tries the two safe overrides (sole
    verified candidate; tied top score) via _resolve_ambiguous_right_event,
    then -- only if the right side actually resolved -- tries a left_event
    fix. A record whose right side resolves but whose left side does not is
    left "partial" (an honest, real improvement over the original
    diagnostics, not a false "bound"); a record neither right-side override
    applies to (e.g. telecom_055_order#b0, genuinely ambiguous among
    candidates most of which have no verified_tool_names at all) is left
    completely untouched.

    2. right_event already "bound" (to exactly one real tool) but left_event
    is still semantic_event_deferred (e.g. telecom_056_norm#b0, whose right
    side the frozen scorer already resolved cleanly): tries a left_event fix
    directly, no right-side change needed.
    """
    status_counts: Counter[str] = Counter()
    branches = []
    for branch in extended_binding_set["branches"]:
        records = []
        for record in branch["bindings"]:
            runtime = record.get("runtime_binding") if isinstance(record.get("runtime_binding"), Mapping) else {}
            right_event = runtime.get("right_event") if isinstance(runtime, Mapping) else None
            left_event = runtime.get("left_event") if isinstance(runtime, Mapping) else None
            is_temporal = runtime.get("extractor_kind") == "temporal_relation"
            left_is_deferred = (
                isinstance(left_event, Mapping) and left_event.get("binding_kind") == "semantic_event_deferred"
            )
            if (
                record["binding_status"] == "partial"
                and is_temporal
                and isinstance(right_event, Mapping)
                and right_event.get("binding_status") == "partial"
            ):
                new_right, right_diagnostic = _resolve_ambiguous_right_event(
                    right_event, record.get("requirement_id"), generic_requested_operation_vocabulary,
                )
                if new_right is not None:
                    record = deepcopy(dict(record))
                    record.pop("binding_fingerprint", None)
                    new_runtime = deepcopy(dict(runtime))
                    new_runtime["right_event"] = new_right
                    new_runtime["program_source"] = PROGRAM_SOURCE
                    diagnostics_to_add = [right_diagnostic]
                    new_status = "partial"
                    if left_is_deferred:
                        new_left, left_diagnostic = _try_left_event_fixes(
                            left_event, dialogue_act_keywords, effort_prerequisite_vocabulary,
                            prerequisite_lookup_by_semantic_unit_vocabulary,
                            fact_verification_prerequisite_vocabulary,
                        )
                        if new_left is not None:
                            new_runtime["left_event"] = new_left
                            diagnostics_to_add.append(left_diagnostic)
                            new_status = "bound"
                    record["runtime_binding"] = new_runtime
                    record["binding_status"] = new_status
                    record["diagnostics"] = list(record["diagnostics"]) + diagnostics_to_add + [
                        {"code": "compiled_by_extension_v1"},
                    ]
                    record["binding_fingerprint"] = content_sha256(record)
                else:
                    record = deepcopy(dict(record))
            elif (
                record["binding_status"] == "partial"
                and is_temporal
                and left_is_deferred
                and isinstance(right_event, Mapping)
                and right_event.get("binding_status") == "bound"
                and len(right_event.get("exact_tool_names") or []) == 1
            ):
                new_left, left_diagnostic = _try_left_event_fixes(
                    left_event, dialogue_act_keywords, effort_prerequisite_vocabulary,
                    prerequisite_lookup_by_semantic_unit_vocabulary,
                    fact_verification_prerequisite_vocabulary,
                )
                if new_left is not None:
                    record = deepcopy(dict(record))
                    record.pop("binding_fingerprint", None)
                    new_runtime = deepcopy(dict(runtime))
                    new_runtime["left_event"] = new_left
                    new_runtime["program_source"] = PROGRAM_SOURCE
                    record["runtime_binding"] = new_runtime
                    record["binding_status"] = "bound"
                    record["diagnostics"] = list(record["diagnostics"]) + [
                        left_diagnostic, {"code": "compiled_by_extension_v1"},
                    ]
                    record["binding_fingerprint"] = content_sha256(record)
                else:
                    record = deepcopy(dict(record))
            else:
                record = deepcopy(dict(record))
            records.append(record)
            status_counts[record["binding_status"]] += 1
        branch_statuses = {r["binding_status"] for r in records}
        if "needs_adjudication" in branch_statuses:
            branch_status = "needs_adjudication"
        elif branch_statuses <= {"bound"}:
            branch_status = "bound"
        else:
            branch_status = "partially_bound"
        branches.append({**deepcopy(dict(branch)), "bindings": records, "binding_status": branch_status})

    binding_count = sum(len(branch["bindings"]) for branch in branches)
    result = {
        key: value
        for key, value in extended_binding_set.items()
        if key not in ("branches", "summary", "unresolved_branches", "binding_set_fingerprint")
    }
    result["branches"] = branches
    result["summary"] = {
        "branch_count": len(branches),
        "binding_count": binding_count,
        "binding_status_counts": {
            status: status_counts[status] for status in sorted(BINDING_STATUSES)
        },
        "fully_bound_branch_count": sum(
            branch["binding_status"] == "bound" for branch in branches
        ),
        "partially_bound_branch_count": sum(
            branch["binding_status"] == "partially_bound" for branch in branches
        ),
        "adjudication_branch_count": sum(
            branch["binding_status"] == "needs_adjudication" for branch in branches
        ),
    }
    result["unresolved_branches"] = [
        branch["branch_id"] for branch in branches if branch["binding_status"] == "partially_bound"
    ]
    result["binding_set_fingerprint"] = content_sha256(result)
    return result


def _rebuilt_binding_set(
    extended_binding_set: Mapping[str, Any], branches: list[dict[str, Any]]
) -> dict[str, Any]:
    """Same set-level summary/fingerprint recomputation the two post-processing
    patches above perform inline, shared by the section 173 patches below."""
    status_counts: Counter[str] = Counter(
        record["binding_status"] for branch in branches for record in branch["bindings"]
    )
    result = {
        key: value
        for key, value in extended_binding_set.items()
        if key not in ("branches", "summary", "unresolved_branches", "binding_set_fingerprint")
    }
    result["branches"] = branches
    result["summary"] = {
        "branch_count": len(branches),
        "binding_count": sum(len(branch["bindings"]) for branch in branches),
        "binding_status_counts": {
            status: status_counts[status] for status in sorted(BINDING_STATUSES)
        },
        "fully_bound_branch_count": sum(branch["binding_status"] == "bound" for branch in branches),
        "partially_bound_branch_count": sum(
            branch["binding_status"] == "partially_bound" for branch in branches
        ),
        "adjudication_branch_count": sum(
            branch["binding_status"] == "needs_adjudication" for branch in branches
        ),
    }
    # Preserve the persisted document's own unresolved_branches ORDER (the real
    # airline document lists airline_075_norm#b0 last, appended by an earlier
    # patch) so an unchanged branch set leaves this field byte-identical.
    partially_bound = [
        branch["branch_id"] for branch in branches if branch["binding_status"] == "partially_bound"
    ]
    original_order = list(extended_binding_set.get("unresolved_branches") or [])
    result["unresolved_branches"] = [
        branch_id for branch_id in original_order if branch_id in partially_bound
    ] + [branch_id for branch_id in partially_bound if branch_id not in original_order]
    result["binding_set_fingerprint"] = content_sha256(result)
    return result


def apply_fact_verification_left_event_patches(
    extended_binding_set: Mapping[str, Any],
    fact_verification_prerequisite_vocabulary: Mapping[Any, Sequence[str]],
) -> dict[str, Any]:
    """Post-processing patch (section 173, task_eba70b14) over an already-
    persisted binding set, same discipline as apply_semantic_event_alias_
    patches/resolve_ambiguous_temporal_right_events: re-derives ONLY the
    records whose left_event was compiled by the literal dialogue-act grammar
    for a semantic unit that now has a fact-verification entry (identified by
    the record's own real "compiled_dialogue_act_grammar" diagnostic, since a
    bound record's left_event list no longer carries the semantic_unit_id);
    every other record is copied unchanged. The right_event, relation and
    binding_status are untouched -- only left_event is replaced, with the
    exact shape _fact_verification_before_operation produces on a fresh
    compile, so a from-scratch recompile and this patch agree."""
    branches = []
    for branch in extended_binding_set["branches"]:
        records = []
        for record in branch["bindings"]:
            record = deepcopy(dict(record))
            runtime = record.get("runtime_binding") if isinstance(record.get("runtime_binding"), Mapping) else {}
            dialogue_unit = next(
                (
                    diagnostic.get("matched_semantic_unit_id")
                    for diagnostic in record.get("diagnostics") or []
                    if isinstance(diagnostic, Mapping)
                    and diagnostic.get("code") == "compiled_dialogue_act_grammar"
                ),
                None,
            )
            left_event = runtime.get("left_event")
            if (
                dialogue_unit is not None
                and runtime.get("extractor_kind") == "temporal_relation"
                and isinstance(left_event, list)
                and all(
                    isinstance(item, Mapping) and item.get("binding_kind") == "assistant_dialogue_act"
                    for item in left_event
                )
            ):
                new_left, diagnostic = _fact_verification_before_operation(
                    dialogue_unit, fact_verification_prerequisite_vocabulary,
                )
                if new_left is not None:
                    record.pop("binding_fingerprint", None)
                    new_runtime = deepcopy(dict(runtime))
                    new_runtime["left_event"] = new_left
                    new_runtime["program_source"] = PROGRAM_SOURCE
                    record["runtime_binding"] = new_runtime
                    record["diagnostics"] = list(record["diagnostics"]) + [
                        {
                            "code": "superseded_dialogue_act_left_event",
                            "superseded_left_event": deepcopy(left_event),
                        },
                        diagnostic,
                        {"code": "compiled_by_extension_v1"},
                    ]
                    record["binding_fingerprint"] = content_sha256(record)
            records.append(record)
        branches.append({**deepcopy(dict(branch)), "bindings": records})
    return _rebuilt_binding_set(extended_binding_set, branches)


# docs/agentcoveragetesting_reuse_log.md section 173 (Bug A): a Then of the
# shape "The agent must not state that <P>" whose accepted requirement is a
# policy-FACT semantic unit (e.g. airline_030_arg#b0::OR01 "costs $30 per
# passenger", sourced via candidate_origin coverage_semantic_unit_via_sibling_
# lineage) rather than the Then sentence itself. The expectation compiler
# (oracle_expectation_compiler_v1.py: "Then modality directly determines the
# expected observation") applies the Then's OUTER "must not" to that
# requirement_text as if it were the forbidden statement, and the frozen
# binding compiler copies the text verbatim into runtime_binding.requirement_
# text -- so when <P> is the NEGATION of the policy fact ("travel insurance
# does not cost $30 per passenger"), the compiled check forbids the CORRECT
# fact ("The following must NOT be true: 'costs $30 per passenger'"). The
# LLM selection step itself had it right ("the two insurance facts the then
# says the agent must not contradict"); the polarity relation between <P>
# and the fact was lost when the operator was applied. Real online effect:
# airline_030_arg#b0 OR01 failed every round for stating "$30 per passenger"
# (tau2 airline policy.md line 101), and airline_048_arg#b0 OR01 (the mirror
# shape: <P> "prices ... will be updated", fact "their prices will not be
# updated", policy.md line 112) failed pass4/5/6 for correctly telling the
# user kept segments keep their original price.
#
# Detection is mechanical and conservative: the Then must be a statement-
# prohibition; the requirement_text must not itself be a normative sentence
# (the section-71 "The agent must not state ..." texts are already correct);
# the <P> clause the requirement is about is chosen by content-token overlap
# (unique best clause, >= 2 shared tokens); and the rewrite fires only when
# that clause's negation polarity DIFFERS from the requirement_text's. When
# they agree (a Then that really forbids stating the fact itself) nothing is
# changed. The rewrite keeps operator="absent" and turns the text into "state
# anything that contradicts the policy fact ...", which is exactly what the
# double-negative Then forbids.
_STATEMENT_PROHIBITION_THEN = re.compile(
    r"^\s*the agent\s+(?:must|should|shall|may)\s+(?:not|never)\s+"
    r"(?:state|say|claim|tell|inform|imply|suggest|assert|indicate|mention|promise|represent)"
    r"(?:\s+or\s+(?:state|say|claim|imply|suggest))?"
    r"(?:\s+(?:to\s+)?(?:the\s+)?(?:user|customer))?"
    r"\s+that\s+(?P<proposition>.+?)\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_NORMATIVE_REQUIREMENT_TEXT = re.compile(r"^\s*the agent\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:not|never|no|cannot)\b|n't\b", re.IGNORECASE)
_PROPOSITION_CLAUSE_SPLIT = re.compile(r",?\s+or\s+(?:that\s+)?|;\s*", re.IGNORECASE)
_POLARITY_TOKEN = re.compile(r"[a-z0-9$]+")
_POLARITY_STOPWORDS = frozenset({
    "a", "an", "the", "that", "this", "to", "of", "for", "if", "is", "are", "be",
    "will", "would", "does", "do", "doe", "not", "never", "no", "cannot", "and", "or",
    "their", "its", "it", "they", "by", "on", "in", "at", "due", "given", "with",
})


def _polarity_tokens(text: str) -> set[str]:
    tokens = set()
    for raw in _POLARITY_TOKEN.findall(text.casefold().replace("n't", " not")):
        token = raw[:-1] if raw.endswith("s") and len(raw) > 3 else raw
        if token not in _POLARITY_STOPWORDS:
            tokens.add(token)
    return tokens


def statement_prohibition_polarity_rewrite(
    then: str, requirement_text: str
) -> dict[str, Any] | None:
    """Returns {"requirement_text": <rewritten>, "matched_then_clause": ...,
    ...} when this (Then, requirement_text) pair is the confirmed section 173
    polarity-inversion shape, else None. Pure function; see the comment block
    above for the exact gate."""
    if not then or not requirement_text or _NORMATIVE_REQUIREMENT_TEXT.search(requirement_text):
        return None
    match = _STATEMENT_PROHIBITION_THEN.match(then)
    if match is None:
        return None
    clauses = [c.strip() for c in _PROPOSITION_CLAUSE_SPLIT.split(match.group("proposition")) if c.strip()]
    requirement_tokens = _polarity_tokens(requirement_text)
    scored = sorted(
        ((len(_polarity_tokens(clause) & requirement_tokens), clause) for clause in clauses),
        key=lambda item: -item[0],
    )
    if not scored or scored[0][0] < 2 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
        return None
    clause = scored[0][1]
    clause_negated = bool(_NEGATION.search(clause))
    requirement_negated = bool(_NEGATION.search(requirement_text))
    if clause_negated == requirement_negated:
        return None
    rewritten = (
        f"state anything that contradicts the policy fact \"{requirement_text.strip()}\" "
        f"(the rule being checked: {then.strip()}) -- correctly stating or "
        "paraphrasing that policy fact itself complies with the rule and is not a violation"
    )
    return {
        "requirement_text": rewritten,
        "source_requirement_text": requirement_text,
        "matched_then_clause": clause,
        "then_clause_negated": clause_negated,
        "requirement_text_negated": requirement_negated,
    }


def _polarity_rewritten_record(record: Mapping[str, Any], then: str) -> dict[str, Any] | None:
    runtime = record.get("runtime_binding") if isinstance(record.get("runtime_binding"), Mapping) else {}
    if (
        record.get("binding_status") != "semantic_deferred"
        or runtime.get("extractor_kind") != "semantic_event_matcher_deferred"
        or runtime.get("source_requirement_text") is not None
    ):
        return None
    rewrite = statement_prohibition_polarity_rewrite(then, str(runtime.get("requirement_text") or ""))
    if rewrite is None:
        return None
    new_record = deepcopy(dict(record))
    new_record.pop("binding_fingerprint", None)
    new_runtime = deepcopy(dict(runtime))
    new_runtime["requirement_text"] = rewrite["requirement_text"]
    new_runtime["source_requirement_text"] = rewrite["source_requirement_text"]
    new_record["runtime_binding"] = new_runtime
    new_record["diagnostics"] = list(new_record.get("diagnostics") or []) + [
        {
            "code": "rewrote_statement_prohibition_polarity_inversion",
            "matched_then_clause": rewrite["matched_then_clause"],
            "then_clause_negated": rewrite["then_clause_negated"],
            "requirement_text_negated": rewrite["requirement_text_negated"],
        },
        {"code": "compiled_by_extension_v1"},
    ]
    new_record["binding_fingerprint"] = content_sha256(new_record)
    return new_record


def compile_runtime_observation_bindings_reproducible(
    accepted_requirement_set: Mapping[str, Any],
    expectation_set: Mapping[str, Any],
    runtime_observation_profile: Mapping[str, Any],
    execution_matching_contracts: Mapping[str, Any],
    tool_catalog_document: Mapping[str, Any] | None = None,
    semantic_unit_model: Mapping[str, Any] | None = None,
    given_conditions_document: Mapping[str, Any] | None = None,
    *,
    binding_vocabulary_config: Mapping[str, Any],
    stamp_source_lineage: bool = True,
) -> dict[str, Any]:
    """Section 194 (task_fd3f3d3b): the full from-scratch Step4a derivation
    for a domain whose persisted bindings were built as "compile, then the
    documented post-processing patches" (retail): compile_runtime_
    observation_bindings_extended, then -- in the order the sections that
    introduced them applied them -- apply_semantic_event_alias_patches
    (section 44), resolve_ambiguous_temporal_right_events (sections 47/49/
    50/106), apply_fact_verification_left_event_patches and apply_statement_
    prohibition_polarity_patches (section 173, both idempotent), and finally
    the two top-level lineage keys Step5's compile_contracts_v2 checks
    (section 109.6). compile_runtime_observation_bindings_extended itself is
    unchanged, so every existing caller is byte-identical.

    Section 197: airline and telecom main use the same chain (telecom with
    its converted tool catalog and semantic unit model, exactly like retail);
    their persisted sets never carried the section 109.6 lineage keys, so they
    pass stamp_source_lineage=False."""
    doc = compile_runtime_observation_bindings_extended(
        accepted_requirement_set, expectation_set, runtime_observation_profile, execution_matching_contracts,
        tool_catalog_document, semantic_unit_model, given_conditions_document,
        binding_vocabulary_config=binding_vocabulary_config,
    )
    doc = apply_semantic_event_alias_patches(doc, binding_vocabulary_config.get("semantic_event_alias_vocabulary") or {})
    doc = resolve_ambiguous_temporal_right_events(
        doc,
        binding_vocabulary_config.get("dialogue_act_keywords") or {},
        binding_vocabulary_config.get("effort_prerequisite_vocabulary") or {},
        binding_vocabulary_config.get("generic_requested_operation_vocabulary") or {},
        binding_vocabulary_config.get("prerequisite_lookup_by_semantic_unit_vocabulary") or {},
        binding_vocabulary_config.get("fact_verification_prerequisite_vocabulary") or {},
    )
    doc = apply_fact_verification_left_event_patches(
        doc, binding_vocabulary_config.get("fact_verification_prerequisite_vocabulary") or {}
    )
    doc = apply_statement_prohibition_polarity_patches(doc)
    if not stamp_source_lineage:
        return doc
    doc = dict(doc)
    doc.pop("binding_set_fingerprint", None)
    doc["source_accepted_set_fingerprint"] = accepted_requirement_set["accepted_set_fingerprint"]
    doc["source_expectation_set_fingerprint"] = expectation_set["expectation_set_fingerprint"]
    doc["binding_set_fingerprint"] = content_sha256(doc)
    return doc


def apply_statement_prohibition_polarity_patches(
    extended_binding_set: Mapping[str, Any],
) -> dict[str, Any]:
    """Post-processing patch (section 173, Bug A) over an already-persisted
    binding set: rewrites only the semantic_deferred records the pure
    statement_prohibition_polarity_rewrite gate selects, reading each branch's
    own real branch_context.gwt.then; every other record is copied unchanged.
    Idempotent (a record already carrying source_requirement_text is skipped).
    """
    branches = []
    for branch in extended_binding_set["branches"]:
        then = str(((branch.get("branch_context") or {}).get("gwt") or {}).get("then") or "")
        records = []
        for record in branch["bindings"]:
            rewritten = _polarity_rewritten_record(record, then)
            records.append(rewritten if rewritten is not None else deepcopy(dict(record)))
        branches.append({**deepcopy(dict(branch)), "bindings": records})
    return _rebuilt_binding_set(extended_binding_set, branches)


# docs/agentcoveragetesting_reuse_log.md section 137.5.3/140 (task_09886b79):
# the frozen _argument_disclosure_match (runtime_observation_binding_v1.py,
# pinned by scripts/prepare_v5_step4_v0_1.py's legacy-compiler-drift check --
# never edited here) requires every required_argument_path's scalar leaf to
# appear as a literal, contiguous substring of the left (assistant) message.
# A real retail_055_order_modify_pending_order_items#b0 transcript message
# describes a real item change purely via its real, already-disclosed
# human-readable attributes ("scent family: fresh, size: 50ml, gender:
# men") instead of the literal internal item_id, and describes a real
# payment method as "credit card ending in 5051208" instead of the literal
# credit_card_5051208 token -- reasonable, real customer-service phrasing
# the frozen strict matcher cannot recognize. Following this module's own
# "never edit the frozen file, only add a genuinely separate extension
# path" discipline (see the module docstring), this is a NEW, tolerant
# matcher living entirely here, used ONLY as a second-chance RETRY after
# the frozen evaluator already said "fail" for this exact narrow shape (see
# _shape_needs_tolerant_argument_disclosure_retry in
# oracle_evaluator_contract_extension_v1.py) -- it never runs as the
# primary evaluation path, so every check the frozen matcher already passes
# is completely unaffected.
def _surface_disclosed(required_surface: str, content_surface: str) -> bool:
    """Widened, relative to the frozen matcher's plain substring test: for a
    MULTI-token required surface only (a bare single token, e.g. a raw
    numeric id with no internal separator, still requires an exact
    substring hit, unchanged), also accepts the same tokens appearing in
    the same ORDER with other real words allowed in between -- what "X
    ending in Y" and similar real paraphrasing actually looks like. Every
    one of the required value's own tokens must still individually appear,
    in order, so an unrelated sentence that merely scatters the same words
    in a different order (or omits one) still correctly fails."""
    if not required_surface:
        return False
    if required_surface in content_surface:
        return True
    required_tokens = required_surface.split(" ")
    if len(required_tokens) < 2:
        return False
    content_tokens = content_surface.split(" ")
    pos = 0
    for token in required_tokens:
        try:
            pos = content_tokens.index(token, pos) + 1
        except ValueError:
            return False
    return True


# Real tau2 fixture payload keys used for "this record is currently valid/
# selectable" bookkeeping, not a real descriptive attribute of the thing
# itself -- excluded from both the grounding-dict search (a coincidental
# `"available": true` shared between two unrelated records must never count
# as a match) and the alias fingerprint it produces.
_ALIAS_SKIP_KEYS = frozenset({"available", "success"})


def _iter_nested_mappings(value: Any) -> "list[Mapping[str, Any]]":
    out: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        out.append(value)
        for child in value.values():
            out.extend(_iter_nested_mappings(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(_iter_nested_mappings(child))
    return out


def _parse_event_payload(event: Mapping[str, Any]) -> Any:
    content = event.get("content")
    if isinstance(content, (Mapping, list)):
        return content
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return None


def _find_grounding_dict(
    value: Any, prior_events: Sequence[Mapping[str, Any]]
) -> "tuple[str, Mapping[str, Any]] | None":
    """A real internal catalog identifier (e.g. a retail item_id) is
    routinely never spoken aloud by a reasonable live agent at all -- the
    real retail_055_order_modify_pending_order_items#b0 transcript
    describes the item change purely via its real, already-disclosed
    human-readable attributes, attributes that came from a REAL, earlier
    tool_result event in the SAME transcript whose own payload keys this
    exact id alongside those attributes. Only ever fires when a real prior
    event in this SAME transcript already established, structurally, that
    this exact scalar value co-occurs with those other real values (see
    _attribute_alias_surfaces) -- grounded in what was actually observed,
    not guessed. Searches most-recent-first and skips any event whose
    content isn't real, parseable JSON (ordinary assistant/user prose)."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    for event in reversed(prior_events):
        payload = _parse_event_payload(event)
        if payload is None:
            continue
        for mapping in _iter_nested_mappings(payload):
            for key, candidate in mapping.items():
                if key in _ALIAS_SKIP_KEYS or isinstance(candidate, (Mapping, list)):
                    continue
                if candidate == value:
                    return key, mapping
    return None


def _attribute_alias_surfaces(matched_key: str, grounding: Mapping[str, Any]) -> list[str]:
    """The real sibling descriptive fields of the grounding dict found by
    _find_grounding_dict -- prefers a real "options" sub-mapping (this
    domain's own established convention for an item variant's real
    descriptive attributes) when present; otherwise falls back to the
    grounding dict's own other real scalar fields (excluding the matched id
    field itself and _ALIAS_SKIP_KEYS)."""
    descriptive: Any = grounding.get("options")
    if not isinstance(descriptive, Mapping):
        descriptive = {
            key: val
            for key, val in grounding.items()
            if key != matched_key and key not in _ALIAS_SKIP_KEYS
        }
    surfaces = []
    for leaf in _scalar_leaves(descriptive):
        surface = _surface(leaf)
        if surface:
            surfaces.append(surface)
    return surfaces


# docs/agentcoveragetesting_reuse_log.md section 180 (task_2724cc75): two more
# real, grounded paraphrase shapes, found by replaying every stored transcript
# of every contains_right_call_argument_values check (3 domains). airline
# policy.md line 7 requires the agent to "list the action details and obtain
# explicit user confirmation (yes)"; in every real transcript that still failed
# after section 140 the confirmation-request message DID list every action
# detail, and the only values it did not repeat literally were:
#   1. the internal id of the conversation's own subject -- the reservation
#      being modified (airline_074_order#b1/#b4) or the requesting user's own
#      account (airline_074_order#b0, airline_079_order#b0, whose Then is
#      literally "obtain the user id from the user") -- which the user had
#      already supplied and the agent had already looked up;
#   2. a payment id described by its real type ("your gift card on file") when
#      the user's real profile holds exactly one payment method of that type.
# Both are accepted ONLY when grounded in this same transcript (see the two
# helpers' docstrings) and only inside the section-140 second-chance retry,
# so a check the frozen matcher already passes is untouched and nothing can
# flip pass->fail.
def _established_subject_identifier(
    path: str, value: Any, groundable_events: Sequence[Mapping[str, Any]]
) -> bool:
    """True when `value` (a top-level scalar argument ``arguments.<name>``) is
    the one, unambiguous subject the conversation already established: the
    user stated it in an earlier user message, the agent passed it as
    ``<name>`` to an earlier tool call, and the agent never passed any OTHER
    value as ``<name>`` before this message (e.g. it looked up only this one
    reservation/order/user). If the agent looked at two reservations, the
    confirmation must still say which one."""
    parts = path.split(".")
    if len(parts) != 2 or parts[0] != "arguments":
        return False
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return False
    name = parts[1]
    surface = _surface(value)
    if not surface:
        return False
    stated_by_user = any(
        event.get("event_kind") == "user_message" and _surface_disclosed(surface, _surface(event.get("content")))
        for event in groundable_events
    )
    if not stated_by_user:
        return False
    looked_up_values = {
        str((event.get("arguments") or {}).get(name))
        for event in groundable_events
        if event.get("event_kind") == "assistant_tool_call"
        and isinstance(event.get("arguments"), Mapping)
        and name in event["arguments"]
    }
    return looked_up_values == {str(value)}


def _unique_payment_source_surface(
    value: Any, groundable_events: Sequence[Mapping[str, Any]]
) -> str | None:
    """The surface of a payment method's real ``source`` (e.g. "gift card")
    when an earlier tool result in this transcript shows `value` as one entry
    of a payment-method mapping (keyed by id, each entry carrying a real
    ``source``) in which it is the ONLY entry with that source -- so naming
    the type alone identifies it. None otherwise (e.g. two credit cards)."""
    if isinstance(value, bool) or not isinstance(value, str):
        return None
    for event in reversed(groundable_events):
        if event.get("event_kind") != "tool_result":
            continue
        payload = _parse_event_payload(event)
        if payload is None:
            continue
        for mapping in _iter_nested_mappings(payload):
            entry = mapping.get(value)
            if not isinstance(entry, Mapping) or not isinstance(entry.get("source"), str):
                continue
            same_source = [
                other
                for other in mapping.values()
                if isinstance(other, Mapping) and other.get("source") == entry["source"]
            ]
            if len(same_source) == 1:
                return _surface(entry["source"]) or None
            return None
    return None


def _argument_disclosure_match_tolerant(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    matcher: Mapping[str, Any],
    prior_events: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    content = _surface(left.get("content"))
    required = []
    missing_paths = []
    for path in matcher.get("required_argument_paths") or []:
        value = _path(right, path)
        if value is None:
            missing_paths.append(path)
            continue
        required.extend(
            {"path": path, "value": leaf, "surface": _surface(leaf)}
            for leaf in _scalar_leaves(value)
        )
    left_index = left.get("event_index")
    groundable_events = [
        event
        for event in prior_events
        if left_index is None or event.get("event_index", -1) < left_index
    ]
    missing_values = []
    aliased_values = []
    established_values = []
    for item in required:
        if not item["surface"]:
            # Matches the frozen matcher's own pre-existing behavior: an
            # empty-string scalar leaf carries nothing real to verify
            # against the message's own content, so it is not counted as a
            # missing value.
            continue
        if _surface_disclosed(item["surface"], content):
            continue
        grounding = _find_grounding_dict(item["value"], groundable_events)
        alias_surfaces = _attribute_alias_surfaces(*grounding) if grounding is not None else []
        if alias_surfaces and all(_surface_disclosed(s, content) for s in alias_surfaces):
            aliased_values.append({**item, "alias_surfaces": alias_surfaces})
            continue
        # section 180: see _established_subject_identifier /
        # _unique_payment_source_surface above.
        if _path(right, item["path"]) == item["value"] and _established_subject_identifier(
            item["path"], item["value"], groundable_events
        ):
            established_values.append(item)
            continue
        source_surface = _unique_payment_source_surface(item["value"], groundable_events)
        if source_surface and _surface_disclosed(source_surface, content):
            aliased_values.append({**item, "alias_surfaces": [source_surface]})
            continue
        missing_values.append(item)
    passed = not missing_paths and not missing_values and bool(required)
    evidence = {
        "required_value_count": len(required),
        "missing_argument_paths": missing_paths,
        "missing_values": missing_values,
        "aliased_values": aliased_values,
    }
    if established_values:
        evidence["established_subject_values"] = established_values
    return passed, evidence


def extract_temporal_precedes_observations_tolerant_argument_disclosure(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
) -> "dict[str, Any] | None":
    """Mirrors the frozen _extract_temporal_observations's real "precedes"
    relation logic (runtime_observation_binding_v1.py) exactly, with ONE
    difference: uses _argument_disclosure_match_tolerant (above) instead of
    the frozen file's strict _argument_disclosure_match, and ONLY for a
    left_event whose matcher.kind is "contains_right_call_argument_values"
    (returns None for any other real shape, deferring to the frozen
    result -- this function is only ever consulted as a second-chance retry,
    see oracle_evaluator_contract_extension_v1.py's
    _shape_needs_tolerant_argument_disclosure_retry, never as a primary
    evaluation path)."""
    left_contract = _mapping(runtime.get("left_event"), "$.left_event")
    right_contract = _mapping(runtime.get("right_event"), "$.right_event")
    if (
        left_contract.get("binding_kind") == "semantic_event_deferred"
        or right_contract.get("binding_status") != "bound"
    ):
        return None
    matcher = left_contract.get("matcher") or {}
    if matcher.get("kind") != "contains_right_call_argument_values":
        return None
    right_filter = _mapping(right_contract.get("event_filter"), "$.right_event.event_filter")
    right_events = []
    missing_bindings = []
    for event in events:
        if not _event_matches_filter(event, right_filter):
            continue
        scope_ok, missing = _scope_matches(
            event, right_contract.get("scope_constraints") or [], driver_bindings
        )
        missing_bindings.extend(missing)
        if scope_ok:
            right_events.append(event)
    if missing_bindings:
        return {
            "status": "missing_driver_bindings",
            "missing_driver_bindings": sorted(set(missing_bindings)),
            "matches": [],
        }
    left_filter = _mapping(left_contract.get("event_filter"), "$.left_event.event_filter")
    matches = []
    unmatched_targets = []
    for right in right_events:
        candidates = [
            event
            for event in events
            if event.get("event_index", -1) < right.get("event_index", -1)
            and _event_matches_filter(event, left_filter)
        ]
        witness = None
        witness_evidence = None
        for left in reversed(candidates):
            passed, evidence = _argument_disclosure_match_tolerant(left, right, matcher, events)
            if not passed:
                continue
            witness = left
            witness_evidence = evidence
            break
        if witness is None:
            unmatched_targets.append(right.get("event_index"))
            continue
        matches.append(
            {
                "event_index": right.get("event_index"),
                "source_message_index": right.get("source_message_index"),
                "value": {
                    "relation": "precedes",
                    "left_event": deepcopy(dict(witness)),
                    "right_event": deepcopy(dict(right)),
                    "matcher_evidence": witness_evidence,
                },
            }
        )
    return {
        "status": "observed",
        "target_event_count": len(right_events),
        "matched_target_count": len(matches),
        "unmatched_target_event_indices": unmatched_targets,
        "matches": matches,
    }


# docs/agentcoveragetesting_reuse_log.md section 145.5.2 (task_8c73cea0):
# contains_any_phrase is a literal-substring matcher (over _surface's
# lowercased/tokenized rendering, same convention as the frozen
# _argument_disclosure_match's own literal matching) -- real-confirmed too
# strict for retail_055_order_cancel_pending_order#b1::OR02 (matcher.phrases
# == ["confirm"]), the only real check in the current 3-domain corpus using
# this exact literal phrase for this exact requirement (real corpus-wide
# scan confirmed in 145.5.2). Real transcript comparison (pass2 vs pass3
# message 13): pass2's real assistant text was "Do you **confirm** you'd
# like to proceed with this modification? (yes/no)" (contains the literal
# word); pass3's real text was "**Would you like me to** proceed with this
# modification? (yes/no)" -- a semantically-equivalent yes/no confirmation
# offer (the user's real next turn was literally "Yes, I confirm. Please
# proceed.") that never uses the word "confirm" itself.
#
# Same bug CLASS as task_09886b79 (140.3, contains_right_call_argument_
# values' literal matcher not tolerating paraphrase) but a DIFFERENT
# concrete matcher/code path -- and unlike task_09886b79's target function
# (_argument_disclosure_match, in the FROZEN runtime_observation_binding_v1.
# py, forcing an additive second-chance-retry layer in a separate extension
# module), contains_any_phrase's _phrase_match already lives in THIS
# non-frozen extension module (contains_any_phrase is an extension-only
# matcher kind -- the frozen compiler never produces it; see
# _witness_for_left_event's matcher.get("kind") dispatch below), so no
# retry-layer indirection is needed here: the tolerance is added directly,
# but scoped narrowly per-phrase (not a blanket loosening) so it stays
# principled rather than a one-off hack for this single branch's wording.
#
# A real, precise state-machine-adjacent scan (every real assistant message
# across the full 432-transcript corpus -- retail/telecom/airline pass3 --
# containing an explicit "(yes/no)"-style binary-answer marker, the exact
# real convention tau2's own retail/airline confirm-before-state-change
# policy prose uses) found 85 real hits, 69 of which also contain the
# literal word "confirm" (already handled) and 16 of which do not -- a real,
# manual read of the full text of all 16 (not a sample; spans retail_020/
# 022/047/055/055_exchange/060/102 and airline_018/019/023/025/034/035/102/
# 118/127) found EVERY one to be a genuine "Shall/Would/Will I proceed...
# (yes/no)" state-change confirmation offer -- zero were an unrelated
# yes/no question repurposing the same marker. telecom had zero hits either
# way, consistent with 140.3's real finding that telecom uses none of these
# matchers at all.
_YES_NO_CONFIRMATION_OFFER_RE = re.compile(r"\(\s*yes\s*(?:/|or)\s*no\s*\)", re.IGNORECASE)

_PHRASE_SEMANTIC_EQUIVALENTS: dict[str, re.Pattern[str]] = {
    "confirm": _YES_NO_CONFIRMATION_OFFER_RE,
}


def _phrase_match(
    left: Mapping[str, Any], matcher: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    raw_content = str(left.get("content") or "")
    content = _surface(raw_content)
    phrases = [_surface(phrase) for phrase in matcher.get("phrases") or []]
    matched = [phrase for phrase in phrases if phrase and phrase in content]
    if matched:
        return True, {"matched_phrases": matched}
    # Only consulted when the literal match above found nothing -- every
    # check whose literal phrase already matches is completely unaffected by
    # this addition, matching the same "purely additive" discipline 140.3's
    # retry layer used.
    semantic_matched = [
        phrase
        for phrase in phrases
        if phrase in _PHRASE_SEMANTIC_EQUIVALENTS
        and _PHRASE_SEMANTIC_EQUIVALENTS[phrase].search(raw_content)
    ]
    if semantic_matched:
        return True, {
            "matched_phrases": [],
            "matched_semantic_equivalent_phrases": semantic_matched,
        }
    return False, {"matched_phrases": matched}


def _witness_for_left_event(
    left_contract: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    right: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, dict[str, Any] | None]:
    left_filter = _mapping(left_contract.get("event_filter"), "$.left_event.event_filter")
    matcher = left_contract.get("matcher") or {}
    if left_contract.get("binding_kind") == "prerequisite_tool_call_any_of":
        # Extension-only matcher kind (see _prerequisite_lookups_before_operation) --
        # ANY ONE of matcher["tool_names"] counts as a witness, which the frozen
        # _event_matches_filter's exact field_equals cannot express.
        tool_names = set(matcher.get("tool_names") or [])
        candidates = [
            event
            for event in events
            if event.get("event_index", -1) < right.get("event_index", -1)
            and event.get("event_kind") == left_filter.get("event_kind")
            and event.get("tool_name") in tool_names
        ]
        return (candidates[-1], {"matched_tool_name": candidates[-1].get("tool_name")}) if candidates else (None, None)
    candidates = [
        event
        for event in events
        if event.get("event_index", -1) < right.get("event_index", -1)
        and _event_matches_filter(event, left_filter)
    ]
    for left in reversed(candidates):
        if matcher.get("kind") == "contains_right_call_argument_values":
            passed, evidence = _argument_disclosure_match(left, right, matcher, events)
            if not passed:
                continue
            return left, evidence
        if matcher.get("kind") == "contains_any_phrase":
            passed, evidence = _phrase_match(left, matcher)
            if not passed:
                continue
            return left, evidence
        return left, None
    return None, None


def _right_event_content_matches(
    event: Mapping[str, Any], right_contract: Mapping[str, Any]
) -> bool:
    # Unlike _witness_for_left_event's matcher handling (left side), the
    # frozen extraction never applies any content check to the right side at
    # all -- event_filter/scope_constraints were always sufficient there
    # because right_event has always been a tool operation (tool_name equality
    # is exact). airline_077_order#b0's message-literal right_event needs a
    # genuine content check (a real agent message will contain surrounding
    # text around the required literal, not equal it exactly), so this reuses
    # the same "contains_literal" convention the frozen assistant_literal/
    # explicit_assistant_literal paths already use, just applied here instead.
    # A record with no matcher (every existing tool-operation right_event)
    # is unaffected -- this returns True unconditionally, matching the prior
    # behavior exactly.
    matcher = right_contract.get("matcher") or {}
    if not matcher:
        return True
    if matcher.get("kind") == "contains_literal":
        content = str(event.get("content") or "")
        literal = str(matcher.get("literal") or "")
        if matcher.get("case_sensitive", True):
            return literal in content
        return literal.casefold() in content.casefold()
    if matcher.get("kind") == "tool_name_in":
        # section 49's tied-top-score disjunctive right_event override
        # (_resolve_ambiguous_right_event): right_event.event_filter carries
        # no field_equals (any real tool_name from the tied set counts), so
        # the actual disjunction check lives here instead.
        return event.get("tool_name") in set(matcher.get("tool_names") or [])
    return True


def _extract_temporal_observations_extended(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Only for the list-valued left_event shape this module produces (conjunctive
    prerequisites). extract_bound_observations_extended routes here only when
    left_event is a list; a dict is delegated straight to the frozen
    extract_bound_observations, unchanged.
    """
    left_contracts = [_mapping(item, "$.left_event[]") for item in runtime.get("left_event") or []]
    right_contract = _mapping(runtime.get("right_event"), "$.right_event")
    if not left_contracts or right_contract.get("binding_status") != "bound":
        return {"status": "unavailable", "reason": "temporal side is unbound", "matches": []}
    right_filter = _mapping(right_contract.get("event_filter"), "$.right_event.event_filter")

    right_events = []
    missing_bindings = []
    for event in events:
        if not _event_matches_filter(event, right_filter):
            continue
        if not _right_event_content_matches(event, right_contract):
            continue
        scope_ok, missing = _scope_matches(
            event, right_contract.get("scope_constraints") or [], driver_bindings
        )
        missing_bindings.extend(missing)
        if scope_ok:
            right_events.append(event)
    if missing_bindings:
        return {
            "status": "missing_driver_bindings",
            "missing_driver_bindings": sorted(set(missing_bindings)),
            "matches": [],
        }

    matches = []
    unmatched_targets = []
    for right in right_events:
        witnesses = []
        evidences = []
        for left_contract in left_contracts:
            witness, evidence = _witness_for_left_event(left_contract, events, right)
            if witness is None:
                witnesses = None
                break
            witnesses.append(witness)
            evidences.append(evidence)
        if witnesses is None:
            unmatched_targets.append(right.get("event_index"))
            continue
        matches.append(
            {
                "event_index": right.get("event_index"),
                "source_message_index": right.get("source_message_index"),
                "value": {
                    "relation": "precedes",
                    "left_events": [deepcopy(dict(w)) for w in witnesses],
                    "right_event": deepcopy(dict(right)),
                    "matcher_evidence": evidences if any(e is not None for e in evidences) else None,
                },
            }
        )
    return {
        "status": "observed",
        "target_event_count": len(right_events),
        "matched_target_count": len(matches),
        "unmatched_target_event_indices": unmatched_targets,
        "matches": matches,
    }


def extract_bound_observations_extended(
    binding: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if binding.get("binding_status") != "bound":
        return {
            "status": "unavailable",
            "reason": f"binding_status={binding.get('binding_status')}",
            "matches": [],
        }
    runtime = binding.get("runtime_binding") or {}
    if (
        runtime.get("extractor_kind") == "temporal_relation"
        and isinstance(runtime.get("left_event"), list)
    ):
        return _extract_temporal_observations_extended(runtime, events, driver_bindings)
    return extract_bound_observations(binding, events, driver_bindings)


def compile_runtime_observation_bindings_extended_file(
    *,
    accepted_requirements_path,
    expectations_path,
    runtime_observation_profile_path,
    execution_matching_contracts_path,
    tool_catalog_path=None,
    semantic_unit_model_path=None,
    given_conditions_path=None,
    binding_vocabulary_config_path=None,
    output_path,
) -> dict[str, Any]:
    import json as _json
    from pathlib import Path as _Path

    def load(path):
        return _json.loads(_Path(path).read_text(encoding="utf-8"))

    result = compile_runtime_observation_bindings_extended(
        load(accepted_requirements_path),
        load(expectations_path),
        load(runtime_observation_profile_path),
        load(execution_matching_contracts_path),
        load(tool_catalog_path) if tool_catalog_path is not None else None,
        load(semantic_unit_model_path) if semantic_unit_model_path is not None else None,
        load(given_conditions_path) if given_conditions_path is not None else None,
        binding_vocabulary_config=(
            load_runtime_observation_binding_vocabulary_config(binding_vocabulary_config_path)
            if binding_vocabulary_config_path is not None
            else None
        ),
    )
    target = _Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
