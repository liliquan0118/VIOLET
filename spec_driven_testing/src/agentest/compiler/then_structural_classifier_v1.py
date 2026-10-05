"""Classifies a branch's full Then clause into one of the structural types Step4
actually needs to distinguish in order to know which compiler mechanism applies.

Why this exists: Step3's own task contract explicitly defers "runtime evaluator
implementation" to later (see candidates.json packets' task_contract.deferred_decisions)
-- Step3's job is "is this worth checking", not "how do we check it". That split is
reasonable. What was missing is a single, explicit "how do we check it" classification
layer in Step4; without it, each new branch got its own hand-written regex (Gap1/2a/2b),
which is the same brittle keyword-matching pattern the Step8 "classify-then-dispatch"
critique was originally about, just smaller in scale so far.

Deliberately classifies at the BRANCH level using the full `then` sentence, not the
per-requirement `requirement_text` fragment Step3 produces -- real data shows fragments
can drop essential structure (e.g. airline_078_norm#b0v1's requirement_text is "give
subjective recommendations or comments", silently missing the "must not" that the full
`then` carries). See docs/v5_step4_then_structural_classifier_v0_1.md for the full
rule rationale and a real-data validation run across all 155 branches.

This module does not compile anything and does not touch any frozen file or existing
extension module -- it only classifies. Building compiler mechanisms for the
newly-identified, not-yet-covered types is explicitly out of scope for this round.

v0.2 correction (see docs/v5_step4_then_structural_classifier_v0_1.md §7): the v0.1
`cross_event_lookup` bucket made the exact same mistake this whole exercise was trying
to fix -- it used `target_action`'s name as a coarse proxy ("looks like a baggage/
compensation table") instead of checking the branch's real Given database_conditions.
Reading the real data for all 51 "cross_event_lookup" branches showed most of them are
NOT cross-event at all: ~19 have a Given database condition that already fixes the
relevant state as a compile-time constant (member tier, flight status, insurance) --
Step4's real job there is the same tool-call presence/absence check
precondition_gated_call_presence already covers, gated on Step7 building the right
fixture. ~9 more have empty database_conditions but Given text describing a
deliberately-fake identifier or ownership mismatch -- also a Step7 fixture concern, not
a Step4 gap. Only ~11 (baggage/compensation arithmetic, where the formula's constant
comes from a Given-fixed membership tier) and ~6 (payment-profile/seat-inventory/
existing-reservation checks with Given="True", no constant to substitute) are real,
different, not-yet-built mechanisms -- see same_event_arithmetic_with_given_constant
and the narrowed cross_event_lookup below.
"""
from __future__ import annotations

import re
from typing import Any


STRUCTURAL_TYPES = frozenset(
    {
        "single_argument_scalar_constraint",
        "same_event_field_comparison",
        "same_event_arithmetic_with_given_constant",
        "list_element_uniformity",
        "list_sequence_validity",
        "cross_source_state_comparison",
        "cross_event_lookup",
        "precondition_gated_call_presence",
        "temporal_precedence",
        "literal_message_content",
        "turn_structure_constraint",
        "conditional_permissive_obligation",
        "open_ended_semantic_judgment",
        "relative_time_comparison",
        "unclassified",
    }
)

# Each rule: (structural_type, matched_rule_name, regex or callable predicate over the
# normalized `then` text). Rules are tried in order; the first match wins. Order
# matters where patterns could otherwise overlap (e.g. a "must not update ... with a
# different X" cross_source_state_comparison text also contains "update", so more
# specific patterns are listed before more general ones).

_FORMAT_HINTS = re.compile(
    r"\b(?:iata|3-letter airport code|yyyy-mm-dd|airline code followed by|"
    r"format(?:ted)? like|standard format)\b",
    re.IGNORECASE,
)

# General "must provide/set/select/specify X ... (a string/integer/array/either 'a' or
# 'b')" shape -- catches simple single-argument type/enum/shape constraints directly
# from `then` text, without needing Step3's requirement_type (which isn't available at
# pure branch-level classification).
_SCALAR_ARG_SHAPE = re.compile(
    r"\bmust (?:provide|set|select|specify)\b.{0,80}\b(?:a string|an integer|"
    r"either\s+['‘“]|an array of objects|a positive integer)\b",
    re.IGNORECASE,
)

_LIST_UNIFORMITY = re.compile(
    r"\bsame\b.{0,40}\b(?:across all|in the same)\b|"
    r"\ball\b.{0,60}\b(?:same|assigned the same|are booked on the same)\b",
    re.IGNORECASE,
)

_BEFORE_AFTER_ORDERING = re.compile(
    r"\bbefore\b|\bafter\b|\bfirst\b.{0,40}\bthen\b",
    re.IGNORECASE,
)

_FIELD_COMPARISON_TEXT = re.compile(
    r"\bmust not exceed\b|\bmust be different\b|\bmust not be greater than\b|"
    r"\bare different\b|\bis greater than\b|\bis less than\b|\bis fewer than\b",
    re.IGNORECASE,
)

_COUNT_LIMIT = re.compile(
    r"\bmore than (?:one|two|three|four|five|\d+)\b|\bat most\b|\bat least\b",
    re.IGNORECASE,
)

_LIST_SEQUENCE = re.compile(
    r"\bchronologically\b|\bcontiguous\b|\bvalid path\b|\bno missing or extraneous\b",
    re.IGNORECASE,
)

_CROSS_SOURCE_STATE = re.compile(
    r"\bmust not update\b.{0,60}\b(?:with a different|to reduce|to increase|"
    r"with more|with fewer|with a different number)\b",
    re.IGNORECASE,
)

# Genuinely needs a value read from real trajectory/profile/inventory data that no
# Given database condition already fixes as a compile-time constant -- see the v0.2
# correction note above. Deliberately narrower than the v0.1 version: "does not
# exist"/"not owned by" moved to _GIVEN_FIXTURE_IDENTITY_TEXT (those are Step7 fixture
# concerns, not real cross-event correlation).
_CROSS_EVENT_LOOKUP = re.compile(
    r"already (?:be )?present in the user'?s profile|"
    r"already in the user'?s profile|"
    r"sufficient (?:balance|credit)|"
    r"sufficient seat inventory|"
    r"available for booking",
    re.IGNORECASE,
)

# Given text describing a deliberately-fake identifier or an ownership mismatch --
# real data shows these branches' database_conditions are consistently empty (the
# fixture itself is expected to supply a fake ID or a non-owned reservation, per
# docs/v5_step123_generality_audit_v0_1.md §2's "deliberately supply a nonexistent ID"/"not_owner"
# categories). Once the fixture holds that state, Step4's actual check is still just
# tool-call presence/absence -- not a real runtime lookup.
_GIVEN_FIXTURE_IDENTITY_TEXT = re.compile(
    r"does not (?:correspond to|belong to)|not owned by|not own(?:ed)? by|"
    r"do not correspond to|does not exist|not in the airline'?s current network",
    re.IGNORECASE,
)

# Same-event arithmetic where the formula's constant comes from a Given database
# condition fixed at fixture time (e.g. membership tier), not a live lookup -- see the
# v0.2 correction note. "exactly N free checked bag(s)" / "$X times the number of
# passengers" are the two real phrasings found across the 155 branches.
_ARITHMETIC_FORMULA = re.compile(
    r"\bexactly \d+ free checked bags?\b|"
    r"\$\d+ (?:times|per) (?:the number of )?passengers?\b",
    re.IGNORECASE,
)

# Last-resort fallback: any remaining "The agent must(not) <verb> ..." sentence not
# caught by a more specific rule above is treated as a simple single-action
# presence/absence obligation -- the same shape match_cardinality already handles
# mechanically (see docs/v5_step4_then_structural_classifier_v0_1.md).
_TOOL_CALL_PRESENCE = re.compile(r"^the agent must(?:\s+not)?\s+\w+", re.IGNORECASE)

_LITERAL_MESSAGE = re.compile(r"\bsend the message ['‘“]", re.IGNORECASE)

_TURN_STRUCTURE = re.compile(
    r"\bat the same time\b|\bone tool call\b|\bnot send a message.{0,20}and make a tool call\b",
    re.IGNORECASE,
)

_PERMISSIVE_MAY = re.compile(r"^the agent may\b", re.IGNORECASE)

_OPEN_ENDED_SEMANTIC = re.compile(
    r"\bmust not state that\b|"
    r"not provided by the user or available tools|"
    r"subjective recommendations or comments|"
    r"without the user explicitly requesting|"
    r"not listed in the policy",
    re.IGNORECASE,
)

_RELATIVE_TIME = re.compile(
    r"\bbefore the current date\b|\bin the past\b|\byears in the past\b|"
    r"\bbeyond the airline'?s published schedule\b|\bfar in the future\b",
    re.IGNORECASE,
)

_SCALAR_FORMAT_TYPES = {"tool_argument"}

# Facts already compiled today, either by the frozen _compile_semantic_left_event's two
# original branches (action details / explicit confirmation, the 074_order family) or
# by runtime_observation_binding_extension_v1's Gap2a/2b extension (user id; trip type,
# origin, destination; user id + reservation id together). Does NOT include
# "reservation id" alone, "help locate it", or "reason for cancellation" -- those stay
# unresolved (091, 085) per docs/v5_step4_generality_audit_v0_1.md §8.
_KNOWN_GAP2_FACTS = re.compile(
    r"\buser id\b|\btrip type\b|\baction details\b|\bexplicit user confirmation\b",
    re.IGNORECASE,
)


def classify_then_structure(
    kind: str | None,
    then_text: str | None,
    target_action: str | None = None,
    *,
    requirement_type: str | None = None,
    observation_contract: dict[str, Any] | None = None,
    given_text: str | None = None,
) -> dict[str, Any]:
    """Classify one branch's full Then clause (not a requirement_text fragment).

    `requirement_type`/`observation_contract` are optional -- when available (i.e. this
    branch already went through Step3 candidate acceptance) they sharpen a few rules
    that are otherwise ambiguous from `then` text alone (same_event_field_comparison,
    temporal_precedence, literal_message_content). Without them the classifier still
    works off `then`/`kind`/`target_action` alone, which is what makes it possible to
    validate against all 155 real branches instead of only the ones already compiled.

    `given_text` is also optional, from assembly.json's given_requirements.text (not
    carried by accepted_requirement_set's branch_context, which only keeps the same raw
    Given text -- so this is not extra data, just plumbed through explicitly since the
    classifier is what consumes it). See the v0.2 correction note in the module
    docstring for why this matters: whether a branch's Given already fixes the relevant
    fact as a compile-time constant is what actually determines the mechanism, not the
    target_action name -- but that requires reading assembly.json's
    given_requirements.database_conditions, which is a bigger plumbing change than this
    classifier alone should make. What's checked here instead is a narrower, sufficient
    proxy: Given text describing a deliberately-fake identifier or ownership mismatch,
    which is detectable from the text alone.
    """
    text = (then_text or "").strip()
    target = (target_action or "").strip()
    observation_contract = observation_contract or {}
    given = (given_text or "").strip()

    if _GIVEN_FIXTURE_IDENTITY_TEXT.search(given):
        return _result("precondition_gated_call_presence", "given_fixture_identity_text")

    if _ARITHMETIC_FORMULA.search(text) and not _OPEN_ENDED_SEMANTIC.search(text):
        # Guards against e.g. airline_030_arg#b0: "must not state that travel
        # insurance does not cost $30 per passenger" superficially matches the "$N per
        # passenger" phrasing but is a claim-grounding check on assistant message
        # content, not a same-event tool-argument arithmetic formula.
        return _result("same_event_arithmetic_with_given_constant", "arithmetic_formula_phrase")

    if requirement_type == "tool_argument_constraint":
        constraint_text = str(observation_contract.get("constraint_text") or text)
        if _FIELD_COMPARISON_TEXT.search(constraint_text) or re.search(
            r"\bmust be different\b", constraint_text, re.IGNORECASE
        ):
            return _result("same_event_field_comparison", "known_gap1_grammar", already_covered_by="oracle_evaluator_contract_extension_v1")
        if _LIST_UNIFORMITY.search(constraint_text):
            return _result("list_element_uniformity", "list_uniformity_phrase")
        if _LIST_SEQUENCE.search(constraint_text):
            return _result("list_sequence_validity", "sequence_phrase")
        if _CROSS_EVENT_LOOKUP.search(constraint_text):
            return _result("cross_event_lookup", "cross_event_lookup_phrase")

    if requirement_type == "temporal_relation":
        already = "runtime_observation_binding_extension_v1" if _KNOWN_GAP2_FACTS.search(text) else None
        return _result("temporal_precedence", "requirement_type_temporal_relation", already_covered_by=already)

    if requirement_type == "assistant_literal":
        return _result("literal_message_content", "requirement_type_assistant_literal")

    # Rules below work from `then`/`kind`/`target_action` alone (no Step3 acceptance
    # data required), which is what makes branch-level validation across all 155 real
    # branches possible instead of only the ones already compiled.

    if _LIST_SEQUENCE.search(text):
        return _result("list_sequence_validity", "sequence_phrase")

    if _LIST_UNIFORMITY.search(text):
        return _result("list_element_uniformity", "list_uniformity_phrase")

    if _FIELD_COMPARISON_TEXT.search(text):
        return _result("same_event_field_comparison", "field_comparison_phrase")

    if _COUNT_LIMIT.search(text) and not _CROSS_SOURCE_STATE.search(text):
        return _result("single_argument_scalar_constraint", "count_limit_phrase")

    if _CROSS_SOURCE_STATE.search(text):
        return _result("cross_source_state_comparison", "update_with_different_phrase")

    if _CROSS_EVENT_LOOKUP.search(text):
        return _result("cross_event_lookup", "cross_event_lookup_phrase")

    if _LITERAL_MESSAGE.search(text):
        return _result("literal_message_content", "send_the_message_phrase")

    if _TURN_STRUCTURE.search(text):
        return _result("turn_structure_constraint", "turn_shape_phrase")

    if _OPEN_ENDED_SEMANTIC.search(text):
        return _result("open_ended_semantic_judgment", "open_ended_phrase")

    if _RELATIVE_TIME.search(text):
        return _result("relative_time_comparison", "relative_time_phrase")

    if _SCALAR_ARG_SHAPE.search(text) or _FORMAT_HINTS.search(text):
        return _result("single_argument_scalar_constraint", "scalar_arg_shape_phrase")

    if _BEFORE_AFTER_ORDERING.search(text):
        already = "runtime_observation_binding_extension_v1" if _KNOWN_GAP2_FACTS.search(text) else None
        return _result("temporal_precedence", "before_after_phrase", already_covered_by=already)

    if _PERMISSIVE_MAY.match(text):
        return _result("conditional_permissive_obligation", "leading_may_phrase")

    if kind == "STATE" and _TOOL_CALL_PRESENCE.match(text):
        return _result("precondition_gated_call_presence", "state_call_presence_phrase")

    if _TOOL_CALL_PRESENCE.match(text):
        return _result("precondition_gated_call_presence", "call_presence_phrase")

    return _result("unclassified", "no_rule_matched")


def _result(
    structural_type: str,
    matched_rule: str,
    *,
    already_covered_by: str | None = None,
    extracted: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert structural_type in STRUCTURAL_TYPES
    return {
        "structural_type": structural_type,
        "matched_rule": matched_rule,
        "already_covered_by": already_covered_by,
        "extracted": extracted or {},
    }
