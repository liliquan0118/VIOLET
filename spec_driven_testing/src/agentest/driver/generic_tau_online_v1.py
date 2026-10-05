"""Online τ-bench runner for GenericBoundDriverPlan v0.1."""

from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..compiler.artifacts import content_sha256
# evaluate_oracle_contract_extended, not the frozen evaluate_oracle_contract: the
# frozen oracle_evaluator_contract_v1.py's bytes are pinned by
# scripts/prepare_v5_step4_v0_1.py's legacy-compiler-drift check (a real regression
# found and reverted in section 63 -- editing that file directly to add
# matches_regex support broke 5 real Step4 reproducibility tests). The _extended
# variant already supports matches_regex (built for Step4's own extension grammar)
# and falls back to the frozen evaluator unchanged for every predicate_kind it
# doesn't itself own -- see oracle_evaluator_contract_extension_v1.py's own
# module docstring for why the frozen file must never be edited in place.
from ..compiler.oracle_evaluator_contract_extension_v1 import evaluate_oracle_contract_extended as evaluate_oracle_contract
from ..compiler.runtime_observation_binding_v1 import normalize_tau_execution
from .generic_tau_airline_v1 import load_default_tau_environment, validate_generic_bound_driver_plan_set


def _real_semantic_judge(*, model: str, criterion: str, target_value: Any, transcript_excerpt: Any,
                         llm_args: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Real LLM-backed implementation of the semantic_judge interface
    oracle_evaluator_contract_extension_v1.evaluate_predicate_extended's own
    semantic_transcript_judgment predicate expects (criterion/target_value/
    transcript_excerpt in, {"passed": bool, "reason": str} out) -- "the one real
    place... where passed cannot be computed by pure, local Python logic" per that
    module's own docstring (section 65). Uses litellm directly (the same library
    tau2's own create_llm_agent uses internally) against the SAME real provider
    already configured for the live conversation. Mechanically parses a real,
    structured JSON response -- never trusts free text beyond that; a malformed
    response is an honest fail, not a silently assumed pass."""
    import litellm

    litellm_model = model if "/" in model else f"openai/{model}"
    prompt = (
        "You are a strict, mechanical judge evaluating one real criterion against "
        "a real conversation transcript excerpt. Base your answer only on what the "
        "transcript actually shows -- never assume or invent facts not present.\n\n"
        # section 107: real, confirmed false-fail risk (docs/
        # agentcoveragetesting_reuse_log.md section 106/107, telecom_063's
        # real online rerun) -- criterion text is a literal, unparaphrased
        # span of the domain's own policy prose (this project's own
        # no-fabrication discipline), which sometimes names a function- or
        # tool-style identifier like "toggle_roaming()" as shorthand for a
        # real-world action, not a literal phrase the assistant must utter.
        # A judge told to treat the criterion "verbatim" (see below) can
        # otherwise misread this as requiring the literal string, rejecting
        # a real, correct, human-phrased guidance ("turn on Data Roaming in
        # your settings") for not saying the API name out loud -- confirmed
        # via a real, content-linked split verdict on the SAME criterion for
        # functionally identical agent behavior across two real reruns.
        "If the criterion or rule text names a function- or tool-style "
        "identifier (e.g. `toggle_roaming()`), that name identifies the "
        "real-world action or outcome the rule is about -- it is not a "
        "literal phrase anyone in the conversation must utter. The rule is "
        "satisfied by conveying that action/outcome in ordinary human "
        "language, or by taking the equivalent real action, even if the "
        "exact function name is never said verbatim.\n\n"
        f"Criterion:\n{criterion}\n\n"
        f"Value being judged:\n{json.dumps(target_value, ensure_ascii=False, default=str)}\n\n"
        f"Transcript excerpt:\n{json.dumps(transcript_excerpt, ensure_ascii=False, default=str)}\n\n"
        'Respond with ONLY a single JSON object, no other text: '
        '{"passed": true or false, "reason": "one concise sentence explaining why"}'
    )
    response = litellm.completion(
        model=litellm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        num_retries=0,
        **dict(llm_args or {}),
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        return {"passed": False, "reason": f"judge_response_not_valid_json: {content[:200]!r}"}
    if not isinstance(parsed, Mapping) or "passed" not in parsed:
        return {"passed": False, "reason": "judge_response_missing_passed_field"}
    result = {"passed": bool(parsed["passed"]), "reason": str(parsed.get("reason", ""))}
    # docs/agentcoveragetesting_reuse_log.md section 141 (task_b852c7cb):
    # _evaluate_semantic_event_matcher_deferred's own criterion now asks the
    # judge for one extra field (never_had_opportunity) alongside passed/
    # reason, in the SAME call/response, to detect premature-termination
    # false negatives on the semantic_judge lane. Every OTHER caller of this
    # function (the two semantic_transcript_judgment call sites in
    # oracle_evaluator_contract_extension_v1.py) never asks for this field in
    # their own criterion text, so the model never returns it for them and
    # this loop is a no-op there -- passing through whatever extra keys the
    # model actually returned (never inventing or requiring any) keeps every
    # existing caller that only reads passed/reason byte-identical.
    for key, value in parsed.items():
        if key not in result:
            result[key] = value
    return result


GENERIC_ONLINE_VERSION = "tau-generic-online-runtime-adapter/v0.1"
GENERIC_EXECUTION_VERSION = "agentspectesting.generic-tau-online-execution/v0.1"


class GenericTauOnlineError(ValueError):
    """Raised when a generic bound test cannot be executed safely."""


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return deepcopy(dict(message))
    try:
        return message.model_dump(exclude_none=True, mode="json")
    except Exception as exc:
        raise GenericTauOnlineError(f"cannot serialize tau message: {exc}") from exc


def _messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [_message_dict(message) for message in messages]


# docs/agentcoveragetesting_reuse_log.md section 162 (task_a2867400): real
# tau2 (tau2/orchestrator/orchestrator.py's Orchestrator._check_
# communication_error) enforces a real half-duplex protocol -- a message
# must be EITHER pure text OR a pure tool call, never both -- and raises
# AgentError/UserError otherwise. Real tau2 LLM-driven message generation
# (both tau2/agent/llm_agent.py's create_llm_agent and tau2/user/
# user_simulator.py's UserSimulator._generate_next_message) builds its
# message straight from whatever the underlying model returns, with no
# mutual-exclusion enforcement of its own, so a real model that emits a
# short acknowledgment alongside its tool_calls in the same turn (a common,
# harmless real LLM behavior, not a genuine protocol violation) trips this
# check and ends the conversation via AGENT_ERROR/USER_ERROR. Extracted as
# its own pure, directly testable function (previously inlined only for the
# AGENT role, inside run_generic_tau_online_plan's local BudgetedOrchestrator
# class) so this same, already-established tolerance can be reused for the
# USER role too -- a real, live telecom_065_order#b0 run (outputs/
# generic_tau_baselines_v0_1_telecom_full_verification_pass6/results/
# telecom_065_order__b0.json) confirmed deepseek-v4-flash, playing the real
# tau2 UserSimulator for this branch (section 106's _TELECOM_USER_SIMULATOR_
# BRANCHES -- the only branches whose USER role can ever carry real
# tool_calls at all; the scripted GenericDeterministicTauUser never emits
# tool_calls, so this is a real no-op for every other branch/domain), hits
# the exact same mixed-message shape on the USER side: "Sure, let me run
# those checks now." together with real run_speed_test()/check_network_
# status() tool_calls, tripping termination_reason=user_error 4 messages
# before any real diagnostic fact could be established. Safe to tolerate:
# tau2's own message routing (orchestrator.py: "if not last_message.
# is_tool_call(): to_role = AGENT else: to_role = ENV") only ever inspects
# is_tool_call(), never text content, so a real mixed message still routes
# correctly once this check stops rejecting it.
def _tolerates_mixed_content_and_tool_calls(role_name: str, value: Mapping[str, Any]) -> bool:
    return (
        role_name in ("agent", "user")
        and bool(value.get("tool_calls"))
        and isinstance(value.get("content"), str)
        and bool(value["content"].strip())
    )


# docs/agentcoveragetesting_reuse_log.md section 95 (sub-pattern B): markers
# split into two tiers. The strong tier is unambiguous -- every real corpus
# instance found (a full scan of 1075 real result files, 737 real marker
# hits) is a genuine operation yes/no confirmation, so it's matched
# unconditionally. The weak tier's substrings ("confirm"-based) can ALSO
# appear in a genuine identity-verification/fact request ("Could you please
# confirm your user ID?", "This is required to confirm your identity") --
# 9 real, confirmed false positives found in real transcripts, 2 of which
# (airline_033_arg#b0, airline_120_state#b0) were directly observed
# derailing a real conversation: the driver's canned "Yes, I confirm. Please
# proceed." doesn't answer the agent's actual question, so the agent asks
# again in words the marker list no longer matches, and the driver -- with
# nothing left to say -- falls through to ###STOP### before the agent ever
# gets the id it asked for. A weak-tier hit is excluded only when no strong
# marker is also present in the same message (a real message can legitimately
# carry both, e.g. "Shall I go ahead and cancel? Please confirm the order
# ID and the reason." -- must still match).
_STRONG_CONFIRMATION_MARKERS = (
    "would you like to proceed",
    "do you want me to proceed",
    "shall i proceed",
    # Real phrasings found missing this marker set while reading the real
    # telecom smoke-run transcripts (section 63): "Would you like me to
    # send you a payment request..." was a real, genuine confirmation
    # request the original marker list didn't recognize. Only the
    # interrogative "would you like me to" is kept -- "you'd like me to"/
    # "you would like me to" (no leading "would") turned out to be a real
    # false positive too, found in a retail smoke run: "the mathematical
    # expression you'd like me to evaluate" is a relative clause asking
    # for MISSING input, not a confirmation request at all.
    "would you like me to",
    # Real phrasing found on a real airline batch run (docs/
    # agentcoveragetesting_reuse_log.md section 84): "Shall I go ahead and
    # make this change? (yes/no)" -- a genuine confirmation request that
    # slipped past every existing marker ("shall i proceed" doesn't match
    # "shall i go ahead").
    "shall i go ahead",
)
_WEAK_CONFIRMATION_MARKERS = (
    "please confirm",
    "confirm that",
    "confirm you",
    "just to confirm",
    "to confirm:",
    # Real phrasing found in a later real telecom smoke-run round (section
    # 63): "Just to make sure we're on the same page: ..." before resuming
    # a suspended line -- a genuine confirmation request, no "confirm"
    # substring at all. Real corpus scan (section 95) found no false
    # positive for this one, but it's kept in the weak tier (checked against
    # the same exclusion) out of caution since it's a looser phrase match
    # than the strong tier's unambiguous ones.
    "on the same page",
)
# Excludes a weak-tier hit that is really asking the user to STATE/PROVIDE an
# identity fact ("confirm your user ID"/"confirm your identity"), or is the
# assistant declaring it already confirmed something about the user's
# account ("I can confirm your details on file") -- neither is a yes/no
# operation-confirmation request.
#
# docs/agentcoveragetesting_reuse_log.md section 137.5.2 (task_11d608cc): the
# first alternative above real, reproducibly over-matched a genuine
# OPERATION confirmation on a real retail_077_norm#b0 transcript message:
# "...Please confirm the order ID and reason, and I'll proceed. (Reply
# "yes" to confirm.)" -- confirming the act of cancelling THIS order, not a
# request to verify identity, but "confirm the order id" still matched the
# bare "confirm (your|the) <noun> id" shape, wrongly excluding it. Real,
# direct call confirmed this deterministically returns False pre-fix. The
# real, distinguishing signal between this and a genuine identity-
# verification ask (e.g. "please confirm your order ID to verify your
# identity", or the existing "Could you please confirm the user ID on
# file?" regression test below, which must stay excluded) is whether the
# SAME sentence also says the operation is about to proceed once the id is
# given back -- a real identity-verification request has no "proceed"
# anywhere nearby (it's asking the user to STATE a fact before anything can
# happen at all), while a real operation-confirmation request restates the
# id as part of "confirm these details, then I'll proceed". Scoped to a
# negative lookahead requiring no "proceed" anywhere in the rest of the
# CURRENT sentence (bounded by the next ./!/? or newline) after the id
# phrase -- narrow enough that it does not touch the plain identity-
# verification case (no "proceed" in that sentence at all) or the
# "confirm your identity" / "can/could confirm your" alternatives below
# (untouched, not implicated in this real bug).
_IDENTITY_REQUEST_EXCLUSION_RE = re.compile(
    r"confirm\w*\s+(your|the)\s+(correct\s+)?"
    r"(user|customer|account|reservation|order|line|member)[\s_-]*id\b"
    r"(?![^.!?\n]*\bproceed\b)"
    r"|confirm\w*\s+your\s+identity\b"
    r"|\b(can|could|i've|i have|we've|we have)\s+confirm(ed)?\s+your\b",
    re.IGNORECASE,
)

# docs/agentcoveragetesting_reuse_log.md section 152.1 (task_d31870ce,
# found by section 150.4.2's real online verification of
# retail_055_order_modify_user_address#b0 pass4): every fixed-gap heuristic
# in this module (the reply-verb .. "yes" gap in _REPLY_YES_CONFIRMATION_RE
# below, the "confirm (your|the) <noun> id" gap in
# _IDENTITY_REQUEST_EXCLUSION_RE above, the "reply with" gap in
# _DISAMBIGUATION_OPTION_REQUEST_RE below) assumes its two anchor words sit
# adjacent to each other (only literal whitespace, or an explicitly-listed
# word, in between). A real assistant message -- 'Please reply **"yes"** to
# confirm and I'll proceed.' (verbatim, retail_055_order_modify_user_
# address#b0 pass4 transcript, message 16) -- inserts a markdown
# bold-emphasis marker (**) directly between the reply-verb and the quoted
# "yes", which none of those fixed gaps tolerate: _REPLY_YES_CONFIRMATION_RE
# real-confirmed failed to match this real text (and matched instantly once
# the ** was stripped), and this was the LAST of the branch's confirmation/
# nudge budget, so the miss sent the scripted driver straight to
# ###STOP### with the real target tool call never observed.
#
# Fix: a single, reusable normalization applied once, at the top of every
# function in this module that runs a fixed-gap/substring heuristic over
# assistant message text, rather than patching each regex's gap
# individually -- chosen over the per-regex approach because (a) this
# exact defect shape (a markdown emphasis marker landing in an
# assumed-adjacent gap) is not unique to _REPLY_YES_CONFIRMATION_RE, so a
# single normalization closes every current AND future instance of it at
# the source instead of requiring a bespoke gap-widening patch to be
# remembered on every future heuristic added to this module, and (b) it is
# provably a strict widening of what already matches, never a narrowing:
# _strip_markdown_emphasis only ever REMOVES `*`/`_` runs that are not part
# of a snake_case-style identifier (see the doc string below for the exact
# rule), so any text that matched before still matches after -- it can only
# ever make MORE gap-interrupted real text match, never make previously-
# matching text stop matching, except where the interrupted markdown
# marker was itself sitting *inside* an unrelated identifier (never the
# case for these prose heuristics). A real, direct function-call corpus
# scan (not a text scan) against every real assistant message across every
# available real online-run result transcript in all 3 domains (1,897
# real result files, 12,229 real assistant messages, 489 distinct
# branches) confirmed this precisely: 18 real messages across 12 real
# branches (airline_008_arg#b0, airline_043_arg#b0, airline_120_state#b0,
# retail_020_arg#b0v0/v1/v4, retail_027_arg#b0, retail_055_order_modify_
# user_address#b0 (this ticket's own target), retail_055_order_return_
# delivered_order_items#b0, retail_060_order_modify_pending_order_items#
# b0v1, retail_069_state#b0, telecom_116_state#b0) really change verdict
# once markdown emphasis is stripped from the gap -- 17 of the 18 are real
# false negatives (a genuine "reply ... yes"/weak-marker confirmation
# request the markdown gap was hiding) that the fix newly, correctly
# recognizes; the 1 remaining (airline_120_state#b0 message 8, "...could
# you please confirm your **user id**? I need it to verify the
# reservation.") is a real, CORRECT flip the other direction -- pre-fix,
# the ** between "your" and "user id" broke _IDENTITY_REQUEST_EXCLUSION_RE'
# s own "confirm (your|the) <noun> id" match, so the weak marker
# "please confirm" fired unexcluded and the message was wrongly treated as
# an operation-confirmation request; post-fix, the exclusion regex sees the
# now-adjacent "confirm your user id" and correctly excludes it as a
# genuine identity-verification ask (the same real class this exclusion
# regex already exists to filter, per the "Could you please confirm the
# user ID on file?" regression test in
# tests/test_generic_tau_online_v1_confirmation_markers.py) -- not a
# regression, a second real bug this same normalization incidentally fixes
# for free. Zero of the 18 real hits were spurious (every one, read in
# full, is a genuine markdown-interrupted instance of an existing,
# already-approved heuristic -- this normalization recognizes no new
# heuristic shape and introduces no new false-positive risk beyond what
# each pre-existing regex already accepted).
_MARKDOWN_EMPHASIS_RE = re.compile(r"\*+|(?<!\w)_+|_+(?!\w)")


def _strip_markdown_emphasis(text: str) -> str:
    """Removes markdown emphasis delimiters (`**bold**`, `*italic*`,
    `__bold__`, `_italic_`) from real LLM-generated assistant prose before
    running any fixed-gap/substring text heuristic in this module, so a
    marker landing between two words a heuristic assumes are adjacent (e.g.
    'reply **"yes"**') does not defeat the match. Asterisk runs are always
    stripped (assistant prose has no other legitimate use for `*`).
    Underscore runs are stripped only when at least one side is NOT a word
    character -- i.e. only when the underscore(s) sit at a word/prose
    boundary, matching markdown's own emphasis-delimiter flanking rule --
    so a real identifier like `toggle_roaming` (underscore flanked by word
    characters on both sides) is left untouched, while `_yes_`/`reply
    _with_` (underscore flanked by whitespace/punctuation on at least one
    side) is correctly stripped. This is a pure widening: it can only turn
    a previously-non-matching gap into a matching one, never the reverse,
    for every heuristic in this module (verified by the real corpus scan
    in the comment above)."""
    return _MARKDOWN_EMPHASIS_RE.sub("", text)


# docs/agentcoveragetesting_reuse_log.md section 138.5 (task_592feb18): real,
# natural confirmation-request phrasings like 'Please reply "yes" to confirm
# and I'll proceed with the booking.' and 'The only thing I need from you is
# your explicit confirmation to proceed. Just reply "yes" and I'll complete
# the booking.' (both real assistant messages from the real
# airline_040_arg#b0 transcript) matched NEITHER tier above, even though each
# message contains "confirm"-related words ("confirm"/"confirmation") and
# "proceed" *somewhere* in its text -- neither message contains any of the
# WEAK tier's exact fixed substrings ("please confirm"/"confirm that"/
# "confirm you"/"just to confirm"/"to confirm:"), and neither contains any
# STRONG tier phrase either. A wide-enough "confirm" near "yes"/"proceed"
# proximity heuristic might happen to catch these two specific messages, but
# that approach has no principled window size -- too narrow and it's just
# another fragile phrase variant, too wide and it risks matching two
# unrelated sentences that each merely happen to contain one of the words.
# What both real messages actually share, and what makes them unambiguous
# confirmation requests regardless of what other words surround them, is a
# recognizable GRAMMATICAL PATTERN instead: the assistant is telling the user
# the literal expected answer is the word "yes" (<reply-verb> ... "yes").
# That signal is precise independent of proximity/window tuning, and (unlike
# a bare "yes"-anywhere check, which would false-positive on ordinary
# assistant prose like "Yes, I can help with that -- I'll reply once I
# check.") only fires when the reply-verb and "yes" are adjacent. Treated as
# STRONG (bypasses
# _IDENTITY_REQUEST_EXCLUSION_RE entirely, like the rest of the strong tier):
# unlike "confirm your account ID" (a request to STATE a fact, which
# _IDENTITY_REQUEST_EXCLUSION_RE exists to exclude), there is no reading of
# "reply with the word yes" that means anything other than a yes/no
# confirmation gesture, even when it also mentions an id/identity in the same
# breath (e.g. "reply 'yes' to confirm your identity" is still, unambiguously,
# a request for a literal "yes"). Requires the reply-verb to come immediately
# before "yes" (only "back "/"with " allowed in between) so it does not match
# unrelated later use of "yes" in the same message (e.g. "I'll reply once I
# hear back, yes, that's the plan.").
_REPLY_YES_CONFIRMATION_RE = re.compile(
    r"\b(?:reply|respond|answer)\w*\s+(?:back\s+)?(?:with\s+)?"
    r"[\"'‘’“”]?yes[\"'‘’“”]?\b",
    re.IGNORECASE,
)


def _assistant_has_confirmation_request(message: Mapping[str, Any]) -> bool:
    text = _strip_markdown_emphasis(str(message.get("content") or "")).casefold()
    if any(marker in text for marker in _STRONG_CONFIRMATION_MARKERS):
        return True
    if _REPLY_YES_CONFIRMATION_RE.search(text):
        return True
    if any(marker in text for marker in _WEAK_CONFIRMATION_MARKERS):
        return not _IDENTITY_REQUEST_EXCLUSION_RE.search(text)
    return False


# docs/agentcoveragetesting_reuse_log.md section 137.5.4 (task_b81f1f07): the
# generate_next_message disambiguation fallback below used to gate on a
# literal `.rstrip().endswith("?")` check. Real, direct replay of the real
# retail_055_order_modify_user_address#b0 transcript's own real
# transport_action_log (confirmation -> nudge -> [this decision] -> stop)
# against the real GenericDeterministicTauUser state machine pinpointed the
# EXACT real assistant message this decision point actually evaluates for
# that branch's real stop -- not message 14 (which is evaluated one step
# earlier and correctly triggers the nudge branch instead), but message 16:
# a real, substantive two-part clarification (an address a/b/c option list
# plus a Yes/No item-modification choice, "To make it easy, please reply
# with your choices: ...") that contains NO "?" character anywhere at all
# (it is phrased as an imperative option list, not a literal question), so
# even a same-sentence "?" search cannot recognize it. What it shares with a
# real "?" question, at this exact point in the state machine (confirmation/
# facts/nudge already exhausted), is that it explicitly tells the user a
# reply/choice is expected -- the same "reply with <answer>" grammatical
# signal _REPLY_YES_CONFIRMATION_RE already recognizes for the literal "yes"
# case, generalized here to ANY expected reply content. A real, precise
# corpus-wide check (replaying the real driver state machine against every
# real assistant message that actually reaches this exact decision point
# across 432 real cross-domain transcripts -- not just scanning raw message
# text) found only 2 real messages anywhere in that set matching "reply
# with": this real retail branch, and one genuinely analogous real telecom
# branch (telecom_118_state#b0v0, "...reply with the number/operation you
# want, and I'll take care of it.") with the identical real gap (a numbered
# option list + an explicit "reply with" request) -- confirming this is
# narrow and precise, not a broad, budget-costly false-positive risk.
_DISAMBIGUATION_OPTION_REQUEST_RE = re.compile(r"\breply\s+with\b", re.IGNORECASE)


# docs/agentcoveragetesting_reuse_log.md section 145.4.4 (task_3e77531b): a
# real, precise state-machine replay (the SAME technique 140.4 used --
# GenericDeterministicTauUser replayed against each real transcript's own
# real transport_action_log as ground truth, not a raw text scan) against the
# CURRENT full 432-transcript corpus (retail/telecom/airline, same pass3
# corpus sections 143/144/145 already replayed) found 422 real assistant
# messages that this exact gated elif actually evaluates (confirmation/
# facts/nudge already dispositioned by the three elif branches above this
# one). Of the 113 where BOTH real heuristics above (endswith("?"), "reply
# with") returned False, 55 contain a real "?" character SOMEWHERE in the
# message but not as the final character -- retail_055_order_modify_user_
# address#b0's real message 10 ("...Could you please tell me exactly which
# action(s) you'd like to take? ... Let me know what you'd like to do!") is
# one of them: a real embedded question followed by a non-question trailer
# sentence. A real, manual read of the full text of all 55 (not a sample)
# found every single one to be a genuine question the user is expected to
# answer -- zero were rhetorical, already-resolved, or otherwise spurious.
# This is not a coincidence of this particular sample: by the time this
# elif is reached, confirmation/facts/nudge have already been dispositioned
# by the three earlier elif branches (each gated on its own live-condition,
# checked before this one), so any "?" that survives to this decision point
# is never a stray confirmation ask -- confirmation asks are already
# consumed upstream by _assistant_has_confirmation_request's own dedicated
# gate. The remaining 58 of the 113 (no "?" anywhere) are legitimate
# non-disambiguation stops (transfer-to-human-agent notices, "already
# cancelled, nothing to do" completions, etc.) and correctly stay
# unmatched -- broadening to "?" ANYWHERE (a strict superset of the old
# endswith("?") check, which only recognized a same-sentence trailing "?")
# does not touch them, since none contain a "?" at all.
#
# docs/agentcoveragetesting_reuse_log.md section 160 (task_87a928f3, found
# by section 159.7's real online verification of airline_085_order#b0 pass6
# -- the branch's FIRST-EVER real "incomplete" verdict in this project's
# airline history): the real, final assistant message that branch produced
# after its one-shot nudge was already spent -- 'To proceed, I just need you
# to confirm the **reason for cancellation** (change of plan, airline
# cancelled flight, or other reasons), and then I'll confirm the details
# with you before finalizing.' -- matches NONE of the three existing
# fallbacks: no strong/weak confirmation marker (it isn't a yes/no ask), no
# "?" anywhere (an imperative sentence, not a question), and no "reply
# with". This is a FOURTH real, distinct phrasing gap in this same
# heuristic family (after task_b81f1f07's "reply with" gap, task_3e77531b's
# "?" not at the end gap, and task_d31870ce's markdown-interrupted-gap
# fix) -- an imperative request for ONE specific piece of missing
# information the assistant needs from the user before it can continue the
# SAME operation already in progress, grammatically distinct from both a
# literal question and an explicit "reply with <X>" instruction, but every
# bit as much something the scripted driver must answer (or, when the exact
# answer isn't literally re-derivable, at least acknowledge) rather than
# silently give up on.
#
# A real, precise, cross-domain state-machine replay (the SAME technique
# 140.4/145.4.4/152.1 all used -- GenericDeterministicTauUser replayed
# against every real transcript's own real message sequence, not a raw text
# scan) against every real online-run result file available across all 3
# domains (3,257 real result files under every generic_tau_baselines_v0_1_*
# output directory, 2,882 real branches successfully replayed, 10,204 real
# assistant-message decision points) found 191 real, distinct (branch_id,
# message) pairs that reach this exact decision point (confirmation/facts/
# nudge already dispositioned, disambiguation not yet sent) and resolve to
# a real stop with NEITHER existing heuristic above matching. A real,
# precise sub-scan of those 191 for the shared grammatical shape every one
# of these real messages turned out to have in common -- a first-person
# present-tense "I need ..." statement naming what the assistant still
# needs from the user, as distinct from an offer, a completed-transaction
# notice, or a policy-blocked refusal -- found 24 real matches, spanning
# ALL 3 domains (airline_085_order#b0 (this ticket's own target),
# airline_012_arg#b0, airline_018_arg#b0, airline_089_norm#b0v0/v1,
# airline_095_norm#b0, airline_099_norm#b0, airline_113_state#b1 (x2),
# airline_119_state#b0, airline_124_state#b0, airline_129_state#b0 (x2),
# retail_023_arg#e2, telecom_018_arg#b0, telecom_019_arg#b0,
# telecom_048_state#b0, telecom_118_state#b0v1/b0v5(x2)/b2v0/b2v4(x2)).
# A real, manual read of the full text of all 24 (not a sample) found 23
# genuine matches (a real, specific missing-input ask blocking the same
# operation the conversation is already pursuing) and exactly 1 real false
# positive: telecom_117_state#b1's "...disabling roaming isn't an action I
# need to take because roaming is already disabled..." -- a real completion
# notice whose "I need to take" describes an action the ASSISTANT itself
# does not need to perform, not something it needs FROM the user. The
# regex below is scoped with a negative lookahead specifically excluding
# that one real false-positive shape ("need to take"); re-run against the
# same 191-message corpus post-exclusion, the false positive is the ONLY
# one of the 24 that drops out (23/23 real remaining matches confirmed
# genuine), and zero of the real 167 non-matching messages newly match.
#
# Narrow-vs-generalize decision (this is the 4th real instance of the same
# underlying gap in this heuristic family): chose a single, principled
# GRAMMATICAL PATTERN match ("I need ...", the same style of reasoning
# task_592feb18's _REPLY_YES_CONFIRMATION_RE and task_b81f1f07's
# _DISAMBIGUATION_OPTION_REQUEST_RE already use -- a recognizable sentence
# construction, not a proximity/keyword-window heuristic) over either (a) a
# single-branch-specific literal-phrase patch (would not have covered the
# other 22 real, independently-found instances across all 3 domains -- a
# narrower patch than the real evidence justifies) or (b) a fully general
# semantic "is this asking for anything" classifier (real corpus evidence
# does not support this: reading all 191 real candidates found the vast
# majority -- 167 of 191 -- are legitimate non-actionable stops with no
# shared textual signal at all, e.g. "this order is already cancelled, let
# me know if there's anything else", and a semantic classifier would risk
# matching many of those too, reintroducing the "385/3017 over-broad
# candidate" mistake task_b81f1f07's own real methodology note above
# already once rejected in favor of exact replay). The "I need" shape sits
# precisely between those two extremes and is the one the real evidence
# actually supports: precise (23/23 verified genuine on the full real
# candidate population), general enough to close all 4 real instances of
# this bug class found so far without requiring a 5th ticket for the next
# branch that happens to phrase the same request slightly differently.
_MISSING_INPUT_REQUEST_RE = re.compile(
    r"\bI(?:'d|\s+would)?\s+(?:still\s+|just\s+|genuinely\s+|really\s+|do\s+)?need\b"
    r"(?!\s+to\s+take\b)",
    re.IGNORECASE,
)


def _assistant_message_needs_disambiguation_response(content: Any) -> bool:
    # docs/agentcoveragetesting_reuse_log.md section 152.1 (task_d31870ce):
    # normalized through the same _strip_markdown_emphasis used by
    # _assistant_has_confirmation_request above, so a markdown marker
    # landing between "reply" and "with" (e.g. "reply **with**") cannot
    # defeat _DISAMBIGUATION_OPTION_REQUEST_RE's own fixed-gap match any
    # more than it can defeat _REPLY_YES_CONFIRMATION_RE's. The real
    # corpus-wide scan for this ticket (see the comment above
    # _MARKDOWN_EMPHASIS_RE) found zero real messages across all 3 domains
    # where this particular regex's own verdict actually flips (no real
    # transcript in the corpus happens to markdown-interrupt "reply with"
    # specifically), but the gap is real and structurally identical, so it
    # is closed here too rather than left latent for a future transcript to
    # trip over -- "?" in stripped is unaffected either way ("?" is never a
    # markdown emphasis character).
    stripped = _strip_markdown_emphasis(str(content or "")).rstrip()
    if not stripped:
        return False
    if "?" in stripped:
        return True
    if _DISAMBIGUATION_OPTION_REQUEST_RE.search(stripped):
        return True
    # docs/agentcoveragetesting_reuse_log.md section 160 (task_87a928f3):
    # see the real corpus-scan-backed rationale in the comment above
    # _MISSING_INPUT_REQUEST_RE for why this third condition was added and
    # why its scope was chosen the way it was.
    return bool(_MISSING_INPUT_REQUEST_RE.search(stripped))


# docs/agentcoveragetesting_reuse_log.md section 162 (task_3a6cdbde, part 2):
# a real, distinct gap in this same disambiguation-heuristic family (after
# task_b81f1f07/task_3e77531b/task_d31870ce/task_87a928f3, sections 140/145/
# 152/160) -- investigated first against section 161's open question of
# whether this is (a) another driver-heuristic gap or (b) a real fixture/
# binder-layer problem, per the user's explicit instruction not to assume.
# Real finding: tau2's own default fixture (tau2-bench/data/tau2/domains/
# telecom/db.toml) gives customer C1001 an account-level `phone_number`
# field ("555-123-2002") that is simply the SAME db.toml's line L1002's own
# phone_number -- a coincidence of the SHARED default fixture every telecom
# branch binds against, not something telecom_116_state#b0's own binder
# introduced (confirmed: a real corpus grep of every bound plan in
# outputs/generic_bound_driver_plans_v0_1_telecom/plans.json found the exact
# same account-phone/line-phone mismatch on telecom_049_state#b0/telecom_
# 091_state#b0/telecom_094_state#b0/telecom_105_state#b0 too -- all 5 share
# one single root fixture fact, not 5 independent binder bugs). Confirmed
# NOT (b): this is not telecom_116-specific, and its own real object_
# bindings/operation_argument_fact_bundle already state the one, unambiguous
# correct line_id explicitly (no genuine binding ambiguity at the fixture
# layer -- the branch's own Given is concrete). It IS (a): a real, precise
# state-machine + verdict scan across every current-generation telecom
# result file (pass3-6/full/section136_fullverify, excluding pre-known_facts_
# message-fix historical smoke runs) found this SAME "2+ real candidate line
# IDs, one of them ours, and the assistant asking which one" shape recurring
# non-deterministically on telecom_091_state#b0/telecom_105_state#b0 too (not
# just telecom_116_state#b0) -- telecom_105_state#b0 itself shows a real
# verdict=incomplete instance in the section136_fullverify pass, the exact
# same live shape as this ticket's own target, confirming this is a genuine,
# reusable driver gap and not a single-branch quirk. The real, immediate
# cause: _assistant_has_confirmation_request's own weak "please confirm"
# marker (declared above) over-matches "...could you please confirm which
# line you'd like to refuel data for?" as a plain yes/no proceed-confirmation
# -- it is not one, it is a request for a SPECIFIC identifier from an
# explicit small set -- so the driver's generic "Yes, I confirm. Please
# proceed." answer (confirm_operation) never actually names a line, spending
# both of the branch's 2 confirmation attempts on a non-answer before the
# real call budget runs out. The fix does not touch
# _assistant_has_confirmation_request/_IDENTITY_REQUEST_EXCLUSION_RE (a
# narrow phrase-level exclusion there would not generalize to the OTHER real
# phrasing this same corpus scan found -- "Please confirm one of the
# following: 1. Line L1001... 2. Line L1002...", which contains no "which"
# adjacent to "confirm" at all); instead, checked ahead of the confirmation
# branch entirely, this recognizes the real, structural signal common to
# every one of the real matches found (informational-only line listings,
# like telecom_051_order#b0's own "you have three lines: L1001, L1002, and
# L1003 (Suspended)" recap, were checked too and do NOT match -- that
# message never names a "which" choice at all) and, uniquely among this
# heuristic family so far, answers with a real, already-known SPECIFIC fact
# (this branch's own line_id, already stated once already in known_facts_
# message) rather than a generic restatement -- because unlike the other 4
# gaps in this family (where restating the full original request truthfully
# answers any question shape), a generic restatement here would still not
# say "L1001", so it would not actually resolve this specific ambiguity.
_LINE_ID_TOKEN_RE = re.compile(r"\bL\d{3,}\b")
_LINE_DISAMBIGUATION_CUE_RE = re.compile(r"\bwhich\b", re.IGNORECASE)


def _assistant_message_needs_line_id_disambiguation_response(
    content: Any, known_line_id: str | None
) -> bool:
    if not known_line_id:
        return False
    stripped = _strip_markdown_emphasis(str(content or ""))
    if not stripped:
        return False
    tokens = set(_LINE_ID_TOKEN_RE.findall(stripped))
    if len(tokens) < 2 or known_line_id not in tokens:
        return False
    return bool(
        _LINE_DISAMBIGUATION_CUE_RE.search(stripped)
        or _DISAMBIGUATION_OPTION_REQUEST_RE.search(stripped)
    )


def _render_known_facts(bindings: Mapping[str, Any]) -> str:
    return "; ".join(f"{key}: {value}" for key, value in sorted(bindings.items()))


# docs/agentcoveragetesting_reuse_log.md section 82, Category G: telecom's
# real device/surroundings state (TelecomUserDB, a SEPARATE db from the
# agent-side TelecomDB) is never materialized for any of the 136 telecom
# bound plans (generic_tau_v2_bound_plan_adapter_v1.py hardcodes
# initial_state_patch.user_data to None for every domain, and nothing
# downstream ever applies it even when non-None -- a real, general gap, not
# specific to these 2 branches). The device-facing tools that would surface
# this state (check_status_bar/check_network_status/check_vpn_status/
# toggle_roaming/disconnect_vpn) all live on tau2's telecom-only
# TelecomUserTools, callable only by the SIMULATED USER, never the agent --
# and this driver's user is a deterministic script with no tool-calling
# capability at all (a deliberate, general, documented design choice,
# section 3.10: reproducibility / zero extra model calls on the user side).
# Rather than reconciling those two real constraints with a bigger change
# (either giving the user real tool-calling via tau2's own native
# UserSimulator(tools=...), which conflicts with that design choice, or
# building general device-state materialization nothing downstream would
# even consume yet), this is scoped to the 2 real branches confirmed to need
# it (telecom_063_order#b0/telecom_065_order#b0): the user's one-shot facts
# message states the relevant device/surroundings facts as prose (matching
# what the real user_tools would report if actually called against a
# correctly-materialized TelecomUserDB, per tau2/domains/telecom/user_tools.
# py's own real check_network_status/check_vpn_status output shapes), so a
# policy-compliant agent has what it needs to reach "you must guide them to
# use toggle_roaming()"/"...disconnect_vpn()" without ever needing to ask a
# diagnostic question the deterministic user has no way to answer. The
# semantic judge for both real requirements only inspects the agent's own
# utterance (never whether a user_tools call occurred), so this is
# sufficient for what's actually being checked.
_TELECOM_DEVICE_FACTS_OVERRIDE = {
    "telecom_063_order#b0": (
        "I checked my status bar and network status: I'm currently outside my "
        "carrier's coverage area (roaming), and Data Roaming shows as OFF."
    ),
    "telecom_065_order#b0": (
        "I checked my VPN status: VPN is ON and connected, and the connection "
        "performance shows as Poor."
    ),
    # docs/agentcoveragetesting_reuse_log.md section 124: telecom_078_order#b2-
    # #b6 each need the SUT to actually reach one distinct real conditional
    # fork of tech_support_workflow.md's Path 2.1 (2.1.2 traveling+roaming-
    # off / 2.1.2 traveling+roaming-on-but-line-not-enabled / 2.1.3 mobile-
    # data-off / 2.1.4 data-usage-exceeded / 2.1.4 data-usage-not-exceeded),
    # not just the shared Step 2.1.1 (confirm cellular service) #b1 already
    # covers -- same real Category C gap and same real fix as telecom_063_
    # order#b0 above (a deterministic scripted user has no other channel to
    # report a device/account state fact that isn't a real database
    # condition on these branches' own Given "True"). Each fact statement
    # states cellular service is already confirmed available (so the agent
    # doesn't loop back to Step 2.1.1) and the ONE distinct downstream
    # condition this branch's Then is grounded in -- the real tau2 telecom
    # deterministic-driver principle (section 3.10) means the agent is never
    # expected to call check_network_status()/run_speed_test() itself (real
    # USER-side tools, tau2/domains/telecom/user_tools.py) and verify this
    # independently -- it can only ever act on what the user reports, same
    # as telecom_063_order#b0's own already-verified pattern.
    "telecom_078_order#b2": (
        "I checked my status bar and network status: I have cellular "
        "service, but I'm currently traveling outside my usual service "
        "area, and Data Roaming shows as OFF."
    ),
    "telecom_078_order#b3": (
        "I checked my status bar and network status: I have cellular "
        "service, I'm currently traveling outside my usual service area, "
        "and Data Roaming shows as ON, but my mobile data still isn't "
        "working."
    ),
    "telecom_078_order#b4": (
        "I checked my status bar and network status: I have cellular "
        "service, I'm not traveling outside my usual service area, and "
        "Mobile Data shows as OFF."
    ),
    "telecom_078_order#b5": (
        "I checked my status bar and network status: I have cellular "
        "service, I'm not traveling, and Mobile Data shows as ON, but I "
        "still have no connectivity. I also checked and my data usage for "
        "this line has exceeded my plan's data limit this month."
    ),
    "telecom_078_order#b6": (
        "I checked my status bar and network status: I have cellular "
        "service, I'm not traveling, and Mobile Data shows as ON, but I "
        "still have no connectivity. I also checked and my data usage for "
        "this line has NOT exceeded my plan's data limit."
    ),
    # docs/agentcoveragetesting_reuse_log.md section 125: telecom_083_order#b2-
    # #b5 each need the SUT to actually reach one distinct real conditional
    # fork of tech_support_workflow.md's Path 3 (3.3 network technology/3.4
    # Wi-Fi Calling/3.5 messaging app permissions/3.6 APN settings), not just
    # the shared Steps 3.1/3.2 (confirm cellular service + mobile data
    # connectivity) #b1 already covers -- same real Category C gap and same
    # real fix as telecom_078_order#b2-#b6 above. Each fact statement states
    # cellular service and mobile data connectivity are already confirmed
    # working (so the agent doesn't loop back to Step 3.1/3.2) and the ONE
    # distinct downstream condition this branch's Then is grounded in. Real,
    # verified default fixture values this override deliberately contradicts
    # (tau2-bench/data/tau2/domains/telecom/user_db.toml, the SEPARATE
    # TelecomUserDB the agent has no tool access to at all -- see the
    # comment above _TELECOM_DEVICE_FACTS_OVERRIDE's declaration): default
    # network_technology_connected="5G" (already >=3G), wifi_calling_enabled
    # =false (already off), messaging app permissions sms=true/storage=true
    # (already both granted), active_apn_settings.mmsc_url is already set --
    # i.e. the default fixture state is the "already fine" branch of every
    # one of these 4 conditionals, so instantiating each branch's own "bad"
    # state is ONLY possible through this injected prose, exactly as for
    # telecom_078_order#b2/#b3/#b4 above. Unlike telecom_078_order#b5 (section
    # 124.11's real, documented vacuous-pass problem), none of these 4 fields
    # are ever readable by any real AGENT-callable tool (check_network_mode_
    # preference/check_wifi_calling_status/check_app_permissions/check_apn_
    # settings are all real tau2 USER-only tools, tau2/domains/telecom/
    # user_tools.py) -- so there is no real tool call through which the agent
    # could independently observe a fixture value contradicting this prose;
    # the prose is structurally the only, self-consistent source of truth the
    # agent has for these 4 branches (see docs/agentcoveragetesting_reuse_
    # log.md section 125 for the full real analysis).
    "telecom_083_order#b2": (
        "I checked and I have cellular service, and my mobile data "
        "connectivity is working, but my phone shows it's connected to a "
        "2G network only."
    ),
    "telecom_083_order#b3": (
        "I checked and I have cellular service, mobile data connectivity "
        "is working, and my phone is connected to a 4G network. I also "
        "checked and Wi-Fi Calling is turned ON."
    ),
    "telecom_083_order#b4": (
        "I checked and I have cellular service, mobile data connectivity "
        "is working, my phone is on a 4G network, and Wi-Fi Calling is "
        "off. I also checked my messaging app's permissions and it's "
        "missing the storage permission."
    ),
    "telecom_083_order#b5": (
        "I checked and I have cellular service, mobile data connectivity "
        "is working, my phone is on a 4G network, Wi-Fi Calling is off, "
        "and the messaging app has both the storage and SMS permissions "
        "granted. I also checked my APN settings and the MMSC URL field "
        "is empty."
    ),
    # docs/agentcoveragetesting_reuse_log.md section 160.2/161: telecom_113_
    # order#b0's own _INITIAL_MESSAGE_OVERRIDE (see above) already grounds
    # this branch's opening request in tech_support_workflow.md's real Path
    # 1 ("No Service / No Connection Troubleshooting") -- but the real
    # section 134 production transcript (outputs/generic_tau_baselines_v0_1_
    # telecom_full_verification_pass6/results/telecom_113_order__b0.json)
    # confirmed a real, live termination_reason=max_steps/verdict=incomplete:
    # the agent (correctly) verified the line is Active/not suspended, then
    # asked the user to run real check_status_bar()/check_network_status()
    # diagnostics on their phone (Path 1 Step 1.0/1.1/1.2) -- both real
    # tau2 USER-only TelecomUserTools this deterministic script has no way
    # to call -- and the conversation ran out its step budget with the
    # user repeatedly deflecting ("Is there anything else you're able to
    # do...") rather than ever supplying a real answer, so this branch's
    # own real OR01 semantic check ("the transcript must show the agent
    # attempted the relevant, available diagnostic/fix tool actions...
    # before this transfer_to_human_agents call") never got the chance to
    # observe anything. Same real Category G gap and same real fix as
    # telecom_063_order#b0/telecom_078_order#b2-#b6/telecom_083_order#b2-#b5
    # above: this branch's own real, frozen operation_argument_fact_bundle
    # (outputs/generic_bound_driver_plans_v0_1_telecom/plans.json) already
    # targets transfer_to_human_agents, and the real default fixture device
    # state (tau2-bench/data/tau2/domains/telecom/user_db.toml's single
    # [device] block: airplane_mode=false, network_signal_strength="good",
    # network_connection_status="connected", sim_card_status="active") is
    # once again the "everything fine" branch of every relevant Path 1
    # conditional -- the same structural reason telecom_083's own comment
    # above documents for its 4 conditionals -- so this branch's own real
    # Given ("no signal at all", from its own opening message) can only be
    # instantiated via this injected prose, not the shared default fixture.
    # Grounded in tau2's own real check_status_bar()/check_network_status()/
    # check_apn_settings() return-value vocabulary (tau2-bench/src/tau2/
    # domains/telecom/user_tools.py: "Airplane Mode: ON/OFF",
    # SimStatus.ACTIVE="active", NetworkStatus.NO_SERVICE="no_service",
    # SignalStrength.NONE="none", and the real default active_apn_settings
    # in tau2-bench/data/tau2/domains/telecom/user_db.toml -- apn_name=
    # "internet", mmsc_url="http://mms.carrier.com/mms/wapenc", both real,
    # already-set, non-empty defaults) -- Airplane Mode OFF and SIM Active
    # rule out the first 2 real, agent-guidable Path 1 fixes (toggle_
    # airplane_mode()/reseat_sim_card(), Steps 1.1/1.2), and normal APN
    # settings rule out the 3rd (reset_apn_settings()/reboot_device(), Step
    # 1.3). A real, live first attempt at this fix (only stating Steps
    # 1.1/1.2's facts) empirically confirmed the APN gap: deepseek-v4-flash
    # correctly used the injected Airplane Mode/SIM facts without asking
    # again, then proceeded to Step 1.3 on its own initiative ("Could you
    # please run check_apn_settings() on your device...") -- another real
    # USER-only tool this deterministic script still had no way to answer,
    # reproducing the same real max_steps/incomplete shape one diagnostic
    # step later. Stating the APN facts up front closes that gap the same
    # way telecom_083_order#b5's own override bundles 2 related facts (data
    # status + data usage) into one statement. With the line already
    # confirmed Active/not-suspended (from the account lookup the agent
    # already performs, Step 1.4) and every real, agent-available Path 1
    # fix now ruled out, no further real diagnostic remains to attempt --
    # consistent with this branch's own real target tool
    # (transfer_to_human_agents), same as telecom_063_order#b0's own
    # already-verified pattern above. A 2nd real, live online-verification
    # attempt (Steps 1.1/1.2/1.3's facts) empirically found a further real
    # gap: deepseek-v4-flash used all 3 stated facts correctly, but --
    # instead of proceeding to Step 1.4 (already answerable from the
    # account lookup it had already done) -- it invented 2 further
    # diagnostics of its own accord: re-asking to reseat_sim_card() despite
    # SIM already being stated Active (not a real tech_support_workflow.md
    # Step 1.2 branch -- reseating is only called for when SIM is MISSING),
    # and check_network_mode_preference() (a real Path 2.2 slow-DATA step,
    # not a real Path 1 no-service step at all). Real, live LLM depth on
    # this branch varies by sample; stating the network mode preference
    # fact too (real default, tau2-bench/data/tau2/domains/telecom/user_db.
    # toml: network_mode_preference="4g_5g_preferred", NetworkModePreference
    # .FOUR_G_5G_PREFERRED -- already the recommended setting) and making
    # the SIM fact explicitly rule out a reseat closes both real observed
    # avenues without asserting anything not already true of the real
    # default fixture. A 3rd real, live sample (same final text as this
    # entry, no code change) reproduced a genuine, non-vacuous pass (the
    # agent used all the stated facts correctly and called transfer_to_
    # human_agents with a grounded summary), confirming the fix works --
    # but a 4th real sample hit max_steps again after spending its budget
    # checking the account's 2 OTHER, unrelated lines (L1002/L1003) and
    # then asking whether the user is "currently located outside your...
    # [coverage area]" -- a real roaming/traveling question tech_support_
    # workflow.md's own Path 1 never asks (that belongs to Path 2.1.2, a
    # different, mobile-DATA path), but one this branch's real default
    # fixture already answers: tau2's own UserSurroundings.is_abroad
    # defaults to False (tau2-bench/src/tau2/domains/telecom/user_data_
    # model.py), and this branch's own initial_state_patch.user_data is
    # null (no override), so the real default applies. Stating it directly
    # (matching telecom_063_order#b0's own sibling override, which states
    # the opposite -- IS abroad -- for that branch's own real Given)
    # removes one more real avenue an agent can spend its limited turn
    # budget exploring instead of reaching this branch's real target
    # action.
    "telecom_113_order#b0": (
        "I checked my status bar and network status: Airplane Mode is OFF, "
        "my SIM card status shows as Active and properly seated (not "
        "missing or locked), but my cellular connection shows as 'no "
        "service' with no signal at all. I'm not traveling -- I'm in my "
        "home coverage area. I also checked my APN settings: Current APN "
        "Name is 'internet' and the MMSC URL is set (not empty), so those "
        "look normal too. My network mode preference is already set to "
        "4G/5G Preferred, the recommended setting."
    ),
}

# docs/agentcoveragetesting_reuse_log.md section 106: a real, narrowly-scoped
# exception to the project-wide deterministic-driver principle (section
# 3.10), for exactly these 2 branches, by explicit user decision after being
# shown the real tradeoff (extra real LLM call per user turn, real
# non-determinism for these 2 branches specifically) -- not a change to the
# default for any other branch in any domain. Real tau2 native
# UserSimulator(tools=...) (see tau2/user/user_simulator.py, tau2/runner/
# build.py's build_user for the canonical real reference invocation) plays
# the user with real access to the real TelecomUserTools
# (check_network_status/check_vpn_status/toggle_roaming/disconnect_vpn),
# replacing _TELECOM_DEVICE_FACTS_OVERRIDE above for just these 2 (that
# override becomes moot for them -- the real fact is now established via
# the simulated user's own persona instructions + real, materialized device
# state below, not injected prose).
#
# real initialization_actions (env_type="user", tau2's own EnvFunctionCall
# shape) applied via Environment.set_state -- confirmed safe (unlike
# initial_state.initialization_data.user_data, which triggers a real,
# destructive self.tools.db = self.user_tools.db reassignment documented in
# the comment on materialize_generic_tau_task's initial_state below;
# initialization_actions never touches that path, it only calls the named
# real method via Environment.run_env_function_call). Real fixture values
# (not invented): AgentSpecTesting's own default telecom fixture line
# L1001's real phone_number is "555-123-2001" (confirmed against this
# project's own db.toml, NOT tau2's upstream example task's "555-123-2002"
# customer contact number, which is a different, unrelated field) and
# customer C1001's real name is "John Smith". telecom_063 needs real
# surroundings.is_abroad=True (set_user_location) for the LLM-user's
# persona to genuinely know it's traveling -- device.roaming_enabled is
# already real-False by default in this project's own user_db.toml, no
# patch needed there. telecom_065 needs a real VPN connection with real
# Poor performance -- break_vpn() (user_tools.py) does exactly this in one
# real call (connect_vpn() + server_performance=Poor), the exact state
# telecom_065's own Given describes.
_TELECOM_USER_SIMULATOR_BRANCHES = {
    "telecom_063_order#b0": {
        "instructions": (
            "You are John Smith, calling your telecom carrier. You are currently "
            "traveling and outside your carrier's home coverage area (roaming). "
            "You've noticed your phone's mobile data isn't working. If the agent "
            "asks you to check your phone's status or network settings, use the "
            "real check_network_status/check_status_bar tools available to you "
            "and report back exactly what they show -- don't invent details "
            "beyond what the tools report. If the agent tells you what to do "
            "(e.g. to turn on data roaming), you don't need to actually perform "
            "the action yourself -- just acknowledge their guidance."
        ),
        # Real identity facts this real customer/line actually has (db.toml),
        # the same ones the deterministic driver's own known_facts mechanism
        # would state -- without these, the LLM-user genuinely cannot answer
        # the agent's real, correct identity-verification step (confirmed:
        # a first real run gave up with ###OUT-OF-SCOPE### here).
        "known_info": (
            "Your full name is John Smith, date of birth 1985-06-15, "
            "customer ID C1001, and the phone number for the line you're "
            "asking about is 555-123-2001."
        ),
        "initialization_actions": [
            {"env_type": "user", "func_name": "set_user_info",
             "arguments": {"name": "John Smith", "phone_number": "555-123-2001"}},
            {"env_type": "user", "func_name": "set_user_location",
             "arguments": {"abroad": True}},
        ],
    },
    "telecom_065_order#b0": {
        "instructions": (
            "You are John Smith, calling your telecom carrier. Your mobile data "
            "connection has been slow. If the agent asks you to check your "
            "phone's VPN or network status, use the real check_vpn_status/"
            "check_network_status tools available to you and report back "
            "exactly what they show -- don't invent details beyond what the "
            "tools report. If the agent tells you what to do (e.g. to "
            "disconnect your VPN), you don't need to actually perform the "
            "action yourself -- just acknowledge their guidance."
        ),
        "known_info": (
            "Your full name is John Smith, date of birth 1985-06-15, "
            "customer ID C1001, and the phone number for the line you're "
            "asking about is 555-123-2001."
        ),
        "initialization_actions": [
            {"env_type": "user", "func_name": "set_user_info",
             "arguments": {"name": "John Smith", "phone_number": "555-123-2001"}},
            {"env_type": "user", "func_name": "break_vpn", "arguments": {}},
        ],
    },
}

# docs/agentcoveragetesting_reuse_log.md section 90: telecom_030_arg#b0's own
# real request/given/when text ("The user requests to suspend a line.") never
# states a reason, so generic_tau_telecom_v2_fixture_binder_v1.py's
# free_form_placeholders honestly fills "reason" with a generic placeholder
# ("Customer requested this change.") -- but this branch's own OR01 check
# specifically demands the reason be "specific, substantive... not a generic
# placeholder", which a fully honest, undecorated placeholder can never
# satisfy no matter how the agent behaves. Confirmed real (not a compiler/
# binder bug): same "abstract source scenario vs. a check that demands
# concrete content" gap as _TELECOM_DEVICE_FACTS_OVERRIDE and
# _INITIAL_MESSAGE_OVERRIDE below, fixed the same way -- a scoped, hand-
# verified content override at the driver layer (per the user's explicit
# choice, matching section 82's Category G precedent) rather than rewriting
# the Step1-2 source GWT text and re-cascading the whole pipeline. The
# replacement reason is concrete and grounded in a real, common suspend_line
# scenario (a lost/stolen device -- explicitly one of the judge criterion's
# own examples of a valid specific reason) without asserting any fact the
# conversation doesn't already carry once the user states it.
_TELECOM_REASON_OVERRIDE = {
    "telecom_030_arg#b0": (
        "My phone was lost, so I'd like to suspend this line right away to "
        "prevent unauthorized use until I get a replacement."
    ),
}

# docs/agentcoveragetesting_reuse_log.md section 162 (task_3a6cdbde, part 1):
# a real, structurally distinct scripted-driver gap from
# _TELECOM_DEVICE_FACTS_OVERRIDE above -- that dict answers a real,
# read-only USER-side diagnostic question the driver can state up front, in
# known_facts_message, before the agent ever asks (the fact already exists
# before the conversation starts). This one is a real WRITE-then-confirm
# loop: the agent (correctly, per main_policy.md's require-payment-before-
# resuming rule) calls the real agent-side send_payment_request() tool
# (tau2-bench/src/tau2/domains/telecom/tools.py), then asks the user to run
# the real USER-only check_payment_request()/make_payment() tools
# (tau2-bench/src/tau2/domains/telecom/user_tools.py) and report back that
# the payment went through -- an event that cannot even be known until
# AFTER the agent's own send_payment_request() call, so no upfront
# known_facts prose can ever answer it. A real, live telecom_094_state#b0
# pass6 transcript (outputs/generic_tau_baselines_v0_1_telecom_full_
# verification_pass6/results/telecom_094_state__b0.json) confirmed the
# deterministic driver has no scripted branch that ever answers this:
# known_facts_message (already spent earlier in the same conversation) and
# the one-shot nudge_toward_target_action ("Is there anything else...",
# also already spent right where this gap bites) both fire before this
# point, and neither one's canned text ever asserts the payment actually
# completed -- the real conversation exhausted its call budget
# (observed_agent_calls=8, termination_reason=max_steps) with the agent
# still waiting on confirmation it could never receive, before ever
# reaching this branch's real target tool (resume_line). Deliberately NOT
# added to _TELECOM_DEVICE_FACTS_OVERRIDE above: that dict's own triggering
# condition (stated unconditionally, once, in known_facts_message, before
# any agent question) is the wrong shape for a fact that can only become
# true once a specific real WRITE tool call (send_payment_request) has
# already happened -- this needs its own trigger (see
# GenericDeterministicTauUser.generate_next_message's own
# _tool_called(state, "send_payment_request") gate below), not a
# conflation with the unrelated device-diagnostics gap. Deliberately
# doesn't name a specific bill ID or dollar amount: the agent -- not the
# bound plan -- discovers at runtime which bill is overdue (via get_bills_
# for_customer/get_details_by_id), so a hardcoded amount here would risk
# asserting a value that doesn't match what actually happened in a given
# run; the phrasing is grounded only in check_payment_request()'s/make_
# payment()'s own real return-value vocabulary ("You have a payment
# request for bill ..."/"Payment of ... has been made for bill ...")
# without repeating either tool's specific numbers back.
# docs/agentcoveragetesting_reuse_log.md section 167: real, live online-
# verification finding (telecom_055_order#b0 postfix reruns) that section
# 162's own original trigger condition -- self._tool_called(state,
# "send_payment_request") -- can NEVER actually observe a real
# send_payment_request() call in the REAL live orchestrator, regardless of
# branch. Real read of tau2's own Orchestrator.step() (tau2-bench/src/tau2/
# orchestrator/orchestrator.py): "user_msg, self.user_state = self.user.
# generate_next_message(self.message, self.user_state)" -- self.user_state
# is threaded PURELY by this class's own return value from call to call
# (get_init_state() is called once, with an empty history, at the start of
# a fresh simulation; nothing ever re-seeds it with the full trajectory
# mid-conversation). This class's own generate_next_message only ever does
# `state["messages"].append(result)` with `result` being its OWN outgoing
# reply -- the incoming `message` argument (the agent's own text) is never
# appended, and the agent's real tool-call messages are NEVER routed to
# the user role at all (tau2's own half-duplex routing sends a tool-call
# message to Role.ENV, and that tool's response back to Role.AGENT, never
# to Role.USER) -- confirmed by real-replaying a live post-fix transcript
# (outputs/_section167_online_verify/telecom_055_order_b0_postfix_seed0/
# results/telecom_055_order__b0.json) side-by-side against a manual replay
# using the SAME bound plan: the manual replay (which artificially seeds
# state["messages"] with the full real transcript, exactly as this file's
# own existing unit tests already did) fires the payment-confirmation
# branch correctly, but the REAL live run -- which never had that
# artificial seeding -- fell through past it every time. In other words:
# state["messages"] as tracked through a real live conversation is really
# just this class's own private log of what IT has said, never a real
# window onto the agent's tool calls -- so _tool_called(state, ...)/
# _target_tool_called(state) can never be true in a real live run,
# independent of this section's own reordering fix. (This also means the
# existing nudge branch's own `not self._target_tool_called(state)` guard,
# section 95, has always been structurally inert in real live execution --
# out of scope to change here since it is harmless there: nudge already
# fires at most once, gated by self.sent_nudge, so an always-True "target
# not yet called" read only means nudge always gets its one real shot,
# never that it fires twice.)
#
# The real, structural fix: since the agent's own real tool calls are
# genuinely invisible to the user role, the only real signal available is
# the SAME one every other heuristic in this file already relies on -- the
# agent's own CURRENT plain-text message content. Real corpus-wide scan
# (all 5 real full-corpus passes, pass3 through pass7, every telecom
# result file) confirms this is reliable: every real branch that ever
# calls send_payment_request follows it, in the very next real assistant
# turn, with a message that (a) reports the request as already sent
# ("has been sent"/"has already been sent"/"was sent"/"I've sent"/"sent to
# you"/"sent for") AND (b) tells the user to check_payment_request -- both
# directly grounded in main_policy.md's own scripted procedure ("Inform
# the user that a payment request has been sent. They should: Check their
# payment requests using the check_payment_request tool."). A real,
# precise scan requiring BOTH conditions across all 5 passes found 47
# distinct real branch/pass combinations whose real first post-payment
# turn matches -- 0 misses (every one of them is caught on its real FIRST
# occurrence, exactly where this branch needs to fire) and, after adding
# an explicit "no unable/not able/can't/cannot/no further action" whole-
# message exclusion (to correctly exclude a real, different, genuinely
# unrelated shape -- telecom_120_state#b0/telecom_047_state#b0's own real
# messages, which mention "check your payment requests" and "sent" only
# while explaining why a bill CANNOT get a NEW request sent for it, e.g.
# because one is already pending -- firing a stand-in "I already paid it"
# claim there would misrepresent a real policy-driven refusal as a
# resolved payment and risk steering that branch's own, unrelated,
# already-passing real behavior), 0 real false positives across the full
# 5-pass corpus either.
_PAYMENT_REQUEST_JUST_SENT_RE = re.compile(
    r"payment request.{0,100}?(?:has\s+(?:already\s+)?been\s+sent|was\s+sent|"
    r"sent\s+(?:to\s+you|for))"
    r"|(?:i(?:'|’)?ve|i\s+have|i\s+already)\s+(?:already\s+)?sent.{0,100}?payment request",
    re.IGNORECASE | re.DOTALL,
)
# section 175: `payment[\s_]+` (was `payment\s+`) -- the bare tool name
# "check_payment_request" (underscore between payment and request) never
# matched, so a report like "A payment request has been sent to you ...
# Please review it using the check_payment_request tool" (real, telecom_055_
# order#b0 section-175 seed 2, idx12) fell through to known-facts/stop. Real
# rescan of pass3-pass8 + section 167/175 runs: 14 newly matched assistant
# messages, all 14 after a real send_payment_request call, 0 new matches
# without one.
_CHECK_PAYMENT_REQUEST_MENTION_RE = re.compile(
    r"check[\s_]+(?:your\s+)?payment[\s_]+requests?", re.IGNORECASE
)
_PAYMENT_REQUEST_NEGATION_RE = re.compile(
    r"\b(?:unable to|not able to|can(?:'|’)?t\b|cannot\b|won(?:'|’)?t be able|"
    r"no\s+(?:further|additional)\s+(?:action|payment request))\b",
    re.IGNORECASE,
)


def _assistant_message_reports_payment_request_sent(content: str | None) -> bool:
    if not content:
        return False
    text = _strip_markdown_emphasis(str(content))
    if _PAYMENT_REQUEST_NEGATION_RE.search(text):
        return False
    return bool(_PAYMENT_REQUEST_JUST_SENT_RE.search(text)) and bool(
        _CHECK_PAYMENT_REQUEST_MENTION_RE.search(text)
    )


_TELECOM_PAYMENT_CONFIRMATION_OVERRIDE = {
    "telecom_094_state#b0": (
        "Okay, I just checked my payment requests using check_payment_request "
        "and found the one you sent me -- I went ahead and accepted it with "
        "make_payment, and it confirmed the payment has been made "
        "successfully. That bill should be all paid now."
    ),
}

# docs/agentcoveragetesting_reuse_log.md section 167: real corpus-wide scan
# (outputs/generic_bound_driver_plans_v0_1_telecom/plans.json cross-referenced
# against 5 real full-corpus online verification passes -- pass3 through
# pass7 -- under outputs/generic_tau_baselines_v0_1_telecom_full_
# verification_pass{3,4,5,6,7}/) found this is NOT a one-branch gap: the
# SAME real WRITE-then-confirm shape (agent calls the real send_payment_
# request() tool, then asks the user to run the real USER-only check_
# payment_request()/make_payment() tools and report back, which the
# deterministic scripted driver has no real channel to answer) genuinely,
# repeatedly recurs across multiple, structurally distinct telecom
# branches -- not just telecom_094_state#b0 (section 162, already fixed
# above): telecom_095_state#b0 (2/5 real historical passes hit it,
# `requires_observation_unmet_due_to_premature_termination`/max_steps),
# telecom_055_order#b0 (3/5 real historical passes hit it -- its own OR01
# target isn't resume_line, but a technical-support "resolution action"
# family that genuinely includes resume_line, so the same stuck payment
# loop starves it of the resolution event it needs before budget runs
# out), telecom_049_state#b0 (1/5, real pass3 transcript: agent finds a
# real overdue bill B1002, sends the payment request, then dead-ends
# waiting on confirmation it can never get), and telecom_008_arg#b0 (1/5,
# real pass5 transcript, same shape, terminates via ###STOP### instead of
# max_steps but the same real `missing_required_observation` gap). Real
# per-pass nondeterminism (the SAME fixture/branch sometimes has the model
# take the cautious "verify payment first" path and sometimes doesn't) means
# a branch-ID dict can only ever catalog gap instances this specific
# 5-pass sample of runs happened to surface -- any other resume_line-
# adjacent branch (telecom_009_arg#b0/telecom_117_state#b3, both share the
# identical L1003/B1002 fixture as telecom_008/009, 0/5 hits in this sample
# but no structural reason they couldn't hit it on a future sample) is a
# real, live risk of the exact same gap recurring under a branch ID this
# dict doesn't yet know about. Genuinely warrants generalizing past a
# closed enumeration: _TELECOM_PAYMENT_CONFIRMATION_OVERRIDE's own real
# telecom_094_state#b0 entry was ALREADY written branch-agnostically
# (section 162.4's own declaration: "doesn't name a specific bill ID or
# dollar amount") specifically so it would stay true for any branch, not
# just that one -- so the override dict was never actually doing branch-
# specific customization, only branch-specific gating. The trigger itself
# is already precise and domain-safe without needing a branch-ID allowlist:
# it only fires on a real, live, observed send_payment_request tool-call
# event in the transcript (a genuine telecom-only tool name -- confirmed
# absent from every real retail/airline tool list) with the branch's own
# real target tool (if any) genuinely not yet called, so it can only ever
# activate on a live conversation that has already taken the real
# WRITE-then-confirm path this fix answers -- never a text-pattern match,
# never something a differently-shaped branch could accidentally trip.
# _TELECOM_PAYMENT_CONFIRMATION_OVERRIDE itself is kept (not deleted) as
# the historical/documented per-branch override point (still consulted
# first, so a future branch that genuinely needs different wording -- e.g.
# one grounded in a specific, hardcoded bill amount -- can still override
# this default), but every telecom branch without its own entry now falls
# back to this same, already-grounded, already branch-agnostic message
# rather than silently getting no coverage at all. Deliberately scoped to
# telecom only (checked by the real "telecom_" branch_id prefix at the call
# site below, the same convention every other telecom-only dict in this
# file already uses) -- retail and airline have no send_payment_request
# tool at all, so this is a structural no-op for both, but the explicit
# prefix check keeps that guarantee independent of tau2's own tool list
# never changing underneath this code.
_TELECOM_PAYMENT_CONFIRMATION_DEFAULT_MESSAGE = (
    "Okay, I just checked my payment requests using check_payment_request "
    "and found the one you sent me -- I went ahead and accepted it with "
    "make_payment, and it confirmed the payment has been made "
    "successfully. That bill should be all paid now."
)

# docs/agentcoveragetesting_reuse_log.md section 90: telecom_032_arg#b0 and
# retail_051_arg#b0's own real request/when text is the abstract GWT
# description itself ("The user asks for help that requires escalation to a
# human agent." / "...that cannot be resolved through automated
# assistance.") used verbatim as the user's opening (and, since this
# deterministic user has nothing else to say, only) message -- the same
# Category C "abstract GWT text never instantiated into a concrete request"
# gap already documented for retail_058/059 and telecom_037 (reuse log
# section 74.4, deferred there as needing real Step1-3 content authoring).
# Each of these 2 branches' own OR01 check specifically demands the
# transfer_to_human_agents summary describe a real, concrete issue the user
# raised -- unsatisfiable when the user never raised one. Fixed the same way
# as _TELECOM_REASON_OVERRIDE above: a scoped, hand-verified concrete issue
# for just these 2 branches, each one genuinely outside every real tool this
# domain's agent has (confirmed by reading tau2's real tool lists -- telecom
# has no plan-change tool, retail has no membership-tier-change tool), so a
# policy-compliant agent transferring after trying to help is still correct,
# not something the real toolset should have resolved instead.
_INITIAL_MESSAGE_OVERRIDE = {
    # Real-rerun finding: the first scenario tried here ("switch to a
    # different phone plan") wasn't clean -- get_available_plan_ids/get_
    # details_by_id let the agent partially engage (browse/list plans) before
    # ever reaching the point of realizing no tool can actually apply the
    # change, burning the whole call budget and terminating on max_steps
    # before a transfer decision. Replaced with an issue that has NO tool
    # path at all, not even a read-only one (confirmed against the real
    # tau2 telecom tool list -- no tool reads or writes a customer's phone
    # number/email/contact info), for a clean, immediate transfer decision.
    "telecom_032_arg#b0": (
        "Hi, I'd like to update the phone number on file for my account, "
        "but I don't see that as an option here -- can you connect me with "
        "someone who can help?"
    ),
    # section 206: telecom_038_norm#b0's Then is an if-and-only-if; the
    # "cannot be handled -> must transfer" direction was chosen (its bound
    # target tool is transfer_to_human_agents). The spec's own request ("The
    # user makes a request.") names nothing, so this is a concrete request
    # no telecom tool can serve (no tool changes a customer's name), without
    # asking for a transfer itself.
    "telecom_038_norm#b0": (
        "Hi, I'm John Smith, customer ID C1001. I recently got married and "
        "I'd like to change the name on my account to John Smith-Lee."
    ),
    "retail_051_arg#b0": (
        "Hi, I'd like to upgrade my membership tier from regular to gold, "
        "but I don't see an option for that here -- could you connect me "
        "with someone who can help?"
    ),
    # docs/agentcoveragetesting_reuse_log.md section 95: same Category C gap
    # as telecom_032/retail_051 above, but these 2 branches' own OR01 checks
    # additionally require the agent to genuinely ATTEMPT resolution
    # ("try its best to resolve" / "try all possible ways") before
    # transferring -- an abstract request gives the agent nothing to
    # genuinely investigate, so real-rerun confirmed the agent either gets
    # stuck reflexively asking for details it never receives (telecom_040)
    # or transfers immediately with no real attempted-resolution step to
    # observe (telecom_056). A disputed bill charge is a real issue the
    # agent CAN genuinely investigate (get_bills_for_customer/get_details_
    # by_id) before concluding it's unresolvable -- no real tool disputes,
    # waives, or removes a charge (confirmed against the real tau2 telecom
    # tool list), so a policy-compliant agent still ends up transferring
    # after a real, observable attempt.
    "telecom_040_norm#b0": (
        "Hi, I noticed a charge on my last bill that I don't recognize and "
        "I'd like to dispute it -- can you help me figure out what it is "
        "and get it removed?"
    ),
    "telecom_056_norm#b0": (
        "Hi, there's a charge on my most recent bill I don't recognize and "
        "I want it disputed and removed -- can you look into this for me?"
    ),
    # docs/agentcoveragetesting_reuse_log.md section 99: same Category C gap
    # again -- these 2 branches' abstract source When text ("The user
    # requests assistance with mobile data functionality." /
    # "...troubleshooting for MMS issues.") gives the deterministic driver
    # nothing to sustain a follow-up conversation with once the agent
    # reasonably asks what the specific problem is; real rerun confirmed
    # the agent gets stuck asking a clarifying question the driver can't
    # answer (078) or ends the scripted turns right as it reaches the real
    # diagnostic-guidance step, before finishing it (083) -- both real,
    # session's-own-authored branches whose new Then requires observing
    # that guidance step actually happen, so a concrete, immediately
    # diagnosable complaint (no clarifying question needed) is required for
    # the check to get a fair chance to be exercised at all.
    "telecom_078_order#b1": (
        "Hi, my mobile data isn't working at all -- I have no internet "
        "connection on my phone. Can you help me figure out what's wrong?"
    ),
    "telecom_083_order#b1": (
        "Hi, I can't send or receive picture messages (MMS) on my phone -- "
        "can you help me troubleshoot this?"
    ),
    # docs/agentcoveragetesting_reuse_log.md section 125: telecom_083_order#b2-
    # #b5, 4 new branches sharing #b0/#b1's own abstract source When text
    # ("The user requests troubleshooting for MMS issues.") -- same Category C
    # gap as #b1 itself (section 99), reusing #b1's own already-verified
    # opening line so the conversation naturally starts at tech_support_
    # workflow.md Step 3.1 (verify network service status) for all 4, same as
    # #b1's own online run -- the branch-specific Path 3 conditional state
    # (network technology/Wi-Fi Calling/app permissions/APN) is supplied via
    # _TELECOM_DEVICE_FACTS_OVERRIDE below, once the agent asks a follow-up
    # diagnostic question (matching telecom_063_order#b0's and section 124's
    # telecom_078_order#b2-#b6's own established pattern for the same reason).
    "telecom_083_order#b2": (
        "Hi, I can't send or receive picture messages (MMS) on my phone -- "
        "can you help me troubleshoot this?"
    ),
    "telecom_083_order#b3": (
        "Hi, I can't send or receive picture messages (MMS) on my phone -- "
        "can you help me troubleshoot this?"
    ),
    "telecom_083_order#b4": (
        "Hi, I can't send or receive picture messages (MMS) on my phone -- "
        "can you help me troubleshoot this?"
    ),
    "telecom_083_order#b5": (
        "Hi, I can't send or receive picture messages (MMS) on my phone -- "
        "can you help me troubleshoot this?"
    ),
    # docs/agentcoveragetesting_reuse_log.md section 100: same Category C
    # gap as telecom_032/retail_051/telecom_040/telecom_056 above --
    # left open at section 74.4/90.5 pending this exact fix. All 3 below
    # share the same abstract source When text ("The user makes a request
    # that is against the policy."/"...cannot be handled within the scope
    # of the agent's actions.") used verbatim as the driver's opening line,
    # giving the agent nothing concrete to deny/transfer against. Each
    # replacement is grounded in a real, customer-data-independent policy
    # rule or a real, structural tool-catalog gap (confirmed against the
    # real tau2 tools.py for each domain), not a specific customer's
    # fictional state -- the branches' own Given is "True" (no database
    # anchor), so the injected request must hold regardless of which real
    # customer/order gets bound.
    "telecom_037_norm#b0": (
        # main_policy.md: "The maximum amount of data that can be refueled
        # is 2GB." -- refuel_data() itself never enforces this (confirmed:
        # no gb_amount cap check in tools.py), so this is a pure
        # policy-level denial, not something a real tool call would reject
        # on its own; true for any real customer/line.
        "Hi, I'd like to refuel my line with 5GB of extra data, please."
    ),
    "retail_058_norm#b0": (
        # policy.md: "You can only help one user per conversation... and
        # must deny any requests for tasks related to any other user." --
        # true regardless of which real user is bound.
        "Hi, while you're helping me, could you also go ahead and cancel "
        "my friend's order for them? Their name is Alex."
    ),
    "retail_059_order#b0": (
        # Real retail tools.py has no tool to place a brand-new order at
        # all (only cancel/modify/exchange/return an EXISTING order) --
        # confirmed by reading every real tool in the file; structurally
        # out of scope for any real customer, not a policy judgment call.
        "Hi, I'd like to place a new order for a pair of running shoes."
    ),
    # airline_076_norm#b0: real-rerun finding (same section 100 fix,
    # discovered while verifying the transfer_to_human_agents::summary
    # scope_constraints fix online for the first time) -- this branch's
    # own Given ("The user's request cannot be handled within the scope
    # of the agent's actions") was ALSO never given a concrete request,
    # same Category C gap as the 3 branches above (not originally in
    # scope for this fix, added because the real online rerun surfaced
    # it). Real airline tools.py has no tool for seat assignment/
    # selection at all (confirmed: no such tool among the 8 real airline
    # tools) -- structurally out of scope for any real reservation.
    "airline_076_norm#b0": (
        "Hi, I'd like to request a specific seat assignment (a window "
        "seat) for my upcoming flight -- can you help me with that?"
    ),
    # airline_085_order#b0 (section 74.4/103): real Given is "the user does
    # not know their reservation id", requesting an unnamed "action that
    # requires a reservation id" -- OR01 only checks whether the agent calls
    # get_user_details (real tau2 tools.py: returns the user's full
    # reservations: List[str]), OR02 that this happens BEFORE the requested
    # operation. Real bound user chen_jackson_3290 genuinely owns 4
    # reservations (confirmed via the real fixture db), so "cancel my
    # reservation" alone is ambiguous -- the route+date below are real
    # facts of the specific bound reservation (#4WQ150: DFW->LAX round
    # trip, outbound 2024-05-22), grounded, not invented, and specific
    # enough for a real agent to identify it among the 4 via get_user_
    # details + get_reservation_details without ever being told the ID.
    "airline_085_order#b0": (
        "Hi, I'd like to cancel my upcoming trip from Dallas to Los "
        "Angeles, departing around May 22nd -- but I don't have my "
        "reservation ID on hand. Can you help me find it and cancel it?"
    ),
    # telecom_055_order#b0 (section 47/106): same Category C gap as the
    # others above -- real Given is "True" (unconstrained), real When is
    # the abstract "The user requests technical support.", which left the
    # agent's own natural next question ("what's the actual issue?") with
    # nothing concrete to answer once the customer-identification step (the
    # branch's own left_event) was done. Real, verified-against-the-fixture
    # scenario: the bound default customer C1001's OWN third line (L1003,
    # phone 555-123-2003) is genuinely Suspended in the real db.toml, with
    # no overdue bill for C1001 (confirmed: none of C1001's real bills
    # B1001/B1002/B1003 has status "Overdue") and a real contract_end_date
    # (2026-06-30) still in the future relative to the real simulated clock
    # (main_policy.md: "2025-02-25") -- main_policy.md's own Line
    # Suspension section names exactly these two conditions as the only
    # real blockers to lifting a suspension, and neither applies, so this
    # is a genuinely liftable suspension, not a scenario engineered to
    # succeed. resume_line is also one of the real 5 tools the new case C
    # override (section 106) recognizes as satisfying this branch's
    # "technical support operation" requirement.
    "telecom_055_order#b0": (
        "Hi, I'm calling about technical support -- my line 555-123-2003 "
        "got suspended and I'd like to get it reactivated, please."
    ),
    # docs/agentcoveragetesting_reuse_log.md section 176 (the one-shot
    # disambiguation gate question, section 171.4): a real cross-domain
    # state-machine replay of all 2,730 pass2-pass8 transcripts found the
    # one-shot restate gate is NOT what keeps this branch failing -- its
    # dialogue_contract's initial_user_message is the abstract GWT text "The
    # user requests to cancel, modify, return, or exchange an order, or to
    # modify their default user address." (the only retail plan with that
    # text), so restate_original_request_for_disambiguation's own truthful
    # answer ("Please go ahead with everything I originally asked for: ...")
    # repeats a list of mutually exclusive actions, and real agents answer it
    # with an explicit refusal (pass5 idx12 "I can't perform a vague 'do
    # everything' request"; pass8 idx12 "I can't perform all of those
    # together"). A 2nd identical restate would get the same refusal. Same
    # Category C gap as the other entries in this dict, fixed the same way:
    # the concrete request this branch's own bound values already encode --
    # its OR01 (modify_pending_order_items: #W5918442, 1725100896 ->
    # 9007697085, credit_card_5051208) and OR02 (modify_pending_order_address:
    # 215 River Road, Suite 991, New York, NY 10083, USA) both require BOTH
    # changes on the same pending order. Real fixture (tau2 retail db.json):
    # #W5918442 is sofia_rossi_8776's pending order, item 1725100896 is its
    # Perfume (oriental/30ml/unisex), 9007697085 is an available Perfume
    # variant (fresh/50ml/men), credit_card_5051208 is her real card.
    # tests/test_generic_tau_online_v1_retail_055_disambiguation_gate_v1.py
    # pins these values against the live bound plan so a later fixture
    # rebuild cannot silently desync this text.
    "retail_055_order_modify_user_address#b0": (
        "Hi, I'd like to make two changes to my pending order #W5918442, "
        "please: change its shipping address to 215 River Road, Suite 991, "
        "New York, NY 10083, USA, and also swap the Perfume in that order "
        "(item 1725100896) for the variant with item ID 9007697085, paying "
        "any price difference with my credit card credit_card_5051208."
    ),
    # docs/agentcoveragetesting_reuse_log.md section 108 (item 7, sub-pattern C
    # scoping report): same Category C gap as telecom_032/airline_077 below --
    # real Given is "True", real When is the abstract "A user request or
    # situation requires transfer to a human agent.", giving the driver
    # nothing concrete to open with. main_policy.md:5 scopes the telecom
    # agent to exactly 4 areas (technical support, overdue bill payment, line
    # suspension, plan options); cross-checked against the real telecom
    # tools.py tool list -- no tool ports a number to another carrier, a
    # real, common, structurally out-of-scope request distinct from every
    # other already-fixed override's scenario.
    "telecom_039_order#b0": (
        "Hi, I'd like to port my phone number to a different carrier -- "
        "can you help me get that started?"
    ),
    # docs/agentcoveragetesting_reuse_log.md section 108 (item 7): same
    # Category C gap, but retail_059_order#b1v0/v1's own Given flips to "the
    # request CAN be handled within scope" (the opposite direction from the
    # already-fixed #b0's out-of-scope grounding) -- the check only demands
    # the agent NOT transfer, so any genuinely in-scope, resolvable request
    # satisfies it. Grounded in the real bound customer's own real order
    # #W2611340 (confirmed via tau2's real retail db.json: james_li_5688 owns
    # this order) -- a plain status lookup (get_order_details) is real,
    # in-scope, and resolvable regardless of the order's real status
    # ("processed", not "pending" -- deliberately NOT worded as a cancel
    # request, since cancel_pending_order would genuinely reject a
    # non-pending order and that would be a structurally wrong grounding).
    "retail_059_order#b1v0": (
        "Hi, I'd like to check the status of my order #W2611340, please."
    ),
    "retail_059_order#b1v1": (
        "Hi, I'd like to check the status of my order #W2611340, please."
    ),
    # docs/agentcoveragetesting_reuse_log.md section 108 (item 7): same
    # Category C gap as telecom_039/airline_076 -- real Given is "True", real
    # When is the abstract "A transfer to a human agent is required.". A
    # lost-and-found item claim is a real, structurally out-of-scope request
    # distinct from airline_076's seat-assignment grounding -- confirmed
    # against the real airline tools.py's full tool list: no tool touches
    # lost items.
    "airline_077_order#b0": (
        "Hi, I think I left my jacket on my flight -- can you help me get "
        "it back?"
    ),
    # docs/agentcoveragetesting_reuse_log.md section 111: same Category C
    # gap -- real online rerun confirmed it (not assumed): telecom_090's
    # real Given is "True", real When is the abstract "The agent receives
    # an id and retrieves details using get_details_by_id." -- the
    # deterministic driver's own literal When text as the opening message
    # gives the agent no concrete id to look up, so the agent asked which
    # id the user meant instead of ever calling get_details_by_id at all,
    # and the check's own required observation never fired (match_count=0,
    # not a real substantive result). Replaced with a concrete request
    # naming this branch's own real, frozen bound fixture line id (L1001,
    # confirmed against this branch's real operation_argument_fact_bundle
    # in outputs/generic_bound_driver_plans_v0_1_telecom/plans.json) so the
    # agent has an immediate, unambiguous reason to call get_details_by_id.
    "telecom_090_order#b0": (
        "Hi, can you pull up the details for L1001 for me?"
    ),
    # telecom_113's real Given is "True", real When is the abstract "The
    # user requests support or escalation to a human agent." -- same real
    # rerun-confirmed gap: the agent (correctly) asked what the actual
    # problem was, the deterministic driver had no concrete issue to
    # supply, and the conversation ran out its turn budget with neither a
    # resolution nor a transfer_to_human_agents call ever happening, so the
    # check's required observation never fired either. Replaced with a
    # real, immediately-diagnosable Path 1 (No Service) issue from
    # tech_support_workflow.md, giving the agent concrete troubleshooting
    # steps it can actually attempt before any potential transfer decision.
    "telecom_113_order#b0": (
        "Hi, my phone has no signal at all right now -- can you help me "
        "figure out what's wrong?"
    ),
    # docs/agentcoveragetesting_reuse_log.md section 124: telecom_078_order#b2-
    # #b6, 5 new branches sharing #b0/#b1's own abstract source When text
    # ("The user requests assistance with mobile data functionality.") --
    # same Category C gap as #b1 itself (section 99), reusing #b1's own
    # already-verified opening line so the conversation naturally starts at
    # tech_support_workflow.md Step 2.1.1 (confirm cellular service) for all
    # 5, same as #b1's own online run -- the branch-specific Path 2.1
    # conditional state (traveling/roaming/mobile-data/data-usage) is
    # supplied via _TELECOM_DEVICE_FACTS_OVERRIDE below, once the agent asks
    # a follow-up diagnostic question (matching telecom_063_order#b0's own
    # established pattern for the same reason).
    "telecom_078_order#b2": (
        "Hi, my mobile data isn't working at all -- I have no internet "
        "connection on my phone. Can you help me figure out what's wrong?"
    ),
    "telecom_078_order#b3": (
        "Hi, my mobile data isn't working at all -- I have no internet "
        "connection on my phone. Can you help me figure out what's wrong?"
    ),
    "telecom_078_order#b4": (
        "Hi, my mobile data isn't working at all -- I have no internet "
        "connection on my phone. Can you help me figure out what's wrong?"
    ),
    "telecom_078_order#b5": (
        "Hi, my mobile data isn't working at all -- I have no internet "
        "connection on my phone. Can you help me figure out what's wrong?"
    ),
    "telecom_078_order#b6": (
        "Hi, my mobile data isn't working at all -- I have no internet "
        "connection on my phone. Can you help me figure out what's wrong?"
    ),
    # docs/agentcoveragetesting_reuse_log.md section 143.7/146: telecom_027_
    # arg#b0's own real source trigger text ("the user provides a phone
    # number to look up their account") is the same Category C gap as every
    # override above -- an abstract GWT description of the scenario, not a
    # concrete value -- used verbatim as dialogue_contract.initial_user_
    # message (generic_tau_telecom_v2_fixture_binder_v1.py's canonical_
    # request := plan["test_point"]["when"]["supplied_user_request"], itself
    # v5_given_when_assembly_v1.py's user["trigger"]["trigger"], an LLM
    # (gpt-4.1) paraphrase of the GWT "when" clause -- this pattern is
    # shared by all ~40 telecom ARG branches' trigger text and is normally
    # harmless, since a policy-compliant agent recognizes it isn't a real
    # value and asks a clarifying question before the driver supplies the
    # real fact on the next turn (exactly as GenericDeterministicTauUser's
    # own known_facts_message mechanism is designed to do). For most ARG
    # branches this is inert either because the agent asks first, or
    # because the check itself is undiscriminating (e.g. telecom_000_arg#b0
    # only requires phone_number to be *a string*, which the placeholder
    # text itself already satisfies). telecom_027_arg#b0's own real OR01
    # check is different: it's a strict format predicate
    # (`^\d{3}-\d{3}-\d{4}$`) on whatever value get_customer_by_phone is
    # actually called with. A real online rerun (section 143.7,
    # outputs/generic_tau_baselines_v0_1_telecom_full_verification_pass3/
    # results/telecom_027_arg__b0.json) confirmed the target agent can,
    # non-deterministically, skip the clarifying question and pass the
    # placeholder text straight through as a real get_customer_by_phone
    # phone_number argument, producing a real tool error ("Customer with
    # phone number the user provides a phone number to look up their
    # account not found") and a real, genuine OR01 fail (event_index 2's
    # actual value fails the regex) -- a prior run (section 136) of the
    # exact same branch had the agent ask first and pass. Fixed the same
    # way as telecom_090_order#b0 above: replace the abstract trigger text
    # with a concrete request naming this branch's own real, frozen bound
    # fixture phone number for line L1001 (confirmed against this branch's
    # real operation_argument_fact_bundle.arguments.phone_number in
    # outputs/generic_bound_driver_plans_v0_1_telecom/plans.json:
    # "555-123-2001", the same real L1001/C1001/John Smith fixture already
    # used by telecom_063_order#b0/telecom_065_order#b0/telecom_090_order#b0
    # above -- not invented), so OR01 gets a fair chance to be exercised
    # against a real, valid value regardless of whether the agent asks a
    # clarifying question first.
    "telecom_027_arg#b0": (
        "Hi, my phone number is 555-123-2001 -- can you look up my account "
        "using that?"
    ),
}

# docs/agentcoveragetesting_reuse_log.md section 103: a branch whose own
# real Given is "the user does NOT know X" must not have GenericDeterministic
# TauUser's mechanical fact-fallback hand X over anyway just because it's in
# object_bindings (object_bindings is the oracle's own authoritative real
# value -- correct to use for verifying what tool call the agent eventually
# makes, but wrong to disclose to the user upfront when the Given explicitly
# says they don't have it). Scoped per-branch, not a general suppression --
# every other branch's object_bindings are real facts the user DOES know.
_UNKNOWN_TO_USER_FIELDS_OVERRIDE = {
    "airline_085_order#b0": frozenset({"reservation_id"}),
}


# docs/agentcoveragetesting_reuse_log.md section 80, Category F: a bounded
# cap, not an unconditional "always confirm" -- see generate_next_message's
# own use of this for the real evidence. No real branch in the corpus needed
# a 3rd confirmation round.
MAX_CONFIRMATIONS = 2


# docs/agentcoveragetesting_reuse_log.md section 178: the retail fields a
# real identity lookup takes (tau2 retail tools.py: find_user_id_by_email
# (email), find_user_id_by_name_zip(first_name, last_name, zip)). Telecom's
# lookups (get_customer_by_phone/_by_id/_by_name) and airline's
# (get_user_details(user_id)) were scanned too: every bound value there
# already resolves to the bound customer/user and no target tool shares a
# key with them, so neither needs this.
_RETAIL_IDENTITY_LOOKUP_FIELDS = ("email", "first_name", "last_name", "zip")
_IDENTITY_FIELD_LABELS = {
    "email": "email address", "first_name": "first name", "last_name": "last name", "zip": "zip code",
}
_ADDRESS_GROUP_FIELDS = ("address1", "address2", "city", "state", "country", "zip")


def _retail_identity_profile(bound_plan: Mapping[str, Any], agent_db: Any) -> dict[str, Any] | None:
    """section 178: the authenticated retail user's real profile, from the real
    environment DB. "Authenticated" = the user the scripted user's own email
    fact resolves to (every one of the 185 retail plans carries one), i.e.
    exactly who find_user_id_by_email would return -- so the name+zip path
    lands on the SAME account. Deliberately not keyed on user_id: section
    128's identity_binding_overrides make user_id the adversarial TARGET on
    retail_087_state#b1/retail_096_state#b1 while email/name stay the
    authenticated user's. None when nothing resolves (no change then)."""
    facts = {
        **((bound_plan.get("operation_argument_fact_bundle") or {}).get("arguments") or {}),
        **(bound_plan.get("object_bindings") or {}),
    }
    email = facts.get("email")
    users = getattr(agent_db, "users", None)
    if not email or not users:
        return None
    for user_id, user in users.items():
        if user.email == email:
            return {
                "user_id": user_id,
                "email": user.email,
                "first_name": user.name.first_name,
                "last_name": user.name.last_name,
                "zip": user.address.zip,
                "address": {field: getattr(user.address, field) for field in _ADDRESS_GROUP_FIELDS},
            }
    return None


def _conversation_statement_requirements(bound_plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """docs/agentcoveragetesting_reuse_log.md section 182 (task_e9deccb5): the
    Given clauses this branch's Step7 binder realized as something the user
    has to SAY (dialogue_contract.conversation_requirements, realization
    "user_statement") -- e.g. retail_023_arg#e1's "the user chooses more than
    one payment method, all different from the original payment method", for
    which no db field exists and which a flat single-valued fact bundle cannot
    express. Each entry carries the statement text its binder built from the
    branch's own real Given clause and real bound values; this driver never
    authors one. Empty (and every derived message byte-identical to before)
    for every bound plan without that key -- every telecom/airline plan and
    every retail plan except the branches the retail binder opts in."""
    requirements = (bound_plan.get("dialogue_contract") or {}).get("conversation_requirements") or []
    return [
        requirement for requirement in requirements
        if isinstance(requirement, Mapping)
        and requirement.get("realization") == "user_statement"
        and requirement.get("statement")
    ]


def _override_branch_key(bound_plan: Mapping[str, Any]) -> str | None:
    """The key for the hand-verified per-branch override tables above. Section
    236: a targeted variant plan (skip_branch_overrides) keeps its seed's
    source_branch_id for the oracle but brings its own opening and facts, so
    none of the seed's overrides may leak into it; production plans never set
    the flag, so nothing changes for them."""
    if bound_plan.get("skip_branch_overrides"):
        return None
    return bound_plan.get("source_branch_id")


def _bound_initial_user_message(bound_plan: Mapping[str, Any]) -> str:
    """The scripted user's opening request: the hand-verified
    _INITIAL_MESSAGE_OVERRIDE or the dialogue contract's own request, followed
    by every section-182 conversation statement (the Given's conversation-only
    clauses are stated AS PART OF the request, so the restate/"(Regarding:
    ...)" paths that reuse the opening message carry them too). Shared by
    GenericDeterministicTauUser and materialize_generic_tau_task's
    reason_for_call so both always say the same thing."""
    base = _INITIAL_MESSAGE_OVERRIDE.get(
        _override_branch_key(bound_plan), bound_plan["dialogue_contract"]["initial_user_message"]
    )
    statements = [requirement["statement"] for requirement in _conversation_statement_requirements(bound_plan)]
    if not statements:
        return base
    return " ".join([base, *statements])


class GenericDeterministicTauUser:
    """One-shot factual user with a deterministic confirmation response, and a
    mechanical fallback that shares the branch's own real object_bindings (never
    invents a value -- the exact same rendering materialize_generic_tau_task already
    puts in the task's own known_info field) whenever the agent asks for information
    beyond the bare initial request and isn't asking for a final confirmation. This
    matches dialogue_contract.answer_agent_fact_requests_from's own declared intent
    (see docs/agentcoveragetesting_reuse_log.md section 63) -- previously unimplemented,
    the user had no way to answer a real "what's your customer ID?" follow-up and
    just gave up with ###STOP### instead."""

    def __init__(
        self, bound_plan: Mapping[str, Any], *, user_tool_names: Any = None,
        identity_profile: Mapping[str, Any] | None = None, target_tool_argument_names: Any = None,
    ) -> None:
        dialogue = bound_plan["dialogue_contract"]
        # section 182: the override/contract request plus any conversation-only
        # Given clause the binder realized as a user statement.
        self.initial_message = _bound_initial_user_message(bound_plan)
        self.confirmation_message = dialogue["confirmation_message"]
        # object_bindings is the oracle-verified, authoritative source: it's derived
        # from real driver_bindings.* references INSIDE the compiled oracle's own
        # runtime_observation_binding checks (v5_driver_plan_v2.py's binding_names),
        # meaning every one of its fields is something a real tool call argument is
        # actually checked against -- these must read as operational parameters to
        # actively USE, not just background facts to mention. A real finding
        # (section 63, telecom_010): a flat, undifferentiated "key: value; key:
        # value" dump buried a real required argument (limit=5) among identity
        # fields, and the agent never picked it up as something to pass to the tool
        # call. operation_argument_fact_bundle is only supplementary real context
        # (e.g. a customer's real full_name) for the agent's other real questions --
        # never itself oracle-checked. {tool_name, arguments} is the real,
        # established shape (see generic_tau_airline_v1.py's own v1-lineage bound
        # plans and tests/test_generic_tau_online_v1.py) -- only "arguments" holds
        # fact key/value pairs; "tool_name" is metadata, never a fact to share.
        object_bindings = dict(bound_plan["object_bindings"])
        unknown_to_user = _UNKNOWN_TO_USER_FIELDS_OVERRIDE.get(_override_branch_key(bound_plan)) or frozenset()
        object_bindings_for_display = {
            key: value for key, value in object_bindings.items() if key not in unknown_to_user
        }
        fact_bundle = bound_plan.get("operation_argument_fact_bundle") or {}
        supplementary = {
            key: value for key, value in (fact_bundle.get("arguments") or {}).items()
            # Excluded from display for two distinct reasons: it's already
            # covered by object_bindings (the normal case), or the branch's
            # own Given says the user doesn't know it (unknown_to_user) --
            # checked against the FULL object_bindings, not the filtered
            # display copy, so a field this override suppresses from
            # object_bindings can't leak back in here as "supplementary".
            if key not in object_bindings and key not in unknown_to_user
        }
        object_bindings = object_bindings_for_display
        self.target_tool_name = fact_bundle.get("tool_name")
        # docs/agentcoveragetesting_reuse_log.md section 162 (task_3a6cdbde,
        # part 2): the one already-known, already-stated "line_id" fact this
        # branch's own operation_argument_fact_bundle/object_bindings
        # carries -- kept separately (not just buried in known_facts_message)
        # so generate_next_message can restate it, specifically and by
        # itself, the instant the assistant's own message shows it's
        # genuinely torn between 2+ real candidate line IDs (see
        # _needs_line_id_disambiguation_response below for the real,
        # corpus-scan-backed root cause this answers).
        self.line_id_fact = object_bindings.get("line_id") or supplementary.get("line_id")
        reason_override = _TELECOM_REASON_OVERRIDE.get(_override_branch_key(bound_plan))
        if reason_override and "reason" in supplementary:
            supplementary["reason"] = reason_override
        # docs/agentcoveragetesting_reuse_log.md section 178: separate "who I
        # am" from "what I want changed". The fact bundle is flat, so a key
        # like `zip` can hold the NEW address's zip (retail modify_user_
        # address / modify_pending_order_address branches) while the agent's
        # name+zip lookup (find_user_id_by_name_zip) needs the zip ON FILE.
        # identity_profile is the authenticated user's real profile, read at
        # runtime from the real environment DB (see _retail_identity_profile);
        # only keys whose bound value genuinely differs from it are touched,
        # so every non-conflicting branch keeps a byte-identical message.
        identity_notes: list[tuple[str, Any, Any]] = []
        self.identity_fact_corrections: dict[str, tuple[Any, Any]] = {}
        if identity_profile:
            target_args = set(target_tool_argument_names or ())
            for key in _RETAIL_IDENTITY_LOOKUP_FIELDS:
                profile_value = identity_profile.get(key)
                if profile_value is None or (key not in object_bindings and key not in supplementary):
                    continue
                bound_value = object_bindings[key] if key in object_bindings else supplementary[key]
                if str(bound_value) == str(profile_value):
                    continue
                # zip never travels alone: it is one field of an address, so
                # the whole address group moves/corrects together (otherwise
                # e.g. "996 Cedar Street, San Antonio, TX" would be paired
                # with a different user's zip).
                group = _ADDRESS_GROUP_FIELDS if key == "zip" else (key,)
                if key in object_bindings or key in target_args:
                    # A real argument of the requested operation (the new
                    # value): keep it exactly, shown with the operation's own
                    # values, and state the on-file value separately.
                    for field in group:
                        if field in supplementary and (field in target_args or field == key):
                            object_bindings[field] = supplementary.pop(field)
                    identity_notes.append((key, bound_value, profile_value))
                else:
                    # Not an argument of the requested operation at all, so
                    # it can only be read as an account detail -- and as one
                    # it was wrong-sourced (e.g. the address of ANOTHER user's
                    # order). State the authenticated user's real one.
                    for field in group:
                        real_value = identity_profile.get("address", {}).get(field) if key == "zip" else profile_value
                        if field in supplementary and field not in target_args and real_value is not None:
                            if str(supplementary[field]) != str(real_value):
                                self.identity_fact_corrections[field] = (supplementary[field], real_value)
                            supplementary[field] = real_value
        # docs/agentcoveragetesting_reuse_log.md section 182: a conversation
        # statement can say MORE than a single bound value can (retail_023_
        # arg#e1: the user chooses TWO payment methods; object_bindings keeps
        # the first as the oracle's absent-check scope). Presenting the lone
        # bound value as "use exactly these values" would contradict the
        # statement and invite exactly the single-method change the Given
        # rules out, so the displayed value is the user's full stated choice.
        # Display only -- object_bindings (the oracle's authoritative scope)
        # is never touched. A no-op for every plan without such statements.
        for requirement in _conversation_statement_requirements(bound_plan):
            for key, value in (requirement.get("stated_values") or {}).items():
                if key in object_bindings:
                    object_bindings[key] = deepcopy(value)
                elif key in supplementary:
                    supplementary[key] = deepcopy(value)
        parts = []
        if object_bindings:
            parts.append(f"Please use exactly these values for the operation: {_render_known_facts(object_bindings)}.")
        if supplementary:
            parts.append(f"Additional account details, if needed: {_render_known_facts(supplementary)}.")
        for key, new_value, on_file in identity_notes:
            label = _IDENTITY_FIELD_LABELS.get(key, key)
            parts.append(
                f"To be clear, the {key} {new_value} above is part of the new details I'm asking you to "
                f"change to, not my current account info -- the {label} currently on file for my account is "
                f"{on_file}, so please use that one if you need to look up my account."
            )
        device_facts = _TELECOM_DEVICE_FACTS_OVERRIDE.get(_override_branch_key(bound_plan))
        if device_facts:
            parts.append(device_facts)
        # Real finding (section 63): a bare fact reply with no restated goal
        # sometimes reads to the agent as a brand-new, contextless message -- it
        # loses track of the original request and asks "what would you like help
        # with?" even right after being given full identifying details. Restating
        # the original request alongside the facts every time keeps that context
        # anchored.
        if parts:
            parts.append(f"(Regarding: {self.initial_message})")
        self.known_facts_message = " ".join(parts)
        # docs/agentcoveragetesting_reuse_log.md section 167: real
        # corpus-wide scan (see _TELECOM_PAYMENT_CONFIRMATION_DEFAULT_
        # MESSAGE's own declaration above for the full real evidence) found
        # this gap genuinely recurs across multiple telecom branches, not
        # just telecom_094_state#b0 -- every telecom branch now gets this
        # same, already branch-agnostic message (an explicit per-branch
        # entry in _TELECOM_PAYMENT_CONFIRMATION_OVERRIDE still wins if one
        # is ever added); real "telecom_" branch_id prefix check keeps this
        # a structural no-op for retail/airline, matching every other
        # telecom-only dict/gate in this file.
        _payment_branch_id = bound_plan.get("source_branch_id") or ""
        if _payment_branch_id.startswith("telecom_"):
            self.payment_confirmation_message = _TELECOM_PAYMENT_CONFIRMATION_OVERRIDE.get(
                _override_branch_key(bound_plan), _TELECOM_PAYMENT_CONFIRMATION_DEFAULT_MESSAGE
            )
        else:
            self.payment_confirmation_message = None
        # docs/agentcoveragetesting_reuse_log.md section 175 (task_e2474c72):
        # when the real environment genuinely exposes tau2's own user-side
        # check_payment_request()/make_payment() tools (telecom only), the
        # payment confirmation is no longer just a scripted claim -- the user
        # really calls both tools (tau2's own half-duplex USER->ENV->USER
        # tool-call routing, the same channel tau2's native UserSimulator
        # uses), and only then states the outcome, quoting the tools' real
        # return values. Combined with materialize_generic_tau_task's new
        # set_user_info() action (which activates TelecomEnvironment.
        # sync_tools()), make_payment() genuinely flips the agent-side bill to
        # PAID, so an agent that re-checks get_bills_for_customer sees the
        # truth. Without those tools (unit-test replays with no live
        # environment, or any non-telecom domain), the pre-section-175 text-
        # only behavior is kept unchanged.
        self.executes_real_payment_tools = bool(
            self.payment_confirmation_message
            and {"check_payment_request", "make_payment"} <= set(user_tool_names or ())
        )
        self.payment_flow_stage: str | None = None
        self.payment_check_result: str | None = None
        self.user_tool_call_count = 0
        self.sent_initial = False
        self.confirmation_count = 0
        self.sent_facts = False
        self.sent_nudge = False
        self.sent_disambiguation_response = False
        self.sent_payment_confirmation = False
        self.line_disambiguation_count = 0
        self.transport_action_log: list[dict[str, Any]] = []

    def get_init_state(self, message_history: list[Any] | None = None) -> dict[str, Any]:
        return {"messages": list(message_history or [])}

    def _tool_called(self, state: dict[str, Any], tool_name: str | None) -> bool:
        if not tool_name:
            return False
        for message in state.get("messages") or []:
            value = _message_dict(message)
            for call in value.get("tool_calls") or []:
                if isinstance(call, Mapping) and call.get("name") == tool_name:
                    return True
        return False

    def _target_tool_called(self, state: dict[str, Any]) -> bool:
        return self._tool_called(state, self.target_tool_name)

    def _user_tool_call(self, name: str) -> Any:
        from tau2.data_model.message import ToolCall

        self.user_tool_call_count += 1
        return ToolCall(
            id=f"scripted_user_tool_call_{self.user_tool_call_count}",
            name=name, arguments={}, requestor="user",
        )

    def _continue_real_payment_flow(self, value: Mapping[str, Any]) -> tuple[str | None, list[Any] | None, str]:
        """docs/agentcoveragetesting_reuse_log.md section 175: the 2nd/3rd
        steps of the real user-side payment flow started by generate_next_
        message's payment-confirmation branch. `value` is the real ToolMessage
        tau2 routes back to the USER role (ENV -> USER) with the real return
        value of the user's own previous tool call. Every outcome is reported
        truthfully from that real return value -- nothing is claimed that the
        tools didn't actually say."""
        result_text = str(value.get("content") or "").strip()
        errored = bool(value.get("error"))
        if self.payment_flow_stage == "awaiting_check_result":
            if not errored and result_text.startswith("You have a payment request"):
                self.payment_check_result = result_text
                self.payment_flow_stage = "awaiting_payment_result"
                return None, [self._user_tool_call("make_payment")], "make_payment_via_user_tool"
            self.payment_flow_stage = None
            return (
                "I just ran check_payment_request on my phone, but it says: "
                f"\"{result_text}\" -- I don't see a payment request from you, so I "
                "haven't been able to pay anything yet.",
                None,
                "report_payment_request_not_found",
            )
        # awaiting_payment_result
        self.payment_flow_stage = None
        if not errored and result_text.startswith("Payment of"):
            return (
                f"{self.payment_confirmation_message} (check_payment_request showed: "
                f"\"{self.payment_check_result}\"; make_payment returned: \"{result_text}\")",
                None,
                "confirm_payment_request_completed",
            )
        return (
            "I found your payment request with check_payment_request "
            f"(\"{self.payment_check_result}\"), but when I tried make_payment it said: "
            f"\"{result_text}\" -- so the payment did not go through.",
            None,
            "report_payment_not_completed",
        )

    def generate_next_message(self, message: Any, state: dict[str, Any]):
        from tau2.data_model.message import UserMessage

        tool_calls = None
        if not self.sent_initial:
            content = self.initial_message
            action_kind = "provide_initial_request_bundle"
            self.sent_initial = True
        else:
            value = _message_dict(message)
            if self.payment_flow_stage is not None:
                # section 175: a real user-side payment tool call is in
                # flight; this incoming message is its real ToolMessage.
                content, tool_calls, action_kind = self._continue_real_payment_flow(value)
            elif (
                self.line_disambiguation_count < MAX_CONFIRMATIONS
                and _assistant_message_needs_line_id_disambiguation_response(
                    value.get("content"), self.line_id_fact
                )
            ):
                # docs/agentcoveragetesting_reuse_log.md section 162: see
                # _assistant_message_needs_line_id_disambiguation_response's
                # own declaration for the real root-cause analysis. Checked
                # BEFORE the confirmation branch below on purpose -- this
                # exact message shape ("...could you please confirm which
                # line...") would otherwise be swallowed by
                # _assistant_has_confirmation_request's weak "please
                # confirm" marker and answered with a generic non-answer
                # that never names a line. Bounded the same way confirmation
                # itself is (MAX_CONFIRMATIONS attempts, not unconditionally
                # forever) so a genuinely confused agent that keeps asking
                # in some other unrecognized shape still falls through to
                # the existing confirmation/nudge/disambiguation-restate/
                # stop chain below rather than looping on this branch alone.
                content = (
                    f"Please use line {self.line_id_fact} -- that's the specific "
                    "line I meant for this request."
                )
                action_kind = "restate_line_id_for_disambiguation"
                self.line_disambiguation_count += 1
            elif (
                self.payment_confirmation_message
                and not self.sent_payment_confirmation
                and _assistant_message_reports_payment_request_sent(value.get("content"))
                and not self._target_tool_called(state)
            ):
                # docs/agentcoveragetesting_reuse_log.md section 167 (real
                # online-verification findings, telecom_055_order#b0):
                # TWO real, live bugs found and fixed together here.
                #
                # (1) Ordering: this branch used to sit AFTER the generic
                # confirmation elif below (section 162's original
                # placement, right before the generic nudge). A real live
                # rerun (seed 1) found a real assistant message that is
                # BOTH the post-send_payment_request turn AND happens to
                # end with a generic proceed question the existing
                # _assistant_has_confirmation_request heuristic's own
                # strong marker set already recognizes ("...let me know
                # once you've accepted it so I can process the payment.
                # Would you like to proceed?") -- with the old ordering,
                # the generic confirmation elif fired FIRST and consumed
                # the turn with a bare "Yes, I confirm. Please proceed."
                # that never actually asserts the payment completed,
                # silently reproducing the exact real gap this fix exists
                # to close. Moved here, immediately after the line-ID
                # disambiguation branch and BEFORE the generic confirmation
                # elif -- the same precedence discipline section 162.6
                # already established for line-ID disambiguation over
                # confirmation: this branch's own trigger is strictly more
                # specific than the generic heuristic's weak/strong text
                # markers, so it must win whenever both would otherwise
                # match the same real turn.
                #
                # (2) Trigger condition itself: section 162's original
                # self._tool_called(state, "send_payment_request") can
                # never actually be true in a real live run -- see
                # _assistant_message_reports_payment_request_sent's own
                # declaration above for the full real root-cause analysis
                # (tau2's own message routing never delivers the agent's
                # tool-call events to the user role at all) and the real,
                # 5-pass, 47-branch/pass corpus scan backing the replacement
                # text-based detector. Still gated by self.
                # sent_payment_confirmation (fires once) and by the real
                # target tool not yet being called, so this cannot resurrect
                # a stale fact once the real operation has already gone
                # through, and confirmation_count/sent_facts are left
                # completely untouched on the turns where THIS branch fires
                # (a real confirmation ask that still needs an actual "yes"
                # still gets one, on a later turn, once this one-shot fact
                # has already been delivered).
                self.sent_payment_confirmation = True
                if self.executes_real_payment_tools:
                    # section 175 (task_e2474c72): actually do it -- run the
                    # real user-side check_payment_request() now; make_payment
                    # () and the (now truthful) confirmation text follow in
                    # _continue_real_payment_flow once tau2 routes each real
                    # tool result back to this user.
                    content = None
                    tool_calls = [self._user_tool_call("check_payment_request")]
                    action_kind = "check_payment_request_via_user_tool"
                    self.payment_flow_stage = "awaiting_check_result"
                else:
                    content = self.payment_confirmation_message
                    action_kind = "confirm_payment_request_completed"
            elif self.confirmation_count < MAX_CONFIRMATIONS and _assistant_has_confirmation_request(value):
                # Real bug found on real retail/airline batch runs (docs/
                # agentcoveragetesting_reuse_log.md section 80, Category F):
                # a real, policy-compliant agent sometimes asks for
                # confirmation TWICE in the same conversation -- e.g. an
                # early "confirm both changes?" ask before the fact bundle
                # was fully populated, then a later, final "confirm you'd
                # like me to proceed?" ask after the agent did more real due
                # diligence (checking price/availability) -- and a single
                # boolean here permanently spent the one "yes" on the first
                # ask, so the real mutating tool call for the second,
                # genuine confirmation never got a "yes" and the driver fell
                # through to STOP before it fired. A bounded counter (not
                # unconditionally always confirming) keeps the original
                # section-69 intent of not looping forever on a confused
                # agent, while allowing the one extra round every real
                # transcript in this corpus actually needed -- no real
                # branch needed a 3rd round.
                content = self.confirmation_message
                action_kind = "confirm_operation"
                self.confirmation_count += 1
            elif self.known_facts_message and not self.sent_facts:
                content = self.known_facts_message
                action_kind = "provide_requested_facts"
                self.sent_facts = True
            elif (
                not self.sent_nudge
                and self.target_tool_name
                and not self._target_tool_called(state)
            ):
                # docs/agentcoveragetesting_reuse_log.md section 95: real
                # branches (telecom_040) whose real request needs genuine
                # investigation before the agent can conclude the real
                # target action (e.g. transfer_to_human_agents) is actually
                # needed -- the agent does real, substantive work (looks up
                # the account, pulls up real bill data, presents it) but
                # its message neither asks a question nor requests
                # confirmation, so with nothing scripted left to say the
                # driver used to fall straight to ###STOP### before the
                # agent ever reached the real target action. Bounded to
                # once (like confirmation above) and gated on the real
                # target tool genuinely not having been called yet -- once
                # it has (the real, common case), this is skipped entirely
                # and falls through to the same unconditional stop as
                # before, so this cannot reintroduce section 69's real bug
                # (repeating a completed operation's facts message forever).
                content = (
                    "Is there anything else you're able to do to help with this, or would this "
                    "need to be handled some other way?"
                )
                action_kind = "nudge_toward_target_action"
                self.sent_nudge = True
            elif (
                not self.sent_disambiguation_response
                and _assistant_message_needs_disambiguation_response(value.get("content"))
            ):
                # docs/agentcoveragetesting_reuse_log.md section 81.3/106
                # (sub-pattern C, part 1): a real, genuine disambiguation
                # question ("did you want the address changed, the item
                # exchanged, or both?") can be left over after both facts
                # and the target-action nudge have already been sent --
                # real cause found on retail_055_order_modify_user_address
                # #b0: the driver's own initial_message phrased two real,
                # both-intended operations with an ambiguous "or", and a
                # reasonable agent asks which one is meant. Restating the
                # FULL original request verbatim answers this truthfully
                # (it already states everything this user actually wants;
                # nothing new is fabricated) for any real question shape,
                # not just this one branch's specific wording -- bounded to
                # once, same discipline as confirmation/nudge above, so a
                # genuinely unanswerable repeated question still reaches
                # the real ###STOP### below rather than looping.
                #
                # section 176: deliberately still one-shot (NOT raised to
                # MAX_CONFIRMATIONS). Real replay of all 2,730 pass2-pass8
                # transcripts (3 domains): this gate blocked a disambiguation-
                # shaped message 1,063 times, but only 6 of those were in a
                # non-pass run whose target tool had not yet been called, and
                # in none of the 6 would a 2nd, byte-identical restate answer
                # what was actually asked (which single charge, a policy
                # refusal, an identical-address fixture no-op, or the agent
                # having already rejected the first restate as too vague).
                # Raising the bound would instead add a spurious turn to
                # every one of the other 1,057 blocked conversations (491 of
                # them after the target already completed -- section 69's
                # documented temporal-check hazard). See the reuse log.
                content = (
                    f"Please go ahead with everything I originally asked for: {self.initial_message}"
                )
                action_kind = "restate_original_request_for_disambiguation"
                self.sent_disambiguation_response = True
            else:
                # Real bug found on a real retail batch run (docs/
                # agentcoveragetesting_reuse_log.md section 69): the facts
                # message is one fixed, complete dump of everything this
                # user knows, sent verbatim -- restating it a second time
                # tells the assistant nothing new, so a real, correct
                # completion turn (e.g. "your order has been modified")
                # used to fall through to "restate_known_facts" and repeat
                # forever, turn after turn, until the real agent-call
                # budget ran out. That produced spurious extra dialogue
                # turns AFTER the real operation had already succeeded,
                # which then tripped temporal_relation precedence checks
                # expecting a fresh confirm/list-details step before each
                # of those spurious turns. Once every real thing this user
                # has to say (initial request, confirmation if asked,
                # known facts, and -- section 95 -- one nudge toward the
                # real target action if it still hasn't happened) has been
                # said once, there is nothing left to contribute -- stop
                # rather than loop.
                content = "###STOP###"
                action_kind = "stop"
        entry = {
            "transport_action_index": len(self.transport_action_log),
            "action_kind": action_kind,
            "content": content,
        }
        if tool_calls:
            # section 175: a real user-side tool call carries no text (tau2's
            # half-duplex protocol); content stays None, which normalize_tau_
            # execution already skips when aligning action kinds to real
            # user_message events.
            entry["tool_calls"] = [{"name": call.name, "arguments": dict(call.arguments)} for call in tool_calls]
        self.transport_action_log.append(entry)
        result = UserMessage(role="user", content=content, tool_calls=tool_calls, cost=0.0)
        state["messages"].append(result)
        return result, state

    def set_seed(self, seed: int) -> None:
        return None

    def stop(self, message: Any = None, state: Any = None) -> None:
        return None


def _telecom_scenario_user_info(bound_plan: Mapping[str, Any], agent_db: Any) -> dict[str, str] | None:
    """docs/agentcoveragetesting_reuse_log.md section 175 (task_e2474c72):
    the real (name, phone_number) pair tau2's own TelecomUserTools.set_user_
    info() needs so TelecomEnvironment.sync_tools() -- the ONLY real bridge
    between agent-side DB writes (send_payment_request, resume_line,
    enable_roaming, refuel_data, ...) and the user-side surroundings
    (payment_request, line_active, roaming_allowed, mobile_data_usage_
    exceeded) -- actually runs instead of returning early on
    surroundings.phone_number is None.

    Derived only from this branch's own real, bound fixture, never invented:
    the branch's own line_id (object_bindings first, then operation_argument_
    fact_bundle.arguments -- the same lookup order line_id_fact already uses),
    resolved against the real (already-patched) agent DB to that line's own
    real phone_number, and the owning customer's real full_name. The line's
    phone -- not a bound `phone_number` fact -- is used on purpose: 6/147 real
    telecom plans bind `phone_number` to the CUSTOMER's contact number (e.g.
    telecom_094_state#b0: bound 555-123-2002, but its line L1003 is 555-123-
    2003), and tau2's own sync_tools() keys everything (line status, roaming,
    data usage) on _get_line_by_phone(phone_number), i.e. the phone number of
    the LINE the scenario is about -- exactly what tau2's own official tasks
    do (service_issues.py/mobile_data_issues.py: set_user_info with the
    user's own line number). Returns None (the pre-section-175 behavior,
    sync stays inert) whenever anything can't be resolved for real: no
    line_id, line not in the DB, line not owned by the bound customer, or
    the line's plan missing -- sync_tools() itself raises on the last two,
    and a raise inside Environment.get_response's post-call sync would turn
    every later tool call into an error, so this refuses rather than risk
    that."""
    object_bindings = bound_plan.get("object_bindings") or {}
    arguments = (bound_plan.get("operation_argument_fact_bundle") or {}).get("arguments") or {}
    line_id = object_bindings.get("line_id") or arguments.get("line_id")
    customer_id = object_bindings.get("customer_id") or arguments.get("customer_id")
    if not line_id or agent_db is None:
        return None
    # One real exception to "the bound line_id": when the user's own opening
    # message (the hand-verified _INITIAL_MESSAGE_OVERRIDE, or the dialogue
    # contract's) explicitly names exactly one real line of the SAME bound
    # customer, that named line IS the line the scenario is about. Real case:
    # telecom_055_order#b0 -- its unconstrained fact bundle defaults to
    # L1001, but its opening message (section 47/106) is "my line 555-123-
    # 2003 got suspended", i.e. L1003 (the only real mismatch among all 147
    # plans; telecom_027_arg#b0/telecom_090_order#b0 name their bound line).
    opening = _INITIAL_MESSAGE_OVERRIDE.get(
        _override_branch_key(bound_plan),
        (bound_plan.get("dialogue_contract") or {}).get("initial_user_message") or "",
    )
    owner_of_bound = next((item for item in agent_db.customers if line_id in item.line_ids), None)
    if owner_of_bound is not None and opening:
        named_phones = set(re.findall(r"\b\d{3}-\d{3}-\d{4}\b", opening))
        named_ids = set(_LINE_ID_TOKEN_RE.findall(opening))
        named = [
            item for item in agent_db.lines
            if item.line_id in owner_of_bound.line_ids
            and (item.phone_number in named_phones or item.line_id in named_ids)
        ]
        if len(named) == 1:
            line_id = named[0].line_id
    line = next((item for item in agent_db.lines if item.line_id == line_id), None)
    if line is None or not line.phone_number:
        return None
    owner = next((item for item in agent_db.customers if line_id in item.line_ids), None)
    if owner is None or (customer_id and owner.customer_id != customer_id):
        return None
    if not any(plan.plan_id == line.plan_id for plan in agent_db.plans):
        return None
    return {"name": owner.full_name, "phone_number": line.phone_number}


def _telecom_scenario_agent_db(bound_plan: Mapping[str, Any]) -> Any:
    environment = load_default_tau_environment("telecom")
    patch = (bound_plan.get("initial_state_patch") or {}).get("agent_data")
    if patch:
        environment.tools.update_db(patch)
    return environment.tools.db


def materialize_generic_tau_task(
    bound_plan: Mapping[str, Any], *, domain: str = "airline", agent_db: Any = None,
) -> dict[str, Any]:
    known = _render_known_facts(bound_plan["object_bindings"])
    branch_id = bound_plan.get("source_branch_id")
    # section 230: a targeted variant plan may carry its own real-device user
    # simulator setup (same shape as _TELECOM_USER_SIMULATOR_BRANCHES entries);
    # absent for every production plan, so nothing changes for them.
    user_simulator_override = (bound_plan.get("telecom_user_simulator")
                               or _TELECOM_USER_SIMULATOR_BRANCHES.get(branch_id))
    # docs/agentcoveragetesting_reuse_log.md section 175 (task_e2474c72): every
    # telecom branch now gets tau2's own set_user_info() initialization action
    # (the same real EnvFunctionCall channel _TELECOM_USER_SIMULATOR_BRANCHES
    # already uses, never the destructive initialization_data path documented
    # below), so TelecomEnvironment.sync_tools() genuinely activates. The 2
    # _TELECOM_USER_SIMULATOR_BRANCHES keep their own hand-written actions
    # verbatim -- they already call set_user_info with the identical real
    # value (L1001 -> 555-123-2001, John Smith), i.e. sync was already live
    # for exactly those 2 and nothing changes for them.
    initialization_actions = (
        user_simulator_override["initialization_actions"]
        if user_simulator_override is not None else None
    )
    if domain == "telecom" and user_simulator_override is None:
        user_info = _telecom_scenario_user_info(
            bound_plan, agent_db if agent_db is not None else _telecom_scenario_agent_db(bound_plan)
        )
        if user_info is not None:
            initialization_actions = [
                {"env_type": "user", "func_name": "set_user_info", "arguments": user_info},
            ]
    return {
        "id": f"generic::{bound_plan['source_branch_id']}",
        "description": {
            "purpose": "Execute one compiled AgentSpecTesting branch with a deterministic Driver.",
            "relevant_policies": None,
            "notes": "The deterministic Driver, not an LLM user simulator, controls user messages.",
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "domain": domain,
                # section 182: same opening request the scripted user sends
                # (override/contract request + conversation statements).
                "reason_for_call": _bound_initial_user_message(bound_plan),
                "known_info": (
                    user_simulator_override["known_info"] if user_simulator_override is not None
                    else (known or None)
                ),
                "unknown_info": None,
                # section 106: for the 2 real branches in
                # _TELECOM_USER_SIMULATOR_BRANCHES, this task is actually
                # consumed by a real tau2 UserSimulator (run_generic_tau_
                # online_plan branches on source_branch_id), which reads
                # this field as real persona/situation content -- not
                # transport metadata, unlike every other branch.
                "task_instructions": (
                    user_simulator_override["instructions"] if user_simulator_override is not None
                    else "Transport metadata only; use the deterministic Driver messages."
                ),
            },
        },
        "initial_state": {
            # The bound plan's initial_state_patch.agent_data is already
            # applied directly via environment.tools.update_db(patch) before
            # the orchestrator is built (run_generic_tau_online_plan). Do not
            # ALSO embed it here as Task.initial_state.initialization_data --
            # tau2's own Environment.set_state (real telecom-specific
            # behavior, environment.py's set_state -> sync_tools) reassigns
            # self.user_tools.db = self.tools.db as part of applying it,
            # which destroys telecom's real distinct user-facing DB (the only
            # one carrying a real "surroundings" field) and crashes every
            # branch with a non-empty patch (see
            # docs/agentcoveragetesting_reuse_log.md section 68). Airline/
            # retail never hit this because their user_tools is None, so the
            # reassignment is a harmless no-op there -- but this is a real
            # correctness fix regardless of domain, not a telecom-only patch:
            # the value was always redundant with the update_db call above.
            "initialization_data": None,
            # section 106: initialization_actions (env_type="user"
            # EnvFunctionCall entries) is a real, SEPARATE channel from
            # initialization_data.user_data above -- Environment.set_state
            # applies it via run_env_function_call (just calls the named
            # real method), never through the destructive self.tools.db =
            # self.user_tools.db reassignment path documented above. Only
            # populated for the 2 branches in _TELECOM_USER_SIMULATOR_
            # BRANCHES; every other branch keeps this None, unchanged.
            # (section 175: superseded for telecom -- see
            # initialization_actions' own computation above.)
            "initialization_actions": initialization_actions,
            "message_history": None,
        },
        "evaluation_criteria": {
            "actions": [],
            "communicate_info": [],
            "nl_assertions": [],
            "reward_basis": [],
        },
        "annotations": None,
    }


def _capture_state(environment: Any, bound_plan: Mapping[str, Any]) -> dict[str, Any]:
    bindings = bound_plan["object_bindings"]
    result: dict[str, Any] = {"db_hash": environment.get_db_hash()}
    reservation_id = bindings.get("reservation_id")
    if not reservation_id:
        return result
    reservation = environment.tools.get_reservation_details(reservation_id)
    value = reservation.model_dump(mode="json")
    result["reservation"] = value
    result["flight_statuses"] = [
        {
            "flight_number": item["flight_number"],
            "date": item["date"],
            "status": environment.tools.get_flight_status(item["flight_number"], item["date"]),
        }
        for item in value.get("flights") or []
    ]
    return result


def _transcript_excerpt_for_judge(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Real, full-conversation excerpt for a judge with no single anchor event
    to work from (see _evaluate_semantic_event_matcher_deferred). Mirrors
    oracle_evaluator_contract_extension_v1._extract_semantic_transcript_
    observations's own {role, content} shape for user/assistant messages, and
    additionally surfaces real tool calls (name + arguments) -- a claim like
    "guide them to use toggle_roaming()" or "the agent must not state X" can
    be about a real tool call the agent made, not only its message text.

    docs/agentcoveragetesting_reuse_log.md section 142 (task_84b94fa8,
    airline_078_norm#b0v0): real, confirmed evidence-construction bug --
    same class and same root cause as section 134 bug 3
    (oracle_evaluator_contract_extension_v1._semantic_transcript_excerpt_
    up_to, telecom_090_order#b0), just never ported to this sibling
    function. This excerpt categorically dropped every real tool_result
    event, so a claim_source_support-style requirement_text ("... not
    provided by the user or shown in any tool result") was judged against a
    transcript that could NEVER contain a tool result, regardless of what
    the real conversation actually contained -- structurally guaranteeing
    the judge can never see the very evidence category (prior_tool_result)
    that configs/semantic_judge_profiles/tau_bench_open_semantics_v0_1.json
    explicitly lists as an allowed grounding source for this rule family.
    Confirmed real: airline_078_norm#b0v0's real message 6 is a genuine
    get_reservation_details tool return containing flight_number/date/
    passenger fields the assistant's real message 7 verbatim repeats back,
    yet the judge's real judge_reason claimed none of that data appeared
    "in any tool result" -- literally true of what this function showed it,
    false of the real transcript. Now includes tool_result (role=tool,
    content=the real return value), the same shape section 134 bug 3 added
    to its sibling extractor -- strictly a widening of what the judge is
    shown, never a narrowing, so no check that used to see a sufficient
    excerpt can regress."""

    excerpt = []
    for event in events:
        kind = event.get("event_kind")
        if kind in ("user_message", "assistant_message"):
            excerpt.append({"role": kind.removesuffix("_message"), "content": event.get("content")})
        elif kind == "assistant_tool_call":
            excerpt.append(
                {
                    "role": "assistant",
                    "tool_call": {"name": event.get("tool_name"), "arguments": event.get("arguments")},
                }
            )
        elif kind == "tool_result":
            excerpt.append({"role": "tool", "content": event.get("content")})
    return excerpt


# Added later (docs/agentcoveragetesting_reuse_log.md section 80): Category
# E, semantic-judge lane -- these 2 requirements' own real When is
# conditioned on a specific real tool call actually firing (gated by real
# business logic that legitimately doesn't always hold), but this
# evaluator's judge call has no mechanism analogous to the mechanical lane's
# requires_observation flag to skip asking when the precondition never
# happened. See _evaluate_semantic_event_matcher_deferred's own use of this
# for the real evidence and scoping rationale.
_SEMANTIC_JUDGE_VACUOUS_UNLESS_TOOL_CALLED = {
    "telecom_043_order#b0::OR01": "send_payment_request",
    "telecom_044_norm#b0::OR01": "send_payment_request",
}


# Section 134 bug 1 (telecom_035_norm#b0v0, docs/agentcoveragetesting_reuse_
# log.md section 131.2/131.6/134.1): the generic "no-fabrication" deferred
# requirement_text ("... information, knowledge, or procedures that were not
# provided by the user or are not available through tools") is the runtime
# copy of the SAME requirement class the compiled semantic-judge-contract
# pipeline calls "claim_source_support"
# (configs/semantic_judge_profiles/tau_bench_open_semantics_v0_1.json's
# "claim_source_support" rule) -- that profile explicitly lists 4
# allowed_source_channels: prior_user_message, prior_tool_result,
# system_policy, agent_tool_schema. The compiled claim_source_support/
# claim_atomization pipeline (compiler/semantic_judge_contract_v1.py) reads
# and honors those channels; this deferred, no-anchor lane (no compiled
# contract exists for these 2 telecom requirements -- diagnostics say
# no_exact_runtime_anchor) never did -- it asked the judge a bare yes/no
# question with no source-channel guidance at all. Confirmed real, materially
# identical assistant turns (the agent restating its own supported-categories
# menu -- technical support / overdue bill payment / line suspension / plan
# options -- derived from its own system policy, not from the user or a tool
# result) got a real "pass" in one online run and, per the section 131.2/
# 131.6 fresh-run report, a real "fail" in another: a system-policy-sourced
# self-description is legitimately allowed under allowed_source_channels, but
# nothing in the old prompt ever told the judge that "system_policy" and
# "agent_tool_schema" count as available sources, so a literal reading of
# "not provided by the user or available through tools" could reasonably (and
# inconsistently, run to run) flag the agent's own stated scope as a
# violation. Matches only the generic claim-source-support wording (the sole
# telecom deferred requirement with this shape is telecom_035_norm#b0v0::
# OR01) -- every other deferred requirement_text in the corpus is a
# domain-specific "must/must not state X" or "must guide to use tool Y" rule
# that was never ambiguous about sourcing and must not receive this hint.
#
# docs/agentcoveragetesting_reuse_log.md section 152.2 (task_a7bf981a, found
# by section 150.5.1/150.5.2's real online verification of retail_056_norm#
# b0v0 pass4): the SAME real defect class recurred under a real SIBLING
# requirement_text wording this pattern did not cover. Real pass4 message
# 12 -- "Under our policy, I can only take action (cancel, modify, return,
# or exchange) on orders that are pending or delivered. A 'processed'
# order isn't eligible for any of those actions..." -- is verbatim-correct
# real tau2 retail policy (<TAU2_BENCH_DIR>/data/
# tau2/domains/retail/policy.md line 82: "Generally, you can only take
# action on pending or delivered orders."), a legitimate system_policy-
# sourced restatement, not fabrication -- but the check's real
# requirement_text is just "fabricate information, knowledge, or
# procedures" (no "provided by the user ... available ... tools" clause at
# all), so the pattern above never matched and the judge got no
# source-channel guidance, real-confirmed to fail with judge_reason
# "...without any supporting tool result or source in the transcript,
# which appears to be fabricated procedure."
#
# A real corpus-wide scan of every distinct requirement_text on every
# semantic_judge-mode check across all 3 domains' real plans.json files
# (retail 185 branches / telecom 147 / airline 153; retail contributes 10
# distinct requirement_text strings, telecom 31, airline 10) found exactly
# 3 checks anywhere in the corpus sharing this exact "fabricate
# information, knowledge, or procedures" text -- all 3 retail
# (retail_056_norm's #b0v0/#b0v1/#b0v2 siblings), 0 in telecom/airline --
# and confirmed (via the same scan, re-run against the ALREADY-widened
# pattern below) that telecom's own no-fabrication wording ("...information,
# knowledge, or procedures that were not provided by the user or are not
# available through tools") and airline's own no-fabrication wording
# ("...information, knowledge, or procedures not provided by the user or
# available tools") BOTH already matched the original pattern above before
# this change (verified directly: `_CLAIM_SOURCE_SUPPORT_PATTERN.search(...)`
# on each real string returns a match pre-fix) -- so this is not a
# rediscovery of the section 134 bug, it is a genuinely new, narrower gap:
# retail's variant is worded as a bare imperative ("fabricate information,
# knowledge, or procedures") with no "provided by ... available ... tools"
# qualifying clause at all, so no widening of the ORIGINAL alternative
# could ever have matched it -- it needs its own alternative. The same
# corpus scan, searched for any OTHER requirement_text anywhere in the 3
# domains expressing a similar no-fabrication/no-unsupported-claim intent
# via different wording (fabricat*, invent*, made-up, guess*, assum*,
# unsupported, unverified, no basis, not based on, hallucinat*, "not
# provided by", "available ... tools", "knowledge, or procedures") found
# NO further uncovered variant -- these 3 wordings (telecom's, airline's,
# retail's) are the complete real set. Given that, this is fixed as a
# second, narrowly-scoped literal alternative (matching this pattern's own
# established style, not a broad semantic-intent rewrite) rather than a
# general "match by meaning" mechanism: a broader semantic mechanism would
# be solving a problem the real corpus does not currently have (zero
# further uncovered variants found), at the real cost of being harder to
# reason about and regression-test than two precise, corpus-verified
# literal alternatives -- if a 4th real wording is ever found, it should be
# added the same way, backed by the same kind of real evidence.
_CLAIM_SOURCE_SUPPORT_PATTERN = re.compile(
    r"provided by the user.{0,40}available (through )?tools"
    r"|fabricat\w*.{0,40}information,?\s*knowledge,?\s*or\s*procedures",
    re.IGNORECASE,
)


def _claim_source_support_channel_hint(
    requirement_text: str | None, *, domain_policy_available: bool = False,
) -> str | None:
    if not requirement_text or not _CLAIM_SOURCE_SUPPORT_PATTERN.search(requirement_text):
        return None
    if domain_policy_available:
        # docs/agentcoveragetesting_reuse_log.md section 172: the judge now
        # really sees the domain policy (see _domain_policy_evidence_block),
        # so system_policy is no longer limited to the assistant restating
        # its own supported-request scope -- any statement actually found in
        # (or faithfully paraphrased from) that verbatim text is grounded.
        # The no-policy wording below is kept byte-identical for every
        # caller that does not supply the policy.
        return (
            " For this rule specifically, \"provided by the user\" or \"available "
            "through tools\" is not limited to text the user typed or a literal "
            "tool call result -- the allowed sources for the assistant's "
            "statements are: prior_user_message, prior_tool_result, "
            "system_policy (anything actually stated in, or faithfully "
            "paraphrased from, the DOMAIN POLICY text shown below, which the "
            "assistant received verbatim as its system instructions -- this "
            "covers both its declared scope of supported requests/categories "
            "AND substantive policy content such as fees, prices, timelines, "
            "eligibility conditions, and procedures), and agent_tool_schema "
            "(facts about what a tool does or requires, drawn from the tool's "
            "own name/schema/description rather than a tool call's return "
            "value). Content grounded in system_policy or agent_tool_schema "
            "does NOT count as unsupported, fabricated, or outside information. "
            "Before flagging a statement as unsupported, check it against the "
            "DOMAIN POLICY text itself. A specific fact, number, amount, fee, "
            "timeline, condition, or procedure that does NOT appear in (and "
            "cannot be faithfully derived from) the DOMAIN POLICY text, a prior "
            "tool result, the tool schema, or the user's own messages is still "
            "unsupported and IS a violation -- sounding like a plausible policy "
            "is not enough, and a value that differs from what the policy "
            "actually says (e.g. a different number of days, a different fee) "
            "is a violation."
        )
    return (
        " For this rule specifically, \"provided by the user\" or \"available "
        "through tools\" is not limited to text the user typed or a literal "
        "tool call result -- the allowed sources for the assistant's "
        "statements are: prior_user_message, prior_tool_result, "
        "system_policy (the assistant restating or paraphrasing its own "
        "declared scope of supported requests/categories -- what it is and "
        "is not able to help with, per its own system instructions), and "
        "agent_tool_schema (facts about what a tool does or requires, drawn "
        "from the tool's own name/schema/description rather than a tool "
        "call's return value). Content grounded in system_policy or "
        "agent_tool_schema does NOT count as unsupported, fabricated, or "
        "outside information -- only flag a violation for content that is a "
        "substantive external fact, procedure, or recommendation the "
        "assistant could not have grounded in any of these 4 sources."
    )


# docs/agentcoveragetesting_reuse_log.md section 153.5 (task_200b51ad, found by
# section 153.5's real online verification of retail_056_norm#b0v3 pass5):
# real, confirmed judge instability on the "neutral clarification vs.
# subjective comment" boundary for this deferred lane's generic "give
# subjective recommendations or comments" rule (configs/semantic_judge_
# profiles/tau_bench_open_semantics_v0_1.json's own "subjective_expression"
# rule_id/judge_kind names this exact requirement_text, but that compiled
# judge_kind pipeline (compiler/semantic_judge_contract_v1.py +
# driver_plan_lowering_v1.py) is a SEPARATE, not-currently-exercised-at-
# runtime architecture from this deferred lane -- every real online run in
# this corpus routes semantic_judge-mode checks through THIS module's own
# generic "must/must not" criterion builder instead, which had no rule-
# specific grounding for this shape at all, the same structural gap
# _claim_source_support_channel_hint above was added to close for the
# no-fabrication shape).
#
# Two real, independent online generations of the SAME check
# (retail_056_norm#b0v3::OR01) on the SAME judge/model/policy produced
# opposite verdicts for functionally identical assistant behavior: pass4's
# real message used wording like "neutral clarification and procedural
# options" (the judge's own real judge_reason quote) and correctly passed;
# pass5's real message -- "I'm not sure I understand exactly what you'd
# like me to do -- your request is a bit general. I can't take any action
# ... without knowing the specific request and getting your confirmation.
# Could you please tell me specifically what you'd like?" -- is, in
# substance, the same kind of ordinary disambiguation/clarification request
# this project's own driver logic deliberately generates (see
# _assistant_message_needs_disambiguation_response), not a subjective
# opinion or recommendation about the user's situation, a product, or a
# decision -- but the SAME judge flagged it, with a real judge_reason of
# "The assistant made a subjective comment by saying the user's request was
# 'a bit general' and that it was not sure it understood." A literal
# reading of the bare requirement_text plausibly treats ANY evaluative-
# sounding wording (even a remark about how clear the user's OWN request
# was) as a "comment" -- confirmed against the real tau2 retail policy
# (policy.md: "You should not make up any information or knowledge or
# procedures not provided by the user or the tools, or give subjective
# recommendations or comments"), whose real intent is plainly about not
# opining on products/services/decisions, not about a meta-level remark on
# request ambiguity.
#
# Real corpus-wide scan (all 3 domains' real plans.json files, every
# distinct requirement_text on every semantic_judge-mode check) found this
# EXACT literal text ("give subjective recommendations or comments") on
# exactly 3 checks total -- one per domain, all operator="absent":
# retail_056_norm#b0v3::OR01, telecom_035_norm#b0v1::OR01, airline_078_norm#
# b0v1::OR01 -- no other wording anywhere in the corpus expresses this same
# no-subjective-commentary intent. All 3 share this SAME shared module/
# function, so this fix (like _claim_source_support_channel_hint above)
# does not require touching any telecom- or airline-specific file: adding
# the grounding hint here benefits all 3 real checks uniformly, the same
# "one shared fix, corpus-verified blast radius" discipline section 152
# established.
_SUBJECTIVE_EXPRESSION_PATTERN = re.compile(
    r"subjective (recommendations?|comments?)",
    re.IGNORECASE,
)


def _subjective_expression_grounding_hint(requirement_text: str | None) -> str | None:
    if not requirement_text or not _SUBJECTIVE_EXPRESSION_PATTERN.search(requirement_text):
        return None
    return (
        " For this rule specifically, a genuine violation is the assistant "
        "expressing its OWN subjective opinion, preference, evaluative "
        "judgment, or recommendation about the user's situation, a product, "
        "a service, or which option the user should choose -- e.g. \"I "
        "think the blue one looks nicer\", \"I'd recommend the express "
        "shipping\", \"that seems like a bad idea\", \"in my opinion the "
        "morning flight is the best choice\". It is NOT a violation for the "
        "assistant to ask the user to clarify or be more specific because "
        "the user's OWN request was ambiguous or underspecified -- even "
        "when that clarification question uses evaluative-sounding words "
        "about the REQUEST ITSELF rather than about the user's situation, a "
        "product, or a decision (e.g. \"your request is a bit general\", "
        "\"I'm not sure I understand exactly what you'd like\", \"could you "
        "be more specific about what you'd like me to do\"). A remark about "
        "how clear, specific, or general the user's OWN message was is a "
        "routine procedural disambiguation question, not a subjective "
        "recommendation or comment -- do not flag it as a violation on that "
        "basis alone. Also not a violation: the assistant declining to give "
        "an opinion (e.g. \"I can't recommend one option, but I can list "
        "the choices\") or stating a plain, neutral fact drawn from a tool "
        "result or system policy."
    )


# docs/agentcoveragetesting_reuse_log.md section 172 (section 171.3's
# infrastructure bug, confirmed real on airline_078_norm#b0v0 pass8): every
# semantic-judge call in this runtime -- this module's deferred lane AND the
# compiled semantic_transcript_judgment / declined-alternative lane in
# oracle_evaluator_contract_extension_v1.py, which receives the SAME judge
# callable -- was judged against a transcript excerpt that never contains the
# domain policy the agent itself received as its system prompt
# (create_llm_agent(domain_policy=environment.get_policy())). Real pass8
# message: "Refund: $4,986 will be returned to your original payment method
# ... within 5 to 7 business days" -- verbatim tau2 airline policy.md line
# 152 ("The refund will go to original payment methods within 5 to 7
# business days.") -- was failed as "not supported by ... the available tool
# schema/policy in the transcript". Literally true of what the judge was
# shown, false of what the agent was given. The section 131.6/134
# _claim_source_support_channel_hint only told the judge system_policy
# EXISTS as a source (and only as "restating its own declared scope"); it
# never gave the judge the policy itself, so a policy fact could not be
# verified either way.
#
# Scope (which judge calls get the policy): only rules whose verdict
# genuinely depends on what the policy says -- (1) the claim-source-support
# ("not provided by the user or available tools" / "fabricate ...") rule,
# whose allowed sources explicitly include system_policy; (2) the
# subjective-expression rule, whose hint explicitly exempts "a plain,
# neutral fact drawn from ... system policy"; (3) any rule/criterion whose
# own text refers the judge to the policy or the workflow document ("...
# timing that contradicts the policy", "... against the current instruction
# policy set", "policy-permitted actions", "tech_support_workflow.md steps")
# -- a judge asked to compare against a document it cannot see can only
# guess. Every other judge call (the "must/must not state <explicit fact>"
# and "must guide the user to <tool>" rules, which carry their own facts in
# the rule text) keeps a byte-identical prompt, so none of them can regress.
# Real corpus scan (section 172): 11 of 50 deferred checks and 4 of 19
# compiled criterion checks (retail_089_state#b0/#b1, retail_101_order#b0v0,
# telecom_113_order#b0) across all 3 production plans.json files match.
#
# Policy text: the FULL text returned by environment.get_policy() -- the
# exact string the agent got -- never a subset. A subset (e.g. the compiled
# policy_semantic_unit_model.json) is exactly what produced section 151's
# wrong conclusion that "5 to 7 business days" is not airline policy: the
# semantic-unit extraction had dropped that sentence. Size is bounded and
# small next to what the agent itself consumes: airline 7,676 chars / retail
# 6,699 / telecom 23,318 (main_policy + tech_support workflow), i.e. at most
# roughly 6K tokens added to a judge call on a gated rule, while the agent
# receives the same text on every one of its own calls.
_POLICY_REFERENCE_PATTERN = re.compile(r"\bpolic(y|ies)\b|workflow", re.IGNORECASE)


def _requires_domain_policy_evidence(rule_text: str | None) -> bool:
    if not rule_text:
        return False
    return bool(
        _CLAIM_SOURCE_SUPPORT_PATTERN.search(rule_text)
        or _SUBJECTIVE_EXPRESSION_PATTERN.search(rule_text)
        or _POLICY_REFERENCE_PATTERN.search(rule_text)
    )


def _domain_policy_evidence_block(domain_policy: str) -> str:
    return (
        "\n\nDOMAIN POLICY: the text between the <domain_policy> tags below is "
        "the exact, complete domain policy the assistant was given as its "
        "system instructions before this conversation began. It is not part "
        "of the transcript excerpt, but it IS information the assistant had "
        "available. Use it to decide whether something the assistant said is "
        "grounded in its own policy, and to interpret any reference the rule "
        "makes to \"the policy\". Only content that actually appears in (or is "
        "a faithful paraphrase of) this text counts as policy-grounded -- do "
        "not treat a claim as policy-grounded merely because it sounds like a "
        "plausible policy.\n<domain_policy>\n"
        f"{domain_policy}\n</domain_policy>"
    )


def _with_domain_policy_evidence(semantic_judge: Any, domain_policy: str | None) -> Any:
    """Wrap a judge callable for the compiled semantic_transcript_judgment /
    declined-alternative lane (oracle_evaluator_contract_extension_v1.py),
    which builds its own criterion from compiled plan text and has no policy
    parameter of its own. Appends the same policy evidence block the deferred
    lane uses, gated by the same _requires_domain_policy_evidence rule; every
    other criterion passes through unchanged. Done here rather than in the
    compiler module so the runtime-only policy string stays a runtime concern
    (section 172)."""

    if semantic_judge is None or not domain_policy:
        return semantic_judge

    def judge(*, criterion, target_value, transcript_excerpt):
        if isinstance(criterion, str) and _requires_domain_policy_evidence(criterion):
            criterion = f"{criterion}{_domain_policy_evidence_block(domain_policy)}"
        return semantic_judge(
            criterion=criterion, target_value=target_value, transcript_excerpt=transcript_excerpt,
        )

    return judge


def _evaluate_semantic_event_matcher_deferred(
    check: Mapping[str, Any], events: list[Mapping[str, Any]], semantic_judge: Any,
    *, termination_reason: str | None = None, domain_policy: str | None = None,
) -> dict[str, Any]:
    """Real judge-backed evaluation for a check whose WHOLE evaluation_mode is
    semantic_judge (docs/agentcoveragetesting_reuse_log.md section 70) --
    unlike the mechanical-mode semantic_transcript_judgment predicate (which
    already has a real anchor tool_call/argument to project, handled by
    evaluate_oracle_contract's own semantic_judge param), this category's
    runtime_observation_binding never resolved to any concrete anchor at all
    (binding_status: semantic_deferred, diagnostics: no_exact_runtime_anchor)
    -- there is only a real, free-text requirement_text describing a claim
    about the agent's behavior somewhere in the conversation, and an
    expected_observation.operator saying whether that claim should (present)
    or should not (absent) hold. Asks the judge a single, direct yes/no
    question against the FULL real transcript, then applies operator
    polarity locally -- the judge is never asked about polarity itself, to
    keep its question unambiguous and mirror the semantic_transcript_
    judgment convention of "passed" meaning "the claim is true".

    docs/agentcoveragetesting_reuse_log.md section 136.5/141 (task_b852c7cb,
    telecom_065_order#b0): real, confirmed premature-termination false-
    negative on this lane, the same SHAPE as section 134 bug 4's mechanical
    missing_required_observation fix (oracle_evaluator_contract_extension_
    v1.evaluate_oracle_contract_extended's termination_reason kwarg) but NOT
    the same fix -- a corpus-wide scan (this ticket, section 141) found 20
    real telecom operator="present" semantic_judge checks, and a real
    cross-reference against section 136's own full-corpus online results
    found 8 of them really failed with a non-agent_stop termination_reason.
    Blindly copying bug 4's blanket rule (soften whenever missing/
    termination_reason != agent_stop) onto this lane would have been WRONG:
    real transcript review of all 8 showed 7 of them (telecom_078_order#b1/
    b2/b4/b6, telecom_083_order#b1/b2/b5) are genuine agent bugs -- the agent
    DID already reach and engage the relevant situation (it performed the
    fix action) and simply omitted a required follow-up instruction -- not
    cases where the conversation ended before the agent had any real chance.
    Only telecom_065_order#b0 (agent never even reached the VPN topic before
    a real user_error cut the conversation short at 4/8 calls) is the true
    premature-termination shape. Unlike the mechanical lane, this lane has
    no concrete anchor event to check "was it ever observed at all" against
    -- only the judge itself, having read the full transcript, can tell
    "the assistant already engaged with this situation but fell short" apart
    from "this situation never came up before the conversation ended". So
    the criterion below now asks the SAME judge call (no extra API cost) for
    one additional field, never_had_opportunity, and softening only applies
    when the judge itself says the situation never arose -- never from
    termination_reason alone. This can only ever turn a real fail into
    "unavailable" (never into "pass"), exactly mirroring bug 4's own
    contract, and only for operator="present" (an operator="absent"
    violation is, by construction, something that DID happen -- premature
    termination can only make a prohibited act LESS likely to be observed,
    never manufacture a false one, so there is no analogous false-negative
    shape to soften there)."""

    runtime_binding = (check["runtime_observation_binding"] or {}).get("runtime_binding") or {}
    requirement_text = runtime_binding.get("requirement_text")
    operator = check["evaluator_contract"]["expected_observation"]["operator"]
    if not requirement_text or operator not in ("present", "absent"):
        return {
            "verdict": "unavailable",
            "observation": {"status": "not_observed", "reason": "no requirement_text or unsupported operator"},
        }
    requirement_id = check["evaluator_contract"].get("requirement_id")
    trigger_tool = _SEMANTIC_JUDGE_VACUOUS_UNLESS_TOOL_CALLED.get(requirement_id)
    if trigger_tool is not None and not any(
        event.get("event_kind") == "assistant_tool_call" and event.get("tool_name") == trigger_tool
        for event in events
    ):
        # Category E, semantic-judge lane (docs/agentcoveragetesting_reuse_
        # log.md section 80): this requirement's own real When is
        # conditioned on a specific real tool call actually firing
        # (telecom_043/044's own rule_text names send_payment_request by
        # name -- gated by real business logic, bill must be Overdue) --
        # but this evaluator asks the judge a literal, unconditional
        # compliance question against the whole transcript regardless of
        # whether that precondition ever became true, unlike the mechanical
        # all_matches_predicate lane's own requires_observation flag (see
        # _REQUIRES_OBSERVATION_FALSE_OVERRIDE in
        # oracle_evaluator_contract_extension_v1.py, the same real defect
        # class, different evaluation lane). Confirmed via real transcript:
        # the agent correctly determined no payment request was legitimately
        # sendable and never got the chance to satisfy (or violate) the
        # Then, and was wrongly scored fail by asking the judge anyway.
        # Scoped to these 2 confirmed real cases, not a general "skip the
        # judge whenever a named tool wasn't called" mechanism.
        return {
            "verdict": "pass",
            "observation": {
                "status": "vacuous",
                "reason": f"trigger tool {trigger_tool!r} was never called; the Then's precondition never held",
            },
        }
    transcript_excerpt = _transcript_excerpt_for_judge(events)
    # section 172: see _requires_domain_policy_evidence. domain_policy=None
    # (hand-built executions, tests, any legacy caller) keeps the pre-172
    # criterion byte-identical.
    include_domain_policy = bool(domain_policy) and _requires_domain_policy_evidence(requirement_text)
    source_channel_hint = _claim_source_support_channel_hint(
        requirement_text, domain_policy_available=include_domain_policy,
    )
    subjective_expression_hint = _subjective_expression_grounding_hint(requirement_text)
    # docs/agentcoveragetesting_reuse_log.md section 71: requirement_text is
    # carried through unmodified from candidate generation and is NOT
    # consistently a bare descriptive claim -- some entries are a plain
    # topical phrase ("costs $30 per passenger"), others are already a full
    # normative sentence with their own polarity baked in ("The agent must
    # not state that the price ... is not 50 dollars"). The original design
    # here asked the judge a bare "is this claim true" question and then
    # combined that with `operator` locally in Python; for the
    # already-normative entries that double-applies polarity (operator=
    # absent on top of a requirement_text that already says "must not"),
    # producing an inverted verdict even though the judge's own stated
    # reasoning was correct. Real, observed bug (airline_047/airline_092),
    # not a hypothetical. Fixed by asking the judge one self-contained
    # compliance question per the ACTUAL rule -- built from requirement_text
    # + operator together -- rather than asking about a bare claim and
    # re-applying polarity afterward; `passed` is used as the verdict
    # directly, with no further polarity math on this side.
    if operator == "present":
        rule = (
            "The following must be true of the ASSISTANT's behavior in this "
            f"conversation: {requirement_text!r}. (If this text is itself "
            "already phrased as an instruction to the assistant, e.g. \"the "
            "agent must ...\", treat it as the rule verbatim -- do not add "
            "another layer of negation.)"
        )
    else:
        rule = (
            "The following must NOT be true of the ASSISTANT's behavior in "
            f"this conversation -- it must not happen: {requirement_text!r}. "
            "(If this text is itself already phrased as a prohibition, e.g. "
            "\"the agent must not ...\", that prohibition IS the rule -- "
            "judge whether the assistant obeyed it. Do not add another layer "
            "of negation on top of a requirement_text that already says "
            "\"must not\".)"
        )
    if source_channel_hint:
        rule = f"{rule}{source_channel_hint}"
    if subjective_expression_hint:
        rule = f"{rule}{subjective_expression_hint}"
    if include_domain_policy:
        rule = f"{rule}{_domain_policy_evidence_block(domain_policy)}"
    result = semantic_judge(
        criterion=(
            f"{rule}\n\nBased only on what the ASSISTANT actually said or "
            "did (its messages and tool calls, never what the user said or "
            "asked for), did the assistant's real behavior in this "
            "transcript COMPLY with the rule above? Return passed=true if "
            "it complied, passed=false if it violated the rule.\n\n"
            "Also answer one more question, independent of the verdict "
            "above: never_had_opportunity=true only if the conversation "
            "ended (for any reason) before the situation, topic, or "
            "precondition this rule is about ever genuinely came up or "
            "became actionable -- i.e. the assistant never had a real "
            "chance to address it at all. Set never_had_opportunity=false "
            "if the assistant already started engaging with the relevant "
            "situation (even partially, even imperfectly, even if what it "
            "did was incomplete or in the wrong order) -- never_had_"
            "opportunity is NOT for \"the assistant's response fell short\" "
            "or \"missed a follow-up step\", only for \"this never came up "
            "at all before the conversation ended\". Return ONLY a single "
            "JSON object: {\"passed\": true or false, \"reason\": \"...\", "
            "\"never_had_opportunity\": true or false}."
        ),
        target_value=None,
        transcript_excerpt=transcript_excerpt,
    )
    passed = bool(result.get("passed"))
    evaluation: dict[str, Any] = {
        "verdict": "pass" if passed else "fail",
        "observation": {
            "status": "observed",
            "operator": operator,
            "judge_reason": result.get("reason"),
        },
    }
    if include_domain_policy:
        evaluation["observation"]["domain_policy_evidence"] = {
            "included": True,
            "sha256": hashlib.sha256(domain_policy.encode("utf-8")).hexdigest(),
            "chars": len(domain_policy),
        }
    if (
        not passed
        and operator == "present"
        and termination_reason is not None
        and termination_reason != "agent_stop"
        and bool(result.get("never_had_opportunity"))
    ):
        evaluation["verdict"] = "unavailable"
        evaluation["reason"] = "requires_observation_unmet_due_to_premature_termination"
        evaluation["premature_termination_reason"] = termination_reason
    return evaluation


def load_reference_tables(domain: str, *, root: Path | None = None) -> dict[str, Any]:
    """Real static reference tables a handful of predicates need beyond the
    observed event stream (docs/agentcoveragetesting_reuse_log.md section
    70): flight_route_reference/flight_schedule_reference (airline) and
    retail_user_directory_reference (retail), built by
    scripts/prepare_v5_step4_flight_route_reference_v0_1.py,
    scripts/prepare_v5_step4_flight_schedule_reference_v0_1.py, and
    scripts/prepare_v5_step4_retail_user_directory_reference_v0_1.py at
    their own default output paths. These are real, static artifacts (a
    projection of tau2's own fixture database) that never change unless the
    fixture data itself does -- loaded once here rather than threaded in as
    a CLI argument, mirroring how the compiler itself only ever asked for
    them as file paths with real, checked-in defaults. Returns an empty
    dict for a domain with no reference tables (e.g. telecom) -- callers
    unpack with .get(...), so a missing key is the correct "not supplied"
    behavior, not an error.
    """

    root = root or Path(__file__).resolve().parents[3]
    if domain == "airline":
        return {
            "flight_route_reference": json.loads(
                (root / "configs/tau_airline_flight_route_reference_v0_1.json").read_text()
            )["routes"],
            "flight_schedule_reference": json.loads(
                (root / "configs/tau_airline_flight_schedule_reference_v0_1.json").read_text()
            )["schedule"],
        }
    if domain == "retail":
        return {
            "retail_user_directory_reference": json.loads(
                (root / "configs/tau_retail_user_directory_reference_v0_1.json").read_text()
            )["match_counts"],
        }
    return {}


def evaluate_generic_mechanical_oracle(
    bound_plan: Mapping[str, Any], execution: Mapping[str, Any],
    *, semantic_judge: Any = None, reference_tables: Mapping[str, Any] | None = None,
    domain_policy: str | None = None,
) -> dict[str, Any]:
    reference_tables = reference_tables or {}
    events = normalize_tau_execution(execution)
    # section 172: the real domain policy text the agent was given (None keeps
    # every judge prompt byte-identical to pre-172). The compiled-criterion
    # lane below gets it via a gated wrapper around the same judge callable.
    compiled_lane_judge = _with_domain_policy_evidence(semantic_judge, domain_policy)
    # Section 134 bug 4 (telecom_113_order#b0, docs/agentcoveragetesting_
    # reuse_log.md section 131.2/131.6): real, plain string/enum value (tau2's
    # own TerminationReason -- a str-Enum, so comparisons against a plain
    # string like "agent_stop" work either way) threaded through to
    # evaluate_oracle_contract (aliased to oracle_evaluator_contract_
    # extension_v1.evaluate_oracle_contract_extended)'s new termination_reason
    # kwarg -- lets it distinguish "the guarded event was never observed
    # because the conversation was cut off before the agent could plausibly
    # reach it" from "the agent had every real opportunity and never acted"
    # for any check surfacing missing_required_observation=True. None when
    # absent (e.g. a hand-built execution dict in a test), which keeps this
    # wrapper's softening inert -- exactly today's unmodified behavior.
    termination_reason = execution.get("termination_reason")
    records = []
    counts = {"pass": 0, "fail": 0, "pending_semantic_judge": 0, "unavailable": 0}
    for check in bound_plan["oracle_plan"]["checks"]:
        if check["evaluation_mode"] == "semantic_judge":
            # A check whose WHOLE evaluation_mode is semantic_judge (as opposed to
            # a "mechanical" check with a semantic_transcript_judgment predicate
            # inside it, handled by evaluate_oracle_contract's own semantic_judge
            # param below) -- a real, separate, coarser category. Real judge
            # wiring added in docs/agentcoveragetesting_reuse_log.md section 70;
            # honest "pending" fallback kept when no judge is configured.
            if semantic_judge is None:
                verdict = "pending_semantic_judge"
                evaluation = {"reason": "semantic judge requires separately approved model calls"}
            else:
                evaluation = _evaluate_semantic_event_matcher_deferred(
                    check, events, semantic_judge, termination_reason=termination_reason,
                    domain_policy=domain_policy,
                )
                verdict = evaluation["verdict"]
        else:
            evaluation = evaluate_oracle_contract(
                check["evaluator_contract"],
                check["runtime_observation_binding"],
                events,
                bound_plan["oracle_scope_values"],
                semantic_judge=compiled_lane_judge,
                termination_reason=termination_reason,
                **reference_tables,
            )
            verdict = evaluation["verdict"]
        counts[verdict] += 1
        records.append(
            {
                "binding_id": check["binding_id"],
                "evaluation_mode": check["evaluation_mode"],
                "verdict": verdict,
                "evaluation": evaluation,
            }
        )
    # docs/agentcoveragetesting_reuse_log.md section 132.2/135 (real bug,
    # task_5d41846b): a real, confirmed alternate-route shape (sections
    # 62.1-62.2 -- e.g. airline_049_arg#b0's own OR01/OR03 real book_
    # reservation route vs OR02's real update_reservation_flights route,
    # only one is ever really exercised in a live conversation) previously
    # had no OR-semantics at all here -- every check counted independently
    # (AND-style), so the candidate route the live agent genuinely did NOT
    # take hard-failed the whole branch even when every check on the route
    # it DID take genuinely passed. bound_plan["oracle_plan"].get(
    # "alternate_route_groups") (generic_tau_v2_bound_plan_adapter_v1.py's
    # _compute_alternate_route_groups) is additive-only -- an empty list
    # for every branch that isn't a real, precisely-confirmed twin-shaped
    # alternate-route branch (see that function's own docstring for the
    # real, narrow detection rule, deliberately excluding real SEQUENCE
    # shapes like airline_084/085/090_order#b0's own real precedes checks)
    # -- so branch_verdict's computation for every OTHER branch (the
    # overwhelming majority, every domain) is byte-identical to before.
    alternate_route_groups = (bound_plan.get("oracle_plan") or {}).get("alternate_route_groups") or []
    if alternate_route_groups:
        record_by_id = {record["binding_id"]: record for record in records}
        grouped_binding_ids = {binding_id for group in alternate_route_groups for binding_id in group}
        group_verdicts = []
        for group in alternate_route_groups:
            group_counts = {"pass": 0, "fail": 0, "unavailable": 0, "pending_semantic_judge": 0}
            for binding_id in group:
                group_counts[record_by_id[binding_id]["verdict"]] += 1
            if group_counts["fail"]:
                group_verdicts.append("fail")
            elif group_counts["unavailable"] or group_counts["pending_semantic_judge"]:
                group_verdicts.append("incomplete")
            else:
                group_verdicts.append("pass")
        # OR across groups: the underlying requirement is satisfied the
        # moment ANY one real alternate route is genuinely, fully
        # satisfied -- a "fail" on a route the live agent never took
        # (and never needed to) does not, by itself, fail the branch.
        if "pass" in group_verdicts:
            or_verdict = "pass"
        elif "incomplete" in group_verdicts:
            or_verdict = "incomplete"
        else:
            or_verdict = "fail"
        ungrouped_counts = {"pass": 0, "fail": 0, "unavailable": 0, "pending_semantic_judge": 0}
        for record in records:
            if record["binding_id"] not in grouped_binding_ids:
                ungrouped_counts[record["verdict"]] += 1
        # Every ungrouped check (a real requirement that applies regardless
        # of which route was taken) stays hard AND-required, same as today.
        if ungrouped_counts["fail"] or or_verdict == "fail":
            verdict = "fail"
        elif ungrouped_counts["unavailable"] or ungrouped_counts["pending_semantic_judge"] or or_verdict == "incomplete":
            verdict = "incomplete"
        else:
            verdict = "pass"
        result = {
            "branch_verdict": verdict,
            "counts": counts,
            "checks": records,
            "alternate_route_groups_evaluation": {
                "groups": alternate_route_groups,
                "group_verdicts": group_verdicts,
                "resolved_verdict": or_verdict,
            },
        }
    else:
        if counts["fail"]:
            verdict = "fail"
        elif counts["unavailable"] or counts["pending_semantic_judge"]:
            verdict = "incomplete"
        else:
            verdict = "pass"
        result = {"branch_verdict": verdict, "counts": counts, "checks": records}
    result["evaluation_fingerprint"] = content_sha256(result)
    return result


def run_generic_tau_online_plan(
    bound_plan: Mapping[str, Any], *, model: str, approved_agent_call_budget: int,
    seed: int = 0, timeout_seconds: float = 300.0, domain: str = "airline",
    enable_semantic_judge: bool = False,
    user_class: Any = None, user_extra_kwargs: Mapping[str, Any] | None = None,
    message_history: list[Mapping[str, Any]] | None = None, user_factory: Any = None,
    telecom_policy_type: str | None = None,
    agent_llm_args: Mapping[str, Any] | None = None,
    user_model: str | None = None, user_llm_args: Mapping[str, Any] | None = None,
    judge_model: str | None = None, judge_llm_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """user_class/user_extra_kwargs/message_history (section 201) exist for
    the perturbation pilot only: a GenericDeterministicTauUser subclass, extra
    constructor kwargs for it, and a stored message prefix that tau2 replays
    (Environment.set_state re-executes its mutating tool calls) before the
    conversation continues. All default to None, which is the unchanged path.

    telecom_policy_type (section 250): "manual" or "workflow" overrides the
    plan's own field. The policy text belongs to the agent under test, not to
    the case, so the plan (and its fingerprint) stays unchanged.

    section 493 (second underlying model): `model` is the agent under test.
    user_model / judge_model (default: `model`, the unchanged path) let the
    simulated user and the semantic judge stay on another model, and the
    *_llm_args (e.g. api_base / api_key) let each role talk to its own
    OpenAI-compatible provider without changing the process environment."""
    if bound_plan.get("schema_version") != "agentspectesting.generic-bound-driver-plan/v0.1":
        raise GenericTauOnlineError("runner requires GenericBoundDriverPlan v0.1")
    if approved_agent_call_budget < 1:
        raise GenericTauOnlineError("approved target-Agent call budget must be positive")
    from tau2.agent.llm_agent import create_llm_agent
    from tau2.data_model.tasks import Task
    from tau2.orchestrator.orchestrator import Orchestrator, Role
    from tau2.data_model.simulation import TerminationReason

    environment = load_default_tau_environment(domain)
    if telecom_policy_type not in (None, "manual", "workflow"):
        raise GenericTauOnlineError(f"unknown telecom_policy_type {telecom_policy_type!r}")
    if domain == "telecom" and (telecom_policy_type or bound_plan.get("telecom_policy_type")) == "manual":
        # section 231: a targeted variant may run under tau2's default manual
        # tech-support policy (same DB and tools, only the policy text differs);
        # absent for every production plan, which keep the section 203 workflow.
        from tau2.domains.telecom.environment import get_environment as _telecom_environment

        environment = _telecom_environment(policy_type="manual")
    patch = (bound_plan.get("initial_state_patch") or {}).get("agent_data")
    if patch:
        environment.tools.update_db(patch)
    task_value = materialize_generic_tau_task(
        bound_plan, domain=domain,
        # section 175: the SAME already-patched agent DB this run uses, so the
        # set_user_info() phone number is resolved against the real state the
        # agent will actually see.
        agent_db=environment.tools.db if domain == "telecom" else None,
    )
    if message_history:
        task_value["initial_state"]["message_history"] = deepcopy(list(message_history))
    task = Task.model_validate(task_value)
    litellm_model = model if "/" in model else f"openai/{model}"
    _user_model = user_model or model
    user_litellm_model = _user_model if "/" in _user_model else f"openai/{_user_model}"
    if bound_plan.get("telecom_user_simulator") or bound_plan.get("source_branch_id") in _TELECOM_USER_SIMULATOR_BRANCHES:
        # section 106: real, narrowly-scoped exception (see
        # _TELECOM_USER_SIMULATOR_BRANCHES' own docstring) -- a real tau2
        # UserSimulator, not the deterministic driver, plays the user for
        # just these 2 branches. Mirrors tau2's own canonical construction
        # (tau2/runner/build.py's build_user): real instructions from the
        # materialized task's own user_scenario, real user-side tools from
        # the real environment, same real model/provider config as the
        # target Agent (not a different, unapproved model).
        from tau2.user.user_simulator import UserSimulator

        user = UserSimulator(
            llm=user_litellm_model,
            instructions=str(task.user_scenario),
            tools=environment.get_user_tools() or None,
            llm_args={"num_retries": 0, **dict(user_llm_args or {})},
        )
    else:
        # section 175: hand the scripted user the real names of the user-side
        # tools this environment actually exposes (tau2's own UserSimulator
        # gets the same list via tools=environment.get_user_tools(); retail/
        # airline have no user_tools at all, so this is an empty set there).
        # section 178: the authenticated user's real profile + the real
        # argument names of this branch's target tool, both read from the
        # real environment, so the scripted user can tell its on-file
        # identity apart from the new values it is asking for.
        target_tool = environment.tools.get_tools().get(
            (bound_plan.get("operation_argument_fact_bundle") or {}).get("tool_name") or ""
        )
        user = (user_class or GenericDeterministicTauUser)(
            bound_plan,
            **(user_extra_kwargs or {}),
            user_tool_names=(
                set(environment.user_tools.get_tools()) if environment.user_tools is not None else set()
            ),
            identity_profile=(
                _retail_identity_profile(bound_plan, environment.tools.db) if domain == "retail" else None
            ),
            target_tool_argument_names=(
                set((target_tool.params.model_json_schema().get("properties") or {})) if target_tool else set()
            ),
        )
    if user_factory is not None:
        # section 210: an LLM user (tactical_user_v0_1) built from what the
        # scripted user above would have said (its opening and known facts).
        user = user_factory(bound_plan, task, environment, user_litellm_model, domain, user)
    # section 172: the SAME string is handed to the agent and, later, to the
    # semantic judge (evaluate_generic_mechanical_oracle(domain_policy=...)).
    domain_policy = environment.get_policy()
    agent = create_llm_agent(
        tools=environment.get_tools(), domain_policy=domain_policy,
        llm=litellm_model, llm_args={"num_retries": 0, **dict(agent_llm_args or {})},
    )

    class BudgetedOrchestrator(Orchestrator):
        def __init__(self) -> None:
            super().__init__(
                domain=domain, agent=agent, user=user, environment=environment,
                task=task, max_steps=max(8, approved_agent_call_budget * 4 + 4),
                seed=seed, timeout=timeout_seconds, validate_communication=True,
            )
            self.message_cursor = 0
            self.observed_agent_calls = 0

        def _count_new_agent_calls(self) -> None:
            for message in self.trajectory[self.message_cursor:]:
                value = _message_dict(message)
                if value.get("role") == "assistant":
                    self.observed_agent_calls += 1
            self.message_cursor = len(self.trajectory)

        def _check_communication_error(self) -> None:
            value = _message_dict(self.message)
            # docs/agentcoveragetesting_reuse_log.md section 162
            # (task_a2867400): see _tolerates_mixed_content_and_tool_calls'
            # own declaration above for the full real root-cause analysis
            # (originally only AGENT-scoped here; now shared with the USER
            # role too for exactly the 2 real branches that can ever route
            # real tool_calls through the USER side at all).
            if _tolerates_mixed_content_and_tool_calls(str(self.from_role.value), value):
                return
            super()._check_communication_error()

        def initialize(self) -> None:
            super().initialize()
            self._count_new_agent_calls()

        def step(self) -> None:
            super().step()
            self._count_new_agent_calls()
            # Mirror the real tau2 Orchestrator's own _check_termination convention
            # (orchestrator.py: "Skip termination checks if we're waiting for
            # environment to respond") -- truncating while to_role == Role.ENV would
            # leave a real tool call dangling with no response, which _finalize()
            # itself rejects ("Environment should not receive the last message"),
            # a real crash found while reading a real telecom smoke-run transcript
            # (section 63). Deferring one step is safe: the budget is rechecked on
            # the very next real step, right after the environment's response is
            # delivered back to the agent.
            if (
                self.observed_agent_calls >= approved_agent_call_budget
                and not self.done
                and self.to_role != Role.ENV
            ):
                self.done = True
                self.termination_reason = TerminationReason.MAX_STEPS

    pre_state = _capture_state(environment, bound_plan)
    orchestrator = BudgetedOrchestrator()
    started = time.monotonic()
    simulation = orchestrator.run()
    elapsed = time.monotonic() - started
    if orchestrator.observed_agent_calls > approved_agent_call_budget:
        raise GenericTauOnlineError("approved target-Agent call budget was exceeded")
    message_values = _messages(simulation.messages)
    execution = {
        "schema_version": GENERIC_EXECUTION_VERSION,
        "adapter_version": GENERIC_ONLINE_VERSION,
        "branch_id": bound_plan["source_branch_id"],
        "bound_driver_plan_id": bound_plan["bound_driver_plan_id"],
        "bound_driver_plan_fingerprint": bound_plan["bound_driver_plan_fingerprint"],
        "driver_bindings": deepcopy(bound_plan["object_bindings"]),
        "model": model,
        "budget": {
            "approved_agent_call_budget": approved_agent_call_budget,
            "observed_agent_calls": orchestrator.observed_agent_calls,
            "orchestrator_steps": orchestrator.step_count,
            "request_retries": 0,
        },
        "duration_seconds": elapsed,
        "termination_reason": simulation.termination_reason,
        "pre_state": pre_state,
        "terminal_state": _capture_state(environment, bound_plan),
        "messages": message_values,
        # section 106: a real tau2 UserSimulator (for the 2 branches in
        # _TELECOM_USER_SIMULATOR_BRANCHES) has no transport_action_log
        # attribute at all (that's specific to GenericDeterministicTauUser)
        # -- getattr(..., []) degrades to an empty list rather than
        # crashing; downstream (runtime_observation_binding_v1.py's
        # normalize_tau_execution) already treats an empty/missing list as
        # "no driver_action_kind annotations available" and falls back to
        # its own empty action_log, so this is a safe, real degradation,
        # not a silent behavior change to any oracle check.
        "runtime_driver": {"transport_action_log": deepcopy(getattr(user, "transport_action_log", []))},
        "materialized_tau_task": task_value,
    }
    if hasattr(user, "perturbation_record"):
        execution["perturbation"] = user.perturbation_record()
    judge = None
    if enable_semantic_judge:
        def judge(*, criterion, target_value, transcript_excerpt):
            return _real_semantic_judge(
                model=judge_model or model, criterion=criterion, target_value=target_value,
                transcript_excerpt=transcript_excerpt, llm_args=judge_llm_args,
            )
    execution["mechanical_oracle"] = evaluate_generic_mechanical_oracle(
        bound_plan, execution, semantic_judge=judge, reference_tables=load_reference_tables(domain),
        domain_policy=domain_policy,
    )
    execution["execution_fingerprint"] = content_sha256(execution)
    return execution


def load_bound_plan_set(path: Path) -> dict[str, Any]:
    return validate_generic_bound_driver_plan_set(json.loads(path.read_text(encoding="utf-8")))
