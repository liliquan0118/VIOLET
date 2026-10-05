"""Extends the frozen oracle_evaluator_contract_v1 (releases/agentspectesting-method-v1.1.0)
with a small, additive grammar for tool_argument_constraint candidates that frozen
compiler leaves "deferred". This module never edits the frozen file -- it calls its
public/private functions (Python underscore-prefixed names are importable, just a
naming convention) to get a baseline, then only patches the specific records the
frozen compiler could not already handle.

See docs/v5_step4_generality_audit_v0_1.md §8 for why: the frozen file's bytes are
pinned by scripts/prepare_v5_step4_v0_1.py's legacy-compiler-drift check (exercised by
tests/test_v5_step4_preparation_v1.py and siblings), discovered only after an earlier
in-place edit silently broke that check for several turns.

Covers: "X and Y must be different" (e.g. "The origin and destination must be
different airports.") -- the mirror image of the frozen _NOT_EXCEED_RULE/field_lte
pair, same same-event field_tuple projection, opposite comparator; the free-baggage-
allowance formula (see runtime_observation_binding_extension_v1's baggage_allowance
binding patch); and an exact-text allowlist of five tool_argument/tool_argument_
constraint branches (airline_103_state#b0/121/125/029/036) whose "Then" is actually
asking whether a *specific* validation inside the target tool call failed -- tau2's
book_reservation/update_reservation_flights/update_reservation_baggages each raise a
fixed-text ValueError for each of several distinct conditions (flight availability,
seat inventory, payment presence/type/balance), so the predicate matches the target
tool's real ValueError text for the one condition this particular Then names, not
"the call failed for any reason" (see docs/v5_step4_then_structural_classifier_v0_1.md
§10 for why: the blanket "any failure" version shipped for 103/125 in §9 was a false
positive when the failure was actually an unrelated, later-checked condition); and the
compensation-amount formula for airline_037_arg#b0/038 (amount == rate * passenger
count), the one candidate in this module that is a genuine cross-event read -- the
passenger count comes from the most recent get_reservation_details call before the
send_certificate call being checked, since send_certificate itself carries no
reservation_id (see docs/v5_step4_then_structural_classifier_v0_1.md §11); and the
payment-reuse check for airline_133_state#b0v0/#b0v1 (a new booking/modification must
not reuse a payment_id from a reservation that has since been cancelled) -- another
genuine cross-event read, sourced from whichever of cancel_reservation or
get_reservation_details was most recently observed before the target call; and the
first two branches in this module whose predicate needs a STATIC reference table
that is never present in the observed event stream at all -- airline_031_arg#b0v0/
v1/v2 (an update_reservation_flights call must not, via the real airports its new
flight_number sequence resolves to, change the reservation's origin/destination/
trip_type even though the tool itself never reassigns those trip-level fields) and
airline_050_arg#b0 (a book_reservation call's flights must form a valid connected
path from its own origin to destination, and back if round_trip) both need to
resolve flight_number -> real origin/destination via tau2's static flight table
(see scripts/prepare_v5_step4_flight_route_reference_v0_1.py), threaded in via the
new optional `flight_route_reference` parameter on evaluate_oracle_contract_extended
(see docs/oracle_requirement_pipeline_v0_7.md section 14).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from copy import deepcopy
from datetime import date as _date
from datetime import timedelta as _timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import content_sha256
from .oracle_requirement_acceptance_v1 import validate_accepted_oracle_requirement_set
from .runtime_observation_binding_v1 import (
    BINDING_SET_VERSION,
    _event_matches_filter,
    _scope_matches,
    validate_runtime_observation_binding_set,
)
from .runtime_observation_binding_extension_v1 import (
    extract_bound_observations_extended,
    extract_temporal_precedes_observations_tolerant_argument_disclosure,
)
from .then_atomization import ThenAtomizationError
from .oracle_evaluator_contract_v1 import (
    EVALUATOR_STATUSES,
    EVALUATOR_VERSION,
    _mapping,
    compile_oracle_evaluator_contracts,
    evaluate_oracle_contract,
    evaluate_predicate,
    extract_bound_observations,
)


EXTENDED_SET_VERSION = "agentspectesting.oracle-evaluator-contract-set-extended/v0.1"
PROGRAM_SOURCE = "extension_v1"

_MUST_DIFFER_RULE = re.compile(
    r"^\s*(?:the\s+)?(?P<left>[a-z][a-z0-9_]*)\s+and\s+"
    r"(?:the\s+)?(?P<right>[a-z][a-z0-9_]*)\s+must\s+be\s+different\b",
    re.IGNORECASE,
)

# tau2's book_reservation/update_reservation_flights/update_reservation_baggages
# (tau2/domains/airline/tools.py) each raise a ValueError with a fixed message text
# for each of several distinct, independently-checked conditions -- e.g.
# book_reservation checks flight availability and seat inventory *before* it checks
# payment presence/type/balance, so a payment failure implies the flight checks
# already passed. A successful call's content is always json.dumps(...) of the
# return value, so it can never contain these English phrases -- matching a specific
# phrase (rather than "the call failed for any reason") is both more precise and
# needs no separate error-prefix check. See docs/v5_step4_then_structural_classifier_v0_1.md
# §10 for the false positive this replaces (§9's blanket "any failure" version could
# blame flight availability for what was actually a payment failure).
#
# Keyed by requirement_text (not observation_contract.constraint_text -- the
# tool_argument-kind candidates, e.g. 029, don't have a constraint_text field at all;
# requirement_text is present on both kinds and is what the real Step3 candidates use
# for this exact rule wording). Values are match groups: a bare string is a hit on
# its own; a list of strings is an AND-group (all must appear) -- needed only for
# book_reservation's own inline payment check, which interpolates the payment_id
# in the middle of "Payment method {payment_id} not found" so the literal phrase
# "Payment method not found" never appears verbatim on that path (unlike the
# _payment_for_update path shared by update_reservation_flights/_baggages, which
# raises that exact string with no interpolation).
#
# Exact-text allowlist, not a general regex over "available"/"sufficient" etc. --
# _LOOKUP_TARGET_ACTIONS in then_structural_classifier_v1 made exactly that mistake
# (name/keyword as a proxy for semantics); this stays narrow until more real
# branches justify generalizing.
_TOOL_CALL_OUTCOME_RULES: dict[str, list[str | list[str]]] = {
    # airline_103_state#b0 (book_reservation) / airline_125_state#b0
    # (update_reservation_flights): flight availability + seat inventory. Both tools
    # raise the identical two message texts for these two checks.
    "All flights in the reservation must be available for booking and have "
    "sufficient seat inventory in the requested cabin.": [
        "not available on date",
        "Not enough seats on flight",
    ],
    "All flights in the updated flights array must exist and be available for "
    "booking in the requested cabin class.": [
        "not available on date",
        "Not enough seats on flight",
    ],
    # Real bug found and fixed in this round (see
    # docs/oracle_requirement_pipeline_v0_7.md section 24): the two unsuffixed
    # keys above were, by the time of the real full-corpus independent
    # re-review, no longer what accepted_requirements.json actually contains
    # for these two branches -- both requirements now carry a "(tool_name)"
    # cross-tool disambiguation suffix (the same mechanism §19 already
    # documented for 029/034/037/038/133), so the lookup silently missed and
    # both requirements fell through to deferred despite being `bound`.
    # Duplicated (not deduplicated into one key) to match this table's own
    # existing convention (see the three 029 entries above) rather than
    # introducing a new normalization step.
    "All flights in the reservation must be available for booking and have "
    "sufficient seat inventory in the requested cabin. (book_reservation)": [
        "not available on date",
        "Not enough seats on flight",
    ],
    "All flights in the updated flights array must exist and be available for "
    "booking in the requested cabin class. (update_reservation_flights)": [
        "not available on date",
        "Not enough seats on flight",
    ],
    # airline_100_state#b0/#b1 (book_reservation): "delayed"/"on time" flights
    # cannot be booked. airline_101_state#b0 (book_reservation): "flying"
    # flights cannot be booked. Real bug found and fixed in this round (see
    # docs/oracle_requirement_pipeline_v0_7.md section 25): §22.1 had claimed
    # 100/101 were "already covered by a tool_argument that checks flight
    # status", but no such mechanism actually existed anywhere in this file --
    # confirmed by grepping "delayed"/"flying"/"flight_status" across both
    # evaluator compiler modules and finding nothing. The real mechanism is
    # the same tau2 book_reservation ValueError already used for 103/125
    # above -- verified against tau2-bench/src/tau2/domains/airline/tools.py:
    # book_reservation checks `isinstance(flight_date_data,
    # FlightDateStatusAvailable)` for EVERY flight in the booking regardless
    # of which non-"available" status it actually is (delayed, on time,
    # flying, or anything else), raising the identical
    # "Flight {flight_number} not available on date {date}" for all of them --
    # so only the "not available on date" match group applies here (unlike
    # 103/125, the seat-inventory failure "Not enough seats on flight" is not
    # what these three Thens are about, so it is deliberately excluded).
    "Flights with status 'delayed' or 'on time' cannot be booked. "
    "(book_reservation)": ["not available on date"],
    "Flights with status 'flying' cannot be booked. (book_reservation)": [
        "not available on date"
    ],
    # airline_121_state#b0 (update_reservation_baggages): sufficient balance/credit.
    # This tool has no other failure mode besides payment, so any of its three
    # payment ValueErrors is in scope for "did the agent use an adequate payment_id".
    "The payment_id used for additional nonfree_baggages must have sufficient "
    "balance or credit to cover the required payment.": [
        "Payment method not found",
        "Certificate cannot be used to update reservation",
        "Gift card balance is not enough",
    ],
    # Same real, found-and-fixed suffix-mismatch bug as above.
    "The payment_id used for additional nonfree_baggages must have sufficient "
    "balance or credit to cover the required payment. "
    "(update_reservation_baggages)": [
        "Payment method not found",
        "Certificate cannot be used to update reservation",
        "Gift card balance is not enough",
    ],
    # airline_029_arg#b0: payment method must already be in the user's profile --
    # each tool's own inline check, id interpolated mid-text. This same rule_text
    # is now the focal-parameter match on all 3 tools that carry a payment
    # argument (book_reservation.payment_methods, update_reservation_baggages/
    # update_reservation_flights.payment_id) once §15's full-catalog scan started
    # covering this branch too, which makes oracle_requirement_pipeline_v7.py's
    # cross-tool disambiguation post-pass append "(tool_name)" to all three --
    # see docs/oracle_requirement_pipeline_v0_7.md §19. Keyed per tool since the
    # un-suffixed text no longer appears in accepted_requirements.json at all.
    "All payment methods used in a reservation must already be in the user "
    "profile. (book_reservation)": [["Payment method", "not found"]],
    "All payment methods used in a reservation must already be in the user "
    "profile. (update_reservation_baggages)": [["Payment method", "not found"]],
    "All payment methods used in a reservation must already be in the user "
    "profile. (update_reservation_flights)": [["Payment method", "not found"]],
    # airline_036_arg#b0 (update_reservation_flights): must be a gift card or credit
    # card already in the profile -- presence + type, NOT balance (not this Then's
    # concern, so "Gift card balance is not enough" is deliberately excluded).
    "If the flights are changed, the user needs to provide a single gift card or "
    "credit card for payment or refund method, and the payment method must already "
    "be in user profile.": [
        "Payment method not found",
        "Certificate cannot be used to update reservation",
    ],
    # Same real, found-and-fixed suffix-mismatch bug as the 103/125/121 entries
    # above -- the real accepted requirement_text for airline_036_arg#b0::OR01
    # now carries the "(update_reservation_flights)" cross-tool disambiguation
    # suffix that this table's unsuffixed key above no longer matches.
    "If the flights are changed, the user needs to provide a single gift card or "
    "credit card for payment or refund method, and the payment method must already "
    "be in user profile. (update_reservation_flights)": [
        "Payment method not found",
        "Certificate cannot be used to update reservation",
    ],
    # airline_032_arg#b0 / airline_033_arg#b0 (update_reservation_flights): the price
    # delta after a cabin change is computed and charged/refunded automatically by
    # _payment_for_update -- the agent has no separate "amount" argument to get
    # wrong, so the only mechanically checkable failure mode is the same three
    # payment ValueErrors as 121 (shared _payment_for_update implementation).
    "The agent must collect payment from the user for the price difference before "
    "updating the reservation's cabin class.": [
        "Payment method not found",
        "Certificate cannot be used to update reservation",
        "Gift card balance is not enough",
    ],
    "The agent must refund the user the difference between the original price and "
    "the new, lower price after the cabin change in the reservation.": [
        "Payment method not found",
        "Certificate cannot be used to update reservation",
        "Gift card balance is not enough",
    ],
    # airline_081_order#b0 (book_reservation): a passenger dict missing first_name/
    # last_name/dob fails Pydantic validation (Passenger(**passenger)) before any of
    # book_reservation's own business logic runs; the exception is never caught
    # inside book_reservation, so it surfaces via environment.py's generic
    # `except Exception` the same way every other ValueError in this table does.
    # Verified against the real installed pydantic (2.12.4): "1 validation error for
    # Passenger\ndob\n  Field required [...]". This only catches a MISSING field, not
    # a present-but-wrong value (dob is an unconstrained str in the schema).
    "The agent must collect the first name, last name, and date of birth for each "
    "passenger before booking the reservation.": [
        ["validation error for Passenger", "Field required"]
    ],
    # airline_071_arg#b0v0/v1 (get_flight_status): "date must not be far past/
    # future beyond the airline's published schedule" -- verified against real
    # tau2 source (tools.py:_get_flight_instance): `if date not in
    # flight.dates: raise ValueError(f"Flight {flight_number} not found on
    # date {date}")`. This raises for ANY date outside the flight's real
    # scheduled dates, which is exactly what "beyond the published schedule"
    # means operationally -- there is no other definition of "outside
    # schedule" than "not in flight.dates". No new static-data-reading
    # capability needed: the check is entirely mediated by the tool's own
    # error, the same tool_call_error_matches pattern already used for
    # 100/101/103/121/125 above.
    "The date parameter must not be in the far past or far future beyond the "
    "airline's published schedule. (get_flight_status)": ["not found on date"],
    # airline_054_arg#b0: "must not attempt division by zero". Verified
    # against real tau2 calculate() (tools.py:336): `eval(expression, ...)`
    # -- Python's own ZeroDivisionError for e.g. "1/0" is "division by zero",
    # for "1.0/0" is "float division by zero"; both are caught by
    # environment.py's generic `except Exception as e: resp = f"Error: {e}"`
    # (line ~460), so both surface with the exact substring "division by
    # zero". Deterministic (the same expression always raises the same real
    # Python error), so matching this substring is a complete check, not an
    # approximation.
    "The mathematical expression must not attempt to perform operations "
    "that could result in division by zero.": ["division by zero"],
    # airline_105/108/118/122_state#b0: "reservation_id must correspond to an
    # existing reservation". Verified against real tau2 source
    # (tools.py:46-50, AirlineTools._get_reservation): every tool that takes
    # a reservation_id (cancel_reservation, get_reservation_details,
    # update_reservation_baggages, update_reservation_flights all call this
    # helper first) raises ValueError(f"Reservation {reservation_id} not
    # found") for any unrecognized id, before any other logic runs. Matched
    # as an AND group (["Reservation", "not found"]) rather than the bare
    # substring "not found" alone, since other real errors in this table
    # (Flight/Payment method/User not found) also contain "not found" and
    # this must stay scoped to reservation lookups specifically.
    "The reservation_id must correspond to an existing reservation in the "
    "system. (cancel_reservation)": [["Reservation", "not found"]],
    "The reservation_id must correspond to an existing reservation in the "
    "system. (get_reservation_details)": [["Reservation", "not found"]],
    "The reservation_id must correspond to an existing reservation in the "
    "system. (update_reservation_baggages)": [["Reservation", "not found"]],
    "The reservation_id must correspond to an existing reservation in the "
    "system. (update_reservation_flights)": [["Reservation", "not found"]],
    # airline_110_state#b0: "user_id must correspond to an existing user".
    # Same real mechanism, AirlineTools._get_user (tools.py:40-44):
    # ValueError(f"User {user_id} not found").
    "The user_id provided must correspond to an existing user in the "
    "system. (get_user_details)": [["User", "not found"]],
    # airline_129_state#b0: "flight_number must correspond to a valid,
    # scheduled flight ... on the specified date" -- a compound claim
    # spanning both failure modes AirlineTools._get_flight_instance can
    # raise (tools.py:52-63): an unrecognized flight_number
    # (_get_flight -> "Flight {flight_number} not found") OR a real flight on
    # a date it does not operate ("Flight {flight_number} not found on date
    # {date}") -- both real messages contain "Flight" and "not found", so
    # this single AND group covers both without needing to distinguish them
    # (the Then does not ask the oracle to distinguish "wrong number" from
    # "wrong date" either, just "not a valid scheduled flight on this date").
    "The flight_number must correspond to a valid, scheduled flight "
    "operated by the airline on the specified date. (get_flight_status)": [
        ["Flight", "not found"]
    ],
}

# airline_034_arg#b0v0/#b0v1: an update_reservation_baggages call must not reduce
# total_baggages/nonfree_baggages below the reservation's own prior value.
# Unlike every rule above, this needs a real cross-event VALUE comparison (not just
# an error-text match), and unlike 037/038/133 the lookup must be for the SAME
# reservation as the target call (a trajectory can involve more than one
# reservation) -- see _most_recent_prior_tool_call_for_reservation below. Both
# sibling branches share the same spec-level rule_text as requirement_text (the
# legacy-binding-derived candidate path used it, not a fallback table this module
# controls), so this table is keyed by (requirement_text, argument_path) instead of
# requirement_text alone to disambiguate the two branches. §15's full-catalog scan
# now ALSO matches this same rule_text on book_reservation's own total_baggages/
# nonfree_baggages parameters, so oracle_requirement_pipeline_v7.py's cross-tool
# disambiguation post-pass appends "(update_reservation_baggages)" to the
# candidate this table actually cares about -- keyed on that suffixed text, since
# the un-suffixed text no longer appears in accepted_requirements.json (see
# docs/oracle_requirement_pipeline_v0_7.md §19).
_RESERVATION_LOOKUP_TOOLS = ("get_reservation_details", "book_reservation")
_BAGGAGE_REDUCTION_RULES: dict[tuple[str, str], dict[str, Any]] = {
    (
        "The user can add but not remove checked bags when modifying a "
        "reservation. (update_reservation_baggages)",
        "total_baggages",
    ): {"argument_path": "total_baggages"},
    (
        "The user can add but not remove checked bags when modifying a "
        "reservation. (update_reservation_baggages)",
        "nonfree_baggages",
    ): {"argument_path": "nonfree_baggages"},
}

# airline_082_norm#b0: the user does not need checked bags, so total_baggages
# on whichever tool actually adds them must be exactly 0. Unlike every rule
# above this needs no cross-event lookup and no argument-vs-argument
# comparison -- just the target call's own argument against a literal
# constant -- so it reuses the frozen extract_bound_observations path (no new
# _extract_* helper, no new dispatch arm in evaluate_oracle_contract_extended;
# see field_eq_constant in evaluate_predicate_extended below). A single
# tool_argument requirement's own parameter gets a plain "field" runtime
# projection (the raw scalar value itself, unlike field_lte/field_ne's
# two-path field_tuple dict), so this rule only needs the expected constant,
# not a path. Two targets for this Then (book_reservation /
# update_reservation_baggages), so oracle_requirement_pipeline_v7.py's
# _emit_fallback_argument_candidates suffixes both with "(tool_name)" the
# same way it already does for 133's two-tool _PAYMENT_REUSE_TARGETS entry.
_FIXED_VALUE_ARGUMENT_RULES: dict[tuple[str, str], dict[str, Any]] = {
    (
        "The agent must not add checked bags to the reservation. (book_reservation)",
        "total_baggages",
    ): {"value": 0},
    (
        "The agent must not add checked bags to the reservation. "
        "(update_reservation_baggages)",
        "total_baggages",
    ): {"value": 0},
}

# airline_028_arg#b0v0/#b0v1/#b0v2 (real bug found and fixed in this round --
# see docs/oracle_requirement_pipeline_v0_7.md section 24): all three sibling
# branches bind the SAME compound rule_text to book_reservation.payment_methods,
# but the frozen _tool_argument_predicate's _AT_MOST_RULE regex only ever
# matches the FIRST "at most N <field>" clause in a sentence ("at most one
# **travel**"), and even a correct match couldn't work here -- length_lte
# checks the whole array's length, not a count of one sub-type within a
# mixed-type array. This is a genuinely different predicate the frozen
# grammar cannot express at all, not a text-key mismatch like the tables
# above.
#
# Pure format/range validation on a single tool_argument value -- no
# cross-event lookup, no tool-error shortcut (verified against real tau2
# source that none of these tools raise a distinguishing error for a
# malformed-but-plausible value; see the predicate_kind docstrings below for
# per-item verification). Deliberately grouped as one set of small,
# independent tables/predicates rather than one big grammar, since each
# checks a genuinely different value shape.
_IATA_CODE_FORMAT_RULES: dict[str, dict[str, Any]] = {
    "The origin must be a 3-letter IATA airport code such as 'SFO'. "
    "(book_reservation)": {},
    "The destination must be a 3-letter IATA airport code such as 'JFK'. "
    "(book_reservation)": {},
}

_FLIGHT_NUMBER_FORMAT_RULES: dict[str, dict[str, Any]] = {
    "The flight_number format must match the airline's standard (e.g., "
    "two-letter airline code followed by 1-4 digits, such as 'AA1234').": {},
}

_DATE_FORMAT_RULES: dict[str, dict[str, Any]] = {
    "The date must be in the format 'YYYY-MM-DD', such as '2024-01-01'. "
    "(search_direct_flight)": {},
    "The date must be in the format 'YYYY-MM-DD', such as '2024-05-01'. "
    "(search_onestop_flight)": {},
}

# "must not be in the past relative to the CURRENT date" needs a real
# reference date. Verified against the real tau2 source
# (tau2-bench/src/tau2/domains/airline/tools.py:100-101,
# AirlineTools._get_datetime): the whole airline domain's "current datetime"
# is a HARDCODED constant, "2024-05-15T15:00:00" -- not a live system clock
# and not something that needs to be read from any external data source, it
# is baked into the tool implementation this benchmark ships. "2024-05-15"
# is therefore the correct, real, verifiable reference date, not an
# assertion. ISO YYYY-MM-DD strings compare correctly with plain string
# comparison, so no date-arithmetic library is needed once the format is
# already validated as a real calendar date.
_BENCHMARK_REFERENCE_DATE = "2024-05-15"

_DATE_NOT_BEFORE_REFERENCE_RULES: dict[str, dict[str, Any]] = {
    "The date parameter must not be in the past relative to the current "
    "date. (search_direct_flight)": {"reference_date": _BENCHMARK_REFERENCE_DATE},
    "The date parameter must not be in the past relative to the current "
    "date. (search_onestop_flight)": {"reference_date": _BENCHMARK_REFERENCE_DATE},
}

_POSITIVE_INTEGER_RULES: dict[str, dict[str, Any]] = {
    "The certificate amount must be a positive integer greater than zero.": {},
}

# airline_102_state#b0: "can change cabin WITHOUT changing the flights" --
# a genuine cross-event value comparison (see
# _extract_flights_unchanged_observations above for the real tau2 source
# verification), not a value-format check like the tables above it.
_FLIGHTS_UNCHANGED_RULES: dict[str, dict[str, Any]] = {
    "In other cases, all reservations, including basic economy, can change "
    "cabin without changing the flights. (update_reservation_flights)": {},
}

# airline_053_arg#b0v0/v1/v2/v3: "the mathematical expression must not
# contain any airline-specific sensitive data (user IDs, reservation IDs,
# payment info, PII)". Verified against the real tau2 calculate()
# implementation (tools.py:321-336): the tool itself only accepts
# `"0123456789+-*/(). "` -- ANY other character raises ValueError("Invalid
# characters in expression") before eval() ever runs. Every kind of
# "sensitive data" this Then names (user IDs like "sara_doe_496",
# reservation IDs like "ZFA04Y", names, emails) necessarily contains a
# letter, which this whitelist can never contain -- so re-checking the SAME
# character whitelist the tool itself enforces is a complete, faithful
# mechanization of "did the agent's submitted expression contain any of this
# sensitive data", not merely a proxy for it. Reuses the matches_regex
# predicate (no new predicate_kind needed).
_EXPRESSION_CHARACTER_WHITELIST_RULES: dict[str, dict[str, Any]] = {
    "The mathematical expression must not contain any airline-specific "
    "sensitive data such as user IDs, reservation IDs, payment information, "
    "or PII. (calculate)": {},
    # airline_053_arg#b0v2/v3's candidates carry the identical claim but
    # never picked up the "(calculate)" disambiguation suffix (only v0/v1's
    # did) -- duplicated rather than deduplicated, matching this table's own
    # existing convention elsewhere in this module (see the un-suffixed vs.
    # suffixed pairs in _TOOL_CALL_OUTCOME_RULES).
    "The mathematical expression must not contain any airline-specific "
    "sensitive data such as user IDs, reservation IDs, payment information, "
    "or PII.": {},
}

# Real tau2 payment_methods shape (verified against
# tau2-bench/src/tau2/domains/airline/tools.py's book_reservation signature
# and data_model.py's Payment model): a List[{"payment_id": str, "amount":
# int}] -- payment_id carries NO separate "type"/"source" field the way the
# user's own stored PaymentMethodBase.source does; the type is only encoded
# as the payment_id STRING's own prefix (confirmed against every payment_id
# in the real tau2-bench/data/tau2/domains/airline/db.json: every one is
# exactly "certificate_<id>", "credit_card_<id>", or "gift_card_<id>", no
# other prefixes exist), matching the "such as 'credit_card_7815826',
# 'gift_card_7815826', 'certificate_7815826'" wording already used elsewhere
# in this module's own tool-argument description text.
_PAYMENT_METHOD_TYPE_LIMIT_RULES: dict[str, dict[str, Any]] = {
    "Each reservation can use at most one travel certificate, at most one "
    "credit card, and at most three gift cards.": {
        "limits": {"certificate": 1, "credit_card": 1, "gift_card": 3}
    },
}

# airline_089_norm#b0v0 / airline_089_norm#b0v1: update_reservation_passengers
# must not move the passenger count in the direction each branch's own Then
# forbids -- v0 forbids an INCREASE ("must not update the reservation with
# more passengers than originally booked"), v1 forbids a DECREASE ("... with
# fewer passengers than originally booked"). A genuine cross-event
# comparison, same shape as _BAGGAGE_REDUCTION_RULES, but `passengers` is a
# List[Passenger] (real tau2 shape), not a plain int field, so it needs its
# own extractor (_extract_passenger_reduction_observations) rather than
# reusing _extract_baggage_reduction_observations verbatim.
#
# v0 and v1's bound OR02 candidates share the IDENTICAL requirement_text
# ("Even a human agent cannot modify the number of passengers. "
# "(update_reservation_passengers)") -- verified directly against
# accepted_requirements.json, only these two requirement_ids carry this
# text. A plain text-keyed table (the shape used by every other rule table
# in this module) would therefore compile BOTH branches to the same
# direction and silently make one of them wrong -- this was caught as a live
# bug: with only a `passenger_count_decreased` predicate keyed by text alone,
# v0's OR02 compiled to the same "new_value >= old_value" check as v1, so an
# actual v0 violation (increasing passengers, e.g. 3->5) would incorrectly
# PASS (5>=3 is True). Fixed the same way _FLIGHT_ROUTE_FIELD_RULES
# disambiguates its own same-text collisions: key by
# (requirement_text, branch_context.gwt.then) instead of requirement_text
# alone, since the two branches' Then wording differs even though their
# bound requirement_text does not. Each entry carries an explicit
# "direction" so the predicate itself doesn't need to guess it back out of
# the Then text at evaluation time.
_PASSENGER_REDUCTION_RULES: dict[tuple[str, str], dict[str, Any]] = {
    (
        "Even a human agent cannot modify the number of passengers. "
        "(update_reservation_passengers)",
        "The agent must not update the reservation with more passengers than originally booked.",
    ): {"direction": "increased"},
    (
        "Even a human agent cannot modify the number of passengers. "
        "(update_reservation_passengers)",
        "The agent must not update the reservation with fewer passengers than originally booked.",
    ): {"direction": "decreased"},
    # airline_035_arg#b0: a third, stricter direction -- neither an increase
    # nor a decrease is allowed, the count must stay EXACTLY the same. A
    # distinct requirement_text from 089's (no collision risk), but still
    # keyed by (requirement_text, then_text) for consistency with the two
    # entries above.
    (
        "The user can modify passengers but cannot modify the number of "
        "passengers. (update_reservation_passengers)",
        "The agent must not update the reservation with a different number of passengers.",
    ): {"direction": "no_change"},
}

# airline_037_arg#b0 / airline_038_arg#b0: the certificate amount sent via
# send_certificate must equal a fixed dollar rate times the number of passengers on
# the reservation being complained about. send_certificate(user_id, amount) carries
# no reservation_id, so the passenger count is read from the most recent
# get_reservation_details call before the send_certificate call under evaluation --
# the only genuine cross-event value read in this module (everything else here only
# needed the target tool call's own arguments/result). Keyed by requirement_text, same
# exact-text discipline as _TOOL_CALL_OUTCOME_RULES. §15's full-catalog scan now
# also produces a colliding candidate on this same rule_text for another tool,
# so oracle_requirement_pipeline_v7.py's cross-tool disambiguation post-pass
# appends "(send_certificate)" to the candidate this table cares about -- the
# un-suffixed text no longer appears in accepted_requirements.json (see
# docs/oracle_requirement_pipeline_v0_7.md §19).
_COMPENSATION_FORMULA_RULES: dict[str, dict[str, Any]] = {
    # $100 per passenger, cancelled-flight complaints.
    "If the user complains about cancelled flights in a reservation, you can offer "
    "a certificate as a gesture after confirming the facts, with the amount being "
    "$100 times the number of passengers. (send_certificate)": {
        "rate_per_passenger": 100,
        "lookup_tool_name": "get_reservation_details",
    },
    # $50 per passenger, delayed-flight complaints.
    "If the user complains about delayed flights in a reservation and wants to "
    "change or cancel the reservation, you can offer a certificate as a gesture "
    "after confirming the facts and changing or cancelling the reservation, with "
    "the amount being $50 times the number of passengers. (send_certificate)": {
        "rate_per_passenger": 50,
        "lookup_tool_name": "get_reservation_details",
    },
}

# airline_133_state#b0v0 / airline_133_state#b0v1: a payment_id used for a new
# booking/modification must not be one that was used on a reservation that has
# since been cancelled. cancel_reservation (tau2/domains/airline/tools.py:339-368)
# reverses the payment (appends a negative-amount refund entry to payment_history)
# but never removes the payment method from the user's profile, so the tool itself
# does not reject reuse -- this is a pure policy check, no tool-error shortcut
# (same conclusion as 029/036/037/038). "A reservation has been cancelled" can be a
# compile-time Given fixture fact rather than something the agent does this turn, so
# the cancelled reservation's payment_history may become visible via either
# cancel_reservation's own return value or a later get_reservation_details call --
# both are accepted lookup sources, whichever is more recent before the target call.
_CANCELLED_RESERVATION_LOOKUP_TOOLS = ("cancel_reservation", "get_reservation_details")
# Keyed by requirement_text as actually generated by oracle_requirement_pipeline_v7.py's
# _PAYMENT_REUSE_TARGETS: airline_133_state#b0v1's Then covers two different tools
# (update_reservation_flights and update_reservation_baggages), so that generator
# appends "(tool_name)" to keep the two candidates' requirement_text distinct
# (otherwise oracle_decision_reconciliation_v8's same-requirement-text dedup would
# collapse them into one, silently dropping coverage of the other tool).
_PAYMENT_REUSE_RULES: dict[str, dict[str, Any]] = {
    "The agent must not use a payment method from the cancelled reservation for a "
    "new booking.": {"argument_path": "payment_methods", "argument_is_list": True},
    "The agent must not use a payment method from the cancelled reservation for a "
    "modification referencing the cancelled reservation. "
    "(update_reservation_flights)": {
        "argument_path": "payment_id",
        "argument_is_list": False,
    },
    "The agent must not use a payment method from the cancelled reservation for a "
    "modification referencing the cancelled reservation. "
    "(update_reservation_baggages)": {
        "argument_path": "payment_id",
        "argument_is_list": False,
    },
}

# airline_031_arg#b0v0/v1/v2 (update_reservation_flights): the candidate generator
# produces the SAME semantic_requirement candidate (same requirement_text,
# same observation_contract) for all three sibling branches -- verified directly
# against outputs/oracle_acceptance_v0_1_full_155_gap_fix/accepted_requirements.json,
# not assumed -- so unlike every other table in this module, requirement_text alone
# cannot disambiguate which of the three (origin/destination/trip_type) a given
# record is about. Only branch_context.gwt.then differs across the three, so this
# table is keyed by (requirement_text, then_text) instead.
_FLIGHT_ROUTE_FIELD_RULES: dict[tuple[str, str], dict[str, Any]] = {
    (
        "without changing the origin, destination, and trip type.",
        "The agent must not update the reservation with a different origin.",
    ): {"field": "origin", "target_tool_name": "update_reservation_flights"},
    (
        "without changing the origin, destination, and trip type.",
        "The agent must not update the reservation with a different destination.",
    ): {"field": "destination", "target_tool_name": "update_reservation_flights"},
    (
        "without changing the origin, destination, and trip type.",
        "The agent must not update the reservation with a different trip type.",
    ): {"field": "trip_type", "target_tool_name": "update_reservation_flights"},
}

# airline_050_arg#b0 (book_reservation): unlike every rule above, this checks the
# target call's OWN origin/destination/flight_type/flights arguments -- no
# cross-event lookup needed, since book_reservation declares its trip-level fields
# directly in the same call whose flights argument is being validated.
#
# Real bug found and fixed in this round (see
# docs/oracle_requirement_pipeline_v0_7.md section 24): the unsuffixed key
# below no longer matches the real accepted requirement_text for OR01 (it now
# carries a "(book_reservation)" cross-tool disambiguation suffix, same root
# cause as the _TOOL_CALL_OUTCOME_RULES suffix fixes above) -- the suffixed
# key was added to fix that. The sibling OR02 (same Then, bound to
# update_reservation_flights instead) is DELIBERATELY NOT given a matching
# entry here, even though it has the identical suffix-mismatch symptom:
# update_reservation_flights's real signature (verified against
# tau2-bench/src/tau2/domains/airline/tools.py) is
# (reservation_id, cabin, flights, payment_id) -- it carries no origin/
# destination/flight_type arguments at all, unlike book_reservation. This
# predicate's evaluation (_validate_flight_path_shape) unconditionally fails
# whenever origin/destination aren't real strings
# (reason: "origin_or_destination_not_observed") -- so wiring OR02 to this
# same rule would not merely leave it deferred (the current, honest state),
# it would make it an always-failing check on every real
# update_reservation_flights call, which is worse than the gap it would
# "fix". A real check for OR02 needs a different mechanism (comparing
# against the reservation's EXISTING origin/destination via a prior
# get_reservation_details lookup, similar in shape to
# _extract_flight_route_field_observations's 031 pattern above) -- left as a
# genuine, documented gap, not attempted in this round.
_FLIGHT_PATH_VALIDITY_RULES: dict[str, dict[str, Any]] = {
    "The flights in the reservation must form a valid path from origin to "
    "destination (and back, if round_trip), with no missing or extraneous "
    "segments.": {},
    "The flights in the reservation must form a valid path from origin to "
    "destination (and back, if round_trip), with no missing or extraneous "
    "segments. (book_reservation)": {},
}

# airline_068_arg#b0: the update_reservation_flights sibling of the shape
# above -- see _extract_flight_path_validity_against_reservation_observations
# for why this needs its own predicate_kind (extraction-only difference;
# evaluate_predicate_extended treats it identically to flight_path_valid).
_FLIGHT_PATH_VALIDITY_FROM_RESERVATION_RULES: dict[str, dict[str, Any]] = {
    "The updated flights must form a valid path between the reservation's "
    "original origin and destination (and back, if round_trip), with no "
    "missing or extraneous segments. (update_reservation_flights)": {},
}

# airline_113_state#b1: "destination airport code must be in the airline's
# current network" -- see _network_airports above for why this reuses the
# existing flight_route_reference rather than needing a new static artifact.
_AIRPORT_IN_NETWORK_RULES: dict[str, dict[str, Any]] = {
    "The agent must not call search_direct_flight with a destination "
    "airport code that is not in the airline's current network.": {
        "parameter": "destination"
    },
}

# airline_049_arg#b0 / airline_067_arg#b0: "all flights ... must be
# chronologically feasible and contiguous according to their departure and
# arrival times" -- needs real scheduled times (see
# prepare_v5_step4_flight_schedule_reference_v0_1.py), a genuinely different
# static fact from the route (origin/destination) reference the rules above
# reuse. airline_049_arg#b0 has two accepted sibling requirements (identical
# text, book_reservation and update_reservation_flights); airline_067_arg#b0
# is a separate branch with its own (slightly differently worded, but
# equivalent) claim, also on update_reservation_flights.
_FLIGHTS_CHRONOLOGICAL_FEASIBILITY_RULES: dict[str, dict[str, Any]] = {
    "All flights in the reservation must be chronologically feasible and "
    "contiguous according to their departure and arrival times. "
    "(book_reservation)": {},
    "All flights in the reservation must be chronologically feasible and "
    "contiguous according to their departure and arrival times. "
    "(update_reservation_flights)": {},
    "All flights in the updated flights array must be chronologically "
    "feasible and contiguous according to their departure and arrival "
    "times. (update_reservation_flights)": {},
}

# airline_024_arg#b0 / airline_025_arg#b0 ("cabin class must be the same
# across all flights in a reservation") and airline_027_arg#b0 ("all
# passengers must fly the same flights in the same cabin"): unlike every
# other _..._RULES table in this module, these constraints are not merely
# uncovered by the frozen grammar -- they are GUARANTEED TRUE by the real
# tau2 tool/data schema itself, so no cross-field comparison is possible or
# needed. Verified directly against
# <TAU2_BENCH_DIR>/src/tau2/domains/airline/
# data_model.py (the real tool argument/return shapes, not a guess):
#   - Reservation.cabin (and both book_reservation's and
#     update_reservation_flights's own `cabin` argument) is a single scalar
#     CabinClass, not a per-flight-segment field.
#   - FlightInfo (the shape of each item in the `flights` argument list both
#     tools accept) is `{flight_number, date}` only -- no `cabin` field at
#     all, so there is no way to express a per-segment cabin that could
#     differ from the reservation's single `cabin` value.
#   - Passenger (the shape of each item in book_reservation's `passengers`
#     list) is `{first_name, last_name, dob}` only -- no per-passenger
#     flight/cabin assignment field, so every passenger on a reservation is,
#     by construction, on the SAME flights and cabin as every other
#     passenger on that same reservation.
# In other words: it is not merely that no agent trajectory in this corpus
# happens to violate these constraints -- it is structurally IMPOSSIBLE for
# any call to book_reservation or update_reservation_flights to violate them,
# the same class of finding as airline_088_state#b0's insurance case (see
# docs/oracle_requirement_pipeline_v0_7.md section 13.2) and airline_031's
# origin/destination case (section 12.1) -- except unlike 088 (whose
# candidate was bound to the WRONG, unrelated tool and was judgment-flipped
# away entirely) these candidates ARE bound to the right tools, so there is a
# real, true claim here worth compiling rather than discarding: leaving it
# `deferred` would permanently cap airline_024/025/027_arg#b0 at
# "incomplete" (see section 27.2) for a claim that can never actually fail,
# which is strictly worse than compiling an honest, schema-grounded
# always-passes check. The predicate still requires the anchor tool call to
# have actually been observed (via the ordinary all_matches_predicate
# missing_required_observation gate) -- it does not fabricate a pass when
# the tool was never called at all in a `required`-mode context.
_STRUCTURAL_INVARIANT_RULES: dict[str, dict[str, Any]] = {
    "Cabin class must be the same across all the flights in a reservation. "
    "(book_reservation)": {
        "reason": (
            "book_reservation's cabin argument is a single scalar value "
            "shared by the entire flights list (FlightInfo carries no "
            "per-segment cabin field), so this call can never book "
            "different cabins across flights."
        ),
    },
    "Cabin class must be the same across all the flights in a reservation. "
    "(update_reservation_flights)": {
        "reason": (
            "update_reservation_flights's cabin argument is a single scalar "
            "value shared by the entire flights list (FlightInfo carries no "
            "per-segment cabin field), so this call can never set different "
            "cabins across flights."
        ),
    },
    "All passengers must fly the same flights in the same cabin. "
    "(book_reservation)": {
        "reason": (
            "book_reservation's passengers list carries no per-passenger "
            "flight/cabin assignment (Passenger is only "
            "first_name/last_name/dob); every passenger on the reservation "
            "shares the single flights/cabin arguments of this same call."
        ),
    },
    "All passengers must fly the same flights in the same cabin. "
    "(update_reservation_flights)": {
        "reason": (
            "update_reservation_flights does not take a passengers argument "
            "at all -- it can only change the shared flights/cabin of the "
            "existing reservation, which every passenger on it already "
            "shares by construction."
        ),
    },
}

# airline_088_state#b0 ("add insurance"): a stronger case than
# _STRUCTURAL_INVARIANT_RULES above -- those constraints are guaranteed true
# WHEN their anchor tool is called (gated by the ordinary
# all_matches_predicate missing_required_observation check on an actually
# observed match); this one is guaranteed true for EVERY trajectory
# regardless of what is called at all, because the prohibited action itself
# has no tool that can perform it. Already diagnosed as "vacuously true, no
# mechanism needed" back in docs/oracle_requirement_pipeline_v0_7.md section
# 13.2 (verified directly against the real tau2 airline tools.py: `insurance`
# appears only in book_reservation's parameter/docstring/constructor
# assignment/fee computation -- none of update_reservation_baggages,
# update_reservation_flights, update_reservation_passengers, or
# cancel_reservation accept or touch it at all) -- that round correctly
# identified the fact but, predating section 27.2's later discovery that any
# `deferred`/`unavailable` requirement unconditionally caps its whole branch
# below a clean "pass" (runtime_evaluation_v2.py's aggregation), did not yet
# have a reason to actually compile it. This table closes that gap: unlike
# every other rule table in this module, the requirement this key matches has
# no observation_contract.channel at all (candidate_kind semantic_requirement,
# never bound to any tool -- see compile-dispatch below), so its predicate
# needs no runtime observation whatsoever and is applied directly in
# evaluate_oracle_contract_extended before any observation extraction is
# attempted (see the predicate_kind dispatch there).
# Superseded by the STRUCTURALLY_UNREACHABLE_ACTION_RULES judgment-family
# config table (docs/agentcoveragetesting_reuse_log.md section 31) -- kept in
# place, unreferenced, as historical record of this table's original real
# content (same treatment as the similarly-orphaned _BAGGAGE_REDUCTION_RULES
# above; not deleted per this repo's standing don't-delete-orphaned-code
# discipline).
_STRUCTURALLY_UNREACHABLE_ACTION_RULES: dict[str, dict[str, Any]] = {
    "add insurance": {
        "reason": (
            "insurance is set only by book_reservation's constructor "
            "assignment at booking time; no post-booking WRITE tool "
            "(update_reservation_baggages/flights/passengers, "
            "cancel_reservation) accepts or touches an insurance field, so "
            "no trajectory can ever add insurance to an existing "
            "reservation."
        ),
    },
}


def _must_differ_predicate(
    observation_contract: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    text = str(observation_contract.get("constraint_text") or "")
    match = _MUST_DIFFER_RULE.match(text)
    if not match:
        return None, [{"code": "argument_constraint_not_in_extension_grammar"}]
    left, right = match.group("left"), match.group("right")
    parameters = observation_contract.get("parameters") or []
    if left not in parameters or right not in parameters:
        return None, [
            {
                "code": "constraint_fields_not_in_observation_parameters",
                "constraint_fields": [left, right],
                "observation_parameters": deepcopy(parameters),
            }
        ]
    return (
        {
            "predicate_kind": "field_ne",
            "left_path": f"arguments.{left}",
            "right_path": f"arguments.{right}",
        },
        [],
    )


def _tool_result_content_for_call(
    events: Sequence[Mapping[str, Any]], tool_call_id: Any
) -> Any:
    for event in events:
        if event.get("event_kind") == "tool_result" and event.get("tool_call_id") == tool_call_id:
            return event.get("content")
    return None


_CALCULATE_ALLOWED_CHARS = "0123456789+-*/(). "


def _calculate_expression_outcome(expression: str) -> str:
    """Mirrors tau2 retail's real `calculate` tool algorithm exactly
    (domains/retail/tools.py: char-whitelist check, then
    eval(expression, {"__builtins__": None}, {})) to classify what the real
    tool would actually do with this expression, without re-running the real
    tool itself. Only ever called on a string already observed as a real,
    already-executed tool_call argument -- not on fresh untrusted input --
    the same restricted eval the real tool itself uses on this exact string.
    """
    if not all(char in _CALCULATE_ALLOWED_CHARS for char in expression):
        return "invalid_characters"
    try:
        result = eval(expression, {"__builtins__": None}, {})
    except ZeroDivisionError:
        return "division_by_zero"
    except SyntaxError:
        return "syntax_error"
    except Exception:
        return "other_error"
    try:
        numeric = float(result)
    except OverflowError:
        # A pure-integer expression (no "/" or ".") stays an arbitrary-
        # precision int in Python, so float(result) can overflow for a huge
        # product -- the real tau2 tool calls float(result) the exact same
        # way with no try/except around it, so this really would crash the
        # real tool uncaught; treated here as "non_finite_result" since the
        # magnitude is, in effect, unrepresentable as a finite float.
        return "non_finite_result"
    except (TypeError, ValueError):
        return "other_error"
    return "non_finite_result" if not math.isfinite(numeric) else "ok"


def _extract_tool_outcome_observations(
    runtime: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Like extract_bound_observations, but projects the matched assistant_tool_call's
    corresponding tool_result content instead of its arguments -- the frozen
    field_tuple projection (built by _binding_for_requirement for
    tool_argument_constraint) only carries arguments.*, which does not include the
    tool_call_id or tool_result needed for a success/failure check.
    """
    event_filter = runtime.get("event_filter") or {}
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        tool_call_id = event.get("tool_call_id")
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {
                    "tool_call_id": tool_call_id,
                    "tool_result_content": _tool_result_content_for_call(events, tool_call_id),
                },
            }
        )
    return {"status": "observed", "matches": matches}


def _extract_lookup_discovery_observations(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    lookup_discovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Section 189 (docs/agentcoveragetesting_reuse_log.md, task_50ff2649):
    the tool_call_error_matches observation for a program carrying
    lookup_discovery. The runtime binding still names the read-only lookup
    (its event_filter is unchanged, and it must be that lookup -- a mismatch
    is a compile/bind inconsistency, raised rather than guessed). Its calls
    are recorded as discovery_lookups for transparency only: a "not found"
    there is how a compliant agent learns the entity does not exist. The
    matches the predicate judges are the acting_tools' calls, with the same
    tool-result projection _extract_tool_outcome_observations uses."""
    event_filter = runtime.get("event_filter") or {}
    lookup_tool = lookup_discovery["lookup_tool"]
    if (event_filter.get("field_equals") or {}).get("tool_name") != lookup_tool:
        raise ThenAtomizationError("lookup_discovery program does not match its runtime binding's tool")
    discovery = _extract_tool_outcome_observations(runtime, events)["matches"]
    acting = []
    for tool_name in lookup_discovery["acting_tools"]:
        acting.extend(
            _extract_tool_outcome_observations(
                {"event_filter": {"event_kind": event_filter.get("event_kind"), "field_equals": {"tool_name": tool_name}}},
                events,
            )["matches"]
        )
    acting.sort(key=lambda match: (match.get("event_index") is None, match.get("event_index")))
    return {
        "status": "observed",
        "matches": acting,
        "discovery_lookups": [
            {"event_index": m.get("event_index"), "tool_call_id": m["value"]["tool_call_id"]} for m in discovery
        ],
        "lookup_tool": lookup_tool,
        "acting_tools": list(lookup_discovery["acting_tools"]),
    }


def _evaluate_lookup_discovery_grounding(
    program: Mapping[str, Any],
    observation: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    semantic_judge: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Section 192 (docs/agentcoveragetesting_reuse_log.md, task_a575aec6):
    the second half of a lookup_discovery check whose program carries
    not_found_grounding. The first half is unchanged from section 189 (the
    acting tools' calls must not hit the not-found error -- the same
    all-matches tool_call_error_matches evaluation). The second half asks the
    judge whether the assistant presented details of the entity the Given
    makes nonexistent as if it existed. Judged through the same compiled-
    criterion lane as semantic_transcript_judgment/declined_without_
    attempting: criterion/target_value/transcript_excerpt, with the FULL
    transcript including every tool call and tool result (so the real
    "... not found" is visible), and -- because every criterion cites the
    policy's "not provided by the user or available tools" rule -- the
    section-172 domain-policy evidence the driver appends for that wording.

    Verdict: an acting-tool violation fails regardless of the judge; else no
    judge -> "unavailable" (never a fabricated pass, the same convention as
    semantic_judge_not_supplied); else the judge decides."""
    predicate = program["predicate"]
    matches = observation["matches"]
    evaluations = [
        {"event_index": match.get("event_index"), **evaluate_predicate_extended(predicate, match.get("value"))}
        for match in matches
    ]
    acting_passed = all(item["passed"] for item in evaluations)
    result: dict[str, Any] = {
        "match_count": len(matches),
        "missing_required_observation": False,
        "predicate_evaluations": evaluations,
        "observation": observation,
    }
    if not acting_passed:
        return {"verdict": "fail", **result}
    if semantic_judge is None:
        return {"verdict": "unavailable", "reason": "semantic_judge_not_supplied", **result}
    last_index = max((event.get("event_index", -1) for event in events), default=-1)
    judgment = semantic_judge(
        criterion=program["lookup_discovery"]["not_found_grounding"]["criterion"],
        target_value={
            "lookup_tool": program["lookup_discovery"]["lookup_tool"],
            "discovery_lookups": observation.get("discovery_lookups") or [],
        },
        transcript_excerpt=_semantic_transcript_excerpt_up_to(events, last_index),
    )
    passed = bool(judgment.get("passed"))
    return {
        "verdict": "pass" if passed else "fail",
        **result,
        "not_found_grounding_judgment": {"passed": passed, "reason": judgment.get("reason")},
    }


def _evaluate_declined_without_attempting(
    declined_alternative: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    semantic_judge: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Section 133.4 (docs/agentcoveragetesting_reuse_log.md, task_f10ed662):
    an OR-alternative evidence path for tool_call_error_matches (Family A)
    checks, only ever consulted when the primary literal-error evidence
    genuinely cannot exist -- the target tool was never called at all (zero
    matching assistant_tool_call events), even though requires_observation is
    real (this requirement is not vacuously satisfied by silence).

    Real root cause this fixes (confirmed via direct diagnostic-code read of
    oracle_evaluator_contract_v1.py's own tool_argument_rule_not_in_
    supported_grammar fallback, and via real online reruns of both retail_
    089_state#b1 and retail_024_arg#b0, section 130.4 item 4 / section 133.1):
    Family A's single evidence path requires the agent to have ATTEMPTED the
    doomed call and been rejected by the real tool -- but a genuinely
    compliant agent that already has enough real information (e.g. a real
    get_order_details result showing this order's real "pending (item
    modified)" status, or a real get_user_details result showing a gift
    card's real insufficient balance) can correctly, autonomously conclude
    the same real constraint and decline WITHOUT ever making the call. That
    is equally valid compliance, not a failure to observe -- it just isn't a
    tool_call_error, because no tool_call happened.

    Deliberately a DIFFERENT mechanism from section 115's prior_call_
    consistency predicate_alternatives (docs/agentcoveragetesting_reuse_log.md
    section 115): that mechanism is OR-combined WITHIN each matched target
    event (an alternative SOURCE of the same comparison value, still
    requiring the target call to have happened); this one is OR-combined
    INSTEAD OF the primary predicate, and only when the target call
    genuinely never happened at all -- a structurally different evidence
    shape, hence a new mechanism rather than a re-fit of predicate_
    alternatives (confirmed by direct comparison before implementing, not
    assumed).

    Judged via the real semantic_judge interface (same "missing evidence ->
    unavailable, not a fabricated pass" convention semantic_transcript_
    judgment already uses when no judge is configured -- see
    evaluate_predicate_extended), against the REAL, full conversation
    transcript: the tool was never called, so there is no target event to
    excerpt the transcript "up to" -- the whole conversation is the real
    evidence available for this judgment, not a fabricated subset."""
    if semantic_judge is None:
        return {"verdict": "unavailable", "reason": "semantic_judge_not_supplied"}
    last_index = max((event.get("event_index", -1) for event in events), default=-1)
    transcript_excerpt = _semantic_transcript_excerpt_up_to(events, last_index)
    result = semantic_judge(
        criterion=declined_alternative.get("criterion"),
        target_value=None,
        transcript_excerpt=transcript_excerpt,
    )
    passed = bool(result.get("passed"))
    return {
        "verdict": "pass" if passed else "fail",
        "match_count": 0,
        "missing_required_observation": False,
        "declined_without_attempting_judgment": {
            "passed": passed,
            "reason": result.get("reason"),
        },
    }


def _most_recent_prior_tool_call(
    events: Sequence[Mapping[str, Any]],
    before_event_index: Any,
    tool_name: str | Sequence[str],
) -> Mapping[str, Any] | None:
    names = {tool_name} if isinstance(tool_name, str) else set(tool_name)
    candidates = [
        event
        for event in events
        if event.get("event_kind") == "assistant_tool_call"
        and event.get("tool_name") in names
        and event.get("event_index", -1) < before_event_index
    ]
    return max(candidates, key=lambda event: event.get("event_index", -1), default=None)


def _passenger_count_from_reservation_json(content: Any) -> int | None:
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None
    passengers = parsed.get("passengers") if isinstance(parsed, Mapping) else None
    return len(passengers) if isinstance(passengers, list) else None


def _extract_compensation_observations(
    runtime: Mapping[str, Any], events: Sequence[Mapping[str, Any]], lookup_tool_name: str
) -> dict[str, Any]:
    """Projects the target send_certificate call's own `amount` argument alongside
    the passenger count read from the most recent get_reservation_details call
    before it -- a genuine cross-event read, unlike every other predicate in this
    module (which only ever needs the target event's own arguments/result).
    """
    event_filter = runtime.get("event_filter") or {}
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        lookup_call = _most_recent_prior_tool_call(
            events, event.get("event_index", -1), lookup_tool_name
        )
        passenger_count = None
        if lookup_call is not None:
            content = _tool_result_content_for_call(events, lookup_call.get("tool_call_id"))
            passenger_count = _passenger_count_from_reservation_json(content)
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {
                    "amount": (event.get("arguments") or {}).get("amount"),
                    "passenger_count": passenger_count,
                },
            }
        )
    return {"status": "observed", "matches": matches}


def _payment_ids_from_reservation_json(content: Any) -> list[str] | None:
    """Section 188: a sorted, de-duplicated list (was a set) -- this value is
    embedded in the observation, which content_sha256 hashes with plain
    json.dumps, so a set crashed the whole evaluation (same defect as the
    airline_113_state#b1 frozenset) the first time a target event matched."""
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None
    history = parsed.get("payment_history") if isinstance(parsed, Mapping) else None
    if not isinstance(history, list):
        return None
    return sorted({
        entry["payment_id"]
        for entry in history
        if isinstance(entry, Mapping) and isinstance(entry.get("payment_id"), str)
    })


def _extract_payment_reuse_observations(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    argument_path: str,
    argument_is_list: bool,
) -> dict[str, Any]:
    """Projects the target call's payment_id(s) alongside the set of payment_ids
    used on the most recently visible cancelled reservation -- read from whichever
    of cancel_reservation/get_reservation_details was called most recently before
    the target event (see _CANCELLED_RESERVATION_LOOKUP_TOOLS docstring above).
    """
    event_filter = runtime.get("event_filter") or {}
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        lookup_call = _most_recent_prior_tool_call(
            events, event.get("event_index", -1), _CANCELLED_RESERVATION_LOOKUP_TOOLS
        )
        cancelled_payment_ids = None
        if lookup_call is not None:
            content = _tool_result_content_for_call(events, lookup_call.get("tool_call_id"))
            cancelled_payment_ids = _payment_ids_from_reservation_json(content)
        argument_value = (event.get("arguments") or {}).get(argument_path)
        # Section 188: sorted lists, never sets (see _payment_ids_from_
        # reservation_json) -- the predicate does its own set arithmetic.
        if argument_is_list:
            used_payment_ids = sorted({
                item["payment_id"]
                for item in argument_value or []
                if isinstance(item, Mapping) and isinstance(item.get("payment_id"), str)
            }) if isinstance(argument_value, list) else None
        else:
            used_payment_ids = [argument_value] if isinstance(argument_value, str) else None
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {
                    "used_payment_ids": used_payment_ids,
                    "cancelled_payment_ids": cancelled_payment_ids,
                },
            }
        )
    return {"status": "observed", "matches": matches}


def _most_recent_prior_tool_call_for_reservation(
    events: Sequence[Mapping[str, Any]],
    before_event_index: Any,
    tool_names: Sequence[str],
    reservation_id: str,
) -> Mapping[str, Any] | None:
    """Like _most_recent_prior_tool_call, but also requires the call's own
    tool_result to be about the same reservation_id -- a trajectory can involve
    more than one reservation, so "most recent of this tool" alone (as 037/038/133
    use) is not enough here; get_reservation_details/book_reservation both return a
    Reservation whose own `reservation_id` field says which one it is.
    """
    names = set(tool_names)
    candidates = []
    for event in events:
        if (
            event.get("event_kind") != "assistant_tool_call"
            or event.get("tool_name") not in names
            or event.get("event_index", -1) >= before_event_index
        ):
            continue
        content = _tool_result_content_for_call(events, event.get("tool_call_id"))
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, Mapping) and parsed.get("reservation_id") == reservation_id:
            candidates.append(event)
    return max(candidates, key=lambda event: event.get("event_index", -1), default=None)


def _reservation_int_field_from_json(content: Any, field: str) -> int | None:
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None
    value = parsed.get(field) if isinstance(parsed, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _most_recent_prior_tool_call_correlated(
    events: Sequence[Mapping[str, Any]],
    before_event_index: Any,
    tool_names: Sequence[str],
    correlate: Mapping[str, str] | None,
    target_arguments: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Generalizes _most_recent_prior_tool_call_for_reservation to any
    correlating argument pair, not just reservation_id, and to correlating on
    the PRIOR call's own arguments (not its result) -- e.g. matching a later
    enable_roaming(line_id=X) to the most recent prior get_details_by_id(id=X)
    by their shared identifier, before ever looking at get_details_by_id's
    result. When `correlate` is None, any prior call to tool_names qualifies
    (single-subject-per-conversation assumption -- e.g. retail's
    get_user_details(user_id) has no matching argument on cancel_pending_order
    to correlate against; tau2 tasks are one customer per conversation).
    """
    names = set(tool_names)
    target_value = None
    if correlate:
        target_value = target_arguments.get(correlate["target_arg"])
        if target_value is None:
            return None
    candidates = []
    for event in events:
        if (
            event.get("event_kind") != "assistant_tool_call"
            or event.get("tool_name") not in names
            or event.get("event_index", -1) >= before_event_index
        ):
            continue
        if correlate:
            source_value = (event.get("arguments") or {}).get(correlate["source_arg"])
            if source_value != target_value:
                continue
        candidates.append(event)
    return max(candidates, key=lambda event: event.get("event_index", -1), default=None)


def _json_path_value(content: Any, path: str) -> Any:
    """Extracts a dotted path from a tool_result's JSON content; "" (empty
    path) returns the whole parsed content, for tools like
    find_user_id_by_email whose result IS the id, not an object containing it.

    Real bug found and fixed in this round (see docs/agentcoveragetesting_
    reuse_log.md section 94): the real tau2 retail find_user_id_by_email/
    find_user_id_by_name_zip tools both really return a bare Python str
    (confirmed against their real signatures, `-> str`), and this project's
    real transcript capture stores that bare string as tool_result content
    UNQUOTED (e.g. `noah_brown_6181`), not as JSON-encoded text (`"noah_
    brown_6181"`) -- so json.loads(content) always raised and this function
    always silently returned None for these tools' real results, regardless
    of the real value, making every real prior_call_consistency check
    sourced from them unconditionally fail. Only the "" (whole-content)
    path case can sensibly fall back to the raw string -- a non-empty path
    genuinely means "dig into a real JSON object", and content that isn't
    real JSON has no real field to dig into.
    """
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return content if not path else None
    if not path:
        return parsed
    value: Any = parsed
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _single_prior_call_consistency_observation(
    events: Sequence[Mapping[str, Any]], event: Mapping[str, Any], predicate: Mapping[str, Any]
) -> dict[str, Any]:
    """Computes one {"prior_call_found", "prior_value", "target_value"}
    observation for a single (event, predicate-params) pair. Factored out of
    _extract_prior_call_consistency_observations (section 115, docs/
    agentcoveragetesting_reuse_log.md) so the same lookup logic can run once
    for a predicate's primary source_tools/target_arg params and again, with
    an independently-parametrized dict, for each of predicate['alternatives']
    -- see that function's docstring for why alternatives exist.

    target_value is sourced one of two ways:
    - predicate['target_arg']: the target event's OWN argument by that name
      (original, sole behavior before section 115) -- e.g. cancel_pending_
      order's own order_id.
    - predicate['identity_source_tools'] (+ optional 'identity_result_path'):
      NEW in section 115 -- instead of an argument on the target call itself,
      looks up the most recent prior call to one of identity_source_tools
      (single-subject-per-conversation assumption, same as source_tools with
      correlate=None) and extracts identity_result_path from ITS result. This
      is what lets an alternative compare two independently-observed prior
      facts to each other (e.g. get_order_details(order_id).user_id vs.
      find_user_id_by_email(...)'s returned user_id) instead of comparing one
      prior fact to the target call's own argument -- the target call
      (e.g. cancel_pending_order) may not even carry a user_id argument to
      compare against.

    prior_value/prior_call_found are sourced one of two ways:
    - predicate['source_tools'] (+ optional 'correlate'): the most recent
      STRICTLY EARLIER matching call's result at result_path (original, sole
      behavior before section 118).
    - predicate['prior_call_is_target_event'] (NEW in section 118, docs/
      agentcoveragetesting_reuse_log.md): reads result_path from the TARGET
      event's OWN real tool_result instead of an earlier call -- for a
      branch whose target action IS the tool the evidence must come from
      (retail_083_state#b0's target is get_order_details itself), where a
      self-referencing source_tools=[<same tool>] alternative would be
      circular/degenerate (see this branch's own comment below).
    """
    event_filter_arguments = event.get("arguments") or {}
    source_tools = predicate.get("source_tools") or []
    correlate = predicate.get("correlate")
    result_path = predicate.get("result_path", "")
    target_arg = predicate.get("target_arg")
    identity_source_tools = predicate.get("identity_source_tools")
    if predicate.get("prior_call_is_target_event"):
        # Section 118 addition (docs/agentcoveragetesting_reuse_log.md): for a
        # branch whose OWN target action IS the tool call the evidence must
        # come from (retail_083_state#b0's target is get_order_details
        # itself -- unlike the 6 section-115 branches, whose target is a
        # SEPARATE write tool like cancel_pending_order), there is no non-
        # circular STRICTLY EARLIER call to treat as "prior": _most_recent_
        # prior_tool_call_correlated's event_index < before_event_index guard
        # means a self-referencing source_tools=["get_order_details"]
        # alternative would only ever match a genuinely earlier, REDUNDANT
        # duplicate call to get_order_details for the same order -- confirmed
        # empirically against tau2's own 114 real gold retail trajectories
        # that compliant agents essentially never make that duplicate call.
        # This reads result_path directly from the TARGET event's own real
        # tool_result (the event currently being evaluated) instead -- non-
        # circular because the fact being read (order_id's real owning
        # user_id, from get_order_details' own response) is not being used
        # to justify the very call that produced it; it is compared against
        # an entirely independent fact (the real authenticated identity),
        # established via a separate identity_source_tools lookup below, not
        # against anything about this same call's own request. The event
        # being evaluated always genuinely happened, so prior_call_found is
        # unconditionally True in this mode -- a missing/malformed real
        # tool_result still surfaces as prior_value=None, which will
        # correctly fail any real equals_argument/member_of_list comparison
        # rather than being silently treated as "no evidence".
        lookup_call: Mapping[str, Any] | None = event
        content = _tool_result_content_for_call(events, event.get("tool_call_id"))
        prior_value = _json_path_value(content, result_path)
    else:
        lookup_call = _most_recent_prior_tool_call_correlated(
            events, event.get("event_index", -1), source_tools, correlate, event_filter_arguments
        )
        prior_value = None
        if lookup_call is not None:
            content = _tool_result_content_for_call(events, lookup_call.get("tool_call_id"))
            prior_value = _json_path_value(content, result_path)
    if identity_source_tools:
        identity_result_path = predicate.get("identity_result_path", "")
        identity_call = _most_recent_prior_tool_call_correlated(
            events, event.get("event_index", -1), identity_source_tools, None, {}
        )
        target_value = None
        if identity_call is not None:
            identity_content = _tool_result_content_for_call(events, identity_call.get("tool_call_id"))
            target_value = _json_path_value(identity_content, identity_result_path)
    else:
        target_value = event_filter_arguments.get(target_arg) if target_arg else None
    return {
        "prior_call_found": lookup_call is not None,
        "prior_value": prior_value,
        "target_value": target_value,
    }


def _extract_prior_call_consistency_observations(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    predicate: Mapping[str, Any],
    alternatives: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """For each matched target tool call, locates the most recent prior call to
    one of predicate['source_tools'] (optionally correlated via
    predicate['correlate']), extracts predicate['result_path'] from its real
    tool_result, and packages it alongside the target call's own
    predicate['target_arg'] argument for the "prior_call_consistency" predicate
    evaluator (see evaluate_predicate_extended) to compare.

    This is Family J (docs/agentcoveragetesting_reuse_log.md section 29): real
    telecom/retail candidates where the state-changing tool itself never
    checks or raises for the claimed precondition (so Family A/
    tool_call_error_matches has no real error text to match), but the real
    precondition IS independently observable through an earlier, real tool
    call the agent itself could have made -- e.g. a line's real status via
    get_details_by_id before enable_roaming/refuel_data, or an order/payment-
    method/user id's real membership in get_user_details'/
    find_user_id_by_*'s own prior result. The check is about whether the
    LATER call is consistent with what an earlier, real call already
    revealed -- not about whether the later call's own tool enforces it.

    Section 115 addition (docs/agentcoveragetesting_reuse_log.md): a real,
    honest limitation found by online verification (section 112.9) is that
    this predicate's single source_tools/result_path/check/target_arg shape
    only recognizes ONE evidence path -- e.g. retail's 6 order-ownership
    branches only accepted get_user_details(user_id).orders containing the
    target order_id, not the equally legitimate alternative of calling
    get_order_details(order_id) directly and checking ITS OWN returned
    user_id field against the authenticated user's real identity (established
    via find_user_id_by_email/find_user_id_by_name_zip, which tau2 retail's
    policy.md line 10 mandates at the start of every conversation). Rather
    than special-casing these 6 branches, `alternatives` (from the compiled
    program's OPTIONAL sibling `predicate_alternatives` key -- see
    compile_oracle_evaluator_contracts_extended's PRIOR_CALL_CONSISTENCY_
    ALTERNATIVES_RULES lookup -- deliberately NOT a key inside `predicate`
    itself, so program["predicate"] stays byte-identical to before this
    addition for every requirement, including the 6 target branches) is a
    generic, opt-in list of independently-parametrized observation dicts
    (same shape as the primary source_tools/correlate/result_path/check/
    target_arg params, plus the new identity_source_tools/identity_result_path
    fields -- see _single_prior_call_consistency_observation) each evaluated
    the same way and OR-combined with the primary check in
    evaluate_predicate_extended. No `alternatives` passed behaves byte-for-
    byte as before -- retail_087/096 and every telecom prior_call_consistency
    rule have none configured and are unaffected.

    Section 118 addition (docs/agentcoveragetesting_reuse_log.md):
    retail_083_state#b0 shares the section 115 limitation's shape but not its
    fix -- its OWN target action is get_order_details itself, so a section-
    115-style alternative (source_tools=["get_order_details"], correlated on
    order_id) would need a genuinely EARLIER call to the very same tool,
    which real compliant agents essentially never make (confirmed against
    tau2's 114 real gold retail trajectories) -- section 112.5/115.8
    correctly flagged this as circular/degenerate and deliberately left it
    unfixed. Real gold-trajectory evidence (10/64 real tasks that call
    get_order_details never call get_user_details first) confirmed the
    underlying false-fail risk is real, so retail_083 now gets its OWN
    alternative using the new `prior_call_is_target_event` flag (see
    _single_prior_call_consistency_observation) -- reads the evidence from
    the target get_order_details call's OWN result instead of an earlier
    call, non-circular because it is compared against an independently-
    established identity, not against anything about the same call's own
    request. retail_087/096 remain unaffected (their claim is about user_id
    ownership itself, not order_id ownership -- get_order_details's user_id
    field was independently judged NOT to constitute the same kind of
    evidence for them; see reuse log section 118).
    """
    event_filter = runtime.get("event_filter") or {}
    alternatives = list(alternatives) if alternatives else []
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        primary = _single_prior_call_consistency_observation(events, event, predicate)
        value: dict[str, Any] = dict(primary)
        if alternatives:
            value["alternatives"] = [
                {"check": alt.get("check"), **_single_prior_call_consistency_observation(events, event, alt)}
                for alt in alternatives
            ]
        matches.append({
            "event_index": event.get("event_index"),
            "source_message_index": event.get("source_message_index"),
            "value": value,
        })
    return {"status": "observed", "matches": matches}


def _semantic_transcript_excerpt_up_to(
    events: Sequence[Mapping[str, Any]], anchor_event_index: int
) -> list[dict[str, Any]]:
    """Section 134 bug 3 (telecom_090_order#b0::OR01, docs/agentcoveragetesting_
    reuse_log.md section 131.2/131.6/134.3): real, confirmed evidence-
    construction bug -- the transcript_excerpt this used to build (inline,
    now factored out here) filtered events to `event_kind in ("user_message",
    "assistant_message")` ONLY, which silently drops every
    assistant_tool_call and tool_result event from the judge's view
    entirely, REGARDLESS of event_index -- not a narrow off-by-one in the
    `<=` comparison itself (that comparison is correct), but the categorical
    exclusion had the same real, observed effect the ticket described as
    "off-by-one": for telecom_090_order#b0, the anchor event (the real
    get_details_by_id assistant_tool_call this whole check is about) is
    itself an assistant_tool_call, so it -- and every earlier tool call/
    result -- silently never appeared in transcript_excerpt at all, leaving
    only the 2 leading user/assistant text messages. The real judge's own
    `predicate_evaluations[0].reason` ("transcript does not show any call to
    get_details_by_id") was a materially correct description of what it was
    actually shown, just not of the real transcript. Family M's own
    criterion text (see this predicate's docstring) explicitly needs to know
    which tool was called AND what it returned ("the REAL type of object
    that was actually returned") to judge consistency at all, so this
    includes both assistant_tool_call (as a role=assistant tool_call, same
    shape driver/generic_tau_online_v1.py's own
    _transcript_excerpt_for_judge already uses for its own, separate judge
    lane) and tool_result (role=tool, content=the real return value) --
    strictly a widening of what the judge is shown, never a narrowing, so no
    check that used to see a real, sufficient excerpt can regress.
    """
    excerpt = []
    for event in events:
        if event.get("event_index", -1) > anchor_event_index:
            continue
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


def _extract_semantic_transcript_observations(
    runtime: Mapping[str, Any], events: Sequence[Mapping[str, Any]], predicate: Mapping[str, Any]
) -> dict[str, Any]:
    """Family M (docs/agentcoveragetesting_reuse_log.md section 35): real
    telecom/retail candidates whose real subject is either (a) whether a
    free-text tool argument (summary/reason) has a semantic property no
    regex can check (accurate to the actual conversation, not a generic
    placeholder, contains no PII) or (b) whether the agent's own dialogue
    behavior satisfied a procedural requirement (confirmed something with
    the user, didn't fabricate a value) -- neither is a fact about a single
    tool_call's structure the way Family A/J/K's predicates are, so this
    projects the target call's own predicate['target_arg'] value (if any)
    together with the real conversation transcript up to that point, for a
    judge (see evaluate_predicate_extended's "semantic_transcript_judgment"
    branch) to assess against predicate['criterion'] -- a real natural-
    language description of what must hold, not a regex or fixed value.
    """
    event_filter = runtime.get("event_filter") or {}
    target_arg = predicate.get("target_arg")
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        target_arguments = event.get("arguments") or {}
        transcript_excerpt = _semantic_transcript_excerpt_up_to(events, event.get("event_index", -1))
        matches.append({
            "event_index": event.get("event_index"),
            "source_message_index": event.get("source_message_index"),
            "value": {
                "target_value": target_arguments.get(target_arg) if target_arg else None,
                "transcript_excerpt": transcript_excerpt,
            },
        })
    return {"status": "observed", "matches": matches}


def _extract_cumulative_sum_observations(
    runtime: Mapping[str, Any], events: Sequence[Mapping[str, Any]], predicate: Mapping[str, Any]
) -> dict[str, Any]:
    """Family N (docs/agentcoveragetesting_reuse_log.md section 39): a real
    trajectory-level aggregate check -- unlike every other predicate here,
    the constraint isn't a fact about ONE call's own argument, or a diff
    between exactly two calls (baggage_count_decreased/passenger_count_
    changed) -- it's "the running SUM of predicate['sum_arg'] across every
    real call to this SAME tool sharing the same real value of
    predicate['scope_arg'], up to and including this one, must not exceed
    predicate['threshold']". The target tool itself never tracks or checks
    this (e.g. real refuel_data has no session-cumulative cap at all -- each
    call only validates its own gb_amount is positive), so there is no
    tool-side error text (Family A) and no single prior call whose result
    settles it (Family J) -- the only real evidence is the trajectory's own
    sequence of matching tool calls.
    """
    event_filter = runtime.get("event_filter") or {}
    scope_arg = predicate.get("scope_arg")
    sum_arg = predicate.get("sum_arg")
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        target_arguments = event.get("arguments") or {}
        scope_value = target_arguments.get(scope_arg) if scope_arg else None
        cumulative_sum = 0.0
        for other in events:
            if (
                other.get("event_kind") != "assistant_tool_call"
                or other.get("tool_name") != event.get("tool_name")
                or other.get("event_index", -1) > event.get("event_index", -1)
            ):
                continue
            other_arguments = other.get("arguments") or {}
            if scope_arg and other_arguments.get(scope_arg) != scope_value:
                continue
            value = other_arguments.get(sum_arg)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cumulative_sum += value
        matches.append({
            "event_index": event.get("event_index"),
            "source_message_index": event.get("source_message_index"),
            "value": {
                "cumulative_sum": cumulative_sum,
                "current_value": target_arguments.get(sum_arg),
                "scope_value": scope_value,
            },
        })
    return {"status": "observed", "matches": matches}


def _extract_call_count_observations(
    runtime: Mapping[str, Any], events: Sequence[Mapping[str, Any]], predicate: Mapping[str, Any]
) -> dict[str, Any]:
    """Family O (docs/agentcoveragetesting_reuse_log.md section 77): a real
    "must not call X more than N times" cardinality bound, scoped by a real
    argument (e.g. order_id) the same way cumulative_sum_le scopes its
    running sum -- for each real matching call, count every matching call to
    the SAME tool sharing the SAME real scope value, up to and including
    this one. Mirrors _extract_cumulative_sum_observations exactly, just
    counting occurrences instead of summing a numeric argument.
    """
    event_filter = runtime.get("event_filter") or {}
    scope_arg = predicate.get("scope_arg")
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        target_arguments = event.get("arguments") or {}
        scope_value = target_arguments.get(scope_arg) if scope_arg else None
        cumulative_count = 0
        for other in events:
            if (
                other.get("event_kind") != "assistant_tool_call"
                or other.get("tool_name") != event.get("tool_name")
                or other.get("event_index", -1) > event.get("event_index", -1)
            ):
                continue
            other_arguments = other.get("arguments") or {}
            if scope_arg and other_arguments.get(scope_arg) != scope_value:
                continue
            cumulative_count += 1
        matches.append({
            "event_index": event.get("event_index"),
            "source_message_index": event.get("source_message_index"),
            "value": {
                "cumulative_count": cumulative_count,
                "scope_value": scope_value,
            },
        })
    return {"status": "observed", "matches": matches}


def _extract_tool_call_any_of_observations(
    runtime: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Family P (docs/agentcoveragetesting_reuse_log.md section 79): a real
    disjunctive "the agent must call X, OR Y" tool_call existence check --
    e.g. retail_053_order_find_user_id_by_email#b0's real Then ("locating
    their user id via email, or via name and zip code"), whose accepted
    candidate only ever named ONE of the two named-valid methods
    (find_user_id_by_name_zip), so an agent that legitimately used the OTHER
    one (find_user_id_by_email) was wrongly scored fail. The frozen
    extractor's event_filter is exact-match only (field_equals one
    tool_name); this reuses the "matcher": {"kind": "tool_name_in",
    "tool_names": [...]} convention this module already established for
    temporal_relation's disjunctive right_event (_right_event_content_
    matches, section 49), applied here to a plain (non-temporal) existence
    check instead. event_filter itself stays bare ({"event_kind":
    "assistant_tool_call"}, no field_equals) so driver_plan_lowering_v1.py's
    _event_anchor still gets a real Mapping to anchor Step6's abstract
    observation plan to -- the actual disjunction lives in `matcher`, read
    here, not in event_filter.
    """
    matcher = runtime.get("matcher") or {}
    tool_names = set(matcher.get("tool_names") or [])
    matches = []
    for event in events:
        if event.get("event_kind") != "assistant_tool_call":
            continue
        if event.get("tool_name") not in tool_names:
            continue
        matches.append({
            "event_index": event.get("event_index"),
            "source_message_index": event.get("source_message_index"),
            "value": {"matched_tool_name": event.get("tool_name")},
        })
    return {"status": "observed", "matches": matches}


def _extract_baggage_reduction_observations(
    runtime: Mapping[str, Any], events: Sequence[Mapping[str, Any]], argument_path: str
) -> dict[str, Any]:
    """Projects the target update_reservation_baggages call's new baggage count
    alongside the same reservation's prior value, read from the most recent
    get_reservation_details/book_reservation call for that SAME reservation_id
    before the target event.
    """
    event_filter = runtime.get("event_filter") or {}
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        arguments = event.get("arguments") or {}
        reservation_id = arguments.get("reservation_id")
        new_value = arguments.get(argument_path)
        old_value = None
        if isinstance(reservation_id, str):
            lookup_call = _most_recent_prior_tool_call_for_reservation(
                events, event.get("event_index", -1), _RESERVATION_LOOKUP_TOOLS, reservation_id
            )
            if lookup_call is not None:
                content = _tool_result_content_for_call(events, lookup_call.get("tool_call_id"))
                old_value = _reservation_int_field_from_json(content, argument_path)
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {"new_value": new_value, "old_value": old_value},
            }
        )
    return {"status": "observed", "matches": matches}


def _extract_passenger_reduction_observations(
    runtime: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """airline_089_norm#b0v1: like _extract_baggage_reduction_observations, but
    `passengers` is a List[Passenger], not a plain int field (confirmed against
    real tau2 source -- update_reservation_passengers/book_reservation both take
    `passengers: List[Passenger | dict]`), so both sides need a length, not a
    direct field read. The new-value side is len(arguments["passengers"]); the
    old-value side reuses _passenger_count_from_reservation_json (already
    written for 037/038's compensation-formula passenger-count lookup) against
    the most recent prior get_reservation_details/book_reservation for the SAME
    reservation_id -- same cross-event lookup as baggage, just a different leaf
    shape. See docs/oracle_requirement_pipeline_v0_7.md section 24.
    """
    event_filter = runtime.get("event_filter") or {}
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        arguments = event.get("arguments") or {}
        reservation_id = arguments.get("reservation_id")
        passengers = arguments.get("passengers")
        new_value = len(passengers) if isinstance(passengers, list) else None
        old_value = None
        if isinstance(reservation_id, str):
            lookup_call = _most_recent_prior_tool_call_for_reservation(
                events, event.get("event_index", -1), _RESERVATION_LOOKUP_TOOLS, reservation_id
            )
            if lookup_call is not None:
                content = _tool_result_content_for_call(events, lookup_call.get("tool_call_id"))
                old_value = _passenger_count_from_reservation_json(content)
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {"new_value": new_value, "old_value": old_value},
            }
        )
    return {"status": "observed", "matches": matches}


def _flight_segment_key(item: Any) -> list[Any] | None:
    """Section 188: a 2-item list, not a tuple. The keys are embedded in the
    observation and the predicate result; a tuple hashes the same through
    content_sha256 but reads back from the stored JSON as a list, so the
    in-memory evaluation differed from its own persisted copy. Lists compare
    and sort exactly like tuples here, so verdicts and fingerprints are
    unchanged."""
    if not isinstance(item, Mapping):
        return None
    flight_number = item.get("flight_number")
    date = item.get("date")
    if flight_number is None or date is None:
        return None
    return [flight_number, date]


def _extract_cabin_unchanged_observations(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Section 193 (airline_087_state#b0): the submitted `cabin` of every
    matching update call, next to the reservation's cabin as the most recent
    prior get_reservation_details/book_reservation for the SAME reservation_id
    reported it (the flights_unchanged_from_reservation lookup shape). Unlike
    that extractor this one honors the binding's scope_constraints (narrowed
    to the bound reservation by _with_prohibition_observation_scope), because
    the Given is a fact about THAT reservation."""
    event_filter = runtime.get("event_filter") or {}
    constraints = runtime.get("scope_constraints") or []
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        matched, missing = _scope_matches(event, constraints, driver_bindings)
        if missing:
            return {"status": "unavailable", "missing_driver_bindings": sorted(set(missing))}
        if not matched:
            continue
        arguments = event.get("arguments") or {}
        reservation_id = arguments.get("reservation_id")
        old_cabin = None
        if isinstance(reservation_id, str):
            lookup_call = _most_recent_prior_tool_call_for_reservation(
                events, event.get("event_index", -1), _RESERVATION_LOOKUP_TOOLS, reservation_id
            )
            if lookup_call is not None:
                content = _tool_result_content_for_call(events, lookup_call.get("tool_call_id"))
                try:
                    parsed = json.loads(content) if isinstance(content, str) else None
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, Mapping) and isinstance(parsed.get("cabin"), str):
                    old_cabin = parsed["cabin"]
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {"reservation_id": reservation_id, "new_cabin": arguments.get("cabin"), "old_cabin": old_cabin},
            }
        )
    return {"status": "observed", "matches": matches}


# Section 193 (task_918eba9d): the arguments that name the entity an operation
# acts ON. A prohibition's scope_constraints pin EVERY scope parameter to the
# bound values, so a violating call that differs in any other argument
# (another payment_id, other flights, another address, another amount, another
# route, ...) was never observed. For a check that requires no observation,
# only these entity arguments keep scoping it -- and never the argument the
# check itself projects (its value IS what is being judged).
_PROHIBITION_TARGET_ENTITY_ARGUMENTS = frozenset(
    {"reservation_id", "order_id", "user_id", "customer_id", "line_id", "bill_id"}
)
# predicate_kinds whose extension extractor ignores scope_constraints entirely
# (every branch of _evaluate_oracle_contract_extended_core's dispatch chain
# except cabin_unchanged_from_reservation, plus the extraction-free
# structurally_unreachable_action; a test re-derives it from the source).
# Narrowing their scope would change nothing but the program bytes.
_SCOPE_IGNORING_PREDICATE_KINDS = frozenset({
    "airport_in_network", "baggage_count_decreased", "call_count_le", "compensation_formula",
    "cumulative_sum_le", "flight_path_valid", "flight_path_valid_from_reservation",
    "flight_route_field_unchanged", "flights_chronologically_feasible", "flights_unchanged_from_reservation",
    "no_concurrent_message_and_tool_call", "passenger_count_changed", "payment_reused_from_cancelled_reservation",
    "prior_call_consistency", "semantic_transcript_judgment", "structurally_unreachable_action",
    "tool_call_any_of", "tool_call_error_matches",
})


def _projected_argument_names(runtime: Mapping[str, Any]) -> set[str]:
    projection = runtime.get("projection") or {}
    paths = [projection["path"]] if isinstance(projection.get("path"), str) else []
    paths += [p for p in projection.get("paths") or [] if isinstance(p, str)]
    return {p.removeprefix("arguments.") for p in paths}


def _with_prohibition_observation_scope(record: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    """Section 193: declare, in the program, which of a prohibition's scope
    constraints still select the observed calls. Obligations (requires_
    observation true) keep every pin -- there they identify the requested
    operation. The binding itself is untouched: its full scope still tells the
    driver which facts the user states. Only programs whose evaluation path
    really filters by scope (the frozen evaluator, or an extension predicate
    evaluated through extract_bound_observations) are annotated."""
    program = record.get("program")
    if record.get("evaluator_status") != "executable" or not isinstance(program, Mapping):
        return dict(record)
    if (record.get("expected_observation") or {}).get("requires_observation"):
        return dict(record)
    predicate_kind = (program.get("predicate") or {}).get("predicate_kind")
    if program.get("program_source") == PROGRAM_SOURCE and predicate_kind in _SCOPE_IGNORING_PREDICATE_KINDS:
        return dict(record)
    runtime = binding.get("runtime_binding") or {}
    projected = _projected_argument_names(runtime)
    kept, dropped = [], []
    for constraint in runtime.get("scope_constraints") or []:
        path = str(constraint.get("actual_path") or "")
        name = path.removeprefix("arguments.")
        entity = path.startswith("arguments.") and name in _PROHIBITION_TARGET_ENTITY_ARGUMENTS and name not in projected
        (kept if entity else dropped).append(path)
    if not dropped:
        return dict(record)
    record = deepcopy(dict(record))
    record.pop("evaluator_contract_fingerprint", None)
    record["program"] = {**record["program"], "observation_scope": {"kept_scope_paths": kept, "dropped_scope_paths": dropped}}
    record["diagnostics"] = list(record.get("diagnostics") or []) + [
        {"code": "compiled_prohibition_observation_scope"},
        {"code": "compiled_by_extension_v1"},
    ]
    record["evaluator_contract_fingerprint"] = content_sha256(record)
    return record


def _with_cabin_unchanged_program(
    record: Mapping[str, Any], requirement: Mapping[str, Any], tables: Mapping[str, Any]
) -> dict[str, Any]:
    """Section 193: replace the frozen enum program of a CABIN_UNCHANGED_RULES
    requirement (only when the requirement's own tool is the rule's target
    tool) with cabin_unchanged_from_reservation."""
    rule = _lookup_requirement_rule(tables.get("CABIN_UNCHANGED_RULES") or {}, requirement.get("requirement_text"))
    tool = (requirement.get("observation_contract") or {}).get("tool_name")
    if rule is None or rule.get("target_tool_name") != tool or record.get("evaluator_status") != "executable":
        return dict(record)
    record = deepcopy(dict(record))
    record.pop("evaluator_contract_fingerprint", None)
    record["program"] = {
        "evaluator_kind": "all_matches_predicate",
        "requires_observation": bool((record.get("expected_observation") or {}).get("requires_observation")),
        "predicate": {"predicate_kind": "cabin_unchanged_from_reservation"},
        "program_source": PROGRAM_SOURCE,
    }
    record["diagnostics"] = list(record.get("diagnostics") or []) + [
        {"code": "compiled_cabin_unchanged_grammar"},
        {"code": "compiled_by_extension_v1"},
    ]
    record["evaluator_contract_fingerprint"] = content_sha256(record)
    return record


def _narrowed_runtime_binding(runtime_binding: Mapping[str, Any], program: Mapping[str, Any]) -> Mapping[str, Any]:
    scope = program.get("observation_scope")
    if not isinstance(scope, Mapping):
        return runtime_binding
    kept = set(scope.get("kept_scope_paths") or [])
    narrowed = deepcopy(dict(runtime_binding))
    runtime = dict(narrowed.get("runtime_binding") or {})
    runtime["scope_constraints"] = [c for c in runtime.get("scope_constraints") or [] if c.get("actual_path") in kept]
    narrowed["runtime_binding"] = runtime
    return narrowed


def _extract_flights_unchanged_observations(
    runtime: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """airline_102_state#b0: "can change cabin WITHOUT changing the flights" --
    verified against real tau2 update_reservation_flights (tools.py:592-660):
    it decides whether an incoming flight_info is the SAME existing segment by
    matching flight_number+date (plus cabin == reservation.cabin, which is
    exactly the thing changing here, so that side of the tool's own match
    never fires when cabin changes -- the tool re-validates/re-prices every
    segment as if new). "Unchanged flights" is therefore a real claim about
    the submitted `flights` argument's (flight_number, date) set matching the
    reservation's PRE-existing flights set exactly (order-independent -- nothing
    in the real Then or tool semantics cares about array order), read from the
    most recent get_reservation_details/book_reservation for the same
    reservation_id, same cross-event lookup shape as baggage/passenger-count
    above.
    """
    event_filter = runtime.get("event_filter") or {}
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        arguments = event.get("arguments") or {}
        reservation_id = arguments.get("reservation_id")
        new_flights = arguments.get("flights")
        new_keys = (
            sorted(filter(None, (_flight_segment_key(item) for item in new_flights)))
            if isinstance(new_flights, list)
            else None
        )
        old_keys = None
        if isinstance(reservation_id, str):
            lookup_call = _most_recent_prior_tool_call_for_reservation(
                events, event.get("event_index", -1), _RESERVATION_LOOKUP_TOOLS, reservation_id
            )
            if lookup_call is not None:
                content = _tool_result_content_for_call(events, lookup_call.get("tool_call_id"))
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                    except (TypeError, ValueError):
                        parsed = None
                    old_flights = parsed.get("flights") if isinstance(parsed, Mapping) else None
                    if isinstance(old_flights, list):
                        old_keys = sorted(
                            filter(None, (_flight_segment_key(item) for item in old_flights))
                        )
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {"new_flights": new_keys, "old_flights": old_keys},
            }
        )
    return {"status": "observed", "matches": matches}


def _extract_turn_shape_observations(
    runtime: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Groups every event by source_message_index (normalize_tau_messages tags
    every event derived from one raw message with the same index) and reports,
    per turn, whether that turn produced an assistant_message and/or an
    assistant_tool_call event -- the co-occurrence airline_073_order#b0v0/v1's
    Then forbids. Ignores runtime.get("event_filter") entirely (there is none
    for this extractor_kind) since the check is turn-shaped, not one-event-
    shaped; every match is one turn, not one tool call.
    """
    by_message: dict[int, dict[str, bool]] = {}
    for event in events:
        index = event.get("source_message_index")
        if index is None:
            continue
        group = by_message.setdefault(index, {"has_message": False, "has_tool_call": False})
        if event.get("event_kind") == "assistant_message":
            group["has_message"] = True
        elif event.get("event_kind") == "assistant_tool_call":
            group["has_tool_call"] = True
    matches = [
        {"event_index": index, "source_message_index": index, "value": value}
        for index, value in sorted(by_message.items())
    ]
    return {"status": "observed", "matches": matches}


def _flight_numbers_from_argument(flights_argument: Any) -> list[str] | None:
    """Extracts the ordered flight_number list from a `flights` tool-call argument
    (a list of {flight_number, date} objects, per tau2's FlightInfo shape). Returns
    None on any malformed shape -- not a list, or an element missing a string
    flight_number -- so the caller can distinguish "couldn't even read the
    argument" from "read it, and it does/doesn't resolve".
    """
    if not isinstance(flights_argument, list):
        return None
    flight_numbers = []
    for item in flights_argument:
        if not isinstance(item, Mapping):
            return None
        flight_number = item.get("flight_number")
        if not isinstance(flight_number, str):
            return None
        flight_numbers.append(flight_number)
    return flight_numbers


def _resolve_flight_path(
    flight_numbers: list[str] | None, route_table: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    """Resolves an ordered flight_number list to its real origin/destination per
    segment via the static route table. Deliberately does only resolution here --
    connectivity/shape validity is a separate, purpose-specific step per caller (see
    _validate_flight_path_shape for airline_050_arg#b0, and the field_unchanged
    predicate branch below for airline_031_arg#b0v0/v1/v2), since the two Thens need
    different things from the same resolved segments.
    """
    if flight_numbers is None:
        return {"status": "malformed_flights_argument", "segments": []}
    if not flight_numbers:
        return {"status": "empty", "segments": []}
    unknown = sorted({fn for fn in flight_numbers if fn not in route_table})
    if unknown:
        return {"status": "unknown_flight_number", "unknown_flight_numbers": unknown, "segments": []}
    segments = [
        {
            "flight_number": fn,
            "origin": route_table[fn]["origin"],
            "destination": route_table[fn]["destination"],
        }
        for fn in flight_numbers
    ]
    return {"status": "resolved", "segments": segments}


def _validate_flight_path_shape(
    segments: list[dict[str, str]], origin: Any, destination: Any, flight_type: Any
) -> dict[str, Any]:
    """Checks whether a resolved segment chain forms a valid path from `origin` to
    `destination` (and back, if `flight_type == "round_trip"`), with no missing or
    extraneous segments. tau2's FlightType is exactly {"one_way", "round_trip"} --
    no multi-city -- which bounds how much shape-checking is needed.

    "No missing" = every consecutive pair of segments must connect nose-to-tail
    (segment[i].destination == segment[i+1].origin); any gap fails.
    "No extraneous" = the airport sequence must not revisit any airport beyond what
    the declared shape requires: for one_way, no airport may repeat at all; for
    round_trip, `destination` must appear exactly once (the turnaround) and `origin`
    exactly twice (start and end), with no repeats WITHIN either leg. Interior stops
    shared between the outbound and return legs (e.g. the same connecting hub used
    both ways) are allowed -- the Then text only calls out origin/destination, not
    interior stops, and re-using a hub on both legs of a round trip is normal
    routing, not padding.
    """
    if not segments:
        return {"valid": False, "reason": "no_segments"}
    if not isinstance(origin, str) or not isinstance(destination, str):
        return {"valid": False, "reason": "origin_or_destination_not_observed"}
    for left, right in zip(segments, segments[1:]):
        if left["destination"] != right["origin"]:
            return {
                "valid": False,
                "reason": "connectivity_gap",
                "gap_between": [left["flight_number"], right["flight_number"]],
            }
    airports = [segments[0]["origin"]] + [s["destination"] for s in segments]
    if flight_type == "one_way":
        if airports[0] != origin or airports[-1] != destination:
            return {"valid": False, "reason": "endpoints_do_not_match_declared_origin_destination"}
        if len(set(airports)) != len(airports):
            return {"valid": False, "reason": "repeated_airport_in_one_way_path"}
        return {"valid": True}
    if flight_type == "round_trip":
        if airports[0] != origin or airports[-1] != origin:
            return {"valid": False, "reason": "path_does_not_start_and_end_at_origin"}
        destination_positions = [i for i, a in enumerate(airports) if a == destination]
        if len(destination_positions) != 1 or destination_positions[0] in (0, len(airports) - 1):
            return {"valid": False, "reason": "destination_not_visited_exactly_once_as_turnaround"}
        if airports.count(origin) != 2:
            return {"valid": False, "reason": "origin_revisited_outside_endpoints"}
        turnaround = destination_positions[0]
        outbound_leg, return_leg = airports[: turnaround + 1], airports[turnaround:]
        if len(set(outbound_leg)) != len(outbound_leg) or len(set(return_leg)) != len(return_leg):
            return {"valid": False, "reason": "repeated_airport_within_a_leg"}
        return {"valid": True}
    return {"valid": False, "reason": "unrecognized_flight_type"}


def _reservation_route_fields_from_json(content: Any) -> dict[str, str] | None:
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    origin, destination, flight_type = (
        parsed.get("origin"),
        parsed.get("destination"),
        parsed.get("flight_type"),
    )
    if (
        not isinstance(origin, str)
        or not isinstance(destination, str)
        or flight_type not in ("one_way", "round_trip")
    ):
        return None
    return {"origin": origin, "destination": destination, "trip_type": flight_type}


def _extract_flight_route_field_observations(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    target_tool_name: str,
    route_table: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """For airline_031_arg#b0v0/v1/v2: unlike every other _extract_* helper in this
    module, the semantic_requirement candidate's runtime_binding carries no usable
    event_filter (it is semantic_event_matcher_deferred, not a real tool-call
    binding), so the target call is identified directly by target_tool_name instead
    of runtime["event_filter"]. `runtime` is accepted only to keep this function's
    signature consistent with its siblings; it is otherwise unused.
    """
    del runtime
    event_filter = {"event_kind": "assistant_tool_call", "field_equals": {"tool_name": target_tool_name}}
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        arguments = event.get("arguments") or {}
        reservation_id = arguments.get("reservation_id")
        resolved = _resolve_flight_path(
            _flight_numbers_from_argument(arguments.get("flights")), route_table
        )
        old_reservation = None
        if isinstance(reservation_id, str):
            lookup_call = _most_recent_prior_tool_call_for_reservation(
                events, event.get("event_index", -1), _RESERVATION_LOOKUP_TOOLS, reservation_id
            )
            if lookup_call is not None:
                content = _tool_result_content_for_call(events, lookup_call.get("tool_call_id"))
                old_reservation = _reservation_route_fields_from_json(content)
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {"resolved": resolved, "old_reservation": old_reservation},
            }
        )
    return {"status": "observed", "matches": matches}


def _extract_flight_path_validity_observations(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    route_table: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """For airline_050_arg#b0: OR01 is a normally-bound tool_argument_constraint
    (book_reservation), so runtime["event_filter"] is real and reused directly,
    unlike the 031 extractor above.
    """
    event_filter = runtime.get("event_filter") or {}
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        arguments = event.get("arguments") or {}
        resolved = _resolve_flight_path(
            _flight_numbers_from_argument(arguments.get("flights")), route_table
        )
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {
                    "resolved": resolved,
                    "origin": arguments.get("origin"),
                    "destination": arguments.get("destination"),
                    "flight_type": arguments.get("flight_type"),
                },
            }
        )
    return {"status": "observed", "matches": matches}


def _extract_flight_path_validity_against_reservation_observations(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    route_table: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """For airline_068_arg#b0 (update_reservation_flights): unlike
    airline_050_arg#b0's book_reservation check above, update_reservation_flights's
    real signature (reservation_id, cabin, flights, payment_id) carries no
    origin/destination/flight_type arguments at all (verified against real
    tau2 source) -- the trip's origin/destination/trip_type are properties of
    the EXISTING reservation, not this call. Reuses
    _reservation_route_fields_from_json (already written for 031's
    field_unchanged mechanism) against the most recent prior
    get_reservation_details/book_reservation for the same reservation_id,
    same cross-event lookup shape used throughout this module. Produces the
    identical {"resolved", "origin", "destination", "flight_type"} value
    shape _validate_flight_path_shape expects, so no new predicate
    evaluation logic is needed -- only extraction differs from
    _extract_flight_path_validity_observations.
    """
    event_filter = runtime.get("event_filter") or {}
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        arguments = event.get("arguments") or {}
        resolved = _resolve_flight_path(
            _flight_numbers_from_argument(arguments.get("flights")), route_table
        )
        origin = destination = flight_type = None
        reservation_id = arguments.get("reservation_id")
        if isinstance(reservation_id, str):
            lookup_call = _most_recent_prior_tool_call_for_reservation(
                events, event.get("event_index", -1), _RESERVATION_LOOKUP_TOOLS, reservation_id
            )
            if lookup_call is not None:
                content = _tool_result_content_for_call(events, lookup_call.get("tool_call_id"))
                route_fields = _reservation_route_fields_from_json(content)
                if route_fields is not None:
                    origin = route_fields["origin"]
                    destination = route_fields["destination"]
                    flight_type = route_fields["trip_type"]
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {
                    "resolved": resolved,
                    "origin": origin,
                    "destination": destination,
                    "flight_type": flight_type,
                },
            }
        )
    return {"status": "observed", "matches": matches}


def _network_airports(route_table: Mapping[str, Mapping[str, str]]) -> list[str]:
    """airline_113_state#b1: the real tau2 list_all_airports() tool
    (tools.py:403-429) returns a hardcoded 20-airport list -- verified that
    this is EXACTLY the set of origin/destination values across every real
    flight in db.json (both computed independently and compared, identical
    20 codes), so the airline's real network is already fully recoverable
    from the SAME flight_route_reference this module already threads through
    for 050/068, with no separate static artifact needed.

    Section 188 (task_e31fae3b, root cause of task_237a02ff): returned a
    frozenset until then. The value is embedded in every matched
    search_direct_flight observation (see _extract_airport_in_network_
    observations), and evaluate_generic_mechanical_oracle hashes the whole
    evaluation with content_sha256 (plain json.dumps, no default=), so ANY
    run where the agent called search_direct_flight raised "Object of type
    frozenset is not JSON serializable" -- the round-8 airline_113_state#b1
    error. A sorted list is JSON-safe, deterministic (so fingerprints are
    stable across runs and PYTHONHASHSEED values) and still supports the
    predicate's membership test."""
    airports: set[str] = set()
    for record in route_table.values():
        origin = record.get("origin")
        destination = record.get("destination")
        if isinstance(origin, str):
            airports.add(origin)
        if isinstance(destination, str):
            airports.add(destination)
    return sorted(airports)


def _extract_airport_in_network_observations(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    parameter: str,
    route_table: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    event_filter = runtime.get("event_filter") or {}
    network = _network_airports(route_table)
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        arguments = event.get("arguments") or {}
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {"airport": arguments.get(parameter), "network": network},
            }
        )
    return {"status": "observed", "matches": matches}


_SCHEDULE_TIME_DAY_OFFSET_RE = re.compile(r"^(?P<time>\d{2}:\d{2}:\d{2})(?:\+(?P<offset>\d+))?$")


def _schedule_datetime(date: str, time_field: str) -> str | None:
    """Combines a `date` (YYYY-MM-DD) with a real tau2 scheduled_*_time_est
    value into a single "YYYY-MM-DD HH:MM:SS" string that sorts identically
    to real chronological order (no datetime-library comparison needed, same
    trick as date_not_before's reference-date comparison above) -- EXCEPT
    that scheduled_arrival_time_est can carry a "+N" day-rollover suffix for
    overnight flights (verified against the real db.json: 36 of 300 real
    flights have this on their arrival time, e.g. HAT002 departs 21:00:00 and
    arrives "01:30:00+1" -- the next calendar day; confirmed no real flight
    ever carries this suffix on its DEPARTURE time, and no offset other than
    "+1" appears anywhere in the real data). Naively concatenating date +
    "01:30:00+1" would silently miscompare against same-day times, so the
    offset must be resolved into an actual date roll-forward first.
    """
    match = _SCHEDULE_TIME_DAY_OFFSET_RE.match(time_field)
    if match is None:
        return None
    try:
        base_date = _date.fromisoformat(date)
    except ValueError:
        return None
    offset_days = int(match.group("offset") or 0)
    actual_date = base_date + _timedelta(days=offset_days) if offset_days else base_date
    return f"{actual_date.isoformat()} {match.group('time')}"


def _extract_flights_chronological_feasibility_observations(
    runtime: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    schedule_table: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """airline_049_arg#b0/airline_067_arg#b0: "all flights ... must be
    chronologically feasible and contiguous according to their departure and
    arrival times". Resolves each segment's real scheduled_departure_time_est/
    scheduled_arrival_time_est (from flight_schedule_reference, a static
    artifact -- see prepare_v5_step4_flight_schedule_reference_v0_1.py) via
    _schedule_datetime, which also resolves the real "+1" overnight-arrival
    day-rollover suffix. Only resolution happens here;
    evaluate_predicate_extended's flights_chronologically_feasible branch
    does the actual ordering check, same extraction/evaluation split as the
    flight-path-shape mechanism.
    """
    event_filter = runtime.get("event_filter") or {}
    matches = []
    for event in events:
        if not _event_matches_filter(event, event_filter):
            continue
        arguments = event.get("arguments") or {}
        flights = arguments.get("flights")
        segments: list[dict[str, Any]] | None = []
        if not isinstance(flights, list) or not flights:
            segments = None
        else:
            for item in flights:
                if not isinstance(item, Mapping):
                    segments = None
                    break
                flight_number = item.get("flight_number")
                date = item.get("date")
                schedule = schedule_table.get(flight_number) if isinstance(flight_number, str) else None
                departure = arrival = None
                if schedule is not None and isinstance(date, str):
                    departure = _schedule_datetime(date, schedule["scheduled_departure_time_est"])
                    arrival = _schedule_datetime(date, schedule["scheduled_arrival_time_est"])
                segments.append(
                    {
                        "flight_number": flight_number,
                        "date": date,
                        "departure": departure,
                        "arrival": arrival,
                    }
                )
        matches.append(
            {
                "event_index": event.get("event_index"),
                "source_message_index": event.get("source_message_index"),
                "value": {"segments": segments},
            }
        )
    return {"status": "observed", "matches": matches}


def _match_group_hits(group: str | list[str], content: str) -> bool:
    if isinstance(group, str):
        return group in content
    return all(s in content for s in group)


def _apply_prior_call_consistency_check(
    check: str | None, prior_value: Any, target_value: Any, *, expected_values: Any = None
) -> bool | None:
    """The real prior_value-vs-target_value comparison at the heart of the
    "prior_call_consistency" predicate (see evaluate_predicate_extended),
    factored into its own function so it can be applied identically to the
    primary source_tools/target_arg observation AND, since section 115, to
    each independently-sourced value['alternatives'] entry -- one shared
    comparison, reused, not two copies that could silently drift apart.
    Returns None (not False) for an unrecognized check name so the caller can
    distinguish "this check kind is unsupported" from "this check kind ran
    and did not pass".
    """
    if check == "equals_argument":
        return prior_value is not None and prior_value == target_value
    if check == "equals_constant":
        return prior_value in (expected_values or [])
    if check == "member_of_list":
        return isinstance(prior_value, list) and target_value in prior_value
    if check == "key_in_dict":
        return isinstance(prior_value, Mapping) and target_value in prior_value
    return None


def evaluate_predicate_extended(
    predicate: Mapping[str, Any],
    value: Any,
    *,
    semantic_judge: Callable[..., Mapping[str, Any]] | None = None,
    retail_user_directory_reference: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if predicate.get("predicate_kind") == "semantic_transcript_judgment":
        # Family M (section 35): the one real place in this module where
        # "passed" cannot be computed by pure, local Python logic -- it needs
        # actual judgment of free text against a natural-language criterion
        # (does this summary really reflect the conversation, is this reason
        # a generic placeholder, does this transcript show the agent
        # confirmed X with the user). `semantic_judge` is the real interface
        # point for that judgment (a real LLM call in production); with none
        # supplied, this follows the same "missing evidence -> fail, not a
        # silent pass" convention every other predicate in this module
        # already uses (see prior_call_consistency's "no prior call found"
        # case) rather than fabricating a verdict with no real judge behind
        # it.
        if semantic_judge is None:
            return {"passed": False, "reason": "semantic_judge_not_configured"}
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        result = semantic_judge(
            criterion=predicate.get("criterion"),
            target_value=value.get("target_value"),
            transcript_excerpt=value.get("transcript_excerpt"),
        )
        return {"passed": bool(result.get("passed")), "reason": result.get("reason")}
    if predicate.get("predicate_kind") == "matches_regex":
        if not isinstance(value, str):
            return {"passed": False, "reason": "value is not a string"}
        passed = re.fullmatch(predicate.get("pattern", ""), value) is not None
        return {"passed": passed, "actual": value}
    if predicate.get("predicate_kind") == "list_elements_match_regex":
        # Same fullmatch semantics as matches_regex, but for a list-typed
        # tool_argument (e.g. item_ids) where the frozen "field" projection
        # gives `value` as the whole raw list, not a single element --
        # matches_regex's own isinstance(value, str) guard would reject
        # every real list value outright (see docs/agentcoveragetesting_
        # reuse_log.md section 43).
        if not isinstance(value, list):
            return {"passed": False, "reason": "value is not a list"}
        pattern = predicate.get("pattern", "")
        non_matching = [
            item for item in value
            if not (isinstance(item, str) and re.fullmatch(pattern, item))
        ]
        return {"passed": not non_matching, "actual": value, "non_matching": non_matching}
    if predicate.get("predicate_kind") == "unique_by_directory_reference":
        # Same "static external reference table" pattern
        # flight_route_field_unchanged/flights_chronologically_feasible use
        # (see prepare_v5_step4_flight_route_reference_v0_1.py) -- a fact
        # only present in tau2's static benchmark database, never in an
        # observed trajectory's tool-call arguments/results.
        # find_user_id_by_name_zip's own real code has no uniqueness check
        # at all (it returns the first match); this checks the real,
        # frozen retail user directory's actual match count for the
        # observed (first_name, last_name, zip) argument triple (see
        # docs/agentcoveragetesting_reuse_log.md section 46).
        if retail_user_directory_reference is None:
            return {"passed": False, "reason": "retail_user_directory_reference_not_supplied"}
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        field_paths = predicate.get("field_paths") or []
        raw = [value.get(path) for path in field_paths]
        if not all(isinstance(item, str) for item in raw):
            return {"passed": False, "reason": "missing name/zip fields"}
        first, last, zip_code = raw
        key = f"{first.strip().lower()}|{last.strip().lower()}|{zip_code.strip()}"
        match_count = retail_user_directory_reference.get(key, 0)
        return {"passed": match_count <= 1, "match_count": match_count, "key": key}
    if predicate.get("predicate_kind") == "calculate_expression_outcome":
        if not isinstance(value, str):
            return {"passed": False, "reason": "value is not a string"}
        outcome = _calculate_expression_outcome(value)
        forbidden = predicate.get("forbidden_outcome")
        return {"passed": outcome != forbidden, "actual_outcome": outcome, "expression": value}
    if predicate.get("predicate_kind") == "valid_calendar_date":
        if not isinstance(value, str):
            return {"passed": False, "reason": "value is not a string"}
        try:
            _date.fromisoformat(value)
        except ValueError:
            return {"passed": False, "reason": "not a valid YYYY-MM-DD calendar date", "actual": value}
        return {"passed": True, "actual": value}
    if predicate.get("predicate_kind") == "date_not_before":
        if not isinstance(value, str):
            return {"passed": False, "reason": "value is not a string"}
        try:
            _date.fromisoformat(value)
        except ValueError:
            return {"passed": False, "reason": "not a valid YYYY-MM-DD calendar date", "actual": value}
        reference_date = predicate.get("reference_date", "")
        # ISO YYYY-MM-DD strings sort lexicographically the same as
        # chronologically, so plain string comparison is exact once the
        # format above is already confirmed valid. `direction` defaults to
        # the original "not_before" (value must be on/after reference_date,
        # e.g. airline's own "must not be in the past" search-date checks);
        # "before" is its mirror (value must be strictly before
        # reference_date, e.g. a real telecom candidate needing "date of
        # birth must be in the past, not today or future" -- see
        # docs/agentcoveragetesting_reuse_log.md section 21).
        direction = predicate.get("direction", "not_before")
        if direction == "before":
            passed = value < reference_date
        elif direction == "not_before":
            passed = value >= reference_date
        else:
            return {"passed": False, "reason": f"unsupported direction: {direction!r}"}
        return {"passed": passed, "actual": value, "reference_date": reference_date, "direction": direction}
    if predicate.get("predicate_kind") == "argument_le_constant":
        # docs/agentcoveragetesting_reuse_log.md section 111: a real telecom
        # candidate (telecom_025_arg#b0, "the agent must refuel no more than
        # 2GB of data in a single refuel_data operation") needs a plain,
        # SINGLE-CALL numeric upper bound on one tool argument -- not a
        # cross-call aggregate the way cumulative_sum_le is (that predicate
        # was tried first and correctly rejected in independent review: it
        # sums gb_amount across every call sharing a line_id, so two
        # separate, individually-compliant 1.5GB calls would wrongly fail a
        # branch whose own Then is explicitly about "a single ... operation",
        # not a session total -- that distinct claim is telecom_119_state#b0's,
        # which cumulative_sum_le already correctly covers). This predicate
        # uses the same generic single-value projection every other
        # `tool_argument` predicate here uses (extract_bound_observations),
        # no new extractor needed -- it only adds the comparison itself.
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
            return {"passed": False, "reason": "value is not numeric"}
        threshold = predicate.get("threshold")
        return {"passed": value <= threshold, "actual": value, "threshold": threshold}
    if predicate.get("predicate_kind") == "positive_integer":
        # `numeric_type` defaults to "integer" (airline's own use is a whole
        # dollar certificate amount, genuinely integer-only); "number" allows
        # a real float too, needed for a real telecom candidate whose own
        # requirement_text says "strictly positive" without saying "integer"
        # and whose real tool argument type is a float (gb_amount) -- see
        # docs/agentcoveragetesting_reuse_log.md section 21.
        numeric_type = predicate.get("numeric_type", "integer")
        if numeric_type == "integer":
            passed = isinstance(value, int) and not isinstance(value, bool) and value > 0
        elif numeric_type == "number":
            passed = isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        else:
            return {"passed": False, "reason": f"unsupported numeric_type: {numeric_type!r}"}
        return {"passed": passed, "actual": value}
    if predicate.get("predicate_kind") == "structural_invariant_holds":
        # See _STRUCTURAL_INVARIANT_RULES above: this predicate is
        # deliberately content-independent -- the real tau2 tool/data schema
        # makes the constraint impossible to violate, so any observed match
        # (the anchor tool call itself, already gated by the ordinary
        # all_matches_predicate missing_required_observation check above this
        # function) passes unconditionally.
        return {"passed": True, "reason": predicate.get("reason", "")}
    if predicate.get("predicate_kind") == "cabin_unchanged_from_reservation":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        if value.get("old_cabin") is None:
            return {"passed": False, "reason": "reservation cabin not observed before the update",
                    "new_cabin": value.get("new_cabin")}
        return {"passed": value.get("new_cabin") == value.get("old_cabin"),
                "new_cabin": value.get("new_cabin"), "old_cabin": value.get("old_cabin")}
    if predicate.get("predicate_kind") == "flights_unchanged_from_reservation":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        new_flights = value.get("new_flights")
        old_flights = value.get("old_flights")
        if new_flights is None or old_flights is None:
            return {"passed": False, "reason": "new or old flights list not observed"}
        return {
            "passed": new_flights == old_flights,
            "new_flights": new_flights,
            "old_flights": old_flights,
        }
    if predicate.get("predicate_kind") == "flight_route_field_unchanged":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        resolved = value.get("resolved")
        old_reservation = value.get("old_reservation")
        field = predicate.get("field")
        if not isinstance(resolved, Mapping) or resolved.get("status") != "resolved":
            return {
                "passed": False,
                "reason": f"new_flight_path_not_resolved:{resolved.get('status') if isinstance(resolved, Mapping) else 'missing'}",
            }
        if not isinstance(old_reservation, Mapping) or old_reservation.get(field) is None:
            return {"passed": False, "reason": "original_reservation_route_not_observed"}
        segments = resolved["segments"]
        if field == "origin":
            new_value = segments[0]["origin"]
        elif field == "destination":
            # docs/agentcoveragetesting_reuse_log.md section 132.2/135 (real
            # bug, task_20c7e79b, reopens section 91's conclusion for
            # airline_031_arg#b0v1): a real reservation's own "destination"
            # field means where the OUTBOUND portion of the itinerary really
            # ends -- for a real round trip (old_reservation.trip_type==
            # "round_trip") that is segments[0]["destination"] (symmetric
            # with the "origin" field above, segments[0]["origin"]), never
            # segments[-1], which for a round trip is the RETURN leg's own
            # real arrival (back at the real origin). Real, confirmed
            # against db.json: reservation 4WQ150 (DFW<->LAX, round_trip)
            # has old_reservation.destination=="LAX", genuinely equal to
            # segments[0]["destination"]=="LAX" (the real outbound leg,
            # DFW->LAX); segments[-1]["destination"] is genuinely "DFW" (the
            # real return leg's own arrival) -- comparing THAT against
            # old_reservation.destination=="LAX" is a real, structural
            # false-positive fail, unrelated to whether the agent's real
            # outbound destination genuinely changed.
            #
            # A real ONE-WAY itinerary is the mirror case and must keep the
            # ORIGINAL segments[-1] extraction: tests/test_flight_route_
            # extension_v1.py's own pre-existing Airline031B0v1
            # DestinationTests (SFO->ORD->JFK, flight_type=="one_way", a
            # real 2-hop CONNECTING one-way route) already, correctly,
            # requires the real FINAL segment's destination (JFK, the real
            # end of the trip) -- segments[0]["destination"] there would
            # wrongly extract "ORD", a real intermediate layover, not the
            # real destination. Naively always using segments[0] (an
            # earlier version of this fix, before this exact regression
            # was caught by this existing test) would have silently broken
            # this real, already-verified one-way-with-connections case.
            # Branching on the real old_reservation.trip_type (always
            # present on a real reservation) is what correctly
            # distinguishes the two real shapes; anything other than the
            # real literal "round_trip" conservatively keeps the original,
            # already-correct segments[-1] behavior.
            new_value = (
                segments[0]["destination"]
                if old_reservation.get("trip_type") == "round_trip"
                else segments[-1]["destination"]
            )
        else:
            # Deliberately different from both branches above -- trip_type
            # (round_trip vs one_way) genuinely needs to compare the FIRST
            # segment's real origin against the LAST segment's real
            # destination (do the outbound and the very end of the
            # itinerary land back where it started), not either endpoint
            # alone. Left untouched by the real destination-field fix above
            # (section 132.2/135, task_20c7e79b): a real, separate real
            # invariant, verified still correct.
            new_value = "round_trip" if segments[0]["origin"] == segments[-1]["destination"] else "one_way"
        old_value = old_reservation[field]
        return {"passed": new_value == old_value, "new_value": new_value, "old_value": old_value}
    if predicate.get("predicate_kind") == "flights_chronologically_feasible":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        segments = value.get("segments")
        if not isinstance(segments, list) or not segments:
            return {"passed": False, "reason": "flights argument not observed or malformed"}
        for segment in segments:
            if segment.get("departure") is None or segment.get("arrival") is None:
                return {
                    "passed": False,
                    "reason": "unresolved_segment_schedule",
                    "flight_number": segment.get("flight_number"),
                }
        for left, right in zip(segments, segments[1:]):
            if right["departure"] <= left["arrival"]:
                return {
                    "passed": False,
                    "reason": "connection_not_chronologically_feasible",
                    "arriving_flight": left["flight_number"],
                    "departing_flight": right["flight_number"],
                }
        return {"passed": True, "segment_count": len(segments)}
    if predicate.get("predicate_kind") == "airport_in_network":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        airport = value.get("airport")
        network = value.get("network") or []
        if not isinstance(airport, str):
            return {"passed": False, "reason": "airport code not observed"}
        return {"passed": airport in network, "airport": airport}
    if predicate.get("predicate_kind") in ("flight_path_valid", "flight_path_valid_from_reservation"):
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        resolved = value.get("resolved")
        if not isinstance(resolved, Mapping) or resolved.get("status") != "resolved":
            return {
                "passed": False,
                "reason": f"flight_path_not_resolved:{resolved.get('status') if isinstance(resolved, Mapping) else 'missing'}",
            }
        shape = _validate_flight_path_shape(
            resolved["segments"], value.get("origin"), value.get("destination"), value.get("flight_type")
        )
        return {"passed": shape["valid"], **shape}
    if predicate.get("predicate_kind") == "prior_call_consistency":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        primary_found = bool(value.get("prior_call_found"))
        primary_prior_value = value.get("prior_value")
        primary_target_value = value.get("target_value")
        primary_passed = False
        if primary_found:
            primary_check = predicate.get("check")
            dispatched = _apply_prior_call_consistency_check(
                primary_check, primary_prior_value, primary_target_value,
                expected_values=predicate.get("expected_values"),
            )
            if dispatched is None:
                return {"passed": False, "reason": f"unsupported check: {primary_check!r}"}
            primary_passed = dispatched
        if primary_passed:
            return {"passed": True, "prior_value": primary_prior_value, "target_value": primary_target_value}
        # section 115 (docs/agentcoveragetesting_reuse_log.md): the primary
        # source_tools/check evidence path didn't establish the claim --
        # before failing outright, OR-check any independently-parametrized
        # value['alternatives'] (populated only when the compiled program
        # carried a predicate_alternatives sibling for this requirement, see
        # _extract_prior_call_consistency_observations) -- e.g. retail's 6
        # order-ownership branches' alternative of get_order_details(order_id)
        # .user_id matching the identity established via find_user_id_by_
        # email/find_user_id_by_name_zip, a real, legitimate evidence path
        # the primary get_user_details-only check never recognized (real
        # online-verification finding, section 112.9/115). A predicate with
        # no alternatives configured (the overwhelming majority, including
        # every branch already adversarially confirmed in section 114) takes
        # an empty alternatives list here and falls straight through to the
        # exact same fail outcomes as before this addition.
        alternative_evaluations = []
        for alternative in value.get("alternatives") or []:
            alt_found = bool(alternative.get("prior_call_found"))
            alt_prior_value = alternative.get("prior_value")
            alt_target_value = alternative.get("target_value")
            alt_passed = False
            if alt_found:
                alt_passed = bool(
                    _apply_prior_call_consistency_check(
                        alternative.get("check"), alt_prior_value, alt_target_value
                    )
                )
            alternative_evaluations.append({
                "check": alternative.get("check"),
                "prior_call_found": alt_found,
                "prior_value": alt_prior_value,
                "target_value": alt_target_value,
                "passed": alt_passed,
            })
            if alt_passed:
                return {
                    "passed": True,
                    "prior_value": alt_prior_value,
                    "target_value": alt_target_value,
                    "matched_alternative": True,
                }
        # No evidence path passed. If the PRIMARY source_tools call was never
        # even found, AND no alternative's own source_tools call was found
        # either, this is exactly the pre-section-115 "agent never looked
        # anything up" violation shape -- return the identical reason text
        # this module has always used for it (see RetailOrderOwnership6Branch
        # AdversarialTests.test_missing_prior_lookup_also_genuinely_fails_
        # for_all_6_branches, section 114), regardless of whether alternatives
        # are configured.
        if not primary_found and not any(item["prior_call_found"] for item in alternative_evaluations):
            return {"passed": False, "reason": "no prior call found to establish the real precondition"}
        result: dict[str, Any] = {"passed": False}
        if primary_found:
            result["prior_value"] = primary_prior_value
            result["target_value"] = primary_target_value
        else:
            result["reason"] = "no prior call found via the primary evidence path"
        if alternative_evaluations:
            result["alternatives"] = alternative_evaluations
        return result
    if predicate.get("predicate_kind") == "cumulative_sum_le":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        cumulative_sum = value.get("cumulative_sum")
        valid = isinstance(cumulative_sum, (int, float)) and not isinstance(cumulative_sum, bool)
        if not valid:
            return {"passed": False, "reason": "cumulative sum not observed"}
        threshold = predicate.get("threshold")
        return {
            "passed": cumulative_sum <= threshold,
            "cumulative_sum": cumulative_sum,
            "threshold": threshold,
        }
    if predicate.get("predicate_kind") == "tool_call_any_of":
        # The extraction itself (_extract_tool_call_any_of_observations) is
        # the whole criterion -- a match only ever appears when its tool_name
        # was already in the disjunctive matcher.tool_names set, so there is
        # nothing further to check per-match; existence is handled by the
        # aggregation's own requires_observation flag (present/exists
        # semantics), not by this function returning False.
        return {"passed": True}
    if predicate.get("predicate_kind") == "call_count_le":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        cumulative_count = value.get("cumulative_count")
        valid = isinstance(cumulative_count, int) and not isinstance(cumulative_count, bool)
        if not valid:
            return {"passed": False, "reason": "cumulative count not observed"}
        threshold = predicate.get("threshold")
        return {
            "passed": cumulative_count <= threshold,
            "cumulative_count": cumulative_count,
            "threshold": threshold,
        }
    if predicate.get("predicate_kind") == "baggage_count_decreased":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        new_value = value.get("new_value")
        old_value = value.get("old_value")
        valid = (
            isinstance(new_value, int)
            and not isinstance(new_value, bool)
            and isinstance(old_value, int)
            and not isinstance(old_value, bool)
        )
        if not valid:
            return {"passed": False, "reason": "new or old baggage count not observed"}
        return {"passed": new_value >= old_value, "new_value": new_value, "old_value": old_value}
    if predicate.get("predicate_kind") == "passenger_count_changed":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        new_value = value.get("new_value")
        old_value = value.get("old_value")
        valid = (
            isinstance(new_value, int)
            and not isinstance(new_value, bool)
            and isinstance(old_value, int)
            and not isinstance(old_value, bool)
        )
        if not valid:
            return {"passed": False, "reason": "new or old passenger count not observed"}
        # "decreased" (forbid a drop, e.g. 089v1) checks new_value >= old_value;
        # "increased" (forbid a rise, e.g. 089v0) checks new_value <= old_value;
        # "no_change" (forbid any change at all, e.g. 035) checks equality.
        direction = predicate.get("direction", "decreased")
        if direction == "increased":
            passed = new_value <= old_value
        elif direction == "no_change":
            passed = new_value == old_value
        else:
            passed = new_value >= old_value
        return {
            "passed": passed,
            "direction": direction,
            "new_value": new_value,
            "old_value": old_value,
        }
    if predicate.get("predicate_kind") == "no_concurrent_message_and_tool_call":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        has_message = bool(value.get("has_message"))
        has_tool_call = bool(value.get("has_tool_call"))
        violation = has_message and has_tool_call
        return {
            "passed": not violation,
            "has_message": has_message,
            "has_tool_call": has_tool_call,
        }
    if predicate.get("predicate_kind") == "payment_reused_from_cancelled_reservation":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        used = value.get("used_payment_ids")
        cancelled = value.get("cancelled_payment_ids")
        if used is None or cancelled is None:
            return {
                "passed": False,
                "reason": "used or cancelled payment_id set not observed",
            }
        reused = sorted(set(used) & set(cancelled))
        return {"passed": not reused, "reused_payment_ids": reused}
    if predicate.get("predicate_kind") == "compensation_formula":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        amount = value.get("amount")
        passenger_count = value.get("passenger_count")
        valid = (
            isinstance(amount, int)
            and not isinstance(amount, bool)
            and passenger_count is not None
        )
        if not valid:
            return {"passed": False, "reason": "amount or passenger_count not observed"}
        expected_amount = predicate["rate_per_passenger"] * passenger_count
        return {
            "passed": amount == expected_amount,
            "amount": amount,
            "passenger_count": passenger_count,
            "expected_amount": expected_amount,
        }
    if predicate.get("predicate_kind") == "tool_call_error_matches":
        content = value.get("tool_result_content") if isinstance(value, Mapping) else None
        if content is None:
            return {"passed": False, "reason": "no_tool_result_observed"}
        content_str = str(content)
        matched = [
            group
            for group in predicate.get("match_groups") or []
            if _match_group_hits(group, content_str)
        ]
        return {"passed": not matched, "matched_groups": matched}
    if predicate.get("predicate_kind") == "field_ne":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        left = value.get(predicate.get("left_path"))
        right = value.get(predicate.get("right_path"))
        comparable = left is not None and right is not None
        return {"passed": bool(comparable and left != right), "left": left, "right": right}
    if predicate.get("predicate_kind") == "fields_non_empty_except":
        # Reuses the same generic field_tuple projection field_ne/field_lte
        # do (see runtime_observation_binding_v1.py's "kind": "field_tuple"
        # -- value is {"arguments.<param>": <raw value>, ...} for every
        # parameter the observation_contract names), so no new extraction
        # code was needed, just this evaluator branch. Real retail shape
        # (retail_047_arg, section 33): a 2-field tool_argument_constraint
        # where one field (e.g. address2) is a documented exception with no
        # constraint of its own -- every OTHER projected field must be a
        # non-empty, non-whitespace-only string.
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        exempt = set(predicate.get("exempt_paths") or [])
        failures = [
            path for path, field_value in value.items()
            if path not in exempt and (not isinstance(field_value, str) or not field_value.strip())
        ]
        return {"passed": not failures, "failures": failures}
    if predicate.get("predicate_kind") == "field_eq_constant":
        # Unlike field_lte/field_ne (which compare two paths within the same
        # call and so need a field_tuple projection, a {path: value} dict),
        # this predicate targets a single tool_argument requirement's own
        # parameter -- the runtime binding compiler gives that a plain
        # "field" projection, so `value` here IS the argument's raw scalar
        # value already, not a dict to index by path.
        expected = predicate.get("value")
        return {"passed": value == expected, "actual": value, "expected": expected}
    if predicate.get("predicate_kind") == "payment_method_type_counts_within_limit":
        # Same "plain field projection" shape as field_eq_constant above --
        # value IS payment_methods's raw list of {"payment_id", "amount"}
        # dicts already, not something to extract from a wrapper. A
        # payment_id not matching any configured type prefix (should not
        # happen with real tau2 data, but the trajectory's events are not
        # trusted input) is simply not counted against any limit, rather
        # than failing the whole check -- this predicate only judges the
        # types it has a stated limit for.
        if not isinstance(value, list):
            return {"passed": False, "reason": "payment_methods is not a list"}
        limits = predicate.get("limits") or {}
        counts: Counter[str] = Counter()
        for item in value:
            payment_id = item.get("payment_id") if isinstance(item, Mapping) else None
            if not isinstance(payment_id, str):
                continue
            for type_name in limits:
                if payment_id.startswith(f"{type_name}_"):
                    counts[type_name] += 1
                    break
        violations = {
            type_name: count
            for type_name, count in counts.items()
            if count > limits.get(type_name, 0)
        }
        return {
            "passed": not violations,
            "counts": dict(counts),
            "limits": dict(limits),
            "violations": violations,
        }
    if predicate.get("predicate_kind") == "baggage_allowance_formula":
        if not isinstance(value, Mapping):
            return {"passed": False, "reason": "projected value is not an object"}
        total_baggages = value.get(predicate.get("total_baggages_path"))
        nonfree_baggages = value.get(predicate.get("nonfree_baggages_path"))
        passengers = value.get(predicate.get("passengers_path"))
        numeric = (
            isinstance(total_baggages, int) and not isinstance(total_baggages, bool)
            and isinstance(nonfree_baggages, int) and not isinstance(nonfree_baggages, bool)
            and isinstance(passengers, list)
        )
        if not numeric:
            return {"passed": False, "reason": "projected fields are not the expected types"}
        free_bags_per_passenger = predicate["free_bags_per_passenger"]
        expected_nonfree = max(0, total_baggages - free_bags_per_passenger * len(passengers))
        return {
            "passed": nonfree_baggages == expected_nonfree,
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "passenger_count": len(passengers),
            "expected_nonfree_baggages": expected_nonfree,
        }
    return evaluate_predicate(predicate, value)


def _as_frozen_binding_set(runtime_observation_binding_set: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrap a binding set (possibly runtime_observation_binding_extension_v1's
    EXTENDED_SET_VERSION wrapper) with the frozen BINDING_SET_VERSION so it passes
    validate_runtime_observation_binding_set for the baseline compile_oracle_evaluator_
    contracts call below. Safe: each individual binding record inside already carries
    the frozen module's own schema_version/binding_fingerprint unchanged (extensions
    only ever add new keys to a record's runtime_binding, never touch the record's own
    schema_version) -- only the outer wrapper's version/fingerprint needs restamping.
    The frozen dispatch only reads the specific runtime_binding sub-fields it already
    knows about, so extra keys like program_source/baggage_allowance are inert to it.
    """
    reframed = deepcopy(dict(runtime_observation_binding_set))
    reframed.pop("binding_set_fingerprint", None)
    reframed["schema_version"] = BINDING_SET_VERSION
    reframed["binding_set_fingerprint"] = content_sha256(reframed)
    return reframed


_JUDGMENT_FAMILY_TABLE_NAMES = (
    "BAGGAGE_REDUCTION_RULES", "FIXED_VALUE_ARGUMENT_RULES", "PAYMENT_METHOD_TYPE_LIMIT_RULES",
    "IATA_CODE_FORMAT_RULES", "FLIGHT_NUMBER_FORMAT_RULES", "DATE_FORMAT_RULES",
    "DATE_NOT_BEFORE_REFERENCE_RULES", "POSITIVE_INTEGER_RULES", "EXPRESSION_CHARACTER_WHITELIST_RULES",
    "PASSENGER_REDUCTION_RULES", "FLIGHT_ROUTE_FIELD_RULES", "COMPENSATION_FORMULA_RULES",
    "PAYMENT_REUSE_RULES", "FLIGHTS_UNCHANGED_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 22): Family A
    # (tool_call_error_matches) was originally excluded from the first externalization
    # pass because its *content* (real tool ValueError text) always needs fresh
    # per-domain transcription regardless of storage mechanism -- but the predicate
    # itself is just as generic as the other 14, and real telecom content ended up
    # needing it, so it now goes through the same config mechanism too.
    "TOOL_CALL_OUTCOME_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 29): Family J
    # (prior_call_consistency) -- for claims like "line_id must be active before
    # enable_roaming" or "order_id must belong to the authenticated user", where
    # the state-changing tool itself never checks or raises (so Family A has no
    # real error text to match), but the real precondition IS independently
    # observable through an earlier, real tool call the agent could have made
    # (get_details_by_id(line_id) -> status; get_user_details(user_id) ->
    # orders/payment_methods) -- a genuinely new predicate, not a re-fit of an
    # existing one.
    "PRIOR_CALL_CONSISTENCY_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 115): an
    # OPTIONAL companion table, keyed by the exact same requirement_text as a
    # PRIOR_CALL_CONSISTENCY_RULES entry, that supplies an OR-combined
    # alternative evidence path for that SAME requirement -- see
    # _single_prior_call_consistency_observation / predicate_alternatives.
    # Deliberately a separate table (not new keys merged into the primary
    # PRIOR_CALL_CONSISTENCY_RULES entry's own params) so a requirement's
    # primary program["predicate"] dict is untouched -- byte-identical to
    # before this addition -- when no alternative is configured, and the
    # alternative is surfaced as a sibling program["predicate_alternatives"]
    # key instead, additive and opt-in per requirement_text.
    "PRIOR_CALL_CONSISTENCY_ALTERNATIVES_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 31):
    # structurally_unreachable_action's one real airline entry ("add
    # insurance") migrated in, and its compile-time gate broadened from
    # semantic_requirement-only to also cover tool_argument/
    # tool_argument_constraint -- the evaluator branch itself never looked at
    # candidate_kind, so this was an arbitrary restriction, not a real one.
    "STRUCTURALLY_UNREACHABLE_ACTION_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 33): real
    # retail candidates (retail_047_arg) where a spec has one branch per
    # field ("The agent must not submit ... with address1 empty...", another
    # branch for city, another for state, ...) sharing one rule_text that
    # names a single exempt field (address2) as an exception -- each branch
    # legitimately projects a real 2-field tool_argument_constraint (the
    # branch's own field + the shared exempt field), and all 5 branches'
    # candidates share the identical requirement_text, so one config entry
    # (keyed by that shared text) covers all of them via the generic,
    # already-existing field_tuple projection field_ne/field_lte also use --
    # no new extractor needed, just a new predicate_kind.
    "FIELDS_NON_EMPTY_EXCEPT_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 35): Family
    # M (semantic_transcript_judgment) -- real telecom/retail candidates
    # whose subject is either a free-text argument's semantic quality
    # (summary accuracy, no PII, a reason that isn't a generic placeholder)
    # or the agent's own dialogue behavior (confirmed something with the
    # user, didn't fabricate a value) -- neither is a structural fact about
    # one tool_call the way every other Family here is, so this is the one
    # predicate_kind that calls out to an injected `semantic_judge` (see
    # evaluate_oracle_contract_extended/evaluate_predicate_extended) instead
    # of computing "passed" from pure local logic.
    "SEMANTIC_TRANSCRIPT_JUDGMENT_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 39): Family
    # N (cumulative_sum_le) -- a real telecom candidate (telecom_119_state#b0,
    # "cumulative data refueled to a line in a single session must not exceed
    # 2GB") where the target tool itself (refuel_data) never checks or
    # tracks a running cap across multiple calls -- the real constraint is a
    # trajectory-level aggregate (sum a numeric argument across every real
    # call to the same tool sharing the same scope value), not a fact about
    # any single call in isolation, and not a two-call before/after diff the
    # way baggage_count_decreased/passenger_count_changed are.
    "CUMULATIVE_SUM_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 43): real
    # retail candidates whose parameter is a LIST argument (item_ids/
    # new_item_ids), where every existing format table (IATA_CODE_FORMAT_
    # RULES etc.) assumes a scalar string projection -- matches_regex's own
    # `isinstance(value, str)` guard would reject every list value outright,
    # so this is a genuinely new, list-aware sibling predicate, not a
    # re-fit of matches_regex.
    "LIST_ELEMENT_FORMAT_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 43): real
    # retail `calculate` tool candidates (division-by-zero/non-finite-result/
    # syntax-validity) -- the real tool's own algorithm (tools.py's
    # restricted eval) is mirrored exactly to classify the observed
    # `expression` argument's real outcome, the same "mirror the real
    # business logic" approach compensation_formula/cumulative_sum_le
    # already use, not a new kind of mechanism.
    "CALCULATE_EXPRESSION_OUTCOME_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 46): a real
    # retail candidate (retail_033_arg#b0v1, "the combination of first_name,
    # last_name, and zip must uniquely identify a single user in the
    # system") whose real fact source is tau2's static retail user
    # directory (find_user_id_by_name_zip's own real code has no
    # uniqueness check at all -- it just returns the first match), the same
    # "static external reference table" pattern flight_route_reference
    # already established, applied to a new domain fact.
    "USER_DIRECTORY_UNIQUENESS_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 77):
    # Family O (call_count_le) -- real retail candidates (retail_060's three
    # modify_pending_order_*#b0v0 branches) whose real Then is "must not call
    # X more than N times", a cardinality bound the frozen compiler's
    # modality-prefix matcher can only ever flatten to operator=absent (zero
    # calls allowed at all), never a real upper bound above zero. Keyed by
    # requirement_id, not requirement_text -- unlike every other table here,
    # this Then's generated tool_call requirement_text ("Observe whether the
    # agent calls X") is reused verbatim across dozens of unrelated branches
    # with genuine absent/present semantics, so a text-keyed table would
    # silently misfire on branches this fix was never meant to touch.
    "AT_MOST_N_CALL_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 111):
    # Family P (argument_le_constant) -- a real telecom candidate
    # (telecom_025_arg#b0, "no more than 2GB of data in a single
    # refuel_data operation") needing a plain single-call numeric upper
    # bound on one tool argument. cumulative_sum_le (Family N) was
    # considered first and rejected by independent review because it
    # aggregates across every call sharing a scope value -- wrong shape for
    # a Then that is explicitly about one operation, not a session total.
    # No new extractor needed (reuses the same generic single-value
    # projection every other plain `tool_argument` predicate here uses);
    # only the comparison itself is new.
    "ARGUMENT_UPPER_BOUND_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 133.4,
    # task_f10ed662): an OPTIONAL companion table, keyed by the exact same
    # requirement_text as a TOOL_CALL_OUTCOME_RULES entry, that supplies an
    # OR-combined "agent correctly declined without attempting the doomed
    # call" evidence path for that SAME requirement -- see
    # _evaluate_declined_without_attempting. tool_call_error_matches (Family
    # A) only ever recognizes ONE evidence path: a literal tool_call_error
    # whose text matches match_groups, which requires the agent to have
    # actually ATTEMPTED the target call. A real, compliant agent that
    # already has enough real information (e.g. a real get_order_details
    # result) to know the call would fail this exact way, and correctly
    # declines without ever attempting it, produces zero matching events --
    # genuinely indistinguishable, under the primary predicate alone, from
    # an agent that never even considered the constraint. Same "deliberately
    # a separate, opt-in table" discipline as PRIOR_CALL_CONSISTENCY_
    # ALTERNATIVES_RULES above: a requirement_text with no entry here is
    # byte-identical to before this addition (every other retail Family A
    # requirement -- e.g. the item/product-not-found and payment-difference
    # checks above -- and every telecom/airline Family A requirement have
    # none configured).
    "TOOL_CALL_OUTCOME_DECLINED_ALTERNATIVE_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 189,
    # task_50ff2649): an OPTIONAL companion table keyed by the same
    # requirement_text as a TOOL_CALL_OUTCOME_RULES entry, for the shape
    # where the checked tool is itself the READ-ONLY lookup that is the only
    # way to learn the entity does not exist (airline_108/110/129: "must not
    # call get_reservation_details/get_user_details/get_flight_status with an
    # id that does not exist", under a Given where the user supplies exactly
    # such an id). Scoping the "not found" error to the lookup makes every
    # compliant run fail -- the agent cannot know without looking. The
    # entry's params name that lookup_tool and the acting_tools (the tau2
    # WRITE tools that take the same entity); the compiled program then
    # treats a "not found" on the lookup as discovery (never a violation)
    # and applies the same match_groups to the acting tools instead: acting
    # on an entity already reported not to exist is the violation. Attaches
    # only when the requirement's own observation_contract.tool_name IS the
    # lookup_tool, so the other tools sharing the base requirement_text
    # (e.g. 105/118/122's cancel/update tools) are byte-identical.
    "TOOL_CALL_OUTCOME_LOOKUP_DISCOVERY_RULES",
    # Added later (docs/agentcoveragetesting_reuse_log.md section 193,
    # task_918eba9d): airline_087_state#b0 ("cabin cannot be changed if any
    # flight in the reservation has already been flown"). The frozen compiler
    # turned its update_reservation_flights requirement into an allowed-values
    # enum check on `cabin`, which cannot see a CHANGE of cabin at all. Keyed
    # by requirement_text; params.target_tool_name must equal the
    # requirement's own tool (the same text also exists for book_reservation,
    # which creates a new reservation and changes no existing cabin).
    "CABIN_UNCHANGED_RULES",
)

# Added later (docs/agentcoveragetesting_reuse_log.md section 78, extended
# section 80): Category E sub-cluster 1 -- a real requires_observation
# compiler bug, not a new predicate family, so this stays a plain scoped
# constant rather than a judgment-family config table (which mandates a
# predicate_kind every entry doesn't need here). Each of these branches' own
# real When is "The agent uses tool X" -- describing an OPTIONAL action the
# agent may take, not an unconditional user request -- but
# oracle_expectation_compiler_v1.py's frozen normative-mode derivation has no
# way to express "the agent must do Y whenever it does X" (conditionally
# vacuous when X never happens) apart from "the agent must do Y"
# (unconditionally required), so requires_observation compiles to true: a
# real agent that legitimately uses a DIFFERENT valid tool never gets the
# chance to satisfy the check, and is wrongly scored fail.
#
# retail_010_arg#b0/retail_033_arg#b0 (section 78): confirmed via real
# transcript (the agent used find_user_id_by_email, never find_user_id_by_
# name_zip, and was scored a false fail).
#
# telecom_004_arg#b0::OR01 (section 80): the exact same defect shape on the
# exact same frozen all_matches_predicate/json_type evaluator -- confirmed
# via a corpus-wide sweep (a dedicated Explore agent, real batch-result +
# transcript cross-reference) plus direct verification against the real
# tau2 telecom tool source (tau2/domains/telecom/tools.py):
# get_details_by_id is a generic dispatcher whose id.startswith("C") branch
# literally calls self.get_customer_by_id(id) -- get_customer_by_id is a
# real, functionally-identical alternative for a customer id, and the real
# batch result shows the agent legitimately called get_customer_by_id
# (never get_details_by_id) and was wrongly scored missing_required_
# observation=true.
#
# retail_008_arg#b0::OR01/retail_009_arg#b0::OR01 (section 80): direct
# siblings of retail_010 in the identical requirement family (same When:
# "The agent uses the find_user_id_by_name_zip tool", same target tool,
# different argument each branch checks -- first_name/last_name/zip). Added
# proactively rather than waiting for a future batch run to happen to expose
# the same latent false-fail: in the one real batch run available the agent
# happened to choose find_user_id_by_name_zip and both passed, but the
# defect is structurally identical, not merely similar-looking.
_REQUIRES_OBSERVATION_FALSE_OVERRIDE = frozenset({
    "retail_010_arg#b0::OR01",
    "retail_033_arg#b0::OR01",
    "telecom_004_arg#b0::OR01",
    "retail_008_arg#b0::OR01",
    "retail_009_arg#b0::OR01",
})

# docs/agentcoveragetesting_reuse_log.md section 104/105: the same
# "conditional negation flattened / fabricated rule" meta-pattern as
# sections 75/86/89/98 (retail_068, airline_031, airline_109, airline_126),
# this time found via a structural corpus scan (branch_context.origin==
# "domain_knowledge" + requirement_type=="tool_call") rather than by
# accident during an online rerun, and confirmed by TWO independent
# verification passes (each given the real policy.md/main_policy.md + real
# tools.py/data_model.py, forming its own judgment) before being REJECTed
# here -- same double-check discipline as section 98.4's airline_126
# decision. Keyed by requirement_id, NOT requirement_text: for a "tool_call"
# shaped requirement, requirement_text is a generic "Observe whether the
# agent calls X." shared verbatim by many OTHER, real, legitimate
# requirements across the corpus (confirmed: this exact string for
# cancel_reservation alone is shared by 22 different requirements, several
# of them real positive "the agent SHOULD call this" checks) -- a
# text-keyed table like STRUCTURALLY_UNREACHABLE_ACTION_RULES would either
# never match (miss) or, worse, match and rewrite an unrelated legitimate
# check. Every reason below is a real, verified fact about the actual tau2
# tool source/policy text, not a restatement of the branch's own (fictional)
# rule_text.
_FABRICATED_REQUIREMENT_REJECT = {
    # Airline -- fabricated "ownership check" (real cancel_reservation/
    # update_reservation_* tools take no user_id/requester argument and
    # perform no ownership comparison anywhere in tools.py; policy.md never
    # states an ownership-verification rule for any of these actions).
    "airline_107_state#b0::OR01": (
        "real cancel_reservation(reservation_id) takes no user_id/requester "
        "argument and performs no ownership check anywhere in tools.py; "
        "policy.md never states an ownership-verification rule for cancellation."
    ),
    "airline_123_state#b0::OR01": (
        "real update_reservation_flights(reservation_id, cabin, flights, "
        "payment_id) takes no user_id/requester argument and performs no "
        "ownership check; policy.md is silent on ownership verification for "
        "flight changes."
    ),
    # Section 108: same fabricated ownership-check claim, confirmed by two
    # further independent verification passes (docs/agentcoveragetesting_
    # reuse_log.md section 108) for the 2 branches section 98's Given-
    # materialization fixture fix touched but never re-examined for real
    # basis (that fix was a different layer -- can the fixture engine FIND a
    # real not-owner (user, reservation) pair -- not whether the claim
    # itself is real). Each branch has exactly one real requirement, so
    # rejecting it doesn't strand any other real check on that branch.
    "airline_119_state#b0::OR01": (
        "real update_reservation_baggages(reservation_id, total_baggages, "
        "nonfree_baggages, payment_id) takes no user_id/requester argument; "
        "self._get_user(reservation.user_id) only looks up the RESERVATION's "
        "own owner (to validate payment_id against their payment methods), "
        "never compares against any other identity. policy.md's 'Change "
        "baggage and insurance' subsection states no ownership/verification "
        "rule in any phrasing."
    ),
    "airline_128_state#b0::OR01": (
        "real update_reservation_passengers(reservation_id, passengers) is "
        "even more minimal -- no user_id/requester argument, doesn't even "
        "call _get_user, only validates passenger count matches. policy.md's "
        "'Change passengers' subsection states only the immutable-passenger-"
        "count rule, nothing about ownership."
    ),
    # Airline -- fabricated "cancelled/voided reservation blocks further
    # action" (real cancel_reservation/update_reservation_* tools never read
    # reservation.status at all; Reservation.status's only real values are
    # None or "cancelled" -- "voided"/"expired"/"refunded" cannot occur).
    "airline_106_state#b0::OR01": (
        "real cancel_reservation (tools.py) has no status guard at all -- "
        "re-cancelling an already-cancelled reservation just re-appends "
        "refund Payment rows and re-sets status=\"cancelled\"; policy.md's "
        "cancel section never restricts re-cancellation."
    ),
    "airline_106_state#b2::OR01": (
        "real Reservation.status is Optional[Literal[\"cancelled\"]] -- "
        "\"voided\" is not a constructible value in the real data model at "
        "all, and even the substituted real \"cancelled\" state has no "
        "enforcement basis (same as airline_106_state#b0)."
    ),
    "airline_120_state#b0::OR01": (
        "real update_reservation_baggages (tools.py) never reads "
        "reservation.status; policy.md's baggage/insurance section never "
        "mentions cancelled reservations."
    ),
    "airline_124_state#b0::OR01": (
        "real update_reservation_flights (tools.py) never reads "
        "reservation.status; policy.md's flight-change section never "
        "mentions cancelled reservations."
    ),
    "airline_127_state#b0::OR01": (
        "real update_reservation_passengers (tools.py) only checks "
        "passenger-count match, never reservation.status; policy.md's "
        "passenger-change section never mentions cancelled reservations."
    ),
    "airline_130_state#b0v0::OR01": (
        "real update_reservation_flights (tools.py) never reads "
        "reservation.status -- same fabricated \"cancelled reservation "
        "blocks modification\" claim as airline_124_state#b0."
    ),
    "airline_130_state#b0v0::OR02": (
        "real cancel_reservation has no status guard -- same fabricated "
        "claim as airline_106_state#b0."
    ),
    "airline_130_state#b0v1::OR01": (
        "real update_reservation_baggages never reads reservation.status -- "
        "same fabricated claim as airline_120_state#b0."
    ),
    "airline_130_state#b0v1::OR02": (
        "real cancel_reservation has no status guard -- same fabricated "
        "claim as airline_106_state#b0."
    ),
    "airline_130_state#b0v2::OR01": (
        "real update_reservation_passengers never reads reservation.status "
        "-- same fabricated claim as airline_127_state#b0."
    ),
    "airline_130_state#b0v2::OR02": (
        "real cancel_reservation has no status guard -- same fabricated "
        "claim as airline_106_state#b0."
    ),
    "airline_130_state#b0v3::OR01": (
        "real cancel_reservation has no status guard -- same fabricated "
        "claim as airline_106_state#b0."
    ),
    # Airline -- fabricated "payment method reuse after cancellation"
    # restriction (real PaymentMethod objects live on the user profile only
    # -- CreditCard/GiftCard/Certificate carry no reservation_id field of any
    # kind, so there is no real notion of "a payment method belonging to the
    # cancelled reservation" for a tool to block).
    "airline_133_state#b0v0::OR02": (
        "real PaymentMethod types (data_model.py) carry no reservation_id "
        "field -- payment methods are user-profile-scoped, not "
        "reservation-scoped; book_reservation/_payment_for_update only check "
        "payment_id against user.payment_methods, never against any "
        "reservation's cancellation history."
    ),
    "airline_133_state#b0v1::OR02": (
        "same fabricated claim as airline_133_state#b0v0::OR02, applied to "
        "update_reservation_flights."
    ),
    "airline_133_state#b0v1::OR03": (
        "same fabricated claim as airline_133_state#b0v0::OR02, applied to "
        "update_reservation_baggages."
    ),
    # Telecom -- fabricated "already enabled/disabled blocks a redundant
    # call" (real enable_roaming/disable_roaming explicitly tolerate the
    # redundant case, returning a friendly string rather than raising).
    "telecom_107_state#b0::OR01": (
        "real enable_roaming (tools.py): \"if target_line.roaming_enabled: "
        "return 'Roaming was already enabled'\" -- the tool is designed to "
        "gracefully no-op a redundant call, not reject it; main_policy.md's "
        "\"check before enabling\" text is workflow guidance, not a hard "
        "prohibition (same claim shape already confirmed fabricated for "
        "telecom_106_state#b1, section 75)."
    ),
    "telecom_111_state#b0::OR01": (
        "real disable_roaming (tools.py): \"if not target_line.roaming_"
        "enabled: return 'Roaming was already disabled'\" -- same graceful "
        "no-op design; main_policy.md contains zero occurrences of "
        "\"disable\" at all."
    ),
    # Telecom -- fabricated "suspended line blocks this action" (real
    # enable_roaming/disable_roaming/refuel_data have no LineStatus gate;
    # resume_line's real gate is the OPPOSITE of the claim).
    "telecom_117_state#b0::OR01": (
        "real enable_roaming (tools.py) calls only _get_target_line "
        "(existence/ownership only) -- no LineStatus check anywhere; same "
        "claim already confirmed fabricated for telecom_106_state#b1."
    ),
    "telecom_117_state#b1::OR01": (
        "real disable_roaming calls only _get_target_line -- no LineStatus "
        "check anywhere; same claim already confirmed fabricated for "
        "telecom_106_state#b2."
    ),
    "telecom_117_state#b2::OR01": (
        "real refuel_data's Active-status check is commented-out dead code "
        "in tools.py (\"# if target_line.status != LineStatus.ACTIVE: # "
        "raise ValueError(...)\") -- current runtime enforces nothing."
    ),
    "telecom_117_state#b3::OR01": (
        "real resume_line (tools.py) REQUIRES status in [SUSPENDED, "
        "PENDING_ACTIVATION] and raises ValueError(\"Line must be suspended "
        "to resume\") otherwise -- acting on a suspended line is resume_"
        "line's entire documented purpose; this claim is the exact inverse "
        "of the tool's real, enforced precondition."
    ),
    # Telecom -- fabricated "terminated/deleted line" states (real
    # LineStatus enum is only ACTIVE/SUSPENDED/PENDING_ACTIVATION/CLOSED --
    # "terminated"/"ported out"/"deleted" do not exist anywhere in the real
    # schema, policy, or database; each of the 4 named tools below has no
    # status gate that could enforce this claim regardless).
    "telecom_118_state#b0v0::OR01": (
        "\"terminated\" is not a real LineStatus value (enum: ACTIVE/"
        "SUSPENDED/PENDING_ACTIVATION/CLOSED only, confirmed zero hits for "
        "\"terminated\" anywhere in telecom source/policy/db); real "
        "get_data_usage also has no status check regardless."
    ),
    "telecom_118_state#b0v1::OR01": (
        "same impossible \"terminated\" status as telecom_118_state#b0v0; "
        "real enable_roaming has no status check regardless."
    ),
    "telecom_118_state#b0v2::OR01": (
        "same impossible \"terminated\" status as telecom_118_state#b0v0; "
        "real disable_roaming has no status check regardless."
    ),
    "telecom_118_state#b0v3::OR01": (
        "same impossible \"terminated\" status as telecom_118_state#b0v0; "
        "real refuel_data's status gate is dead code (see "
        "telecom_117_state#b2)."
    ),
    "telecom_118_state#b2v0::OR01": (
        "\"deleted\" is not a real LineStatus value either (same enum fact "
        "as telecom_118_state#b0v0); real get_data_usage has no status "
        "check regardless."
    ),
    "telecom_118_state#b2v1::OR01": (
        "same impossible \"deleted\" status as telecom_118_state#b2v0; real "
        "enable_roaming has no status check regardless."
    ),
    "telecom_118_state#b2v2::OR01": (
        "same impossible \"deleted\" status as telecom_118_state#b2v0; real "
        "disable_roaming has no status check regardless."
    ),
    "telecom_118_state#b2v3::OR01": (
        "same impossible \"deleted\" status as telecom_118_state#b2v0; real "
        "refuel_data's status gate is dead code (see telecom_117_state#b2)."
    ),
    # Telecom -- mirror of the already-retracted telecom_106_state#b1/#b2
    # claim (section 75), independently re-confirmed here for a different
    # pair of branches making the identical "must be active before roaming
    # can be disabled" claim.
    "telecom_110_state#b1::OR01": (
        "real disable_roaming has no LineStatus check at all -- would "
        "succeed identically on a suspended line; same claim already "
        "confirmed fabricated for telecom_106_state#b2."
    ),
    "telecom_110_state#b2::OR01": (
        "real disable_roaming has no LineStatus check at all -- would "
        "succeed identically regardless of status; \"terminated\" (used in "
        "this branch's own rule_text) is also not a real LineStatus value."
    ),
}

_DEFAULT_JUDGMENT_FAMILY_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs/oracle_judgment_families/airline_v0_1.json"
)


def load_judgment_family_config(path: str | Path) -> dict[str, dict[Any, dict[str, Any]]]:
    """Load one domain's judgment-family rule tables from external JSON.

    This is a pure content move, not a new dispatch mechanism: the ~14 tables
    below (covering format/range validation, fixed-value/composite-limit,
    cross-call value-diff, and derived-formula judgments -- the families whose
    predicate evaluators in this module are already fully generic,
    parameterized functions) used to be Python dict literals hardcoded in
    this file. Moving their *content* to an external, per-domain JSON file
    lets a new domain (e.g. telecom, retail) supply its own real
    requirement_text/parameter keys and predicate parameters without editing
    this module's code -- the dispatch chain in
    compile_oracle_evaluator_contracts_extended (which table is tried, in
    what order, under what requirement_type/parameter gate) is unchanged; see
    docs/agentcoveragetesting_reuse_log.md section 19 for the real-data
    validation this design is based on. The ~1/3 of tables NOT covered here
    (tool-error-text matching, static-reference graph/schedule validity,
    schema-derived structural invariants, free-text dialogue-act keywords)
    stay as Python literals for now -- they need genuine per-domain content
    authoring no matter the storage mechanism, so externalizing them buys
    nothing on its own.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "agentspectesting.oracle-judgment-family-config/v0.1":
        raise ValueError(f"unsupported judgment family config schema: {data.get('schema_version')!r}")
    tables: dict[str, dict[Any, dict[str, Any]]] = {name: {} for name in _JUDGMENT_FAMILY_TABLE_NAMES}
    for rule in data.get("rules") or []:
        table = rule.get("table")
        if table not in tables:
            raise ValueError(f"unknown judgment family table: {table!r}")
        key = rule.get("key")
        key = tuple(key) if isinstance(key, list) else key
        if not isinstance(key, (str, tuple)) or key in tables[table]:
            raise ValueError(f"missing or duplicate key in table {table!r}: {key!r}")
        params = dict(rule.get("params") or {})
        params["predicate_kind"] = rule["predicate_kind"]
        tables[table][key] = params
    return tables


_AIRLINE_JUDGMENT_FAMILY_CONFIG = load_judgment_family_config(_DEFAULT_JUDGMENT_FAMILY_CONFIG_PATH)

# docs/agentcoveragetesting_reuse_log.md section 104: oracle_requirement_
# pipeline_v7.py's real disambiguation pass (_dedupe_same_requirement_text,
# roughly) appends a real " (tool_name)" suffix to a candidate's
# requirement_text ONLY when the full-catalog scan binds that exact literal
# text to more than one DISTINCT tool in that particular compile run -- this
# is genuinely recomputed from the current candidate pool every regen, not a
# stable one-time fact, so a judgment-family config table keyed on the
# unsuffixed (or suffixed) text alone inevitably drifts out of sync with
# whichever form a later regen happens to produce (confirmed real recurrence:
# section 32 already stripped these suffixes from airline_v0_1.json once,
# and a later, independent regeneration re-added them for the same 30
# requirement_ids). The two forms can also genuinely coexist in the same
# table at the same moment for different requirement_ids sharing the same
# base text (confirmed: airline_053_arg#b0v0/v1 currently carry the suffixed
# form while #b0v2/v3 carry the exact unsuffixed form) -- so rewriting the
# config to "the current form" is not a stable fix either. This lookup tries
# the requirement's real requirement_text exactly first (preserving every
# existing exact-match table entry/behavior unchanged), and only falls back
# to the suffix-stripped form on a miss -- robust to either direction the
# suffix drifts, without ever touching the config content itself.
_TOOL_SUFFIX_RE = re.compile(r" \([a-z0-9_]+\)$")


def _lookup_requirement_rule(
    table: Mapping[Any, Any], requirement_text: str | None, *rest: Any
) -> Any:
    if requirement_text is None:
        return None
    key = (requirement_text, *rest) if rest else requirement_text
    hit = table.get(key)
    if hit is not None:
        return hit
    stripped = _TOOL_SUFFIX_RE.sub("", requirement_text)
    if stripped == requirement_text:
        return None
    return table.get((stripped, *rest) if rest else stripped)


def _baggage_allowance_predicate(baggage_allowance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "predicate_kind": "baggage_allowance_formula",
        "total_baggages_path": "arguments.total_baggages",
        "nonfree_baggages_path": "arguments.nonfree_baggages",
        "passengers_path": "arguments.passengers",
        "free_bags_per_passenger": baggage_allowance["free_bags_per_passenger"],
    }


def _mark_optional_argument_value_predicate(
    record: Mapping[str, Any], requirement: Mapping[str, Any]
) -> dict[str, Any]:
    """Real tau2 tool schemas mark many arguments optional (observation_contract.
    endpoint.required == False, e.g. get_bills_for_customer's real `limit: int =
    12`) -- a genuinely OMITTED optional argument is not a violation of a real
    value-shape check on it (json_type/positive_integer/in_set/etc, any
    all_matches_predicate program), but extract_bound_observations' projection
    has no way to tell "argument key absent from the real tool call" from
    "argument key present with the wrong value" -- both surface as value=None,
    and evaluate_predicate/evaluate_predicate_extended then apply the predicate
    to None and fail. Confirmed real, corpus-wide (docs/agentcoveragetesting_
    reuse_log.md section 68.1/103): telecom_031_arg#b0's real "positive_integer"
    check on get_bills_for_customer's real optional `limit` genuinely failed a
    live agent that correctly omitted it; the sibling `telecom_011_arg#b0`
    (same real optional-argument shape, compiled to "json_type") only "passed"
    because that particular run happened to supply a value -- same latent bug,
    masked by luck, not a correctly-designed check.

    Scoped to requirement_type=="tool_argument" (a single scalar tool
    argument's own value-shape claim -- not a cross-argument/semantic/state
    claim, where a None value could mean something else entirely) with the
    endpoint EXPLICITLY marked required==False (not just absent, to avoid
    firing on a requirement_type shape that never carries this field at all).
    Marks program.target_argument_required=False; evaluate_oracle_contract_
    extended treats a None-valued match as vacuously satisfied only when this
    flag is present -- never relaxes a check on a value that was genuinely
    supplied, wrong-typed or not."""
    program = record.get("program")
    if not isinstance(program, Mapping) or program.get("evaluator_kind") != "all_matches_predicate":
        return dict(record)
    if requirement.get("requirement_type") != "tool_argument":
        return dict(record)
    endpoint = (requirement.get("observation_contract") or {}).get("endpoint") or {}
    if endpoint.get("required") is not False:
        return dict(record)
    if program.get("target_argument_required") is False:
        return dict(record)
    record = deepcopy(dict(record))
    record.pop("evaluator_contract_fingerprint", None)
    new_program = dict(record["program"])
    new_program["target_argument_required"] = False
    new_program["program_source"] = PROGRAM_SOURCE
    record["program"] = new_program
    record["diagnostics"] = list(record.get("diagnostics") or []) + [
        {"code": "marked_optional_argument_value_predicate"},
        {"code": "compiled_by_extension_v1"},
    ]
    record["evaluator_contract_fingerprint"] = content_sha256(record)
    return record


def _reject_fabricated_requirement(record: Mapping[str, Any]) -> dict[str, Any]:
    """Unconditionally overrides a record whose requirement_id is in
    _FABRICATED_REQUIREMENT_REJECT (see that constant's own docstring) to an
    always-pass structurally_unreachable_action program -- same real,
    verified-basis-free-rule REJECT semantics as sections 75/86/89/98's
    precedent, this time keyed by requirement_id since these are all
    "tool_call"-shaped requirements whose requirement_text is a generic
    string shared with real, unrelated requirements elsewhere in the corpus
    (see the constant's docstring for the confirmed 22-way collision on
    "Observe whether the agent calls cancel_reservation." alone)."""
    reason = _FABRICATED_REQUIREMENT_REJECT.get(record.get("requirement_id"))
    if reason is None:
        return dict(record)
    record = deepcopy(dict(record))
    record.pop("evaluator_contract_fingerprint", None)
    record["evaluator_status"] = "executable"
    record["program"] = {
        "evaluator_kind": "all_matches_predicate",
        "requires_observation": False,
        "predicate": {"predicate_kind": "structurally_unreachable_action", "reason": reason},
        "program_source": PROGRAM_SOURCE,
    }
    record["diagnostics"] = list(record.get("diagnostics") or []) + [
        {"code": "rejected_fabricated_requirement"},
        {"code": "compiled_by_extension_v1"},
    ]
    record["evaluator_contract_fingerprint"] = content_sha256(record)
    return record


def compile_oracle_evaluator_contracts_extended(
    accepted_requirement_set: Mapping[str, Any],
    runtime_observation_binding_set: Mapping[str, Any],
    *,
    judgment_family_config: Mapping[str, Mapping[Any, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    tables = judgment_family_config if judgment_family_config is not None else _AIRLINE_JUDGMENT_FAMILY_CONFIG
    accepted = validate_accepted_oracle_requirement_set(accepted_requirement_set)
    baseline = compile_oracle_evaluator_contracts(
        accepted_requirement_set, _as_frozen_binding_set(runtime_observation_binding_set)
    )
    requirements = {
        item["requirement_id"]: item
        for branch in accepted["branches"]
        for item in branch["requirements"]
    }
    bindings = {
        item["requirement_id"]: item
        for branch in runtime_observation_binding_set["branches"]
        for item in branch["bindings"]
    }

    counts: Counter[str] = Counter()
    branches = []
    deferred_branches = []
    for source_branch in baseline["branches"]:
        contracts = []
        for record in source_branch["evaluator_contracts"]:
            requirement = requirements[record["requirement_id"]]
            binding = bindings.get(record["requirement_id"]) or {}
            baggage_allowance = (binding.get("runtime_binding") or {}).get("baggage_allowance")
            is_turn_shape_constraint = (
                (binding.get("runtime_binding") or {}).get("extractor_kind")
                == "turn_shape_constraint"
            )
            # docs/agentcoveragetesting_reuse_log.md section 79/104: same
            # "unconditional override, not gated on evaluator_status ==
            # deferred" reason as baggage_allowance/is_turn_shape_constraint
            # above -- the binding extension already marks this "bound" via
            # a real OR-route (e.g. retail_053's find_user_id_by_email OR
            # find_user_id_by_name_zip), so the frozen baseline compiler
            # sees binding_status=="bound" and builds its own generic
            # match_cardinality/"present" program (which real-passes for ANY
            # tool call at all, not just the intended disjunction) before
            # ever reaching "deferred". This compile branch had gone
            # missing (confirmed: the live retail_053 record carrying this
            # shape was a hand-splice, not something any current code path
            # produced -- a fresh recompile silently downgraded it to the
            # vacuous match_cardinality shape above).
            is_tool_call_any_of = (
                (binding.get("runtime_binding") or {}).get("extractor_kind")
                == "tool_call_any_of"
            )
            at_most_n_rule = tables["AT_MOST_N_CALL_RULES"].get(record["requirement_id"])
            if (
                record["evaluator_status"] == "deferred"
                and requirement.get("requirement_type")
                in ("tool_argument_constraint", "tool_argument", "semantic_requirement")
            ):
                observation_contract = requirement.get("observation_contract") or {}
                predicate, extra_diagnostics = None, []
                predicate_alternatives = None
                declined_without_attempting_alternative = None
                lookup_discovery = None
                if requirement.get("requirement_type") == "tool_argument_constraint":
                    predicate, extra_diagnostics = _must_differ_predicate(observation_contract)
                if predicate is None and requirement.get("requirement_type") == "tool_argument_constraint":
                    fields_non_empty_rule = _lookup_requirement_rule(tables["FIELDS_NON_EMPTY_EXCEPT_RULES"], requirement.get("requirement_text"))
                    if fields_non_empty_rule is not None:
                        exempt_fields = fields_non_empty_rule.get("exempt_fields") or []
                        predicate = {
                            "predicate_kind": "fields_non_empty_except",
                            "exempt_paths": [f"arguments.{field}" for field in exempt_fields],
                        }
                        extra_diagnostics = [{"code": "compiled_fields_non_empty_except_grammar"}]
                if predicate is None and requirement.get("requirement_type") in (
                    "tool_argument", "tool_argument_constraint",
                ):
                    semantic_rule = _lookup_requirement_rule(tables["SEMANTIC_TRANSCRIPT_JUDGMENT_RULES"], requirement.get("requirement_text"))
                    if semantic_rule is not None:
                        predicate = {
                            "predicate_kind": "semantic_transcript_judgment",
                            "criterion": semantic_rule.get("criterion"),
                            # tool_argument_constraint candidates have no
                            # single "parameter" -- their real subject is the
                            # whole conversation/argument-set the criterion
                            # describes, so target_arg is left unset (None)
                            # and the judge relies on transcript_excerpt.
                            "target_arg": observation_contract.get("parameter"),
                        }
                        extra_diagnostics = [
                            {"code": "compiled_semantic_transcript_judgment_grammar"}
                        ]
                if predicate is None:
                    tool_call_outcome_rule = _lookup_requirement_rule(tables["TOOL_CALL_OUTCOME_RULES"], requirement.get("requirement_text"))
                    match_groups = (
                        tool_call_outcome_rule.get("match_groups")
                        if tool_call_outcome_rule is not None else None
                    )
                    if match_groups is not None:
                        predicate = {
                            "predicate_kind": "tool_call_error_matches",
                            "match_groups": match_groups,
                        }
                        extra_diagnostics = [{"code": "compiled_tool_call_outcome_grammar"}]
                        # section 133.4: an OPTIONAL, additive OR-evidence path
                        # for this SAME requirement_text -- see the
                        # TOOL_CALL_OUTCOME_DECLINED_ALTERNATIVE_RULES table
                        # comment above and _evaluate_declined_without_
                        # attempting. Looked up by the same key so it only
                        # ever attaches to requirements that already resolved
                        # a primary tool_call_error_matches predicate above; a
                        # miss leaves declined_without_attempting_alternative
                        # None, so program["predicate"] and program itself
                        # stay byte-identical to before this addition for
                        # every requirement without a configured entry.
                        declined_alt_rule = _lookup_requirement_rule(
                            tables.get("TOOL_CALL_OUTCOME_DECLINED_ALTERNATIVE_RULES") or {},
                            requirement.get("requirement_text"),
                        )
                        if declined_alt_rule is not None and declined_alt_rule.get("criterion"):
                            declined_without_attempting_alternative = {
                                "criterion": declined_alt_rule["criterion"],
                            }
                            extra_diagnostics = extra_diagnostics + [
                                {"code": "compiled_tool_call_outcome_declined_alternative_grammar"}
                            ]
                        # section 189: see TOOL_CALL_OUTCOME_LOOKUP_DISCOVERY_
                        # RULES in _JUDGMENT_FAMILY_TABLE_NAMES.
                        lookup_rule = _lookup_requirement_rule(
                            tables.get("TOOL_CALL_OUTCOME_LOOKUP_DISCOVERY_RULES") or {},
                            requirement.get("requirement_text"),
                        )
                        if (
                            lookup_rule is not None
                            and lookup_rule.get("lookup_tool") == observation_contract.get("tool_name")
                            and lookup_rule.get("acting_tools")
                        ):
                            lookup_discovery = {
                                "lookup_tool": lookup_rule["lookup_tool"],
                                "acting_tools": sorted(lookup_rule["acting_tools"]),
                            }
                            extra_diagnostics = extra_diagnostics + [
                                {"code": "compiled_tool_call_outcome_lookup_discovery_grammar"}
                            ]
                            # section 192: the optional no-fabrication judgment
                            # (see _evaluate_lookup_discovery_grounding).
                            if lookup_rule.get("not_found_grounding_criterion"):
                                lookup_discovery["not_found_grounding"] = {
                                    "criterion": lookup_rule["not_found_grounding_criterion"],
                                }
                                extra_diagnostics = extra_diagnostics + [
                                    {"code": "compiled_lookup_discovery_not_found_grounding_grammar"}
                                ]
                if predicate is None:
                    prior_call_rule = _lookup_requirement_rule(tables["PRIOR_CALL_CONSISTENCY_RULES"], requirement.get("requirement_text"))
                    if prior_call_rule is not None:
                        predicate = {
                            "predicate_kind": "prior_call_consistency",
                            **prior_call_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_prior_call_consistency_grammar"}]
                        # section 115: an OPTIONAL, additive OR-evidence path
                        # for this SAME requirement_text -- see the
                        # PRIOR_CALL_CONSISTENCY_ALTERNATIVES_RULES table
                        # comment. Looked up by the same key so it only ever
                        # attaches to requirements that already resolved a
                        # primary prior_call_consistency predicate above; a
                        # miss (the overwhelming majority of prior_call_
                        # consistency requirements, e.g. retail_087/096 and
                        # every telecom entry -- see reuse log section 118 for
                        # why those 2 stay unfixed while retail_083 now has an
                        # alternative configured) leaves predicate_alternatives
                        # None, so program["predicate"] and program itself stay
                        # byte-identical to before this addition for them.
                        prior_call_alt_rule = _lookup_requirement_rule(
                            tables.get("PRIOR_CALL_CONSISTENCY_ALTERNATIVES_RULES") or {},
                            requirement.get("requirement_text"),
                        )
                        if prior_call_alt_rule is not None and prior_call_alt_rule.get("alternatives"):
                            predicate_alternatives = prior_call_alt_rule["alternatives"]
                            extra_diagnostics = extra_diagnostics + [
                                {"code": "compiled_prior_call_consistency_alternatives_grammar"}
                            ]
                if predicate is None:
                    cumulative_sum_rule = _lookup_requirement_rule(tables["CUMULATIVE_SUM_RULES"], requirement.get("requirement_text"))
                    if cumulative_sum_rule is not None:
                        predicate = {
                            "predicate_kind": "cumulative_sum_le",
                            **cumulative_sum_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_cumulative_sum_grammar"}]
                if predicate is None:
                    compensation_rule = _lookup_requirement_rule(tables["COMPENSATION_FORMULA_RULES"], requirement.get("requirement_text"))
                    if compensation_rule is not None:
                        predicate = {
                            "predicate_kind": "compensation_formula",
                            **compensation_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_compensation_formula_grammar"}]
                if predicate is None:
                    payment_reuse_rule = _lookup_requirement_rule(tables["PAYMENT_REUSE_RULES"], requirement.get("requirement_text"))
                    if payment_reuse_rule is not None:
                        predicate = {
                            "predicate_kind": "payment_reused_from_cancelled_reservation",
                            **payment_reuse_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_payment_reuse_grammar"}]
                if predicate is None:
                    baggage_reduction_rule = _lookup_requirement_rule(
                        tables["BAGGAGE_REDUCTION_RULES"],
                        requirement.get("requirement_text"),
                        observation_contract.get("parameter"),
                    )
                    if baggage_reduction_rule is not None:
                        predicate = {
                            "predicate_kind": "baggage_count_decreased",
                            **baggage_reduction_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_baggage_reduction_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    fixed_value_rule = _lookup_requirement_rule(
                        tables["FIXED_VALUE_ARGUMENT_RULES"],
                        requirement.get("requirement_text"),
                        observation_contract.get("parameter"),
                    )
                    if fixed_value_rule is not None:
                        predicate = {
                            "predicate_kind": "field_eq_constant",
                            **fixed_value_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_fixed_value_argument_grammar"}]
                if (
                    predicate is None
                    and requirement.get("requirement_type") == "tool_argument"
                    and observation_contract.get("parameter") == "payment_methods"
                ):
                    payment_method_type_limit_rule = _lookup_requirement_rule(tables["PAYMENT_METHOD_TYPE_LIMIT_RULES"], requirement.get("requirement_text"))
                    if payment_method_type_limit_rule is not None:
                        predicate = {
                            "predicate_kind": "payment_method_type_counts_within_limit",
                            **payment_method_type_limit_rule,
                        }
                        extra_diagnostics = [
                            {"code": "compiled_payment_method_type_limit_grammar"}
                        ]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    iata_code_format_rule = _lookup_requirement_rule(tables["IATA_CODE_FORMAT_RULES"], requirement.get("requirement_text"))
                    if iata_code_format_rule is not None:
                        predicate = {
                            "predicate_kind": "matches_regex",
                            **iata_code_format_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_iata_code_format_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    flight_number_format_rule = _lookup_requirement_rule(tables["FLIGHT_NUMBER_FORMAT_RULES"], requirement.get("requirement_text"))
                    if flight_number_format_rule is not None:
                        predicate = {
                            "predicate_kind": "matches_regex",
                            **flight_number_format_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_flight_number_format_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    date_format_rule = _lookup_requirement_rule(tables["DATE_FORMAT_RULES"], requirement.get("requirement_text"))
                    if date_format_rule is not None:
                        predicate = {
                            "predicate_kind": "valid_calendar_date",
                            **date_format_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_date_format_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    date_not_before_rule = _lookup_requirement_rule(tables["DATE_NOT_BEFORE_REFERENCE_RULES"], requirement.get("requirement_text"))
                    if date_not_before_rule is not None:
                        predicate = {
                            "predicate_kind": "date_not_before",
                            **date_not_before_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_date_not_before_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    positive_integer_rule = _lookup_requirement_rule(tables["POSITIVE_INTEGER_RULES"], requirement.get("requirement_text"))
                    if positive_integer_rule is not None:
                        predicate = {
                            "predicate_kind": "positive_integer",
                            **positive_integer_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_positive_integer_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    argument_upper_bound_rule = _lookup_requirement_rule(tables["ARGUMENT_UPPER_BOUND_RULES"], requirement.get("requirement_text"))
                    if argument_upper_bound_rule is not None:
                        predicate = {
                            "predicate_kind": "argument_le_constant",
                            **argument_upper_bound_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_argument_upper_bound_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    expression_whitelist_rule = _lookup_requirement_rule(tables["EXPRESSION_CHARACTER_WHITELIST_RULES"], requirement.get("requirement_text"))
                    if expression_whitelist_rule is not None:
                        predicate = {
                            "predicate_kind": "matches_regex",
                            **expression_whitelist_rule,
                        }
                        extra_diagnostics = [
                            {"code": "compiled_expression_character_whitelist_grammar"}
                        ]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    list_element_format_rule = _lookup_requirement_rule(
                        tables["LIST_ELEMENT_FORMAT_RULES"],
                        requirement.get("requirement_text"),
                        observation_contract.get("parameter"),
                    )
                    if list_element_format_rule is not None:
                        predicate = {
                            "predicate_kind": "list_elements_match_regex",
                            **list_element_format_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_list_element_format_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    calculate_expression_rule = _lookup_requirement_rule(tables["CALCULATE_EXPRESSION_OUTCOME_RULES"], requirement.get("requirement_text"))
                    if calculate_expression_rule is not None:
                        predicate = {
                            "predicate_kind": "calculate_expression_outcome",
                            **calculate_expression_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_calculate_expression_outcome_grammar"}]
                if (
                    predicate is None
                    and requirement.get("requirement_type") == "tool_argument"
                    and observation_contract.get("parameter") == "passengers"
                ):
                    passenger_then_text = (
                        (source_branch.get("branch_context") or {}).get("gwt") or {}
                    ).get("then")
                    passenger_reduction_rule = _lookup_requirement_rule(
                        tables["PASSENGER_REDUCTION_RULES"],
                        requirement.get("requirement_text"),
                        passenger_then_text,
                    )
                    if passenger_reduction_rule is not None:
                        predicate = {
                            "predicate_kind": "passenger_count_changed",
                            **passenger_reduction_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_passenger_reduction_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "semantic_requirement":
                    then_text = ((source_branch.get("branch_context") or {}).get("gwt") or {}).get("then")
                    route_field_rule = _lookup_requirement_rule(
                        tables["FLIGHT_ROUTE_FIELD_RULES"],
                        requirement.get("requirement_text"),
                        then_text,
                    )
                    if route_field_rule is not None:
                        predicate = {
                            "predicate_kind": "flight_route_field_unchanged",
                            **route_field_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_flight_route_field_grammar"}]
                if predicate is None and requirement.get("requirement_type") in (
                    "semantic_requirement", "tool_argument", "tool_argument_constraint",
                ):
                    # Broadened from semantic_requirement-only (docs/agentcoveragetesting_reuse_log.md
                    # section 31): the evaluator branch this compiles to (see
                    # evaluate_oracle_contract_extended's predicate_kind ==
                    # "structurally_unreachable_action" handling) never inspects
                    # candidate_kind or observation_contract shape at all -- it
                    # unconditionally returns "pass" once evaluator_status is
                    # executable, since the fact holds for every possible
                    # trajectory by construction of the real tool schema. A
                    # tool_argument-shaped claim referencing a value/state that
                    # provably cannot exist in the real schema (e.g. a
                    # LineStatus value telecom's real enum doesn't have) is the
                    # same shape of fact, just bound to a tool/parameter instead
                    # of floating free -- there was no principled reason to gate
                    # this to semantic_requirement only.
                    unreachable_action_rule = _lookup_requirement_rule(tables["STRUCTURALLY_UNREACHABLE_ACTION_RULES"], requirement.get("requirement_text"))
                    if unreachable_action_rule is not None:
                        predicate = {
                            "predicate_kind": "structurally_unreachable_action",
                            **unreachable_action_rule,
                        }
                        extra_diagnostics = [
                            {"code": "compiled_structurally_unreachable_action_grammar"}
                        ]
                if predicate is None and requirement.get("requirement_type") == "tool_argument_constraint":
                    path_validity_rule = _FLIGHT_PATH_VALIDITY_RULES.get(
                        requirement.get("requirement_text")
                    )
                    if path_validity_rule is not None:
                        predicate = {
                            "predicate_kind": "flight_path_valid",
                            **path_validity_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_flight_path_validity_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    path_validity_from_reservation_rule = (
                        _FLIGHT_PATH_VALIDITY_FROM_RESERVATION_RULES.get(
                            requirement.get("requirement_text")
                        )
                    )
                    if path_validity_from_reservation_rule is not None:
                        predicate = {
                            "predicate_kind": "flight_path_valid_from_reservation",
                            **path_validity_from_reservation_rule,
                        }
                        extra_diagnostics = [
                            {"code": "compiled_flight_path_validity_from_reservation_grammar"}
                        ]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    airport_in_network_rule = _AIRPORT_IN_NETWORK_RULES.get(
                        requirement.get("requirement_text")
                    )
                    if airport_in_network_rule is not None:
                        predicate = {
                            "predicate_kind": "airport_in_network",
                            **airport_in_network_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_airport_in_network_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument":
                    chronological_feasibility_rule = _FLIGHTS_CHRONOLOGICAL_FEASIBILITY_RULES.get(
                        requirement.get("requirement_text")
                    )
                    if chronological_feasibility_rule is not None:
                        predicate = {
                            "predicate_kind": "flights_chronologically_feasible",
                            **chronological_feasibility_rule,
                        }
                        extra_diagnostics = [
                            {"code": "compiled_flights_chronological_feasibility_grammar"}
                        ]
                if predicate is None and requirement.get("requirement_type") == "tool_argument_constraint":
                    structural_invariant_rule = _STRUCTURAL_INVARIANT_RULES.get(
                        requirement.get("requirement_text")
                    )
                    if structural_invariant_rule is not None:
                        predicate = {
                            "predicate_kind": "structural_invariant_holds",
                            **structural_invariant_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_structural_invariant_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument_constraint":
                    flights_unchanged_rule = _lookup_requirement_rule(tables["FLIGHTS_UNCHANGED_RULES"], requirement.get("requirement_text"))
                    if flights_unchanged_rule is not None:
                        predicate = {
                            "predicate_kind": "flights_unchanged_from_reservation",
                            **flights_unchanged_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_flights_unchanged_grammar"}]
                if predicate is None and requirement.get("requirement_type") == "tool_argument_constraint":
                    user_directory_uniqueness_rule = _lookup_requirement_rule(tables["USER_DIRECTORY_UNIQUENESS_RULES"], requirement.get("requirement_text"))
                    if user_directory_uniqueness_rule is not None:
                        predicate = {
                            "predicate_kind": "unique_by_directory_reference",
                            **user_directory_uniqueness_rule,
                        }
                        extra_diagnostics = [{"code": "compiled_user_directory_uniqueness_grammar"}]
                if predicate is not None:
                    record = deepcopy(record)
                    record.pop("evaluator_contract_fingerprint", None)
                    record["evaluator_status"] = "executable"
                    record["program"] = {
                        "evaluator_kind": "all_matches_predicate",
                        "requires_observation": bool(
                            record["expected_observation"].get("requires_observation")
                        ),
                        "predicate": predicate,
                        "program_source": PROGRAM_SOURCE,
                        **(
                            {"predicate_alternatives": predicate_alternatives}
                            if predicate_alternatives is not None
                            else {}
                        ),
                        **(
                            {"declined_without_attempting_alternative": declined_without_attempting_alternative}
                            if declined_without_attempting_alternative is not None
                            else {}
                        ),
                        **(
                            {"lookup_discovery": lookup_discovery}
                            if lookup_discovery is not None
                            else {}
                        ),
                    }
                    record["diagnostics"] = list(record["diagnostics"]) + extra_diagnostics + [
                        {"code": "compiled_by_extension_v1"}
                    ]
                    record["evaluator_contract_fingerprint"] = content_sha256(record)
            elif baggage_allowance:
                # Unconditional override, not gated on evaluator_status == "deferred":
                # once the binding extension marks a binding "bound" with
                # baggage_allowance data, the frozen baseline compiler sees
                # binding_status == "bound" and expected_observation.operator ==
                # "present" and happily builds its own (wrong, generic "did the tool
                # get called at all") match_cardinality program for it -- it never
                # reaches "deferred" because the *binding* side already looks
                # resolved. This extension's own binding data must be authoritative
                # over whatever the frozen evaluator compiler guessed from it.
                record = deepcopy(record)
                record.pop("evaluator_contract_fingerprint", None)
                record["evaluator_status"] = "executable"
                record["program"] = {
                    "evaluator_kind": "all_matches_predicate",
                    "requires_observation": bool(
                        record["expected_observation"].get("requires_observation")
                    ),
                    "predicate": _baggage_allowance_predicate(baggage_allowance),
                    "program_source": PROGRAM_SOURCE,
                }
                record["diagnostics"] = list(record["diagnostics"]) + [
                    {"code": "compiled_baggage_allowance_grammar"},
                    {"code": "compiled_by_extension_v1"},
                ]
                record["evaluator_contract_fingerprint"] = content_sha256(record)
            elif is_turn_shape_constraint:
                # Unconditional override, same reason as baggage_allowance
                # above: the binding extension already marked this "bound",
                # so the frozen evaluator compiler sees bound + operator ==
                # "absent" (this Then is a "must not" rule) and short-circuits
                # to its own generic match_cardinality program before ever
                # looking at requirement_type -- it never reaches "deferred".
                # That generic program would try extract_bound_observations
                # against an extractor_kind it doesn't recognize and always
                # come back "unavailable", never pass/fail.
                record = deepcopy(record)
                record.pop("evaluator_contract_fingerprint", None)
                record["evaluator_status"] = "executable"
                record["program"] = {
                    "evaluator_kind": "all_matches_predicate",
                    "requires_observation": bool(
                        record["expected_observation"].get("requires_observation")
                    ),
                    "predicate": {"predicate_kind": "no_concurrent_message_and_tool_call"},
                    "program_source": PROGRAM_SOURCE,
                }
                record["diagnostics"] = list(record["diagnostics"]) + [
                    {"code": "compiled_turn_shape_constraint_grammar"},
                    {"code": "compiled_by_extension_v1"},
                ]
                record["evaluator_contract_fingerprint"] = content_sha256(record)
            elif is_tool_call_any_of:
                # Unconditional override, same reason as baggage_allowance/
                # is_turn_shape_constraint above (docs/agentcoveragetesting_
                # reuse_log.md section 79/104): the binding extension already
                # marked this "bound" via a real OR-route disjunction
                # (matcher.tool_names), so the frozen baseline compiler sees
                # bound + operator == "present" and happily builds its own
                # generic match_cardinality program with NO tool_name filter
                # at all -- it would vacuously pass for ANY tool call
                # whatsoever, not just one of the real disjuncts, silently
                # losing the entire point of the check.
                record = deepcopy(record)
                record.pop("evaluator_contract_fingerprint", None)
                record["evaluator_status"] = "executable"
                record["program"] = {
                    "evaluator_kind": "all_matches_predicate",
                    "requires_observation": bool(
                        record["expected_observation"].get("requires_observation")
                    ),
                    "predicate": {"predicate_kind": "tool_call_any_of"},
                    "program_source": PROGRAM_SOURCE,
                }
                record["diagnostics"] = list(record["diagnostics"]) + [
                    {"code": "compiled_tool_call_any_of_grammar"},
                    {"code": "compiled_by_extension_v1"},
                ]
                record["evaluator_contract_fingerprint"] = content_sha256(record)
            elif at_most_n_rule:
                # Unconditional override, same reason as baggage_allowance/
                # is_turn_shape_constraint above: this is a real tool_call
                # requirement the frozen baseline already sees as
                # binding_status == "bound", so it happily builds its own
                # generic match_cardinality program from
                # expected_observation.operator -- but the modality-prefix
                # matcher that produced that operator can only ever emit
                # "absent"/"none" for a "must not X" Then, which is wrong
                # for a real "must not X MORE THAN N TIMES" cardinality bound
                # (docs/agentcoveragetesting_reuse_log.md section 77,
                # Category D sub-cluster 2): retail_060's three real
                # modify_pending_order_*#b0v0 branches ("The agent must not
                # call modify_pending_order_* more than once for the same
                # order") compiled to operator=absent/quantifier=none, which
                # forbids even the ONE legitimate call the agent must make to
                # do the task at all -- an unconditionally-failing false
                # requirement, not a real gap. Keyed by requirement_id (not
                # requirement_text, which this exact phrase shares with dozens
                # of unrelated "must/must not call this tool at all" branches
                # across the corpus -- confirmed via corpus grep before
                # scoping this to requirement_id).
                record = deepcopy(record)
                record.pop("evaluator_contract_fingerprint", None)
                record["evaluator_status"] = "executable"
                record["expected_observation"] = {
                    "operator": "predicate_true",
                    "quantifier": "all_matching_events",
                    "requires_observation": False,
                }
                record["program"] = {
                    "evaluator_kind": "all_matches_predicate",
                    "requires_observation": False,
                    "predicate": {
                        "predicate_kind": "call_count_le",
                        "threshold": at_most_n_rule["threshold"],
                        "scope_arg": at_most_n_rule.get("scope_arg"),
                    },
                    "program_source": PROGRAM_SOURCE,
                }
                record["diagnostics"] = list(record["diagnostics"]) + [
                    {"code": "compiled_at_most_n_call_grammar"},
                    {"code": "compiled_by_extension_v1"},
                ]
                record["evaluator_contract_fingerprint"] = content_sha256(record)
            record = _mark_optional_argument_value_predicate(record, requirement)
            record = _reject_fabricated_requirement(record)
            record = _with_cabin_unchanged_program(record, requirement, tables)
            if (
                record["requirement_id"] in _REQUIRES_OBSERVATION_FALSE_OVERRIDE
                and record.get("program", {}).get("requires_observation")
            ):
                # Real requires_observation fix (see the constant's own
                # docstring above) -- a pure boolean flip applied on top of
                # whatever program shape the baseline/other branches above
                # already built (frozen json_type or this module's own
                # unique_by_directory_reference for these 2 real cases),
                # never replacing the predicate itself.
                record = deepcopy(record)
                record.pop("evaluator_contract_fingerprint", None)
                record["program"] = dict(record["program"])
                record["program"]["requires_observation"] = False
                record["expected_observation"] = dict(record["expected_observation"])
                record["expected_observation"]["requires_observation"] = False
                record["diagnostics"] = list(record["diagnostics"]) + [
                    {"code": "compiled_requires_observation_override"},
                    {"code": "compiled_by_extension_v1"},
                ]
                record["evaluator_contract_fingerprint"] = content_sha256(record)
            record = _with_prohibition_observation_scope(record, binding)
            contracts.append(record)
            counts[record["evaluator_status"]] += 1
        statuses = {item["evaluator_status"] for item in contracts}
        if not statuses:
            # Explicit empty-collection case (docs/agentcoveragetesting_
            # reuse_log.md section 110): an empty set is a subset of any set,
            # so `statuses <= {"executable"}` is vacuously True when `contracts`
            # is empty -- without this branch, a branch whose accepted
            # requirement set (and therefore its evaluator contracts, compiled
            # 1:1 from it) is empty would fall through to "executable", a
            # falsely-healthy status for a branch with zero real contracts to
            # execute. Checked first on purpose: this is a pure addition, not
            # a behavior change, for every branch with at least one real
            # contract (statuses is never empty in that case).
            branch_status = "branch_has_no_oracle_content"
        elif "needs_adjudication" in statuses:
            branch_status = "needs_adjudication"
        elif statuses <= {"executable"}:
            branch_status = "executable"
        else:
            branch_status = "partially_executable"
            deferred_branches.append(source_branch["branch_id"])
        branches.append(
            {
                "branch_id": source_branch["branch_id"],
                "branch_context": deepcopy(source_branch["branch_context"]),
                "evaluator_contracts": contracts,
                "evaluator_status": branch_status,
            }
        )

    result = {
        "schema_version": EXTENDED_SET_VERSION,
        "source_evaluator_contract_set_fingerprint": baseline["evaluator_contract_set_fingerprint"],
        "branches": branches,
        "summary": {
            "branch_count": len(branches),
            "evaluator_contract_count": sum(len(b["evaluator_contracts"]) for b in branches),
            "evaluator_status_counts": {
                status: counts[status] for status in sorted(EVALUATOR_STATUSES)
            },
            "fully_executable_branch_count": sum(
                b["evaluator_status"] == "executable" for b in branches
            ),
            "partially_executable_branch_count": sum(
                b["evaluator_status"] == "partially_executable" for b in branches
            ),
            "adjudication_branch_count": sum(
                b["evaluator_status"] == "needs_adjudication" for b in branches
            ),
        },
        "deferred_branches": deferred_branches,
        "next_stage": "runtime_oracle_evaluation",
    }
    result["evaluator_contract_set_fingerprint"] = content_sha256(result)
    return result


def validate_oracle_evaluator_contract_set_extended(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$evaluator_contract_set_extended")))
    fingerprint = result.pop("evaluator_contract_set_fingerprint", None)
    if result.get("schema_version") != EXTENDED_SET_VERSION or fingerprint != content_sha256(result):
        raise ThenAtomizationError("invalid extended Oracle evaluator contract set")
    identities = []
    counts: Counter[str] = Counter()
    for branch in result.get("branches") or []:
        for raw in branch.get("evaluator_contracts") or []:
            record = deepcopy(dict(_mapping(raw, "$.evaluator_contracts[]")))
            record_fingerprint = record.pop("evaluator_contract_fingerprint", None)
            if record.get("schema_version") != EVALUATOR_VERSION or record_fingerprint != content_sha256(record):
                raise ThenAtomizationError("invalid Oracle evaluator contract")
            if record.get("branch_id") != branch.get("branch_id"):
                raise ThenAtomizationError("Oracle evaluator contract branch mismatch")
            if record.get("evaluator_status") not in EVALUATOR_STATUSES:
                raise ThenAtomizationError("invalid Oracle evaluator status")
            identities.append(record.get("evaluator_contract_id"))
            counts[record["evaluator_status"]] += 1
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("Oracle evaluator contract IDs must be unique")
    summary = _mapping(result.get("summary"), "$.summary")
    if summary.get("evaluator_contract_count") != len(identities):
        raise ThenAtomizationError("Oracle evaluator contract count mismatch")
    expected = {status: counts[status] for status in sorted(EVALUATOR_STATUSES)}
    if summary.get("evaluator_status_counts") != expected:
        raise ThenAtomizationError("Oracle evaluator status counts mismatch")
    result["evaluator_contract_set_fingerprint"] = fingerprint
    return result


# docs/agentcoveragetesting_reuse_log.md section 137.5.3/140 (task_09886b79).
def _shape_needs_tolerant_argument_disclosure_retry(runtime_binding: Mapping[str, Any]) -> bool:
    runtime = runtime_binding.get("runtime_binding") or {}
    if runtime.get("relation") != "precedes":
        return False
    left = runtime.get("left_event")
    if not isinstance(left, Mapping):
        return False
    matcher = left.get("matcher") or {}
    return matcher.get("kind") == "contains_right_call_argument_values"


def _retry_temporal_precedes_with_tolerant_argument_disclosure(
    evaluator_contract: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recomputes the observation using extract_temporal_precedes_
    observations_tolerant_argument_disclosure (runtime_observation_binding_
    extension_v1.py) and assembles a verdict for it the SAME way the frozen
    evaluate_oracle_contract does for evaluator_kind ==
    "all_target_events_have_predecessor" (oracle_evaluator_contract_v1.py
    lines ~510-521) -- a small, stable, 6-line piece of frozen logic
    mirrored here rather than imported, since the frozen file exposes no
    smaller reusable unit for just that step and this file must never edit
    the frozen one."""
    runtime = runtime_binding.get("runtime_binding") or {}
    observation = extract_temporal_precedes_observations_tolerant_argument_disclosure(
        runtime, events, driver_bindings
    )
    if observation is None or observation.get("status") != "observed":
        return None
    program = _mapping(evaluator_contract.get("program"), "$.program")
    if program.get("evaluator_kind") != "all_target_events_have_predecessor":
        return None
    target_count = int(observation.get("target_event_count") or 0)
    matched_count = int(observation.get("matched_target_count") or 0)
    missing_required = bool(program.get("requires_observation")) and target_count == 0
    passed = not missing_required and target_count == matched_count
    return {
        "verdict": "pass" if passed else "fail",
        "target_event_count": target_count,
        "matched_target_count": matched_count,
        "missing_required_observation": missing_required,
        "observation": observation,
    }


def _evaluate_oracle_contract_extended_core(
    evaluator_contract: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
    flight_route_reference: Mapping[str, Mapping[str, str]] | None = None,
    flight_schedule_reference: Mapping[str, Mapping[str, str]] | None = None,
    semantic_judge: Callable[..., Mapping[str, Any]] | None = None,
    retail_user_directory_reference: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    program = evaluator_contract.get("program") or {}
    # section 193: a prohibition's declared observation scope (see
    # _with_prohibition_observation_scope) -- applied here, before the frozen
    # delegation and every extension extractor alike.
    runtime_binding = _narrowed_runtime_binding(runtime_binding, program)
    runtime_for_shape_check = runtime_binding.get("runtime_binding") or {}
    # A list-shaped left_event (runtime_observation_binding_extension_v1's
    # "conjunctive prerequisites" shape -- prerequisite_tool_call and, as of
    # this round, assistant_dialogue_act) can appear on a program the FROZEN
    # Step4 compiler produced on its own (relation_true -> its generic
    # all_target_events_have_predecessor evaluator_kind doesn't inspect
    # left_event's shape at compile time, only extractor_kind/relation), so
    # program_source alone is not a reliable signal here: such a program
    # never gets program_source stamped (only this module's OWN predicate_kinds
    # do), so delegating to the frozen evaluate_oracle_contract purely because
    # program_source is absent would call the frozen extract_bound_observations,
    # which raises on a list left_event ($.left_event must be an object) --
    # confirmed against real airline_097_order#b0-shaped data, see
    # docs/oracle_requirement_pipeline_v0_7.md section 24.
    left_event_is_list = isinstance(runtime_for_shape_check.get("left_event"), list)
    if program.get("program_source") != PROGRAM_SOURCE and not left_event_is_list:
        frozen_result = evaluate_oracle_contract(evaluator_contract, runtime_binding, events, driver_bindings)
        # docs/agentcoveragetesting_reuse_log.md section 137.5.3/140
        # (task_09886b79): a real, narrow second-chance retry for the one
        # frozen shape (temporal "precedes" relation whose left_event is a
        # real assistant_argument_disclosure with matcher.kind ==
        # "contains_right_call_argument_values") the frozen evaluator's own
        # literal-substring matcher is real-confirmed too strict for
        # (reasonable customer-service paraphrasing of internal ids/
        # tokens) -- see _shape_needs_tolerant_argument_disclosure_retry.
        # Purely additive: only consulted when the frozen result already
        # says "fail", and only ADOPTED when the tolerant re-evaluation
        # itself real-passes -- every check that already passes under the
        # frozen evaluator (the overwhelming majority) is completely
        # unaffected, and a genuine fail that the tolerant matcher can't
        # explain stays a fail with its original frozen diagnostics intact.
        if frozen_result.get("verdict") == "fail" and _shape_needs_tolerant_argument_disclosure_retry(
            runtime_binding
        ):
            tolerant_result = _retry_temporal_precedes_with_tolerant_argument_disclosure(
                evaluator_contract, runtime_binding, events, driver_bindings
            )
            if tolerant_result is not None and tolerant_result.get("verdict") == "pass":
                return tolerant_result
        return frozen_result

    if evaluator_contract.get("evaluator_status") != "executable":
        return {
            "verdict": "unavailable",
            "reason": f"evaluator_status={evaluator_contract.get('evaluator_status')}",
        }
    if evaluator_contract.get("binding_id") != runtime_binding.get("binding_id"):
        raise ThenAtomizationError("evaluator contract and runtime binding do not match")
    if evaluator_contract.get("source_binding_fingerprint") != runtime_binding.get("binding_fingerprint"):
        raise ThenAtomizationError("evaluator contract uses a stale runtime binding")
    predicate_kind = (program.get("predicate") or {}).get("predicate_kind")
    if predicate_kind == "structurally_unreachable_action":
        # No observation_contract.channel exists for this requirement at all
        # (see _STRUCTURALLY_UNREACHABLE_ACTION_RULES above) -- unlike every
        # other predicate_kind here, this one needs no runtime_binding/events
        # extraction whatsoever, since the fact it asserts holds for every
        # possible trajectory by construction of the real tool schema.
        return {"verdict": "pass", "reason": program["predicate"].get("reason", "")}
    if (
        predicate_kind
        in (
            "flight_route_field_unchanged",
            "flight_path_valid",
            "flight_path_valid_from_reservation",
            "airport_in_network",
        )
        and flight_route_reference is None
    ):
        return {"verdict": "unavailable", "reason": "flight_route_reference_not_supplied"}
    if predicate_kind == "flights_chronologically_feasible" and flight_schedule_reference is None:
        return {"verdict": "unavailable", "reason": "flight_schedule_reference_not_supplied"}
    if predicate_kind == "unique_by_directory_reference" and retail_user_directory_reference is None:
        return {"verdict": "unavailable", "reason": "retail_user_directory_reference_not_supplied"}
    if predicate_kind == "semantic_transcript_judgment" and semantic_judge is None:
        # Same "required external input not supplied -> unavailable, not fail"
        # convention as the 3 reference-table guards above (see
        # docs/agentcoveragetesting_reuse_log.md section 88): without this
        # guard, no semantic_judge falls through to evaluate_predicate_extended's
        # own semantic_judge_not_configured branch, whose {"passed": False} is
        # folded into this function's own all_matches_predicate "passed" check
        # below and surfaces as a hard "fail" -- indistinguishable from a real,
        # substantive failure to every downstream branch_verdict rollup that
        # was never told to expect an unconfigured judge (batch runs that don't
        # pass --enable-semantic-judge never did).
        return {"verdict": "unavailable", "reason": "semantic_judge_not_supplied"}
    if predicate_kind == "flight_route_field_unchanged":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_flight_route_field_observations(
            runtime, events, program["predicate"]["target_tool_name"], flight_route_reference
        )
    elif predicate_kind == "flight_path_valid":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_flight_path_validity_observations(runtime, events, flight_route_reference)
    elif predicate_kind == "flight_path_valid_from_reservation":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_flight_path_validity_against_reservation_observations(
            runtime, events, flight_route_reference
        )
    elif predicate_kind == "airport_in_network":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_airport_in_network_observations(
            runtime, events, program["predicate"]["parameter"], flight_route_reference
        )
    elif predicate_kind == "flights_chronologically_feasible":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_flights_chronological_feasibility_observations(
            runtime, events, flight_schedule_reference
        )
    elif predicate_kind == "tool_call_error_matches":
        runtime = runtime_binding.get("runtime_binding") or {}
        lookup_discovery = program.get("lookup_discovery")
        if lookup_discovery is not None:
            observation = _extract_lookup_discovery_observations(runtime, events, lookup_discovery)
            if lookup_discovery.get("not_found_grounding") is not None:
                return _evaluate_lookup_discovery_grounding(program, observation, events, semantic_judge)
        else:
            observation = _extract_tool_outcome_observations(runtime, events)
        # section 133.4: the target tool was never called at all (zero real
        # matching events), so the primary literal-error evidence genuinely
        # cannot exist -- if this requirement's real requires_observation is
        # true (so silence would otherwise be missing_required_observation
        # -> a hard fail) AND a real declined_without_attempting_alternative
        # is configured for this requirement (see TOOL_CALL_OUTCOME_DECLINED_
        # ALTERNATIVE_RULES), consult it instead of failing outright -- see
        # _evaluate_declined_without_attempting's own docstring for the full
        # real root cause and why this is a genuinely different mechanism
        # from predicate_alternatives. A requirement with no configured
        # alternative (the overwhelming majority of Family A requirements)
        # falls through to the generic all_matches_predicate handling below,
        # byte-identical to before this addition.
        declined_alternative = program.get("declined_without_attempting_alternative")
        if (
            declined_alternative is not None
            and not observation["matches"]
            and bool(program.get("requires_observation"))
        ):
            return _evaluate_declined_without_attempting(declined_alternative, events, semantic_judge)
    elif predicate_kind == "compensation_formula":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_compensation_observations(
            runtime, events, program["predicate"]["lookup_tool_name"]
        )
    elif predicate_kind == "payment_reused_from_cancelled_reservation":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_payment_reuse_observations(
            runtime,
            events,
            program["predicate"]["argument_path"],
            program["predicate"]["argument_is_list"],
        )
    elif predicate_kind == "prior_call_consistency":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_prior_call_consistency_observations(
            runtime, events, program["predicate"], program.get("predicate_alternatives")
        )
    elif predicate_kind == "semantic_transcript_judgment":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_semantic_transcript_observations(
            runtime, events, program["predicate"]
        )
    elif predicate_kind == "cumulative_sum_le":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_cumulative_sum_observations(
            runtime, events, program["predicate"]
        )
    elif predicate_kind == "call_count_le":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_call_count_observations(
            runtime, events, program["predicate"]
        )
    elif predicate_kind == "tool_call_any_of":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_tool_call_any_of_observations(runtime, events)
    elif predicate_kind == "baggage_count_decreased":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_baggage_reduction_observations(
            runtime, events, program["predicate"]["argument_path"]
        )
    elif predicate_kind == "no_concurrent_message_and_tool_call":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_turn_shape_observations(runtime, events)
    elif predicate_kind == "passenger_count_changed":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_passenger_reduction_observations(runtime, events)
    elif predicate_kind == "flights_unchanged_from_reservation":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_flights_unchanged_observations(runtime, events)
    elif predicate_kind == "cabin_unchanged_from_reservation":
        runtime = runtime_binding.get("runtime_binding") or {}
        observation = _extract_cabin_unchanged_observations(runtime, events, driver_bindings)
    elif left_event_is_list:
        observation = extract_bound_observations_extended(runtime_binding, events, driver_bindings)
    else:
        observation = extract_bound_observations(runtime_binding, events, driver_bindings)
    if observation.get("status") != "observed":
        return {"verdict": "unavailable", "observation": observation}

    matches = observation["matches"]
    if program.get("evaluator_kind") == "all_target_events_have_predecessor":
        # Mirrors the frozen evaluate_oracle_contract's own
        # all_target_events_have_predecessor handling exactly (see
        # oracle_evaluator_contract_v1.py) -- this module has no predicate to
        # apply here (unlike all_matches_predicate below), it only needed the
        # list-aware extraction above instead of the frozen extractor.
        target_count = int(observation.get("target_event_count") or 0)
        matched_count = int(observation.get("matched_target_count") or 0)
        missing_required = bool(program.get("requires_observation")) and target_count == 0
        passed = not missing_required and target_count == matched_count
        return {
            "verdict": "pass" if passed else "fail",
            "target_event_count": target_count,
            "matched_target_count": matched_count,
            "missing_required_observation": missing_required,
            "observation": observation,
        }
    if program.get("evaluator_kind") != "all_matches_predicate":
        raise ThenAtomizationError("unsupported extended evaluator program")
    # section 103: a match whose projected value is None because the real
    # argument was genuinely OMITTED (not explicitly supplied and wrong) is
    # vacuously satisfied for a target this compiler marked optional
    # (program["target_argument_required"] is False, set by
    # _mark_optional_argument_value_predicate) -- never applies to a value
    # that was actually supplied, and never relaxes a required argument.
    optional_argument_absent = program.get("target_argument_required") is False
    evaluations = [
        {
            "event_index": match.get("event_index"),
            **(
                {"passed": True, "vacuous_optional_argument_absent": True}
                if optional_argument_absent and match.get("value") is None
                else evaluate_predicate_extended(
                    program["predicate"], match.get("value"), semantic_judge=semantic_judge,
                    retail_user_directory_reference=retail_user_directory_reference,
                )
            ),
        }
        for match in matches
    ]
    missing_required = bool(program.get("requires_observation")) and not matches
    passed = not missing_required and all(item["passed"] for item in evaluations)
    return {
        "verdict": "pass" if passed else "fail",
        "match_count": len(matches),
        "missing_required_observation": missing_required,
        "predicate_evaluations": evaluations,
        "observation": observation,
    }


# Section 134 bug 4 (telecom_113_order#b0::OR01, docs/agentcoveragetesting_
# reuse_log.md section 131.2/131.6, ticket task_986ef913 -- corrected scope
# from the withdrawn task_db03e6ef): real, confirmed systemic issue, present
# identically in BOTH this module's own all_matches_predicate tail above
# (program_source == "extension_v1") AND, for every check this module
# delegates to it, the FROZEN oracle_evaluator_contract_v1.py's own
# `evaluate_oracle_contract` (lines ~493-521, never edited in place --
# releases/agentspectesting-method-v1.1.0's legacy-compiler-drift check pins
# its bytes): `missing_required = bool(program.get("requires_observation"))
# and not matches` treats "the guarded event was never observed" as an
# unconditional hard fail, with no way to distinguish a real violation
# (the agent had every opportunity and simply never acted) from a
# conversation that was cut off -- for ANY real-world reason -- before the
# agent could plausibly have reached that point. telecom_113_order#b0's real
# online transcript is fully compliant up to where it stops (confirmed
# account, verified the line was not suspended, began legitimate
# troubleshooting) and never gets a chance to decide whether to escalate,
# because the real termination_reason is "user_stop" (a simulated user sent
# ###STOP###), not because the agent refused to call transfer_to_human_
# agents. Corpus-wide scan (this round) of every telecom
# all_matches_predicate + requires_observation=True executable check (52
# real checks: 29 ARG/15 STATE/8 ORDER) confirms this exact code path is the
# SOLE site that computes `missing_required_observation` for every one of
# them, regardless of branch kind or predicate_kind -- so the fix belongs at
# this single, generic site, not duplicated per predicate/branch kind.
#
# Real fix: this thin wrapper (the module's only public entry point from
# here on) reruns the exact same evaluation unchanged when no
# termination_reason is supplied (every existing caller/test -- none of
# which pass this new, optional kwarg -- gets byte-identical behavior), and
# only ever SOFTENS an outcome the core already marked
# missing_required_observation=True: if termination_reason is present and is
# NOT "agent_stop" (tau2's own real
# tau2.data_model.simulation.TerminationReason enum -- AGENT_STOP is the one
# value meaning the AGENT itself judged the conversation complete with
# nothing left to do; every other real value -- user_stop, max_steps,
# timeout, too_many_errors, agent_error, user_error,
# infrastructure_error, context_window_exceeded, unexpected_error -- means
# the conversation ended for a reason external to the agent's own
# compliance decision), the verdict is downgraded from "fail" to
# "unavailable" (the same convention this module already uses for
# semantic_judge_not_supplied/flight_route_reference_not_supplied above --
# "we could not determine compliance", never "compliant"). This can only
# ever turn a hard fail into "unavailable" (which rolls up to branch_verdict
# "incomplete", see driver/generic_tau_online_v1.py's own counts rollup) --
# it can never turn a fail into a pass, and it never touches any check that
# already has at least one real match (a genuine violation with prior
# matches is untouched). When termination_reason IS "agent_stop" and the
# guarded event still never appeared, the original hard fail is preserved
# unchanged -- the agent had a real, natural opportunity to act and did not.
def evaluate_oracle_contract_extended(
    evaluator_contract: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    driver_bindings: Mapping[str, Any],
    flight_route_reference: Mapping[str, Mapping[str, str]] | None = None,
    flight_schedule_reference: Mapping[str, Mapping[str, str]] | None = None,
    semantic_judge: Callable[..., Mapping[str, Any]] | None = None,
    retail_user_directory_reference: Mapping[str, int] | None = None,
    termination_reason: str | None = None,
) -> dict[str, Any]:
    result = _evaluate_oracle_contract_extended_core(
        evaluator_contract,
        runtime_binding,
        events,
        driver_bindings,
        flight_route_reference=flight_route_reference,
        flight_schedule_reference=flight_schedule_reference,
        semantic_judge=semantic_judge,
        retail_user_directory_reference=retail_user_directory_reference,
    )
    if (
        result.get("missing_required_observation") is True
        and termination_reason is not None
        and termination_reason != "agent_stop"
    ):
        result = dict(result)
        result["verdict"] = "unavailable"
        result["reason"] = "requires_observation_unmet_due_to_premature_termination"
        result["premature_termination_reason"] = termination_reason
    return result


def compile_oracle_evaluator_contracts_extended_file(
    *,
    accepted_requirements_path,
    runtime_bindings_path,
    output_path,
) -> dict[str, Any]:
    import json as _json
    from pathlib import Path as _Path

    def load(path):
        return _json.loads(_Path(path).read_text(encoding="utf-8"))

    result = compile_oracle_evaluator_contracts_extended(
        load(accepted_requirements_path), load(runtime_bindings_path)
    )
    target = _Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
