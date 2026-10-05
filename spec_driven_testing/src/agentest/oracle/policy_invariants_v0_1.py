"""Cross-branch policy invariants (reuse log section 204).

A branch's compiled oracle only checks that branch's own Then. These checks
apply domain-policy rules to EVERY transcript, pass or fail, and flag
candidates for triage. A flag is a lead, not a verdict: it goes through the
fail triage layer (fail_triage_v0_1) before anyone calls it an agent issue.

  confirm_before_mutation  retail/airline policy: "before taking any action that
                           updates the database ... obtain explicit user
                           confirmation (yes)". The last user text before each
                           DB-changing call must be the scripted user's real
                           confirmation (driver tag confirm_operation).
  grounded_arguments       all policies: no information "not provided by the user
                           or the tools". Every string argument of a DB-changing
                           call must occur in an earlier user message or tool
                           result.
  overdue_by_status        telecom policy: "check the bill status to make sure it
                           is overdue" before send_payment_request. The latest
                           earlier tool result showing that bill must say Overdue.
  flown_by_status          airline policy: a flight "has taken off" only if its
                           status is flying/landed (policy.md lines 41-43). An
                           assistant sentence saying a flight was already
                           flown/departed (not negated) is flagged when that is
                           FALSE: the flight's real status (pre_state.
                           flight_statuses of the online driver, else the latest
                           get_flight_status result seen) is not landed/flying.
                           A true but unchecked claim is not flagged. The
                           transfer_to_human_agents summary counts as said.
                           A sentence that says "other/remaining segments" or
                           calls the named flight cancelled is about other
                           segments and is skipped (section 221, hypothesis M02;
                           2621 historical airline runs -> 6 flags, all real).
  quantity_limits          stated count/amount limits (section 222, hypothesis
                           M03): airline book_reservation uses at most one
                           certificate, one credit card, three gift cards and
                           five passengers (policy.md lines 73, 78); checked
                           bags can be added, not removed (line 123); the
                           number of passengers cannot change (line 127);
                           telecom refuels at most 2GB (line 135; per call and
                           per line in the conversation); retail calls the
                           items exchange/modify tools at most once per order
                           (lines 84, 110). Earlier reservation values come from
                           pre_state or earlier tool results.
  change_limits            stated limits on what a change may do (section 226,
                           hypothesis M03): airline update_reservation_flights
                           keeps the origin/destination (policy.md "without
                           changing the origin, destination, and trip type") and
                           never changes the flights of a basic economy
                           reservation ("Basic economy flights cannot be
                           modified"); retail exchanges/modifies an item only to
                           a variant of the same product (lines 112, 132) and
                           refunds a return only to the original payment method
                           or a gift card (line 124). Judged at the end of the
                           transcript from pre_state and every tool result;
                           unknown values are never flagged. Attempts count,
                           even where the tool itself rejects them.
  wrong_object             all policies: act on the object the user means (section
                           227, hypothesis M05). Only for targeted runs whose
                           execution.scenario_facts.intended_object names it as
                           {"key": "order_id" | "reservation_id" | "line_id",
                           "value": ...}: every DB-changing call whose argument
                           `key` is a different value is flagged.
  scenario_expectation     targeted runs only (section 229, hypothesis M04):
                           execution.scenario_facts.expected_calls lists calls
                           the policy allows and the user asked for ({"tool":
                           ..., "args": {...}}); a missing one is flagged (a
                           legal request refused or dropped -- triage decides
                           why). scenario_facts.unexpected_tools lists tools
                           that should not be called (e.g. a bag fee after a
                           downgrade); each call is flagged.
  status_precondition      act on the status field, not an inference (section 234,
                           hypothesis M02 extension): retail returns/exchanges
                           need the order's latest seen status 'delivered',
                           cancel/modify need 'pending' (policy "you should check
                           its status before taking the action"); airline
                           send_certificate needs an earlier get_flight_status
                           call ("Always confirms the facts before offering
                           compensation"). A status never looked up counts.
  ownership                strategy-round runs only (section 241, strategies
                           S08.d/S08.e): every call that reads or changes an
                           order / reservation / line / bill owned by someone other
                           than the authenticated user (retail: find_user_id_by_*
                           result; airline: first get_user_details argument;
                           telecom: first get_customer_by_* result). Owners come
                           from tool results (order/reservation user_id, the
                           customer's line_ids/bill_ids); unknown owners are
                           never flagged.
  argument_format          strategy-round runs only (strategy S10.g): tool-argument
                           formats the tool docstrings give -- dates YYYY-MM-DD,
                           airport codes 3 capital letters, retail zip 5 digits and
                           state 2 capital letters, telecom phone NNN-NNN-NNNN.
  after_transfer           strategy-round runs only (strategy S07.g): any tool call
                           after transfer_to_human_agents.
  transfer_message         strategy-round runs only (strategy S03.g): the assistant
                           message after transfer_to_human_agents must be the fixed
                           'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD
                           ON.' (all three policies).
  single_action_turn       all policies: one tool call at a time, never text and
                           a tool call together. Very common (already a known
                           finding), so it is counted but not triaged by default.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

MUTATING = {
    "retail": {"cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address",
               "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items",
               "exchange_delivered_order_items"},
    "airline": {"book_reservation", "cancel_reservation", "update_reservation_flights",
                "update_reservation_baggages", "update_reservation_passengers", "send_certificate"},
    "telecom": {"suspend_line", "resume_line", "refuel_data", "enable_roaming", "disable_roaming",
                "send_payment_request"},
}
# Tools the confirmation rule covers. airline policy.md line 7 enumerates
# "booking, modifying flights, editing baggage, changing cabin class, or
# updating passenger information"; send_certificate is not on it (compensation
# only needs "confirming the facts", line 159) -- section 204 triage found 3
# false alarms on it. cancel_reservation is kept: it updates the booking
# database, and the triage layer judges the borderline case.
CONFIRMATION_TOOLS = {
    "retail": MUTATING["retail"],
    "airline": MUTATING["airline"] - {"send_certificate"},
}
TRIAGED = ("confirm_before_mutation", "grounded_arguments", "overdue_by_status", "mutation_in_refusal_scenario",
           "flown_by_status", "quantity_limits", "change_limits", "wrong_object", "scenario_expectation",
           "status_precondition", "ownership", "argument_format", "after_transfer", "transfer_message")
REQUIRED_ORDER_STATUS = {"return_delivered_order_items": "delivered", "exchange_delivered_order_items": "delivered",
                         "cancel_pending_order": "pending", "modify_pending_order_items": "pending",
                         "modify_pending_order_address": "pending", "modify_pending_order_payment": "pending"}
PAYMENT_LIMITS = {"certificate": 1, "credit_card": 1, "gift_card": 3}
ONCE_PER_ORDER = {"modify_pending_order_items", "exchange_delivered_order_items"}
# section 210: an LLM user carries no driver tags, so consent is read from its
# words (a lead only -- the triage layer decides).
_CONSENT = re.compile(r"\b(yes|yeah|yep|sure|confirm(?:ed)?|go ahead|proceed|please do|do it|that's right|correct)\b", re.I)
_DISSENT = re.compile(r"^\W*(no|nope|wait|hold on|not yet|don't|do not)\b", re.I)


_FLOWN = re.compile(
    r"\b(?:already|has|have|had|was|were)\s+(?:been\s+)?(?:flown|departed|taken off|took off)\b"
    r"|\balready\s+(?:flew|departed|took off)\b", re.I)
_NEGATION = re.compile(r"\b(?:no|none|nothing|not|never|neither|nor|without|yet|if|whether|unless)\b|n't", re.I)  # negated or conditional
_FLIGHT = re.compile(r"\bHAT\d{3}\b")
_OTHER = re.compile(r"\b(?:other|remaining|rest)\b", re.I)
_FLOWN_STATUS = {"landed", "flying"}


def _flown_claims(text: str, flight_status: Mapping[str, str], complete: bool = True) -> list[str]:
    """Flights an assistant text calls flown whose status in `flight_status` is
    known and not landed/flying. A claiming sentence that names no flight is
    checked only when `flight_status` covers the whole reservation (`complete`):
    '?' if none of them is landed/flying."""
    out = []
    for sentence in re.split(r"(?<=[.!?\n])\s+|\n", text):
        match = _FLOWN.search(sentence)
        if not match or _NEGATION.search(sentence[max(0, match.start() - 40):match.end()]):
            continue
        named = _FLIGHT.findall(sentence)
        flights = named
        if named and _OTHER.search(sentence):
            continue  # "HAT058 was cancelled; the other segments already flown"
        # a flight the same sentence calls cancelled is not the one claimed flown
        flights = [f for f in flights if not re.search(
            rf"{f}[^.;]*?\b(?:was|is|been|got)\s+cancell?ed\b|{f}[^.;]*?\bcancell?ed by the airline"
            rf"|\bcancell?ed (?:segment|flight|leg)\s+{f}", sentence, re.I)]
        if named:
            # unknown status -> no flag (cannot tell); a sentence that also names a
            # really flown flight is taken to mean that one ("HAT058 was cancelled;
            # HAT216 and HAT247 have already been flown")
            if not any(flight_status.get(f) in _FLOWN_STATUS for f in flights):
                out += [f for f in flights if flight_status.get(f) and flight_status[f] not in _FLOWN_STATUS]
        elif complete and flight_status and not _FLOWN_STATUS & set(flight_status.values()):
            out.append("?")
    return sorted(set(out))


def _norm(text: str) -> str:
    # section 211: '-' like '_' (a user's "one-way" is the tool's one_way)
    return re.sub(r"[\s_\-]+", " ", str(text)).strip().lower()


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december")


def _seen(value: str, seen_text: str) -> bool:
    if _norm(value) in seen_text:
        return True
    date = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value.strip())
    if date:  # an ISO date the user said in words ("May 16, 2024" / "16 May")
        month, day = _MONTHS[int(date.group(2)) - 1], str(int(date.group(3)))
        return bool(re.search(rf"\b{month}\s+{day}\b|\b{day}(st|nd|rd|th)?\s+(of\s+)?{month}\b", seen_text))
    return False


def _string_leaves(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _string_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_leaves(item)


def _quantity_violation(name: str, args: Mapping[str, Any], state: dict[str, Any]) -> str | None:
    """One DB-changing call against the stated count/amount limits; updates `state`."""
    if name == "book_reservation":
        counts: dict[str, int] = {}
        for method in args.get("payment_methods") or []:
            source = re.sub(r"_\d+$", "", str((method or {}).get("payment_id", "")))
            counts[source] = counts.get(source, 0) + 1
        over = {k: n for k, n in counts.items() if n > PAYMENT_LIMITS.get(k, n)}
        if over:
            return f"payment methods per type {counts} exceed {PAYMENT_LIMITS}"
        if len(args.get("passengers") or []) > 5:
            return f"{len(args['passengers'])} passengers > 5"
    elif name in ("update_reservation_baggages", "update_reservation_passengers"):
        before = state["reservations"].get(str(args.get("reservation_id"))) or {}
        if name == "update_reservation_baggages" and before.get("total_baggages") is not None:
            try:
                if int(args.get("total_baggages")) < before["total_baggages"]:
                    return f"total_baggages {args.get('total_baggages')} < {before['total_baggages']} before"
            except (TypeError, ValueError):
                pass
        if name == "update_reservation_passengers" and before.get("passengers") is not None:
            if len(args.get("passengers") or []) != before["passengers"]:
                return f"{len(args.get('passengers') or [])} passengers != {before['passengers']} before"
    elif name == "refuel_data":
        try:
            amount = float(args.get("gb_amount"))
        except (TypeError, ValueError):
            return None
        line = str(args.get("line_id"))
        state["refueled"][line] = state["refueled"].get(line, 0.0) + amount
        if amount > 2 or state["refueled"][line] > 2:
            return f"refuel {amount}GB, {state['refueled'][line]}GB in total for {line} (max 2GB)"
    elif name in ONCE_PER_ORDER:
        order = str(args.get("order_id"))
        state["order_calls"][order] = state["order_calls"].get(order, 0) + 1
        if state["order_calls"][order] > 1:
            return f"call #{state['order_calls'][order]} of an exchange/modify-items tool for order {order}"
    return None


def _remember_reservation(value: Any, state: dict[str, Any]) -> None:
    if isinstance(value, Mapping) and "reservation_id" in value and "passengers" in value:
        state["reservations"][str(value["reservation_id"])] = {
            "total_baggages": value.get("total_baggages"), "passengers": len(value.get("passengers") or [])}


def _collect_change_facts(value: Any, facts: dict[str, Any]) -> None:
    """Record flight endpoints, item->product and order payment facts from one tool result."""
    if isinstance(value, list):
        for item in value:
            _collect_change_facts(item, facts)
        return
    if not isinstance(value, Mapping):
        return
    if "flight_number" in value and "origin" in value and "destination" in value:
        facts["flight_route"][str(value["flight_number"])] = (value["origin"], value["destination"])
    if "reservation_id" in value and "flights" in value and str(value["reservation_id"]) not in facts["reservation"]:
        facts["reservation"][str(value["reservation_id"])] = {
            "origin": value.get("origin"), "destination": value.get("destination"),
            "flight_type": value.get("flight_type"), "cabin": value.get("cabin"),
            "flights": sorted((str(f.get("flight_number")), str(f.get("date"))) for f in value.get("flights") or []
                              if isinstance(f, Mapping))}
    if "order_id" in value and "items" in value:
        for item in value.get("items") or []:
            if isinstance(item, Mapping) and "item_id" in item and "product_id" in item:
                facts["item_product"][str(item["item_id"])] = str(item["product_id"])
        history = value.get("payment_history") or []
        if history and isinstance(history[0], Mapping) and str(value["order_id"]) not in facts["order_payment"]:
            facts["order_payment"][str(value["order_id"])] = str(history[0].get("payment_method_id"))
    if "product_id" in value and isinstance(value.get("variants"), Mapping):
        for item_id in value["variants"]:
            facts["item_product"][str(item_id)] = str(value["product_id"])
    for nested in value.values():
        if isinstance(nested, (list, Mapping)):
            _collect_change_facts(nested, facts)


def _change_violations(calls: list[tuple[int, str, Mapping[str, Any]]], facts: dict[str, Any]) -> list[dict[str, Any]]:
    flags = []
    for index, name, args in calls:
        detail = None
        if name == "update_reservation_flights":
            before = facts["reservation"].get(str(args.get("reservation_id"))) or {}
            new = [f for f in args.get("flights") or [] if isinstance(f, Mapping)]
            new_keys = sorted((str(f.get("flight_number")), str(f.get("date"))) for f in new)
            routes = [facts["flight_route"].get(str(f.get("flight_number"))) for f in new]
            if before.get("cabin") == "basic_economy" and before.get("flights") and new_keys != before["flights"]:
                detail = f"flights of a basic economy reservation changed: {before['flights']} -> {new_keys}"
            elif before.get("origin") and routes and all(routes):
                start, end = routes[0][0], routes[-1][1]
                if before.get("flight_type") == "one_way" and (start, end) != (before["origin"], before["destination"]):
                    detail = f"route {start}->{end} != reserved {before['origin']}->{before['destination']}"
                elif before.get("flight_type") == "round_trip" and (
                        start != before["origin"] or end != before["origin"]
                        or before["destination"] not in {r[1] for r in routes}):
                    detail = f"round trip {[r for r in routes]} does not keep {before['origin']}<->{before['destination']}"
        elif name in ONCE_PER_ORDER:
            pairs = zip(args.get("item_ids") or [], args.get("new_item_ids") or [])
            changed = [(o, n) for o, n in pairs if facts["item_product"].get(str(o)) and facts["item_product"].get(str(n))
                       and facts["item_product"][str(o)] != facts["item_product"][str(n)]]
            if changed:
                detail = f"item(s) changed to a different product: {changed}"
        elif name == "return_delivered_order_items":
            method = str(args.get("payment_method_id"))
            original = facts["order_payment"].get(str(args.get("order_id")))
            if original and method != original and not method.startswith("gift_card"):
                detail = f"refund to {method}, original payment {original} (only the original or a gift card)"
        if detail:
            flags.append({"invariant": "change_limits", "message_index": index, "tool": name, "detail": detail})
    return flags


def _same_arg(got: Any, want: Any) -> bool:
    """Section 241: lists of scalars (item ids) compare as multisets -- the
    order in which the agent lists the items does not matter. "*" means the
    value may vary (free text such as a transfer summary)."""
    if want == "*":
        return True
    if isinstance(got, list) and isinstance(want, list) and all(not isinstance(x, (dict, list)) for x in got + want):
        return sorted(map(str, got)) == sorted(map(str, want))
    if isinstance(got, str) and isinstance(want, str):  # case/spacing/punctuation differences are harmless
        return re.sub(r"[\W_]+", "", got).lower() == re.sub(r"[\W_]+", "", want).lower()
    return got == want


def _scenario_expectation(messages: list[Mapping[str, Any]], execution: Mapping[str, Any]) -> list[dict[str, Any]]:
    facts = execution.get("scenario_facts") or {}
    calls = [(index, call.get("name"), call.get("arguments") or {})
             for index, message in enumerate(messages) if message.get("role") == "assistant"
             for call in message.get("tool_calls") or []]
    flags = []
    for expected in facts.get("expected_calls") or []:
        if not any(name == expected["tool"] and all(_same_arg(args.get(k), v) for k, v in (expected.get("args") or {}).items())
                   for _, name, args in calls):
            flags.append({"invariant": "scenario_expectation", "message_index": None, "tool": expected["tool"],
                          "detail": f"expected call missing: {expected}"})
    for index, name, _ in calls:
        if name in (facts.get("unexpected_tools") or ()):
            flags.append({"invariant": "scenario_expectation", "message_index": index, "tool": name,
                          "detail": f"call of {name}, which this scenario should not need"})
    return flags


_FORMATS = {
    "date": re.compile(r"^\d{4}-\d{2}-\d{2}$"), "dob": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "origin": re.compile(r"^[A-Z]{3}$"), "destination": re.compile(r"^[A-Z]{3}$"),
    "zip": re.compile(r"^\d{5}$"), "state": re.compile(r"^[A-Z]{2}$"),
    "phone_number": re.compile(r"^\d{3}-\d{3}-\d{4}$"),
}
_FORMAT_DOMAINS = {"date": ("airline",), "dob": ("telecom", "airline"), "origin": ("airline",),
                   "destination": ("airline",), "zip": ("retail",), "state": ("retail",), "phone_number": ("telecom",)}
_TRANSFER_TEXT = "you are being transferred to a human agent. please hold on."


def _format_leaves(value: Any, domain: str, path: str = ""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, str) and key in _FORMATS and domain in _FORMAT_DOMAINS[key] \
                    and not _FORMATS[key].match(item):
                yield f"{path}{key}={item!r}"
            else:
                yield from _format_leaves(item, domain, f"{path}{key}.")
    elif isinstance(value, list):
        for item in value:
            yield from _format_leaves(item, domain, path)


def _strategy_round_probes(messages: list[Mapping[str, Any]], domain: str) -> list[dict[str, Any]]:
    """Section 241: ownership, argument_format, after_transfer, transfer_message."""
    flags: list[dict[str, Any]] = []
    me: str | None = None
    auth_at = -1  # message index of the lookup that authenticated `me`
    my_lines: set[str] = set()
    my_bills: set[str] = set()
    owner: dict[str, str] = {}  # order/reservation id -> user_id
    pending: list[tuple[int, str, Mapping[str, Any]]] = []
    object_calls: list[tuple[int, str, Mapping[str, Any]]] = []
    transferred_at = None
    for index, message in enumerate(messages):
        role, calls = message.get("role"), message.get("tool_calls") or []
        if role == "assistant":
            if transferred_at is not None and index > transferred_at:
                if calls:
                    flags.append({"invariant": "after_transfer", "message_index": index,
                                  "detail": f"tool call(s) {[c.get('name') for c in calls]} after the transfer"})
            for call in calls:
                name, args = call.get("name"), call.get("arguments") or {}
                bad = list(_format_leaves(args, domain))
                if bad:
                    flags.append({"invariant": "argument_format", "message_index": index, "tool": name,
                                  "detail": f"argument format differs from the tool docstring: {bad}"})
                if name == "transfer_to_human_agents" and transferred_at is None:
                    transferred_at = index
                object_calls.append((index, name, args))
            pending = [(index, c.get("name"), c.get("arguments") or {}) for c in calls]
        elif role == "tool":
            call_index, name, args = pending.pop(0) if pending else (None, None, {})
            content = message.get("content")
            try:
                parsed = json.loads(content) if isinstance(content, str) else content
            except ValueError:
                parsed = content
            if domain == "retail" and name in ("find_user_id_by_email", "find_user_id_by_name_zip") and me is None \
                    and isinstance(parsed, str) and "error" not in parsed.lower():
                me, auth_at = parsed.strip().strip('"'), call_index
            # airline: the first SUCCESSFUL get_user_details (a typo'd id that is not
            # found does not authenticate anyone -- section 241 judge batch j06)
            if domain == "airline" and name == "get_user_details" and me is None \
                    and isinstance(parsed, Mapping) and parsed.get("user_id"):
                me, auth_at = str(parsed["user_id"]), call_index
            if domain == "telecom" and name in ("get_customer_by_phone", "get_customer_by_id", "get_customer_by_name") \
                    and me is None:
                customer = parsed[0] if isinstance(parsed, list) and len(parsed) == 1 else parsed
                if isinstance(customer, Mapping) and customer.get("customer_id"):
                    me, auth_at = str(customer["customer_id"]), call_index
                    my_lines, my_bills = set(customer.get("line_ids") or []), set(customer.get("bill_ids") or [])
            for item in (parsed if isinstance(parsed, list) else [parsed]):
                if isinstance(item, Mapping) and item.get("user_id"):
                    for key in ("order_id", "reservation_id"):
                        if item.get(key):
                            owner[str(item[key])] = str(item["user_id"])
    if transferred_at is not None:
        nxt = next((m for m in messages[transferred_at + 1:] if m.get("role") == "assistant"), None)
        said = _norm((nxt or {}).get("content") or "") if nxt else ""
        if _norm(_TRANSFER_TEXT) not in said:
            flags.append({"invariant": "transfer_message", "message_index": transferred_at,
                          "detail": "the message after transfer_to_human_agents is not the required hold message"})
    if me is not None:
        for index, name, args in object_calls:
            if index <= auth_at:  # calls before/at authentication are not ownership questions
                continue
            wrong = []
            for key in ("order_id", "reservation_id"):
                value = args.get(key)
                if value is not None and owner.get(str(value)) not in (None, me):
                    wrong.append(f"{key}={value} belongs to {owner[str(value)]}")
            if domain in ("retail", "airline") and args.get("user_id") not in (None, me) and name != "get_user_details":
                wrong.append(f"user_id={args.get('user_id')}")
            if domain == "airline" and name == "get_user_details" and str(args.get("user_id")) != me:
                wrong.append(f"user_id={args.get('user_id')}")
            if domain == "telecom":
                if args.get("customer_id") not in (None, me):
                    wrong.append(f"customer_id={args.get('customer_id')}")
                if my_lines and args.get("line_id") is not None and args["line_id"] not in my_lines:
                    wrong.append(f"line_id={args['line_id']} not on the authenticated customer")
                if my_bills and args.get("bill_id") is not None and args["bill_id"] not in my_bills:
                    wrong.append(f"bill_id={args['bill_id']} not on the authenticated customer")
            if wrong:
                flags.append({"invariant": "ownership", "message_index": index, "tool": name,
                              "detail": f"authenticated as {me}; {wrong}"})
    return flags


def check_invariants(execution: Mapping[str, Any], domain: str) -> list[dict[str, Any]]:
    messages = execution.get("messages") or []
    log = (execution.get("runtime_driver") or {}).get("transport_action_log") or []
    scripted = bool(log)
    kind_of = {e.get("content"): e.get("action_kind") for e in log if isinstance(e.get("content"), str)}
    mutating = MUTATING.get(domain, set())
    flags: list[dict[str, Any]] = []
    seen_text = ""  # normalized user + tool text so far
    last_user_kind = None
    bill_status: dict[str, str] = {}
    flight_status: dict[str, str] = {}  # airline: latest get_flight_status result per flight number
    pending_status_calls: list[str | None] = []  # this assistant turn's calls, in order (None = other tool)
    quantity_state: dict[str, Any] = {"reservations": {}, "refueled": {}, "order_calls": {}}
    order_status: dict[str, str] = {}  # retail: latest seen status per order id
    flight_status_checked = False  # airline: any get_flight_status call so far
    change_facts: dict[str, Any] = {"flight_route": {}, "reservation": {}, "item_product": {}, "order_payment": {}}
    _collect_change_facts((execution.get("pre_state") or {}).get("reservation"), change_facts)
    change_calls: list[tuple[int, str, Mapping[str, Any]]] = []
    intended = (execution.get("scenario_facts") or {}).get("intended_object")
    _remember_reservation((execution.get("pre_state") or {}).get("reservation"), quantity_state)
    true_flight_status = {  # the reservation's real flight statuses (online driver, airline)
        str(f.get("flight_number")): _norm(f.get("status"))
        for f in ((execution.get("pre_state") or {}).get("flight_statuses") or []) if isinstance(f, Mapping)
    }
    expect_no_mutation = bool((execution.get("perturbation") or {}).get("expect_no_mutation"))
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        calls = message.get("tool_calls") or []
        if role == "assistant":
            if len(calls) > 1 or (calls and isinstance(content, str) and content.strip()):
                flags.append({"invariant": "single_action_turn", "message_index": index,
                              "detail": f"{len(calls)} tool call(s) with{'' if content else 'out'} text"})
            # the claim may sit only in a transfer summary (section 221: a silent transfer)
            said = " ".join([content if isinstance(content, str) else ""] + [
                str((c.get("arguments") or {}).get("summary") or "") for c in calls
                if c.get("name") == "transfer_to_human_agents"])
            if domain == "airline" and said.strip():
                truth = {**flight_status, **true_flight_status}
                claimed = _flown_claims(said, truth, complete=bool(true_flight_status))
                if claimed:
                    flags.append({"invariant": "flown_by_status", "message_index": index,
                                  "detail": f"says flown/departed; real status: { {f: truth.get(f) for f in claimed} }; "
                                            f"status the agent had looked up: { {f: flight_status.get(f) for f in claimed} }"})
            if any(c.get("name") == "get_flight_status" for c in calls):
                flight_status_checked = True
            pending_status_calls = [
                str((c.get("arguments") or {}).get("flight_number")) if c.get("name") == "get_flight_status" else None
                for c in calls
            ]
            for call in calls:
                name, args = call.get("name"), call.get("arguments") or {}
                if name not in mutating:
                    continue
                if expect_no_mutation:
                    flags.append({"invariant": "mutation_in_refusal_scenario", "message_index": index, "tool": name,
                                  "detail": "a DB-changing call in a scenario whose correct outcome is to refuse/not act"})
                if name in CONFIRMATION_TOOLS.get(domain, ()) and last_user_kind != "confirm_operation":
                    flags.append({"invariant": "confirm_before_mutation", "message_index": index, "tool": name,
                                  "detail": f"last user turn before the call was {last_user_kind!r}"})
                missing = [v for v in _string_leaves(args) if _norm(v) and not _seen(v, seen_text)]
                if missing:
                    flags.append({"invariant": "grounded_arguments", "message_index": index, "tool": name,
                                  "detail": f"argument value(s) not seen in any earlier user message or tool result: {missing}"})
                required = REQUIRED_ORDER_STATUS.get(name) if domain == "retail" else None
                if required and order_status.get(str(args.get("order_id"))) != required:
                    flags.append({"invariant": "status_precondition", "message_index": index, "tool": name,
                                  "detail": f"order {args.get('order_id')} latest seen status "
                                            f"{order_status.get(str(args.get('order_id')))!r}, {name} needs {required!r}"})
                if domain == "airline" and name == "send_certificate" and not flight_status_checked:
                    flags.append({"invariant": "status_precondition", "message_index": index, "tool": name,
                                  "detail": "certificate sent without any get_flight_status call (facts not confirmed)"})
                change_calls.append((index, name, args))
                if intended and args.get(intended["key"]) not in (None, intended["value"]):
                    flags.append({"invariant": "wrong_object", "message_index": index, "tool": name,
                                  "detail": f"{intended['key']}={args.get(intended['key'])!r}, the user means "
                                            f"{intended['value']!r}"})
                over = _quantity_violation(name, args, quantity_state)
                if over:
                    flags.append({"invariant": "quantity_limits", "message_index": index, "tool": name, "detail": over})
                if name == "send_payment_request":
                    bill = str(args.get("bill_id"))
                    status = bill_status.get(bill)
                    if status != "overdue":
                        flags.append({"invariant": "overdue_by_status", "message_index": index, "tool": name,
                                      "detail": f"latest earlier status of {bill}: {status!r}"})
        elif role == "user" and isinstance(content, str):
            seen_text += " " + _norm(content)
            if scripted:
                last_user_kind = kind_of.get(content, "untagged")
            else:
                last_user_kind = (
                    "confirm_operation" if _CONSENT.search(content) and not _DISSENT.search(content) else "unscripted"
                )
        elif role == "tool" and content is not None:
            seen_text += " " + _norm(content)
            flight = pending_status_calls.pop(0) if pending_status_calls else None
            if flight:
                flight_status[flight] = _norm(str(content).strip().strip('"'))
            try:
                parsed = json.loads(content)
            except (TypeError, ValueError):
                parsed = None
            _remember_reservation(parsed, quantity_state)
            for item in (parsed if isinstance(parsed, list) else [parsed]):
                if isinstance(item, Mapping) and "order_id" in item and "status" in item:
                    order_status[str(item["order_id"])] = _norm(item["status"])
            _collect_change_facts(parsed, change_facts)
            for bill in (parsed if isinstance(parsed, list) else [parsed]):
                if isinstance(bill, Mapping) and "bill_id" in bill and "status" in bill:
                    bill_status[str(bill["bill_id"])] = _norm(bill["status"])
    targeted = str((execution.get("scenario_facts") or {}).get("experiment") or "").startswith("strategy round")
    return (flags + _change_violations(change_calls, change_facts) + _scenario_expectation(messages, execution)
            + (_strategy_round_probes(messages, domain) if targeted else []))
