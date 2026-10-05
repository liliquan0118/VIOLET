"""Layer 2 (prompt construction) and Layer 3 (mechanical validation) of the
generic tool-call synthesis pipeline (see
docs/oracle_requirement_pipeline_v0_7.md section 53). Layer 2's prompt is
the ONLY place this pipeline asks an LLM anything -- it decides how an
ambiguous Given quantifier reads, which real candidate (if any) the branch
resolves to, whether a single tool call is honestly determinate, and if so
its arguments and the user's initial message. Layer 3 never trusts that
answer -- every claim it makes about real data is re-checked here against
the real facts Layer 1 already gathered.

Domain-agnostic by construction: nothing here knows what a "reservation" or
a "cabin" is. The prompt is built entirely from the branch's own real text
and Layer 1's real structured facts; the schema check is real JSON Schema
validation against whatever tool schema was supplied.
"""
from __future__ import annotations

import json

ID_LIKE_FIELD_TABLES = {"reservation_id": "reservations", "user_id": "users", "flight_number": "flights"}


class ToolCallSynthesisError(ValueError):
    pass


def render_tool_call_synthesis_prompt(branch_context, tool_menu, request_text, given_text, oracle_texts, default_filling_policy=None):
    """branch_context: Layer 1's resolve_fixture_context() output.
    tool_menu: {tool_name: {"description":..., "parameters": <real JSON schema>}} --
    either the single already-classified tool, or all real tools when the
    upstream role classification was honestly "not_determined".
    default_filling_policy: the real, domain-supplied config (e.g. this
    project's inputs/step7/baseline_policy_v0_1.json) describing how to
    fill in fields the request/given text leaves unstated -- a swappable
    per-domain input, not something this module hardcodes; a different
    domain supplies its own. Without it, an opening message that only
    partially specifies a multi-field tool call has no principled way to
    become a concrete, determinate call and would be forced into
    no_operation far more often than the branch actually needs."""
    payload = {
        "branch_id": branch_context["branch_id"],
        "real_request_text": request_text,
        "real_given_text": given_text,
        "real_oracle_requirement_texts": oracle_texts,
        "real_candidates": branch_context["candidates"],
        "real_synthesized_fallback_candidate": branch_context["synthesized_candidate"],
        "real_non_database_given_conditions": branch_context["non_database_conditions"],
        "candidate_tool_schemas": tool_menu,
        "domain_default_filling_policy": default_filling_policy,
        "real_route_reference": branch_context.get("real_route_reference"),
    }
    return f"""You are preparing ONE test case for an automated evaluation of a customer-service agent. You are not the agent -- you are deciding what a simulated CUSTOMER would plausibly say as their opening message, and (only if a single real tool call is genuinely determinate) what that tool call would be.

Below is real data for one branch: the real user request text, the real "Given" precondition text, the real Oracle requirements this branch will be checked against, every real candidate object (user, and where applicable a bound root object) Layer 1 found in the real database, each with its own Given database conditions already evaluated against it (a condition's "truth" is true/false when resolved, or null with raw "matching_witnesses"/"instances" data when the condition involves a collection and the source text doesn't specify whether it means "at least one" or "all" -- read the real given text yourself to resolve this, do not guess), and (when present) this domain's own real default-filling policy -- a config that says how to fill fields the request/given text itself leaves unstated (e.g. "use the requester's own real owned credit card, else a real gift card with enough real balance", "default to 0 baggage", "default cabin economy"). When this policy is present and covers a field, use it to build a concrete, complete, correct tool call -- do NOT answer no_operation merely because the opening message didn't spell out a value the policy already determines. Only answer no_operation when a field has no real, policy-backed or given-backed default AND multiple genuinely different values would be equally plausible with no principled way to choose (a real, irreducible ambiguity), or when no real tool in candidate_tool_schemas can honestly do what's being asked.

When real_route_reference is present, it is a REAL, schedule-valid, correctly priced route already found in the real database for each real cabin class (a plain list of real flight legs with real flight_number/date/origin/destination/price) -- when the branch needs a brand-new booking's flights/origin/destination/price and none of that is otherwise determined by the given text, use the entry for whichever cabin the given text/policy calls for (or the policy's default cabin) as the real basis for those fields and for computing a correct total price; do not invent flight numbers, dates, or prices when a real reference route is available. A null entry for a cabin means no real route was found for it in the configured search horizon -- treat that honestly (try another real cabin's reference route, or given the branch's genuine constraints, conclude no_operation) rather than inventing one anyway.

A field that represents a SERVICE PROVIDER'S OWN DISCRETIONARY DECISION (the clearest real example: send_certificate's "amount", a compensation figure the agent/policy decides, never something the customer states or negotiates) has NO principled value you can pick, even a "plausible, round" one -- inventing one and declaring it fabricated is NOT the right move for this specific kind of field, because unlike an id or a route choice, there is no real fact for the fabrication to approximate; treat it exactly like any other genuinely irreducible ambiguity and answer no_operation. Contrast this with fields the CUSTOMER genuinely would state (how many bags, which cabin, a made-up id they believe is theirs) -- those are fine to fill via the default-filling policy or, when the branch's own given text implies a specific fact, via a concrete customer-stated value.

Real data:
{json.dumps(payload, indent=2, default=str)}

Your job:
1. Resolve the Given: for each candidate (or the synthesized fallback, or none if fixture_root is null), decide whether it genuinely makes the Given true, using ONLY the real observations given -- resolve any null-truth collection condition by reading the real given text's own quantifier language, and do not treat a real consistency check failure as disqualifying unless the given text's meaning is unrelated to it.
2. Decide: does a single tool call from candidate_tool_schemas genuinely, determinately follow from the real request/given/oracle text (using the domain_default_filling_policy above to fill any field it doesn't state)? If yes, which tool, and what arguments (must satisfy that tool's real JSON schema exactly, and any amount charged to a real gift card or certificate must not exceed that real payment method's own real recorded balance). If the request is ambiguous between multiple tools, asks for something no real tool can do, or truly has a field with no default and no principled single answer, answer "no_operation" instead -- do not force a tool call that isn't genuinely singular and correct, but also do not reach for no_operation just because a policy-fillable field wasn't spelled out verbatim.

   IMPORTANT: an Oracle requirement saying the agent must NOT do something, or must refuse, or forbidding some action under the current real state, is NOT a reason to answer no_operation by itself. You are not the agent and are not deciding what the agent should correctly do -- you are constructing what the CUSTOMER would plausibly ask for, which is very often exactly the thing a correctly-behaving agent should refuse (that is the whole point of a prohibition-style Oracle requirement: it can only be tested by a customer genuinely asking for that thing). If a single real tool call would let the customer's literal request be carried out (even if a correct agent should decline it), construct that tool call and message as normal. Reserve no_operation only for when no real, determinate tool call construction is possible at all -- genuine multi-tool ambiguity, a request no real tool schema supports in any form, or a required field with no real/default/given-backed value and no principled way to pick one.
3. Write the initial_user_message a real customer would plausibly say to open this conversation, consistent with the real request/given text (including any real non_database_given_conditions, which describe facts the user states in conversation, not database facts) and with whatever concrete details you filled in via the default-filling policy (the message should state them, since the user is the one supplying these facts).
4. If any argument value is not literally present in the real data above (e.g. you must supply an id that provably does NOT exist, because the given text requires that), list it in fabricated_fields with a one-line reason -- never silently invent a value without declaring it. Values you derived mechanically from the default-filling policy (e.g. computing a real route's total price from real per-date prices) are not "fabricated" -- only flag values with no real basis at all.

Respond with ONLY a JSON object of this exact shape:
{{
  "given_resolution": {{"chosen_source": {{"kind": "candidate", "index": 0}} | {{"kind": "synthesized"}} | {{"kind": "owned", "candidate_index": 0, "table": "...", "object_index": 0}} | {{"kind": "none"}}, "explanation": "..."}},
  "decision": "call_tool" | "no_operation",
  "tool_name": "..." (only if call_tool),
  "arguments": {{...}} (only if call_tool),
  "fabricated_fields": {{"field_name": "reason"}} (optional),
  "initial_user_message": "...",
  "rationale": "..." (required if no_operation, explaining why no single tool call is correct)
}}"""


def _resolve_chosen_source(branch_context, chosen_source):
    kind = chosen_source.get("kind")
    if kind == "none":
        return None, None
    if kind == "synthesized":
        candidate = branch_context["synthesized_candidate"]
        if candidate is None:
            raise ToolCallSynthesisError("chosen_source references synthesized_candidate but none was offered")
        return candidate["user"], candidate["root_object"]
    if kind == "candidate":
        index = chosen_source.get("index")
        matches = [c for c in branch_context["candidates"] if c["candidate_index"] == index]
        if not matches:
            raise ToolCallSynthesisError(f"chosen_source references unknown candidate_index {index}")
        candidate = matches[0]
        return candidate["user"], candidate["root_object"]
    if kind == "owned":
        index, table, obj_index = chosen_source.get("candidate_index"), chosen_source.get("table"), chosen_source.get("object_index")
        matches = [c for c in branch_context["candidates"] if c["candidate_index"] == index]
        if not matches:
            raise ToolCallSynthesisError(f"chosen_source references unknown candidate_index {index}")
        owned = matches[0].get("related_owned_objects", {}).get(table) or []
        if not (isinstance(obj_index, int) and 0 <= obj_index < len(owned)):
            raise ToolCallSynthesisError(f"chosen_source references unknown owned object {table}[{obj_index}]")
        return matches[0]["user"], owned[obj_index]["object"]
    raise ToolCallSynthesisError(f"unsupported chosen_source kind {kind!r}")


def _collect_strings(value, pool):
    if isinstance(value, str):
        pool.add(value)
    elif isinstance(value, dict):
        for v in value.values():
            _collect_strings(v, pool)
    elif isinstance(value, list):
        for v in value:
            _collect_strings(v, pool)


def _real_object_pool(database, user, root_object, branch_context=None):
    """Every real object value this branch legitimately has access to, used
    to check that a non-fabricated argument value is actually real. Includes
    the chosen candidate's own user/root_object, plus (when no candidate
    binding is relevant at all, e.g. calculate/search/flight-status
    branches) the real route reference and any real candidates' own data
    Layer 1 gathered, so a real flight_number/date pulled from there isn't
    mistaken for an unverified invention."""
    pool = set()
    for obj in (user, root_object):
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str):
                    pool.add(v)
    if branch_context:
        _collect_strings(branch_context.get("real_route_reference"), pool)
        for candidate in branch_context.get("candidates") or []:
            _collect_strings(candidate.get("user"), pool)
            _collect_strings(candidate.get("root_object"), pool)
        synthesized = branch_context.get("synthesized_candidate")
        if synthesized:
            _collect_strings(synthesized.get("root_object"), pool)
    return pool


def _validate_payment_amounts(arguments, user, fabricated):
    """Real, structural check -- not a hand-rolled fare calculator: a
    payment_id argument (bare, or inside a list of {payment_id, amount}
    entries -- the real book_reservation/update_reservation_* shapes) must
    be one of the resolved user's real payment methods unless declared
    fabricated, and any amount charged to a real gift_card/certificate must
    not exceed that method's own real recorded balance (credit_card has no
    stored balance in this domain, so no check applies there). This is what
    caught the real gap found in the pilot: an LLM answer charged $4986 to
    a real gift card whose real balance was $245, and nothing checked it."""
    if not isinstance(user, dict):
        return
    real_methods = user.get("payment_methods") or {}

    def check_one(payment_id, amount, field_label):
        if field_label in fabricated or payment_id in fabricated:
            return
        real = real_methods.get(payment_id)
        if real is None:
            raise ToolCallSynthesisError(f"{field_label}={payment_id!r} is not a real payment method owned by the resolved user")
        if real.get("source") in ("gift_card", "certificate") and amount is not None:
            balance = real.get("amount")
            if isinstance(balance, (int, float)) and isinstance(amount, (int, float)) and amount > balance:
                raise ToolCallSynthesisError(f"{field_label} charges {amount} to {payment_id!r} but its real balance is only {balance}")

    for field, value in arguments.items():
        if field == "payment_id" and isinstance(value, str):
            check_one(value, None, field)
        elif isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict) and isinstance(entry.get("payment_id"), str):
                    check_one(entry["payment_id"], entry.get("amount"), field)


def validate_tool_call_synthesis(branch_context, tool_menu, answer, database, tool_schemas):
    """Never trusts the LLM's answer. Re-derives the chosen real object(s)
    from branch_context itself (never from the answer's own claims about
    them), validates arguments against the REAL tool schema (jsonschema,
    not a hand-rolled check), and for any argument shaped like a real
    identity field (reservation_id/user_id/flight_number) not declared in
    fabricated_fields, requires it to match something Layer 1 actually
    supplied; declared-fabricated values are independently checked absent
    from the real database, never trusted at face value."""
    from .generic_tau_airline_v1 import GenericDriverBindingError, _validate_required_arguments

    if not isinstance(answer, dict) or answer.get("decision") not in ("call_tool", "no_operation"):
        raise ToolCallSynthesisError("answer missing a valid decision")
    message = answer.get("initial_user_message")
    if not isinstance(message, str) or not message.strip():
        raise ToolCallSynthesisError("initial_user_message must be a non-empty string")

    given_resolution = answer.get("given_resolution") or {}
    chosen_source = given_resolution.get("chosen_source") or {"kind": "none"}
    user, root_object = _resolve_chosen_source(branch_context, chosen_source)

    if answer["decision"] == "no_operation":
        if not (isinstance(answer.get("rationale"), str) and answer["rationale"].strip()):
            raise ToolCallSynthesisError("no_operation requires a non-empty rationale")
        return {"decision": "no_operation", "rationale": answer["rationale"], "initial_user_message": message,
                "user": user, "root_object": root_object}

    tool_name = answer.get("tool_name")
    if tool_name not in tool_menu:
        raise ToolCallSynthesisError(f"tool_name {tool_name!r} not in the offered candidate_tool_schemas")
    arguments = answer.get("arguments")
    if not isinstance(arguments, dict):
        raise ToolCallSynthesisError("call_tool requires an arguments object")
    try:
        _validate_required_arguments({"tool_name": tool_name, "arguments": arguments}, tool_schemas)
    except GenericDriverBindingError as exc:
        raise ToolCallSynthesisError(f"argument schema validation failed: {exc}") from exc

    fabricated = answer.get("fabricated_fields") or {}
    real_pool = _real_object_pool(database, user, root_object, branch_context)
    for field, value in arguments.items():
        if field not in ID_LIKE_FIELD_TABLES or not isinstance(value, str):
            continue
        if field in fabricated:
            table = ID_LIKE_FIELD_TABLES[field]
            if value in database.get(table, {}):
                raise ToolCallSynthesisError(f"{field}={value!r} was declared fabricated but is a REAL {table} record")
        elif value not in real_pool:
            raise ToolCallSynthesisError(
                f"{field}={value!r} is neither a real value from the resolved candidate nor declared in fabricated_fields")
    _validate_payment_amounts(arguments, user, fabricated)

    return {"decision": "call_tool", "tool_name": tool_name, "arguments": arguments,
            "fabricated_fields": fabricated, "initial_user_message": message, "user": user, "root_object": root_object}
