"""Prepare ordinary baseline cases from reviewed intent and fixed snapshot data.

This module does not execute tools, produce adversarial variations or search feedback.
Step 8 must enforce the declared environment and interaction contracts at runtime.
"""
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta

from ..compiler.artifacts import content_sha256
from ..compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .generic_tau_airline_v1 import _validate_required_arguments, GenericDriverBindingError
from .v5_candidate_binding_v1 import _predicate_observation
from .v5_related_binding_v1 import _dated_flights, _supported_dated_path
from .v5_object_references_v1 import SnapshotObjects, inspect_candidates, seal
from .v5_role_handoff_v1 import check_target_origin


SCHEMA = "agentspectesting.v5-step7-package/v0.1"


class PreparationGap(ValueError):
    pass


def validate_policy(policy):
    if policy["schema_version"] != "agentspectesting.step7-baseline-policy/v0.1":
        raise ValueError("unsupported baseline policy")
    if (set(policy)!={"schema_version","scope","booking_defaults","route_search","existing_reservation","disclosure"}
            or policy["scope"]!="ordinary_functional_baseline_no_prompt_mutation_or_feedback_search"):
        raise ValueError("unsupported baseline policy fields or scope")
    route=policy["route_search"]; defaults=policy["booking_defaults"]
    if (defaults["passenger_selection"]!="requester_self" or defaults["payment_selection"]!="owned_credit_card_else_sufficient_gift_card"
            or defaults["flight_type"]!="one_way" or defaults["insurance"]!="no"
            or type(defaults["total_baggages"]) is not int or defaults["total_baggages"]!=0
            or type(defaults["nonfree_baggages"]) is not int or defaults["nonfree_baggages"]!=0
            or defaults["cabin"] not in ("basic_economy","economy","business")):
        raise ValueError("unsupported baseline booking preferences")
    for key,limit in (("horizon_days",30),("max_legs",3),("min_connection_minutes",1440)):
        if type(route[key]) is not int or not 1 <= route[key] <= limit:
            raise ValueError("route configuration out of supported bounds")
    if route["same_date_connections"] is not True or route["departure_after_reference_date"] is not True:
        raise ValueError("unsupported route time policy")
    if route["order"]!="date_then_flight_number_depth_first_no_repeated_airport":
        raise ValueError("unsupported route order")
    existing=policy["existing_reservation"]
    if existing["candidate_selection"]!="first_supplied_pair_passing_relation_given_and_consistency":
        raise ValueError("unsupported candidate selection policy")
    if any(existing[key] is not True for key in ("require_active","require_future_unflown_segments","require_creation_not_future")):
        raise ValueError("nominal consistency checks cannot be disabled")
    if (policy["disclosure"]["identity_passenger_payment"]!="on_agent_request_only"
            or policy["disclosure"]["full_database_record"]!="never"
            or policy["disclosure"]["oracle_then_expected_verdict"]!="never"):
        raise ValueError("unsupported disclosure policy")


def _index(rows):
    result = {r["branch_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate branch")
    return result


def _clock(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        raise PreparationGap("offset_clock_to_local_flight_time_mapping_not_supported")
    return parsed


def boolean_value(expression, truths):
    if "condition" in expression:
        return truths[expression["condition"]]
    if "not" in expression:
        value = boolean_value(expression["not"], truths)
        return None if value is None else not value
    op = next(iter(expression))
    values = [boolean_value(e, truths) for e in expression[op]]
    if op == "all":
        return False if False in values else None if None in values else True
    if op == "any":
        return True if True in values else None if None in values else False
    raise ValueError("unsupported Given operator")


def _given_observations(contract, candidate, user, reservation, database, clock, semantic):
    if contract["given_contract"]["non_database_conditions"]:
        raise PreparationGap("non_database_given_requires_additional_adapter")
    conditions = contract["given_contract"]["database_conditions"]
    if not conditions:
        text = contract["given_contract"]["text"]
        if text != "True" or contract["given_contract"]["non_database_conditions"]:
            raise PreparationGap("given_requires_additional_representation")
        return {"truth": True, "basis": "literal_true", "observations": []}
    if semantic is None or semantic["expression"] is None or semantic["unrepresented_text"]:
        raise PreparationGap("given_logic_missing_or_unrepresented")
    branch = {"fixture_requirements": contract["fixture_contract"]}
    observations = [_predicate_observation(c, branch, {"user":user,"root":reservation}, clock) for c in conditions]
    dated = [c for c in conditions if c["source_condition"]["table"] == "flights" and _supported_dated_path(c)]
    if dated:
        linked = _dated_flights({"private_fixture_context": {"root": reservation, "root_table": "reservations"}}, dated, database, clock)
        observations = [linked["condition_observations"].get(o["condition_id"], o) for o in observations]
    truths = {f"C{i+1}": o["truth"] for i,o in enumerate(observations)}
    return {"truth": boolean_value(semantic["expression"], truths), "basis": "reviewed_boolean_expression",
            "expression": deepcopy(semantic["expression"]), "observations": observations}


def _reservation_consistency(reservation, database, clock, policy):
    reasons = []
    try:
        created = _clock(reservation["created_at"])
        if policy["require_creation_not_future"] and created > clock:
            reasons.append("reservation_created_after_reference_clock")
    except (KeyError, TypeError, ValueError):
        reasons.append("creation_time_unverifiable")
    if policy["require_active"] and reservation.get("status") in ("cancelled", "canceled"):
        reasons.append("reservation_already_cancelled")
    segments = reservation.get("flights")
    if not isinstance(segments,list) or not segments:
        reasons.append("missing_booked_segments")
    else:
        for segment in segments:
            flight = database["flights"].get(segment.get("flight_number"), {})
            instance = flight.get("dates",{}).get(segment.get("date"), {})
            try:
                future = date.fromisoformat(segment["date"]) > clock.date()
            except (KeyError, ValueError, TypeError):
                future = False
            if policy["require_future_unflown_segments"] and (not future or instance.get("status") != "available"):
                reasons.append("segment_not_verified_future_and_available")
    return {"passed": not reasons, "reasons": sorted(set(reasons)),
            "basis": "configured_nominal_state_consistency_not_inferred_given_logic"}


def _constraint_match(arguments, constraints):
    for c in constraints:
        actual = arguments.get(c["field"])
        if c["operator"] == "eq":
            if content_sha256(actual) != content_sha256(c["value"]): return False
        elif not isinstance(actual,list): return False
        elif c["operator"] == "min_items" and len(actual) < c["value"]: return False
        elif c["operator"] == "max_items" and len(actual) > c["value"]: return False
    return True


def _route(database, clock, cabin, constraints, policy, price_cap=None):
    minimum, maximum, origin, destination = 1, policy["max_legs"], None, None
    for c in constraints:
        if c["field"] == "flights" and c["operator"] == "min_items": minimum = max(minimum,c["value"])
        elif c["field"] == "flights" and c["operator"] == "max_items": maximum = min(maximum,c["value"])
        elif c["field"] == "origin" and c["operator"] == "eq": origin = c["value"]
        elif c["field"] == "destination" and c["operator"] == "eq": destination = c["value"]
    if not 1 <= minimum <= maximum <= policy["max_legs"]:
        raise PreparationGap("flight_count_outside_configured_route_capability")
    by_day = {}
    for number, flight in sorted(database["flights"].items()):
        if flight.get("flight_number") != number: continue
        for day, instance in sorted(flight.get("dates",{}).items()):
            try:
                travel_date = date.fromisoformat(day)
                if not 0 < (travel_date-clock.date()).days <= policy["horizon_days"]: continue
                departure = datetime.fromisoformat(day+"T"+flight["scheduled_departure_time_est"])
                arrival = datetime.fromisoformat(day+"T"+flight["scheduled_arrival_time_est"])
                price = instance.get("prices",{}).get(cabin)
                seats = instance.get("available_seats",{}).get(cabin)
                if (departure.tzinfo is not None or arrival.tzinfo is not None or arrival <= departure
                        or instance.get("status") != "available" or type(seats) is not int or seats < 1
                        or type(price) is not int or price < 0): continue
                if not isinstance(flight.get("origin"),str) or not isinstance(flight.get("destination"),str): continue
                by_day.setdefault(day,[]).append({"flight_number":number,"date":day,"origin":flight["origin"],
                    "destination":flight["destination"],"departure":departure.isoformat(),"arrival":arrival.isoformat(),
                    "price":price,"instance_fingerprint":content_sha256(instance)})
            except (ValueError, TypeError, KeyError):
                continue
    gap = timedelta(minutes=policy["min_connection_minutes"])
    for day, instances in sorted(by_day.items()):
        def search(path, seen):
            if price_cap is not None and sum(s["price"] for s in path) > price_cap: return None
            if len(path) >= minimum and (destination is None or path[-1]["destination"] == destination):
                return path
            if len(path) >= maximum: return None
            for segment in instances:
                if not path:
                    if origin is not None and segment["origin"] != origin: continue
                    used = {segment["origin"]}
                else:
                    if segment["origin"] != path[-1]["destination"]: continue
                    if datetime.fromisoformat(segment["departure"]) < datetime.fromisoformat(path[-1]["arrival"]) + gap: continue
                    used = seen
                if segment["destination"] in used: continue
                found = search(path+[segment], used|{segment["destination"]})
                if found: return found
            return None
        result = search([],set())
        if result: return result
    raise PreparationGap("no_route_satisfies_constraints_within_configured_horizon")


def _booking(user, database, clock, request, policy):
    defaults = deepcopy(policy["booking_defaults"])
    allowed_equalities = {"cabin","flight_type","origin","destination","total_baggages","nonfree_baggages","insurance"}
    seen = {}
    for c in request["constraints"]:
        if c["field"] == "flights" and c["operator"] in ("min_items","max_items"): continue
        if c["operator"] != "eq" or c["field"] not in allowed_equalities:
            raise PreparationGap("explicit_request_constraint_not_supported_by_booking_adapter")
        if c["field"] in seen and seen[c["field"]] != c["value"]:
            raise PreparationGap("conflicting_explicit_constraints")
        seen[c["field"]] = c["value"]
        if c["field"] in defaults: defaults[c["field"]] = c["value"]
    if defaults["flight_type"] != "one_way" or defaults["insurance"] != "no" or defaults["total_baggages"] != 0 or defaults["nonfree_baggages"] != 0:
        raise PreparationGap("pricing_or_trip_configuration_not_supported")
    passenger = {"first_name":user.get("name",{}).get("first_name"),
                 "last_name":user.get("name",{}).get("last_name"),"dob":user.get("dob")}
    if any(not isinstance(v,str) or not v.strip() for v in passenger.values()):
        raise PreparationGap("requester_passenger_identity_incomplete")
    try: date.fromisoformat(passenger["dob"])
    except ValueError: raise PreparationGap("passenger_date_invalid") from None
    methods = user.get("payment_methods",{})
    cards = sorted(k for k,v in methods.items() if v.get("source") == "credit_card" and v.get("id") == k)
    gifts = sorted(k for k,v in methods.items() if v.get("source") == "gift_card" and v.get("id") == k
                   and type(v.get("amount")) in (int,float) and v["amount"] >= 0)
    if not cards and not gifts: raise PreparationGap("no_supported_owned_payment_method")
    cap = None if cards else max(methods[k]["amount"] for k in gifts)
    route = _route(database,clock,defaults["cabin"],request["constraints"],policy["route_search"],cap)
    total = sum(r["price"] for r in route)
    payment_id = cards[0] if cards else next(k for k in gifts if methods[k]["amount"] >= total)
    arguments = {"user_id":user["user_id"],"origin":route[0]["origin"],"destination":route[-1]["destination"],
        "flight_type":defaults["flight_type"],"cabin":defaults["cabin"],
        "flights":[{"flight_number":r["flight_number"],"date":r["date"]} for r in route],
        "passengers":[passenger],"payment_methods":[{"payment_id":payment_id,"amount":total}],
        "total_baggages":0,"nonfree_baggages":0,"insurance":"no"}
    return arguments, {"route":route,"passenger_basis":"verified_requester_self_not_saved_third_party",
        "payment_basis":"verified_owned_method_reference_and_gift_balance_if_applicable",
        "payment_check":{"source":methods[payment_id]["source"],"required_amount":total,
                         "gift_balance":methods[payment_id].get("amount") if payment_id in gifts else None},
        "defaults":defaults,
        "copied_context_reservation_preferences":False}


def _initial_request(tool, args):
    if tool == "cancel_reservation":
        return "Please cancel my reservation."
    flights = ", then ".join(f"{f['flight_number']} on {f['date']}" for f in args["flights"])
    return (f"Please book a one-way {args['cabin']} trip from {args['origin']} to {args['destination']} "
            f"using {flights}. I am travelling alone, with no checked baggage and no travel insurance.")


def compile_package(contracts, plans, references, handoff, requester_bindings, semantics, data, policy):
    validate_policy(policy)
    for value,field in [(contracts,"contract_set_fingerprint"),(plans,"plan_set_fingerprint"),(references,"reference_set_fingerprint"),
                        (handoff,"handoff_fingerprint"),(requester_bindings,"requester_binding_set_fingerprint"),
                        (semantics,"review_fingerprint")]:
        verify_fingerprint(value,field)
    if semantics["source_contract_set_fingerprint"] != contracts["contract_set_fingerprint"]:
        raise ValueError("semantics source mismatch")
    if plans["source_contract_set_fingerprint"] != contracts["contract_set_fingerprint"]:
        raise ValueError("plan source mismatch")
    if (references["source_contract_set_fingerprint"]!=contracts["contract_set_fingerprint"]
            or handoff["source_reference_set_fingerprint"]!=references["reference_set_fingerprint"]
            or requester_bindings["source_handoff_fingerprint"]!=handoff["handoff_fingerprint"]
            or requester_bindings["source_contract_set_fingerprint"]!=contracts["contract_set_fingerprint"]
            or data["manifest"]["source_plan_set_fingerprint"]!=plans["plan_set_fingerprint"]):
        raise ValueError("input dependency lineage mismatch")
    if references["database_fingerprint"] != content_sha256(data["database"]): raise ValueError("database mismatch")
    refs, roles, actors, plan_rows = map(_index,(references["rows"],handoff["rows"],requester_bindings["rows"],plans["plans"]))
    bids={c["branch_id"] for c in contracts["contracts"]}
    if not (bids==refs.keys()==roles.keys()==actors.keys()==plan_rows.keys()): raise ValueError("branch membership mismatch")
    sem = {(r["branch_id"],r["kind"]):r for r in semantics["rows"]}
    if len(sem)!=len(semantics["rows"]) or any(bid not in bids for bid,kind in sem):
        raise ValueError("duplicate or foreign semantic review")
    store=SnapshotObjects(data["database"],references["database_fingerprint"])
    clock=_clock(data["clock"]["value"]); schemas={k:v["parameters"] for k,v in data["tool_snapshot"]["tools"].items()}
    rows=[]
    for contract in contracts["contracts"]:
        bid=contract["branch_id"]; role=roles[bid]
        for value,field in ((contract,"branch_test_contract_fingerprint"),(role,"role_row_fingerprint"),
                            (actors[bid],"requester_binding_fingerprint"),(plan_rows[bid],"driver_plan_fingerprint")):
            verify_fingerprint(value,field)
        if any(value["source_contract_fingerprint"]!=contract["branch_test_contract_fingerprint"]
               for value in (role,actors[bid],plan_rows[bid])):
            raise ValueError("branch dependency lineage mismatch")
        row={"branch_id":bid,"source_contract_fingerprint":contract["branch_test_contract_fingerprint"],
             "source_driver_plan_fingerprint":plan_rows[bid]["driver_plan_fingerprint"],
             "source_user_requirement_contract":deepcopy(contract["user_input_contract"]),
             "source_given_contract":deepcopy(contract["given_contract"]),
             "status":"blocked","blockers":[],"step7_prepared":False,"runtime_executed":False,
             "initial_user_message":None,"private_operation_bundle":None,"user_facts":{},"driver_bindings":{},
             "oracle_handoff":deepcopy(contract["oracle_contract"]),
             "runtime_obligations":["load_exact_database_and_reference_clock","validate_runtime_tool_schema_compatibility",
                                    "open_evidence_capture_before_first_message","realize_source_conversation_requirements",
                                    "enforce_on_request_fact_disclosure","confirm_only_same_proposed_operation",
                                    "observe_when_separately_from_oracle_outcome"]}
        try:
            request_row=sem.get((bid,"request"))
            if request_row is None or request_row["review_status"] != "accepted":
                raise PreparationGap("request_operation_not_determined")
            request=request_row["answer"]; tool=request["tool"]
            if request["unmapped_requirements"] or tool not in ("book_reservation","cancel_reservation"):
                raise PreparationGap("request_semantics_outside_supported_operations")
            if contract["user_input_contract"]["conversation_preconditions"] or contract["user_input_contract"]["when_qualifiers"]:
                raise PreparationGap("additional_conversation_constraints_require_adapter")
            if contract["user_input_contract"]["trigger_record"].get("supplies_queried") is not True:
                raise PreparationGap("upstream_does_not_authorize_supplying_requested_facts")
            given_sem=sem.get((bid,"given"))
            if given_sem and given_sem["review_status"] not in ("accepted","accepted_with_state_guard"):
                raise PreparationGap("given_semantic_review_not_accepted")
            given_answer=given_sem["answer"] if given_sem else None
            if tool=="book_reservation":
                if role["lifecycle"]["decision"] != "create_new" or check_target_origin(role,"future_result")["status"] != "compatible":
                    raise PreparationGap("operation_lifecycle_conflict")
                actor=actors[bid]
                if actor["status"] != "requester_and_context_bound": raise PreparationGap("requester_not_bound")
                checks=[]; chosen=None
                for candidate in inspect_candidates(contract["fixture_contract"],store):
                    check={"candidate_index":candidate["candidate_index"],"candidate_fingerprint":candidate["source_candidate_fingerprint"],
                           "reference_status":candidate["reference_status"],"relation_status":candidate["relation_observation"]["status"]}
                    checks.append(check)
                    if candidate["reference_status"]!="resolved" or candidate["relation_observation"]["status"]!="matched": continue
                    user=store.read(candidate["user_reference"]["handle"])
                    try:
                        given=_given_observations(contract,None,user,None,data["database"],data["clock"]["value"],given_answer)
                        args,evidence=_booking(user,data["database"],clock,request,policy)
                    except PreparationGap as exc:
                        check["preparation_gap"]=str(exc); continue
                    chosen=(candidate,user,args,evidence,given); break
                if chosen is None:
                    row["binding_evidence"]={"candidate_checks":checks}
                    raise PreparationGap("no_supplied_actor_has_complete_supported_booking_facts")
                candidate,user,args,evidence,given=chosen
                row["private_roles"]={"requester":candidate["user_reference"]["handle"],
                    "context_anchor":candidate["root_reference"]["handle"],"operation_target":None}
                row["driver_bindings"]={"user_id":user["user_id"]}
                facts={k:deepcopy(v) for k,v in args.items()}
                row["binding_evidence"]={**evidence,"candidate_checks":checks,
                    "selected_candidate_index":candidate["candidate_index"],
                    "source_identity_only_candidate_index":actor["selected_candidate_index"],
                    "selection_scope":"first_supplied_pair_with_complete_supported_request_facts"}
            else:
                if role["lifecycle"]["decision"] != "use_existing" or check_target_origin(role,"snapshot_object")["status"] != "compatible":
                    raise PreparationGap("operation_lifecycle_conflict")
                if contract["fixture_contract"]["root"]!="reservations": raise PreparationGap("unsupported_existing_object_type")
                if contract["fixture_contract"]["relation_assertions"]["user_requirements"] != {"present":True,"value":"owner"}:
                    raise PreparationGap("existing_object_owner_scope_not_supported")
                checks=[]; chosen=None
                for candidate in inspect_candidates(contract["fixture_contract"],store):
                    check={"candidate_index":candidate["candidate_index"],"candidate_fingerprint":candidate["source_candidate_fingerprint"],
                           "reference_status":candidate["reference_status"],"relation_status":candidate["relation_observation"]["status"]}
                    checks.append(check)
                    if candidate["reference_status"]!="resolved" or candidate["relation_observation"]["status"]!="matched": continue
                    user=store.read(candidate["user_reference"]["handle"]); reservation=store.read(candidate["root_reference"]["handle"])
                    consistency=_reservation_consistency(reservation,data["database"],clock,policy["existing_reservation"])
                    given=_given_observations(contract,candidate,user,reservation,data["database"],data["clock"]["value"],given_answer)
                    args={"reservation_id":reservation["reservation_id"]}
                    check.update(consistency=consistency,given=given)
                    if consistency["passed"] and given["truth"] is True and _constraint_match(args,request["constraints"]):
                        chosen=(candidate,user,args,given); break
                if chosen is None:
                    row["binding_evidence"]={"candidate_checks":checks}
                    raise PreparationGap("no_supplied_existing_object_passes_given_and_nominal_consistency")
                candidate,user,args,given=chosen
                row["private_roles"]={"requester":candidate["user_reference"]["handle"],"operation_target":candidate["root_reference"]["handle"],"context_anchor":None}
                row["driver_bindings"]={"user_id":user["user_id"],"reservation_id":args["reservation_id"]}
                facts=deepcopy(row["driver_bindings"])
                row["binding_evidence"]={"candidate_checks":checks,"selected_candidate_index":candidate["candidate_index"]}
            if given["truth"] is not True: raise PreparationGap("given_not_verified_true")
            if not _constraint_match(args,request["constraints"]): raise PreparationGap("explicit_request_constraint_not_satisfied")
            try:
                _validate_required_arguments({"tool_name":tool,"arguments":args},schemas)
            except GenericDriverBindingError as exc:
                raise PreparationGap("operation_arguments_do_not_satisfy_tool_schema") from exc
            row.update(status="prepared",step7_prepared=True,given_verification=given,
                private_operation_bundle={"tool_name":tool,"arguments":args},
                initial_user_message=_initial_request(tool,args),
                user_facts={key:{"value":value,"timing":"on_agent_request_only"} for key,value in facts.items()},
                confirmation_contract={"requires_agent_confirmation_request":True,"requires_same_operation":True,
                    "comparison":"exact_canonical_operation_bundle","allowed_response":"Yes, please proceed.",
                    "pre_recorded_confirmation_event":False},
                source_request_constraints=deepcopy(request["constraints"]),
                argument_schema_validated=True,complete_user_input_preparation=True)
        except PreparationGap as exc:
            row["blockers"].append(str(exc))
        rows.append(seal(row,"case_fingerprint"))
    return seal({"schema_version":SCHEMA,"source_contract_set_fingerprint":contracts["contract_set_fingerprint"],
        "source_plan_set_fingerprint":plans["plan_set_fingerprint"],"source_semantic_review_fingerprint":semantics["review_fingerprint"],
        "policy":deepcopy(policy),"policy_fingerprint":content_sha256(policy),
        "environment_contract":{"database_fingerprint":store.fingerprint,"reference_clock":deepcopy(data["clock"]),
            "tool_schema_fingerprint":content_sha256(data["tool_snapshot"]),"source_bundle_fingerprint":data["manifest"]["bundle_fingerprint"],
            "compatibility_limits":deepcopy(data["manifest"]["compatibility"]),"runtime_handshake_required":True,
            "database_reset_policy":"fresh_full_snapshot_before_each_case"},
        "cases":rows,"out_of_pilot_branch_ids":deepcopy(contracts["out_of_pilot_branch_ids"]),
        "source_deferred_branch_ids":deepcopy(contracts["source_deferred_branch_ids"]),
        "summary":{"branches":len(rows),"status_counts":dict(Counter(r["status"] for r in rows)),
            "prepared_cases":sum(r["step7_prepared"] for r in rows),"blocked_cases":sum(not r["step7_prepared"] for r in rows),
            "external_calls_during_binding":0,"target_agent_calls":0},
        "step8_entrypoint_implemented":False,"scope":"ordinary functional baseline preparation, not feedback search or online execution"},
        "package_fingerprint")


def user_view(case):
    verify_fingerprint(case,"case_fingerprint")
    if not case["step7_prepared"]: raise PreparationGap("case_not_prepared")
    return {"initial_user_message":case["initial_user_message"],
            "facts_on_request":{k:deepcopy(v["value"]) for k,v in case["user_facts"].items()},
            "confirmation_contract":deepcopy(case["confirmation_contract"])}


def validate_environment_identity(package, *, database_fingerprint, reference_clock, tool_schema_fingerprint):
    """Called with actual environment observations by Step 8 before every case.

    This verifies declared identities only; no environment has been loaded here.
    The runtime is responsible for observing these values, not copying expectations.
    """
    verify_fingerprint(package,"package_fingerprint")
    expected=package["environment_contract"]
    if (database_fingerprint!=expected["database_fingerprint"]
            or reference_clock!=expected["reference_clock"]["value"]
            or tool_schema_fingerprint!=expected["tool_schema_fingerprint"]):
        raise PreparationGap("runtime_environment_identity_mismatch")
    return {"status":"identity_matches","runtime_execution_performed":False,
            "source_version_compatibility_inferred":False}


def requested_facts(case, requested_fields, *, agent_requested):
    view=user_view(case)
    if agent_requested is not True: raise PreparationGap("facts_require_an_actual_agent_request")
    if not isinstance(requested_fields,list) or any(not isinstance(k,str) or k not in view["facts_on_request"] for k in requested_fields):
        raise PreparationGap("requested_field_not_granted")
    return {k:deepcopy(view["facts_on_request"][k]) for k in requested_fields}


def confirmation_response(case, proposed_bundle, *, agent_requested):
    user_view(case)
    if agent_requested is not True or content_sha256(proposed_bundle)!=content_sha256(case["private_operation_bundle"]):
        raise PreparationGap("confirmation_request_or_operation_does_not_match")
    return case["confirmation_contract"]["allowed_response"]


def render_package(result):
    lines=["# Step 7 ordinary-functionality baseline preparation results","","prepared means input preparation is complete; it does not mean the case has run, that When was observed, or that the Oracle passed.","",
           "| Branch | Status | Business operation | Flight segments | Blocking reasons |","| --- | --- | --- | ---: | --- |"]
    for c in result["cases"]:
        op=c["private_operation_bundle"] or {}
        lines.append(f"| {c['branch_id']} | {c['status']} | {op.get('tool_name','undetermined')} | {len(op.get('arguments',{}).get('flights',[]))} | {'; '.join(c['blockers']) or 'none'} |")
    return "\n".join(lines+["","All Oracle content and database context remain in the private contract; the whole package must not be sent to the Agent.",""])
