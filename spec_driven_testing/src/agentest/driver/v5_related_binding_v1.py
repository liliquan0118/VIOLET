"""Ground linked records and dated flight instances, without text re-review."""

from collections import Counter
from copy import deepcopy
from datetime import date

from ..compiler.artifacts import content_sha256
from ..compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .v5_candidate_binding_v1 import bind_candidates, seal, _object, _predicate_observation


SCHEMA = "agentspectesting.v5-related-binding/v0.1"


def validate_parent(parent, assembly, database):
    verify_fingerprint(parent, "binding_set_fingerprint")
    if parent["database_fingerprint"] != content_sha256(database):
        raise ValueError("database differs from bound snapshot")
    expected = bind_candidates(assembly, database, parent["reference_time"])
    excluded = {"binding_set_fingerprint", "source_files", "database_loader"}
    strip = lambda d: {k: v for k, v in d.items() if k not in excluded}
    if strip(expected) != strip(parent):
        raise ValueError("parent differs from candidate-binding reconstruction")


def _on_object(condition, table, obj, clock, *, instance=False):
    scoped = deepcopy(condition)
    if instance:
        steps = scoped["optional_encoding"]["expression"]["subject"]["path_steps"]
        scoped["optional_encoding"]["expression"]["subject"]["path_steps"] = steps[1:]
    observation = _predicate_observation(scoped, {"fixture_requirements": {"root": table}}, {"root": obj}, clock)
    observation["subject_role"] = "linked_reservation" if table == "reservations" else "dated_flight_instance"
    return observation


def _joint_match(observations):
    # Selection of a sufficient common witness is not an inferred Given AST.
    return bool(observations) and all(o["truth"] is True for o in observations)


def _related_orders(branch, conditions, database, clock):
    user = branch["private_fixture_context"]["root"]
    links = user.get("reservations")
    result = {"status": "no_joint_witness", "selection_policy": "first_linked_object_satisfying_all_supplied_filters",
              "source_edge": "users.reservations -> reservations.reservation_id", "checks": [], "selected": None,
              "given_logic_inferred": False}
    if not isinstance(links, list):
        return {**result, "status": "missing_relation_list"}
    seen = set()
    for index, rid in enumerate(links):
        if not isinstance(rid, str) or not rid.strip():
            result["checks"].append({"link_index": index, "error": "invalid_reservation_reference"})
            continue
        if rid in seen:
            continue
        seen.add(rid)
        obj, error = _object(database, "reservations", rid)
        if error or obj.get("user_id") != user["user_id"]:
            result["checks"].append({"link_index": index, "reservation_id": rid, "error": error or "linked_owner_mismatch"})
            continue
        observations = [_on_object(c, "reservations", obj, clock) for c in conditions]
        result["checks"].append({"link_index": index, "reservation_id": rid, "observations": observations})
        if _joint_match(observations):
            result.update(status="bound_joint_witness", selected={"reservation_id": rid, "link_index": index,
                          "user_id": obj["user_id"], "record": deepcopy(obj), "observations": observations})
            break
    return result


def _valid_date(value):
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _flight_instances(branch, database):
    root = branch["private_fixture_context"]["root"]
    kind = branch["private_fixture_context"]["root_table"]
    rows, issues = [], []
    if kind == "reservations":
        segments = root.get("flights")
        if not isinstance(segments, list) or not segments:
            return [], [{"error": "empty_or_missing_reservation_segments"}]
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict) or not isinstance(segment.get("flight_number"), str) or not _valid_date(segment.get("date")):
                issues.append({"segment_index": index, "error": "invalid_segment_reference"})
                continue
            number, day = segment["flight_number"], segment["date"]
            flight, error = _object(database, "flights", number)
            dates = flight.get("dates") if flight is not None else None
            instance = dates.get(day) if isinstance(dates, dict) else None
            if error or not isinstance(instance, dict):
                issues.append({"segment_index": index, "flight_number": number, "date": day, "error": error or "flight_date_not_found"})
                continue
            rows.append({"segment_index": index, "flight_number": number, "date": day, "record": deepcopy(instance)})
    elif kind == "flights":
        dates = root.get("dates")
        if not isinstance(dates, dict) or not dates:
            return [], [{"error": "empty_or_missing_flight_dates"}]
        for day, instance in sorted(dates.items(), key=lambda x: str(x[0])):
            if not _valid_date(day) or not isinstance(instance, dict):
                issues.append({"error": "invalid_flight_date_record"})
                continue
            rows.append({"flight_number": root["flight_number"], "date": day, "record": deepcopy(instance)})
    return rows, issues


def _supported_dated_path(condition):
    enc = condition["optional_encoding"]
    if enc["status"] != "available":
        return False
    steps = enc["expression"]["subject"]["path_steps"]
    return (len(steps) >= 2 and steps[0] == {"field": "dates", "traversal": "object_values"}
            and all(step["traversal"] == "field" for step in steps[1:]))


def _dated_flights(branch, conditions, database, clock):
    instances, issues = _flight_instances(branch, database)
    root_kind = branch["private_fixture_context"]["root_table"]
    observations = {}
    for instance in instances:
        instance["observations"] = [_on_object(c, "flights", instance["record"], clock, instance=True) for c in conditions]
    selection = None
    if root_kind == "flights":
        selection = next((deepcopy(i) for i in instances if _joint_match(i["observations"])), None)
    for position, condition in enumerate(conditions):
        raw = condition["source_condition"]
        atom_results = [{"instance_ref": {k: v for k, v in inst.items() if k not in ("record", "observations")},
                         "observation": inst["observations"][position]} for inst in instances]
        witnesses = [a for a in atom_results if a["observation"]["truth"] is True]
        quant = raw.get("quant")
        observation = {"condition_id": condition["condition_id"], "source_condition": deepcopy(raw),
                       "status": "not_evaluated", "truth": None,
                       "basis": "quantifier_not_supplied_instance_evidence_only", "instances": atom_results,
                       "matching_witnesses": witnesses, "collection_issues": deepcopy(issues),
                       "scope": "reservation_booked_segments" if root_kind == "reservations" else "bound_root_flight_dates",
                       "quantifier_supplied": "quant" in raw}
        # Explicit aggregation is over the linked scope, never every date of
        # each flight number. Empty or incomplete scopes do not pass vacuously.
        if quant in ("all", "any"):
            truths = [a["observation"]["truth"] for a in atom_results]
            if quant == "all" and False in truths:
                truth = False
            elif quant == "any" and True in truths:
                truth = True
            elif truths and not issues and all(t is not None for t in truths):
                truth = all(truths) if quant == "all" else any(truths)
            else:
                truth = None
            observation.update(truth=truth, status="evaluated" if truth is not None else "not_evaluated",
                               basis="explicit_quantifier_over_linked_scope" if truth is not None else "incomplete_or_empty_linked_scope")
        elif quant is not None:
            observation["basis"] = "unsupported_quantifier"
        elif root_kind == "flights" and selection is not None:
            observation.update(truth=selection["observations"][position]["truth"], status="evaluated",
                               basis="comparison_on_selected_flight_date", scope="selected_single_flight_date",
                               selected_instance_ref={"flight_number": selection["flight_number"], "date": selection["date"]})
        elif witnesses:
            observation["status"] = "instance_witness_available"
        observations[condition["condition_id"]] = observation
    return {"source_edge": "reservations.flights[flight_number,date] -> flights.dates[date]" if root_kind == "reservations" else "flights.dates[date]",
            "instances": instances, "issues": issues, "selected_date_witness": selection,
            "selection_policy": "first_date_satisfying_all_supplied_filters" if root_kind == "flights" else "preserve_all_booked_segment_dates",
            "condition_observations": observations}


def enrich_related_bindings(parent, assembly, database):
    validate_parent(parent, assembly, database)
    source = {b["branch_id"]: b for b in assembly["branches"]}
    rows = []
    for original in parent["branches"]:
        b = deepcopy(original)
        b["source_initial_binding_fingerprint"] = b.pop("binding_fingerprint")
        if b["object_binding_status"] == "objects_bound":
            conditions = source[b["branch_id"]]["given_requirements"]["database_conditions"]
            root_kind = b["private_fixture_context"]["root_table"]
            replacements = {}
            if root_kind == "users":
                related = [c for c in conditions if c["source_condition"]["table"] == "reservations"]
                if related:
                    binding = _related_orders(b, related, database, parent["reference_time"])
                    b["linked_reservation_binding"] = binding
                    if binding["selected"] is not None:
                        for o in binding["selected"]["observations"]:
                            o = deepcopy(o)
                            o.update(basis="selected_related_reservation_comparison", scope="selected_related_reservation",
                                     reservation_id=binding["selected"]["reservation_id"])
                            replacements[o["condition_id"]] = o
            if root_kind in ("reservations", "flights"):
                dated = [c for c in conditions if c["source_condition"]["table"] == "flights" and _supported_dated_path(c)]
                if dated:
                    b["flight_instance_binding"] = _dated_flights(b, dated, database, parent["reference_time"])
                    replacements.update(b["flight_instance_binding"]["condition_observations"])
            b["database_condition_observations"] = [replacements.get(o["condition_id"], o) for o in b["database_condition_observations"]]
        rows.append(seal(b, "binding_fingerprint"))
    obs = [o for b in rows for o in b.get("database_condition_observations", [])]
    return seal({"schema_version": SCHEMA, "source_initial_binding_set_fingerprint": parent["binding_set_fingerprint"],
                 "source_assembly_set_fingerprint": assembly["assembly_set_fingerprint"],
                 "database_fingerprint": parent["database_fingerprint"], "reference_time": parent["reference_time"],
                 "scope": "related-record and dated-instance grounding; instance witnesses are not implicit aggregate quantifiers",
                 "branches": rows, "deferred_branch_ids": deepcopy(parent["deferred_branch_ids"]),
                 "summary": {"branch_count": len(rows),
                             "object_binding_status_counts": dict(Counter(b["object_binding_status"] for b in rows)),
                             "linked_reservation_binding_counts": dict(Counter(b["linked_reservation_binding"]["status"] for b in rows if "linked_reservation_binding" in b)),
                             "dated_flight_branch_count": sum("flight_instance_binding" in b for b in rows),
                             "selected_flight_date_count": sum(b.get("flight_instance_binding", {}).get("selected_date_witness") is not None for b in rows),
                             "database_condition_truth_counts": dict(Counter("true" if o["truth"] is True else "false" if o["truth"] is False else "not_aggregated_or_evaluated" for o in obs)),
                             "database_condition_status_counts": dict(Counter(o["status"] for o in obs)),
                             "pending_reasons": dict(Counter(o["basis"] for o in obs if o["truth"] is None)),
                             "user_messages_generated": 0, "database_writes": 0, "external_llm_calls": 0,
                             "target_agent_calls": 0, "test_ready_branch_count": 0}}, "binding_set_fingerprint")
