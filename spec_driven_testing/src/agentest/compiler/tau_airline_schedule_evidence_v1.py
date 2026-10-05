"""Benchmark data-format adapter for a scoped record-relation judge.

Tool names/field conventions belong to a benchmark adapter, never a Spec-ID
dispatch. This decodes evidence only; it does not decide chronological policy.
"""
from copy import deepcopy
from datetime import date, datetime, time, timedelta
import re

from .artifacts import content_sha256
from .step4_evidence_extensions_v1 import join_record_evidence, sealed


def normalize_schedule(day, clock):
    if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("explicit ISO flight date required")
    parsed_day = date.fromisoformat(day)
    match = re.fullmatch(r"(\d{2}:\d{2}:\d{2})(\+1)?", clock) if isinstance(clock, str) else None
    if not match:
        raise ValueError("unsupported schedule format")
    value = datetime.combine(parsed_day, time.fromisoformat(match[1]))
    if match[2]:
        value += timedelta(days=1)
    # Deliberately no airport timezone/DST inference: compare on schema's common
    # EST clock basis, not a claim about actual local civil times or UTC.
    return value.isoformat()


def prepare_schedule_facts(subjects, observations, *, scope_id, target_event_ref, source_basis_fingerprint):
    """Decode explicitly scoped tool-result records and preserve their provenance.

    Each observation references one tool call/result and provides its decoded
    return value. This function does not fetch tools or select a trajectory.
    """
    if not all(isinstance(x, str) and x for x in (scope_id, target_event_ref, source_basis_fingerprint)):
        raise ValueError("explicit scope and reviewed source basis required")
    if not isinstance(subjects, list) or any(not isinstance(s, dict) for s in subjects):
        raise ValueError("subjects must be record objects")
    facts, problems = [], []
    for obs in observations:
        if not isinstance(obs, dict) or obs.get("scope_id") != scope_id or not obs.get("call_ref") or not obs.get("result_ref"):
            problems.append("missing_observation_scope_or_provenance")
            continue
        tool, returned = obs.get("tool_name"), obs.get("result")
        if tool not in {"search_direct_flight", "search_onestop_flight"} or not isinstance(returned, list):
            problems.append("unsupported_source_shape")
            continue
        for index, group in enumerate(returned):
            values = [group] if tool == "search_direct_flight" else group
            if not isinstance(values, list) or (tool == "search_onestop_flight" and len(values) != 2):
                problems.append("malformed_source_record_group")
                continue
            for position, value in enumerate(values):
                if not isinstance(value, dict):
                    problems.append("malformed_source_record")
                    continue
                value = deepcopy(value)
                path = [index] if tool == "search_direct_flight" else [index, position]
                date_basis = "returned_record.date"
                if tool == "search_direct_flight":
                    arguments = obs.get("arguments")
                    call_date = arguments.get("date") if isinstance(arguments, dict) else None
                    if value.get("date") is not None and value["date"] != call_date:
                        problems.append("conflicting_call_and_record_date")
                        continue
                    value["date"] = call_date
                    date_basis = "linked_tool_call.arguments.date"
                try:
                    departure = normalize_schedule(value.get("date"), value.get("scheduled_departure_time_est"))
                    arrival = normalize_schedule(value.get("date"), value.get("scheduled_arrival_time_est"))
                    if not all(isinstance(value.get(k), str) and value[k] for k in ("flight_number", "origin", "destination")):
                        raise ValueError("missing identity or route fields")
                except (TypeError, ValueError, OverflowError) as exc:
                    problems.append(str(exc))
                    continue
                facts.append({"value": {**value, "departure": departure, "arrival": arrival},
                              "source_ref": {"call_ref": obs["call_ref"], "result_ref": obs["result_ref"],
                                             "record_path": path, "date_basis": date_basis}})
    joined = join_record_evidence(subjects, facts, [("flight_number", "flight_number"), ("date", "date")])
    records = []
    for row in joined["rows"]:
        if row["status"] != "joined":
            problems.append(row["reason"])
            continue
        fact = row["fact"]
        records.append({"ref": "schedule:" + str(row["subject_index"]),
                        "subject_index": row["subject_index"],
                        "flight_number": fact["value"]["flight_number"], "date": fact["value"]["date"],
                        "origin": fact["value"]["origin"], "destination": fact["value"]["destination"],
                        "departure": fact["value"]["departure"], "arrival": fact["value"]["arrival"],
                        "source": fact["source_ref"]})
    if not records:
        problems.append("no_scoped_records")
    return sealed({"status": "complete" if not problems else "insufficient", "scope_id": scope_id,
                   "subjects_fingerprint": content_sha256(subjects),
                   "target_event_ref": target_event_ref, "records": records, "diagnostics": sorted(set(problems)),
                   "source_basis_fingerprint": source_basis_fingerprint,
                   "time_convention": "All timestamps share the tool schema EST clock basis. An explicit +1 means the next calendar day; absent +1 uses the flight date. No implicit overnight rollover, airport-zone or DST conversion.",
                   "predicate_evaluated": False})
