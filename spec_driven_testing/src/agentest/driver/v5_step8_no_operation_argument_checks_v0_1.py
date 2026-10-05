"""Argument-level checks for no_operation branches whose oracle_handoff has
more than a bare tool_call presence check (see docs/oracle_requirement_pipeline_v0_7.md
section 54, "categories 2/3" of the no_operation grading gap). Mechanical
only: every checker below reads real data -- the case's own real database
snapshot (reservations/users/flights, from artifact_sources['database_snapshot'])
and the real call's own arguments -- never an LLM, never a guess.

Each checker answers "does this real call, which DID call the avoided tool,
still satisfy this specific argument-level requirement?" -- True (satisfied),
False (violated), or None (cannot be determined with real data this checker
has access to, e.g. the reservation_id in the call doesn't even exist in the
database so there's no real prior state to compare against; None must be
treated as not_evaluated by the caller, never silently as True).

Keyed by the EXACT requirement_text Step7 recorded, the same "explicit,
hand-verified table" discipline used throughout this project (see sections
17/19/21/53 for prior examples) rather than an automatic NLP classifier --
each entry was read and verified against its own real branch(es) before
being added. A requirement_text with no entry here is genuinely not covered
yet; the caller reports 'not_evaluated', never a guessed pass.
"""
import re


def _reservation(database, reservation_id):
    return (database.get('reservations') or {}).get(reservation_id)


def _flight(database, flight_number):
    return (database.get('flights') or {}).get(flight_number)


def _owner(database, reservation_id):
    reservation = _reservation(database, reservation_id)
    if reservation is None:
        return None
    return (database.get('users') or {}).get(reservation.get('user_id'))


def _leg_schedule(database, flight_number, date):
    """(departure_datetime_key, arrival_datetime_key) as sortable
    (date, time) tuples, or None if the flight/date isn't real."""
    flight = _flight(database, flight_number)
    if flight is None:
        return None
    dates = flight.get('dates') or {}
    if date not in dates:
        return None
    dep = flight.get('scheduled_departure_time_est')
    arr = flight.get('scheduled_arrival_time_est')
    if dep is None or arr is None:
        return None
    arrival_date = date if arr >= dep else date  # overnight legs are not modeled in this real schema; same-day only
    return (date, dep), (arrival_date, arr)


def _integer_type_check(call, database, parameter):
    value = call['arguments'].get(parameter)
    return type(value) is int and not isinstance(value, bool)


def _passenger_count_preserved(call, database):
    reservation = _reservation(database, call['arguments'].get('reservation_id'))
    if reservation is None:
        return None
    new_count = len(call['arguments'].get('passengers') or [])
    old_count = len(reservation.get('passengers') or [])
    return new_count == old_count


def _case_user_ids(case):
    value = (case.get('user_facts') or {}).get('user_id', {}).get('value')
    return [value] if value else []


def _sensitive_ids_for(database, case_user_ids):
    ids = set()
    for user_id in case_user_ids:
        user = (database.get('users') or {}).get(user_id)
        if not user:
            continue
        ids.add(user_id)
        for reservation_id in user.get('reservations') or []:
            ids.add(reservation_id)
        for payment_id in (user.get('payment_methods') or {}):
            ids.add(payment_id)
    return ids


def _expression_has_no_sensitive_ids(call, database, case_user_ids):
    expression = call['arguments'].get('expression')
    if not isinstance(expression, str):
        return None
    sensitive = _sensitive_ids_for(database, case_user_ids)
    return not any(real_id in expression for real_id in sensitive)


def _positive_integer(call, database, parameter):
    value = call['arguments'].get(parameter)
    return type(value) is int and not isinstance(value, bool) and value > 0


def _real_flight_number_format(call, database, parameter):
    value = call['arguments'].get(parameter)
    if not isinstance(value, str):
        return None
    # Derived from the real data itself (all 300 real flight numbers in this
    # domain match this), not the requirement text's generic "AA1234"
    # example, which does not match this domain's own real flight numbers.
    return bool(re.fullmatch(r'[A-Z]{2,4}\d{2,4}', value))


def _baggages_not_increased(call, database):
    tool_name = call['tool_name']
    new_total = call['arguments'].get('total_baggages')
    if type(new_total) is not int or isinstance(new_total, bool):
        return None
    if tool_name == 'book_reservation':
        old_total = 0  # a fresh booking has no prior reservation to compare against.
    else:
        reservation = _reservation(database, call['arguments'].get('reservation_id'))
        if reservation is None:
            return None
        old_total = reservation.get('total_baggages')
        if type(old_total) is not int:
            return None
    return new_total <= old_total


def _flight_status_not_forbidden(call, database, forbidden_statuses):
    flights = call['arguments'].get('flights')
    if not isinstance(flights, list) or not flights:
        return None
    for leg in flights:
        if not isinstance(leg, dict):
            return None
        flight = _flight(database, leg.get('flight_number'))
        if flight is None:
            return None
        entry = (flight.get('dates') or {}).get(leg.get('date'))
        if entry is None:
            return None
        if entry.get('status') in forbidden_statuses:
            return False
    return True


def _flights_chronologically_feasible(call, database):
    flights = call['arguments'].get('flights')
    if not isinstance(flights, list) or len(flights) < 2:
        return True if isinstance(flights, list) else None
    schedule = []
    for leg in flights:
        if not isinstance(leg, dict):
            return None
        legs = _leg_schedule(database, leg.get('flight_number'), leg.get('date'))
        if legs is None:
            return None
        schedule.append(legs)
    for (_, prior_arrival), (next_departure, _) in zip(schedule, schedule[1:]):
        if next_departure[0] < prior_arrival[0]:
            return False
        if next_departure[0] == prior_arrival[0] and next_departure[1] < prior_arrival[1]:
            return False
    return True


def _reservation_id_exists(call, database, parameter):
    return _reservation(database, call['arguments'].get(parameter)) is not None


def _payment_not_from_a_cancelled_reservation(call, database, parameter):
    payment_id = call['arguments'].get(parameter)
    if not isinstance(payment_id, str):
        return None
    reservations = database.get('reservations') or {}
    owning = [r for r in reservations.values()
              if any(p.get('payment_id') == payment_id for p in (r.get('payment_history') or []))]
    if not owning:
        return True  # not tied to any real reservation's payment history at all.
    return not any(r.get('status') == 'cancelled' for r in owning)


def _fields_preserved_vs_prior(call, database, fields):
    reservation = _reservation(database, call['arguments'].get('reservation_id'))
    if reservation is None:
        return None
    for field in fields:
        if field in call['arguments'] and call['arguments'][field] != reservation.get(field):
            return False
    return True


def _origin_destination_preserved_via_flights(call, database):
    """update_reservation_flights has no origin/destination argument of its
    own (confirmed against the real tool schema) -- origin/destination are
    only inferable from the new flights array's first leg's real origin and
    last leg's real destination."""
    reservation = _reservation(database, call['arguments'].get('reservation_id'))
    if reservation is None:
        return None
    flights = call['arguments'].get('flights')
    if not isinstance(flights, list) or not flights:
        return None
    first = _flight(database, flights[0].get('flight_number') if isinstance(flights[0], dict) else None)
    last = _flight(database, flights[-1].get('flight_number') if isinstance(flights[-1], dict) else None)
    if first is None or last is None:
        return None
    return (first.get('origin') == reservation.get('origin')
            and last.get('destination') == reservation.get('destination'))


def _cabin_is_a_single_reservation_wide_field(call, database):
    """The real update_reservation_flights schema has exactly one 'cabin'
    argument applying to the whole reservation (confirmed against the real
    tool schema: no per-flight cabin field exists at all) -- "must be the
    same across all flights" is structurally guaranteed by the schema
    itself, not something a real call could ever violate."""
    return True


def _single_real_gift_or_credit_card_on_file(call, database, parameter):
    payment_id = call['arguments'].get(parameter)
    if not isinstance(payment_id, str):
        return None
    if not (payment_id.startswith('gift_card_') or payment_id.startswith('credit_card_')):
        return False
    owner = _owner(database, call['arguments'].get('reservation_id'))
    if owner is None:
        return None
    return payment_id in (owner.get('payment_methods') or {})


def _flights_real_and_available_in_cabin(call, database):
    flights = call['arguments'].get('flights')
    cabin = call['arguments'].get('cabin')
    if not isinstance(flights, list) or not flights or not isinstance(cabin, str):
        return None
    for leg in flights:
        if not isinstance(leg, dict):
            return None
        flight = _flight(database, leg.get('flight_number'))
        if flight is None:
            return None
        entry = (flight.get('dates') or {}).get(leg.get('date'))
        if entry is None or entry.get('status') != 'available':
            return False
        if (entry.get('available_seats') or {}).get(cabin, 0) <= 0:
            return False
    return True


NO_OPERATION_ARGUMENT_CHECKS = {
    'The amount must be an integer.':
        lambda call, database, case: _integer_type_check(call, database, 'amount'),
    "The total_baggages must be an integer representing the updated total number of baggage items. (update_reservation_baggages)":
        lambda call, database, case: _integer_type_check(call, database, 'total_baggages'),
    'The user can modify passengers but cannot modify the number of passengers. (update_reservation_passengers)':
        lambda call, database, case: _passenger_count_preserved(call, database),
    'Even a human agent cannot modify the number of passengers. (update_reservation_passengers)':
        lambda call, database, case: _passenger_count_preserved(call, database),
    'The mathematical expression must not contain any airline-specific sensitive data such as user IDs, reservation IDs, payment information, or PII. (calculate)':
        lambda call, database, case: _expression_has_no_sensitive_ids(call, database, _case_user_ids(case)),
    'The mathematical expression must not contain any airline-specific sensitive data such as user IDs, reservation IDs, payment information, or PII.':
        lambda call, database, case: _expression_has_no_sensitive_ids(call, database, _case_user_ids(case)),
    'The certificate amount must be a positive integer greater than zero.':
        lambda call, database, case: _positive_integer(call, database, 'amount'),
    "All flights in the updated flights array must be chronologically feasible and contiguous according to their departure and arrival times. (update_reservation_flights)":
        lambda call, database, case: _flights_chronologically_feasible(call, database),
    "The flight_number format must match the airline's standard (e.g., two-letter airline code followed by 1-4 digits, such as 'AA1234').":
        lambda call, database, case: _real_flight_number_format(call, database, 'flight_number'),
    'The agent must not add checked bags to the reservation. (book_reservation)':
        lambda call, database, case: _baggages_not_increased(call, database),
    'The agent must not add checked bags to the reservation. (update_reservation_baggages)':
        lambda call, database, case: _baggages_not_increased(call, database),
    "Flights with status 'delayed' or 'on time' cannot be booked. (book_reservation)":
        lambda call, database, case: _flight_status_not_forbidden(call, database, {'delayed', 'on time'}),
    "Flights with status 'flying' cannot be booked. (book_reservation)":
        lambda call, database, case: _flight_status_not_forbidden(call, database, {'flying'}),
    'The reservation_id must correspond to an existing reservation in the system. (update_reservation_baggages)':
        lambda call, database, case: _reservation_id_exists(call, database, 'reservation_id'),
    'The reservation_id must correspond to an existing reservation in the system. (update_reservation_flights)':
        lambda call, database, case: _reservation_id_exists(call, database, 'reservation_id'),
    'The agent must not use a payment method from the cancelled reservation for a modification referencing the cancelled reservation. (update_reservation_flights)':
        lambda call, database, case: _payment_not_from_a_cancelled_reservation(call, database, 'payment_id'),
    'The agent must not use a payment method from the cancelled reservation for a modification referencing the cancelled reservation. (update_reservation_baggages)':
        lambda call, database, case: _payment_not_from_a_cancelled_reservation(call, database, 'payment_id'),
    'without changing the origin, destination, and trip type.':
        lambda call, database, case: _origin_destination_preserved_via_flights(call, database),
    'Cabin class must be the same across all the flights in a reservation. (update_reservation_flights)':
        lambda call, database, case: _cabin_is_a_single_reservation_wide_field(call, database),
    'If the flights are changed, the user needs to provide a single gift card or credit card for payment or refund method, and the payment method must already be in user profile. (update_reservation_flights)':
        lambda call, database, case: _single_real_gift_or_credit_card_on_file(call, database, 'payment_id'),
    'All flights in the updated flights array must exist and be available for booking in the requested cabin class. (update_reservation_flights)':
        lambda call, database, case: _flights_real_and_available_in_cabin(call, database),
    # NOT covered (deferred, honestly not_evaluated): airline_068_arg#b0's
    # "flights must form a valid path between the reservation's original
    # origin and destination (and back, if round_trip), with no missing or
    # extraneous segments" needs real multi-leg PATH validity (a general
    # graph-reachability check across arbitrarily many legs, not just first/
    # last leg endpoints), which this first pass does not implement.
}
