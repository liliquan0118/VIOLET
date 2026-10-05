"""Layer 1 of the generic tool-call synthesis pipeline (see
docs/oracle_requirement_pipeline_v0_7.md section 53): a domain-agnostic,
purely mechanical FACT GATHERER. It resolves real candidate object(s) for a
branch's fixture and evaluates the branch's own structured Given conditions
against them, but makes NO judgment calls -- it never decides whether an
ambiguous quantifier means "any" or "all", never decides whether a failed
baseline consistency check should be waived, never decides which candidate
is "the right one". All of that is Layer 2's job (a real Agent-tool call);
this layer only ever reports real, verifiable facts for Layer 2 to reason
over, and Layer 3 to later cross-check against.

Reuses frozen, already-proven primitives directly (SnapshotObjects,
inspect_candidates, _relation, _predicate_observation, _dated_flights,
_reservation_consistency, synthesize_entity_from_template) rather than
reimplementing their comparison/traversal logic. The one real generalization
made here: _predicate_observation (frozen) only supports a condition whose
table equals the fixture's own root table, or the reverse users-on-
reservations case -- this module instead binds a condition's table against
whichever of the two real objects (user, root_object) actually has that
table, in either direction, for any table pair, by calling
_predicate_observation through a synthetic single-object framing that
exploits its own generic table==root branch (never mutating or bypassing
its comparison logic, which is reused unchanged).
"""
from __future__ import annotations

from copy import deepcopy

from .v5_candidate_binding_v1 import IDENTITY_FIELDS, _predicate_observation
from .v5_object_references_v1 import inspect_candidates
from .v5_related_binding_v1 import _dated_flights, _supported_dated_path
from .v5_step7_intent_synthesis_v1 import IntentSynthesisError, synthesize_entity_from_template
from .v5_step7_package_v1 import PreparationGap, _reservation_consistency, _route

REAL_CABIN_CLASSES = ("basic_economy", "economy", "business")

MAX_CANDIDATES = 8  # real candidate pools run to 20; cap what Layer 2 has to read per branch
MAX_OWNED_OBJECTS = 5  # real owned-object pools (e.g. a user's reservations) can run long too
MAX_DATES_SHOWN = 6  # a real flight's "dates" dict can span 30 real days of price/seat data


def _slim_dated_object(obj):
    """A real flights-table object's "dates" dict can be tens of KB (30 real
    days of price/seat/timestamp data); the condition_observations already
    surface whichever specific dates actually matter for this branch's real
    Given, so only a small, real, deterministic sample is kept here to keep
    prompts a sane size -- never fabricated, just truncated."""
    if not (isinstance(obj, dict) and isinstance(obj.get("dates"), dict) and len(obj["dates"]) > MAX_DATES_SHOWN):
        return obj
    slimmed = deepcopy(obj)
    kept = dict(sorted(slimmed["dates"].items())[:MAX_DATES_SHOWN])
    slimmed["dates"] = kept
    slimmed["_dates_truncated_note"] = f"showing {len(kept)} of {len(obj['dates'])} real dated instances"
    return slimmed


def _evaluate_condition_against_object(condition, obj, clock):
    """Reuses _predicate_observation (frozen) unchanged. The frozen function
    only binds a condition whose table=="fixture root" (or the special-cased
    users-on-reservations reverse). Passing a synthetic branch whose "root"
    IS this condition's own table makes that check trivially true, so the
    function binds `obj` as objects["root"] regardless of what obj really
    represents -- this generalizes to ANY (condition table, object) pair
    without touching or duplicating the frozen comparison logic."""
    fake_branch = {"fixture_requirements": {"root": condition["source_condition"]["table"]}}
    return _predicate_observation(condition, fake_branch, {"root": obj, "user": None}, clock)


MAX_INSTANCE_WITNESSES = 4  # a flights.dates{} collection condition can real-walk 30 real dated instances


def _slim_instance_observation(observation):
    """A real flights.dates{} collection observation (_dated_flights,
    frozen) carries the FULL real per-date instance list (up to 30 real
    entries with real price/seat/timestamp records each) -- what Layer 2
    actually needs to resolve an ambiguous quantifier is a real summary
    (total count, how many match) plus a small, real sample, not the whole
    catalog. Never fabricates a count or a sample; only truncates real data
    already computed by _dated_flights."""
    if "instances" not in observation:
        return observation
    slimmed = {k: v for k, v in observation.items() if k != "instances"}
    total = len(observation["instances"])
    matching = observation.get("matching_witnesses", [])
    slimmed["matching_witnesses"] = matching[:MAX_INSTANCE_WITNESSES]
    slimmed["real_instance_summary"] = {"total_real_dated_instances": total, "matching_count": len(matching),
        "matching_witnesses_truncated": len(matching) > MAX_INSTANCE_WITNESSES}
    return slimmed


def evaluate_condition(condition, user, root_object, root_table, database, clock):
    """Evaluates one structured Given condition against the real bound
    objects available for this branch. Returns the raw observation
    (truth is None and a real reason when genuinely unresolvable -- never
    guessed); Layer 2 reads this, including any raw per-instance witnesses
    for quantifier-ambiguous collection conditions, and makes the actual
    true/false call itself.

    A condition whose table=="flights" and whose path is the real dated-
    collection shape (flights.dates{}...) is a LINKED condition when
    root_table=="reservations" -- it doesn't match the root object's own
    table directly, it's reached by walking the reservation's real booked
    segments (this is the one real indirection _dated_flights/
    _flight_instances, frozen, already implements; reused unchanged)."""
    table = condition["source_condition"]["table"]
    is_dated = _supported_dated_path(condition)
    if table == "flights" and is_dated and root_table in ("flights", "reservations") and root_object is not None:
        linked = _dated_flights({"private_fixture_context": {"root": root_object, "root_table": root_table}}, [condition], database, clock)
        return _slim_instance_observation(linked["condition_observations"][condition["condition_id"]])
    if table == "users":
        obj = user
    elif root_object is not None and table == root_table:
        obj = root_object
    else:
        return {"condition_id": condition["condition_id"], "source_condition": deepcopy(condition["source_condition"]),
                "truth": None, "status": "unbound_condition_subject", "basis": "no_real_bound_object_for_this_table"}
    return _evaluate_condition_against_object(condition, obj, clock)


def _real_consistency_facts(root_object, root_table, database, clock):
    if root_table != "reservations" or root_object is None:
        return None
    policy = {"require_active": True, "require_future_unflown_segments": True, "require_creation_not_future": True}
    return _reservation_consistency(root_object, database, clock, policy)


def _owned_objects(database, table, user_id):
    """Real objects of `table` whose own user_id field matches this user --
    generic version of the "does this user's OWN reservation satisfy these
    conditions" lookup (airline_096_state#b0-style: root=='users' but the
    Given also references reservations-table fields, so there is no single
    bound root_object to check them against; the real candidate set is
    whichever objects this user actually owns)."""
    return [deepcopy(obj) for obj in database.get(table, {}).values() if obj.get("user_id") == user_id]


def _any_real_object_of_table(database, table):
    """Generalized version of any_real_owned_reservation (airline_intent_
    adapter_v1.py): picks the first real object of `table` (sorted by key)
    together with its real owning user, if the table has an owner field
    matching a real users record. Returns (owner_user_or_None, template) or
    (None, None) if the table is empty or has no real records at all."""
    for key, obj in sorted(database.get(table, {}).items()):
        owner_id = obj.get("user_id")
        owner = database.get("users", {}).get(owner_id) if owner_id else None
        return owner, deepcopy(obj)
    return None, None


def _real_route_reference(database, clock, route_policy, route_constraints=()):
    """A real, priced, schedule-valid route for each real cabin class,
    reusing _route (frozen, the same search compile_package's own _booking
    already used) -- its own deterministic date/flight-number search order
    picks one canonical real option per cabin. Exposed so Layer 2 can build
    a NEW booking's flights/price without inventing flight numbers or
    amounts from nothing; not a requirement to use these specific flights,
    just real material to draw on. None for a cabin with genuinely no valid
    route in the configured horizon (honest, not an error).

    route_constraints: the branch's own real classified request constraints
    (e.g. flights min_items>1 for "multiple flights") -- _route already
    natively supports these (min/max legs, fixed origin/destination), so
    they're passed straight through rather than only ever searching for the
    trivial single-leg default; a branch with no such constraint gets the
    same single-leg-search behavior as before."""
    reference = {}
    for cabin in REAL_CABIN_CLASSES:
        try:
            route = _route(database, clock, cabin, list(route_constraints), route_policy)
        except PreparationGap:
            route = None
        reference[cabin] = route
    return reference


def resolve_fixture_context(contract, store, database, clock, route_policy=None, route_constraints=()):
    """Gathers, without judging, everything Layer 2 needs for one branch:
    up to MAX_CANDIDATES real supplied (user, root_object) pairs, each with
    every Given database_condition evaluated against it and the real
    baseline consistency facts (when root_table=="reservations"); plus a
    real template-based synthesis fallback candidate when the fixture's own
    supplied candidate pool is genuinely empty and the Given's own database
    conditions are all plain eq (mechanically synthesizable, same rule
    synthesize_entity_from_template already enforces); plus, when
    route_policy is supplied, a real priced route reference per cabin class
    (see _real_route_reference) for branches that may need to construct a
    brand-new booking."""
    fixture = contract["fixture_contract"]
    given = contract["given_contract"]
    root_table = fixture["root"]
    conditions = given["database_conditions"]

    other_tables = sorted({c["source_condition"]["table"] for c in conditions
                            if c["source_condition"]["table"] not in ("users", root_table, "flights")})

    candidates = []
    for candidate in inspect_candidates(fixture, store)[:MAX_CANDIDATES]:
        if candidate["reference_status"] != "resolved":
            continue
        user = store.read(candidate["user_reference"]["handle"])
        root_object = store.read(candidate["root_reference"]["handle"]) if candidate["root_reference"]["handle"] else None
        observations = [evaluate_condition(c, user, root_object, root_table, database, clock) for c in conditions]
        related = {}
        for table in other_tables:
            owned = _owned_objects(database, table, user.get("user_id"))[:MAX_OWNED_OBJECTS]
            table_conditions = [c for c in conditions if c["source_condition"]["table"] == table]
            related[table] = [{"object": obj, "condition_observations":
                [evaluate_condition(c, user, obj, table, database, clock) for c in table_conditions]}
                for obj in owned]
        candidates.append({"candidate_index": candidate["candidate_index"], "source": "real_supplied_candidate",
            "user": deepcopy(user), "root_object": _slim_dated_object(root_object) if root_object is not None else None,
            "relation_observation": deepcopy(candidate["relation_observation"]),
            "condition_observations": observations, "related_owned_objects": related,
            "consistency_facts": _real_consistency_facts(root_object, root_table, database, clock)})

    synthesized = None
    if not candidates and root_table in IDENTITY_FIELDS:
        eq_conditions = [c["source_condition"] for c in conditions
                         if c["source_condition"]["table"] == root_table and c["source_condition"]["op"] == "eq"]
        non_eq = [c for c in conditions if c["source_condition"]["table"] == root_table and c["source_condition"]["op"] != "eq"]
        template_user, template_object = _any_real_object_of_table(database, root_table)
        if template_object is not None and not non_eq:
            try:
                synthesized_object, applied_fields = synthesize_entity_from_template(template_object, eq_conditions)
                observations = [evaluate_condition(c, template_user or {}, synthesized_object, root_table, database, clock)
                                 for c in conditions]
                synthesized = {"source": "synthesized_from_real_template", "user": deepcopy(template_user),
                    "root_object": _slim_dated_object(synthesized_object), "applied_fields": applied_fields,
                    "condition_observations": observations,
                    "consistency_facts": _real_consistency_facts(synthesized_object, root_table, database, clock)}
            except IntentSynthesisError:
                synthesized = None

    route_reference = _real_route_reference(database, clock, route_policy, route_constraints) if route_policy is not None else None

    return {"branch_id": contract["branch_id"], "fixture_root": root_table,
        "fixture_lookup_status": fixture["lookup"]["lookup_status"], "fixture_n_matches": fixture["lookup"]["n_matches"],
        "relation_assertions": deepcopy(fixture["relation_assertions"]),
        "given_text": given["text"], "given_condition_count": len(conditions),
        "non_database_conditions": [deepcopy(c["source_condition"]) for c in given["non_database_conditions"]],
        "candidates": candidates, "synthesized_candidate": synthesized, "real_route_reference": route_reference}
