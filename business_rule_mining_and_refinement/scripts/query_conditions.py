"""Look up (user, root record) pairs for every grounded-conditions record.

Usage:
    python scripts/query_conditions.py --domain airline \
        --conditions ExperimentResult/spec_v2/airline/specs_airline_gpt41_full_run6_gwt_v3_conditions.json \
        --today 2024-05-15 --now "2024-05-15 15:00:00 EST"
Output (default <conditions base>_matches.json): one row per GWT record with
user_id + root-record pairs; records without conditions or with errors pass
through with matches: null.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mining_v2.grounding import build_vocabulary, infer_links
from src.mining_v2.entity_query import all_persons, find_or_make, reverse_index


def _domain_clock(domain: str, tau_bench_path: str) -> tuple[str, str]:
    """The clock the DOMAIN runs on, not the wall clock.

    tau2 states it in the policy and its tools return the same constant
    (airline's _get_datetime() is literally "2024-05-15T15:00:00"), so every
    stored date sits around it. Falling through to today's real date instead
    silently empties every temporal condition — "booked within the last 24
    hours" matched 41 reservations on the domain clock and 0 on the wall
    clock, and nothing reported a problem.
    """
    from src.agent_interface.tau2_bench import load_tau2_bench_agent
    from src.mining_v2.domain_env import current_time_from_policy
    agent = load_tau2_bench_agent(tau_bench_path, domain)
    stamp = current_time_from_policy(agent.system_prompt or "")
    if not stamp or not stamp[0].isdigit():
        return "", ""
    return stamp.split()[0], stamp


def _relation_of(record: dict) -> str:
    """Ownership relation the grounding step tagged, if any."""
    for entry in record.get("nonDBconditions") or []:
        if entry.get("relation") in ("owner", "not_owner"):
            return entry["relation"]
    return "owner"


def main() -> None:
    p = argparse.ArgumentParser(description="Rule-based entity lookup for grounded conditions")
    p.add_argument("--domain", choices=["airline", "retail", "telecom"], required=True)
    p.add_argument("--conditions", required=True, help="Output JSON of the grounding step")
    p.add_argument("--today", default=None,
                   help="default: the current date stated in the domain policy")
    p.add_argument("--now", default=None,
                   help="default: the current time stated in the domain policy")
    p.add_argument("--tau-bench-path", default="tau2-bench")
    p.add_argument("--limit", type=int, default=20, help="Max (user, root) pairs kept per GWT")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    env = importlib.import_module(f"tau2.domains.{args.domain}.environment").get_environment()
    db = env.tools.db
    vocab = build_vocabulary(db)
    links = infer_links(db, vocab)
    rev = reverse_index(db, links)
    today, now = args.today, args.now
    if not today or not now:
        policy_today, policy_now = _domain_clock(args.domain, args.tau_bench_path)
        today, now = today or policy_today, now or policy_now
    ctx = {k: v for k, v in (("today", today), ("now", now)) if v}
    if not ctx:
        print("WARNING: no domain clock found; temporal conditions will be "
              "resolved against today's real date, which almost certainly "
              "matches nothing in this database")
    else:
        print(f"{args.domain}: domain clock today={today!r} now={now!r}")

    rows = json.load(open(args.conditions))
    out = []
    n_with, n_zero = 0, 0
    for r in rows:
        entry = dict(r)
        relation = _relation_of(r)
        # An ownership premise usually carries no field condition of its own —
        # "the booking is not the requester's" says nothing about the booking.
        # Any record will do, so the query still runs, with no conditions to
        # narrow it; skipping it here would hand back every user and lose the
        # one thing the scenario actually needs.
        # No field condition does not mean no entity: WHEN still acts on a
        # record of `root`, and the user has to be able to name one. Any
        # record of that table will do, so the lookup runs with no conditions.
        queryable = r.get("DBconditions") or r.get("root")
        if r.get("errors") or not queryable:
            entry["matches"] = all_persons(db)
            entry["lookup_status"] = "no_conditions"
            entry["relation"] = relation
            entry["n_matches"] = len(entry["matches"])
        else:
            root = r.get("root") or r["DBconditions"][0]["table"]
            res = find_or_make(db, vocab, args.domain, root, r.get("DBconditions") or [],
                               links=links, rev=rev, ctx=ctx, limit=args.limit,
                               relation=_relation_of(r))
            entry.update(res)
            entry["n_matches"] = len(res["matches"])
            n_with += bool(r.get("DBconditions"))
            n_zero += res["lookup_status"] in ("makeable", "unmakeable")
        out.append(entry)

    output = args.output or f"{os.path.splitext(args.conditions)[0]}_matches.json"
    json.dump(out, open(output, "w"), ensure_ascii=False, indent=2)
    print(f"{args.domain}: {len(out)} gwt records, {n_with} queried, "
          f"{n_zero} not directly matched -> {output}")


if __name__ == "__main__":
    main()
