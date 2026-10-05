"""Ground a formalized GWT set and extract its conversation side.

One pass over every GWT scenario, writing the two artifacts the test
generator consumes:

  <prefix>_conditions.json          root + database conditions + the clauses
                                    no condition was written for (`nonDBconditions`)
  <prefix>_user_requirements.json   trigger, conversation_preconditions (from the clauses
                                    grounding left to the conversation),
                                    when_qualifiers, violation_supplies, and
                                    an untestable_in_domain flag when some
                                    GIVEN clause nobody can establish

The stages were previously driven by hand, which is why the artifacts and the
code that made them drifted apart. Usage:

    python scripts/ground_and_requirements.py --domain airline \
        --gwt ExperimentResult/spec_v2/airline/..._gwt_v4.json \
        --conditions ExperimentResult/spec_v2/airline/..._gwt_v4_conditions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mining_v2.conversation_side import extract_conversation_side
from src.mining_v2.grounding import build_vocabulary
from src.mining_v2.grounding_v2 import compile_condition_v2
from src.mining_v2.domain_env import load_domain_db
from src.mining_v2.user_requirements import (
    extract_trigger, extract_violation_supplies,
)

logger = logging.getLogger(__name__)


def _load_gwts(path: str) -> list[dict]:
    """Accept either the flat GWT list or a SpecSet-shaped file."""
    data = json.load(open(path))
    if isinstance(data, dict):
        data = data.get("specs") or data.get("gwts") or []
    out: list[dict] = []
    for entry in data:
        gwts = entry.get("gwt") or entry.get("gwts")
        if isinstance(gwts, dict):
            gwts = [gwts]
        for i, g in enumerate(gwts or []):
            out.append({
                "spec_id": entry.get("spec_id") or entry.get("id"),
                "kind": entry.get("kind"),
                "rule_text": entry.get("rule_text") or entry.get("rule") or "",
                "origin": entry.get("origin"),
                "gwt_index": i,
                "gwt": g,
            })
    return out


def _ground(rec: dict, vocab, domain: str, model: str) -> dict:
    gwt = rec["gwt"]
    given = str(gwt.get("given", "")).strip()
    root, conditions, dropped, errors = (None, [], [], [])
    # Run even for GIVEN "True": the segmenter returns no clauses then, but it
    # still names the ROOT (the entity WHEN acts on), and without a root the
    # lookup hands back bare users with no record for the user to refer to.
    if when := str(gwt.get("when", "")).strip():
        try:
            root, conditions, dropped, errors = compile_condition_v2(
                given or "True", vocab, trigger=when,
                then=gwt.get("then", ""), rule=rec.get("rule_text", ""),
                domain=domain, model=model)
        except Exception as exc:            # one bad scenario must not end the run
            errors = [f"grounding raised: {exc}"]
    warnings = [e for e in errors if e.startswith("warning:")]
    errors = [e for e in errors if not e.startswith("warning:")]
    return {**rec, "root": root, "DBconditions": conditions,
            "nonDBconditions": dropped, "errors": errors, "warnings": warnings,
            "model": model}


def _requirements(rec: dict, domain: str, model: str) -> dict:
    gwt, rule = rec["gwt"], rec.get("rule_text", "")
    errors: list[str] = []
    try:
        return _requirements_unguarded(rec, domain, model)
    except Exception as exc:            # a 429 mid-run must not sink the batch
        return {**{k: rec.get(k) for k in
                   ("spec_id", "kind", "rule_text", "origin", "gwt_index", "gwt")},
                "root": rec.get("root"), "DBconditions": rec.get("DBconditions") or [],
                "nonDBconditions": rec.get("nonDBconditions") or [],
                "relation": "owner", "untestable_in_domain": False, "untestable_reasons": [],
                "trigger": {"trigger": str(gwt.get("when", "")).strip(), "rephrased": False},
                "conversation_preconditions": [], "when_qualifiers": [],
                "violation_supplies": [], "model": model,
                "errors": [f"requirements raised: {type(exc).__name__}: {str(exc)[:200]}"]}


def _requirements_unguarded(rec: dict, domain: str, model: str) -> dict:
    gwt, rule = rec["gwt"], rec.get("rule_text", "")
    errors: list[str] = []
    trigger, rephrased, errs = extract_trigger(gwt, rule, domain=domain,
                                               model=model)
    errors += errs
    dropped = rec.get("nonDBconditions") or []
    conv, errs = extract_conversation_side(gwt, dropped, rule, trigger=trigger,
                                           domain=domain, model=model)
    errors += errs
    supplies, errs, meta = extract_violation_supplies(
        gwt, rule, trigger=trigger, domain=domain, model=model)
    errors += errs
    # A GIVEN clause nobody can establish — no field, no value, not the
    # user's to say — leaves the scenario unbuildable here. Flag it rather
    # than let a test run against a state that was never planted.
    # "none" here is a verified verdict: the field the classifier looked
    # for was checked against the vocabulary and is absent.
    unrealizable = [d for d in dropped if d.get("realizable") == "none"]
    relation = next((d["relation"] for d in dropped if d.get("relation")), "owner")
    return {**{k: rec[k] for k in
               ("spec_id", "kind", "rule_text", "origin", "gwt_index", "gwt")},
            "root": rec.get("root"), "DBconditions": rec.get("DBconditions") or [],
            "nonDBconditions": dropped, "relation": relation,
            "untestable_in_domain": bool(unrealizable),
            "untestable_reasons": [
                {"clause": d.get("clause", ""), "missing": d.get("missing", ""),
                 "check": d.get("check", ""), "why": d.get("why", "")}
                for d in unrealizable],
            "trigger": {"trigger": trigger, "rephrased": rephrased, **meta,
                        **conv["meta"]},
            "conversation_preconditions": conv["conversation_preconditions"],
            "when_qualifiers": conv["when_qualifiers"],
            "violation_supplies": supplies,
            "errors": errors, "model": model}


def main() -> None:
    p = argparse.ArgumentParser(description="Ground GWTs and extract the user side")
    p.add_argument("--domain", choices=["airline", "retail", "telecom"], required=True)
    p.add_argument("--gwt", required=True)
    p.add_argument("--conditions", required=True, help="conditions output path")
    p.add_argument("--requirements", default=None,
                   help="default: <conditions minus _conditions>_user_requirements.json")
    p.add_argument("--tau-bench-path", default="tau2-bench")
    p.add_argument("--model", default="gpt-4.1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-requirements", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    records = _load_gwts(args.gwt)
    if args.limit:
        records = records[:args.limit]
    db = load_domain_db(args.tau_bench_path, args.domain)
    vocab = build_vocabulary(db, domain=args.domain)
    logger.info("grounding %d scenarios", len(records))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        grounded = list(pool.map(
            lambda r: _ground(r, vocab, args.domain, args.model), records))
    os.makedirs(os.path.dirname(args.conditions) or ".", exist_ok=True)
    json.dump(grounded, open(args.conditions, "w"), indent=2, ensure_ascii=False)
    logger.info("wrote %s", args.conditions)
    if args.skip_requirements:
        return

    out = args.requirements or (
        args.conditions.replace("_conditions.json", "") + "_user_requirements.json")
    logger.info("extracting the user side of %d scenarios", len(grounded))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reqs = list(pool.map(
            lambda r: _requirements(r, args.domain, args.model), grounded))
    json.dump(reqs, open(out, "w"), indent=2, ensure_ascii=False)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
