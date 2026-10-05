"""Audit a spec file for testability in its environment, and filter it.

Reads a mined spec set, anchors every rule's judgement bases in the domain's
policy / database / tool schemas (src.mining_v2.testability), and writes:

  <out>                    the audit: one record per spec with verdict,
                           anchored bases, the model's own verdict, reason
  <filtered>  (optional)   the spec file with every non-testable spec removed
                           (metadata gains a `testability` block)

Usage:
    python scripts/filter_testability.py --domain airline \
        --specs ExperimentResult/spec_v2/airline/specs_airline_gpt41_full_run6.json \
        --origin domain_knowledge \
        --out ExperimentResult/spec_v2/airline/specs_airline_gpt41_full_run6_testability.json \
        --filtered ExperimentResult/spec_v2/airline/specs_airline_gpt41_full_run6_testable.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_interface.tau2_bench import load_tau2_bench_agent
from src.mining_v2.grounding import build_vocabulary
from src.mining_v2.domain_env import load_domain_db
from src.mining_v2.testability import Environment, audit_spec

logger = logging.getLogger("filter_testability")


def _print_summary(audits: list[dict]) -> None:
    by = collections.Counter(a["verdict"] for a in audits)
    print(f"\n{len(audits)} specs audited: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by.items())))
    disagree = [a for a in audits if not a["agree"] and a["verdict"] != "error"]
    print(f"computed vs model verdict disagree on {len(disagree)}")
    for verdict in ("missing_concept", "tool_enforced", "subjective", "partial",
                    "error"):
        rows = [a for a in audits if a["verdict"] == verdict]
        if not rows:
            continue
        print(f"\n== {verdict} ({len(rows)}) ==")
        for a in rows:
            print(f"  {a['spec_id']:<20} {a['rule_text'][:95]}")
            for b in a["bases"]:
                if not b["verified"]:
                    near = f"  nearest: {b['nearest'][:70]}" if b["nearest"] else ""
                    print(f"      - {b['basis'][:80]}  [{b['anchor']}]{near}")
            if a["reason"]:
                print(f"      reason: {a['reason'][:140]}")
    notes = [a for a in audits if a["note"]]
    if notes:
        print(f"\n== stated value wrong for this environment ({len(notes)}) ==")
        for a in notes:
            print(f"  {a['spec_id']:<20} {a['note'][:140]}")
    if disagree:
        print("\n== disagreements (computed / model) ==")
        for a in disagree:
            print(f"  {a['spec_id']:<20} {a['verdict']} / {a['llm_verdict']}"
                  f"  {a['rule_text'][:70]}")


def main() -> None:
    p = argparse.ArgumentParser(description="Testability audit + filter")
    p.add_argument("--domain", choices=["airline", "retail", "telecom"], required=True)
    p.add_argument("--specs", required=True)
    p.add_argument("--origin", default=None,
                   help="audit only specs of this origin (schema/prompt/domain_knowledge)")
    p.add_argument("--out", required=True, help="audit output path")
    p.add_argument("--filtered", default=None,
                   help="write the spec file with non-testable specs removed")
    p.add_argument("--tau-bench-path", default="tau2-bench")
    p.add_argument("--model", default="gpt-4.1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    doc = json.load(open(args.specs))
    specs = doc["specs"] if isinstance(doc, dict) else doc
    targets = [s for s in specs if not args.origin or s.get("origin") == args.origin]
    if args.limit:
        targets = targets[:args.limit]

    agent = load_tau2_bench_agent(os.path.abspath(args.tau_bench_path), args.domain)
    vocab = build_vocabulary(load_domain_db(args.tau_bench_path, args.domain),
                             domain=args.domain)
    env = Environment(agent, vocab, args.domain)
    logger.info("auditing %d specs (%s)", len(targets), args.origin or "all origins")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        audits = list(pool.map(lambda s: audit_spec(s, env, model=args.model),
                               targets))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"domain": args.domain, "model": args.model,
               "source": args.specs, "origin": args.origin,
               "summary": dict(collections.Counter(a["verdict"] for a in audits)),
               "audits": audits},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    logger.info("wrote %s", args.out)

    if args.filtered:
        drop = {a["spec_id"]: a["verdict"] for a in audits
                if a["verdict"] not in ("testable", "partial", "error")}
        kept = [s for s in specs if s.get("spec_id") not in drop]
        out = dict(doc) if isinstance(doc, dict) else {"specs": kept}
        out["specs"] = kept
        meta = dict(out.get("metadata") or {})
        meta["testability"] = {"audited_origin": args.origin,
                               "n_removed": len(drop),
                               "removed": drop, "audit": args.out}
        out["metadata"] = meta
        json.dump(out, open(args.filtered, "w"), indent=2, ensure_ascii=False)
        logger.info("wrote %s (%d -> %d specs)", args.filtered, len(specs), len(kept))

    _print_summary(audits)


if __name__ == "__main__":
    main()
