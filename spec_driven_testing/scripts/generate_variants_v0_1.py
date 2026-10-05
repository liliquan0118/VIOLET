#!/usr/bin/env python3
"""Strategy-case generation from the spec inventory: select (spec, strategy) pairs and synthesize one test case per pair.

  # first strategy round: select pairs (variant_generation_v0_1.select_pairs) and queue one synthesis prompt per pair
  generate_variants_v0_1.py --inventory inputs/spec_inventory_v0_2 --out-dir outputs/round_a --backend queue \
      --per-spec 1 --available-probes S08.d S08.e S10.g S03.g S07.g
  # an LLM writes queue/cases/<case_id>.json for every queue/pending/<case_id>.prompt.md; check with --check <case ids...>
  # re-run the same command to validate every case and write cases.json + generation_summary.json

  # next round, guided by the bug experience of the previous one (variant_generation_v0_1.select_pairs_round2):
  # needs <previous>/cases.json and <previous>/main/violation_check/round_summary.json (summarize_strategy_round_v0_1.py)
  generate_variants_v0_1.py --inventory inputs/spec_inventory_v0_2 --out-dir outputs/round_b --backend queue \
      --round2-from outputs/round_a --available-probes S08.d S08.e S10.g S03.g S07.g

With these arguments and the inventory in inputs/, the selection reproduces the pairs of the paper's strategy rounds.
No target-agent calls.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agentest.strategy.spec_inventory_v0_2 import load_library
from agentest.strategy.variant_generation_v0_1 import (
    QueueGenerationBackend, load_plans, seed_candidates, select_pairs, select_pairs_round2, validate_case,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=("queue",))
    ap.add_argument("--per-spec", type=int, default=2)
    ap.add_argument("--available-probes", nargs="*", default=[], help="strategy ids whose new probe exists")
    ap.add_argument("--only", nargs="*", help="restrict to these case ids (pilot)")
    ap.add_argument("--check", nargs="*", help="only validate these cases and print problems")
    ap.add_argument("--round2-from", type=Path, help="out dir of the previous strategy round: select the next round's pairs by its results")
    args = ap.parse_args()

    library = load_library()
    records = json.loads((args.inventory / "records.json").read_text(encoding="utf-8"))
    classifications = json.loads((args.inventory / "classifications.json").read_text(encoding="utf-8"))
    by_spec = {r["spec_id"]: r for r in records}
    plans = load_plans()
    if args.round2_from:
        r1_cases = {c["case_id"] for c in json.loads((args.round2_from / "cases.json").read_text(encoding="utf-8"))}
        r1_summary = json.loads((args.round2_from / "main" / "violation_check" / "round_summary.json").read_text(encoding="utf-8"))
        pairs = select_pairs_round2(records, classifications, library, r1_cases, r1_summary["by_strategy"],
                                    available_probes=set(args.available_probes))
    else:
        pairs = select_pairs(records, classifications, library, per_spec=args.per_spec,
                             available_probes=set(args.available_probes))
    if args.only:
        pairs = [p for p in pairs if p["case_id"] in set(args.only)]
    backend = QueueGenerationBackend(args.out_dir / "queue")
    policies = args.inventory / "queue" / "policies"
    if not all((policies / f"{d}.md").exists() for d in {p["domain"] for p in pairs}):
        # the inventory was built without the queue backend: write the domain policies next to the queue
        from agentest.strategy.spec_inventory_v0_1 import domain_policy

        policies = args.out_dir / "queue" / "policies"
        policies.mkdir(parents=True, exist_ok=True)
        for d in {p["domain"] for p in pairs}:
            if not (policies / f"{d}.md").exists():
                (policies / f"{d}.md").write_text(domain_policy(d), encoding="utf-8")

    def seeds(pair):
        return seed_candidates(by_spec[pair["spec_id"]], plans[pair["domain"]])

    if args.check is not None:
        for pair in (p for p in pairs if p["case_id"] in set(args.check)):
            case = backend.collect(pair)
            problems = None if case is None else validate_case(case, pair, {s["bound_driver_plan_id"] for s in seeds(pair)})
            print(f"CHECK {pair['case_id']} " + ("MISSING" if case is None else ("OK" if not problems else
                  "PROBLEMS " + " | ".join(problems))), flush=True)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    valid, invalid, pending = [], [], []
    for pair in pairs:
        case = backend.collect(pair)
        if case is None:
            backend.submit(pair, by_spec[pair["spec_id"]], seeds(pair), str(policies / f"{pair['domain']}.md"))
            pending.append(pair["case_id"])
            continue
        problems = validate_case(case, pair, {s["bound_driver_plan_id"] for s in seeds(pair)})
        (invalid if problems else valid).append({**case, "domain": pair["domain"], "pair": pair, "problems": problems})
    (args.out_dir / "cases.json").write_text(json.dumps(valid, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {"pairs": len(pairs), "valid": len(valid), "invalid": [{"case_id": c["case_id"], "problems": c["problems"]}
                                                                       for c in invalid], "pending": len(pending),
               "by_domain": dict(Counter(p["domain"] for p in pairs)),
               "by_strategy": dict(Counter(p["strategy_id"] for p in pairs).most_common()),
               "by_rule_source": dict(Counter(p["rule_source"] for p in pairs))}
    (args.out_dir / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"GENERATION pairs={len(pairs)} valid={len(valid)} invalid={len(invalid)} pending={len(pending)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
