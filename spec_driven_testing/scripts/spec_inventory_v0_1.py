#!/usr/bin/env python3
"""Spec inventory against the strategy library (reuse log section 240).

  # queue every spec for an external classifier (Claude Code subagents fill verdicts/)
  spec_inventory_v0_1.py --out-dir outputs/spec_inventory_v0_1 --backend queue

  # or classify now with an LLM API
  spec_inventory_v0_1.py --out-dir ... --backend api --model <model> --api-file <file> --concurrency 8

  # after verdicts were written: re-run the queue command to rebuild summary.{json,md}

Outputs in <out-dir>/: records.json (every spec with its branches and pipeline
status), queue/{pending,verdicts,policies}/, classifications.json (validated),
summary.json, summary.md. No target-agent calls.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentest.strategy.spec_inventory_v0_1 import (
    ApiInventoryBackend, QueueInventoryBackend, build_spec_records, domain_policy, load_library, summarize,
)


def _md(summary: dict, library: dict) -> str:
    lines = ["# Spec inventory (strategy library v0.1)", "",
             f"{summary['specs']} specs, {summary['classified']} classified, {len(summary['pending'])} pending, "
             f"{len(summary['invalid'])} failed validation.", "", "## By domain", "",
             "| domain | specs | reached main test | testable | testable_with_extension | untestable |", "|---|---|---|---|---|---|"]
    for d, c in summary["by_domain"].items():
        lines.append(f"| {d} | {c.get('specs', 0)} | {c.get('reached_main_test', 0)} | {c.get('testable', 0)} | "
                     f"{c.get('testable_with_extension', 0)} | {c.get('untestable', 0)} |")
    lines += ["", "## Normalized class (by policy basis and observability, ignoring the testability label)", "",
              "| class | reached main test | not reached |", "|---|---|---|"]
    for k, v in summary["normalized"].items():
        lines.append(f"| {k} {v['label']} | {v.get('reached', 0)} | {v.get('not_reached', 0)} |")
    lines += ["", "## Structure types", "", "| type | specs (primary) | specs (any) |", "|---|---|---|"]
    for k, v in summary["primary_structure"].items():
        lines.append(f"| {k} | {v} | {summary['any_structure'].get(k, 0)} |")
    evidence = {s["id"]: (s["label"], s["deepseek_v4_flash"]) for t in library["structure_types"] for s in t["strategies"]}
    lines += ["", "## Specs per applicable strategy", "", "| strategy | specs | effect on DeepSeek |", "|---|---|---|"]
    for k, v in summary["strategy_specs"].items():
        zh, ev = evidence.get(k, ("?", "?"))
        lines.append(f"| {k} {zh} | {v} | {ev} |")
    names = {b["code"]: b["label"] for b in library["blockers"]}
    lines += ["", "## Blockers", "", "| code | specs |", "|---|---|"]
    lines += [f"| {k} {names.get(k, '')} | {v} |" for k, v in summary["blocker_specs"].items()]
    if summary["invalid"]:
        lines += ["", "## Failed validation", ""] + [f"- {x['spec_id']}: {'; '.join(x['problems'])}" for x in summary["invalid"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=("api", "queue"))
    ap.add_argument("--model")
    ap.add_argument("--api-file", type=Path)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--check", nargs="*", help="queue backend: only validate these specs' verdict files and print problems")
    args = ap.parse_args()

    library = load_library()
    records = build_spec_records()
    if args.check is not None:
        backend = QueueInventoryBackend(args.out_dir / "queue")
        policies = {}
        for record in (r for r in records if r["spec_id"] in set(args.check)):
            policy = policies.setdefault(record["domain"], domain_policy(record["domain"]))
            got = backend.collect(record, library, policy)
            print(f"CHECK {record['spec_id']} " + ("MISSING" if got is None else
                  ("OK" if not got["problems"] else "PROBLEMS " + " | ".join(got["problems"]))), flush=True)
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    policies = {d: domain_policy(d) for d in sorted({r["domain"] for r in records})}
    classifications: dict[str, dict | None] = {}
    if args.backend == "queue":
        backend = QueueInventoryBackend(args.out_dir / "queue")
        for record in records:
            got = backend.collect(record, library, policies[record["domain"]])
            if got is None:
                backend.submit(record, library, policies[record["domain"]])
            classifications[record["spec_id"]] = got
    else:
        from agentest.execution_runner import configure_openai_compatible_environment, load_provider_config

        configure_openai_compatible_environment(load_provider_config(api_file=args.api_file))
        backend = ApiInventoryBackend(args.model)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = pool.map(lambda r: backend.classify(r, library, policies[r["domain"]]), records)
            classifications = {r["spec_id"]: c for r, c in zip(records, results)}
    (args.out_dir / "classifications.json").write_text(
        json.dumps({k: v for k, v in classifications.items() if v is not None}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    summary = summarize(records, classifications, library)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "summary.md").write_text(_md(summary, library), encoding="utf-8")
    print(f"INVENTORY specs={summary['specs']} classified={summary['classified']} pending={len(summary['pending'])} "
          f"invalid={len(summary['invalid'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
