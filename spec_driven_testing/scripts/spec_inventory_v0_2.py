#!/usr/bin/env python3
"""Spec inventory revision against strategy library v0.2 (reuse log section 241).

  # queue every spec (prompt includes its v0.1 classification) for Claude Code subagents
  spec_inventory_v0_2.py --out-dir outputs/spec_inventory_v0_2 --previous outputs/spec_inventory_v0_1 --backend queue
  # subagents validate their files with --check <spec ids...>
  # re-run the queue command to rebuild summary.{json,md}
  # or classify with an LLM API: --backend api --model <m> --api-file <f> --concurrency N

No target-agent calls.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentest.strategy.spec_inventory_v0_2 import (
    ApiInventoryBackend, QueueInventoryBackend, build_spec_records, domain_policy, load_library, summarize, tool_source,
)


def _md(summary: dict, library: dict) -> str:
    lines = ["# Spec inventory (strategy library v0.2)", "",
             f"{summary['specs']} specs, {summary['classified']} classified, {len(summary['pending'])} pending, "
             f"{len(summary['invalid'])} failed validation.", "",
             f"Rule source: {json.dumps(summary['rule_source'], ensure_ascii=False)}; not enforced by the tool (tool_gap): "
             f"{summary['tool_gap']['true']}; structurally unlikely: {summary['markers'].get('M-unlikely', 0)}.", "",
             "## Normalized class", "", "| class | reached main test | not reached |", "|---|---|---|"]
    for k, v in summary["normalized"].items():
        lines.append(f"| {k} {v['label']} | {v.get('reached', 0)} | {v.get('not_reached', 0)} |")
    lines += ["", "## By domain", "", "| domain | specs | " + " | ".join(summary["normalized"]) + " |",
              "|---|---|" + "---|" * len(summary["normalized"])]
    for d, c in summary["by_domain"].items():
        lines.append(f"| {d} | {c.get('specs', 0)} | " + " | ".join(str(c.get(k, 0)) for k in summary["normalized"]) + " |")
    lines += ["", "## Primary structure type", "", "| type | specs |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["primary_structure"].items()]
    evidence = {s["id"]: (s["label"], s["deepseek_v4_flash"], bool(s.get("from_gap")))
                for t in library["structure_types"] for s in t["strategies"]}
    lines += ["", "## Specs per applicable strategy (NEW = added in v0.2)", "", "| strategy | specs | effect on DeepSeek |", "|---|---|---|"]
    for k, v in summary["strategy_specs"].items():
        label, ev, new = evidence.get(k, ("?", "?", False))
        lines.append(f"| {k} {label}{' NEW' if new else ''} | {v} | {ev} |")
    names = {b["code"]: b["label"] for b in library["blockers"]}
    lines += ["", "## Blockers", "", "| code | specs |", "|---|---|"]
    lines += [f"| {k} {names.get(k, '')} | {v} |" for k, v in summary["blocker_specs"].items()]
    if summary["invalid"]:
        lines += ["", "## Failed validation", ""] + [f"- {x['spec_id']}: {'; '.join(x['problems'])}" for x in summary["invalid"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--previous", required=True, type=Path, help="v0.1 inventory dir (classifications.json)")
    ap.add_argument("--backend", required=True, choices=("api", "queue"))
    ap.add_argument("--model")
    ap.add_argument("--api-file", type=Path)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--check", nargs="*", help="queue backend: only validate these specs' verdict files")
    args = ap.parse_args()

    library = load_library()
    records = build_spec_records()
    previous = json.loads((args.previous / "classifications.json").read_text(encoding="utf-8"))
    domains = sorted({r["domain"] for r in records})
    policies = {d: domain_policy(d) for d in domains}
    tools = {d: tool_source(d) for d in domains}
    if args.check is not None:
        backend = QueueInventoryBackend(args.out_dir / "queue")
        for record in (r for r in records if r["spec_id"] in set(args.check)):
            got = backend.collect(record, library, policies[record["domain"]], tools[record["domain"]])
            print(f"CHECK {record['spec_id']} " + ("MISSING" if got is None else
                  ("OK" if not got["problems"] else "PROBLEMS " + " | ".join(got["problems"]))), flush=True)
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    classifications: dict[str, dict | None] = {}
    if args.backend == "queue":
        backend = QueueInventoryBackend(args.out_dir / "queue")
        for record in records:
            d = record["domain"]
            got = backend.collect(record, library, policies[d], tools[d])
            if got is None:
                backend.submit(record, library, previous.get(record["spec_id"]), policies[d])
            classifications[record["spec_id"]] = got
    else:
        from agentest.execution_runner import configure_openai_compatible_environment, load_provider_config

        configure_openai_compatible_environment(load_provider_config(api_file=args.api_file))
        backend = ApiInventoryBackend(args.model)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = pool.map(lambda r: backend.classify(r, library, previous.get(r["spec_id"]), policies[r["domain"]],
                                                          tools[r["domain"]]), records)
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
