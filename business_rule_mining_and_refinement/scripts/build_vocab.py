"""Run L0: dump a domain's database vocabulary (tables, fields, types, values).

Deterministic and LLM-free — it reads the domain's pydantic models, so the
result is exact and independent of how much data happens to be stored.

Usage:
    python scripts/build_vocab.py --domain airline
    python scripts/build_vocab.py --domain all --output ExperimentResult/vocab
    python scripts/build_vocab.py --domain airline --table reservations --full
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mining_v2.grounding import Vocabulary, build_vocabulary
from src.mining_v2.domain_env import load_domain_db

DOMAINS = ("airline", "retail", "telecom")


def _print_report(vocab: Vocabulary, *, only: str | None, full: bool,
                  head: int) -> None:
    tables = vocab.tables
    if only:
        tables = {k: v for k, v in tables.items() if k == only}
        if not tables:
            print(f"  (no table named {only!r}; have: "
                  f"{', '.join(vocab.tables)})")
            return

    n_fields = sum(len(t.fields) for t in tables.values())
    n_declared = sum(1 for t in tables.values() for f in t.fields if f.values)
    n_collection = sum(1 for t in tables.values() for f in t.fields
                       if "[]" in f.path or "{}" in f.path)
    print(f"\n{'=' * 72}")
    print(f"  {vocab.domain}: {len(tables)} tables, {n_fields} field paths "
          f"({n_declared} with a declared value domain, "
          f"{n_collection} reaching into a collection)")
    print(f"{'=' * 72}")

    for name, table in tables.items():
        shown = table.fields if full else table.fields[:head]
        print(f"\n  table `{name}`  [{table.record_type}]  "
              f"{table.n_records} records, {len(table.fields)} paths")
        for f in shown:
            if f.values:
                # The declared domain is authoritative: the API rejects
                # anything outside it.
                domain = "∈ " + ", ".join(str(v) for v in f.values)
            elif f.examples:
                # Observed only — illustrates the shape, constrains nothing.
                domain = "e.g. " + ", ".join(str(v)[:22] for v in f.examples)
            else:
                domain = ""
            print(f"      {f.path:<42} {f.type:<7} {domain[:60]}")
        if not full and len(table.fields) > head:
            print(f"      ... {len(table.fields) - head} more "
                  f"(pass --full to list all)")


def main() -> None:
    p = argparse.ArgumentParser(description="L0 — database vocabulary")
    p.add_argument("--domain", default="airline",
                   choices=[*DOMAINS, "all"])
    p.add_argument("--tau-bench-path", default="tau2-bench")
    p.add_argument("--table", default=None, help="Only show this table")
    p.add_argument("--full", action="store_true",
                   help="List every field instead of the first few")
    p.add_argument("--head", type=int, default=12,
                   help="Fields shown per table without --full")
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--output", default=None,
                   help="Directory to write <domain>_vocab.json into")
    args = p.parse_args()

    domains = DOMAINS if args.domain == "all" else (args.domain,)
    for domain in domains:
        db = load_domain_db(args.tau_bench_path, domain)
        vocab = build_vocabulary(db, domain=domain, max_depth=args.max_depth)
        _print_report(vocab, only=args.table, full=args.full, head=args.head)

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            path = os.path.join(args.output, f"{domain}_vocab.json")
            with open(path, "w") as f:
                json.dump(vocab.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"\n  written → {path} "
                  f"({os.path.getsize(path) / 1024:.1f} KB)")
    print()


if __name__ == "__main__":
    main()
