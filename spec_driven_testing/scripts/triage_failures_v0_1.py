#!/usr/bin/env python3
"""Fail triage for an online run directory (reuse log section 202).

  # queue every fail for an external judge (Claude Code subagents fill triage/verdicts/)
  triage_failures_v0_1.py --run-dir outputs/<run> --backend queue

  # or judge them now with an LLM API
  triage_failures_v0_1.py --run-dir outputs/<run> --backend api --model <model> --api-file <file>

  # after verdicts were written: rebuild triage/summary.{json,md}
  triage_failures_v0_1.py --run-dir outputs/<run> --backend queue

Works on results/*.json (run_generic_tau_baselines_v0_1.py) and runs/*.json
(run_perturbation_pilot_v0_1.py). See docs/fail_triage.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentest.oracle.fail_triage_v0_1 import ApiTriageBackend, QueueTriageBackend, triage_directory


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=("api", "queue"))
    ap.add_argument("--verdicts", nargs="+", default=["fail"], help="branch verdicts to triage (default: fail)")
    ap.add_argument("--model", help="api backend: judge model")
    ap.add_argument("--api-file", type=Path, help="api backend: provider config file")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--probes", action="store_true",
                    help="also triage runs flagged by the cross-branch policy invariants (policy_invariants_v0_1), even if they passed")
    ap.add_argument("--out-name", default="triage", help="output subdirectory of --run-dir (default: triage)")
    args = ap.parse_args()
    if args.backend == "api":
        if not (args.model and args.api_file):
            ap.error("--backend api needs --model and --api-file")
        from agentest.execution_runner import configure_openai_compatible_environment, load_provider_config

        configure_openai_compatible_environment(load_provider_config(api_file=args.api_file))
        backend = ApiTriageBackend(args.model)
    else:
        backend = QueueTriageBackend(args.run_dir / args.out_name)
    summary = triage_directory(
        args.run_dir, backend, verdicts=tuple(args.verdicts), concurrency=args.concurrency, out_name=args.out_name,
        include_probes=args.probes,
    )
    print("reviewed:", json.dumps(summary["reviewed_verdict_counts"], ensure_ascii=False))
    print("raw oracle:", json.dumps(summary["raw_verdict_counts"], ensure_ascii=False))
    print("triage categories:", json.dumps(summary["counts"], ensure_ascii=False))
    print(f"report: {args.run_dir / args.out_name / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
