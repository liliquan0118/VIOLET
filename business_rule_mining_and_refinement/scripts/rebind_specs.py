"""Repair under-bound specs in an existing SpecSet: run the binding-coverage
pass (deterministic gap detection + per-tool yes/no adjudication), then
re-specialize the expanded rules into per-tool specs.

Usage:
    python scripts/rebind_specs.py --domain retail \
        --specs ExperimentResult/spec_v3/retail/specs_retail_gpt41_full_run3.json
Output: <specs base>_rebound.json (+ coverage report in metadata).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mining_v2.common import specialize, verify_binding_coverage
from src.mining_v2.spec import SpecSet


def main() -> None:
    p = argparse.ArgumentParser(description="Binding-coverage repair for a mined SpecSet")
    p.add_argument("--domain", choices=["airline", "retail", "telecom"], required=True)
    p.add_argument("--specs", required=True)
    p.add_argument("--tau-bench-path", default="tau2-bench")
    p.add_argument("--provider", choices=["openai", "deepseek", "claude"], default="openai")
    p.add_argument("--model", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from src.core.llm_client import load_provider_env
    model = args.model or load_provider_env(args.provider)

    from src.agent_interface.tau2_bench import load_tau2_bench_agent
    agent = load_tau2_bench_agent(os.path.abspath(args.tau_bench_path), args.domain)

    spec_set = SpecSet.load(args.specs)
    coverage = verify_binding_coverage(agent, spec_set.specs, domain=args.domain,
                                       model=model)
    spec_set.specs = specialize(spec_set.specs)
    spec_set.metadata["binding_coverage"] = coverage

    output = args.output or f"{os.path.splitext(args.specs)[0]}_rebound.json"
    spec_set.save(output)
    print(f"{args.domain}: {coverage['n_checked']} candidates checked, "
          f"{coverage['n_added']} bindings added -> {len(spec_set.specs)} specs "
          f"-> {output}")
    for a in coverage["added"]:
        print(f"  + {a['spec_id']} -> {a['tool']}   ({a['reason'][:70]})")


if __name__ == "__main__":
    main()
