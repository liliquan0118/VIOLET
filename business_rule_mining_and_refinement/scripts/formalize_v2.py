"""Formalize an already-mined SpecSet: split independently-violable
obligations and attach guard_dnf / LTL / semantic representations.

Usage:
    python scripts/formalize_v2.py --domain airline \
        --specs ExperimentResult/spec_v2/airline/specs_airline_gpt41_full_run6.json
    python scripts/formalize_v2.py --domain airline --specs ... --limit 8   # trial
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.types import AgentUnderTest
from src.mining_v2.formalize import formalize_specs
from src.mining_v2.spec import SpecSet


def main() -> None:
    p = argparse.ArgumentParser(description="Formalize mined specs (split + guard_dnf/LTL)")
    p.add_argument("--domain", choices=["airline", "retail", "telecom"], required=True)
    p.add_argument("--specs", required=True, help="Mined SpecSet JSON")
    p.add_argument("--agent-snapshot", default=None,
                   help="AgentUnderTest JSON (default: ExperimentResult/spec/<domain>/agent_<domain>.json)")
    p.add_argument("--tau-bench-path", default="tau2-bench")
    p.add_argument("--provider", choices=["openai", "deepseek", "claude"], default="openai")
    p.add_argument("--model", default=None)
    p.add_argument("--direction", choices=["pos", "neg", "v2"], default=None,
                   help="pos: positive-only GWT; neg: violation-only GWT; "
                        "v2: single IR derivation rendered as pos+neg+ir files; "
                        "default: bidirectional (legacy formalize)")
    p.add_argument("--output", default=None, help="Default: <specs base>_formal.json "
                                                  "(or _gwt_pos/_gwt_neg with --direction)")
    p.add_argument("--out-prefix", default=None,
                   help="v2 only: output path prefix for the _pos/_neg/_ir "
                        "files (default: <specs base>_gwt_v2)")
    p.add_argument("--limit", type=int, default=None, help="Only formalize the first N specs (trial)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    from src.core.llm_client import load_provider_env
    env_model = load_provider_env(args.provider)   # always configure provider base_url/key
    model = args.model or env_model

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    snap = args.agent_snapshot or f"ExperimentResult/spec/{args.domain}/agent_{args.domain}.json"
    if os.path.isfile(snap):
        agent = AgentUnderTest.load(snap)
    else:
        from src.agent_interface.tau2_bench import load_tau2_bench_agent
        agent = load_tau2_bench_agent(os.path.abspath(args.tau_bench_path), args.domain)

    spec_set = SpecSet.load(args.specs)
    specs = spec_set.specs[: args.limit] if args.limit else spec_set.specs

    if args.direction == "v2":
        import json

        from src.mining_v2.formalize_v2 import formalize_specs_v2
        views, report = formalize_specs_v2(agent, specs, model=model)
        base = os.path.splitext(args.specs)[0]
        prefix = args.out_prefix or f"{base}_gwt_v2"
        outputs = {}
        for name in ("gwt", "pos", "neg"):
            out_set = SpecSet(
                domain=spec_set.domain,
                specs=views[name],
                metadata={**spec_set.metadata, "formalize_report": report, "formalize_model": model},
            )
            # The deduplicated single view is the primary artifact; pos/neg
            # remain as reference projections of the same IR.
            path = f"{prefix}.json" if name == "gwt" else f"{prefix}_{name}.json"
            out_set.save(path)
            outputs[name] = path
        ir_path = f"{prefix}_ir.json"
        with open(ir_path, "w") as f:
            json.dump(
                {"domain": spec_set.domain,
                 "metadata": {**spec_set.metadata, "formalize_report": report, "formalize_model": model},
                 "specs": views["ir"]},
                f, indent=2, ensure_ascii=False,
            )
        outputs["ir"] = ir_path
        # Step-1 classification as a standalone artifact: reviewable,
        # hand-calibratable, and re-injectable to pin down run-to-run drift.
        cls_path = f"{prefix}_classification.json"
        with open(cls_path, "w") as f:
            json.dump(
                {"domain": spec_set.domain, "model": model,
                 "classifications": views["classification"]},
                f, indent=2, ensure_ascii=False,
            )
        outputs["classification"] = cls_path

        print(f"\n{'=' * 60}")
        print(f"  Formalize v2 — {args.domain}  ({args.specs})")
        print(f"{'=' * 60}")
        for k, v in report.items():
            if k == "errors":
                print(f"  errors: {len(v)}")
            else:
                print(f"  {k}: {v}")
        for name, path in outputs.items():
            print(f"  Output ({name}) → {path}")
        for name in ("pos", "neg"):
            n_gwt = sum(1 for s in views[name] if s.gwt)
            n_scen = sum(len(s.gwt or []) for s in views[name])
            print(f"  {name}: {n_gwt}/{len(views[name])} specs with GWT, "
                  f"{n_scen} sub-scenarios")
        return

    if args.direction == "pos":
        from src.mining_v2.formalize_pos import formalize_specs_pos
        convert, suffix = formalize_specs_pos, "_gwt_pos"
    elif args.direction == "neg":
        from src.mining_v2.formalize_neg import formalize_specs_neg
        convert, suffix = formalize_specs_neg, "_gwt_neg"
    else:
        convert, suffix = formalize_specs, "_formal"
    new_specs, report = convert(agent, specs, model=model)

    spec_set.specs = new_specs
    spec_set.metadata["formalize_report"] = report
    output = args.output or f"{os.path.splitext(args.specs)[0]}{suffix}.json"
    spec_set.save(output)

    print(f"\n{'=' * 60}")
    print(f"  Formalize — {args.domain}  ({args.specs})")
    print(f"{'=' * 60}")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"  Output → {output}\n")
    n_gwt = sum(1 for s in new_specs if s.gwt)
    n_scenarios = sum(len(s.gwt or []) for s in new_specs)
    n_multi = sum(1 for s in new_specs if len(s.gwt or []) > 1)
    print(f"  GWT attached: {n_gwt}/{len(new_specs)} specs")
    print(f"  GWT sub-scenarios: {n_scenarios} ({n_multi} specs have multiple scenarios)")


if __name__ == "__main__":
    main()
