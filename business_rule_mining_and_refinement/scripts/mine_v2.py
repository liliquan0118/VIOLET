"""Run mining_v2 on a tau2-bench domain.

Usage:
    python scripts/mine_v2.py --domain airline --provider openai
    python scripts/mine_v2.py --domain airline --no-gap-fill        # only grounded rules
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_interface.tau2_bench import load_tau2_bench_agent
from src.mining_v2.pipeline import mine


def main() -> None:
    parser = argparse.ArgumentParser(description="mining_v2 — business-rule spec mining")
    parser.add_argument("--domain", choices=["retail", "airline", "telecom"], default="airline")
    parser.add_argument("--tau-bench-path", default="tau2-bench")
    parser.add_argument("--agent-snapshot", default=None,
                        help="Load a saved AgentUnderTest JSON instead of tau2-bench "
                             "(e.g. ExperimentResult/spec/airline/agent_airline.json)")
    parser.add_argument("--provider", choices=["openai", "deepseek", "claude"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--prompt-miner", choices=["v2"], default="v2",
                        help="v2: coverage audit with code-guaranteed completeness (prompt_miner_v2)")
    parser.add_argument("--no-gap-fill", action="store_true",
                        help="Skip Pass 2 (hypothesized rules); keep only grounded rules")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from src.core.llm_client import load_provider_env
    env_model = load_provider_env(args.provider)
    model = args.model or env_model

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.agent_snapshot:
        from src.core.types import AgentUnderTest
        agent = AgentUnderTest.load(args.agent_snapshot)
        logging.info("Loaded %s agent from snapshot %s: %d tools",
                     args.domain, args.agent_snapshot, len(agent.tools))
    else:
        tau_path = os.path.abspath(args.tau_bench_path)
        agent = load_tau2_bench_agent(tau_path, args.domain)
        logging.info("Loaded %s agent: %d tools", args.domain, len(agent.tools))

    pm = None
    if args.prompt_miner == "v2":
        from src.mining_v2.prompt_miner_v2 import mine_prompt as pm
    spec_set = mine(agent, model=model, gap_fill=not args.no_gap_fill, prompt_miner=pm)

    output = args.output or f"ExperimentResult/spec_v2/{args.domain}/specs_{args.domain}.json"
    # Record what mined this set. Nothing did before, so whether a spec file
    # came from gpt-4.1 or the provider's default could not be told from the
    # artifact — only guessed from its filename.
    from datetime import datetime
    spec_set.metadata["mining"] = {"model": model, "provider": args.provider,
                                   "prompt_miner": args.prompt_miner,
                                   "timestamp": datetime.now().isoformat(timespec="seconds")}
    _save_prompt_audit(spec_set, output)
    spec_set.save(output)

    # Also split by source into separate files (schema / prompt / domain_knowledge)
    # so each set can be reviewed on its own — prompt-grounded specs are the ones
    # that can be checked against the policy by hand.
    _save_by_origin(spec_set, output)

    # Sentence-level coverage is closed inside prompt_miner: a forced-enumeration
    # quote audit marks every prompt sentence occurrence not verbatim-quoted by
    # any spec's evidence, and a per-sentence adjudication call resolves each one
    # (extend an existing spec's evidence / new spec / not-a-rule). Use
    # scripts/coverage_v2.py for an optional (noisy) manual re-check.

    _print_report(spec_set, output)


def _save_prompt_audit(spec_set, output: str) -> None:
    """Move the prompt miner's raw enumeration + adjudication records out of the
    spec-set metadata into a sidecar <output>_prompt_audit.json, so every
    quoted/unquoted judgment and every verdict can be inspected after a run."""
    import json
    import os
    pr = spec_set.metadata.get("prompt_report") or {}
    sentences = pr.pop("sentences", None)
    verdicts = pr.pop("verdicts", None)
    if sentences is None and verdicts is None:
        return
    base, _ = os.path.splitext(output)
    path = f"{base}_prompt_audit.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"sentences": sentences or [], "verdicts": verdicts or []},
                  f, ensure_ascii=False, indent=2)
    print(f"  prompt audit     → {path}")


def _save_by_origin(spec_set, output: str) -> None:
    import json
    import os
    base, _ = os.path.splitext(output)
    groups = {"schema": [], "prompt": [], "domain_knowledge": []}
    for s in spec_set.specs:
        groups.get(s.origin, groups.setdefault(s.origin, [])).append(s)
    for origin, specs in groups.items():
        if not specs:
            continue
        path = f"{base}_{origin}.json"
        payload = {
            "domain": spec_set.domain,
            "source": origin,
            "total": len(specs),
            "specs": [s.to_dict() for s in specs],
        }
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  {origin:16s} {len(specs):3d} specs → {path}")


def _print_report(spec_set, output: str) -> None:
    m = spec_set.metadata
    print(f"\n{'=' * 66}")
    print(f"  mining_v2 — {spec_set.domain}")
    print(f"{'=' * 66}")
    print(f"  Total specs:         {m['n_total']}")
    print(f"    from schema:       {m['n_struct']}")
    print(f"    from prompt:       {m['n_prompt']}")
    print(f"    from domain know.: {m.get('n_domain_knowledge', 0)}")
    print(f"  By kind:             {m['kind_breakdown']}")
    print(f"  By confidence:       {m['confidence_breakdown']}")
    print(f"  Multi-clause specs:  {m.get('n_multi_clause', 0)}  (rule spans >1 statement)")
    print(f"  Multi-tool specs:    {m.get('n_multi_tool', 0)}")
    print(f"  Empty STATE/ORDER cells (tool×kind): {m['coverage_empty_state_order_cells']}")
    print(f"  Output:              {output}")
    print(f"{'=' * 66}")

    # show a few specs per kind
    for kind in ("STATE", "ORDER", "NORM", "ARG"):
        specs = [s for s in spec_set.specs if s.kind == kind]
        print(f"\n  [{kind}] {len(specs)} specs — sample:")
        for s in specs[:6]:
            conf = {"structural": "S", "stated": "✓", "hypothesized": "?"}.get(s.confidence, " ")
            tools = ",".join(s.relevant_tools) or "agent"
            multi = f" [{len(s.clauses)} clauses]" if len(s.clauses) > 1 else ""
            print(f"    ({conf}) <{s.target_action}> [{tools}]{multi} {s.rule_text}")
    print()


if __name__ == "__main__":
    main()
