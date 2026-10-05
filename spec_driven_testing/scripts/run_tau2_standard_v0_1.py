#!/usr/bin/env python3
"""Run tau2-bench's own standard task set for one domain, exactly as tau-bench
does (tau2's native LLM user simulator following each task's user instructions,
the task's own initial state, tau2's native evaluation / reward), and keep the
full traces (reuse log section 213).

Only deviations from tau2's defaults, all by the project owner's decision
(2026-09-25): agent, user simulator and the NL-assertion evaluator all use
--model (tau2's default is gpt-4.1 for all three); telecom is tau2's standard
"telecom" domain (manual tech-support policy). Everything else (temperature 0,
seed 300, max steps 200, max errors 10, split "base") is tau2's default.

Outputs in <output-dir>/<domain>/:
  tau2_results.json   tau2's native results file (copied from tau2's data dir)
  tasks/<task_id>__t<trial>.json  one record per simulation: tau2 reward info
                      (reward, db/action/communicate/nl checks, breakdown) and the
                      full trace (every message, tool call and tool result), plus
                      an `execution` block in this project's execution format
                      (messages, termination_reason, empty transport log) so our
                      policy probes / oracles can later be run on it
  summary.json        pass/fail counts (pass = reward 1.0)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from agentest.execution_runner import (configure_agent_provider, configure_openai_compatible_environment,
                                            load_provider_config)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=("airline", "retail", "telecom"))
    ap.add_argument("--api-file", required=True, type=Path)
    ap.add_argument("--model", required=True,
                    help="agent, user and NL-assertion model; with --agent-api-file, user and NL-assertion only")
    ap.add_argument("--agent-api-file", type=Path,
                    help="section 493: provider file (base URL, model, key) for the agent under test only")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--concurrency", required=True, type=int)
    ap.add_argument("--task-ids", nargs="*", help="only these task ids (smoke tests)")
    ap.add_argument("--seed", type=int, default=300, help="tau2 base seed (default 300 = tau2's default)")
    ap.add_argument("--trial-offset", type=int, default=0,
                    help="top-up runs (reuse log section 510): add this to the trial number in the output records, so "
                         "extra trials run with another --seed sit next to the existing trials")
    ap.add_argument("--run-name", help="tau2 save_to name (default: agentspec_<domain>_<model>)")
    args = ap.parse_args()

    configure_openai_compatible_environment(load_provider_config(api_file=args.api_file))
    llm = args.model if "/" in args.model else f"openai/{args.model}"
    agent_llm = configure_agent_provider(args.agent_api_file) if args.agent_api_file else llm

    # tau2 reads the NL-assertion judge model from a module-level constant (no CLI
    # option); point it at the same model as agent and user.
    import tau2.evaluator.evaluator_nl_assertions as nl_eval

    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = llm
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.batch import run_domain
    from tau2.utils.utils import DATA_DIR

    run_name = args.run_name or f"agentspec_{args.domain}_{agent_llm if args.agent_api_file else args.model}".replace("/", "_")
    config = TextRunConfig(
        domain=args.domain, task_split_name="base", task_ids=args.task_ids or None,
        llm_agent=agent_llm, llm_args_agent={"temperature": 0.0},
        llm_user=llm, llm_args_user={"temperature": 0.0},
        num_trials=args.num_trials, max_concurrency=args.concurrency, save_to=run_name, seed=args.seed,
    )
    results = run_domain(config)

    out = args.output_dir / args.domain
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    native = DATA_DIR / "simulations" / run_name / "results.json"
    if native.exists():
        shutil.copy2(native, out / "tau2_results.json")
    tasks = {str(t.id): t for t in results.tasks}
    counts = {"pass": 0, "fail": 0, "no_reward": 0}
    for sim in results.simulations:
        reward_info = sim.reward_info.model_dump(mode="json") if sim.reward_info else None
        reward = (reward_info or {}).get("reward")
        verdict = "no_reward" if reward is None else ("pass" if reward >= 1.0 - 1e-9 else "fail")
        counts[verdict] += 1
        messages = [m.model_dump(mode="json", exclude_none=True) for m in (sim.messages or [])]
        task = tasks.get(str(sim.task_id))
        record = {
            "schema_version": "agentspectesting.tau2-standard-run/v0.1",
            "domain": args.domain, "task_id": str(sim.task_id), "trial": sim.trial + args.trial_offset, "seed": sim.seed,
            "model": {"agent": agent_llm, "user": llm, "nl_assertions": llm},
            "tau2_verdict": verdict, "reward": reward, "reward_info": reward_info,
            "termination_reason": str(getattr(sim.termination_reason, "value", sim.termination_reason)),
            "duration_seconds": sim.duration, "agent_cost": sim.agent_cost, "user_cost": sim.user_cost,
            "task": task.model_dump(mode="json", exclude_none=True) if task else None,
            "messages": messages,
            "execution": {
                "messages": messages,
                "termination_reason": str(getattr(sim.termination_reason, "value", sim.termination_reason)),
                "runtime_driver": {"transport_action_log": []},
            },
        }
        (out / "tasks" / f"{sim.task_id}__t{sim.trial + args.trial_offset}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    summary = {"domain": args.domain, "model": agent_llm, "user_model": llm, "num_trials": args.num_trials, "task_split": "base",
               "tasks": len(tasks), "simulations": len(results.simulations), "counts": counts,
               "pass_rate": counts["pass"] / max(1, len(results.simulations)), "tau2_run_name": run_name}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("TAU2_STANDARD_SUMMARY", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
