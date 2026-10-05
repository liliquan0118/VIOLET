#!/usr/bin/env python3
"""Run generated strategy cases (reuse log section 241).

Input: cases.json from generate_variants_v0_1.py (every case passed
validate_case). Each case has a trigger arm and a control arm; each arm
becomes a variant of its seed plan:
  - initial_user_message = the arm's user_goal (LLM user, tactical no_tactic),
    or the arm's telecom_user_simulator (tau2's real user simulator with
    device fault injection);
  - known facts, state patch, telecom policy type from the arm;
  - skip_branch_overrides (section 236);
  - scenario_facts carry the rule, its source quote, violation_check and the
    arm's expected_calls / unexpected_tools / intended_object, so the
    scenario_expectation and wrong_object probes and the violation check
    (oracle/violation_check_v0_1) can judge the run.
When a case has no seed plan (its spec never reached the main test) the
domain's first bound plan is the carrier; its branch oracle is then marked not
applicable and only the violation check counts.
Results: <output-dir>/runs/<case_id>__<arm>__r<rep>.json; existing non-error
files are skipped. --dry-run prints the jobs not on disk yet and the budget.
"""
from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock

from agentest.compiler.artifacts import content_sha256
from agentest.driver.generic_tau_online_v1 import run_generic_tau_online_plan
from agentest.driver.tactical_user_v0_1 import tactical_user_factory
from agentest.execution_runner import (configure_agent_provider, configure_openai_compatible_environment,
                                            load_provider_config)
from agentest.strategy.variant_generation_v0_1 import load_plans


def arm_plan(seed: dict, case: dict, arm: str) -> dict:
    a = case["arms"][arm]
    plan = deepcopy(seed)
    plan["dialogue_contract"]["initial_user_message"] = a["user_goal"]
    plan["skip_branch_overrides"] = True
    facts = dict(a.get("known_facts") or {})
    bindings = dict(case.get("oracle_bindings") or {}) if case.get("seed_oracle_applies") else \
        {k: v for k, v in facts.items() if isinstance(v, (str, int, float))}
    plan["object_bindings"] = bindings
    plan["oracle_scope_values"] = dict(bindings)
    plan["operation_argument_fact_bundle"]["arguments"] = facts
    plan["initial_state_patch"] = a.get("initial_state_patch") or {}
    if case.get("user") == "telecom_user_sim":
        plan["telecom_user_simulator"] = a["telecom_user_simulator"]
    if case.get("telecom_policy_type") == "manual":
        plan["telecom_policy_type"] = "manual"
    plan["bound_driver_plan_id"] = f"{seed['bound_driver_plan_id']}::SR1::{case['case_id']}::{arm}"
    plan.pop("bound_driver_plan_fingerprint", None)
    plan["bound_driver_plan_fingerprint"] = content_sha256(plan)
    return plan


def scenario_facts(case: dict, arm: str) -> dict:
    a, pair = case["arms"][arm], case["pair"]
    return {"experiment": "strategy round 1 (reuse log section 241)", "case_id": case["case_id"],
            "spec_id": case["spec_id"], "strategy_id": case["strategy_id"], "arm": arm,
            "rule_source": pair.get("rule_source"), "rule_quote": pair.get("source_quote"),
            "violation_form": pair.get("violation_form"), "violation_check": case["violation_check"],
            "seed_oracle_applies": bool(case.get("seed_oracle_applies")),
            "telecom_policy_type": case.get("telecom_policy_type") or "workflow",
            "expected_calls": a.get("expected_calls") or [], "unexpected_tools": a.get("unexpected_tools") or [],
            "intended_object": a.get("intended_object"),
            "user_goal_given_to_the_simulated_user": a["user_goal"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--api-file", required=True, type=Path)
    ap.add_argument("--model", required=True,
                    help="agent model; with --agent-api-file, the model of the simulated user and the judge")
    ap.add_argument("--agent-api-file", type=Path,
                    help="section 493: provider file (base URL, model, key) for the agent under test only; "
                         "the simulated user and the semantic judge stay on --model / --api-file")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--reps", type=json.loads, default={"trigger": 2, "control": 1})
    ap.add_argument("--agent-calls", type=json.loads, default={"retail": 16, "airline": 20, "telecom": 20,
                                                                "telecom_user_sim": 30})
    ap.add_argument("--only", nargs="*", help="restrict to these case ids")
    ap.add_argument("--approved-total-agent-calls", required=True, type=int)
    ap.add_argument("--concurrency", required=True, type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--telecom-policy", choices=("workflow", "manual"),
                    help="section 250: run every telecom case under this tau2 tech-support policy instead of the "
                         "case's own telecom_policy_type (both policies are part of the agent under test)")
    args = ap.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["case_id"] in set(args.only)]
    if args.telecom_policy:
        # section 250: override the policy config (arm_plan and scenario_facts read this field,
        # so the run and the violation check both see the policy the agent actually got)
        cases = [{**c, "telecom_policy_type": args.telecom_policy} if c["domain"] == "telecom" else c for c in cases]
    plans = load_plans()
    by_id = {p["bound_driver_plan_id"]: p for ps in plans.values() for p in ps}
    run_dir = args.output_dir / "runs"

    def calls(case):
        return args.agent_calls["telecom_user_sim" if case.get("user") == "telecom_user_sim" else case["domain"]]

    jobs = [dict(case=c, arm=arm, rep=rep) for c in cases for arm in ("trigger", "control") for rep in range(args.reps[arm])]

    def _path(job):
        return run_dir / f"{job['case']['case_id']}__{job['arm']}__r{job['rep']}.json"

    todo = [j for j in jobs if not (_path(j).exists()
                                    and json.loads(_path(j).read_text(encoding="utf-8")).get("status") != "error")]
    budget = sum(calls(j["case"]) for j in todo)
    print(f"SR1_PLAN new_jobs={len(todo)} of {len(jobs)} hard_budget={budget}", flush=True)
    if args.dry_run:
        for j in todo:
            print(f"  {_path(j).name}")
        return 0
    if budget > args.approved_total_agent_calls:
        ap.error(f"hard budget {budget} exceeds approved total {args.approved_total_agent_calls}")
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_openai_compatible_environment(load_provider_config(api_file=args.api_file))
    agent_model = configure_agent_provider(args.agent_api_file) if args.agent_api_file else args.model
    user_litellm = args.model if "/" in args.model else f"openai/{args.model}"
    lock, done = Lock(), [0]

    def _run(job):
        case, arm = job["case"], job["arm"]
        seed = by_id.get(case.get("seed_plan_id") or "") or plans[case["domain"]][0]
        record = {"branch": seed["source_branch_id"], "domain": case["domain"], "case": case["case_id"],
                  "spec_id": case["spec_id"], "strategy_id": case["strategy_id"], "arm": arm,
                  "user": case.get("user"), "rep": job["rep"],
                  "models": {"agent": agent_model, "user": args.model, "judge": args.model}}
        try:
            factory = None
            if case.get("user") != "telecom_user_sim":
                base = tactical_user_factory("no_tactic")

                def factory(bound_plan, task, environment, litellm_model, domain, scripted_user):
                    return base(bound_plan, task, environment, user_litellm, domain, scripted_user)

            result = run_generic_tau_online_plan(
                arm_plan(seed, case, arm), model=agent_model, approved_agent_call_budget=calls(case), seed=job["rep"],
                domain=case["domain"], enable_semantic_judge=True, user_factory=factory,
                user_model=args.model, judge_model=args.model)
            result["scenario_facts"] = scenario_facts(case, arm)
            record.update(status="completed", result=result, error=None)
            verdict = result["mechanical_oracle"]["branch_verdict"]
        except Exception as exc:
            record.update(status="error", result=None,
                          error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            verdict = "error"
        _path(job).write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return _path(job).name, record["status"], verdict

    # Section 499: the driver imports these tau2 modules lazily; first imports from many
    # worker threads at once deadlock on Python's module locks, so import them here first.
    import tau2.agent.llm_agent  # noqa: F401
    import tau2.data_model.simulation  # noqa: F401
    import tau2.data_model.tasks  # noqa: F401
    import tau2.domains.telecom.environment  # noqa: F401
    import tau2.orchestrator.orchestrator  # noqa: F401
    import tau2.user.user_simulator  # noqa: F401

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for future in as_completed([pool.submit(_run, j) for j in todo]):
            name, status, verdict = future.result()
            with lock:
                done[0] += 1
                print(f"SR1_PROGRESS {done[0]}/{len(todo)} {name} status={status} verdict={verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
