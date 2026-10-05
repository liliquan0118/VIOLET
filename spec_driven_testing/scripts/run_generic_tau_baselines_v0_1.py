#!/usr/bin/env python3
"""Run an approved batch of generic τ-bench baselines for a domain."""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from agentest.driver.generic_tau_online_v1 import (
    load_bound_plan_set,
    run_generic_tau_online_plan,
)
from agentest.execution_runner import (
    configure_agent_provider,
    configure_openai_compatible_environment,
    load_provider_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-plans", required=True, type=Path)
    parser.add_argument("--domain", default="airline", choices=("airline", "retail", "telecom"))
    parser.add_argument("--api-file", required=True, type=Path)
    parser.add_argument("--model", required=True,
                        help="agent model; with --agent-api-file, the model of the simulated user and the judges")
    parser.add_argument("--agent-api-file", type=Path,
                        help="section 493: provider file (base URL, model, key) for the agent under test only")
    parser.add_argument("--per-case-agent-calls", required=True, type=int)
    parser.add_argument("--approved-total-agent-calls", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--offset", type=int, default=0, help="skip the first N bound plans (for batching a full corpus)")
    parser.add_argument("--only", nargs="+", help="run only these branch ids (targeted verification)")
    parser.add_argument("--limit", type=int, help="only run the next N bound plans after --offset (a real subset, not a sample)")
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="real branches to run concurrently (each is its own real conversation against a fresh "
        "environment, independent of the others -- safe to parallelize; deepseek-flash allows up to "
        "2500 concurrent real requests per account, well above any real batch size used here)",
    )
    parser.add_argument(
        "--enable-semantic-judge", action="store_true",
        help="make a real extra LLM call (same model/provider) to evaluate semantic_transcript_judgment "
        "predicates instead of the honest default 'semantic_judge_not_configured' fail (section 65)",
    )
    parser.add_argument(
        "--triage", choices=("off", "api", "queue"), default="off",
        help="section 202: after the batch, triage every fail (api = judge now with --triage-model; "
        "queue = write triage/pending/ for an external judge, see docs/fail_triage.md)",
    )
    parser.add_argument("--triage-model", help="judge model for --triage api (default: --model)")
    parser.add_argument(
        "--telecom-policy", choices=("workflow", "manual"),
        help="section 250: run telecom under this tau2 tech-support policy (both are part of the agent "
        "under test); stamped into each result as telecom_policy_type. Default: the plan's own "
        "(workflow unless the plan says manual)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the jobs and the hard budget, run nothing")
    args = parser.parse_args()
    if args.telecom_policy and args.domain != "telecom":
        parser.error("--telecom-policy is for --domain telecom")

    source = load_bound_plan_set(args.bound_plans)
    sliced = source["bound_plans"][args.offset :]
    plans = sliced[: args.limit] if args.limit else sliced
    if args.only:
        missing = set(args.only) - {p["source_branch_id"] for p in plans}
        if missing:
            parser.error(f"--only branches not in the plan set: {sorted(missing)}")
        plans = [p for p in plans if p["source_branch_id"] in set(args.only)]
    hard_budget = len(plans) * args.per_case_agent_calls
    if args.per_case_agent_calls < 1 or hard_budget > args.approved_total_agent_calls:
        parser.error(
            f"requested hard budget {hard_budget} exceeds approved total "
            f"{args.approved_total_agent_calls}"
        )
    if args.dry_run:
        existing = {p.stem for p in (args.output_dir / "results").glob("*.json")} if (args.output_dir / "results").exists() else set()
        print(f"DRY_RUN cases={len(plans)} hard_budget={hard_budget} telecom_policy={args.telecom_policy} "
              f"already_on_disk={sum(p['source_branch_id'].replace('#', '__') in existing for p in plans)}")
        for p in plans:
            print(f"  {p['source_branch_id']}")
        return 0
    provider = load_provider_config(api_file=args.api_file)
    configure_openai_compatible_environment(provider)
    agent_model = configure_agent_provider(args.agent_api_file) if args.agent_api_file else args.model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.output_dir / "results"
    result_dir.mkdir(exist_ok=True)
    results = []
    observed = 0
    progress_lock = Lock()
    completed_so_far = 0
    print(
        f"GENERIC_EXECUTION_START cases={len(plans)} offset={args.offset} per_case={args.per_case_agent_calls} "
        f"hard_budget={hard_budget} model={agent_model} user_judge_model={args.model} concurrency={args.concurrency}", flush=True
    )

    def _write_record(record):
        safe = record["branch_id"].replace("#", "__")
        (result_dir / f"{safe}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return record

    def _run_one(plan):
        # section 117: a plan whose real compiled generation_readiness.
        # runtime_execution_allowed is False (generic_tau_v2_bound_plan_adapter_v1.py
        # -- set when the branch's real oracle_plan.checks list is empty) has nothing
        # a runtime execution could ever check. Skip it before launching any real
        # online conversation -- burning a real, full LLM conversation against
        # deepseek-v4-flash on it would only ever produce a vacuous "pass" (nothing
        # was ever checked), wasting real API budget for zero signal. This check runs
        # BEFORE the try/except below on purpose: no real network call is made for a
        # skipped branch, so it can never land in the "error" bucket either.
        if not plan["generation_readiness"]["runtime_execution_allowed"]:
            return _write_record({
                "branch_id": plan["source_branch_id"],
                "status": "skipped_not_scriptable",
                "result": None,
                "error": None,
            })
        try:
            result = run_generic_tau_online_plan(
                plan, model=agent_model, user_model=args.model, judge_model=args.model,
                approved_agent_call_budget=args.per_case_agent_calls,
                seed=args.seed, timeout_seconds=args.timeout_seconds,
                domain=args.domain,
                enable_semantic_judge=args.enable_semantic_judge,
                telecom_policy_type=args.telecom_policy,
            )
            if args.telecom_policy:
                result["telecom_policy_type"] = args.telecom_policy
            status = "completed"
            error = None
        except Exception as exc:
            result = None
            status = "error"
            # Keep the traceback: airline_113_state#b1's one-off frozenset
            # TypeError (reuse log section 170.11) could not be located without it.
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        return _write_record({
            "branch_id": plan["source_branch_id"],
            "status": status,
            "result": result,
            "error": error,
        })

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {executor.submit(_run_one, plan): plan for plan in plans}
        for future in as_completed(futures):
            record = future.result()
            result = record["result"]
            with progress_lock:
                observed += result["budget"]["observed_agent_calls"] if result else 0
                results.append(record)
                completed_so_far += 1
                index = completed_so_far
            verdict = result["mechanical_oracle"]["branch_verdict"] if result else record["status"]
            calls = result["budget"]["observed_agent_calls"] if result else 0
            print(
                f"GENERIC_EXECUTION_PROGRESS {index}/{len(plans)} branch={record['branch_id']} "
                f"status={record['status']} verdict={verdict} calls={calls}", flush=True
            )
    if observed > args.approved_total_agent_calls:
        raise RuntimeError("approved total target-Agent call budget was exceeded")
    summary = {
        "schema_version": "agentspectesting.generic-tau-online-batch/v0.1",
        "domain": args.domain,
        "model": agent_model,
        "user_judge_model": args.model,
        "provider_source": provider.source,
        "provider_base_url": provider.base_url,
        "offset": args.offset,
        "case_count": len(plans),
        "completed_count": sum(x["status"] == "completed" for x in results),
        "error_count": sum(x["status"] == "error" for x in results),
        # section 117: a branch whose real compiled generation_readiness.
        # runtime_execution_allowed is False is skipped before any online conversation
        # is launched (see _run_one above) -- never counted as completed/error, and
        # never counted as pass/fail/incomplete in branch_verdict_counts below (there
        # is no real mechanical_oracle result for it at all). Honestly represented in
        # its own bucket instead.
        "skipped_not_scriptable_count": sum(x["status"] == "skipped_not_scriptable" for x in results),
        "branch_verdict_counts": {
            verdict: sum(
                x["status"] == "completed" and x["result"]["mechanical_oracle"]["branch_verdict"] == verdict
                for x in results
            )
            for verdict in ("pass", "fail", "incomplete")
        },
        "per_case_agent_call_budget": args.per_case_agent_calls,
        "telecom_policy_type": args.telecom_policy,
        "approved_total_agent_call_budget": args.approved_total_agent_calls,
        "hard_scheduled_agent_call_budget": hard_budget,
        "observed_agent_calls": observed,
        "budget_respected": observed <= args.approved_total_agent_calls,
        "results": [
            {
                "branch_id": x["branch_id"], "status": x["status"],
                "verdict": x["result"]["mechanical_oracle"]["branch_verdict"] if x["result"] else None,
                "error": x["error"],
            }
            for x in results
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.triage != "off":
        from agentest.oracle.fail_triage_v0_1 import (
            ApiTriageBackend, QueueTriageBackend, triage_directory,
        )

        backend = (
            ApiTriageBackend(args.triage_model or args.model) if args.triage == "api"
            else QueueTriageBackend(args.output_dir / "triage")
        )
        triage = triage_directory(args.output_dir, backend, concurrency=args.concurrency, include_probes=True)
        print(f"TRIAGE {args.triage} reviewed={json.dumps(triage['reviewed_verdict_counts'])} raw={json.dumps(triage['raw_verdict_counts'])} triage={json.dumps(triage['counts'])} report={args.output_dir / 'triage' / 'summary.md'}", flush=True)
    return 0 if summary["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
