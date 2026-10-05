#!/usr/bin/env python3
"""Violation check for a strategy round (reuse log section 241).

Two tiers (feedback: verified verdicts, not raw judge output):
  1. --tier api    every run is judged once by the API judge (--model);
  2. --tier queue  Claude Code subagents re-judge blind: every API `violation`,
                   `case_invalid`, low-confidence or unparseable verdict, every
                   run with a policy probe flag, and a seeded --sample share of
                   the remaining API `compliant` / `not_reached` verdicts
                   (to measure API misses).
Final verdict per run = the queue verdict where one exists, else the API one.
Re-run with --tier queue after the verdicts are written to rebuild
violation_check/summary.{json,md}. No target-agent calls.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentest.oracle.policy_invariants_v0_1 import TRIAGED, check_invariants
from agentest.oracle.violation_check_v0_1 import (
    ApiViolationBackend, QueueViolationBackend, build_request, validate_verdict,
)


def _sampled(key: str, share: float) -> bool:
    """Stable per-run sample (independent of the other runs' routing)."""
    import hashlib

    return int(hashlib.sha256(f"241:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def _policy(domain: str, manual: bool, cache: dict) -> tuple[str, str]:
    key = f"{domain}{'_manual' if manual else ''}"
    if key not in cache:
        if manual:
            from tau2.domains.telecom.environment import get_environment

            cache[key] = get_environment(policy_type="manual").get_policy()
        else:
            from agentest.driver.generic_tau_online_v1 import load_default_tau_environment

            cache[key] = load_default_tau_environment(domain).get_policy()
    return cache[key], key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--tier", required=True, choices=("api", "queue"))
    ap.add_argument("--model")
    ap.add_argument("--api-file", type=Path)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sample", type=float, default=0.15, help="share of API compliant/not_reached re-judged in the queue")
    args = ap.parse_args()

    out = args.run_dir / "violation_check"
    out.mkdir(parents=True, exist_ok=True)
    api_path = out / "api_verdicts.json"
    api_verdicts = json.loads(api_path.read_text(encoding="utf-8")) if api_path.exists() else {}
    cache: dict = {}
    requests = {}
    for path in sorted((args.run_dir / "runs").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        execution = record["result"]
        flags = [f for f in check_invariants(execution, record["domain"]) if f["invariant"] in TRIAGED]
        req = build_request(path.stem, record, str(path), flags)
        manual = (execution.get("scenario_facts") or {}).get("telecom_policy_type") == "manual"
        requests[path.stem] = (req, record, manual)

    if args.tier == "api":
        from agentest.execution_runner import configure_openai_compatible_environment, load_provider_config

        configure_openai_compatible_environment(load_provider_config(api_file=args.api_file))
        backend = ApiViolationBackend(args.model)
        todo = [k for k in requests if k not in api_verdicts]
        policies = {k: _policy(requests[k][1]["domain"], requests[k][2], cache)[0] for k in todo}
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for key, verdict in zip(todo, pool.map(lambda k: backend.judge(requests[k][0], policies[k]), todo)):
                api_verdicts[key] = verdict
        api_path.write_text(json.dumps(api_verdicts, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"VIOLATION_API judged={len(todo)} total={len(api_verdicts)}", flush=True)

    queue = QueueViolationBackend(out / "queue")
    final, routed = {}, Counter()
    for key in sorted(requests):
        req, record, manual = requests[key]
        api = api_verdicts.get(key)
        if api and api.get("category"):  # re-validate with the current checker
            api = {**validate_verdict(req, {k: v for k, v in api.items() if k not in ("problems", "evidence_verified")}),
                   "judge": api.get("judge")}
        needs_queue = (api is None or api.get("category") in (None, "violation", "case_invalid")
                       or api.get("confidence") == "low" or api.get("problems") or req["policy_probe_flags"]
                       or _sampled(key, args.sample))
        # an existing blind verdict is always used, routed or not (routing can
        # shift when probes change)
        verdict = queue.collect(req)
        if verdict is not None:
            needs_queue = True
        if needs_queue:
            if verdict is None:
                text, name = _policy(record["domain"], manual, cache)
                queue.submit(req, text, name)
                routed["queued_pending"] += 1
            else:
                routed["queue_verdict"] += 1
        else:
            routed["api_only"] += 1
        final[key] = {"record": {k: record.get(k) for k in ("case", "spec_id", "strategy_id", "arm", "domain", "rep")},
                      "rule_source": (record["result"].get("scenario_facts") or {}).get("rule_source"),
                      "api": api and {k: api.get(k) for k in ("category", "confidence", "violation")},
                      "final": (verdict or api or {}).get("category") if (verdict or not needs_queue) else None,
                      "final_from": "queue" if verdict else ("api" if not needs_queue else "pending"),
                      "violation": (verdict or api or {}).get("violation")}

    # agreement of the API tier on runs the queue re-judged
    agree = Counter((v["api"] or {}).get("category") == v["final"] for v in final.values()
                    if v["final_from"] == "queue" and v["api"])
    by_case = defaultdict(lambda: {"trigger": Counter(), "control": Counter()})
    for v in final.values():
        by_case[v["record"]["case"]][v["record"]["arm"]][v["final"] or "pending"] += 1
    summary = {"runs": len(final), "routing": dict(routed),
               "api_agreement_on_rejudged": {"agree": agree[True], "disagree": agree[False]},
               "final_counts": {arm: dict(sum((c[arm] for c in by_case.values()), Counter())) for arm in ("trigger", "control")},
               "by_case": {k: {a: dict(c) for a, c in v.items()} for k, v in sorted(by_case.items())}}
    (out / "final_verdicts.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"VIOLATION_SUMMARY routing={json.dumps(summary['routing'])} final={json.dumps(summary['final_counts'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
