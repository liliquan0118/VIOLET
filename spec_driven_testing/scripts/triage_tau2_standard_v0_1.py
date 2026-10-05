#!/usr/bin/env python3
"""Fail triage for tau2-bench standard-set runs (reuse log section 214).

Same judging scheme as fail_triage_v0_1 (categories, `outcome`, verbatim-evidence
validation, reviewed verdicts), but the requirement is the tau2 task's own
evaluation criteria instead of a compiled GWT: the gold action sequence (whose
replay defines the expected final DB), communicate_info, nl_assertions and
env_assertions, with reward_basis saying which of them count.

  triage_tau2_standard_v0_1.py --run-dir outputs/tau2_standard_v0_1/main          # queue failed tasks
  (judges write <run-dir>/triage/verdicts/<request_id>.json)
  triage_tau2_standard_v0_1.py --run-dir outputs/tau2_standard_v0_1/main          # re-run: rebuild summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentest.oracle.fail_triage_v0_1 import (
    CATEGORIES, KNOWN_HARNESS_LIMITATIONS, OUTCOMES, REVIEWED_LABELS, VERDICT_FORMAT,
    _transcript, reviewed_verdict, validate_verdict,
)

TAU2_NOTE = (
    "This is a run of tau2-bench's OWN standard task (not a compiled GWT test case). The user is tau2's LLM "
    "user simulator following the task's user instructions (task.user_scenario). The requirement is the task's "
    "evaluation criteria (task.evaluation_criteria): `actions` is the gold tool-call sequence -- tau2 replays it "
    "to build the expected final database and compares it with the real final database (DB check); "
    "communicate_info must appear in the agent's messages; nl_assertions are judged by an LLM; env_assertions "
    "check the final environment (e.g. the user's device). reward_info.reward_basis says which of these count "
    "and reward_info shows which failed. Decide WHY the task failed:\n"
    "- agent_behavior: the agent deviated from its policy or from what the user asked (missing/extra/wrong "
    "action, wrong arguments, wrong object, skipped step, gave in to an unallowed request, stopped early);\n"
    "- harness_artifact: the user simulator deviated from its instructions (gave wrong/invented information, asked "
    "for something else, ended early, refused a step it was told to accept) so the agent could not reach the gold "
    "outcome -- outcome not_tested;\n"
    "- oracle_bug: the task's expected outcome is questionable -- the gold actions contradict the policy, or the "
    "instructions are ambiguous so the agent's different outcome is also valid -- outcome agent_correct (or "
    "invalid_test if no agent could reach it);\n"
    "- infra_or_nondeterminism: errors/crashes; uncertain otherwise.\n"
    "Compare the agent's actual tool calls with the gold `actions` to locate the divergence, and quote it."
)


def _policy(domain: str) -> str:
    # tau2's own standard domains (telecom = manual tech-support policy), not this
    # project's workflow override (section 203)
    if domain == "airline":
        from tau2.domains.airline.environment import get_environment
    elif domain == "retail":
        from tau2.domains.retail.environment import get_environment
    else:
        from tau2.domains.telecom.environment import get_environment
    return get_environment().get_policy()


def _prompt(request: dict, policy_path: Path) -> str:
    body = {k: v for k, v in request.items() if k not in ("schema_version", "source_file")}
    return "\n\n".join([
        "You are auditing ONE failed run of a customer-service LLM agent on a tau2-bench task. Be skeptical in "
        "both directions; use only what the transcript and task show.",
        TAU2_NOTE,
        "Also: unless the category is agent_behavior, set `outcome` (" + ", ".join(OUTCOMES) + "). Every evidence "
        "quote must be copied exactly from the cited transcript message (content or tool_calls JSON).",
        f"The domain policy the agent was given is in {policy_path}.",
        "Case:\n" + json.dumps(body, ensure_ascii=False, indent=1, default=str),
        "Respond with ONLY one JSON object in this format:\n" + json.dumps(VERDICT_FORMAT, ensure_ascii=False, indent=1),
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()
    out = args.run_dir / "triage"
    for sub in ("pending", "verdicts", "policies"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows, runs = [], []
    for domain in ("airline", "retail", "telecom"):
        policy_path = out / "policies" / f"{domain}.md"
        if not policy_path.exists():
            policy_path.write_text(_policy(domain), encoding="utf-8")
        for path in sorted((args.run_dir / domain / "tasks").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            raw = record["tau2_verdict"]
            request_id = f"{domain}__{path.stem}"
            request = None
            if raw != "pass":
                request = {
                    "schema_version": "agentspectesting.tau2-triage-request/v0.1",
                    "request_id": request_id, "source_file": str(path), "domain": domain,
                    "task_id": record["task_id"], "branch_verdict": raw,
                    "termination_reason": record["termination_reason"],
                    "task": record["task"], "reward_info": record["reward_info"],
                    "transcript": _transcript(record["messages"]),
                    "known_harness_limitations": KNOWN_HARNESS_LIMITATIONS, "categories": CATEGORIES,
                }
                (out / "pending" / f"{request_id}.json").write_text(
                    json.dumps(request, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
                (out / "pending" / f"{request_id}.prompt.md").write_text(_prompt(request, policy_path), encoding="utf-8")
            verdict = None
            vpath = out / "verdicts" / f"{request_id}.json"
            if request is not None and vpath.exists():
                v = json.loads(vpath.read_text(encoding="utf-8"))
                judge = v.pop("judge", None)
                verdict = {**validate_verdict(request, v), "judge": judge}
            reviewed = reviewed_verdict("pass" if raw == "pass" else "fail", verdict, request is not None)
            runs.append({"request_id": request_id, "domain": domain, "task_id": record["task_id"],
                         "raw_verdict": raw, "reviewed_verdict": reviewed})
            if request is not None:
                rows.append({"request_id": request_id, "domain": domain, "task_id": record["task_id"],
                             "failed_components": sorted(k for k, x in (record["reward_info"].get("reward_breakdown") or {}).items() if x is not None and x < 1),
                             "reviewed_verdict": reviewed, "triage": verdict})
    counts: dict = {}
    for r in runs:
        key = (r["domain"], r["reviewed_verdict"])
        counts[key] = counts.get(key, 0) + 1
    summary = {"reviewed_by_domain": {f"{d}:{v}": n for (d, v), n in sorted(counts.items())}, "rows": rows, "runs": runs}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# tau2 standard set failure re-review", "", "| domain | reviewed verdict | count |", "|---|---|---|"]
    lines += [f"| {d} | {REVIEWED_LABELS.get(v, v)} | {n} |" for (d, v), n in sorted(counts.items())]
    lines += ["", "| request | failed components | reviewed verdict | category | outcome | confidence | notes |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["triage"] or {}
        note = (t.get("agent_violation") or t.get("explanation") or "").replace("|", "/").replace("\n", " ")
        if t.get("problems"):
            note = "⚠ " + "; ".join(t["problems"]) + " -- " + note
        lines.append(f"| {r['request_id'][:60]} | {', '.join(r['failed_components'])} | {r['reviewed_verdict']} | "
                     f"{t.get('category', 'pending')} | {t.get('outcome', '')} | {t.get('confidence', '')} | {note[:300]} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["reviewed_by_domain"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
