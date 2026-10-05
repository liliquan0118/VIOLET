#!/usr/bin/env python3
"""Review of the IntellAgent baseline (reuse log section 472), same judging scheme as
fail_triage_v0_1 / triage_tau2_standard_v0_1 (categories, `outcome`, verbatim
evidence, reviewed verdicts). The turn rule ("one tool call at a time / no reply
while calling a tool") is reported separately and never counted with the other
violations (user decision).

Input: outputs/intellagent_review_v0_1/dialogs.jsonl (export_intellagent_dialogs_v0_1.py).

Stage A -- split IntellAgent's own reason into claims. Every claim is an exact quote
of the reason, labelled `turn_rule` or `other`; claims IntellAgent itself retracts
("the feedback clarifies X was correct") are not claims.
  review_intellagent_v0_1.py stage-a-prepare [--sample N] [--chunk-size K]
  (judges write stage_a/output/<chunk>.jsonl)
  review_intellagent_v0_1.py stage-a-collect

Stage B -- blind review of every dialog with at least one `other` claim: does the
agent really violate its policy as claimed? One request per dialog.
  review_intellagent_v0_1.py stage-b-prepare
  (judges write stage_b/verdicts/<dialog_id>.json)

  review_intellagent_v0_1.py summarize      # tables: turn rule separate, other violations reviewed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from agentest.oracle.fail_triage_v0_1 import (
    CATEGORIES, CONSENT_RULE, OUTCOMES, REVIEWED_LABELS, VERDICT_FORMAT, reviewed_verdict, validate_verdict,
)

ROOT = Path(__file__).resolve().parents[1]
# section 506: IA_REVIEW_OUT selects the review directory (Qwen rerun: intellagent_review_v0_1_qwen38flash)
OUT = ROOT / "outputs" / os.environ.get("IA_REVIEW_OUT", "intellagent_review_v0_1")
CLAIM_KINDS = ("turn_rule", "other")

TURN_RULE = (
    "The TURN RULE is the policy sentence \"You should only make one tool call at a time, and if you make a "
    "tool call, you should not respond to the user simultaneously.\" A claim is `turn_rule` if what it blames "
    "is only that: several tool calls issued together/at once/in parallel, or a message to the user sent in the "
    "same turn as a tool call. Everything else (wrong action, missing confirmation, invented information, wrong "
    "object, wrong refund, no transfer, ...) is `other`, even if the same sentence also mentions the turn rule -- "
    "then quote the part about the other problem as its own claim."
)

STAGE_A_INSTRUCTIONS = "\n".join([
    "You split the final judgement of an automated test tool (IntellAgent) into the violations it claims.",
    "Input: a JSONL file; each line has dialog_id, domain and ia_reason (IntellAgent's final reason for judging "
    "the agent under test to have FAILED). You see only the reason, not the dialog.",
    TURN_RULE,
    "Rules:",
    "- A claim is something the reason still asserts the agent did wrong. Points the reason itself withdraws "
    "(\"the feedback clarifies the refusal was correct\", \"that is not the failure\") are NOT claims.",
    "- Each claim's `quote` must be copied exactly (character for character) from ia_reason; keep it short but "
    "self-explanatory (one clause or sentence).",
    "- Merge repeats of the same point into one claim.",
    "- If the reason asserts no concrete violation at all, return an empty list.",
    "Output: one JSON object per input line, same order, written as JSONL:",
    json.dumps({"dialog_id": "...", "claims": [{"quote": "exact substring of ia_reason", "kind": "turn_rule | other"}]}),
    "Write every line yourself; do not write a script that generates the labels.",
])

IA_NOTE = (
    "This dialog comes from IntellAgent, an external test generator used as a baseline. IntellAgent sampled a few "
    "policies, wrote the `scenario` and `expected_behaviour` from them, generated the initial database, and let an "
    "LLM user simulator talk to the agent; its own judge then claimed the agent violated the policy "
    "(`claims_to_check`, exact quotes of its reason). The scenario and expected behaviour can contradict the full "
    "policy or the generated database: judge ONLY by the full policy the agent was given (policy file) and by what "
    "the trajectory and tool outputs show. The trajectory lists user/assistant messages and every tool_call with "
    "its arguments and output, in time order; tool calls before an assistant message were made in that turn. "
    "Do NOT judge the turn rule (one tool call at a time / no reply while calling a tool) -- it is counted "
    "separately from data."
)

CLAIM_VERDICT_FORMAT = {
    "claim_index": "int, index in claims_to_check",
    "verdict": "confirmed | rejected | uncertain",
    "category": "agent_behavior if confirmed; otherwise the category that explains the claim (no_violation if the "
                "agent's behaviour was allowed, harness_artifact if the simulated user caused it, oracle_bug if the "
                "claim rests on a scenario/expected behaviour that contradicts the policy or the data)",
    "policy_quote": "for confirmed: exact sentence(s) of the policy that the agent broke; else null",
    "evidence": [{"message_index": "int, index in trajectory", "quote": "exact substring of that entry"}],
    "reason": "1-3 sentences",
}


def _load_dialogs() -> dict[str, dict]:
    with open(OUT / "dialogs.jsonl", encoding="utf-8") as fh:
        return {r["dialog_id"]: r for r in map(json.loads, fh)}


def _stable_sample(ids: list[str], n: int) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:n]


def stage_a_prepare(args) -> None:
    dialogs = _load_dialogs()
    ids = [i for i, r in dialogs.items() if r["ia_verdict"] == "violation"]
    if args.sample:
        per = args.sample // 2
        ids = sum((_stable_sample([i for i in ids if dialogs[i]["domain"] == d], per) for d in ("airline", "retail")), [])
    done = set()
    for path in (OUT / "stage_a" / "output").glob("*.jsonl"):
        done |= {json.loads(l)["dialog_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    ids = sorted(set(ids) - done)
    tag = args.tag
    out = OUT / "stage_a" / "input"
    out.mkdir(parents=True, exist_ok=True)
    (OUT / "stage_a" / "output").mkdir(parents=True, exist_ok=True)
    (OUT / "stage_a" / "INSTRUCTIONS.md").write_text(STAGE_A_INSTRUCTIONS + "\n", encoding="utf-8")
    chunks = [ids[k:k + args.chunk_size] for k in range(0, len(ids), args.chunk_size)]
    for k, chunk in enumerate(chunks):
        with open(out / f"{tag}_{k:02d}.jsonl", "w", encoding="utf-8") as fh:
            for i in chunk:
                r = dialogs[i]
                fh.write(json.dumps({"dialog_id": i, "domain": r["domain"], "ia_reason": r["ia_reason"]},
                                    ensure_ascii=False) + "\n")
    print(f"{len(ids)} reasons in {len(chunks)} chunks -> {out} (tag {tag})")


def stage_a_collect(_args) -> dict[str, list[dict]]:
    dialogs = _load_dialogs()
    claims, problems = {}, []
    for path in sorted((OUT / "stage_a" / "output").glob("*.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            i = row.get("dialog_id")
            if i not in dialogs:
                problems.append(f"{path.name}:{n} unknown dialog_id {i!r}")
                continue
            reason = str(dialogs[i]["ia_reason"])
            ok = []
            for c in row.get("claims") or []:
                if c.get("kind") not in CLAIM_KINDS:
                    problems.append(f"{i}: unknown kind {c.get('kind')!r}")
                elif not c.get("quote") or c["quote"] not in reason:
                    problems.append(f"{i}: quote not in reason: {str(c.get('quote'))[:80]!r}")
                else:
                    ok.append({"quote": c["quote"], "kind": c["kind"]})
            claims[i] = ok
    (OUT / "stage_a" / "claims.json").write_text(json.dumps(claims, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = Counter()
    for i, cs in claims.items():
        ks = {c["kind"] for c in cs}
        kinds["turn_rule_only" if ks == {"turn_rule"} else "has_other" if "other" in ks else "no_claim"] += 1
    print(json.dumps({"dialogs": len(claims), **kinds, "problems": len(problems)}, ensure_ascii=False))
    for p in problems[:30]:
        print("  ", p)
    return claims


def _stage_b_request(r: dict, other: list[dict]) -> dict:
    return {
        "schema_version": "agentspectesting.intellagent-review-request/v0.1",
        "request_id": r["dialog_id"], "domain": r["domain"], "batch": r["batch"], "event_id": r["event_id"],
        "claims_to_check": [c["quote"] for c in other],
        "scenario": r["scenario"], "expected_behaviour": r["expected_behaviour"],
        "initial_database": r["initial_database"],
        "transcript": r["trajectory"],
    }


def _stage_b_prompt(request: dict, policy_path: Path) -> str:
    fmt = {**VERDICT_FORMAT, "claim_verdicts": [CLAIM_VERDICT_FORMAT]}
    fmt["category"] = ("dialog level: agent_behavior if at least one claim is confirmed, else the category of the "
                       "rejected claims")
    return "\n\n".join([
        "You are auditing ONE dialog of a customer-service LLM agent (tau2-bench policy) that an external test "
        "tool judged as a policy violation. Be skeptical in both directions; use only what the trajectory, the "
        "tool outputs and the initial database show.",
        IA_NOTE,
        "For each entry of claims_to_check decide: confirmed (the agent really broke an explicit rule of the policy "
        "it was given, given what the user actually said and what the tools returned), rejected, or uncertain. "
        "Categories: " + json.dumps(CATEGORIES, ensure_ascii=False) + ". Unless the dialog-level category is "
        "agent_behavior or no_violation, set `outcome` (" + ", ".join(OUTCOMES) + ").",
        "A user simulator that invents facts, changes its goal, or ends early is harness_artifact. A claim that only "
        "holds because the scenario/expected behaviour says so, while the policy or the tool data say otherwise, is "
        "oracle_bug (outcome agent_correct). Every evidence quote must be copied exactly from the cited trajectory "
        "entry (content, arguments or output); every policy_quote exactly from the policy file.",
        "Use the same confirmation standard as the rest of this study (section 474):\n" + CONSENT_RULE + "\n"
        "- A reply such as 'yes', 'sure', 'go ahead', 'sounds good, do it' to a request to proceed after the details "
        "were listed is explicit confirmation.",
        f"The policy the agent was given is in {policy_path}.",
        "Case:\n" + json.dumps({k: v for k, v in request.items() if k != "schema_version"}, ensure_ascii=False,
                               indent=1, default=str),
        "Respond with ONLY one JSON object in this format:\n" + json.dumps(fmt, ensure_ascii=False, indent=1),
    ])


def stage_b_prepare(args) -> None:
    dialogs = _load_dialogs()
    claims = json.loads((OUT / "stage_a" / "claims.json").read_text(encoding="utf-8"))
    pending = OUT / "stage_b" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    (OUT / "stage_b" / "verdicts").mkdir(parents=True, exist_ok=True)
    n = 0
    for i, cs in sorted(claims.items()):
        other = [c for c in cs if c["kind"] == "other"]
        if not other:
            continue
        r = dialogs[i]
        request = _stage_b_request(r, other)
        policy_path = OUT / "policies" / f"{r['domain']}.md"
        (pending / f"{i}.json").write_text(json.dumps(request, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        (pending / f"{i}.prompt.md").write_text(_stage_b_prompt(request, policy_path), encoding="utf-8")
        n += 1
    print(f"{n} stage-B requests -> {pending}")


def _for_validation(request: dict) -> dict:
    """validate_verdict looks for a quote in an entry's `content` or its JSON dump; a tool_call's output is a
    JSON string, so a quote copied from it (with plain quote marks) only matches the raw text. Give every
    entry a `content` that holds all of its raw text fields (validation only; the prompt is unchanged)."""
    transcript = []
    for e in request["transcript"]:
        parts = [str(e.get("content") or "")]
        if e.get("role") == "tool_call":
            args = e.get("arguments")
            parts += [args if isinstance(args, str) else json.dumps(args, ensure_ascii=False), str(e.get("output") or ""),
                      str(e.get("name") or "")]
        transcript.append({**e, "content": "\n".join(parts)})
    return {**request, "transcript": transcript}


def _validate_stage_b(request: dict, verdict: dict, policy: str) -> dict:
    request = _for_validation(request)
    v = validate_verdict(request, verdict)
    problems = list(v["problems"])
    n_claims = len(request["claims_to_check"])
    seen = set()
    for cv in verdict.get("claim_verdicts") or []:
        k = cv.get("claim_index")
        if not isinstance(k, int) or not 0 <= k < n_claims:
            problems.append(f"claim_index {k!r} out of range")
            continue
        seen.add(k)
        if cv.get("verdict") not in ("confirmed", "rejected", "uncertain"):
            problems.append(f"claim {k}: unknown verdict {cv.get('verdict')!r}")
        if cv.get("verdict") == "confirmed":
            q = cv.get("policy_quote") or ""
            if not q or q not in policy:
                problems.append(f"claim {k}: policy_quote not found verbatim")
            sub = validate_verdict(request, {"category": "agent_behavior", "confidence": "high",
                                             "evidence": cv.get("evidence") or []})
            problems += [f"claim {k}: {p}" for p in sub["problems"]]
    if seen != set(range(n_claims)):
        problems.append(f"claims without a verdict: {sorted(set(range(n_claims)) - seen)}")
    confirmed = any(cv.get("verdict") == "confirmed" for cv in verdict.get("claim_verdicts") or [])
    if confirmed != (verdict.get("category") == "agent_behavior"):
        problems.append("dialog category inconsistent with claim verdicts")
    return {**v, "problems": problems}


PASS_SAMPLE_FRACTION = 0.15  # same stable 15% as the strategy-round review


def _pass_sample_ids(dialogs: dict) -> list[str]:
    ids = []
    for d in ("airline", "retail"):
        passes = [i for i, r in dialogs.items() if r["domain"] == d and r["ia_verdict"] == "pass"]
        ids += _stable_sample(passes, max(1, round(len(passes) * PASS_SAMPLE_FRACTION)))
    return sorted(ids)


def _pass_request(r: dict) -> dict:
    return {**_stage_b_request(r, []), "claims_to_check": [],
            "open_question": "IntellAgent judged this dialog as compliant. Did the agent violate any explicit rule of "
                             "its policy OTHER than the turn rule? If yes, list each violation as a claim in "
                             "claim_verdicts (claim_index = order you list them, verdict confirmed) and add "
                             "`claim_text` to each; if no, return claim_verdicts [] and category no_violation."}


def pass_sample_prepare(_args) -> None:
    dialogs = _load_dialogs()
    pending = OUT / "pass_sample" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    (OUT / "pass_sample" / "verdicts").mkdir(parents=True, exist_ok=True)
    ids = _pass_sample_ids(dialogs)
    for i in ids:
        r = dialogs[i]
        request = _pass_request(r)
        (pending / f"{i}.json").write_text(json.dumps(request, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        prompt = _stage_b_prompt(request, OUT / "policies" / f"{r['domain']}.md")
        prompt = prompt.replace("For each entry of claims_to_check decide", "Answer the case's open_question. For each "
                                "violation you find, decide")
        (pending / f"{i}.prompt.md").write_text(prompt, encoding="utf-8")
    print(f"{len(ids)} pass-sample requests -> {pending}")


def _pass_sample_rows(dialogs: dict, policies: dict) -> list[dict]:
    rows = []
    for i in _pass_sample_ids(dialogs):
        r = dialogs[i]
        path = OUT / "pass_sample" / "verdicts" / f"{i}.json"
        v = None
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            judge = raw.pop("judge", None)
            n = len(raw.get("claim_verdicts") or [])
            request = {**_pass_request(r), "claims_to_check": [cv.get("claim_text", "") for cv in raw.get("claim_verdicts") or []]}
            v = {**_validate_stage_b(request, raw, policies[r["domain"]]), "judge": judge, "n_found": n}
        status = ("pending" if v is None else "needs_review" if v["problems"]
                  else "missed_violation" if v.get("category") == "agent_behavior" else "no_other_violation")
        rows.append({"dialog_id": i, "domain": r["domain"], "reviewed_other": status, "review": v})
    return rows


def _read_verdict(path: Path, request: dict, policy: str) -> dict | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    judge = raw.pop("judge", None)
    return {**_validate_stage_b(request, raw, policy), "judge": judge}


def _claim_signature(verdict: dict) -> tuple:
    """What the two reviewers must agree on: confirmed or not, per claim (uncertain counts as its own value)."""
    return tuple(sorted((cv.get("claim_index"), cv.get("verdict")) for cv in verdict.get("claim_verdicts") or []))


def summarize(_args) -> None:
    dialogs = _load_dialogs()
    claims_path = OUT / "stage_a" / "claims.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8")) if claims_path.exists() else {}
    policies = {d: (OUT / "policies" / f"{d}.md").read_text(encoding="utf-8") for d in ("airline", "retail")}
    rows = []
    for i, r in sorted(dialogs.items()):
        cs = claims.get(i)
        other = [c for c in (cs or []) if c["kind"] == "other"]
        ia_turn = any(c["kind"] == "turn_rule" for c in (cs or []))
        verdict, second, agreement = None, None, None
        if other:
            request = _stage_b_request(r, other)
            verdict = _read_verdict(OUT / "stage_b" / "verdicts" / f"{i}.json", request, policies[r["domain"]])
            second = _read_verdict(OUT / "stage_b" / "verdicts_second" / f"{i}.json", request, policies[r["domain"]])
            adjudicated = _read_verdict(OUT / "stage_b" / "adjudicated" / f"{i}.json", request, policies[r["domain"]])
            if verdict and second:
                agreement = _claim_signature(verdict) == _claim_signature(second)
            if adjudicated:
                verdict = adjudicated
            elif second and not agreement:
                verdict = {**verdict, "problems": verdict["problems"] + ["first and second reviewer disagree"]}
        if r["ia_verdict"] == "pass":
            reviewed = "ia_pass"
        elif cs is None:
            reviewed = "claims_pending"
        elif not other:
            reviewed = "turn_rule_only" if ia_turn else "no_concrete_claim"
        elif verdict is None:
            reviewed = "needs_review"
        elif verdict["problems"]:
            reviewed = "needs_review"
        else:
            reviewed = reviewed_verdict("fail", verdict, True)
        rows.append({"dialog_id": i, "domain": r["domain"], "ia_verdict": r["ia_verdict"],
                     "parallel_calls": r["turn_rule_data"]["has_parallel_calls"], "ia_claims_turn_rule": ia_turn,
                     "n_other_claims": len(other), "reviewed_other": reviewed, "review": verdict,
                     "second_review": second, "reviewers_agree": agreement})
    pass_rows = _pass_sample_rows(dialogs, policies)
    turn = Counter((x["domain"], x["parallel_calls"]) for x in rows)
    other = Counter((x["domain"], x["reviewed_other"]) for x in rows)
    agree = Counter(x["reviewers_agree"] for x in rows if x["n_other_claims"])
    summary = {"turn_rule_parallel_calls": {f"{d}:{v}": n for (d, v), n in sorted(turn.items())},
               "other_violations_reviewed": {f"{d}:{v}": n for (d, v), n in sorted(other.items())},
               "second_reviewer_agreement": {str(k): n for k, n in agree.items()},
               "pass_sample": Counter(f"{x['domain']}:{x['reviewed_other']}" for x in pass_rows),
               "rows": rows, "pass_sample_rows": pass_rows}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    labels = {**REVIEWED_LABELS, "ia_pass": "IntellAgent judged pass (no violation reported)", "turn_rule_only": "Only the turn rule was reported",
              "no_concrete_claim": "No concrete violation in the rationale", "claims_pending": "Claims pending split"}
    lines = ["# IntellAgent Baseline Re-review", "", "## Turn rule (listed separately, by tool-call timestamps)", "",
             "| Domain | Has parallel calls | Does not |", "|---|---|---|"]
    for d in ("airline", "retail"):
        lines.append(f"| {d} | {turn[(d, True)]} | {turn[(d, False)]} |")
    lines += ["", "## Violations other than the turn rule (after re-review)", "", "| Domain | Verdict | Dialogs |", "|---|---|---|"]
    lines += [f"| {d} | {labels.get(v, v)} | {n} |" for (d, v), n in sorted(other.items())]
    lines += ["", "| Dialog | Verdict | Category | outcome | Note |", "|---|---|---|---|---|"]
    for x in rows:
        t = x["review"]
        if not t:
            continue
        note = (t.get("agent_violation") or t.get("explanation") or "").replace("|", "/").replace("\n", " ")
        if t.get("problems"):
            note = "⚠ " + "; ".join(t["problems"]) + " -- " + note
        lines.append(f"| {x['dialog_id']} | {x['reviewed_other']} | {t.get('category', '')} | {t.get('outcome', '')} | {note[:300]} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stage-a-prepare")
    a.add_argument("--sample", type=int, default=0, help="stable sample of N violation dialogs (half per domain)")
    a.add_argument("--chunk-size", type=int, default=60)
    a.add_argument("--tag", default="full")
    sub.add_parser("stage-a-collect")
    sub.add_parser("stage-b-prepare")
    sub.add_parser("pass-sample-prepare")
    sub.add_parser("summarize")
    args = ap.parse_args()
    {"stage-a-prepare": stage_a_prepare, "stage-a-collect": stage_a_collect,
     "stage-b-prepare": stage_b_prepare,
     "pass-sample-prepare": pass_sample_prepare, "summarize": summarize}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
