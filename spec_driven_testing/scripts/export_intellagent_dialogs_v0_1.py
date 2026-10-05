#!/usr/bin/env python3
"""Export the valid IntellAgent baseline dialogs for review (reuse log section 472).

Runs in the `intellagent` conda env (the dataset pickles need IntellAgent's own
classes) from the IntellAgent checkout, and writes plain JSON into this project:

  cd $INTELLAGENT_DIR
  <python of the intellagent environment> <this repo>/scripts/export_intellagent_dialogs_v0_1.py

Output (outputs/intellagent_review_v0_1/):
  dialogs.jsonl        one record per valid dialog (leaked-thought dialogs excluded,
                       same rule as tools/count_valid.py)
  policies/<domain>.md the system prompt the agent under test was given

Each record keeps IntellAgent's own verdict (score 0 = violation) and final reason,
the scenario and expected behaviour IntellAgent generated, the event's initial
database, and the trajectory: user/agent messages and every tool call with its
output, in time order. IntellAgent stores an agent message after the tool calls of
its turn, so tool calls between two user messages belong to the agent turn that
follows them. Calls less than 1 s apart are one `call_group` (the agent issued them
together in one step; a sequential call needs a model round trip of seconds), which
is the data-side evidence for the turn rule ("one tool call at a time").
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

IA = Path(os.environ.get("INTELLAGENT_DIR", "."))  # IntellAgent checkout
RESULTS = IA / "results"
# section 506 (Qwen rerun): IA_RUN_KIND=qwen reads results/<domain>_qwen_<tag>, IA_REVIEW_OUT names the output dir
RUN_KIND = os.environ.get("IA_RUN_KIND", "full")
OUT = Path(__file__).resolve().parents[1] / "outputs" / os.environ.get("IA_REVIEW_OUT", "intellagent_review_v0_1")
VALID_TAGS = ["b1", "b2r", "b3", "b4", "b5", "b6", "b7", "b8", "b9"]  # b2 aborted, excluded
# section 510 (fill-up to all-runs parity): IA_TAGS="b10 ... b17" exports only these batches into a separate review dir
if os.environ.get("IA_TAGS"):
    VALID_TAGS = os.environ["IA_TAGS"].split()
DOMAINS = ("airline", "retail")
PARALLEL_GAP_MS = 1000
LEAK = re.compile(r"\bThought\s*:|User Response\s*:")


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _trajectory(con: sqlite3.Connection, thread_id: str) -> tuple[list[dict], list[list[int]]]:
    items = [("msg", int(t), role, msg) for role, msg, t in
             con.execute("select role, message, time from Dialog where thread_id=?", (thread_id,))]
    items += [("tool", int(t), name, (inp, out)) for name, inp, out, t in
              con.execute("select tool_name, input, output, time from Tools where thread_id=?", (thread_id,))]
    items.sort(key=lambda x: x[1])
    trajectory, groups, last_tool_t, group = [], [], None, None
    for kind, t, a, b in items:
        if kind == "msg":
            role = "user" if a == "Human" else "assistant"
            trajectory.append({"index": len(trajectory), "role": role, "content": b, "t_ms": t})
            last_tool_t, group = None, None
            continue
        try:
            arguments = json.loads(b[0])
        except (TypeError, ValueError):
            arguments = b[0]
        if last_tool_t is None or t - last_tool_t >= PARALLEL_GAP_MS:
            group = len(groups)
            groups.append([])
        groups[group].append(len(trajectory))
        trajectory.append({"index": len(trajectory), "role": "tool_call", "name": a, "arguments": arguments,
                           "output": b[1], "t_ms": t, "call_group": group})
        last_tool_t = t
    return trajectory, groups


def main() -> int:
    sys.path.insert(0, str(IA))
    leaked = json.load(open(RESULTS / "leaked_thought_dialogs.json"))["threads"]
    (OUT / "policies").mkdir(parents=True, exist_ok=True)
    n = {d: 0 for d in DOMAINS}
    with open(OUT / "dialogs.jsonl", "w", encoding="utf-8") as fh:
        for domain in DOMAINS:
            for tag in VALID_TAGS:
                if not (RESULTS / f"{domain}_{RUN_KIND}_{tag}").is_dir():  # b9 is airline only
                    continue
                exp = glob.glob(str(RESULTS / f"{domain}_{RUN_KIND}_{tag}" / "experiments" / "*") + "/")
                if not exp or not Path(exp[0], "results.csv").exists():
                    continue
                exp = Path(exp[0])
                policy = (exp / "prompt.txt").read_text(encoding="utf-8")
                policy_path = OUT / "policies" / f"{domain}.md"
                if not policy_path.exists():
                    policy_path.write_text(policy, encoding="utf-8")
                elif policy_path.read_text(encoding="utf-8") != policy:
                    raise SystemExit(f"{domain}_{tag}: agent prompt differs from earlier batches")
                results = pd.read_csv(exp / "results.csv")
                events = pickle.load(open(glob.glob(str(RESULTS / f"{domain}_{RUN_KIND}_{tag}" / "datasets" / "*.pickle"))[0], "rb"))[0]
                events = {e.id: e for e in events}
                con = sqlite3.connect(exp / "memory.db")
                leak = {t for t, m in con.execute("select thread_id, message from Dialog where role='Human'")
                        if LEAK.search(m or "")}
                bad = set(leaked.get(f"{domain}_{RUN_KIND}_{tag}", [])) | leak
                for row in results.itertuples():
                    if row.thread_id in bad:
                        continue
                    event = events[row.id]
                    if str(event.scenario)[:200] != str(row.scenario)[:200]:
                        raise SystemExit(f"{domain}_{tag} id {row.id}: scenario does not match its event")
                    trajectory, groups = _trajectory(con, row.thread_id)
                    parallel = [g for g in groups if len(g) > 1]
                    record = {
                        "dialog_id": f"{domain}_{tag}_{int(row.id):03d}",
                        "domain": domain, "batch": tag, "event_id": int(row.id), "thread_id": row.thread_id,
                        "ia_verdict": "violation" if int(row.score) == 0 else "pass",
                        "ia_reason": row.reason,
                        "scenario": row.scenario, "expected_behaviour": row.expected_behaviour,
                        "initial_database": {k: _records(v) for k, v in event.database.items()},
                        "trajectory": trajectory,
                        "turn_rule_data": {"tool_calls": sum(len(g) for g in groups),
                                           "parallel_call_groups": [[trajectory[i]["name"] for i in g] for g in parallel],
                                           "has_parallel_calls": bool(parallel)},
                    }
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    n[domain] += 1
    print(json.dumps(n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
