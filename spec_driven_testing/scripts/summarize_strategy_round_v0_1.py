#!/usr/bin/env python3
"""Summarize a strategy round after the violation check (reuse log section 241).

Reads <run-dir>/violation_check/final_verdicts.json (check_violations_v0_1.py),
the round's cases.json and the inventory classifications, and writes
<run-dir>/violation_check/round_summary.{json,md}:
  - per case: trigger / control violations over reached runs;
  - by strategy, rule_source (policy / tool_doc / domain_knowledge), domain,
    tool_gap;
  - cases whose runs got different verdicts across reps or were judged
    case_invalid (for coordinator verification).
A case "violates" when at least one reviewed trigger run is a violation;
control-arm violations are reported separately (the rule was broken without
the trigger).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--inventory", required=True, type=Path)
    args = ap.parse_args()
    ov_path = args.run_dir / "violation_check" / "coordinator_overrides.json"
    overrides = json.loads(ov_path.read_text(encoding="utf-8")) if ov_path.exists() else {}

    final = json.loads((args.run_dir / "violation_check" / "final_verdicts.json").read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in json.loads(args.cases.read_text(encoding="utf-8"))}
    cls = json.loads((args.inventory / "classifications.json").read_text(encoding="utf-8"))
    per = defaultdict(lambda: {"trigger": Counter(), "control": Counter(), "violations": []})
    for key, v in final.items():
        r = v["record"]
        per[r["case"]][r["arm"]][v["final"] or "pending"] += 1
        if v["final"] == "violation":
            per[r["case"]]["violations"].append({"run": key, "arm": r["arm"], "violation": v.get("violation")})
    rows = []
    for cid, p in sorted(per.items()):
        c = cases[cid]
        spec = cls.get(c["spec_id"], {})
        t, k = p["trigger"], p["control"]
        rows.append({
            "case_id": cid, "spec_id": c["spec_id"], "strategy_id": c["strategy_id"], "domain": c["domain"],
            "rule_source": (overrides.get("rule_source", {}).get(cid) or {}).get("to") or spec.get("rule_source"),
            "contested": cid in overrides.get("contested", {}), "tool_gap": bool(spec.get("tool_gap")),
            "u9": "U9" in {b.get("code") for b in spec.get("blockers") or []},
            "trigger": dict(t), "control": dict(k),
            "trigger_violations": t["violation"], "trigger_reached": t["violation"] + t["compliant"],
            "control_violations": k["violation"], "control_reached": k["violation"] + k["compliant"],
            "mixed": len({x for x in t if x != "pending"}) > 1 or t["case_invalid"] > 0,
            "violations": p["violations"]})

    def agg(key):
        out = defaultdict(lambda: Counter())
        for r in rows:
            g = out[r[key]]
            g["cases"] += 1
            g["cases_violating"] += r["trigger_violations"] > 0
            g["trigger_viol"] += r["trigger_violations"]
            g["trigger_reached"] += r["trigger_reached"]
            g["control_viol"] += r["control_violations"]
            g["control_reached"] += r["control_reached"]
        return {str(k): dict(v) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}

    summary = {
        "cases": len(rows),
        "cases_with_trigger_violation": sum(r["trigger_violations"] > 0 for r in rows),
        "contested": sorted(overrides.get("contested", {})),
        "cases_with_control_violation": sum(r["control_violations"] > 0 for r in rows),
        "runs": {"trigger": dict(sum((Counter(r["trigger"]) for r in rows), Counter())),
                 "control": dict(sum((Counter(r["control"]) for r in rows), Counter()))},
        "by_rule_source": agg("rule_source"), "by_domain": agg("domain"), "by_strategy": agg("strategy_id"),
        "by_tool_gap": agg("tool_gap"), "by_u9": agg("u9"),
        "to_verify": [r["case_id"] for r in rows if r["mixed"] or r["control_violations"]],
        "cases_detail": rows,
    }
    out = args.run_dir / "violation_check"
    (out / "round_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    def table(title, data):
        lines = [f"## {title}", "", "| group | cases | cases with a violation | trigger violations/reached | control violations/reached |", "|---|---|---|---|---|"]
        for k, v in data.items():
            lines.append(f"| {k} | {v['cases']} | {v['cases_violating']} | {v['trigger_viol']}/{v['trigger_reached']} | "
                         f"{v['control_viol']}/{v['control_reached']} |")
        return lines + [""]

    md = [f"# Strategy round results (after review): {args.run_dir.parent.name}", "",
          f"{summary['cases']} cases; {summary['cases_with_trigger_violation']} with a trigger violation; "
          f"{summary['cases_with_control_violation']} with a control violation as well.", "",
          f"Run verdicts: trigger {json.dumps(summary['runs']['trigger'], ensure_ascii=False)}; "
          f"control {json.dumps(summary['runs']['control'], ensure_ascii=False)}", ""]
    md += table("By rule source", summary["by_rule_source"]) + table("By domain", summary["by_domain"])
    md += table("Enforced by the tool? (tool_gap=True: not enforced)", summary["by_tool_gap"]) + table("Spec itself wrong (U9, tested against the real rule)", summary["by_u9"])
    md += table("By strategy", summary["by_strategy"])
    md += ["## Cases with a violation", "", "| case | source | trigger | control | violation (one example) |", "|---|---|---|---|---|"]
    for r in rows:
        if r["trigger_violations"] or r["control_violations"]:
            ex = (r["violations"][0]["violation"] or "").replace("|", "/")[:160]
            md.append(f"| {r['case_id']}{' (contested)' if r['contested'] else ''} | {r['rule_source']} | {r['trigger_violations']}/{r['trigger_reached']} | "
                      f"{r['control_violations']}/{r['control_reached']} | {ex} |")
    md += ["", "## To verify", ""] + [f"- {c}" for c in summary["to_verify"]]
    (out / "round_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"ROUND cases={summary['cases']} violating={summary['cases_with_trigger_violation']} "
          f"control_violating={summary['cases_with_control_violation']} to_verify={len(summary['to_verify'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
