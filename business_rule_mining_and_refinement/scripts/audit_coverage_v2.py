"""Run the v2 coverage audit on an EXISTING mined spec set (no re-extraction).

Segments the policy in code, asks one small claim call per spec, verifies every
claim against the text, and adjudicates the unquoted occurrences WITHOUT
modifying the spec set. Writes <specs>_prompt_audit_v2.json in the same shape
as the miner's audit file so the same audit scripts read both.

Usage:
    python scripts/audit_coverage_v2.py --domain telecom \
        --specs ExperimentResult/spec_v2/telecom/specs_telecom_gpt41_full_run2.json
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_interface.tau2_bench import load_tau2_bench_agent
from src.mining_v2.common import call_json, tool_context
from src.mining_v2.prompt_miner_v2 import _ADJ_SYSTEM, _ADJ_USER, audit_coverage
from src.mining_v2.spec import SpecSet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["airline", "retail", "telecom"], required=True)
    p.add_argument("--specs", required=True)
    p.add_argument("--tau-bench-path", default="tau2-bench")
    p.add_argument("--model", default="gpt-4.1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-adjudicate", action="store_true", help="only mark quoted/unquoted")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    agent = load_tau2_bench_agent(a.tau_bench_path, a.domain)
    specs = [s for s in SpecSet.load(a.specs).specs if s.origin == "prompt"]
    sentences, rejected = audit_coverage(agent, specs, model=a.model)
    unquoted = [s for s in sentences if not s["quoted"]]
    policy = agent.system_prompt or ""
    last_line = [l for l in policy.splitlines() if l.strip()][-1].strip()
    print(f"{a.domain}: occurrences {len(sentences)} (last = policy's last line: "
          f"{sentences[-1]['text'].strip() == last_line}) | quoted {len(sentences) - len(unquoted)} "
          f"| unquoted {len(unquoted)} | claims rejected {len(rejected)}")

    verdicts: list[dict] = []
    if not a.no_adjudicate:
        tl = tool_context(agent)
        block = json.dumps([{"spec_id": s.spec_id, "kind": s.kind, "rule_text": s.rule_text,
                             "target_action": s.target_action, "bindings": s.bindings,
                             "quotes": [e.quote for e in s.evidence]} for s in specs],
                           ensure_ascii=False, indent=1)
        system = _ADJ_SYSTEM.format(domain=a.domain)

        def judge(c):
            user = _ADJ_USER.format(domain=a.domain, tool_list=tl, specs=block, policy=policy,
                                    text=c["text"], anchor=c["anchor"], occ=c["occ"] or "unique")
            v = call_json(system, user, a.model, 0.0, max_tokens=1500, label="coverage_adjudicate")
            return {"idx": c["idx"], "text": c["text"], "anchor": c["anchor"], "occ": c["occ"],
                    "verdict": (v.get("verdict") or "?").strip(),
                    "spec_id": (v.get("spec_id") or "").strip(),
                    "reason": (v.get("reason") or "").strip(), "applied": False}
        with ThreadPoolExecutor(a.workers) as pool:
            verdicts = list(pool.map(judge, unquoted))
        print("  verdicts:", dict(collections.Counter(v["verdict"] for v in verdicts)))

    out = a.specs.replace(".json", "") + "_prompt_audit_v2.json"
    json.dump({"audit": "v2", "domain": a.domain, "model": a.model,
               "sentences": sentences, "verdicts": verdicts, "rejected_claims": rejected},
              open(out, "w"), indent=2, ensure_ascii=False)
    print("  ->", out)


if __name__ == "__main__":
    main()
