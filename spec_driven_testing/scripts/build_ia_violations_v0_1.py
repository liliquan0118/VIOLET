#!/usr/bin/env python3
"""Build the spec-mapping input from a reviewed IntellAgent run (reuse log section 510, fill-up batches).

For every dialog reviewed in stage B, the final verdict is the adjudicated one when present, else the first
reviewer's (the reviewers agreed). Each claim with verdict `confirmed` becomes one violation line, in the format of
outputs/intellagent_review_v0_1/spec_mapping/<domain>_violations.jsonl. Spec files are copied from the old run.

  IA_REVIEW_OUT=intellagent_review_v0_2_fill build_ia_violations_v0_1.py
"""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / os.environ.get("IA_REVIEW_OUT", "intellagent_review_v0_2_fill")
OLD = ROOT / "outputs" / "intellagent_review_v0_1" / "spec_mapping"


def main() -> int:
    sb = OUT / "stage_b"
    sm = OUT / "spec_mapping"
    sm.mkdir(exist_ok=True)
    for d in ("airline", "retail"):
        shutil.copy(OLD / f"{d}_specs.txt", sm / f"{d}_specs.txt")
    shutil.copy(OLD / "INSTRUCTIONS.md", sm / "INSTRUCTIONS.md")
    lines = {"airline": [], "retail": []}
    c = Counter()
    for p in sorted((sb / "pending").glob("*.json")):
        did = p.stem
        req = json.loads(p.read_text())
        final = sb / "adjudicated" / f"{did}.json"
        if not final.exists():
            final = sb / "verdicts" / f"{did}.json"
        v = json.loads(final.read_text())
        claims = req.get("claims_to_check") or []
        for cv in v.get("claim_verdicts") or []:
            if cv.get("verdict") != "confirmed":
                continue
            i = cv.get("claim_index", 0)
            claim = claims[i] if i < len(claims) else None
            text = claim.get("quote") if isinstance(claim, dict) else claim
            lines[req["domain"]].append({
                "violation_id": f"{did}#c{i}", "domain": req["domain"], "claim": text,
                "reviewer_reason": cv.get("reason") or v.get("explanation"),
                "policy_quote": cv.get("policy_quotes") or cv.get("policy_quote"), "evidence": cv.get("evidence") or v.get("evidence"),
                "scenario": req.get("scenario")})
            c[req["domain"]] += 1
    for d, ls in lines.items():
        (sm / f"{d}_violations.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ls))
    print(dict(c), {d: len({x["violation_id"].split("#")[0] for x in ls}) for d, ls in lines.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
