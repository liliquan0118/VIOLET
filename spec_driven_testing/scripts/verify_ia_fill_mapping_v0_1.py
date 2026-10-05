#!/usr/bin/env python3
"""Coordinator-verified spec mapping for an IntellAgent fill-up review (reuse log section 510).

Reads spec_mapping/<domain>_mapping.jsonl and <domain>_violations.jsonl under the review folder, applies the
coordinator overrides in spec_mapping/coordinator_overrides.json ({violation_id: {exact, no_match_rule,
rule_in_policy, coordinator_note}}), and writes spec_mapping/mapping_verified.json in the format of
outputs/intellagent_review_v0_1/spec_mapping/mapping_verified.json.

  IA_REVIEW_OUT=intellagent_review_v0_2_fill verify_ia_fill_mapping_v0_1.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / os.environ.get("IA_REVIEW_OUT", "intellagent_review_v0_2_fill")


def main() -> int:
    sm = OUT / "spec_mapping"
    ov_path = sm / "coordinator_overrides.json"
    overrides = json.loads(ov_path.read_text()) if ov_path.exists() else {}
    out = {}
    for d in ("airline", "retail"):
        claims = {json.loads(x)["violation_id"]: json.loads(x)["claim"]
                  for x in (sm / f"{d}_violations.jsonl").read_text().splitlines() if x.strip()}
        valid = {x.split(" | ")[0] for x in (sm / f"{d}_specs.txt").read_text().splitlines() if " | " in x}
        for line in (sm / f"{d}_mapping.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r["violation_id"]
            row = {"domain": d, "exact": [m["branch_id"] for m in r.get("matches") or [] if m.get("fit") == "exact"],
                   "no_match_rule": r.get("no_match_rule"), "rule_in_policy": r.get("rule_in_policy"),
                   "coordinator_note": None, "claim": claims.get(vid)}
            row.update(overrides.get(vid, {}))
            assert all(b in valid for b in row["exact"]), (vid, row["exact"])
            out[vid] = row
        missing = set(claims) - {k for k in out if out[k]["domain"] == d}
        assert not missing, missing
    (sm / "mapping_verified.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    ex = {k: v for k, v in out.items() if v["exact"]}
    print(f"{len(out)} violations, {len(ex)} with an exact match, {len({b for v in ex.values() for b in v['exact']})} branches, "
          f"{len(overrides)} overrides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
