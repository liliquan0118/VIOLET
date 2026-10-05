"""Filter a domain's v5 artifacts down to the specs the testability audit kept.

Takes the full artifact set (formalize outputs + grounding + conversation side
+ entity matches) and the `_testable.json` spec file written by
scripts/filter_testability.py, removes every withheld spec and its GWT
records from each file, and writes the result into <dir>/testable_v5/ with a
FILTER_MANIFEST.json recording what was removed and why. Originals untouched.

Usage:
    python scripts/build_testable.py --domain retail \
        --specs ExperimentResult/spec_v2/retail/specs_retail_gpt41_full_run2.json \
        --prefix ExperimentResult/spec_v2/retail/specs_retail_gpt41_full_run2_gwt_v5
"""
from __future__ import annotations

import argparse
import collections
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--specs", required=True, help="the mined spec file")
    p.add_argument("--prefix", required=True, help="path prefix of the _gwt_v5* artifacts")
    p.add_argument("--out-dir", default=None, help="default: <prefix dir>/testable_v5")
    a = p.parse_args()

    base = a.specs.replace(".json", "")
    full = json.load(open(a.specs))
    testable = json.load(open(base + "_testable.json"))
    audits = {x["spec_id"]: x for x in json.load(open(base + "_testability.json"))["audits"]}
    kept = {s["spec_id"] for s in testable["specs"]}
    removed = [s["spec_id"] for s in full["specs"] if s["spec_id"] not in kept]

    out_dir = a.out_dir or os.path.join(os.path.dirname(a.prefix), "testable_v5")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(a.prefix)
    report = {}
    for suffix, key in ((".json", "specs"), ("_pos.json", "specs"), ("_neg.json", "specs"),
                        ("_ir.json", "specs"), ("_classification.json", "classifications")):
        d = json.load(open(a.prefix + suffix))
        before = len(d[key])
        gb = sum(len(x.get("gwt") or []) for x in d[key]) if key == "specs" else None
        d[key] = [x for x in d[key] if x["spec_id"] not in removed]
        ga = sum(len(x.get("gwt") or []) for x in d[key]) if key == "specs" else None
        if isinstance(d.get("metadata"), dict):
            d["metadata"] = {**d["metadata"], "testability_filter": {"removed": removed}}
        json.dump(d, open(os.path.join(out_dir, name + suffix), "w"), indent=2, ensure_ascii=False)
        report[suffix] = {"specs": f"{before}->{len(d[key])}",
                          **({"gwts": f"{gb}->{ga}"} if gb is not None else {})}
    for suffix in ("_conditions.json", "_user_requirements.json", "_conditions_matches.json"):
        rows = json.load(open(a.prefix + suffix))
        out = [r for r in rows if r["spec_id"] not in removed]
        json.dump(out, open(os.path.join(out_dir, name + suffix), "w"), indent=2, ensure_ascii=False)
        report[suffix] = {"records": f"{len(rows)}->{len(out)}"}
    json.dump({"domain": a.domain, "source_specs": a.specs, "prefix": a.prefix,
               "removed": [{"spec_id": s, "verdict": audits.get(s, {}).get("verdict"),
                            "reason": (audits.get(s, {}).get("reason") or "")[:300]} for s in removed],
               "files": report},
              open(os.path.join(out_dir, "FILTER_MANIFEST.json"), "w"), indent=2, ensure_ascii=False)
    # consistency across the filtered set
    gw = {(x["spec_id"], i) for x in json.load(open(os.path.join(out_dir, name + ".json")))["specs"]
          for i, _ in enumerate(x.get("gwt") or [])}
    for suffix in ("_conditions.json", "_user_requirements.json", "_conditions_matches.json"):
        ks = {(r["spec_id"], r["gwt_index"]) for r in json.load(open(os.path.join(out_dir, name + suffix)))}
        assert ks == gw, f"{suffix}: keys differ from the GWT file"
    print(f"{a.domain}: removed {len(removed)} specs "
          f"({dict(collections.Counter(audits.get(s, {}).get('verdict') for s in removed))}); "
          f"GWTs {report['.json']['gwts']} -> {out_dir}")


if __name__ == "__main__":
    main()
