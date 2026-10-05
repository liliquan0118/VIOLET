#!/usr/bin/env python3
"""Import two v5 upstream artifacts and report source quality without executing tests."""

import argparse
import json
from pathlib import Path

from agentest.compiler.v5_source_intake_v1 import intake_files, audit_view, render_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-requirements", type=Path, required=True)
    parser.add_argument("--condition-matches", type=Path, required=True)
    parser.add_argument("--v3-reference", type=Path, default=None,
        help="Optional prior-generation spec file, only used to render a version-drift diff report. "
             "A brand-new domain with no real prior generation should omit this rather than supply a "
             "fake placeholder -- intake_status/summary/review_reasons never depend on it.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty; use a new directory to preserve prior intake evidence")
    try:
        result = intake_files(args.user_requirements, args.condition_matches, args.v3_reference)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("intake.json", result), ("audit.json", audit_view(result))):
        (args.output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "review.md").write_text(render_review(result), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
