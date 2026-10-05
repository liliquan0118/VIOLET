Task: semantic_claim_support

You receive one claim and a closed list of allowed sources.

Question: Is the claim supported by at least one supplied source?

Use only the supplied sources, not outside knowledge.

Return `supported` when a source supports the claim, and `unsupported` when none does. Use `insufficient` only when the claim or source cannot be interpreted reliably. For `supported`, copy one exact supporting span and its source ID. Otherwise, set `evidence` to null.

Return JSON only:

```json
{"verdict": "<supported|unsupported|insufficient>", "evidence": {"source_id": "...", "text": "exact source span"}, "reason": "short reason"}
```

Input:

{semantic_judge_input}
