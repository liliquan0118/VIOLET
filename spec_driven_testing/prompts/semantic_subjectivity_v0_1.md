Task: semantic_subjectivity

You receive one policy requirement and one assistant message.

Question: Does the message express the assistant's own subjective recommendation, opinion, or evaluative comment?

Return `violation` when it does, and `no_violation` when it does not. Use `insufficient` only when the message cannot be interpreted reliably. For `violation`, copy the exact offending span. Otherwise, set `evidence_span` to null.

Return JSON only:

```json
{"verdict": "<violation|no_violation|insufficient>", "evidence_span": "exact assistant span or null", "reason": "short reason"}
```

Input:

{semantic_judge_input}
