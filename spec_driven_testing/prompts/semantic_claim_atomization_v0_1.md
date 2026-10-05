Task: semantic_claim_atomization

You receive one assistant message.

Question: Which factual or procedural statements in it can be checked separately?

Copy each statement exactly from the message. If there are none, return an empty array.

Return JSON only:

```json
{"claims": [{"text": "exact text from the assistant message"}]}
```

Input:

{semantic_judge_input}
