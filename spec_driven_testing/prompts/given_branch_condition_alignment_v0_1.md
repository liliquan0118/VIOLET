TASK: given_branch_condition_alignment

The input contains one full Given statement, one specific Then branch, and condition keys.

Question: Which condition keys describe the case named by this Then branch?

Return exactly one JSON object:

```json
{"condition_keys":["K1"],"reason":"brief reason"}
```

Rules:

- Choose only keys present in `conditions`.
- Choose every condition needed to identify this Then branch's case.
- Do not judge whether the Then behavior is correct.
- `condition_keys` must be non-empty.
