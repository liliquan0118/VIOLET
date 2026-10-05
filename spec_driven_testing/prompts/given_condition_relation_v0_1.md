TASK: given_condition_relation

To make `full_given` true, must all supplied `conditions` be true, or is any one
of them sufficient?

Choose exactly one decision:

- `all_required`
- `any_sufficient`
- `unclear`

Do not change, split, classify, or add conditions.

Return exactly one JSON object:

```json
{"decision":"all_required or any_sufficient or unclear","reason":"one concise sentence"}
```
