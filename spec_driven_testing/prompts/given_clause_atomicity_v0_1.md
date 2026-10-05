TASK: given_clause_atomicity

Does `source_clause` express exactly one independently truth-valued condition,
or does it combine two or more independently truth-valued conditions?

Choose exactly one decision:

- `single_condition`: one condition, even if it contains descriptive modifiers.
- `multiple_conditions`: two or more conditions that could be true or false separately.
- `unclear`: the wording does not make this decidable.

Do not split the clause and do not decide where its evidence comes from.

Return exactly one JSON object:

```json
{"decision":"single_condition or multiple_conditions or unclear","reason":"one concise sentence"}
```
