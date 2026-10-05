# Binary semantic boundary decision v0.6

## One question

Does the cue enclosed by `⟦` and `⟧` connect two claims that could have
different truth values?

Answer:

- `yes`: the two sides make separate claims, so one could be true while the
  other is false;
- `no`: at least one side is only part of the same phrase or value list;
- `ambiguous`: the complete sentence does not determine this.

Do not find another boundary. Do not classify roles, modality, or relations.

Return JSON only:

```json
{
  "target_id": "copy from input",
  "boundary_id": "copy from input",
  "decision": "yes | no | ambiguous",
  "reason": "One short sentence."
}
```

## Input

```json
{boundary_decision_input}
```
