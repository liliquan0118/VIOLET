# Oracle requirement membership v0.8

## One question

Does the oracle need to check `candidate_observation` to evaluate
`then_requirement`?

- `yes`: it is one part of the behavior or result judged by the Then. It does
  not need to be sufficient by itself.
- `no`: it is not part of what this Then judges.
- `ambiguous`: these two statements are insufficient to decide.

Judge only behavior explicitly stated by this Then. A behavior that could
overlap, cause, or co-occur with it is not enough for `yes`.

Return JSON only:

```json
{
  "decision": "yes | no | ambiguous",
  "reason": "One short sentence."
}
```

## Input

```json
{oracle_decision_input}
```
