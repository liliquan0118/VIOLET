# Oracle requirement selection v0.7

## One question

Is `proposed_observation` one of the things the oracle should check for this
GWT Then?

- `yes`: it is one part of the agent behavior judged by Then. It does not need
  to be sufficient by itself.
- `no`: it is not part of the agent behavior judged by Then; for example, it
  is only test setup from Given/When.
- `ambiguous`: the supplied branch and candidate are insufficient to decide.

Use the current Then as the authority. Judge only this candidate. Do not create
another observation and do not choose an evaluator.

Return JSON only:

```json
{
  "branch_id": "copy from branch",
  "candidate_id": "copy from proposed_observation",
  "decision": "yes | no | ambiguous",
  "reason": "One short sentence."
}
```

## Input

```json
{oracle_requirement_input}
```
