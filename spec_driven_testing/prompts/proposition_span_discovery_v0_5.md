# Proposition Span Discovery v0.5

## The only question

Which exact substrings of `primary_text` express proposition candidates that
can be judged true or false separately?

A proposition candidate is one action, event, state, condition, comparison, or
scope fact. This task only finds its literal boundary. Do not decide what role
it plays or whether it is required, prohibited, or permitted.

## Rules

- Copy every answer exactly from `primary_text`. Never paraphrase, normalize,
  remove negation, or add words.
- Include both the governed/asserted outcome and propositions mentioned as its
  conditions, temporal endpoints, or scope. In `X if Y`, return both `X` and
  `Y`, not only `Y`.
- Split two candidates when either one could be true while the other is false.
- Keep one comparison or one allowed-value set together. For example,
  `flight_type is either 'round_trip' or 'one_way'` is one proposition, not two.
- Keep every qualifier that changes meaning. For example, never rewrite
  `information not provided by the user or tools` as `information`.
- Return spans in textual order. Do not generate IDs.
- Use `ambiguous` and an empty list only when the literal boundaries genuinely
  cannot be chosen from the supplied sentence.

Examples:

- From `The agent may cancel the flight if the booking was made within 24
  hours`, the candidate list must include an exact span for the cancellation
  and an exact span for `the booking was made within 24 hours`.
- From `The agent must not provide information not provided by the user or
  tools`, preserve the two occurrences of `not`; do not create a positive
  paraphrase.

## Output

Return JSON only:

```json
{
  "target_id": "copy target_id from the input",
  "resolution_status": "resolved",
  "proposition_spans": ["exact substring from primary_text"],
  "reason": "Briefly explain why these are the complete literal boundaries."
}
```

The only valid `resolution_status` values are `resolved` and `ambiguous`.

## Input

```json
{proposition_span_discovery_input}
```
