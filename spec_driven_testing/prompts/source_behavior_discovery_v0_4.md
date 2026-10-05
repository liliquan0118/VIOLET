# Source Behavior Discovery

## One question only

Which independently assessable **positive behaviors or states** are governed by
this source anchor?

Do not decide modality, polarity, activation conditions, biconditional reverse
cases, GWT support, relations, or test observability. Do not generate IDs.

## Rules

- Write each candidate in positive form without `must`, `may`, `should`, or
  `not`. Example: from “must not cancel”, return `cancel the reservation`.
- Split candidates when one can be true while the other is false.
- A value/type/quantity constraint that can fail independently is its own
  state candidate.
- Do not return an event mentioned only as the condition or scope of another
  governed behavior.
- Preserve one allowed-value set or one comparison as a unit.
- Cite exact substrings of the supplied source. The normalized positive
  candidate itself may paraphrase those spans.
- Return `ambiguous` with an empty list only when the behavior boundaries
  themselves cannot be determined.

## Output

```json
{
  "spec_id": "airline_example",
  "resolution_status": "resolved",
  "behaviors": [
    {
      "behavior": "One positive independently assessable behavior or state.",
      "evidence_spans": ["exact source substring"]
    }
  ],
  "reason": "Why this list contains all and only governed behaviors/states."
}
```

Return candidates in source order. Do not add identifiers; the compiler assigns
stable IDs after validation.

## Input

```json
{source_behavior_discovery_input}
```
