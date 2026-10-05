# Then Entity Discovery

## One question only

Which independently assessable **positive event or state candidates** are
mentioned in this Then?

Do not decide whether a candidate is asserted or reference-only. Do not decide
modality, polarity, source support, relations, or observability. Do not generate
IDs.

## Rules

- Write every candidate in positive form without `must`, `may`, `should`, or
  `not`. Example: from “must not cancel”, return `cancel the reservation`.
- Include both governed outcomes and events mentioned only as conditions,
  scope anchors, or `before`/`after` endpoints. Role classification happens in
  a later task.
- Split candidates when one can be true while another is false.
- A parameter's presence and its independently violable type/value/quantity
  constraint are separate candidates.
- Keep one allowed-value set or one comparison together.
- Cite exact substrings of the Then. The normalized positive candidate may
  paraphrase those spans.
- Return `ambiguous` with an empty list only when candidate boundaries cannot
  be determined.

For “provide a user_id that is a string”, presence and string validity are two
candidates. For “obtain approval before submitting”, approval and submission
are two candidates; this task does not yet decide that submission is only a
temporal reference.

## Output

```json
{
  "branch_id": "airline_example#b0",
  "resolution_status": "resolved",
  "entities": [
    {
      "event_or_state": "One positive independently assessable event or state.",
      "evidence_spans": ["exact Then substring"]
    }
  ],
  "reason": "Why this is the complete candidate list."
}
```

Return candidates in textual order. Do not add identifiers; the compiler
assigns stable IDs after validation.

## Input

```json
{then_entity_discovery_input}
```
