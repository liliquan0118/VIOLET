# Then Assertion Extraction

## One question

What outcomes are literally asserted by this `then`, and which mentioned
events are only references?

Use `given` and `when` only to resolve words in the Then. Do not judge whether
the assertions are supported by a source policy, bind tools, decide fixture
reachability, or decide test observability.

## Assertion versus reference

- An **assertion** is an outcome whose satisfaction or violation the Then
  itself asks us to judge.
- A **reference event** is mentioned only to qualify scope, state a condition,
  or serve as an endpoint for `before`/`after`. Its occurrence is not itself
  required or prohibited by the Then.
- Split two outcomes when one can be satisfied while the other is violated.
- Keep a guard, quantity comparison, allowed-value set, or scope qualifier
  with the assertion it qualifies.
- Add relations only when explicit text relates two assertions or relates an
  assertion to a reference event.

Example: “The agent must obtain approval before submitting the change” has one
required assertion (`obtain approval`), one temporal reference event (`submit
the change`), and one `before` relation. It does not by itself require the
submission to occur.

Every evidence span must be an exact substring of the Then. Return
`ambiguous` with empty collections when materially different parses remain.

## Output

```json
{
  "branch_id": "airline_example#b0",
  "resolution_status": "resolved",
  "assertions": [
    {
      "assertion_id": "T01",
      "modality": "required",
      "claim": "One independently assessable asserted outcome.",
      "evidence_spans": ["exact Then substring"]
    }
  ],
  "reference_events": [
    {
      "reference_id": "E01",
      "role": "temporal_anchor",
      "event": "An event mentioned only as a temporal endpoint.",
      "evidence_spans": ["exact Then substring"]
    }
  ],
  "relations": [
    {
      "relation_id": "R01",
      "relation": "before",
      "left_entity_id": "T01",
      "right_entity_id": "E01",
      "evidence_spans": ["exact Then substring"]
    }
  ],
  "reason": "Why this is the complete literal parse."
}
```

IDs are consecutive in source order. Reference roles are `condition`,
`temporal_anchor`, or `scope_anchor`. Relations are `before`, `after`,
`same_turn`, `mutually_exclusive`, or `requires`.

## Input

```json
{then_assertion_input}
```
