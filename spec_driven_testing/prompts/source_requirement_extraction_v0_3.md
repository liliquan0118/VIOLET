# Source Requirement Extraction

## One question

What normative requirements are explicitly stated by this source anchor?

Use only `source_context`. Do not judge a GWT, infer tool behavior, search for
another policy, or decide how a test will observe the result.

## Representation rules

- A requirement is one behavior with one modality: `required`, `prohibited`,
  or `permitted`.
- Keep a condition with the behavior it governs.
- `applicability.mode=unconditional` means the requirement has no stated
  condition and must have no alternatives.
- `applicability.mode=any_of` means any listed alternative activates the same
  requirement. Each alternative is a self-contained condition; conjunctions
  that form one path stay together in that condition.
- When several conditions activate the same behavior with the same modality,
  return **one requirement** with several alternatives. Do not repeat that
  behavior as one requirement per condition.
- Expand an explicit “if and only if” only when the source states normative
  behavior in both directions. Do not invent a closed-world prohibition from
  an ordinary one-way “if”.
- Every evidence span must be an exact substring of the supplied source.
- If more than one materially different parse remains possible, return
  `ambiguous` with empty `requirements`.

## Output

```json
{
  "spec_id": "airline_example",
  "resolution_status": "resolved",
  "requirements": [
    {
      "requirement_id": "S01",
      "modality": "permitted",
      "claim": "One self-contained source requirement.",
      "evidence_spans": ["exact source substring"],
      "applicability": {
        "mode": "any_of",
        "alternatives": [
          {
            "alternative_id": "A01",
            "condition": "One complete activation path.",
            "evidence_spans": ["exact source substring"]
          }
        ]
      }
    }
  ],
  "reason": "Why this is the complete source parse."
}
```

Requirement IDs are consecutive `S01`, `S02`, ... in source order.
Alternative IDs restart within each requirement and are consecutive `A01`,
`A02`, ... in source order.

For example, “a request may be approved if condition X, Y, or Z holds” is one
`permitted` approval requirement whose applicability has three alternatives;
it is not three separate approval requirements.

## Input

```json
{source_requirement_input}
```
