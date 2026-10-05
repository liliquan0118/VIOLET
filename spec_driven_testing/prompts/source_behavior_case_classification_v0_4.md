# Source Behavior Case Classification

## One question only

For this one already-discovered positive behavior, which normative cases does
the supplied source explicitly state?

The behavior is positive, so modality always applies to that positive event:
`must not cancel` means `behavior=cancel`, `modality=prohibited`.

Return one row per distinct condition/modality case. Several OR conditions for
the same behavior remain several cases in this one response; never duplicate
the behavior. Use condition `always` only when no condition is stated. An
explicit iff reverse case may use `biconditional_complement`; do not invent the
complement of an ordinary one-way if.

Do not map GWT branches, extract relations, search other Specs, or decide test
observability. Evidence must be exact source substrings. Do not generate case
IDs.

## Output

```json
{
  "spec_id": "airline_example",
  "behavior_id": "B01",
  "resolution_status": "resolved",
  "cases": [
    {
      "modality": "required",
      "condition": "A complete condition for this positive behavior.",
      "evidence_spans": ["exact source substring"],
      "derivation": "explicit"
    }
  ],
  "reason": "Why these are all normative cases for this behavior."
}
```

Modalities: `required`, `prohibited`, `permitted`. Derivations: `explicit`,
`biconditional_complement`.

## Input

```json
{source_behavior_case_input}
```
