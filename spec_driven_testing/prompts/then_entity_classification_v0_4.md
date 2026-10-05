# Then Entity Classification

## One question only

For this one already-discovered positive Then entity, is it an asserted outcome
or reference-only, and what single modality or reference role applies?

- `asserted`: the Then asks us to judge whether this positive event/state
  occurs or holds. Set modality to `required`, `prohibited`, or `permitted` and
  set `reference_role` to null.
- `reference`: it is mentioned only as a condition, temporal endpoint, or scope
  anchor. Set modality to null and reference role to `condition`,
  `temporal_anchor`, or `scope_anchor`.

Because the entity is already positive, “must not X” always means X is
`prohibited`, never `required`.

Do not judge source support, extract relations, bind tools, or decide
observability.

## Output

```json
{
  "branch_id": "airline_example#b0",
  "entity_id": "E01",
  "role": "asserted",
  "modality": "prohibited",
  "reference_role": null,
  "reason": "Why this single classification follows from the Then."
}
```

## Input

```json
{then_entity_classification_input}
```
