# Proposition Modality Classification v0.5

## The only question

Does the complete sentence make this already-confirmed outcome required,
prohibited, or permitted?

The proposition's role has already been fixed. Do not reconsider its boundary
or role. Do not normalize it, map it to another text, or decide test
observability.

## Choose exactly one label

- `required`: the outcome is obligatory or is the expected result.
- `prohibited`: the outcome must not occur.
- `permitted`: the outcome is explicitly allowed but not required.
- `ambiguous`: the supplied sentence does not distinguish these choices.

Copy into `cue_spans` the exact word or phrase in `primary_text` that signals
the choice, such as `must`, `must not`, `may`, `should`, or a declarative verb
that states the expected result. Negation inside an object qualifier is not a
prohibition cue. For example, in `must not provide information not supplied by
tools`, `must not` governs the action; the later `not supplied by tools`
describes the information.

If and only if the modality is `ambiguous`, return an empty `cue_spans` list.

## Output

Return JSON only:

```json
{
  "target_id": "copy target_id from the input",
  "proposition_id": "copy proposition_id from the input",
  "modality": "required | prohibited | permitted | ambiguous",
  "cue_spans": ["exact substring from primary_text"],
  "reason": "One short sentence explaining the selected modality."
}
```

## Input

```json
{proposition_modality_input}
```
