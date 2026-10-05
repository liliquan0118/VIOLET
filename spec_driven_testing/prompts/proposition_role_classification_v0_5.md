# Proposition Role Classification v0.5

## The only question

What one role does the supplied exact proposition play in its complete original
sentence, `primary_text`?

Do not decide modality, polarity, source support, normalization, relations, or
test observability.

## Choose exactly one label

When `text_kind` is `source_rule`:

- `governed_outcome`: the action or state that the rule regulates—what the rule
  says must, must not, or may happen.
- `activation_condition`: a fact that controls whether the rule applies, such
  as an `if`, `unless`, or eligibility condition.
- `scope_reference`: a mentioned entity, event, or context that only locates
  the rule's scope and is not itself regulated or an activation condition.
- `ambiguous`: the sentence does not determine one of the roles above.

When `text_kind` is `gwt_then`:

- `asserted_outcome`: an action or state that the Then says should hold. A
  required property of an output or tool argument is also an asserted outcome.
- `condition_reference`: a fact that qualifies when or for which case the
  asserted outcome holds.
- `temporal_reference`: an event used only as a `before` or `after` endpoint.
- `scope_reference`: a mentioned entity, action, or context that only locates
  the assertion's scope.
- `ambiguous`: the sentence does not determine one of the roles above.

Use the exact proposition together with the full sentence. Do not classify
other propositions in the sentence.

## Output

Return JSON only:

```json
{
  "target_id": "copy target_id from the input",
  "proposition_id": "copy proposition_id from the input",
  "role": "one allowed label",
  "reason": "One short sentence explaining this role in the original sentence."
}
```

## Input

```json
{proposition_role_input}
```
