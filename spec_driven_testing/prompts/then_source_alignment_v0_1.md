# GWT Then Source Alignment

## Goal

Determine whether the supplied `then` faithfully states the correctness
requirement supported by the supplied rule and evidence for this one GWT
branch.

The `then` text is the candidate test expectation. The source rule, clauses,
and evidence are the authority used to audit it. `given` and `when` identify
the branch context; they do not add new outcome requirements.

Check only:

- omitted requirements that change what counts as correct;
- unsupported details;
- changed modality (`must`, `must not`, `may`/`can`);
- changed polarity;
- changed operation, object, value, quantity, scope, or condition;
- unresolved references that make the expectation materially ambiguous.

Use `aligned` only when the Then is fully supported. Use `partial` when the
Then is supported but omits an independently testable part. Use `conflict`
when it changes or adds a requirement. Use `ambiguous` only when the supplied
source supports materially different readings.

Do not repair or rewrite the Then. Do not infer requirements from tool names,
common practice, or domain knowledge not included in the input.

## Output

Return exactly one JSON object:

```json
{
  "alignment_status": "aligned",
  "findings": [],
  "reason": "The Then faithfully preserves the source requirement."
}
```

For a non-aligned result, each finding has this shape:

```json
{
  "finding_type": "omitted_requirement",
  "then_spans": [],
  "source_quotes": ["exact supplied source quote"],
  "description": "Description of the mismatch."
}
```

Allowed finding types are `omitted_requirement`, `unsupported_detail`,
`incorrect_modality`, `incorrect_polarity`, `incorrect_scope`, and
`unresolved_context`. `then_spans` must contain exact substrings of `then` and
may be empty for an omission. `source_quotes` must quote only supplied source
material and may be empty for an unsupported Then detail.

## Input

```json
{source_alignment_input}
```
