# Spec-to-GWT Allocation and Then Alignment

## Goal

Audit how one source spec is allocated across all of its sibling GWT branches.
For every branch, determine whether its `then` faithfully states the source
requirement selected by that branch's `given` and `when`.

A branch is not required to repeat the whole parent spec. Do not report an
omission when an independent source requirement is explicitly allocated to a
different sibling branch. Instead, audit the sibling set jointly for complete,
non-conflicting allocation.

Check:

- whether each branch Then is supported under its Given/When context;
- omitted requirements that should apply to that branch;
- unsupported details, changed modality or polarity, and changed scope;
- source requirements absent from every sibling branch;
- unresolved references that prevent a reliable allocation.

Do not repair or rewrite a Then. Do not infer requirements from tools,
Coverage Models, common practice, or outside domain knowledge.

Use branch status `aligned`, `partial`, `conflict`, or `ambiguous`. Use overall
status `aligned` only when every sibling branch is aligned and there are no
unallocated requirements. Otherwise use `issues_found`, or `ambiguous` when
materially different allocations remain supported.

## Output

Return exactly one JSON object:

```json
{
  "spec_id": "airline_000_arg",
  "overall_status": "aligned",
  "branch_results": [
    {
      "branch_id": "airline_000_arg#b0",
      "alignment_status": "aligned",
      "findings": [],
      "reason": "This branch faithfully represents its assigned source claim."
    }
  ],
  "unallocated_requirements": [],
  "reason": "The sibling GWT set completely and faithfully allocates the source spec."
}
```

A branch finding has exactly:

```json
{
  "finding_type": "omitted_requirement",
  "then_spans": [],
  "source_quotes": ["exact supplied source quote"],
  "description": "Description of the branch-local mismatch."
}
```

Allowed finding types are `omitted_requirement`, `unsupported_detail`,
`incorrect_modality`, `incorrect_polarity`, `incorrect_scope`, and
`unresolved_context`. Then spans must be exact substrings of that branch's
Then. Source quotes must be exact excerpts from the supplied source context.

Each unallocated requirement has exactly `source_quotes` and `description`.
Return every sibling branch once and in input order.

## Input

```json
{spec_source_alignment_input}
```
