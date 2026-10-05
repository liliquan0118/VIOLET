# Branch Requirement Mapping

## One question

How does this one branch map its already-parsed Then assertions to the supplied
source requirements under its Given/When?

Do not reparse the source or the Then. Do not judge siblings, search other
Specs, bind tools, or decide test observability.

## Mapping rules

For each source requirement, choose one branch relation:

- `direct_case`: Given/When selects the requirement or one activation path;
- `complement_case`: Given/When tests the stated boundary when its activation
  condition is false;
- `irrelevant`: the requirement does not govern this branch;
- `ambiguous`: the relation cannot be determined from the supplied input.

For each Then assertion, choose one support status:

- `supported`: source requirements support the assertion under this context;
- `contradicted`: the assertion reverses a supplied source requirement;
- `unsupported`: no supplied source requirement supports the assertion;
- `ambiguous`: support cannot be decided from the supplied input.

This task answers support only against the supplied source requirements. An
`unsupported` assertion may later be linked to another Spec by a separate
lineage resolver.

## Output

```json
{
  "branch_id": "airline_example#b0",
  "overall_status": "aligned",
  "requirement_mappings": [
    {
      "requirement_id": "S01",
      "branch_relation": "direct_case",
      "matched_alternative_ids": ["A01"],
      "assertion_ids": ["T01"],
      "coverage_status": "covered",
      "reason": "Why this requirement has this relation and coverage."
    }
  ],
  "assertion_mappings": [
    {
      "assertion_id": "T01",
      "support_status": "supported",
      "requirement_ids": ["S01"],
      "reason": "Why the supplied source supports this assertion."
    }
  ],
  "reason": "Overall mapping result."
}
```

Return requirements and assertions once each and in input order. Coverage
statuses are `covered`, `omitted`, `not_expected`, or `ambiguous`. Overall is
`aligned` only when no mapping issue exists; otherwise use `issues_found`, or
`ambiguous` when the mapping itself is materially ambiguous.

## Input

```json
{branch_mapping_input}
```
