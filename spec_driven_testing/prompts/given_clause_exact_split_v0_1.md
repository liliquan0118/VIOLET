TASK: given_clause_exact_split

Copy each independently truth-valued condition from `source_clause` as an exact
text span, in source order.

Do not paraphrase, add missing words, classify the conditions, or decide whether
all or any must be true.

Return exactly one JSON object:

```json
{"conditions":["first exact span","second exact span"]}
```
