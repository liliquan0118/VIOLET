TASK: given_atom_evidence_source

At the moment described by `when_event`, what evidence must the test Driver
construct so that `condition_clause` is independently checkable as true?

Choose exactly one `source_kind` from the supplied options. Use
`multiple_sources` only when at least two distinct source kinds are inherently
necessary. Use `insufficient` when the input does not determine the answer.

Do not treat a user's unsupported claim about an external fact as proof of that
fact. Choose `user_message` only when saying something is itself the condition,
such as making a request or complaint.

The upstream mapping note only explains why the clause was not represented as
one database-field filter. It is context, not an answer.

Return exactly one JSON object with these two fields:

```json
{"source_kind":"one supplied source_kind","reason":"one concise sentence"}
```
