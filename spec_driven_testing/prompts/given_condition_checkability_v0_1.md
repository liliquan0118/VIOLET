TASK: given_condition_checkability

Question: Does the supplied evidence define a repeatable test procedure for deciding whether
the exact `condition` is true at `when`?

Choose one:

- `checkable`: yes. This includes a user request/complaint, a user-authoritative reason or
  knowledge state, direct fixture state or entity absence, a fully defined deterministic rule,
  a staged prior runtime event, or comparison with the listed Agent actions.
- `source_undefined`: no. The condition uses a concept, but the complete accepted evidence
  supplied here gives no rule, field, event, or authorized fact that defines its truth.
- `insufficient_context`: the source may be defined, but the supplied evidence is incomplete
  or ambiguous, so this packet cannot decide.

Do not choose an evidence-source family. Do not invent an unstated rule. In particular, a
timestamp does not define “expired” unless an accepted expiration rule is also supplied.

Return exactly one JSON object:

```json
{"checkability":"checkable","reason":"one concise sentence"}
```
