TASK: given_evidence_source_v2

At the time described by `when`, what evidence source must the test Driver use to make
the exact `condition` independently checkable as true?

Choose exactly one `source_kind` from `allowed_source_kinds`.

- Classify the condition, not the expected behavior in the rule.
- A request or complaint is a `user_speech_act` because saying it creates the condition.
- A reason, preference, or knowledge state supplied by the user is a `user_supplied_fact`.
- Use `fixture_state` only for directly stored state; use `derived_state` when a deterministic
  comparison, count, time rule, set calculation, or eligibility evaluator is required.
- Use `prior_runtime_event` only when an earlier trace event must establish the condition.
- Use `agent_capability` only for comparison with the Agent's declared action scope.
- Use `multiple_sources` only when two or more source kinds are inherently required.
- Use `source_undefined` when the accepted rule uses a concept but provides no authoritative
  way to determine it.
- Use `insufficient_context` when such a source may exist but the accepted evidence supplied
  here is not enough to select it.

Return exactly one JSON object:

```json
{"source_kind":"one supplied source_kind","reason":"one concise sentence"}
```
