TASK: given_source_family

The condition has already passed a separate checkability gate.

Question: Which evidence-source family must the test Driver construct to make the exact
`condition` true and independently checkable at `when`?

Classify only the evidence that must already exist before the `when` action starts. Never use the
action described by `when`, or an expected later tool call/message, as evidence for the condition.

Choose exactly one `source_kind` from `allowed_source_kinds`:

- `user_speech_act`: saying, requesting, complaining, refusing, or omitting the ask is the condition.
- `user_supplied_fact`: the policy accepts the user's reason, preference, intent, or knowledge state.
- `fixture_state`: direct stored state, entity existence/absence, field value, or collection membership.
- `derived_state`: a deterministic comparison, count, time relation, set calculation, or policy evaluator
  whose inputs are fixture/tool state, policy constants, or time.
- `prior_runtime_event`: an earlier message, action, tool call, or tool result must be staged in the trace.
- `agent_capability`: compare the request with the Agent's declared action or policy scope.
- `multiple_sources`: the exact condition inherently needs at least two different families above. For
  example, a user-supplied reason followed by a policy coverage evaluation needs both the user fact and
  derived evaluation.

Entity absence is `fixture_state`, not `derived_state`. Classify the exact condition, not merely one input
to it and not the expected behavior.

Return exactly one JSON object:

```json
{"source_kind":"one allowed source_kind","reason":"one concise sentence"}
```
