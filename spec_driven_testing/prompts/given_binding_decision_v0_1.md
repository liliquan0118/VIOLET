TASK: given_binding_decision

Answer only the single `question` in the input.

The `condition` must already be true before the `when` action starts. Use only the supplied accepted
context and only the choices supplied in the input. Do not invent a field, tool, rule, or event. Do not
design the later fixture, evaluator operands, conversation, or expected Agent behavior.

Classify only the exact `condition`. Do not import another Given condition from the same branch. A
condition may intentionally describe an invalid or policy-forbidden request; decide how its truth is
established, not whether the Agent is allowed to carry it out. For fixture-state conditions, prefer an
initial fixture collection or a read-only observation of existing state; never choose a write action that
creates the condition being tested.

Return exactly one JSON object with exactly the fields described by `answer_contract`. The `reason`
must be one concise sentence.
