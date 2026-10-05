Specification source: {{source_origin}}
Original rule: {{source_rule}}
Given: {{given}}
When: {{when}}
Then: {{then}}

Tool: {{tool_name}}
Tool description: {{tool_description}}
Input field: {{field_name}}
Input field description: {{field_description}}

Proposed checking scope:
When the Given holds, check this field requirement on the arguments of every invocation of this tool, including invocations that return an error rather than completing the operation successfully.

Question: Is this proposed checking scope supported by the specification?

Return only {"decision":"yes|no|unclear","reason":"one short explanation"}.
Use yes if the proposed scope follows from the specification, no if it conflicts with it, and unclear if the supplied information does not determine the scope. Do not assume a tool invocation and a successful operation are the same event.
Do not judge whether the Agent must invoke the tool, whether a task was completed, or whether a field value passes its type check.
