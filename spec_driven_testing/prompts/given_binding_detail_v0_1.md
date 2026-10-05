TASK: given_binding_detail

Answer only the single `question` in the input. The condition must be true before the When starts.
Use only supplied policy lines, endpoints, operands, relation contracts, or scenarios. Do not invent a
field, tool, rule, value, or later Agent behavior. Do not import another Given condition.

If the supplied material cannot define the exact condition, use the supplied `insufficient` outcome
when it exists, or set `decision` to `insufficient` when the answer contract permits it. Do not make an
ambiguous condition concrete by guessing.

Return exactly one JSON object with exactly the fields in `answer_contract`. Keep `reason` to one
concise sentence.
