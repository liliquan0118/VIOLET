"""Positive-only GWT conversion: each sub-scenario re-expresses one allowed
case of the rule (Given = an eligible state, Then = what the agent must do).

The positive branches are written verbatim in the rule text, so this is a
transcription task and is stable across runs. The violation side (negated
Givens) is deliberately NOT represented here — see formalize_neg.py.
"""

from __future__ import annotations

import copy
import json
import logging

from src.core.types import AgentUnderTest
from src.mining_v2.common import call_json, tool_context
from src.mining_v2.formalize import _clean_gwt, _coverage_errors
from src.mining_v2.spec import Spec

logger = logging.getLogger(__name__)

_POS_SYSTEM = """\
You rewrite ONE mined business rule of a {domain} agent as a set of Given-When-Then (GWT) sub-scenarios in plain natural language.

Use the traditional GWT meaning:
- Given: the state that already holds before the action — typically a property
  of the entities involved, such as the reservation being in economy cabin or
  the user being a silver member. It is the context a test sets up rather than
  checks; "True" if the rule always applies.
- When: the triggering action — what happens, stated plainly, with no condition
  attached to it.
- Then: the assertion that must hold once it does. This carries the substance of
  the rule and is what a test verifies, so it can fail.

Put the rule's constraint in Then, never in When. For a rule about a tool call's
arguments, When is the agent calling that tool and Then is the property the
arguments must satisfy, not merely that the call is allowed or refused. For a rule about the agent's decision or
behaviour, When is the user's request or situation and Then is how the agent
must respond. Then always describes the AGENT's observable behaviour and must
be something the agent could fail to do; never write a property of the user or
of the world as Then.

Express each rule as one or more GWT sub-scenarios. A sub-scenario states what
must hold, so its negation is already the violation — do not write a separate
scenario for the invalid case. Emit several sub-scenarios only when the rule
covers distinct concrete cases, exceptions, or eligibility branches, giving each
its own Given rather than combining alternatives inside one. Write each
sub-scenario in plain natural language.

Some rules impose no positive behaviour at all: pure prohibitions ("do not X")
and pure restrictions ("only do X if ..."), whose entire content is what they
forbid. For such a rule return an empty "gwt" list — its violation side is
represented separately. A rule that states when a service IS available
("X can be done if ...") does have positive sub-scenarios: one per branch,
with Then stating that the agent performs or allows the requested action.

For a rule that obliges the agent at all times — an invariant with no eligible
state — write a single sub-scenario with Given "True" and When "True", and
keep the complete constraint in Then. Do not split an invariant by moving part
of the constraint into When.

Preserve the original meaning, including all conditions, exceptions, tool
names, parameter constraints, ordering requirements, and numeric limits.

Use RULE and EVIDENCE as the source of the rule. Use SYSTEM PROMPT only as
context, to understand what the references in the rule mean. Do not introduce
requirements, conditions, exceptions, tools, parameters, or numeric values that
are not supported by RULE or EVIDENCE. Reference only tools and parameters
listed in TOOLS.

Before returning, audit the scenarios against RULE and EVIDENCE:
- every concrete case and exception is covered;
- no two scenarios describe the same case;
- no scenario adds an unsupported condition, tool, parameter, or limit.

Return only a valid JSON object matching the requested structure.
"""

_POS_USER = """\
DOMAIN: {domain}

TOOLS (only these tool names and parameters may be referenced):
{tool_list}

SYSTEM PROMPT FOR CONTEXT:
{policy}

RULE ([{kind}], tools: {tools}):
{rule}

EVIDENCE:
{evidence}

Return JSON:
{{
  "gwt": [
    {{
      "given": "<state that already holds, or \\"True\\", in natural language>",
      "when": "<triggering action in natural language>",
      "then": "<assertion that must hold, in natural language>"
    }}
  ],
  "coverage": {{
    "all_cases_covered": true,
    "missing_cases": [],
    "unsupported_additions": []
  }}
}}
"""


def formalize_specs_pos(
    agent: AgentUnderTest,
    specs: list[Spec],
    *,
    model: str = "gpt-4.1",
    temperature: float = 0.0,
) -> tuple[list[Spec], dict]:
    """Attach positive-only GWT sub-scenarios to each spec."""
    tool_list = tool_context(agent)
    policy = agent.system_prompt
    domain = agent.domain or "general"
    system = _POS_SYSTEM.format(domain=domain)

    output_specs: list[Spec] = []
    n_formalized = 0
    n_scenarios = 0
    n_failed = 0
    n_no_positive = 0
    no_positive_ids: list[str] = []
    n_partial_retries = 0
    errors: list[dict] = []

    for spec in specs:
        evidence = (
            "\n".join(f"  - {item.quote}" for item in spec.evidence)
            or "  (none)"
        )
        user = _POS_USER.format(
            domain=domain,
            tool_list=tool_list,
            policy=policy,
            kind=spec.kind,
            tools=",".join(spec.relevant_tools) or "agent",
            rule=spec.rule_text,
            evidence=evidence,
        )

        result: dict = {}
        gwt: list[dict] = []
        validation_errors: list[str] = []
        for attempt in range(2):
            try:
                result = call_json(system, user, model, temperature, max_tokens=3500)
            except Exception as exc:
                validation_errors = [f"{type(exc).__name__}: {exc}"]
                logger.exception("Positive GWT call failed for %s", spec.spec_id)
                break
            raw_gwt = result.get("gwt")
            if raw_gwt == []:
                # The model judged the rule purely prohibitive/restrictive:
                # no positive side; the neg file represents it.
                gwt, validation_errors = [], []
                break
            gwt, validation_errors = _clean_gwt(raw_gwt)
            validation_errors += _coverage_errors(result)
            if gwt and not validation_errors:
                break
            if attempt == 0:
                n_partial_retries += 1
                user += (
                    "\n\nRETRY: The previous response failed validation:\n- "
                    + "\n- ".join(validation_errors)
                    + "\nReturn a complete corrected JSON object."
                )

        new_spec = copy.deepcopy(spec)
        if not validation_errors:
            new_spec.gwt = gwt
            if gwt:
                n_formalized += 1
                n_scenarios += len(gwt)
            else:
                n_no_positive += 1
                no_positive_ids.append(spec.spec_id)
        else:
            new_spec.gwt = None
            n_failed += 1
            logger.warning(
                "Positive GWT failed for %s: %s; raw=%s",
                spec.spec_id,
                "; ".join(validation_errors),
                json.dumps(result, ensure_ascii=False)[:800],
            )
            errors.append({
                "spec_id": spec.spec_id,
                "errors": validation_errors,
                "raw_response": result,
            })
        output_specs.append(new_spec)

    report = {
        "direction": "positive",
        "n_in": len(specs),
        "n_out": len(output_specs),
        "n_formalized": n_formalized,
        "n_scenarios": n_scenarios,
        "n_no_positive": n_no_positive,
        "no_positive_ids": no_positive_ids,
        "n_failed": n_failed,
        "n_retried": n_partial_retries,
        "errors": errors,
    }
    logger.info(
        "Positive GWT: %d specs (%d with GWT, %d sub-scenarios, "
        "%d no positive side, %d failed)",
        len(specs), n_formalized, n_scenarios, n_no_positive, n_failed,
    )
    return output_specs, report
