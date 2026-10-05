"""Violation-only GWT conversion: each sub-scenario describes one state in
which the requested action does NOT satisfy the rule (Given = an ineligible
state, Then = the refusal/enforcement the agent must produce).

The violation states are not written in the rule text — they must be derived
by negating the allowed branches (De Morgan on compound branches), which is
why this runs as its own focused call with an explicit "analysis" slot: the
derivation has to happen in generated tokens before the GWTs are written.
See formalize_pos.py for the transcription-only positive side.
"""

from __future__ import annotations

import copy
import json
import logging

from src.core.types import AgentUnderTest
from src.mining_v2.common import call_json, tool_context
from src.mining_v2.formalize import _clean_gwt
from src.mining_v2.spec import Spec

logger = logging.getLogger(__name__)

_NEG_SYSTEM = """\
You rewrite ONE mined business rule of a {domain} agent as Given-When-Then (GWT)
sub-scenarios in plain natural language. This file represents the FORBIDDEN side
of the rule: each GWT is one situation in which a specific agent behaviour is
not allowed, and Then states what the agent must not do (or must refuse).
You are given the rule, the FULL system prompt for context, and the tool list.

Use the traditional GWT meaning:
- Given: the state that already holds before the action — typically a property
  of the entities involved. It is the context a test sets up rather than
  checks; "True" if the rule always applies.
- When: the triggering action — what happens, stated plainly, with no condition attached to it.
- Then: the assertion that must hold once the action occurs. This carries the
  substance of the rule and is what a test verifies, so it can fail.

First decide how the rule is stated, then follow the matching case:

1. The rule is stated positively — it says when an action IS allowed or what
   the agent should do ("X can be done if ...", "do X when ..."). Derive its
   violation scenarios and write one GWT per scenario:
   - one GWT for each meaningfully different way the stated conditions fail;
   - if a condition requires both D and E, "D fails" and "D holds but E fails"
     are two different scenarios — one GWT each;
   - in each GWT, every other allowing alternative must also be false;
   - each Given describes exactly one state: a Given containing "or" or
     "either" merges two states and must be split into two GWTs.

2. The rule is stated as a prohibition or restriction — "do not X", "never X",
   "only do X if ...". Represent the rule directly, without inventing new
   cases: one GWT per case the rule itself distinguishes. For "only do X
   if C": Given is a state where C does not hold, When is the user requesting
   X, Then is that the agent must not do X.

For argument and ordering rules, normally generate one GWT and keep the complete
argument or ordering constraint in Then.

Use RULE and EVIDENCE as the only sources of requirements. SYSTEM PROMPT is
context only. Do not add unsupported conditions, tools, parameters, or values.

Return only valid JSON matching the requested structure.
"""

_NEG_USER = """\
DOMAIN: {domain}

TOOLS:
{tool_list}

SYSTEM PROMPT:
{policy}

RULE ([{kind}], tools: {tools}):
{rule}

EVIDENCE:
{evidence}

Return:
{{
  "analysis": "<first, work it out: state whether the rule is positive (case 1) or a prohibition/restriction (case 2). For case 1, list the allowed condition branches and, for each compound branch, each distinct way it can fail. For case 2, list the cases the rule itself distinguishes. State how many GWTs follow>",
  "gwt": [
    {{
      "given": "<external state before the trigger, or True>",
      "when": "<triggering user request, event, or constrained tool call>",
      "then": "<complete observable requirement on the agent>"
    }}
  ]
}}
"""


def formalize_specs_neg(
    agent: AgentUnderTest,
    specs: list[Spec],
    *,
    model: str = "gpt-4.1",
    temperature: float = 0.0,
) -> tuple[list[Spec], dict]:
    """Attach violation-only GWT sub-scenarios to each spec."""
    tool_list = tool_context(agent)
    policy = agent.system_prompt
    domain = agent.domain or "general"
    system = _NEG_SYSTEM.format(domain=domain)

    output_specs: list[Spec] = []
    n_formalized = 0
    n_scenarios = 0
    n_failed = 0
    n_skipped_non_prompt = 0
    n_partial_retries = 0
    errors: list[dict] = []

    for spec in specs:
        # The violation file only represents prompt-sourced business rules;
        # schema/domain-knowledge specs are out of scope here.
        if spec.origin != "prompt":
            new_spec = copy.deepcopy(spec)
            new_spec.gwt = None
            n_skipped_non_prompt += 1
            output_specs.append(new_spec)
            continue

        evidence = (
            "\n".join(f"  - {item.quote}" for item in spec.evidence)
            or "  (none)"
        )
        user = _NEG_USER.format(
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
                logger.exception("Violation GWT call failed for %s", spec.spec_id)
                break
            gwt, validation_errors = _clean_gwt(result.get("gwt"))
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
        if gwt and not validation_errors:
            new_spec.gwt = gwt
            n_formalized += 1
            n_scenarios += len(gwt)
        else:
            new_spec.gwt = None
            n_failed += 1
            logger.warning(
                "Violation GWT failed for %s: %s; raw=%s",
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
        "direction": "negative",
        "n_in": len(specs),
        "n_out": len(output_specs),
        "n_formalized": n_formalized,
        "n_scenarios": n_scenarios,
        "n_skipped_non_prompt": n_skipped_non_prompt,
        "n_failed": n_failed,
        "n_retried": n_partial_retries,
        "errors": errors,
    }
    logger.info(
        "Violation GWT: %d specs (%d with GWT, %d sub-scenarios, "
        "%d non-prompt skipped, %d failed)",
        len(specs), n_formalized, n_scenarios,
        n_skipped_non_prompt, n_failed,
    )
    return output_specs, report
