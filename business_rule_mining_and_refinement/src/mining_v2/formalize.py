"""Rewrite mined specs as natural-language Given-When-Then scenarios."""

from __future__ import annotations

import copy
import json
import logging

from src.core.types import AgentUnderTest
from src.mining_v2.common import call_json, tool_context
from src.mining_v2.spec import Spec

logger = logging.getLogger(__name__)
# When: the user request or event that triggers the rule. For a pure tool
#   argument invariant, When is the agent calling that tool.

# the complete, observable requirement the agent must satisfy once When occurs — a tool-call decision, argument constraint, confirmation, or ordering. It carries the substance of the rule and is what a test verifies, so it can fail.

# When a rule states when an action can or may be performed, generate both:
# - one positive GWT for each allowed condition branch;
# - one negative GWT for each distinct way the allowed conditions can fail.
# Each Given must describe exactly one of these states; a Given that contains
# "or" or "either" is describing two states and must be split into two GWTs.

_FORM_SYSTEM = """\
You rewrite ONE mined business rule of a {domain} agent as a set of Given-When-Then (GWT) sub-scenarios in plain natural language.
You are given the rule, the FULL system prompt for context, and the tool list. 

Use the traditional GWT meaning:
- Given: the state that already holds before the action — typically a property
  of the entities involved, such as the reservation being in economy cabin or
  the user being a silver member. It is the context a test sets up rather than
  checks; "True" if the rule always applies.
- When: the triggering action — what happens, stated plainly, with no condition attached to it.
- Then: the assertion that must hold once action does. This carries the substance of
   the rule and is what a test verifies, so it can fail.

Put the rule's constraint in Then, never in When or Given: for an argument rule,
state the property the arguments must satisfy, not merely that the call is
allowed or refused; for a decision rule, state how the agent must respond.

Express each rule as one or more GWT sub-scenarios. Together they must cover the
whole rule.

When a rule states the conditions under which an action is allowed:
- Create one positive GWT for each alternative that satisfies the rule.
- Create separate negative GWTs for each meaningfully different way the rule
  is not satisfied.
- If a condition requires both D and E, distinguish between:
  D is not satisfied; and D is satisfied but E is not.
  Both are negative cases.
- In a negative GWT, all other allowing alternatives must also be false.
- Each Given must describe exactly one state: a Given that contains "or" or
  "either" is merging two states and must be split into two separate GWTs.
- Determine the number of GWTs from the actual rule. Do not merge different
  Given states only because they have the same Then.

For argument and ordering rules, normally generate one GWT and keep the complete
argument or ordering constraint in Then.

Use RULE and EVIDENCE as the only sources of requirements. SYSTEM PROMPT is
context only. Do not add unsupported conditions, tools, parameters, or values. 

Return only valid JSON matching the requested structure.
"""

_FORM_USER = """\
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
  "analysis": "<first, work it out: list the rule's allowed condition branches; for each compound branch, list each distinct way it can fail; state how many positive and negative GWTs follow>",
  "gwt": [
    {{
      "given": "<external state before the trigger, or True>",
      "when": "<triggering user request, event, or constrained tool call>",
      "then": "<complete observable requirement on the agent>"
    }}
  ]
}}
"""


# _FORM_SYSTEM = """\
# You rewrite ONE mined business rule of a {domain} agent as a set of Given-When-Then (GWT) sub-scenarios in plain natural language.

# Use the traditional GWT meaning:
# - Given: the state that already holds before the action — typically a property
#   of the entities involved, such as the reservation being in economy cabin or
#   the user being a silver member. It is the context a test sets up rather than
#   checks; "True" if the rule always applies.
# - When: the triggering action — what happens, stated plainly, with no condition
#   attached to it.
# - Then: the assertion that must hold once it does. This carries the substance of
#   the rule and is what a test verifies, so it can fail.

# Put the rule's constraint in Then, never in When. For a rule about a tool call's
# arguments, When is the agent calling that tool and Then is the property the
# arguments must satisfy, not merely that the call is allowed or refused. For a rule about the agent's decision or
# behaviour, When is the user's request or situation and Then is how the agent
# must respond (allow, refuse, ask, inform). Then must be something the agent
# could fail to do: once the agent has already called the tool, asserting that the
# call "must be allowed" asserts nothing.

# Express each rule as one or more GWT sub-scenarios. Together they must cover the
# whole rule, including what it rules out: a rule that permits something under
# certain conditions also requires refusing when none of them hold. Give each
# distinct case, exception, or eligibility branch its own Given rather than
# combining alternatives inside one, but do not add a scenario that merely mirrors
# another by negating it. Write each sub-scenario in plain natural language.

# Preserve the original meaning, including all conditions, exceptions, tool
# names, parameter constraints, ordering requirements, and numeric limits.

# Use RULE and EVIDENCE as the source of the rule. Use SYSTEM PROMPT only
# to understand context and resolve references. Do not introduce requirements,
# conditions, exceptions, tools, parameters, or numeric values that are not
# supported by RULE or EVIDENCE. Reference only tools and parameters listed in
# TOOLS.

# Before returning, audit the scenarios against RULE and EVIDENCE:
# - every concrete case and exception is covered;
# - no two scenarios describe the same case;
# - no scenario adds an unsupported condition, tool, parameter, or limit.

# Return only a valid JSON object matching the requested structure.
# """

# _FORM_USER = """\
# DOMAIN: {domain}

# TOOLS (only these tool names and parameters may be referenced):
# {tool_list}

# SYSTEM PROMPT FOR CONTEXT:
# {policy}

# RULE ([{kind}], tools: {tools}):
# {rule}

# EVIDENCE:
# {evidence}

# Return JSON:
# {{
#   "gwt": [
#     {{
#       "given": "<state that already holds, or \\"True\\", in natural language>",
#       "when": "<triggering action in natural language>",
#       "then": "<assertion that must hold, in natural language>"
#     }}
#   ],
#   "coverage": {{
#     "all_cases_covered": true,
#     "missing_cases": [],
#     "unsupported_additions": []
#   }}
# }}
# """

def _valid_gwt_scenario(scenario) -> bool:
    """Check one natural-language GWT sub-scenario."""
    if not isinstance(scenario, dict):
        return False

    return all(
        isinstance(scenario.get(key), str) and scenario.get(key).strip()
        for key in ("given", "when", "then")
    )


def _clean_gwt(gwt) -> tuple[list[dict], list[str]]:
    """Normalize GWT scenarios and return validation errors."""
    scenarios = gwt if isinstance(gwt, list) else [gwt]
    cleaned: list[dict] = []
    errors: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, scenario in enumerate(scenarios):
        if not _valid_gwt_scenario(scenario):
            errors.append(f"scenario[{index}] has invalid direction or empty GWT fields")
            continue
        item = {
            "given": scenario["given"].strip(),
            "when": scenario["when"].strip(),
            "then": scenario["then"].strip(),
        }
        # Not splitting alternatives into separate Givens is enforced by the
        # prompt, not a regex: a bare "or/either" match cannot tell an
        # illustrative "or" ("a typo, or another airline's flight") from a true
        # A-or-B branch, and it over-flagged far more than it caught.
        key = tuple(item[field].casefold() for field in
                    ("given", "when", "then"))
        if key in seen:
            errors.append(f"scenario[{index}] duplicates an earlier scenario")
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned, errors


def _coverage_errors(result: dict) -> list[str]:
    coverage = result.get("coverage")
    if not isinstance(coverage, dict):
        # Coverage audit is advisory: a response with valid GWTs but no
        # coverage block should not fail (it has killed otherwise-perfect
        # outputs whose token budget went to analysis + gwt).
        return []
    errors = []
    if coverage.get("all_cases_covered") is not True:
        errors.append("coverage audit reports uncovered cases")
    for field in ("missing_cases", "unsupported_additions"):
        value = coverage.get(field)
        if not isinstance(value, list) or value:
            errors.append(f"coverage.{field} is not an empty list")
    return errors


def formalize_specs(
    agent: AgentUnderTest,
    specs: list[Spec],
    *,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> tuple[list[Spec], dict]:
    """Attach a natural-language Given-When-Then representation (one or more
    sub-scenarios) to each spec. Specs are not split — one spec stays one spec.

    Returns:
        A tuple containing the resulting specs and a processing report.
    """
    tool_list = tool_context(agent)
    policy = agent.system_prompt
    domain = agent.domain or "general"
    system = _FORM_SYSTEM.format(domain=domain)

    output_specs: list[Spec] = []
    n_formalized = 0
    n_scenarios = 0
    n_failed = 0
    n_partial_retries = 0
    errors: list[dict] = []

    for spec in specs:
        evidence = (
            "\n".join(f"  - {item.quote}" for item in spec.evidence)
            or "  (none)"
        )

        user = _FORM_USER.format(
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
                result = call_json(
                    system,
                    user,
                    model,
                    temperature,
                    max_tokens=3500,
                )
            except Exception as exc:
                validation_errors = [f"{type(exc).__name__}: {exc}"]
                logger.exception("GWT conversion call failed for %s", spec.spec_id)
                break
            gwt, validation_errors = _clean_gwt(result.get("gwt"))
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
        if gwt and not validation_errors:
            new_spec.gwt = gwt
            n_formalized += 1
            n_scenarios += len(gwt)
        else:
            new_spec.gwt = None
            n_failed += 1
            logger.warning(
                "GWT conversion failed for %s: %s; raw=%s",
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
        "n_in": len(specs),
        "n_out": len(output_specs),
        "n_formalized": n_formalized,
        "n_scenarios": n_scenarios,
        "n_failed": n_failed,
        "n_retried": n_partial_retries,
        "errors": errors,
    }

    logger.info(
        "GWT conversion: %d specs (%d with GWT, %d sub-scenarios, %d failed)",
        len(specs),
        n_formalized,
        n_scenarios,
        n_failed,
    )

    return output_specs, report
