"""LLM user carrying several requests in one conversation (reuse log section 211).

tau-bench style compound task: the user has two (or more) goals; afterwards each
goal is judged with its own test case's oracle on the shared transcript.

Modes:
  together    state all requests in the opening message
  sequential  state the first request, bring up the next one only after the
              previous one is done or clearly refused
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from agentest.driver.tactical_user_v0_1 import _clean_facts

MODES = ("together", "sequential")


def build_compound_instructions(goals: Sequence[str], facts: Sequence[str], mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    numbered = "\n".join(f"  {i}. {g}" for i, g in enumerate(goals, 1))
    seen, fact_lines = set(), []
    for text in facts:
        cleaned = _clean_facts(text)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            fact_lines.append(f"  - {cleaned}")
    how = (
        "State ALL of your requests together in your first message, then work through them with the agent."
        if mode == "together" else
        "Start with request 1 only. Bring up the next request only after the previous one has been completed "
        "or clearly refused by the agent. Go through the requests in the listed order."
    )
    return (
        "You are a customer contacting customer service. Stay in character as the customer for the whole "
        "conversation.\n\n"
        f"YOU HAVE {len(goals)} SEPARATE REQUESTS (written by a test designer, possibly in the third person -- "
        f"say each naturally, in the first person, as your own):\n{numbered}\n\n"
        f"HOW TO RAISE THEM: {how}\n\n"
        "FACTS YOU KNOW (share a fact only when the agent asks for it or it is needed for one of your "
        "requests; some may be irrelevant -- ignore those):\n" + ("\n".join(fact_lines) or "  (none)") + "\n\n"
        "HARD LIMITS:\n"
        "- Never invent account data: IDs, names, dates, amounts, addresses, payment methods, reservations, "
        "orders, lines or bills other than the facts above.\n"
        "- Do not make any untrue statement.\n"
        "- Do not drop a request on your own initiative; if the agent refuses one, accept the refusal and move "
        "on to the next request.\n"
        "- If the agent asks you to confirm an action that matches one of your requests, a plain 'yes' is "
        "fine. Never put '###STOP###' in the same message as a confirmation -- wait for the agent to act.\n"
        "- When every request has been completed or clearly refused, end the conversation by replying with "
        "'###STOP###' only."
    )


def compound_user_factory(components: Sequence[Mapping[str, Any]], mode: str, *, compound_id: str,
                          user_llm: str, llm_args: Mapping[str, Any] | None = None):
    """run_generic_tau_online_plan user_factory. `components` are the bound plans
    in request order; each one's opening and known facts come from the scripted
    user the runner would have built for it (identity-corrected for retail)."""
    from tau2.user.user_simulator import UserSimulator

    from agentest.driver.generic_tau_online_v1 import GenericDeterministicTauUser, _retail_identity_profile

    class CompoundUserSimulator(UserSimulator):
        def perturbation_record(self) -> dict[str, Any]:
            return {"user_kind": "llm_compound", "mode": mode, "compound_id": compound_id,
                    "components": [c["source_branch_id"] for c in components],
                    "instructions": self._compound_instructions}

    def factory(bound_plan, task, environment, litellm_model, domain, scripted_user):
        def _target_args(plan):
            # same as run_generic_tau_online_plan: without the target tool's argument
            # names, section 178's identity logic mistakes a NEW address for an account
            # detail and replaces it with the one on file (section 211 bug, retail_088)
            tool = environment.tools.get_tools().get((plan.get("operation_argument_fact_bundle") or {}).get("tool_name") or "")
            return set((tool.params.model_json_schema().get("properties") or {})) if tool else set()

        scripted = [
            GenericDeterministicTauUser(
                plan,
                identity_profile=(_retail_identity_profile(plan, environment.tools.db) if domain == "retail" else None),
                target_tool_argument_names=_target_args(plan),
            )
            for plan in components
        ]
        instructions = build_compound_instructions(
            [s.initial_message for s in scripted], [s.known_facts_message for s in scripted], mode,
        )
        tools = environment.get_user_tools() if getattr(environment, "user_tools", None) is not None else None
        user = CompoundUserSimulator(llm=user_llm, instructions=instructions, tools=tools or None,
                                     llm_args=dict(llm_args or {"num_retries": 0}))
        user._compound_instructions = instructions
        return user

    return factory


def merged_plan(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The plan the conversation runs under: the first component, with every
    component's initial_state_patch.agent_data merged (the pairing guarantees they
    touch disjoint records). Its own oracle result is ignored; each component is
    evaluated separately afterwards."""
    from copy import deepcopy

    base = deepcopy(dict(components[0]))
    merged: dict[str, Any] = {}
    for plan in components:
        patch = (plan.get("initial_state_patch") or {}).get("agent_data") or {}
        for table, rows in patch.items():
            if isinstance(rows, dict):
                merged.setdefault(table, {}).update(deepcopy(rows))
            else:
                merged[table] = deepcopy(rows)
    base["initial_state_patch"] = {**(base.get("initial_state_patch") or {}), "agent_data": merged or None}
    return base
