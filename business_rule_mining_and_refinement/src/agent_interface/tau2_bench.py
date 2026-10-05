"""Adapter that loads a τ²-bench domain into AgentUnderTest.

Bypasses τ²-bench's heavy __init__ import chain (litellm, fastapi, etc.)
by registering a lightweight stub module and importing only what we need:
domain environments, toolkits, and data models.
"""

from __future__ import annotations

import inspect
import os
import sys
import types
from typing import Any

from src.core.types import AgentUnderTest, ToolParameter, ToolSchema

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT_TEMPLATE = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


def _ensure_tau2_importable(tau2_root: str) -> None:
    """Add τ²-bench src to sys.path and register a stub for the tau2 package."""
    tau2_src = os.path.join(os.path.abspath(tau2_root), "src")
    if tau2_src not in sys.path:
        sys.path.insert(0, tau2_src)
    if "tau2" not in sys.modules or hasattr(sys.modules["tau2"], "__tau2_stub__"):
        stub = types.ModuleType("tau2")
        stub.__path__ = [os.path.join(tau2_src, "tau2")]
        stub.__package__ = "tau2"
        stub.__tau2_stub__ = True
        sys.modules["tau2"] = stub


def _parse_parameters(schema: dict, tool_name: str) -> list[ToolParameter]:
    """Extract ToolParameter list from a JSON schema's properties."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    params: list[ToolParameter] = []
    for pname, pinfo in props.items():
        ptype = pinfo.get("type", "string")
        if ptype == "array":
            item_type = pinfo.get("items", {}).get("type", "object")
            ptype = f"array[{item_type}]"
        params.append(ToolParameter(
            name=pname,
            type=ptype,
            description=pinfo.get("description", ""),
            enum_values=pinfo.get("enum"),
            required=pname in required,
            belongs_to_tool=tool_name,
        ))
    return params


def get_environment(tau2_root: str, domain: str):
    """Create and return a τ²-bench Environment for the given domain."""
    _ensure_tau2_importable(tau2_root)

    import importlib
    supported = ["retail", "airline", "telecom"]
    if domain not in supported:
        raise ValueError(
            f"Unknown domain '{domain}'. Available: {supported}"
        )
    env_module = importlib.import_module(f"tau2.domains.{domain}.environment")
    return env_module.get_environment()


def load_tau2_bench_agent(tau2_bench_root: str, domain: str) -> AgentUnderTest:
    """Load a τ²-bench domain as an AgentUnderTest.

    Args:
        tau2_bench_root: Path to the cloned tau2-bench repository.
        domain: "retail", "airline", or "telecom".
    """
    env = get_environment(tau2_bench_root, domain)

    tools = env.get_tools()
    tool_schemas: list[ToolSchema] = []
    for tool in tools:
        oai = tool.openai_schema
        func = oai.get("function", oai)
        name = func["name"]
        desc = func.get("description", "")
        params_schema = func.get("parameters", {})

        source_code = None
        try:
            source_code = inspect.getsource(tool._func)
        except (OSError, TypeError):
            pass

        tool_schemas.append(ToolSchema(
            name=name,
            description=desc,
            parameters=_parse_parameters(params_schema, name),
            raw_schema=oai,
            source_code=source_code,
        ))

    policy = env.get_policy()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        agent_instruction=AGENT_INSTRUCTION,
        domain_policy=policy,
    )

    return AgentUnderTest(
        tools=tool_schemas,
        system_prompt=system_prompt,
        rules=[],
        domain=domain,
    )
