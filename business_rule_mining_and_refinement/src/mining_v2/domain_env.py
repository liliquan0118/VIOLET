"""Access to a τ²-bench domain's database and clock, without building an agent."""

from __future__ import annotations

import importlib
import os
import re
import sys


def ensure_tau2(tau2_root: str) -> None:
    tau2_src = os.path.join(os.path.abspath(tau2_root), "src")
    if tau2_src not in sys.path:
        sys.path.insert(0, tau2_src)


def load_domain_db(tau2_root: str, domain: str):
    """The domain's pydantic DB object, without building an agent.

    Grounding and entity queries need the real models (so a bad condition fails loudly),
    but not an LLM."""
    ensure_tau2(tau2_root)
    env_mod = importlib.import_module(f"tau2.domains.{domain}.environment")
    return env_mod.get_environment().tools.db


_NOW_RE = re.compile(r"current time is ([^\n.]+)", re.IGNORECASE)


def current_time_from_policy(policy: str) -> str:
    """The domain's clock as stated in its policy ('The current time is ...').

    Judging dates against the real present instead of this is a silent source
    of wrong results: the τ² airline database lives in 2024."""
    match = _NOW_RE.search(policy or "")
    return match.group(1).strip() if match else "(not stated in the policy)"
