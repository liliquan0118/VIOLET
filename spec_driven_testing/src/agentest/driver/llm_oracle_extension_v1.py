"""LLM oracle for test cases whose Then compiled to no decidable check
(reuse log section 206).

Before this, such a bound plan had an empty oracle_plan.checks and was marked
not_scriptable_deterministic_driver (section 117): 48 of 485 test cases never
ran. User decision (2026-09-25):
  - Thens the compiler cannot turn into an observable check -> judged by the LLM;
  - "may" permissions -> rewritten as "must";
  - "if and only if" -> one direction, chosen for the branch's scenario;
  - multi-step procedures whose steps were non-decisive -> judged by the LLM.
The per-branch requirement text (and which rule produced it) lives in
configs/llm_oracle_uncompilable_thens_v0_1.json so every rewrite is auditable.

The check has exactly the shape of the existing whole-transcript semantic
checks (evaluation_mode semantic_judge, extractor semantic_event_matcher_
deferred), so generic_tau_online_v1._evaluate_semantic_event_matcher_deferred
evaluates it unchanged. It never touches a plan that has any compiled check.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from agentest.compiler.artifacts import content_sha256

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/llm_oracle_uncompilable_thens_v0_1.json"
PROGRAM_SOURCE = "llm_oracle_extension_v1"


@lru_cache(maxsize=1)
def _entries() -> dict[str, Any]:
    if not CONFIG.exists():
        return {}
    return json.loads(CONFIG.read_text(encoding="utf-8"))["branches"]


def llm_oracle_checks(branch_id: str) -> list[dict[str, Any]]:
    """The LLM-judged check(s) for a branch listed in the config; [] otherwise."""
    entry = _entries().get(branch_id)
    if entry is None:
        return []
    requirement_id = f"{branch_id}::LLM01"
    binding_id = f"{requirement_id}::RB01"
    expected = {"operator": entry["operator"], "quantifier": "exists", "requires_observation": True}
    diagnostics = [{"code": "llm_oracle_for_uncompilable_then", "rewrite": entry["rewrite"]}]
    binding = {
        "schema_version": "agentspectesting.runtime-observation-binding/v0.1",
        "binding_id": binding_id,
        "requirement_id": requirement_id,
        "expectation_id": f"{requirement_id}::E01",
        "branch_id": branch_id,
        "binding_status": "semantic_deferred",
        "runtime_binding": {
            "extractor_kind": "semantic_event_matcher_deferred",
            "event_stream": "canonical_tau_events",
            "semantic_unit_ids": [],
            "requirement_text": entry["requirement_text"],
            "candidate_event_kinds": ["assistant_message", "assistant_tool_call"],
            "program_source": PROGRAM_SOURCE,
        },
        "expected_observation": expected,
        "diagnostics": diagnostics,
    }
    binding["binding_fingerprint"] = content_sha256(binding)
    contract = {
        "schema_version": "agentspectesting.oracle-evaluator-contract/v0.1",
        "evaluator_contract_id": f"{binding_id}::EC01",
        "binding_id": binding_id,
        "requirement_id": requirement_id,
        "expectation_id": f"{requirement_id}::E01",
        "branch_id": branch_id,
        "evaluator_status": "deferred",
        "program": {"evaluator_kind": "unavailable"},
        "expected_observation": expected,
        "diagnostics": diagnostics,
        "source_binding_fingerprint": binding["binding_fingerprint"],
    }
    contract["evaluator_contract_fingerprint"] = content_sha256(contract)
    return [{
        "binding_id": binding_id,
        "evaluation_mode": "semantic_judge",
        "evaluator_contract": contract,
        "runtime_observation_binding": binding,
    }]


def is_llm_oracle_check(check: Mapping[str, Any]) -> bool:
    runtime = (check.get("runtime_observation_binding") or {}).get("runtime_binding") or {}
    return runtime.get("program_source") == PROGRAM_SOURCE
