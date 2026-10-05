"""mining_v2 orchestrator — three sources, then de-duplication.

  1. schema mining      (schema_miner)   grounded ARG + ORDER  -> structural
                                         (deterministic facts + LLM read of the
                                          schema text + data-flow sequencing)
  2. prompt mining      (prompt_miner)   grounded, 4 kinds      -> stated
  3. domain-knowledge   (domain_knowledge) LLM prior, 4 kinds   -> hypothesized
                                         (per-tool + cross-tool)
  -> dedup reconciles overlaps across the three sources.

Produces a SpecSet of natural-language business rules with kind, source
evidence, and confidence (structural / stated / hypothesized).
"""

from __future__ import annotations

import logging

from src.core.types import AgentUnderTest
from src.mining_v2.common import build_coverage_grid
from src.mining_v2.dedup import dedup
from src.mining_v2.domain_knowledge import mine_domain_knowledge
from src.mining_v2.prompt_miner_v2 import mine_prompt   # snapshot: v1 prompt_miner removed, v2 is the default
from src.mining_v2.spec import KINDS, SpecSet
from src.mining_v2.schema_miner import mine_schema

logger = logging.getLogger(__name__)


def mine(
    agent: AgentUnderTest,
    *,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    gap_fill: bool = True,
    prompt_miner=None,) -> SpecSet:
    logger.info("mining_v2 on domain=%s (%d tools)", agent.domain, len(agent.tools))

    # 1. schema — grounded ARG + ORDER (deterministic facts + LLM read of schema)
    struct_specs = mine_schema(agent, model=model, temperature=temperature)
    # 2. prompt — grounded, all four kinds
    _mine_prompt = prompt_miner or mine_prompt      # prompt_miner_v2.mine_prompt is a drop-in
    prompt_specs, prompt_report = _mine_prompt(agent, model=model, temperature=temperature)
    # 3. domain knowledge — hypothesized (known = schema + prompt, so it does not restate them)
    dk_specs = []
    if gap_fill:
        dk_specs = mine_domain_knowledge(
            agent, prompt_specs + struct_specs, model=model, temperature=temperature,
        )

    # Reconcile overlaps across the three sources.
    raw = struct_specs + prompt_specs + dk_specs
    all_specs = dedup(raw, model=model, temperature=temperature)
    n_dups = len(raw) - len(all_specs)

    # renumber spec_ids globally for stable identity
    for i, s in enumerate(all_specs):
        s.spec_id = f"{agent.domain or 'x'}_{i:03d}_{s.kind.lower()}"

    metadata = {
        "n_total": len(all_specs),
        "n_struct": len(struct_specs),
        "n_prompt": len(prompt_specs),
        "n_domain_knowledge": len(dk_specs),
        "n_duplicates_removed": n_dups,
        "n_multi_clause": sum(1 for s in all_specs if len(s.clauses) > 1),
        "n_multi_tool": sum(1 for s in all_specs if len(s.relevant_tools) > 1),
        "kind_breakdown": {k: sum(1 for s in all_specs if s.kind == k) for k in KINDS},
        "confidence_breakdown": {
            c: sum(1 for s in all_specs if s.confidence == c)
            for c in ("structural", "stated", "hypothesized")
        },
        "coverage_empty_state_order_cells": len(build_coverage_grid(agent, all_specs)),
        "prompt_report": prompt_report,
    }

    logger.info("mining_v2 done: %d specs %s", len(all_specs), metadata["kind_breakdown"])
    return SpecSet(domain=agent.domain or "unknown", specs=all_specs, metadata=metadata)
