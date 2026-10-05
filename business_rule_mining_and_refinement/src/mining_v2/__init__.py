"""mining_v2 — business-rule spec mining from three sources.

Independent of src/mining (v1). Mines only specs (no test-case material) from:
  1. tool schema        → deterministic ARG constraints (struct_miner)
  2. system prompt      → grounded rules, all kinds (prompt_miner)
  3. domain knowledge   → LLM-prior hypothesized rules (domain_knowledge)
then de-duplicates across the three (dedup). Does not assume a policy document.
"""

from src.mining_v2.pipeline import mine
from src.mining_v2.spec import Evidence, Spec, SpecSet

__all__ = ["mine", "Spec", "SpecSet", "Evidence"]
