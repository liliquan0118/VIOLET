# Stage 1: business rule mining and refinement

This folder implements §3.2 of the paper. It mines business rules for a τ²-bench agent from three sources (tool
definitions, system prompt, LLM domain knowledge), deduplicates and filters them, and refines them into
Given–When–Then (GWT) specifications. The Given clauses are grounded on the environment's data model. The outputs
in `../specs/` were produced with GPT-4.1.

## Setup

All scripts resolve paths relative to **this folder**:

- τ²-bench is expected at `./tau2-bench`; `setup_tau2.sh` at the repository root links it there. Otherwise pass
  `--tau-bench-path`.
- API keys are read from `./.env`:

  | Variable | Use |
  |---|---|
  | `OPENAI_API_KEY`, `OPENAI_MODEL` | default provider and model of every stage (`gpt-4.1` in the paper) |
  | `DEEPSEEK_API_KEY` | `--provider deepseek` |
  | `CLAUDE_API_KEY` | `--provider claude` |
  | `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL` | `--provider qwen` (OpenAI-compatible endpoint) |

- Outputs go to `./ExperimentResult/` unless a path is given.

## Pipeline

`<S>` below is the mined-spec file without `.json`. `<G>` is the GWT output prefix.

| Step | Script | Paper | Output |
|---|---|---|---|
| 1 | `mine_v2.py --domain D --output <S>.json` | §3.2.1 mining from tool definitions, system prompt (with a sentence-level coverage audit) and domain knowledge; deduplication | `<S>.json` plus per-source files and `<S>_prompt_audit.json` |
| 1a | `rebind_specs.py --domain D --specs <S>.json` | §3.2.1 missing tool bindings | `<S>_rebound_final.json` |
| 1b | `audit_coverage_v2.py --domain D --specs <S>.json` | optional: redo the prompt coverage audit only | `<S>_prompt_audit_v2.json` |
| 2 | `formalize_v2.py --domain D --specs <S>.json --direction v2 --out-prefix <S>_gwt_v5 --model gpt-4.1` | §3.2.2 scenario derivation (disjunctive conditions, negation of the whole condition) and GWT encoding | `<G>.json`, `<G>_pos.json`, `<G>_neg.json`, `<G>_ir.json` |
| 3 | `ground_and_requirements.py --domain D --gwt <G>.json --conditions <G>_conditions.json` | §3.3.1 splitting Given into environment / user-side / ownership clauses; environment clauses encoded as `{table, path, op, value}` and validated against the data model; user-side requirements extracted from When/Then | `<G>_conditions.json`, `<G>_user_requirements.json` |
| 4 | `query_conditions.py --domain D --conditions <G>_conditions.json --output <G>_conditions_matches.json` | §3.3.1 searching the environment for objects that satisfy the conditions and ownership relation | `<G>_conditions_matches.json` |
| 5 | `filter_testability.py --domain D --specs <S>.json --out <S>_testability.json --filtered <S>_testable.json` | §3.2.1 filtering (the paper describes it as part of mining; the implementation runs it last, and `build_testable.py` then drops the GWTs of filtered specs): removes rules that rely on concepts absent from the environment | audit and filtered set |
| — | `build_vocab.py` | helper: export the data-model vocabulary used to validate grounding | |

`../specs/<domain>/*_gwt_v5_user_requirements.json` and `*_gwt_v5_conditions_matches.json` are the outputs of
steps 3 and 4 for the three domains.

## Code

| Path | Content |
|---|---|
| `src/agent_interface/tau2_bench.py` | loads a τ²-bench agent (system prompt, tool schemas, tool source) as an `AgentUnderTest` |
| `src/core/` | LLM client and provider configuration, shared types, result paths |
| `src/mining_v2/schema_miner.py`, `prompt_miner_v2.py`, `domain_knowledge.py`, `dedup.py`, `pipeline.py` | the three mining sources and deduplication |
| `src/mining_v2/testability.py` | applicability filter |
| `src/mining_v2/formalize_v2.py` (with `formalize.py`, `formalize_pos.py`, `formalize_neg.py`) | scenario derivation and GWT encoding |
| `src/mining_v2/grounding.py`, `grounding_v2.py` | condition language, validation against the data model, grounding of Given clauses |
| `src/mining_v2/conversation_side.py`, `user_requirements.py` | user-side conditions, trigger and violation supplies |
| `src/mining_v2/entity_query.py` | object search (`matched` / `anchored` / `makeable`) |
| `src/mining_v2/domain_env.py` | loads a domain's database and reads its clock from the policy |
