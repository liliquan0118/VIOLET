# AgenTest

Artifact for the paper **"AgenTest: Mining Business Rules to Test LLM Agents for Business Logic Bugs"**.

AgenTest tests tool-using LLM agents for *business logic bugs*: executions in which the preconditions of a business
rule hold but the agent's behaviour violates it. It works in two stages:

1. **Business rule mining and refinement** (paper §3.2). Rules are mined from the system prompt, the tool definitions
   and LLM domain knowledge, then deduplicated and filtered. Each one is refined into scenario-specific
   Given–When–Then (GWT) specifications.
2. **Specification-driven testing** (paper §3.3). For each refined specification, AgenTest constructs a test driver:
   an environment state that satisfies *Given* plus a simulated user that issues *When*. It exercises the agent over
   multi-turn conversations, using an interaction-strategy library and a bug-experience memory. A hybrid oracle
   checks *Then* against messages, tool calls and state changes, using deterministic checks for structured evidence
   and LLM judgment for natural-language conditions.

The evaluation targets the airline, retail and telecom agents of [τ²-bench](https://github.com/sierra-research/tau2-bench)
with two underlying models (DeepSeek-V4-Flash, Qwen-3.8-Flash).

This repository contains the implementation, the mined and refined specifications, and the inputs needed to run the
tests of the paper: the bound test-driver plans and the strategy cases. The run records of the evaluation (conversation
transcripts and verdicts) are not part of this release.

## Repository layout

| Path | Content |
|---|---|
| `business_rule_mining_and_refinement/` | Stage 1 (§3.2): business rule mining and refinement, see [its README](business_rule_mining_and_refinement/README.md) |
| `specs/{airline,retail,telecom}/` | Refined GWT specifications produced by stage 1 (166 + 207 + 178 = 551): `*_gwt_v5_user_requirements.json` (refined specifications) and `*_gwt_v5_conditions_matches.json` (Given conditions grounded on the data model, and matching objects) |
| `spec_driven_testing/` | Stage 2 (§3.3): specification-driven testing and the baselines, see [its README](spec_driven_testing/README.md) |

## Setup

Use Python 3.12 or 3.13 (τ²-bench requires it; the experiments used 3.12). In that environment run

```bash
bash setup_tau2.sh
export TAU2_BENCH_DIR=$PWD/tau2-bench
```

It clones [τ²-bench](https://github.com/sierra-research/tau2-bench) at the commit used in the paper
(`1901a301961cbbe3fd11f3e84a2a376530c759e3`) into `tau2-bench/`, installs it and the dependencies of both stages
(`requirements.txt`), and links the checkout where stage 1 expects it. τ²-bench is not modified.

**API keys.** Stage 1 reads its keys from `business_rule_mining_and_refinement/.env`. Stage 2 scripts that call a
model take an API key file via `--api-file` (simulated user and judge) and `--agent-api-file` (agent under test); the
file has the OpenAI-compatible base URL on its first line and the key on a line starting with `sk-`. The paper used
GPT-4.1 for stage 1, DeepSeek (`https://api.deepseek.com`, model `deepseek-v4-flash`) for the simulated user and the
API judge, and DeepSeek-V4-Flash or Qwen-3.8-Flash (`qwen3.8-flash`) as the agent's model.

## Running AgenTest

1. Mine and refine the business rules of an agent into `specs/`: see
   [`business_rule_mining_and_refinement/README.md`](business_rule_mining_and_refinement/README.md).
2. Test the agent against them: see [`spec_driven_testing/README.md`](spec_driven_testing/README.md).

## License

Released under the MIT License (see `LICENSE`). τ²-bench, whose policies, tool definitions and data appear in the
specifications and test inputs, is distributed under its own MIT license.
