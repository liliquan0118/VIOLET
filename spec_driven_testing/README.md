# Stage 2: specification-driven testing

This folder implements §3.3 of the paper. It tests a τ²-bench agent against the refined specifications in
`../specs/`, produced by `../business_rule_mining_and_refinement/`. For each specification it constructs a test
driver (environment state and simulated user), exercises the agent over multi-turn conversations with interaction
strategies guided by a bug-experience memory, and checks the result with a hybrid oracle.

Run every command from this folder:

```bash
cd spec_driven_testing
export PYTHONPATH=src:scripts
export TAU2_BENCH_DIR=../tau2-bench          # the τ²-bench checkout (default: <repo>/tau2-bench)
```

## Layout

| Path | Content |
|---|---|
| `src/agentest/` | library: test driver construction (`driver/generic_tau_*_v2_*`, `compiler/`), online execution on τ²-bench (`driver/generic_tau_online_v1.py`), oracle (`oracle/`, `compiler/`), strategy selection (`strategy/`) |
| `scripts/` | command-line entry points (below) |
| `configs/` | interaction-strategy library (`strategy_library_v0_2.json`), oracle judgment families, runtime observation bindings, τ²-bench reference data |
| `inputs/` | inputs of the paper's test runs: bound driver plans, strategy applicability per specification, selected (specification, strategy) pairs and synthesized strategy cases |
| `prompts/`, `schemas/` | LLM prompt templates and JSON schemas of intermediate artifacts |
| `outputs/` | written by the runs (not part of the release) |

Module and input-directory names carry version suffixes (`_v0_1`, `_v1`, …) because the stored plans and cases refer
to them; they are kept unchanged.

## Pipeline

**Test driver construction (§3.3.1).** Each refined specification is bound to concrete environment objects that
satisfy its Given (`src/agentest/driver/generic_tau_v2_fixture_binder_v1.py`, with per-domain configuration and
bundle resolvers in `generic_tau_{airline,retail,telecom}_v2_*`). It is then lowered into a bound driver plan with
its oracle checks (`generic_tau_v2_bound_plan_adapter_v1.py`). The bound plans used in the paper are in
`inputs/generic_bound_driver_plans_v0_1_{airline_v2,retail,telecom}/plans.json`. Plans for specifications added
later are in `inputs/as_written_backfill_v0_1/cases.json`.

**Round 1: each specification as written, with the default user behaviour.**

```bash
python scripts/run_generic_tau_baselines_v0_1.py \
  --bound-plans inputs/generic_bound_driver_plans_v0_1_retail/plans.json --domain retail \
  --api-file DS_KEY --model deepseek-v4-flash [--agent-api-file QWEN_KEY] \
  --output-dir outputs/<run> --per-case-agent-calls 30 --approved-total-agent-calls N \
  --telecom-policy manual --enable-semantic-judge --triage queue --concurrency 16   # --dry-run lists the jobs
```

**Rounds 2–3: interaction strategies with the bug-experience memory (§3.3.2).**

1. *Specification intake.* Read the refined specifications of stage 1 into one intake per domain:
   ```bash
   python scripts/intake_v5_sources_v0_1.py \
     --user-requirements ../specs/airline/specs_airline_gpt41_full_run6_gwt_v5_user_requirements.json \
     --condition-matches ../specs/airline/specs_airline_gpt41_full_run6_gwt_v5_conditions_matches.json \
     --output-dir outputs/v5_source_intake_airline          # likewise retail, telecom
   ```
2. *Strategy applicability.* An LLM classifies every specification against the strategy library
   (`configs/strategy_library_v0_1.json`, then the revised `strategy_library_v0_2.json`): its rule structure, the
   applicable strategies, how a violation is observed and whether the tool enforces the rule. Answers are checked
   mechanically (quotes must appear verbatim in the policy or the tool source).
   ```bash
   python scripts/spec_inventory_v0_1.py --out-dir outputs/spec_inventory_v0_1 --backend api --model M --api-file KEY
   python scripts/spec_inventory_v0_2.py --out-dir outputs/spec_inventory_v0_2 --previous outputs/spec_inventory_v0_1 \
     --backend api --model M --api-file KEY
   ```
   The inventory used in the paper is in `inputs/spec_inventory_v0_2/`.
3. *Select pairs and synthesize cases.* `scripts/generate_variants_v0_1.py` selects (specification, strategy) pairs
   (`select_pairs` in `src/agentest/strategy/variant_generation_v0_1.py`) and writes one synthesis prompt per pair to
   `<out-dir>/queue/pending/`. An LLM (we used Claude Opus 5.5 as an agentic coding assistant) answers each prompt
   with a case file in `<out-dir>/queue/cases/`. Re-running the script validates every case (tool and object
   references, state patch, replay of the compliant tool sequence in a fresh environment) and writes
   `<out-dir>/cases.json`.
   ```bash
   python scripts/generate_variants_v0_1.py --inventory inputs/spec_inventory_v0_2 --out-dir outputs/round_a \
     --backend queue --per-spec 1 --available-probes S08.d S08.e S10.g S03.g S07.g
   ```
4. *Run and judge the round* (judging is described below):
   ```bash
   python scripts/run_strategy_round_v0_1.py --cases outputs/round_a/cases.json --output-dir outputs/round_a/main \
     --api-file DS_KEY --model deepseek-v4-flash [--agent-api-file QWEN_KEY] \
     --reps '{"trigger":1,"control":0}' --telecom-policy manual --approved-total-agent-calls N --concurrency 24
   ```
5. *Update the bug experience and select the next round.* `scripts/summarize_strategy_round_v0_1.py` aggregates the
   reviewed verdicts by strategy, rule source, domain and tool enforcement into
   `violation_check/round_summary.json`. `--round2-from` then ranks strategies that exposed violations first and
   demotes strategies that repeatedly failed (`select_pairs_round2`):
   ```bash
   python scripts/summarize_strategy_round_v0_1.py --run-dir outputs/round_a/main --cases outputs/round_a/cases.json \
     --inventory inputs/spec_inventory_v0_2
   python scripts/generate_variants_v0_1.py --inventory inputs/spec_inventory_v0_2 --out-dir outputs/round_b \
     --backend queue --round2-from outputs/round_a --available-probes S08.d S08.e S10.g S03.g S07.g
   ```

With the inventory in `inputs/spec_inventory_v0_2/` and the arguments above, the selection of steps 3 and 5
reproduces the pairs of the paper's strategy rounds. The cases of the paper's final rounds 2–3 are in
`inputs/reanchor_v0_1/complete/cases.json` (selection in `inputs/reanchor_v0_1/complete_plan_{ds,qwen}.json`,
synthesis with `scripts/prepare_complete_supplements_v0_1.py` following
`inputs/reanchor_v0_1/complete/SYNTH_INSTRUCTIONS.md`) and can be run directly with
`run_strategy_round_v0_1.py --cases inputs/reanchor_v0_1/complete/cases.json`.

**Hybrid bug detection (§3.3.3).** Deterministic checks on tool calls and states are part of the bound plan and run
online. Natural-language assertions are judged by `scripts/check_violations_v0_1.py`:

```bash
python scripts/check_violations_v0_1.py --run-dir outputs/<run> --tier api --model deepseek-v4-flash --api-file DS_KEY
python scripts/check_violations_v0_1.py --run-dir outputs/<run> --tier queue   # after the queued reviews are answered
```

`--tier api` judges every run once. `--tier queue` routes every reported violation, every low-confidence verdict
and a sample of the remaining verdicts to a second, blind review (prompts under `violation_check/pending/`,
instructions in `inputs/reanchor_v0_1/complete/JUDGE_INSTRUCTIONS.md`). It then writes
`violation_check/final_verdicts.json`.

## Baselines

The scripts for the baselines of the paper are in `scripts/` as well.

| Baseline | Code |
|---|---|
| τ²-bench task suite | `scripts/run_tau2_standard_v0_1.py`, `scripts/triage_tau2_standard_v0_1.py` |
| IntellAgent | run in an [IntellAgent](https://github.com/plurai-ai/intellagent) checkout; export with `scripts/export_intellagent_dialogs_v0_1.py` (`INTELLAGENT_DIR`), review with `scripts/review_intellagent_v0_1.py` |
| MANTRA | `scripts/run_mantra_baseline_v0_1.sh` (`MANTRA_DIR`), review with `scripts/review_mantra_v0_1.py` |
