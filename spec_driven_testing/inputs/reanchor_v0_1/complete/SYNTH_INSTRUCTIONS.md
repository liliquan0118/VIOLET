# Case synthesis: complete-round cases (reuse log section 510)

Your batch is gen_batches.json[<batch>]["case_ids"] in this folder
(outputs/reanchor_v0_1/complete).

For each case id:
1. Read queue/pending/<case_id>.prompt.md and follow it exactly, INCLUDING the ANCHORING block at its end (the case
   must test that one branch: keep its Given and When, change only the interaction or rebind objects that satisfy the
   same Given; violation_check = anchored_oracle = the branch's Then on your objects; add "target_branch").
   It names the tools file, the tau2 DB and the policy. Inspect the DB with small Python snippets (read only); never
   modify tau2 files.
2. Write ONLY the JSON object to queue/cases/<case_id>.json (write queue/cases/.<case_id>.tmp first, then mv).
3. Validate (from <repo>):
   PYTHONPATH=src /opt/miniconda3/envs/tau-bench/bin/python3 scripts/prepare_complete_supplements_v0_1.py --check <case ids...>
   Fix every PROBLEMS line until the case prints OK. The check replays each arm's expected_calls on the patched DB.
4. If the strategy cannot be realised on this branch faithfully, still write the most faithful valid case that tests
   the branch's Then (anchoring wins over the strategy sketch) and say so in "notes"; do not skip the case.

Rules: do not run the agent or any paid API; do not touch other batches' cases; do not open other outputs folders.
Put helper scripts in your own scratchpad.
Report: one line per case: case_id | OK | one-phrase summary of the trigger scene (+ any notes).
