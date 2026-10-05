#!/usr/bin/env bash
# MANTRA baseline: run the released MANTRA tau2 suites on the agents under test.
# The harness is MANTRA's run_eval.py; run_eval_violet.py is a copy of it that also records the final agent message
# and token usage (place it in the MANTRA checkout).
# usage: run_mantra_baseline_v0_1.sh <deepseek|qwen> <tau2-airline|tau2-retail|tau2-telecom> <retries> <workers> [extra run_eval args, e.g. --max-cases 1]
# environment:
#   MANTRA_DIR        MANTRA checkout
#   MANTRA_PYTHON     python of MANTRA's environment (default: python)
#   DEEPSEEK_API_FILE / QWEN_API_FILE   API key files (base URL on line 1, key on a line starting with sk-;
#                                       the Qwen file also has the model id on a line starting with qwen)
set -u
M=$1; D=$2; R=$3; W=$4; shift 4
: "${MANTRA_DIR:?set MANTRA_DIR to the MANTRA checkout}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/outputs/mantra_baseline_v0_1
if [[ $M == deepseek ]]; then
  : "${DEEPSEEK_API_FILE:?set DEEPSEEK_API_FILE}"
  export OPENAI_BASE_URL=$(sed -n 1p "$DEEPSEEK_API_FILE" | tr -d '[:space:]'); export OPENAI_API_KEY=$(grep '^sk-' "$DEEPSEEK_API_FILE" | tr -d '[:space:]'); MID=deepseek-v4-flash
else
  : "${QWEN_API_FILE:?set QWEN_API_FILE}"
  export OPENAI_BASE_URL=$(sed -n 1p "$QWEN_API_FILE" | tr -d '[:space:]'); export OPENAI_API_KEY=$(grep '^sk-' "$QWEN_API_FILE" | tr -d '[:space:]'); MID=$(grep '^qwen' "$QWEN_API_FILE" | tr -d '[:space:]')
fi
cd "$MANTRA_DIR"; export TAU2_BENCH_SRC=$MANTRA_DIR/tau2_vendor
"${MANTRA_PYTHON:-python}" run_eval_violet.py --suite test_suites/$D --env environments/$D --provider openai --model $MID \
  --retries $R --max-workers $W --out $OUT/${M}__${D}${TAG:-}.json "$@"
