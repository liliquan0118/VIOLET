#!/usr/bin/env bash
# Install τ²-bench at the commit used in the paper into <repo>/tau2-bench, plus the dependencies of both stages.
# Run it inside the Python environment you will use (Python 3.12 or 3.13, as τ²-bench requires).
#   bash setup_tau2.sh
set -euo pipefail
cd "$(dirname "$0")"

TAU2_COMMIT=1901a301961cbbe3fd11f3e84a2a376530c759e3
PYTHON=${PYTHON:-python3}

"$PYTHON" -c 'import sys; v = sys.version_info[:2]; sys.exit(0 if (3, 12) <= v < (3, 14) else f"Python {v[0]}.{v[1]} found; tau2-bench needs 3.12 or 3.13 (set PYTHON=...)")'

if [ ! -d tau2-bench/.git ]; then
  git clone https://github.com/sierra-research/tau2-bench.git tau2-bench
fi
git -C tau2-bench fetch --quiet origin "$TAU2_COMMIT" || true
git -C tau2-bench checkout --quiet "$TAU2_COMMIT"

"$PYTHON" -m pip install -e tau2-bench
"$PYTHON" -m pip install -r requirements.txt

# stage 1 expects the checkout at business_rule_mining_and_refinement/tau2-bench
[ -e business_rule_mining_and_refinement/tau2-bench ] || ln -s ../tau2-bench business_rule_mining_and_refinement/tau2-bench

"$PYTHON" -c 'import tau2; print("tau2-bench ready:", tau2.__file__)'
echo "export TAU2_BENCH_DIR=$PWD/tau2-bench"
