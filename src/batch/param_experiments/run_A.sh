#!/bin/bash
# §5.2.1 projection-scaling: compile C++ microbench, run it, plot.
#
# All three steps in one go. Outputs:
#     output/projection_timings.csv
#     output/projection_scaling.pdf
#
# Override defaults via env vars:
#     N_REPEATS=1000 N_WARMUP=50 bash run_A.sh

set -euo pipefail
cd "$(dirname "$0")"

N_REPEATS="${N_REPEATS:-100}"
N_WARMUP="${N_WARMUP:-50}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p output

echo "[1/3] building cpp/bench_projection ..."
bash cpp/build.sh

echo
echo "[2/3] running benchmark (n_repeats=$N_REPEATS, n_warmup=$N_WARMUP) ..."
cpp/bench_projection "$N_REPEATS" "$N_WARMUP" > output/projection_timings.csv

echo
echo "[3/3] plotting ..."
$PYTHON_BIN plot_projection_scaling.py --csv output/projection_timings.csv

echo
echo "done. outputs:"
echo "  output/projection_timings.csv"
echo "  output/projection_scaling.pdf"
