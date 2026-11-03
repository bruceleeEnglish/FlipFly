#!/bin/bash
# Run all three §5.2 parameter / theory-verification experiments
# end-to-end. Designed for the server layout
#     /data/dataset/<dataset>/...
# Override DATA_ROOT to point elsewhere.
#
# Final outputs (under output/):
#     projection_timings.csv          (raw C++ timings)
#     projection_scaling.pdf          (§5.2.1 fig:projection-scaling)
#     variance_verification.pdf       (§5.2.2 fig:variance-verification)
#     walsh_flatness.tex              (§5.2.2 tab:walsh-flatness)
#     headroom.pdf                    (§5.2.3 fig:headroom)
#     headroom_table.tex              (§5.2.3 tab:headroom-table)
#     proxy_agreement.tex             (§5.2.3 tab:proxy-agreement)

set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/data/dataset}"
N_REPEATS="${N_REPEATS:-1000}"
N_WARMUP="${N_WARMUP:-50}"
M_Q="${M_Q:-400}"          # number of (base, query) pairs sampled in B
R="${R:-256}"               # number of independent SRHT draws per pair in B
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p output

# echo "=================================================================="
# echo "A  projection-scaling  (§5.2.1)"
# echo "=================================================================="
# bash cpp/build.sh
# cpp/bench_projection "$N_REPEATS" "$N_WARMUP" > output/projection_timings.csv
# $PYTHON_BIN plot_projection_scaling.py --csv output/projection_timings.csv

echo
echo "=================================================================="
echo "B  variance-verification  (§5.2.2)"
echo "=================================================================="
$PYTHON_BIN plot_variance_verification.py \
    --data-root "$DATA_ROOT" \
    --M-q "$M_Q" \
    --R "$R"

# echo
# echo "=================================================================="
# echo "C  headroom + proxy-agreement  (§5.2.3)"
# echo "=================================================================="
# $PYTHON_BIN plot_headroom.py --data-root "$DATA_ROOT"

# echo
# echo "=================================================================="
# echo "all done. outputs under $(pwd)/output/"
# ls -1 output/
