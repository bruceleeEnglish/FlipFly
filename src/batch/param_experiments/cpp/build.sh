#!/bin/bash
# Compile the projection-scaling microbenchmark.
#
# Run from anywhere; the script chdirs to its own directory and
# resolves the Eigen include path against the project's third_party.

set -euo pipefail
cd "$(dirname "$0")"

# Repo layout assumed (relative to this script):
#   <repo>/src/batch/param_experiments/cpp/  <-- here
#   <repo>/third_party/eigen/                <-- Eigen header-only
EIGEN_PATH="$(cd ../../../../third_party/eigen && pwd)"

if [ ! -d "$EIGEN_PATH/Eigen" ]; then
    echo "ERROR: Eigen headers not found at $EIGEN_PATH" >&2
    echo "       (expected ../../../../third_party/eigen relative to this script)" >&2
    exit 1
fi

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O3 -DNDEBUG -march=native}"

echo "compiling with $CXX $CXXFLAGS  -I $EIGEN_PATH"
$CXX $CXXFLAGS -I "$EIGEN_PATH" bench_projection.cpp -o bench_projection

echo "built: $(pwd)/bench_projection"
echo ""
echo "next steps:"
echo "  ./bench_projection           > ../output/projection_timings.csv"
echo "  cd .. && python3 plot_projection_scaling.py --csv output/projection_timings.csv"
