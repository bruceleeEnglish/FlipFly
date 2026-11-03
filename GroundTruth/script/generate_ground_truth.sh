#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/GroundTruth
dataset="arxiv-for-fanns-large"
K=256
dataset_path="/data/dataset/${dataset}/arxiv-for-fanns-large_base.fvecs"
query_path="/data/dataset/${dataset}/query.fvecs"
ground_truth_path="/data/dataset/${dataset}/gt_${K}.ivecs"
metric=0  # 0: L2, 1: IP
log_file="${ORIGINAL_DIR}/log/${dataset}_gt_${K}_log.txt"

# Create the log directory so that the output redirection does not fail
mkdir -p "${ORIGINAL_DIR}/log"

./generate_ground_truth -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -m ${metric}  > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} Generate Ground Truth FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"