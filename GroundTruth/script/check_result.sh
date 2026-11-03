#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/GroundTruth
dataset="XUltra"
K=256
dataset_path="/data/dataset/${dataset}/msmacro_base-001.bin"
query_path="/data/dataset/${dataset}/query.bin"
ground_truth_path="/data/dataset/${dataset}/gt_${K}.ivecs"
metric=0  # 0: L2, 1: IP

./check_result -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -m ${metric}

if [ $? -ne 0 ]; then
    echo "${dataset} CHECK FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"