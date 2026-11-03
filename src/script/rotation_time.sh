#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src
dataset="gist"                                                    #**
K=100                                                             # 20, 100
rotation=""
dataset_path="/home/dataset/${dataset}/${rotation}gist_base.fvecs"     #**
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
operator=2                                                        #**
index_path="/home/dataset/${dataset}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
log_file="${ORIGINAL_DIR}/rotation_without_SIMD/hnsw_${dataset}_K${K}_operator${operator}_rotation_time_log_NO_SIMD.txt"
./rotation_without_SIMD -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} Rotation FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"