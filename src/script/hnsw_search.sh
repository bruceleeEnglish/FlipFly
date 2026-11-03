#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src
dataset="gist"                                                    #**
init=64
delta=32
K=100                                                             # 20, 100
rotation=""
dataset_path="/home/dataset/${dataset}/${rotation}gist_base.fvecs"     #**
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
operator=1                                                        #**
index_path="/home/dataset/${dataset}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
log_file="${ORIGINAL_DIR}/test/hnsw_${dataset}_K${K}_operator${operator}_search_log_delta${delta}_init${init}.txt"
./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"