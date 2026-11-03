#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/Construction || exit
dataset="YouTube"                                                    #**
K=100                                                             # 20, 100
rotation=""
dataset_path="/home/zzlin/dataset/${dataset}/${rotation}YouTube_base.fvecs"     #**
query_path="/home/zzlin/dataset/${dataset}/query.bin"           #**
ground_truth_path="/home/zzlin/dataset/${dataset}/gt_256.ivecs"
operator=1                                                        #**
index_path="/home/zzlin/dataset/${dataset}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator${operator}.index"
log_file="${ORIGINAL_DIR}/search/hnsw_${dataset}_K${K}_operator${operator}_search_index_log.txt"
./hnsw_search_index -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit