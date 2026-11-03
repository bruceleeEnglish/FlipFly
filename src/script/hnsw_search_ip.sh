#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit
dataset="wiki"                                                    #**
K=100                                                             # 20, 100
rotation="P"
dataset_path="/home/dataset/${dataset}/${rotation}wiki_base.fvecs"     #**
query_path="/home/dataset/${dataset}/wikipedia_query.bin"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256_ip.ivecs"
operator=14                                                        #**
#index_path="/home/dataset/${dataset}/hnsw_norm_${rotation}${dataset}_M16_efConstruction500_operator8.index"
index_path="/home/dataset/${dataset}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator8.index"
log_file="${ORIGINAL_DIR}/log/hnsw_${dataset}_K${K}_operator${operator}_search_norm_log.txt"
./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" ||  exit