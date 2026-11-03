#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit
dataset="wiki"                                                    #**
rotation=""
dataset_path="/home/dataset/${dataset}/${rotation}wiki_base.fvecs"     #**
query_path="/home/dataset/${dataset}/wikipedia_query.bin"           #**
operator=1                                                        #**
index_path="/home/dataset/${dataset}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
index_new_path="/home/dataset/${dataset}/hnsw_norm_${rotation}${dataset}_M16_efConstruction500_operator0.index"
log_file="${ORIGINAL_DIR}/log/hnsw_${dataset}_operator${operator}_normalize_log.txt"
./hnsw_normalize -n ${dataset_path} -q ${query_path} -i ${index_path} -e ${index_new_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Rota FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit