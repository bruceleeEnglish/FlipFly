#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src
dataset="YouTube"                                                    #**
rotation="P"
dataset_path="/home/dataset/${dataset}/${rotation}YouTube_base.fvecs"     #**
query_path="/home/dataset/${dataset}/query.bin"           #**
operator=2                                                        #**
index_path="/home/dataset/${dataset}/ivf_${dataset}_C4096_operator0.index"
index_new_path="/home/dataset/${dataset}/ivf_${rotation}${dataset}_C4096_operator0.index"
log_file="${ORIGINAL_DIR}/log/ivf_${dataset}_operator${operator}_rotation_log.txt"
./ivf_rotation -n ${dataset_path} -q ${query_path} -i ${index_path} -e ${index_new_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Rota FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"