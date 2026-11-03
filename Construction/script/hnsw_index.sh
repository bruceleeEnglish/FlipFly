#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/Construction
dataset="XUltra"
rotation="O"
M=16
efConstruction=500
operator=0
dataset_path="/data/dataset/${dataset}/OXUltra_base.fvecs"
index_path="/data/dataset/${dataset}/hnsw_${rotation}${dataset}_M${M}_efConstruction${efConstruction}_operator${operator}.index"
log_file="${ORIGINAL_DIR}/log/hnsw_${rotation}${dataset}_M${M}_efConstruction${efConstruction}_operator${operator}_log.txt"

# Create the log directory so that the output redirection does not fail
mkdir -p "${ORIGINAL_DIR}/log"

./hnsw_index -m ${M} -e ${efConstruction} -o ${operator} -n ${dataset_path} -i ${index_path} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} Build HNSW Index FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"