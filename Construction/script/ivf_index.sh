#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/Construction
dataset="YouTube"
rotation=""
C=4096
operator=0
dataset_path="/home/zzlin/dataset/${dataset}/${rotation}YouTube.bin"
centroid_path="/home/zzlin/dataset/${dataset}/${rotation}${dataset}_centroid_${C}.fvecs"
index_path="/home/zzlin/dataset/${dataset}/ivf_${rotation}${dataset}_C${C}_operator${operator}.index"
log_file="${ORIGINAL_DIR}/log/ivf_${rotation}${dataset}_C${C}_operator${operator}_log.txt"

./ivf_index  -o ${operator} -n ${dataset_path} -i ${index_path} -c ${centroid_path}> ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} Build ivf Index FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"