#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src/ || exit
dataset="gist"
rotation="Q"
K=100
operator=7
size=600000
start=600000
end=700000
dataset_path="/home/dataset/${dataset}/train_${size}/Qgist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/tmp_Q${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gist_size${end}_gt.ivecs"
#ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/update/hnsw_${rotation}${dataset}_operator${operator}_update_start${start}_end${end}_log.txt"

./hnsw_update_test -k ${K} -o ${operator} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -s ${start} -e ${end} -i ${index_path} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Update FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit