#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../../build/data/ddc ||  exit

K=100
recall=0.995
dataset="YouTube"
type="ivf"
method="pca"
dataset_path="/home/zzlin/dataset/${dataset}/L${dataset}_base.fvecs"
query_path="/home/zzlin/dataset/${dataset}/${dataset}_train.fvecs"
ground_truth_path="/home/zzlin/dataset/${dataset}/${dataset}_train_ground_truth.ivecs"
index_path="/home/zzlin/dataset/${dataset}/ivf_L${dataset}_C4096_operator0.index"
output_path="/home/zzlin/dataset/${dataset}/ddc/${dataset}_ivf_pca_${K}_training_set.fvecs"
linear_path="/home/zzlin/dataset/${dataset}/ddc/linear_${type}${K}_${method}.txt"
log_file="${ORIGINAL_DIR}/log/ivf_${dataset}_K${K}_logger_hnsw_pca1.txt"

./logger_ivf_pca -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${output_path} -r ${recall}  -l ${linear_path} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} ivf Training Set Generate FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit
##########################
cd ..

log_file="${ORIGINAL_DIR}/log/ivf_${dataset}_ddc_pca_linear_${K}.txt"

python linear.py -d ${dataset} -k ${K} -m ${method} -i ${index_path} -t ${output_path} -s ${linear_path} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} Linear FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit
##########################
cd ../../../build/data/ddc ||  exit

log_file="${ORIGINAL_DIR}/log/ivf_${dataset}_K${K}_logger_hnsw_pca2.txt"

./logger_ivf_pca -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${output_path} -r ${recall}  -l ${linear_path} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} ivf Linear Model Correct FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit
