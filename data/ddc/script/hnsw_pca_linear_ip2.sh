#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../../build/data/ddc ||  exit

K=100
recall=0.995
dataset="wiki"
type="hnsw"
method="pca"
dataset_path="/home/zzlin/dataset/${dataset}/P${dataset}_base.fvecs"
query_path="/home/zzlin/dataset/${dataset}/${dataset}_train.fvecs"
ground_truth_path="/home/zzlin/dataset/${dataset}/${dataset}_train_ground_truth_ip.ivecs"
index_path="/home/zzlin/dataset/${dataset}/hnsw_${dataset}_M16_efConstruction500_operator8.index"
new_index_path="/home/zzlin/dataset/${dataset}/hnsw_norm_P${dataset}_M16_efConstruction500_operator8.index"
output_path="/home/zzlin/dataset/${dataset}/ddc/${dataset}_hnsw_pca_${K}_training_set_ip2.fvecs"
linear_path="/home/zzlin/dataset/${dataset}/ddc/linear_${type}${K}_${method}_ip2.txt"
log_file="${ORIGINAL_DIR}/log/hnsw_${dataset}_K${K}_logger_hnsw_pca1_ip2.txt"

./logger_hnsw_pca_ip2 -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${output_path} -r ${recall}  -l ${linear_path} -e ${new_index_path}> ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Training Set Generate FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit
##########################
cd ..

log_file="${ORIGINAL_DIR}/log/${dataset}_ddc_pca_linear_ip2.txt"

python linear.py -d ${dataset} -k ${K} -m ${method} -i ${index_path} -t ${output_path} -s ${linear_path} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} Linear FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit
##########################
cd ../../../build/data/ddc ||  exit

log_file="${ORIGINAL_DIR}/log/hnsw_${dataset}_K${K}_logger_hnsw_pca2_ip2.txt"

./logger_hnsw_pca_ip2 -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${output_path} -r ${recall}  -l ${linear_path} -e ${new_index_path} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Linear Model Correct FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit
