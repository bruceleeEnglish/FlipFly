#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit
dataset="gist"
K=100
rotation="P"
operator=4
#####################################################
size=100
dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/log3/hnsw_${rotation}${dataset}_operator${operator}_size${size}_log_sta.txt"

./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi
#####################################################
size=500
dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/log3/hnsw_${rotation}${dataset}_operator${operator}_size${size}_log_sta.txt"

./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi
#####################################################
size=1000
dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/log3/hnsw_${rotation}${dataset}_operator${operator}_size${size}_log_sta.txt"

./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi
#####################################################
size=5000
dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/log3/hnsw_${rotation}${dataset}_operator${operator}_size${size}_log_sta.txt"

./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi
#####################################################
size=10000
dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/log3/hnsw_${rotation}${dataset}_operator${operator}_size${size}_log_sta.txt"

./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi
#####################################################
size=50000
dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/log3/hnsw_${rotation}${dataset}_operator${operator}_size${size}_log_sta.txt"

./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi
#####################################################
size=100000
dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/log3/hnsw_${rotation}${dataset}_operator${operator}_size${size}_log_sta.txt"

./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi
#####################################################
size=200000
dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/log3/hnsw_${rotation}${dataset}_operator${operator}_size${size}_log_sta.txt"

./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi
#####################################################
size=300000
dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/log3/hnsw_${rotation}${dataset}_operator${operator}_size${size}_log_sta.txt"

./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi
#####################################################
size=600000
dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
index_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
log_file="${ORIGINAL_DIR}/log3/hnsw_${rotation}${dataset}_operator${operator}_size${size}_log_sta.txt"

./hnsw_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Search FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"