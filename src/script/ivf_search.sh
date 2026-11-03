#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit
delta=32
init=32
dataset="gist"                                                    #**
K=100                                                             # 20, 100
rotation="L"
dataset_path="/home/dataset/${dataset}/${rotation}gist_base.fvecs"     #**
query_path="/home/dataset/${dataset}/query.fvecs"           #**
ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
operator=5                                                        #**
index_path="/home/dataset/${dataset}/ivf_${rotation}${dataset}_C4096_operator0.index"
log_file="${ORIGINAL_DIR}/test/ivf_${dataset}_K${K}_operator${operator}_delta${delta}_init${init}_search_log.txt"
./ivf_search -k ${K} -n ${dataset_path} -q ${query_path} -g ${ground_truth_path} -i ${index_path} -o ${operator} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} ivf Search FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit