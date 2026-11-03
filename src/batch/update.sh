#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit
dataset="gist"
K=100
size=600000
start=600000

dataset="gist"
rotate=("" "" "P" "O" "P" "L" "L" "Q")
ends=(600000 700000 800000 900000 1000000)
K=100

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit

for op in {0..7}; do
  for end in ${ends[*]}; do
    dataset_path="/home/dataset/${dataset}/train_${size}/gist_base.fvecs"
    index_path="/home/dataset/${dataset}/train_${size}/hnsw_${dataset}_M16_efConstruction500_operator0.index"
    query_path="/home/dataset/${dataset}/query.fvecs"           #**
    ground_truth_path="/home/dataset/${dataset}/gist_size${end}_gt.ivecs"
    if [ $end == 1000000 ]; then
      ground_truth_path="/home/dataset/${dataset}/gt_256.ivecs"
    fi
    log_file="${ORIGINAL_DIR}/log/hnsw_${rotate[${op}]}${dataset}_operator${op}_update_start${start}_end${end}_log.txt"

    if [ $op == 5 ]; then
        ./hnsw_update_ip -k "${K}" -o "${op}" -n "${dataset_path}" -q "${query_path}" -g "${ground_truth_path}" -s "${start}" -e "${end}" -i "${index_path}" > "${log_file}" 2>&1
    elif [ $op == 7 ]; then
        ./hnsw_update_opq -k "${K}" -o "${op}" -n "${dataset_path}" -q "${query_path}" -g "${ground_truth_path}" -s "${start}" -e "${end}" -i "${index_path}" > "${log_file}" 2>&1
    else
        ./hnsw_update -k "${K}" -o "${op}" -n "${dataset_path}" -q "${query_path}" -g "${ground_truth_path}" -s "${start}" -e "${end}" -i "${index_path}" > "${log_file}" 2>&1
    fi

    if [ $? -ne 0 ]; then
        echo "${dataset} HNSW Update FAIL"
        exit 1
    fi
  done
done

cd "$ORIGINAL_DIR" || exit