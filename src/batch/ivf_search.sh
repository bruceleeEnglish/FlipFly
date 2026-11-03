#!/bin/bash

# shellcheck disable=SC2054
datasets=("sift" "gist")
query=("query.fvecs" "query.fvecs")
rotate=("" "" "P" "O" "P" "L" "L" "Q")
K=100

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit

for data in {0..1}; do
  for op in {0..7}; do
    dataset_path="/home/dataset/${datasets[$data]}/${rotate[$op]}${datasets[$data]}_base.fvecs"
    query_path="/home/dataset/${datasets[$data]}/${query[$data]}"
    ground_truth_path="/home/dataset/${datasets[$data]}/gt_256.ivecs"
    index_path="/home/dataset/${datasets[$data]}/ivf_${rotate[$op]}${datasets[$data]}_C4096_operator0.index"
    log_file="${ORIGINAL_DIR}/ivf_search/hnsw_${datasets[$data]}_K${K}_operator${op}_search_log.txt"

    ./ivf_search -k "${K}" -n "${dataset_path}" -q "${query_path}" -g "${ground_truth_path}" -i "${index_path}" -o "${op}" >"${log_file}" 2>&1

    if [ $? -ne 0 ]; then
      echo "${datasets[$data]} HNSW OOD Search FAIL"
      exit 1
    fi
  done
done

cd "$ORIGINAL_DIR" || exit