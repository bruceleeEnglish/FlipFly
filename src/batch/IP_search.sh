#!/bin/bash

# shellcheck disable=SC2054
datasets=("glove100" "wiki" "OpenAIArXiv" "msmacro")
query=("query.fvecs" "query_rand.fvecs" "openai_query.bin" "query.bin")
truth=("gt_256_ip.ivecs" "gt_256_rand_ip.ivecs" "gt_256_ip.ivecs" "gt_256_ip.ivecs")
norm=("" "norm_" "norm_" "norm_" "norm_" "" "norm_" "norm_")
OP2IP=(8 9 10 11 12 13 16 17)
rotate=("" "" "P" "O" "P" "L" "P" "Q")
K=100

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit

for data in {0..4}; do
  for op in {0..7}; do
    dataset_path="/home/dataset/${datasets[$data]}/${rotate[$op]}${datasets[$data]}_base.fvecs"
    query_path="/home/dataset/${datasets[$data]}/${query[$data]}"
    ground_truth_path="/home/dataset/${datasets[$data]}/${truth[$data]}"
    index_path="/home/dataset/${datasets[$data]}/hnsw_${norm[$op]}${rotate[$op]}${datasets[$data]}_M16_efConstruction500_operator8.index"
    log_file="${ORIGINAL_DIR}/IP/hnsw_${datasets[$data]}_K${K}_operator${OP2IP[$op]}_search_log.txt"

    ./hnsw_search -k "${K}" -n "${dataset_path}" -q "${query_path}" -g "${ground_truth_path}" -i "${index_path}" -o "${OP2IP[$op]}" >"${log_file}" 2>&1

    if [ $? -ne 0 ]; then
      echo "${datasets[$data]} HNSW IP Search FAIL"
      exit 1
    fi
  done
done

cd "$ORIGINAL_DIR" || exit