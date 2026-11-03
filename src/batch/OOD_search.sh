#!/bin/bash

# shellcheck disable=SC2054
datasets=("laion" "text2image")
query1=("query_img.fvecs" "query_rand.fvecs")
truth1=("gt_img_img_256.ivecs" "gt_256_rand.ivecs")
query2=("query_text.fvecs" "query.public.100K.fbin")
truth2=("gt_img_text_256.ivecs" "gt_256.ivecs")
rotate=("" "" "P" "O" "P" "L" "L" "Q")
K=100

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit

for data in {0..1}; do
  for op in {0..7}; do
    dataset_path="/home/dataset/${datasets[$data]}/${rotate[$op]}${datasets[$data]}_base.fvecs"
    query_path="/home/dataset/${datasets[$data]}/${query1[$data]}"
    ground_truth_path="/home/dataset/${datasets[$data]}/${truth1[$data]}"
    index_path="/home/dataset/${datasets[$data]}/hnsw_${rotate[$op]}${datasets[$data]}_M16_efConstruction500_operator0.index"
    log_file="${ORIGINAL_DIR}/OOD/hnsw_${datasets[$data]}_K${K}_operator${op}_search_log_id.txt"

    ./hnsw_search -k "${K}" -n "${dataset_path}" -q "${query_path}" -g "${ground_truth_path}" -i "${index_path}" -o "${op}" >"${log_file}" 2>&1

    if [ $? -ne 0 ]; then
      echo "${datasets[$data]} HNSW ID Search FAIL"
      exit 1
    fi
  done
done

for data in {0..1}; do
  for op in {0..7}; do
    dataset_path="/home/dataset/${datasets[$data]}/${rotate[$op]}${datasets[$data]}_base.fvecs"
    query_path="/home/dataset/${datasets[$data]}/${query2[$data]}"
    ground_truth_path="/home/dataset/${datasets[$data]}/${truth2[$data]}"
    index_path="/home/dataset/${datasets[$data]}/hnsw_${rotate[$op]}${datasets[$data]}_M16_efConstruction500_operator0.index"
    log_file="${ORIGINAL_DIR}/OOD/hnsw_${datasets[$data]}_K${K}_operator${op}_search_log_ood.txt"

    ./hnsw_search -k "${K}" -n "${dataset_path}" -q "${query_path}" -g "${ground_truth_path}" -i "${index_path}" -o "${op}" >"${log_file}" 2>&1

    if [ $? -ne 0 ]; then
      echo "${datasets[$data]} HNSW OOD Search FAIL"
      exit 1
    fi
  done
done

cd "$ORIGINAL_DIR" || exit
