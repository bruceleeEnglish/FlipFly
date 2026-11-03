#!/bin/bash

ORIGINAL_DIR=$(pwd)

datasets=("glove100" "sift" "gist" "OpenAIArXiv" "trevi")

for data in {0..4}; do
  cd ../../../build/data/ddc || exit
  K=100
  recall=0.995
  type="hnsw"
  method="pca"
  dataset_path="/home/dataset/${datasets[$data]}/L${datasets[$data]}_base.fvecs"
  query_path="/home/dataset/${datasets[$data]}/${datasets[$data]}_train.fvecs"
  ground_truth_path="/home/dataset/${datasets[$data]}/${datasets[$data]}_train_ground_truth.ivecs"
  index_path="/home/dataset/${datasets[$data]}/hnsw_L${datasets[$data]}_M16_efConstruction500_operator0.index"
  output_path="/home/dataset/${datasets[$data]}/ddc/${datasets[$data]}_hnsw_pca_${K}_training_set.fvecs"
  linear_path="/home/dataset/${datasets[$data]}/ddc/linear_${type}${K}_${method}.txt"
  log_file="${ORIGINAL_DIR}/log/hnsw_${datasets[$data]}_K${K}_logger_hnsw_pca1.txt"

  ./logger_hnsw_pca -k ${K} -n "${dataset_path}" -q "${query_path}" -g "${ground_truth_path}" -i "${index_path}" -o "${output_path}" -r "${recall}"  -l "${linear_path}" > "${log_file}" 2>&1

  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} HNSW Training Set Generate FAIL"
      exit 1
  fi

  cd "$ORIGINAL_DIR" || exit
  ##########################
  cd ..

  log_file="${ORIGINAL_DIR}/log/${datasets[$data]}_ddc_pca_linear_${K}.txt"

  python linear.py -d "${datasets[$data]}" -k "${K}" -m "${method}" -i "${index_path}" -t "${output_path}" -s "${linear_path}" > "${log_file}" 2>&1

  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} Linear FAIL"
      exit 1
  fi

  cd "$ORIGINAL_DIR" || exit
  ##########################
  cd ../../../build/data/ddc ||  exit

  log_file="${ORIGINAL_DIR}/log/hnsw_${datasets[$data]}_K${K}_logger_hnsw_pca2.txt"

  ./logger_hnsw_pca -k ${K} -n "${dataset_path}" -q "${query_path}" -g "${ground_truth_path}" -i "${index_path}" -o "${output_path}" -r "${recall}"  -l "${linear_path}" > "${log_file}" 2>&1

  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} HNSW Linear Model Correct FAIL"
      exit 1
  fi

  cd "$ORIGINAL_DIR" || exit

done