#!/bin/bash

ORIGINAL_DIR=$(pwd)

# datasets=(arxiv-for-fanns-large)
# base=(arxiv-for-fanns-large_base.fvecs)

datasets=("XUltra" "OpenAI3072" "trevi")
base=("msmacro_base-001.bin" "openai3072_base.bin" "trevi_base.fvecs")

cd ../../../build/src || exit

mkdir -p "${ORIGINAL_DIR}/log"

for data in "${!datasets[@]}"; do
  dataset_path="/data/dataset/${datasets[$data]}/${base[$data]}"
  learning_path="/data/dataset/${datasets[$data]}/${datasets[$data]}_train.fvecs"
  sampling=10000
  log_file="${ORIGINAL_DIR}/log/${datasets[$data]}_sampling.txt"

  ./hnsw_sample -n "${dataset_path}" -l "${learning_path}" -s "${sampling}" > ${log_file} 2>&1

  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} Sample FAIL"
      exit 1
  fi

done

K=100
for data in "${!datasets[@]}"; do
  dataset_path="/data/dataset/${datasets[$data]}/${base[$data]}"
  learning_path="/data/dataset/${datasets[$data]}/${datasets[$data]}_train.fvecs"
  ground_truth_path="/data/dataset/${datasets[$data]}/${datasets[$data]}_train_ground_truth.ivecs"
  log_file="${ORIGINAL_DIR}/log/${datasets[$data]}_train_groundtruth.txt"

  ./hnsw_bruteforce -k "${K}" -n "${dataset_path}" -l "${learning_path}" -g "${ground_truth_path}" > "${log_file}" 2>&1

  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} Tarin Set GroundTruth FAIL"
      exit 1
  fi

done

cd "${ORIGINAL_DIR}/.." || exit 1
for data in "${!datasets[@]}"; do
  learn_ground_truth="/data/dataset/${datasets[$data]}/${datasets[$data]}_train_ground_truth.ivecs"
  log_file="${ORIGINAL_DIR}/log/${datasets[$data]}_ddc_pca_generate.txt"

  python pca.py -d "${datasets[$data]}" -n "${base[$data]}" -l "${learn_ground_truth}" > ${log_file} 2>&1

  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} PCA FAIL"
      exit 1
  fi

done

cd "$ORIGINAL_DIR" || exit

