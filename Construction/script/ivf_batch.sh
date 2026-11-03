#!/bin/bash

ORIGINAL_DIR=$(pwd)

# shellcheck disable=SC2054
datasets=("glove100" "sift" "gist" "trevi")
base=("glove100.bin" "sift_base.fvecs" "gist_base.fvecs" "trevi_base.fvecs")
rotation=""
C=4096
operator=0

cd ../../data/ || exit 1

for data in {0..3}; do
  log_file="${ORIGINAL_DIR}/log/${datasets[$data]}_ivf_centroid_build.txt"

  python ivf.py -d "${datasets[$data]}" -n "${base[$data]}" -k ${C} > "${log_file}" 2>&1
  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} IVF Centroid FAIL"
      exit 1
  fi
done

cd ../build/Construction || exit 1

for data in {0..3}; do
  dataset_path="/home/dataset/${datasets[$data]}/${rotation}YouTube.bin"
  centroid_path="/home/dataset/${datasets[$data]}/${rotation}${datasets[$data]}_centroid_${C}.fvecs"
  index_path="/home/dataset/${datasets[$data]}/ivf_${rotation}${datasets[$data]}_C${C}_operator${operator}.index"
  log_file="${ORIGINAL_DIR}/log/ivf_${rotation}${datasets[$data]}_C${C}_operator${operator}_log.txt"

  ./ivf_index  -o "${operator}" -n "${dataset_path}" -i "${index_path}" -c "${centroid_path}"> "${log_file}" 2>&1

  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} Build ivf Index FAIL"
      exit 1
  fi
done

cd "$ORIGINAL_DIR" || exit 1