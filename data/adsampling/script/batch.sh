#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd .. || exit 1

# datasets=("XUltra" "OpenAI3072" "trevi")
# base=("msmacro_base-001.bin" "openai3072_base.bin" "trevi_base.fvecs")
datasets=(arxiv-for-fanns-large)
base=(arxiv-for-fanns-large_base.fvecs)

mkdir -p "${ORIGINAL_DIR}/log"

for data in "${!datasets[@]}"; do
  log_file="${ORIGINAL_DIR}/log/${datasets[$data]}_randomized.txt"

  python randomized.py -d "${datasets[$data]}" -n "${base[$data]}" > "${log_file}" 2>&1

  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} Randomized FAIL"
      exit 1
  fi
done

cd "$ORIGINAL_DIR" ||  exit