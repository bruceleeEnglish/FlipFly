#!/bin/bash
# Batch-preprocess all datasets for SRHT-ADSampling.
# Mirrors data/adsampling/script/batch.sh.

ORIGINAL_DIR=$(pwd)
cd .. || exit 1

# datasets=("XUltra" "OpenAI3072" "trevi")
# base=("msmacro_base-001.bin" "openai3072_base.bin" "trevi_base.fvecs")
# query=("query.bin" "query.bin" "query.fvecs")  # new_add: pass it to -q and randomized.py writes Squery.fvecs

datasets=(arxiv-for-fanns-large)
base=(arxiv-for-fanns-large_base.fvecs)
query=("query.fvecs")


mkdir -p "${ORIGINAL_DIR}/log"

for data in "${!datasets[@]}"; do
  log_file="${ORIGINAL_DIR}/log/${datasets[$data]}_srht.txt"

  python randomized.py -d "${datasets[$data]}" -n "${base[$data]}" -q "${query[$data]}" > "${log_file}" 2>&1

  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} SRHT preprocessing FAIL"
      exit 1
  fi
  echo "${datasets[$data]} SRHT preprocessing OK"
done

cd "$ORIGINAL_DIR" || exit
