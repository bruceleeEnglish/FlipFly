#!/bin/bash

# shellcheck disable=SC2054
# datasets=("arxiv-for-fanns-large" "XUltra" "OpenAI3072" "trevi")
# query=("query.fvecs" "query.bin" "query.bin" "query.fvecs")

datasets=("OpenAI3072")
query=("query.bin")

# rotate=("" "" "P" "O" "P" "L" "L" "Q")
# new_add: SRHT family — 18 SRHTADSampling (S), 19 SRHTADSamplingIP (S), 20 SRHTPCAADSampling (T)
rotate=([0]="" [1]="" [2]="P" [3]="O" [4]="P" [5]="L" [18]="S" [19]="S" [20]="T")
# KNN=(20 100)
KNN=(60)

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit

for data in "${!datasets[@]}"; do
  for op in 18 20; do
  # for op in 0 1 2 3 4 5 18 20; do
    for K in ${KNN[*]}; do
      # new_add: the SRHT family (op 18/19/20) must use the padded Squery.fvecs
      if [ "$op" -ge 18 ] && [ "$op" -le 20 ]; then
        qname="Squery.fvecs"
      else
        qname="${query[$data]}"
      fi
      dataset_path="/data/dataset/${datasets[$data]}/${rotate[$op]}${datasets[$data]}_base.fvecs"
      query_path="/data/dataset/${datasets[$data]}/${qname}"
      ground_truth_path="/data/dataset/${datasets[$data]}/gt_256.ivecs"
      index_path="/data/dataset/${datasets[$data]}/hnsw_${rotate[$op]}${datasets[$data]}_M16_efConstruction500_operator0.index"
      log_file="${ORIGINAL_DIR}/query_performance/hnsw_${datasets[$data]}_K${K}_operator${op}_search_log.txt"

      mkdir -p "${ORIGINAL_DIR}/query_performance"

      ./hnsw_search -k "${K}" -n "${dataset_path}" -q "${query_path}" -g "${ground_truth_path}" -i "${index_path}" -o "${op}" >"${log_file}" 2>&1

      if [ $? -ne 0 ]; then
          echo "${datasets[$data]} HNSW Search FAIL"
          exit 1
      fi
    done
  done
done

cd "$ORIGINAL_DIR" || exit