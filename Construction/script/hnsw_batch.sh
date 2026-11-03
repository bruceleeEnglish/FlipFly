#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/Construction || exit 1

# datasets=("trevi" "XUltra" "OpenAI3072")
# base=("trevi_base.fvecs" "XUltra_base.fvecs" "OpenAI3072_base.fvecs")
# datasets=()
# base=()

datasets=(arxiv-for-fanns-large)
base=(arxiv-for-fanns-large_base.fvecs)

rotation="T" # 
# rotation="" # use this line for the baseline methods
#rotate=("" "" "P" "O" "P" "L" "L" "Q") FDScanning, PDScanning, PDScanning+, ADSampling, DADE, DDCres, DDCpca, DDCopq
# new_add: SRHT family rotation prefixes
#   "S" -> SRHTADSampling (op 18), SRHTADSamplingIP (op 19)   uses srht/S.fvecs
#   "T" -> SRHTPCAADSampling (op 20)                          uses srht_pca/S.fvecs

M=16
efConstruction=500
operator=0 # L2: 0, IP: 8 # distance used during the build only; building every index with L2 (0) is fine because the DCO operator matters only at query time
# L2: FDScanning 0, PDScanning 1, PDScanning+ 2, ADSampling 3, DADE 4, DDCres 5, DDCpca 6, DDCopq 7
# new_add: SRHTADSampling 18, SRHTADSamplingIP 19, SRHTPCAADSampling 20

# Create the log directory
mkdir -p "${ORIGINAL_DIR}/log"

for data in "${!datasets[@]}"; do
  dataset_path="/data/dataset/${datasets[$data]}/${rotation}${base[$data]}"
  index_path="/data/dataset/${datasets[$data]}/hnsw_${rotation}${datasets[$data]}_M${M}_efConstruction${efConstruction}_operator${operator}.index"
  log_file="${ORIGINAL_DIR}/log/hnsw_${rotation}${datasets[$data]}_M${M}_efConstruction${efConstruction}_operator${operator}_log.txt"

  ./hnsw_index -m "${M}" -e "${efConstruction}" -o "${operator}" -n "${dataset_path}" -i "${index_path}" > "${log_file}" 2>&1

  if [ $? -ne 0 ]; then
      echo "${datasets[$data]} Build HNSW Index FAIL"
      exit 1
  fi
done

cd "$ORIGINAL_DIR" || exit 1
