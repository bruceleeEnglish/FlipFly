#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ..

dataset="gist"
name="gist_base.fvecs"
size=10000
learn_ground_truth="/home/zzlin/dataset/${dataset}/${dataset}_train_ground_truth.ivecs"
log_file="${ORIGINAL_DIR}/log/${dataset}_ddc_pca_generate_${size}.txt"

python pca.py -d ${dataset} -n ${name} -l ${learn_ground_truth}> ${log_file} 2>&1
#python pca.py -d ${dataset} -n ${name} -l ${learn_ground_truth} -s ${size}> ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} PCA FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"
