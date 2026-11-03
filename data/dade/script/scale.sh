#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ..

dataset="gist"
name="gist_base.fvecs"
size=10000
log_file="${ORIGINAL_DIR}/log/${dataset}_pca_size${size}.txt"

python pca.py -d ${dataset} -n ${name} -s ${size}> ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} Pca FAIL"
    exit 1
fi

log_file="${ORIGINAL_DIR}/log/${dataset}_getEpsilon_size${size}.txt"

python getEpsilon.py -d ${dataset} -n ${name} -s ${size}> ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} GetEpsilon FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"