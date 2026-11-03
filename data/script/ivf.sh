#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ..

dataset="YouTube"
name="YouTube.bin"
K=4096
log_file="${ORIGINAL_DIR}/log/${dataset}_ivf_centroid_build.txt"

python ivf.py -d ${dataset} -n ${name} -k ${K} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} IVF Centroid FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"
