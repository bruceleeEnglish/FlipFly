#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ..

dataset="wiki"
name="wikipedia_base.bin.crop_nb_1000000"
normalize="1"
#log_file="${ORIGINAL_DIR}/log/${dataset}_pca.txt"
log_file="${ORIGINAL_DIR}/log/${dataset}_pca_${normalize}.txt"

python pca.py -d ${dataset} -n ${name} -l ${normalize} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} Pca FAIL"
    exit 1
fi

#log_file="${ORIGINAL_DIR}/log/${dataset}_getEpsilon.txt"
log_file="${ORIGINAL_DIR}/log/${dataset}_getEpsilon_${normalize}.txt"

python getEpsilon.py -d ${dataset} -n ${name} -l ${normalize} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} GetEpsilon FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit