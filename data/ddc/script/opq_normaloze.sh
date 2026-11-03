#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ..

dataset="wiki"
name="wikipedia_base.bin.crop_nb_1000000"
bit=8
normalize=1

log_file="${ORIGINAL_DIR}/log/${dataset}_ddc_opq_generate.txt"

python opq.py -d ${dataset} -n ${name} -b ${bit} -l ${normalize}> ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} OPQ FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit
