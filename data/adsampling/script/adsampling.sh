#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ..

dataset="XUltra"            # **
name="msmacro_base-001.bin"    # **

log_file="${ORIGINAL_DIR}/log/${dataset}_randomized.txt"

# Create the log directory so that the output redirection does not fail
mkdir -p "${ORIGINAL_DIR}/log"

python randomized.py -d ${dataset} -n ${name} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} Randomized FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR"