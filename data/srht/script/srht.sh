# new_add: SRHT preprocessing (not in upstream dco-benchmark)
#!/bin/bash
# Single-dataset SRHT preprocessing.
# Edit the variables below for your dataset.

ORIGINAL_DIR=$(pwd)
cd ..

dataset="XUltra"                  # **
name="msmacro_base-001.bin"       # **
query="query.fvecs"               # ** optional; pads to power-of-2 D, writes Squery.fvecs

log_file="${ORIGINAL_DIR}/log/${dataset}_srht.txt"

# Create the log directory so that the output redirection does not fail
mkdir -p "${ORIGINAL_DIR}/log"

python randomized.py -d ${dataset} -n ${name} -q ${query} > ${log_file} 2>&1

if [ $? -ne 0 ]; then
    echo "${dataset} SRHT preprocessing FAIL"
    exit 1
fi
echo "${dataset} SRHT preprocessing OK"

cd "$ORIGINAL_DIR"
