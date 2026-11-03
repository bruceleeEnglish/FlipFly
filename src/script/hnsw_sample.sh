#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit
dataset="wiki"
dataset_path="/home/dataset/${dataset}/wikipedia_base.bin.crop_nb_1000000"
learning_path="/home/dataset/${dataset}/${dataset}_train.fvecs"
sampling=10000

./hnsw_sample -n ${dataset_path} -l ${learning_path} -s ${sampling}

if [ $? -ne 0 ]; then
    echo "${dataset} Sample FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit