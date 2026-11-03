#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit
dataset="gist"                                                    #**
rotation="L"
size=10000
dataset_path="/home/dataset/${dataset}/train_${size}/${rotation}gist_base.fvecs"     #**
query_path="/home/dataset/${dataset}/query.fvecs"           #**
operator=6                                                        #**
index_path="/home/dataset/${dataset}/hnsw_${dataset}_M16_efConstruction500_operator0.index"
index_new_path="/home/dataset/${dataset}/train_${size}/hnsw_${rotation}${dataset}_M16_efConstruction500_operator0.index"
./hnsw_rotation -n ${dataset_path} -q ${query_path} -i ${index_path} -e ${index_new_path} -o ${operator}

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Rota FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit