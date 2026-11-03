#!/bin/bash

ORIGINAL_DIR=$(pwd)

dataset="gist"
size=1000
name="gist_base.fvecs"
dataset_path="/home/zzlin/dataset/${dataset}/${name}"
learning_path="/home/zzlin/dataset/${dataset}/train_${size}/${dataset}_train.fvecs"
ground_truth_path="/home/zzlin/dataset/${dataset}/train_${size}/${dataset}_train_ground_truth.ivecs"
sampling=10000
M=16
efConstruction=500
operator=0
index_path="/home/zzlin/dataset/${dataset}/train_${size}/hnsw_${dataset}_M${M}_efConstruction${efConstruction}_operator${operator}.index"
train_index_path="/home/zzlin/dataset/${dataset}/train_${size}/train_index.index"
type="hnsw"
method="pca"
K=100
recall=0.995
output_path="/home/zzlin/dataset/${dataset}/train_${size}/ddc/${dataset}_hnsw_pca_${K}_training_set.fvecs"
linear_path="/home/zzlin/dataset/${dataset}/train_${size}/ddc/linear_${type}${K}_${method}.txt"
#######################################
cd ../../../build/src || exit

./hnsw_sample -n ${dataset_path} -l ${learning_path} -s ${sampling} -z ${size}

if [ $? -ne 0 ]; then
    echo "${dataset} Sample FAIL"
    exit 1
fi

echo "${dataset} Sample SUCCESS"

cd "$ORIGINAL_DIR" || exit
#######################################
cd ../../../build/src || exit
./hnsw_bruteforce -k ${K} -n ${dataset_path} -l ${learning_path} -g ${ground_truth_path} -z ${size}

if [ $? -ne 0 ]; then
    echo "${dataset} Tarin Set GroundTruth FAIL"
    exit 1
fi

echo "${dataset} Tarin Set GroundTruth SUCCESS"

cd "$ORIGINAL_DIR" || exit
#######################################
cd ../../../build/Construction || exit
./hnsw_index -m ${M} -e ${efConstruction} -o ${operator} -n ${dataset_path} -i ${index_path} -z ${size}
if [ $? -ne 0 ]; then
    echo "${dataset} Build HNSW Index FAIL"
    exit 1
fi

echo "${dataset} Build HNSW Index SUCCESS"

cd "$ORIGINAL_DIR" || exit
######################################
cd ..
python pca.py -d ${dataset} -n ${name} -l ${ground_truth_path} -s ${size}

if [ $? -ne 0 ]; then
    echo "${dataset} PCA FAIL"
    exit 1
fi

echo "${dataset} PCA SUCCESS"

cd "$ORIGINAL_DIR" || exit
######################################
cd ../../../build/src || exit
./hnsw_rotation -n ${learning_path} -q ${learning_path} -i ${index_path} -e ${train_index_path} -o 6
if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Rota FAIL"
    exit 1
fi
echo "${dataset} HNSW Rota SUCCESS"

cd "$ORIGINAL_DIR" || exit
#######################################
cd ../../../build/data/ddc || exit
./logger_hnsw_pca -k ${K} -n ${learning_path} -q ${learning_path} -g ${ground_truth_path} -i ${train_index_path} -o ${output_path} -r ${recall}  -l ${linear_path}
if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Training Set Generate FAIL"
    exit 1
fi

echo "${dataset} HNSW Training Set Generate SUCCESS"

cd "$ORIGINAL_DIR" || exit
##########################
cd ..

python linear.py -d ${dataset} -k ${K} -m ${method} -i ${type} -t ${output_path} -s ${linear_path}

if [ $? -ne 0 ]; then
    echo "${dataset} Linear FAIL"
    exit 1
fi

echo "${dataset} Linear SUCCESS"

cd "$ORIGINAL_DIR" || exit
##########################
cd ../../../build/data/ddc ||  exit


./logger_hnsw_pca -k ${K} -n ${learning_path} -q ${learning_path} -g ${ground_truth_path} -i ${train_index_path} -o ${output_path} -r ${recall}  -l ${linear_path}

if [ $? -ne 0 ]; then
    echo "${dataset} HNSW Linear Model Correct FAIL"
    exit 1
fi

echo "${dataset} HNSW Linear Model Correct SUCCESS"

cd "$ORIGINAL_DIR" || exit
