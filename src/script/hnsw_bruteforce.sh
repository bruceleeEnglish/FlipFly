#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd ../../build/src || exit
K=100
dataset="wiki"
dataset_path="/home/dataset/${dataset}/wikipedia_base.bin.crop_nb_1000000"
size=-1
metric=1
learning_path="/home/dataset/${dataset}/${dataset}_train.fvecs"
ground_truth_path="/home/dataset/${dataset}/${dataset}_train_ground_truth.ivecs"
if [ $metric == 1 ]; then
  ground_truth_path="/home/dataset/${dataset}/${dataset}_train_ground_truth_ip.ivecs"
fi
#learning_path="/home/dataset/${dataset}/query.fvecs"
#ground_truth_path="/home/dataset/${dataset}/${dataset}_size${size}_gt.ivecs"

./hnsw_bruteforce -k ${K} -n ${dataset_path} -l ${learning_path} -g ${ground_truth_path} -z ${size} -m ${metric}

if [ $? -ne 0 ]; then
    echo "${dataset} Tarin Set GroundTruth FAIL"
    exit 1
fi

cd "$ORIGINAL_DIR" || exit