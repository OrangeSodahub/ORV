#!/bin/bash

# CUDA 11.8 Environment Setup
# YOU MUST SET THE CUDA_HOME AND PATH AND LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate das

cd /your_path

export DATASET_PATH=../your_dataset_path

mkdir $DATASET_PATH/generated

export CUDA_VISIBLE_DEVICES=6,7
PORT=29501
accelerate launch --main_process_port $PORT accelerate_tracking.py --root $DATASET_PATH/videos --outdir $DATASET_PATH/generated --grid_size 70 
# python batch_tracking.py --root $DATASET_PATH/videos --outdir $DATASET_PATH/generated --grid_size 70 