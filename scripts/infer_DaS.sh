#!/bin/bash

# CUDA 12.1 Environment Setup

# YOU MUST SET THE CUDA_HOME AND PATH AND LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogvideo

python testing/inference.py \
    --prompt <"prompt text"> \
    --model_path <model_path> \
    --tracking_path <tracking_path> \
    --image_or_video_path <image_or_video_path> \
    --generate_type i2v