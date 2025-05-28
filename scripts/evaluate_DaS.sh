#!/bin/bash

# YOU MUST SET THE CUDA_HOME AND PATH AND LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate das

echo "start time: $(date)"

python testing/evaluation.py \
    --data_root <data_root> \
    --model_path <model_path> \
    --evaluation_dir <evaluation_dir> \
    --fps 8 \
    --generate_type i2v \
    --tracking_column trackings.txt \
    --video_column videos.txt \
    --caption_column prompt.txt \
    --image_paths repaint.txt \