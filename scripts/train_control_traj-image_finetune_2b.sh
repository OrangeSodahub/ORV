#!/bin/bash

# network

# env
echo -e "\e[31m$CONDA_DEFAULT_ENV\e[0m"

HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
echo "ORV Root DIR: $HOME"

cd $HOME
export WANDB_API_KEY=""
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="~/.cache/huggingface"
export PYTHONPATH='.'

#--------------------------------------------------------------------------------------------------
#                               Multi-GPU Training
#--------------------------------------------------------------------------------------------------

GPU_IDS="all"
NUM_PROCESSES=8
PORT=29500

# ACCELERATE_CONFIG_FILE="accelerate_configs/gpu4.yaml"
ACCELERATE_CONFIG_FILE="accelerate_configs/gpu8.yaml"

# Experiment configurations
BASE_CONFIG_PATH="config/base_train.yaml"
EXP_CONFIG_PATH="config/traj_image_2b_480_320_finetune.yaml"


accelerate launch \
          --config_file $ACCELERATE_CONFIG_FILE \
          --gpu_ids $GPU_IDS \
          --num_processes $NUM_PROCESSES \
          --main_process_port $PORT \
          pipelines/cogvideox_control_to_video_sft.py \
          --base_config $BASE_CONFIG_PATH \
          --config $EXP_CONFIG_PATH ${@:1}


#--------------------------------------------------------------------------------------------------
#                               Single-GPU Training (debugging)
#--------------------------------------------------------------------------------------------------

# GPU_IDS="0"
# NUM_PROCESSES=1
# PORT=29500

# export DEBUG=1

# ACCELERATE_CONFIG_FILE="accelerate_configs/gpu2.yaml"

# # Experiment configurations
# BASE_CONFIG_PATH='config/base_train.yaml'
# EXP_CONFIG_PATH='config/traj_image_2b_480_320_finetune.yaml'


# accelerate launch \
#           --config_file $ACCELERATE_CONFIG_FILE \
#           --num_processes $NUM_PROCESSES \
#           --main_process_port $PORT \
#           pipelines/cogvideox_control_to_video_sft.py \
#           --base_config $BASE_CONFIG_PATH \
#           --config $EXP_CONFIG_PATH \
#           --debug ${@:1}