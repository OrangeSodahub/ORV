#!/bin/bash

# network
source /share/project/cwm/xiuyu.yang/clash.sh
bash /share/project/cwm/xiuyu.yang/clash-for-linux-backup/start.sh
proxy_on

# env
source /share/project/cwm/xiuyu.yang/anaconda3/etc/profile.d/conda.sh
conda config --append envs_dirs /share/project/cwm/xiuyu.yang/.conda/envs
conda activate deepspeed
echo -e "\e[31m$CONDA_DEFAULT_ENV\e[0m"

export WANDB_API_KEY="e006d03e8d44f42a8f72f872b09ac373022aed96"

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/share/project/cwm/xiuyu.yang/.cache/huggingface"
export PYTHONPATH='.'

cd /share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader

GPU_IDS="all"
NUM_PROCESSES=4
PORT=29500

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/L20_4.yaml"

# Experiment configurations
BASE_CONFIG_PATH='/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/config/base_train.yaml'
EXP_CONFIG_PATH='/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/config/traj_image_1.4b_480_320_ref5_finetune_ref1.yaml'


accelerate launch \
          --config_file $ACCELERATE_CONFIG_FILE \
          --gpu_ids $GPU_IDS \
          --num_processes $NUM_PROCESSES \
          --main_process_port $PORT \
          training/cogvideox_control_to_video.py \
          --base_config $BASE_CONFIG_PATH \
          --config $EXP_CONFIG_PATH ${@:1}

# GPU_IDS="0"
# NUM_PROCESSES=1
# PORT=29502

# # Single GPU uncompiled training
# ACCELERATE_CONFIG_FILE="accelerate_configs/L20_2.yaml"

# # Experiment configurations
# BASE_CONFIG_PATH='/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/config/base_train.yaml'
# EXP_CONFIG_PATH='/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/config/traj_image_1.4b_480_320_ref5_finetune_ref1.yaml'


# accelerate launch \
#           --config_file $ACCELERATE_CONFIG_FILE \
#           --num_processes $NUM_PROCESSES \
#           --main_process_port $PORT \
#           training/cogvideox_control_to_video.py \
#           --base_config $BASE_CONFIG_PATH \
#           --config $EXP_CONFIG_PATH \
#           --debug ${@:1}