#!/bin/bash

# network

# env
echo -e "Current ENV: \e[31m$CONDA_DEFAULT_ENV\e[0m"

HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
echo -e "ORV Root DIR: \e[31m$HOME\e[0m"

cd $HOME
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="~/.cache/huggingface"
export PYTHONPATH='.'


BASE_CONFIG_PATH=config/base_eval.yaml
CONFIG_PATH=config/eval_traj_image_2b_finetune_cascaded.yaml  # Eval base model


#------------------------------------------------------------
#                     Single-GPU Running
#------------------------------------------------------------

python pipelines/evaluation_control_to_video.py \
            --base_config $BASE_CONFIG_PATH \
            --config $CONFIG_PATH ${@:1}


#------------------------------------------------------------
#                     Multi-GPU Running
#------------------------------------------------------------
# GPUS=$1  # how many parallel processes?

# torchrun --nnodes=1 --nproc_per_node=$GPUS \
#             --standalone \
#             pipelines/evaluation_control_to_video.py \
#             --base_config  $BASE_CONFIG_PATH \
#             --config  $CONFIG_PATH ${@:2}
