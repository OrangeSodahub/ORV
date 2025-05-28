#!/bin/bash

# network
# source /share/project/cwm/xiuyu.yang/clash.sh
# bash /share/project/cwm/xiuyu.yang/clash-for-linux-backup/start.sh
# proxy_on

# env
# source /share/project/cwm/xiuyu.yang/anaconda3/etc/profile.d/conda.sh
# conda config --append envs_dirs /share/project/cwm/xiuyu.yang/.conda/envs
# conda activate deepspeed
echo -e "\e[31m$CONDA_DEFAULT_ENV\e[0m"

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


BASE_CONFIG_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/config/base_eval.yaml
# CONFIG_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/config/eval_traj_image_2b_480_320_finetune.yaml  # Eval base model
# CONFIG_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/config/eval_traj_image_cond_2b_480_320_finetune.yaml  # Eval visual controlled model
CONFIG_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/config/eval_traj_image_2b_480_320_multiview.yaml  # Eval multiview model
# CONFIG_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/config/eval_traj_image_condfull_2b_480_320_multiview.yaml  # Eval visual controlled multiview model


# ----------------- Single GPU ---------------------
# python testing/evaluation_control.py \
#             --base_config $BASE_CONFIG_PATH \
#             --config $CONFIG_PATH ${@:1}


# ----------------- Multiple GPUs ---------------------
GPUS=$1  # how many parallel processes?

torchrun --nnodes=1 --nproc_per_node=$GPUS \
            --standalone \
            testing/evaluation_control.py \
            --base_config  $BASE_CONFIG_PATH \
            --config  $CONFIG_PATH ${@:2}
