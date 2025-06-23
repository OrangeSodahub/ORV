#!/bin/bash
set -x

# network
source ~/clash.sh
bash ~/clash-for-linux-backup/start.sh
proxy_on

# env
source /share/project/cwm/xiuyu.yang/anaconda3/etc/profile.d/conda.sh
conda config --append envs_dirs /share/project/cwm/xiuyu.yang/.conda/envs
conda activate data
echo -e "\e[31m$CONDA_DEFAULT_ENV\e[0m"

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH='.'

cd /share/project/cwm/xiuyu.yang/work/dev6/

SPLIT=$1

python ivideogpt/dataset/prepare_dataset.py --split $SPLIT --action reconstruction
