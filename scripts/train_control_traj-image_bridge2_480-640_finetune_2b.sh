# home path
echo -e "HOME DIR: \e[31m$HOME\e[0m"

# network
bash $HOME/clash-for-linux-backup/start.sh
source $HOME/clash.sh
proxy_on

# env
# source $HOME/anaconda3/etc/profile.d/conda.sh
# conda config --append envs_dirs $HOME/.conda/envs
conda activate orv
echo -e "Current ENV: \e[31m$CONDA_DEFAULT_ENV\e[0m"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/" && pwd)"
echo -e "Root DIR: \e[31m$ROOT\e[0m"

cd $ROOT

export TORCHDYNAMO_VERBOSE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30
# export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="~/.cache/torch"
export PYTHONPATH='.'

#--------------------------------------------------------------------------------------------------
#                               Multi-GPU Training
#--------------------------------------------------------------------------------------------------

GPU_IDS="all"
NUM_PROCESSES=4  # 4 or 8
PORT=29500

ACCELERATE_CONFIG_FILE="config/accelerate/gpu4.yaml"  # ['gpu4', 'gpu8']

# Experiment configurations
BASE_CONFIG_PATH="config/base_train.yaml"
EXP_CONFIG_PATH="config/traj_image_bridge2_480-640_2b_finetune.yaml"


accelerate launch \
          --config_file $ACCELERATE_CONFIG_FILE \
          --gpu_ids $GPU_IDS \
          --num_processes $NUM_PROCESSES \
          --main_process_port $PORT \
          orv/pipeline/train_cogvideox_control_to_video_sft.py \
          --dataset_type bridgev2 \
          --base_config $BASE_CONFIG_PATH \
          --config $EXP_CONFIG_PATH ${@:1}


#--------------------------------------------------------------------------------------------------
#                               Single-GPU Training (debugging)
#--------------------------------------------------------------------------------------------------

# GPU_IDS="0"
# NUM_PROCESSES=1
# PORT=29500

# export DEBUG=1

# ACCELERATE_CONFIG_FILE="config/accelerate/gpu2.yaml"

# # Experiment configurations
# BASE_CONFIG_PATH="config/base_train.yaml"
# EXP_CONFIG_PATH="config/traj_image_bridge2_480-640_2b_finetune.yaml"


# accelerate launch \
#           --config_file $ACCELERATE_CONFIG_FILE \
#           --num_processes $NUM_PROCESSES \
#           --main_process_port $PORT \
#           orv/pipeline/train_cogvideox_control_to_video_sft.py \
#           --dataset_type bridgev2 \
#           --base_config $BASE_CONFIG_PATH \
#           --config $EXP_CONFIG_PATH \
#           --debug ${@:1}
