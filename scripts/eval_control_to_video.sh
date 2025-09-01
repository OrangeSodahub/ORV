# home path
HOME="~"
echo -e "HOME DIR: \e[31m$HOME\e[0m"

# network
# bash $HOME/clash-for-linux-backup/start.sh
# source $HOME/clash.sh
# proxy_on

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


BASE_CONFIG_PATH="config/base_eval.yaml"
CONFIG_PATH="config/eval_traj_image_2b_finetune.yaml"  # ['eval_traj_image_2b_finetune', 'eval_traj_image_cond_2b_finetune', 'eval_traj_image_condfull_2b_multiview']


python orv/evaluation_control_to_video.py \
            --base_config $BASE_CONFIG_PATH \
            --config $CONFIG_PATH ${@:1}
