# home path

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


GT_PATH="data/bridgev2/embeddings_320_480_sliced_full/val/videos/"
PRED_PATH="outputs/eval_orv_bridge2_traj-image_480-320_multiview_20k/"


python orv/pipeline/compute_metrics.py \
            --gt_dir $GT_PATH \
            --pred_dir $PRED_PATH ${@:1}