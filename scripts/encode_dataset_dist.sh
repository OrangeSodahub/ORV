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

GPUS=$1  # how many parallel processes?
SPLIT=$2 # choose from [train, val]
BATCH=$3 # batch size per gpu


# ===> singleview dataset

torchrun --nnodes=1 --nproc_per_node=$GPUS \
            --standalone \
            orv/dataset/encode_dataset.py \
            --dataset bridgev2 \
            --split $SPLIT \
            --batch_size $BATCH \
            --data_root 'data/bridge' \
            --slice \
            --output_dir 'data/bridge/embeddings_480_640_sliced_full' ${@:4}


# torchrun --nnodes=1 --nproc_per_node=$GPUS \
#             --standalone \
#             training/prepare_dataset.py \
#             --dataset rt1 \
#             --split $SPLIT \
#             --batch_size $BATCH \
#             --data_root 'data/rt1' \
#             --slice \
#             --use_cond \
#             --output_dir 'data/rt1/embeddings_320_480_sliced_full' ${@:4}


# multiview dataset

# torchrun --nnodes=1 --nproc_per_node=$GPUS \
#             --standalone \
#             training/prepare_dataset.py \
#             --dataset droid \
#             --split $SPLIT \
#             --batch_size $BATCH \
#             --data_root 'data/droid' \
#             --slice \
#             --output_dir 'data/droid/embeddings_320_480_sliced_full' ${@:4}


# torchrun --nnodes=1 --nproc_per_node=$GPUS \
#             --standalone \
#             training/prepare_dataset.py \
#             --dataset bridgev2 \
#             --split $SPLIT \
#             --batch_size $BATCH \
#             --data_root 'data/bridgev2' \
#             --slice \
#             --use_cond \
#             --output_dir 'data/bridgev2/embeddings_320_480_sliced_full' ${@:4}


# encode GT depths/labels for single view bridge data
# torchrun --nnodes=1 --nproc_per_node=$GPUS \
#             --standalone \
#             training/prepare_dataset.py \
#             --dataset bridgev2 \
#             --split $SPLIT \
#             --batch_size $BATCH \
#             --data_root 'data/bridge' \
#             --slice \
#             --use_cond \
#             --load_condGT \
#             --output_dir 'data/bridge/embeddings_320_480_sliced_full' ${@:4}


# encode render depths/labels for single view bridge data
# torchrun --nnodes=1 --nproc_per_node=$GPUS \
#             --standalone \
#             training/prepare_dataset.py \
#             --dataset bridgev2 \
#             --split $SPLIT \
#             --batch_size $BATCH \
#             --data_root 'data/bridge' \
#             --slice \
#             --use_cond \
#             --output_dir 'data/bridge/embeddings_320_480_sliced_full' ${@:4}