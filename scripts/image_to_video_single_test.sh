#!/bin/bash
set -x

# network
source ~/clash.sh
bash ~/clash-for-linux-backup/start.sh
proxy_on

# env
source /share/project/cwm/xiuyu.yang/anaconda3/etc/profile.d/conda.sh
conda config --append envs_dirs /share/project/cwm/xiuyu.yang/.conda/envs
conda activate deepspeed
echo -e "\e[31m$CONDA_DEFAULT_ENV\e[0m"

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH='.'

cd /share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader

GPU_IDS="0"
NUM_PROCESSES=1
PORT=29501
# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=1e-4
LR_SCHEDULES=cosine_with_restarts
OPTIMIZERS=adamw
MAX_TRAIN_STEPS=2000
WARMUP_STEPS=100
CHECKPOINT_STEPS=500
TRAIN_BATCH_SIZE=1
GRAD_ACC=4

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/L20_2.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.

# training dataset parameters
DATA_ROOT="data/bridge"
COND_DATA_ROOT="data/occ_bridge"
# MODEL_PATH="THUDM/CogVideoX-5b-I2V"
# MODEL_PATH="THUDM/CogVideoX1.5-5b-I2V"
MODEL_PATH="/share/project/cwm/xiuyu.yang/.cache/huggingface/hub/models--THUDM--CogVideoX1.5-5b-I2V/snapshots/9f310b78e4ed32a15fec50712149d4785d3d0fc4"
OUTPUT_PATH="outputs"
# CAPTION_COLUMN="prompt.txt"
# VIDEO_COLUMN="videos.txt"
# TRACKING_COLUMN="trackings.txt"
CAPTION_COLUMN="prompt"
VIDEO_COLUMN="video"
LATENT_COLUMN="latent"
DEPTH_COLUMN="depth"
SEMANTIC_COLUMN="semantic"

output_dir="${OUTPUT_PATH}/cirasm_bridge_image-test_${MAX_TRAIN_STEPS}_lr_${LEARNING_RATES}/"
accelerate launch --config_file $ACCELERATE_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $PORT training/cogvideox_image_to_video_test.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --data_root $DATA_ROOT \
          --cond_data_root $COND_DATA_ROOT \
          --video_column $VIDEO_COLUMN \
          --latent_column $LATENT_COLUMN \
          --use_cond \
          --caption_column $CAPTION_COLUMN \
          --depth_column $DEPTH_COLUMN \
          --semantic_column $SEMANTIC_COLUMN \
          --no_traj \
          --no_cond \
          --load_tensors \
          --num_tracking_blocks 18 \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 8 \
          --pin_memory \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size $TRAIN_BATCH_SIZE \
          --max_train_steps $MAX_TRAIN_STEPS \
          --checkpointing_steps $CHECKPOINT_STEPS \
          --gradient_accumulation_steps $GRAD_ACC \
          --gradient_checkpointing \
          --learning_rate $LEARNING_RATES \
          --lr_scheduler $LR_SCHEDULES \
          --lr_warmup_steps $WARMUP_STEPS \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --optimizer $OPTIMIZERS \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --noised_image_dropout 0.05 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --nccl_timeout 18000 ${@:1}
        #   --resume_from_checkpoint \"latest\" \