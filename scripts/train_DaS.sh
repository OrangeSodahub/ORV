#!/bin/bash
set -x

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30

GPU_IDS="0,1,2,3,4,5,6,7"
NUM_PROCESSES=8
PORT=29500
# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-4")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("2000")
WARMUP_STEPS=100
CHECKPOINT_STEPS=500
TRAIN_BATCH_SIZE=2

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_2.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.

# training dataset parameters
DATA_ROOT="../datasets/cogshader"
MODEL_PATH="../ckpts/CogVideoX-5b-I2V"
OUTPUT_PATH="../ckpts/your_ckpt_path"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
TRACKING_COLUMN="trackings.txt"

# validation parameters
TRACKING_MAP_PATH="../videos/tracking.mp4"
VALIDATION_PROMPT="text"
VALIDATION_IMAGES="../videos/first_frame.png"

# Launch experiments with different hyperparameters
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="${OUTPUT_PATH}/cogshader_inv-avatar-physics_steps_${steps}__optimizer_${optimizer}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS --num_processes $NUM_PROCESSES --main_process_port $PORT training/cogvideox_image_to_video_sft.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --tracking_column $TRACKING_COLUMN \
          --tracking_map_path $TRACKING_MAP_PATH \
          --num_tracking_blocks 18 \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 8 \
          --pin_memory \
          --validation_prompt \"$VALIDATION_PROMPT\" \
          --validation_images \"$VALIDATION_IMAGES\" \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size $TRAIN_BATCH_SIZE \
          --max_train_steps $steps \
          --checkpointing_steps $CHECKPOINT_STEPS \
          --gradient_accumulation_steps 4 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps $WARMUP_STEPS \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --noised_image_dropout 0.05 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --resume_from_checkpoint \"latest\" \
          --nccl_timeout 1800"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
