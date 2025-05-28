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
export TORCH_HOME="/share/project/cwm/xiuyu.yang/.cache/torch"
export PYTHONPATH='.'

cd /share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader


# GT_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data_old/bridge/embeddings_320_480_sliced_full/val/videos/
# GT_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/droid/embeddings_320_480_sliced_full/val/videos/
GT_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/bridgev2/embeddings_320_480_sliced_full/val/videos/
# GT_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/rt1/embeddings_320_480_sliced_full/val/videos/
# GT_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/rt1/embeddings_320_480_sliced_full/val/videos/
# GT_PATH=/share/project/cwm/shaocong.xu/exp/HMA/logs/folder_gt/
# GT_PATH=/share/project/cwm/shaocong.xu/exp/HMA/logs/bridge_gt/
# GT_PATH=/share/project/cwm/shaocong.xu/exp/HMA/logs/fractal20220817_gt/
# GT_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/bridgev2/embeddings_320_480_sliced_full/val/videos/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_bridge_traj-image_480-320_finetune_2b_30k/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_bridge_traj-image-cond_480-320_finetune_2b_20k/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_bridge_traj-image_480-320_scratch_notext_full_30k/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge_traj-image-label_480-320_finetune_2b_20k/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge_traj-image-labelGT_480-320_finetune_2b_20k/
# PRED_PATH=//share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfullGT_480-320_finetune_2b_20k/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image_320-480_finetune_2b_30k/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge2_traj-image_320-480_finetune_2b_30k/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/
PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_bridge2_traj-image_480-320_multiview_20k/
# PRED_PATH=/share/project/cwm/shaocong.xu/exp/HMA/logs/folder_pred/
# PRED_PATH=/share/project/cwm/shaocong.xu/exp/HMA/logs/bridge_pred/
# PRED_PATH=/share/project/cwm/shaocong.xu/exp/HMA/logs/fractal20220817_pred/
# PRED_PATH=/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_bridge2_traj-image_480-320_multiview_condfull_20k/


python testing/compute_metrics.py \
            --gt_dir $GT_PATH \
            --pred_dir $PRED_PATH ${@:1}