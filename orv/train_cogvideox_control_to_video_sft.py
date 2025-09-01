# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import gc
import logging
import math
import os
import shutil
import random
import fnmatch
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Union
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from PIL import Image

import diffusers
import torch
import transformers
import wandb
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.configuration_utils import FrozenDict
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.t5 import T5EncoderModel

import decord  # isort:skip
decord.bridge.set_bridge('torch')

from orv.models.text_encoder import compute_prompt_embeddings  # isort:skip
from orv.dataset.dataset import BucketSampler, RobotDataset, MultiViewRobotDataset, CollateFunctionControl  # isort:skip
from orv.models.cogvideox_control import CogVideoXTransformer3DModelTraj, CogVideoXImageToVideoPipelineTraj
from orv.utils import get_gradient_norm, get_optimizer, prepare_rotary_positional_embeddings, print_memory, reset_memory, flatten_dict, CONSOLE  # isort:skip


logger = get_logger(__name__)


@torch.no_grad()
def log_validation(
    accelerator: Accelerator,
    pipe: Union[CogVideoXPipeline, CogVideoXImageToVideoPipelineTraj],
    config: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    step,
    epoch,
    index,
):
    logger.info(
        f"Running validation... \n Generating {config.train.num_validation_videos} videos."
    )

    # sliced_frames = pipeline_args.pop('sliced_frames')
    num_views = pipeline_args['num_views']
    num_frames = pipeline_args['num_frames']
    H = pipeline_args['height']
    W = pipeline_args['width']
    sample_name = pipeline_args.pop('sample_name')

    if not config.inference.enable_model_cpu_offload:
        pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed else None

    with torch.no_grad():
        videos = pipe(**pipeline_args, generator=generator, output_type='pil').frames

    # log to local disk
    video_filenames = []
    num_frames = len(videos[0])
    for i in range(0, len(videos), num_views):

        batch_videos = videos[i : i + num_views]
        if num_views > 1:
            video = []
            for j in range(num_frames):
                canvas = Image.new('RGB', (W * num_views, H))
                for k in range(num_views):
                    canvas.paste(batch_videos[k][j], (W * k, 0))
                video.append(canvas)
        else:
            video = batch_videos[0]

        prompt = (
            pipeline_args['prompt'][i][:25]
            .replace(' ', '_')
            .replace("'", '_')
            .replace('"', '_')
            .replace('/', '_')
        )

        # export to video
        video_filename = os.path.join(config.train.output_dir, f'validation_{index}_{sample_name}_s{step}_ep{epoch}_{i}th_{prompt}.mp4')
        os.makedirs(os.path.dirname(video_filename), exist_ok=True)
        export_to_video(video, video_filename, fps=10)
        video_filenames.append(video_filename)

        # export to gif
        merges = video
        # gt_video = sliced_frames
        # assert len(gt_video) == len(video), f'Got mismatched length of video: {len(gt_video)=} v.s. {len(video)=}.'
        # gt_video = [Image.fromarray(frame.astype(np.uint8)) for frame in gt_video]
        # merges = []
        # for i in range(len(gt_video)):
        #     W, H = gt_video[i].size
        #     canvas = Image.new('RGB', (W, H * 2))
        #     canvas.paste(gt_video[i], (0, 0))
        #     canvas.paste(video[i], (0, H))
        #     merges.append(canvas)
        gif_filename = os.path.join(config.train.output_dir, f'validation_{index}_{sample_name}_s{step}_ep{epoch}_{i}th_{prompt}.gif')
        merges[0].save(gif_filename, save_all=True, append_images=merges[1:], duration=100, loop=0)
        CONSOLE.log(f'Exported GIF to {gif_filename}.')

    # log to wandb
    for tracker in accelerator.trackers:
        if tracker.name == 'wandb':
            tracker.log(
                {
                    'validation': [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )
    torch.cuda.empty_cache()
    return videos


def main(config):
    train_config = config.train
    dataset_config = config.dataset
    transformer_config = config.transformer
    pipeline_config = config.inference

    if config.report_to == 'wandb' and config.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and train_config.mixed_precision == 'bf16':
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    os.environ['WANDB_MODE'] = 'offline'
    if config.report_to == 'wandb':
        os.environ['WANDB_MODE'] = 'online'

    logging_dir = Path(train_config.output_dir, train_config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=train_config.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=train_config.find_unused_parameters)
    init_process_group_kwargs = InitProcessGroupKwargs(backend='nccl', timeout=timedelta(seconds=config.nccl_timeout))
    accelerator = Accelerator(
        cpu=False,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=train_config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )
    CONSOLE.log(f'{accelerator.device=}')

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if train_config.output_dir is not None:
            os.makedirs(train_config.output_dir, exist_ok=True)

            # backup configuration
            config_path = os.path.join(train_config.output_dir, 'config.yaml')
            if os.path.exists(config_path) and train_config.resume_from_checkpoint is None and not config.debug:
                raise RuntimeError(f'File already exists: {config_path}')
            OmegaConf.save(config=config, f=config_path)

        # if config.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=config.hub_model_id or Path(train_config.output_dir).name,
        #         exist_ok=True,
        #     ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        transformer_config.pretrained_model_name_or_path,
        subfolder='tokenizer',
        revision=transformer_config.revision,
    )

    text_encoder = T5EncoderModel.from_pretrained(
        transformer_config.pretrained_model_name_or_path,
        subfolder='text_encoder',
        revision=transformer_config.revision,
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if '5b' in transformer_config.pretrained_model_name_or_path.lower() else torch.float16

    transformer: CogVideoXTransformer3DModelTraj
    extra_init_kwargs = dict(
        recon_action=transformer_config.recon_action,
        visual_guidance=transformer_config.visual_guidance,
        num_control_blocks=transformer_config.num_control_blocks,
        multiview=transformer_config.multiview,
        max_n_view=dataset_config.max_n_view,
        num_control_keys=len(dataset_config.control_keys),
        loaded_pretrained_model_name_or_path=None,  # cannot be called as `pretrained_model_name_or_path`
        modulate_encoder_hidden_states=transformer_config.modulate_encoder_hidden_states,
    )
    model_source_key = None
    if train_config.from_scratch:
        # We do not load pretrained CogVideoX model

        CONSOLE.log(f'Loading transformer model from {transformer_config.transformer_model_name_or_path} or {transformer_config.transformer_config_path} ...')
        if not (os.path.exists(str(transformer_config.transformer_model_name_or_path))
                or os.path.exists(str(transformer_config.transformer_config_path))):
            raise FileNotFoundError(f'Not found {transformer_config.transformer_model_name_or_path=}'
                                    f'or {transformer_config.transformer_config_path=}')

        if transformer_config.transformer_model_name_or_path is not None:
            transformer = CogVideoXTransformer3DModelTraj.from_pretrained(
                transformer_config.transformer_model_name_or_path,
                subfolder='transformer',
                torch_dtype=load_dtype,
                revision=transformer_config.revision,
                variant=transformer_config.variant,
                **extra_init_kwargs,  # these kwargs will supass the existing arguments!
            )
            model_source_key = transformer_config.transformer_model_name_or_path
        else:
            transformer = CogVideoXTransformer3DModelTraj.from_config(
                CogVideoXTransformer3DModelTraj.load_config(
                    transformer_config.transformer_config_path),
                **extra_init_kwargs,  # these kwargs will supass the existing arguments!
            )
            model_source_key = transformer_config.transformer_config_path

    elif train_config.from_pretrained:
        # We load pretrained CogVideoX model

        extra_init_kwargs.update(
            sample_height=dataset_config.sample_size[0],
            sample_width=dataset_config.sample_size[1],
            sample_frames=dataset_config.sample_frames,
            loaded_pretrained_model_name_or_path=transformer_config.pretrained_model_name_or_path,
        )

        CONSOLE.log(f'Loading pretrained model from {transformer_config.pretrained_model_name_or_path} ...')
        transformer = CogVideoXTransformer3DModelTraj.from_pretrained(
                transformer_config.pretrained_model_name_or_path,
                subfolder='transformer',
                torch_dtype=load_dtype,
                revision=transformer_config.revision,
                variant=transformer_config.variant,
                **extra_init_kwargs,  # these kwargs will supass the existing arguments!
        )
        model_source_key = transformer_config.pretrained_model_name_or_path

    else:

        raise RuntimeError(f'Need to specify either `from_scratch` or `from_pretrained`!')

    CONSOLE.log(f'Loaded transformer model: [bold yellow]{transformer.__class__.__name__}[/] from [bold yellow]{model_source_key}[/].')

    is_ofs_embed = transformer.config.ofs_embed_dim is not None
    patch_size_t = transformer.config.patch_size_t

    # Compute the model size
    total_params = sum(p.numel() for p in transformer.parameters())
    total_params_in_billion = total_params / 1e9
    CONSOLE.log(f'Transformer model size: {total_params_in_billion:.2f}B')
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    trainable_params_in_billion = trainable_params / 1e9
    CONSOLE.log(f'Train params: {trainable_params_in_billion:.2f}B')

    vae = AutoencoderKLCogVideoX.from_pretrained(
        transformer_config.pretrained_model_name_or_path,
        subfolder='vae',
        revision=transformer_config.revision,
        variant=transformer_config.variant,
    )
    # ! IMPORTANT: only pretrained CogVideoX should set `invert_scale_latents`!
    # TODO: fix this term when using CogVideoX's settings!!!!!
    vae._internal_dict = FrozenDict(**(vae.config | dict(invert_scale_latents=False)))

    scheduler: CogVideoXDPMScheduler = CogVideoXDPMScheduler.from_pretrained(transformer_config.pretrained_model_name_or_path, subfolder='scheduler')
    CONSOLE.log(f'Loaded vae and scheduler.')

    if transformer_config.enable_slicing:
        vae.enable_slicing()
    if transformer_config.enable_tiling:
        vae.enable_tiling()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            'fp16' in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config['fp16']['enabled']
        ):
            weight_dtype = torch.float16
        if (
            'bf16' in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config['bf16']['enabled']
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    CONSOLE.log(f'Moving components to {accelerator.device} ...')
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if train_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model: CogVideoXTransformer3DModelTraj = unwrap_model(model)
                    model.save_pretrained(
                        os.path.join(output_dir, 'transformer'), safe_serialization=True, max_shard_size='5GB'
                    )
                else:
                    raise ValueError(f'Unexpected save model: {model.__class__}')

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None
        init_under_meta = False

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f'Unexpected save model: {unwrap_model(model).__class__}')
        else:
            with init_empty_weights():
                transformer_ = CogVideoXTransformer3DModel.from_config(
                    transformer_config.pretrained_model_name_or_path, subfolder='transformer'
                )
                init_under_meta = True

        load_model = CogVideoXTransformer3DModel.from_pretrained(os.path.join(input_dir, 'transformer'))
        transformer_.register_to_config(**load_model.config)
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if train_config.mixed_precision == 'fp16':
            cast_training_params([transformer_])

    def load_model_hook_trajectory(models, input_dir):
        transformer_ = None
        init_under_meta = False

        # here we load pretrained checkpoint, thus we do not modify any configuration.
        load_model = CogVideoXTransformer3DModelTraj.from_pretrained(os.path.join(input_dir, 'transformer'))
        CONSOLE.log(f'Configurations of checkpoint: {load_model.config}')

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f'Unexpected save model: {unwrap_model(model).__class__}')
        else:
            with init_empty_weights():
                # NOTE we must make sure that new model is constructed from exactly the same configurations as the checkpoint!
                transformer_ = CogVideoXTransformer3DModelTraj.from_config(load_model.config)
                init_under_meta = True

        CONSOLE.log(f'Configurations of current model: {transformer_.config}')
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if train_config.mixed_precision == 'fp16':
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook if train_config.resume_from_checkpoint is None else load_model_hook_trajectory)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if train_config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if train_config.scale_lr:
        train_config.learning_rate = (
            train_config.learning_rate * train_config.gradient_accumulation_steps * train_config.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if train_config.mixed_precision == 'fp16':
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {
        'params': transformer_parameters,
        'lr': train_config.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model['params'])

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and 'optimizer' in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and 'scheduler' in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    CONSOLE.log(f'Setting up optimizer ...')
    optimizer_config = train_config.optimizer
    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=optimizer_config.type,
        learning_rate=train_config.learning_rate,
        beta1=optimizer_config.beta1,
        beta2=optimizer_config.beta2,
        beta3=optimizer_config.beta3,
        epsilon=optimizer_config.epsilon,
        weight_decay=optimizer_config.weight_decay,
        prodigy_decouple=optimizer_config.prodigy_decouple,
        prodigy_use_bias_correction=optimizer_config.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=optimizer_config.prodigy_safeguard_warmup,
        use_8bit=optimizer_config.use_8bit,
        use_4bit=optimizer_config.use_4bit,
        use_torchao=optimizer_config.use_torchao,
        use_deepspeed=use_deepspeed_optimizer,
        use_cpu_offload_optimizer=optimizer_config.use_cpu_offload_optimizer,
        offload_gradients=optimizer_config.offload_gradients,
    )

    # Dataset and DataLoader
    dataset_kwargs = {
        'data_root': dataset_config.data_root,
        'renderings_folder': dataset_config.renderings_folder,
        'embeddings_folder': dataset_config.embeddings_folder,
        'use_cond': dataset_config.use_cond,
        'filter_by_cond': dataset_config.filter_by_cond,
        'num_samples': dataset_config.num_samples,
        'camera_ids': dataset_config.camera_ids,
        'action_dim': dataset_config.action_dim,
        'sequence_interval': dataset_config.sequence_interval,
        'sequence_length': dataset_config.sequence_length,
        'sample_frames': dataset_config.sample_frames,
        'start_frame_interval': dataset_config.start_frame_interval,
        'video_size': dataset_config.video_size,
        'control_keys': dataset_config.control_keys,
        'caption_column': dataset_config.caption_column,
        'video_column': dataset_config.video_column,
        'latent_column': dataset_config.latent_column,
        'depth_column': dataset_config.depth_column,
        'semantic_column': dataset_config.semantic_column,
        'load_actions': dataset_config.load_actions,
        'load_condGT': dataset_config.load_condGT,
        'slice_frame': dataset_config.slice_frame,  # we always train with fixed length
        'empty_prompt': dataset_config.empty_prompt,
        'use_3dvae': dataset_config.use_3dvae,  # some specifications needed when using 3dvae
    }

    if train_config.overfit:
        dataset_kwargs['num_samples'] = 5e2

    # be careful with `n_view` parameter!
    n_view = len(dataset_config.camera_ids)
    if not train_config.multiview:
        n_view = 1

    if n_view == 1:
        # single view inputs

        train_dataset = RobotDataset(
            **dataset_kwargs,
            split='train' if not train_config.overfit else 'val',
            load_tensor=dataset_config.load_tensors,
            ref_num=dataset_config.num_observation,
        )

        val_dataset = RobotDataset(
            **dataset_kwargs,
            split='val',
            load_tensor=dataset_config.load_tensors,
            ref_num=1,
            test_mode=True,  # TODO: here will also laod tensors!!!
        )

    else:
        # multivew inputs

        dataset_kwargs['n_view'] = n_view

        train_dataset = MultiViewRobotDataset(
            **dataset_kwargs,
            split='train' if not train_config.overfit else 'val',
            load_tensor=dataset_config.load_tensors,
            ref_num=dataset_config.num_observation,
        )

        val_dataset = MultiViewRobotDataset(
            **dataset_kwargs,
            split='val',
            load_tensor=dataset_config.load_tensors,
            ref_num=1,
            test_mode=True,
        )

    collate_fn_control = CollateFunctionControl(weight_dtype, dataset_config.load_tensors)
    use_bucket_sampler = train_dataset.num_refs > 1 or n_view > 1

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        sampler=(BucketSampler(train_dataset,
                               batch_size=train_config.train_batch_size,
                               shuffle=True,
                               drop_last=True)
                if use_bucket_sampler else None),
        collate_fn=collate_fn_control,
        drop_last=True,
        shuffle=True if not use_bucket_sampler else False,
        num_workers=train_config.dataloader_num_workers,
        pin_memory=train_config.pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=collate_fn_control,
        num_workers=train_config.dataloader_num_workers,
        pin_memory=train_config.pin_memory,
        shuffle=False,
    )

    def run_validation(step, epoch):

        accelerator.print('===== Memory before validation =====')
        print_memory(accelerator.device)
        torch.cuda.synchronize(accelerator.device)
        transformer.eval()

        pipe: CogVideoXImageToVideoPipelineTraj = CogVideoXImageToVideoPipelineTraj.from_pretrained(
            transformer_config.pretrained_model_name_or_path,
            transformer=unwrap_model(transformer),
            scheduler=scheduler,
            vae=vae,  # ! IMPORTANT
            revision=transformer_config.revision,
            variant=transformer_config.variant,
            torch_dtype=weight_dtype,
        )
        CONSOLE.log(f'Build pipeline {pipe.__class__.__name__}')

        if transformer_config.enable_slicing:
            pipe.vae.enable_slicing()
        if transformer_config.enable_tiling:
            pipe.vae.enable_tiling()
        if pipeline_config.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        for i, batch in enumerate(val_dataloader):

            if i > train_config.num_validation_batch: break

            images = batch['images'] if val_dataset.config.load_tensor else batch['pil_images']
            # prompt_embeds = batch['prompt_embeds'].to(accelerator.device) if val_dataset.config.load_tensor else None
            pipeline_args = {
                'image': images,  # n_batch * n_view * ref_num
                'prompt': batch['prompts'],
                'prompt_embeds': batch['prompt_embeds'].to(accelerator.device),
                'negative_prompt': 'The video is not of a high quality, it has a low resolution. Strange body and strange trajectory. Distortion.',
                'controls_or_guidances': {
                    'actions': batch['controls']['actions'].to(accelerator.device) if not config.no_traj else None,
                    'depths': (
                        batch['controls']['latents_depth'].to(accelerator.device)
                        if config.use_cond and 'depth' in dataset_config.control_keys else None
                    ),
                    'labels': (
                        batch['controls']['latents_label'].to(accelerator.device)
                        if config.use_cond and 'label' in dataset_config.control_keys else None
                    ),
                },
                'num_frames': batch['num_frames'],
                'num_views': batch['num_views'],
                'guidance_scale': pipeline_config.guidance_scale,
                'use_dynamic_cfg': pipeline_config.use_dynamic_cfg,
                'height': batch['image_height'],
                'width': batch['image_width'],
                # 'sliced_frames': batch['sliced_frames'][0],  # [f, h, w, c]
                'max_sequence_length': model_config.max_text_seq_length,
                'sample_name': batch['metainfos'][0]['sample_name'],
            }
            CONSOLE.log(f"Validating episode: {pipeline_args['sample_name']} with (H, W)=({pipeline_args['height']}, {pipeline_args['width']}) num_frames={pipeline_args['num_frames']} num_views={pipeline_args['num_views']}")

            log_validation(
                accelerator=accelerator,
                pipe=pipe,
                config=config,
                pipeline_args=pipeline_args,
                step=step,
                epoch=epoch,
                index=i,
            )

        CONSOLE.log('test done!')

        transformer.train()
        accelerator.print('===== Memory after validation =====')
        print_memory(accelerator.device)
        reset_memory(accelerator.device)

        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_config.gradient_accumulation_steps)
    if train_config.max_train_steps is None:
        train_config.max_train_steps = train_config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if optimizer_config.use_cpu_offload_optimizer:
        lr_scheduler = None
        accelerator.print(
            "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
            "you are training with those settings, they will be ignored."
        )
    else:
        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=train_config.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=train_config.max_train_steps * accelerator.num_processes,
                num_warmup_steps=train_config.lr_warmup_steps * accelerator.num_processes,
            )
        else:
            lr_scheduler = get_scheduler(
                train_config.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=train_config.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=train_config.max_train_steps * accelerator.num_processes,
                num_cycles=train_config.lr_num_cycles,
                power=train_config.lr_power,
            )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        train_config.max_train_steps = train_config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    train_config.num_train_epochs = math.ceil(train_config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = config.tracker_name or 'cogvideox-control'
        accelerator.init_trackers(tracker_name, config=flatten_dict(config))

        accelerator.print('===== Memory before training =====')
        CONSOLE.log(f'{accelerator.device=}')
        reset_memory(accelerator.device)
        print_memory(accelerator.device)

    # Train!
    total_batch_size = train_config.train_batch_size * accelerator.num_processes * train_config.gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {train_config.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {train_config.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {train_config.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {train_config.max_train_steps}")
    accelerator.print(f"  Validation steps = {train_config.validation_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not train_config.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if train_config.resume_from_checkpoint != 'latest':
            path = os.path.basename(train_config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(train_config.output_dir)
            dirs = [d for d in dirs if d.startswith('checkpoint-')]  # exculde the 'checkpoint'!!!
            dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None or not os.path.exists(os.path.join(train_config.output_dir, path)):
            CONSOLE.log(f"[bold red]Checkpoint '{train_config.resume_from_checkpoint}' does not exist with {path=}. Starting a new training run.")
            train_config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f'Resuming from checkpoint {path}')
            accelerator.load_state(os.path.join(train_config.output_dir, path))
            global_step = int(path.split('-')[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            CONSOLE.log(f"[bold blue]Successfully resumed from {os.path.join(train_config.output_dir, path)}!")

    progress_bar = tqdm(
        range(0, train_config.max_train_steps),
        initial=initial_global_step,
        desc='Steps',
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, 'module') else transformer.config

    # Add initial validation before training starts
    if accelerator.is_main_process and not int(os.getenv('NO_INIT_VAL', 0)):

        run_validation(step=global_step, epoch=first_epoch)

        # Hacky code!
        if int(os.getenv('ONLY_INIT_VAL', 0)):
            exit(1)

    if dataset_config.load_tensors:
        del text_encoder
        text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)


    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)

    for epoch in range(first_epoch, train_config.num_train_epochs):
        transformer.train()

        if accelerator.is_main_process:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            CONSOLE.log(f'Allocated {(torch.cuda.memory_allocated() / 1024 ** 3):.2f}GB')
            CONSOLE.log(f'Reserved {(torch.cuda.memory_reserved() / 1024 ** 3):.2f}GB')

        for _, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            gradient_norm_before_clip = None
            gradient_norm_after_clip = None

            batch_n_view = batch['num_views']
            batch_n_frame = batch['num_frames']
            if batch_n_view > n_view:
                raise RuntimeError(f'Invalid batch data with {batch_n_view=} and {n_view=}!')

            with accelerator.accumulate(models_to_accumulate):
                if dataset_config.load_tensors:
                    videos = batch['latents'].to(accelerator.device, non_blocking=True)
                    images = batch['images'].to(accelerator.device, non_blocking=True)
                    prompts = batch['prompt_embeds'].to(accelerator.device, non_blocking=True)
                else:
                    videos = batch['videos'].to(accelerator.device, non_blocking=True)
                    images = batch['images'].to(accelerator.device, non_blocking=True)
                    prompts = batch['prompts']

                # if config.tracking_column is not None:
                #     tracking_maps = batch['tracking_maps'].to(accelerator.device, non_blocking=True)
                #     tracking_image = tracking_maps[:, :1].clone()
                controls = batch['controls']
                actions = depths = labels = None
                if not config.no_traj:
                    actions = controls['actions'].to(accelerator.device, non_blocking=True)
                if config.use_cond:
                    if dataset_config.load_tensors:
                        if 'latents_depth' in controls:
                            depths = controls['latents_depth'].to(accelerator.device, non_blocking=True)
                        if 'latents_label' in controls:
                            labels = controls['latents_label'].to(accelerator.device, non_blocking=True)
                    else:
                        # TODO: encode depths by vae
                        depths = controls['depths'].to(accelerator.device, non_blocking=True)

                latent_dist = DiagonalGaussianDistribution(videos)
                video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                video_latents = video_latents.permute(0, 2, 1, 3, 4)  # -> [B, F, C, H, W]
                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                image_latent_dist = DiagonalGaussianDistribution(images)
                image_latents = image_latent_dist.sample() * VAE_SCALING_FACTOR
                image_latents = image_latents.permute(0, 2, 1, 3, 4)  # -> [B, F, C, H, W]
                image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                # sanity check
                # from einops import rearrange
                # from diffusers.video_processor import VideoProcessor
                # from diffusers.image_processor import VaeImageProcessor
                # save_folder = 'debug_mv_train_input'
                # os.makedirs(save_folder, exist_ok=True)
                # # check videl latents -------------------------------------------------------------------------- #
                # video_processor = VideoProcessor(
                #     vae_latent_channels=vae.config.latent_channels,
                #     vae_scale_factor=VAE_SCALE_FACTOR_SPATIAL,
                # )
                # decode_video_latents = rearrange(video_latents, 'b (v f) c h w -> (b v) f c h w', v=n_view)
                # decode_video_latents = decode_video_latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
                # decode_video_latents = 1 / VAE_SCALING_FACTOR * decode_video_latents
                # decode_video = vae.decode(decode_video_latents).sample
                # decode_video = video_processor.postprocess_video(video=decode_video, output_type='pil')
                # num_frames = len(decode_video[0])
                # W, H = decode_video[0][0].size
                # for i in range(0, len(decode_video), n_view):
                #     batch_videos = decode_video[i : i + n_view]
                #     video = []
                #     for j in range(num_frames):
                #         canvas = Image.new('RGB', (W * n_view, H))
                #         for k in range(n_view):
                #             canvas.paste(batch_videos[k][j], (W * k, 0))
                #         video.append(canvas)
                #     video[0].save(os.path.join(save_folder, f'{global_step}_{i}.gif'), save_all=True, append_images=video[1:], duration=100, loop=0)
                # # -------------------------------------------------------------------------------------------- #

                # check image latents ------------------------------------------------------------------------ #
                # image_processor = VaeImageProcessor(
                #     vae_latent_channels=vae.config.latent_channels,
                #     vae_scale_factor=VAE_SCALE_FACTOR_SPATIAL,
                # )
                # decode_image_latents = rearrange(image_latents, 'b (v f) c h w -> (b v) f c h w', v=n_view)
                # decode_image_latents = decode_image_latents.permute(0, 2, 1, 3, 4)  # [b, c, t, h, w]
                # decode_image_latents = 1 / VAE_SCALING_FACTOR * decode_image_latents
                # decode_image = vae.decode(decode_image_latents).sample
                # decode_image = image_processor.postprocess(decode_image[:, :, 0, ...], output_type='pil')
                # for i, image in enumerate(decode_image):
                #     image.save(os.path.join(save_folder, f'{global_step}_{i}.png'))
                # # -------------------------------------------------------------------------------------------- #

                num_frames = video_latents.size(1)
                pad_frames = 0
                if patch_size_t and num_frames % patch_size_t != 0:
                    if batch_n_view > 1:
                        raise RuntimeError
                    pad_frames = patch_size_t - num_frames % patch_size_t
                    # video_latents = video_latents[:, :-res_frame, ...]
                    video_latents = torch.cat([
                        video_latents,
                        torch.zeros((video_latents.size(0), pad_frames, video_latents.size(2), video_latents.size(3), video_latents.size(4)),
                                    dtype=video_latents.dtype, device=video_latents.device)
                    ], dim=1)
                    if actions is not None:
                        # actions = actions[:, 1:, ...]  # NOTE: hard code!!!
                        actions = torch.cat([
                            actions,
                            torch.zeros((actions.size(0), pad_frames * 4, actions.size(2)),
                                        dtype=actions.dtype, device=actions.device)
                        ], dim=1)
                frame_mask = torch.ones(video_latents.size(1), device=accelerator.device).bool()
                if pad_frames > 0:
                    frame_mask[-pad_frames:] = False

                # zero-pad image latents to the same length of video latents.
                # note that length of image latents may > 1.
                padding_shape = (video_latents.size(0), video_latents.size(1) - image_latents.size(1), *video_latents.shape[2:])
                latent_padding = image_latents.new_zeros(padding_shape)
                image_latents = torch.cat([image_latents, latent_padding], dim=1)
                # CONSOLE.log(f'vidoe_latents shape (B F C H W): {video_latents.shape}')
                # CONSOLE.log(f'image_latents shape (B F C H W): {image_latents.shape}')

                depth_latents = label_latents = None

                if depths is not None:
                    depth_latent_dist = DiagonalGaussianDistribution(depths)
                    depth_latents = depth_latent_dist.sample() * VAE_SCALING_FACTOR
                    depth_latents = depth_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    depth_latents = depth_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                if labels is not None:
                    label_latent_dist = DiagonalGaussianDistribution(labels)
                    label_latents = label_latent_dist.sample() * VAE_SCALING_FACTOR
                    label_latents = label_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    label_latents = label_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                if random.random() < train_config.noised_image_dropout:
                    image_latents = torch.zeros_like(image_latents)

                # Encode prompts
                if not dataset_config.load_tensors:
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        model_config.max_text_seq_length,
                        accelerator.device,
                        weight_dtype,
                        requires_grad=False,
                    )
                else:
                    prompt_embeds = prompts.to(dtype=weight_dtype)

                # Sample noise that will be added to the latents
                noise = torch.randn_like(video_latents)
                batch_size, num_frames, _, height, width = video_latents.shape
                if batch_n_view > 1:
                    num_frames //= batch_n_view  # calculate the correct number of frames

                # TODO: uniform sampling?
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    device=accelerator.device,
                )

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=height * VAE_SCALE_FACTOR_SPATIAL,
                        width=width * VAE_SCALE_FACTOR_SPATIAL,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                        patch_size=model_config.patch_size,
                        patch_size_t=model_config.patch_size_t,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                # Create ofs embeds if required
                ofs_emb = image_latents.new_full((1,), fill_value=2.0) if is_ofs_embed else None

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)  # [B, F, 2C, H, W]
                # CONSOLE.log(f'noisy_model_input shape (B F C H W): {noisy_model_input.shape}')

                if depths is not None:
                    depths = torch.cat([depth_latents, depth_latents], dim=2)
                if labels is not None:
                    labels = torch.cat([label_latents, label_latents], dim=2)

                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    controls_or_guidances=dict(
                        actions=actions,
                        depths=depths,
                        labels=labels,
                    ),
                    timestep=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    num_views=batch_n_view,
                )
                video_output, is_action_mask, actions_recon = model_output

                video_pred = scheduler.get_velocity(video_output, noisy_video_latents, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(video_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = video_latents

                # video denoise loss
                loss = torch.mean(
                    (weights * (video_pred[:, frame_mask] - target[:, frame_mask]) ** 2).reshape(batch_size, -1),
                    dim=1,
                ).mean()

                # action loss if possible
                rot_loss = pos_loss = grip_loss = None
                if model_config.recon_action and (~is_action_mask).sum() > 0:
                    loss_weight = {
                        'rot_loss': 0.4, 'pos_loss': 5, 'grip_loss': 1,
                    }
                    action_recon_loss = CogVideoXTransformer3DModelTraj.compute_action_loss(
                        actions, actions_recon, loss_weight=loss_weight, mask=~is_action_mask,
                    )
                    rot_loss, pos_loss, grip_loss = action_recon_loss
                    loss += (rot_loss + pos_loss + grip_loss)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    gradient_norm_before_clip = get_gradient_norm(transformer.parameters())
                    accelerator.clip_grad_norm_(transformer.parameters(), optimizer_config.max_grad_norm)
                    gradient_norm_after_clip = get_gradient_norm(transformer.parameters())
                    if (gradient_norm_after_clip == 0 or gradient_norm_after_clip is None) and accelerator.state.deepspeed_plugin is None:
                        raise RuntimeError(f'Got invalid gradient {gradient_norm_after_clip=}.')

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                if not optimizer_config.use_cpu_offload_optimizer:
                    lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % train_config.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if train_config.checkpoints_total_limit is not None:
                            checkpoints = list(fnmatch.filter(
                                os.listdir(train_config.output_dir),
                                'checkpoint-*'
                            ))
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= train_config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - train_config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(train_config.output_dir, removing_checkpoint)
                                    try:
                                        shutil.rmtree(removing_checkpoint)
                                    except:
                                        CONSOLE.log(f'Failed to remove checkpoint {removing_checkpoint}')

                        save_path = os.path.join(train_config.output_dir, f'checkpoint-{global_step}')
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else train_config.learning_rate
            logs = {'loss': loss.detach().item(), 'lr': last_lr, 'n_view-n_frame': f'{batch_n_view}/{batch_n_frame}'}
            if model_config.recon_action and (~is_action_mask).sum() > 0:
                logs.update(rot_loss=rot_loss.detach().item(),
                            pos_loss=pos_loss.detach().item(),
                            grip_loss=grip_loss.detach().item())
            # gradnorm + deepspeed: https://github.com/microsoft/DeepSpeed/issues/4555
            if accelerator.distributed_type != DistributedType.DEEPSPEED:
                logs.update(
                    {
                        'gradient_norm_before_clip': gradient_norm_before_clip,
                        'gradient_norm_after_clip': gradient_norm_after_clip,
                    }
                )
            CONSOLE.log(logs)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= train_config.max_train_steps:
                break

            # ! Validation step
            if global_step % train_config.validation_steps == 0 and accelerator.sync_gradients and accelerator.is_main_process:

                run_validation(step=global_step, epoch=epoch)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        dtype = (
            torch.float16
            if train_config.mixed_precision == 'fp16'
            else torch.bfloat16
            if train_config.mixed_precision == 'bf16'
            else torch.float32
        )
        transformer = transformer.to(dtype)

        pipe = CogVideoXImageToVideoPipelineTraj.from_pretrained(
            transformer_config.pretrained_model_name_or_path,
            transformer=transformer,  # Use trained transformer 
            revision=transformer_config.revision,
            variant=transformer_config.variant,
            torch_dtype=dtype,
        )

        pipe.save_pretrained(
            os.path.join(
                train_config.output_dir,
                'checkpoint',
            ),
            safe_serialization=True,
            max_shard_size='5GB',
        )

        # Cleanup trained models to save memory
        if dataset_config.load_tensors:
            del pipe
        else:
            del text_encoder, vae, pipe

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

        # Final test inference
        run_validation(step=global_step, epoch=-1)

    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")
    parser.add_argument(
        "--base_config",
        type=str,
        default='./config/base_train.yaml',
        required=False,
    )
    parser.add_argument(
        "--debug_config",
        type=str,
        default='./config/debug.yaml',
        required=False,
    )
    # NOTE: must set default values to None to avoid override!!!
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    args = parser.parse_args()

    CONSOLE.log(f'Loading configs from {args.config} ...')
    # TODO: fix this!
    base_config = OmegaConf.load(args.base_config)
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(base_config, config)
    config.dataset = OmegaConf.merge(config.dataset, config['dataset'][args.dataset_type])
    if args.debug:
        debug_config = OmegaConf.load(args.debug_config)
        config = OmegaConf.merge(config, debug_config)
    args_config = OmegaConf.create(vars(args))
    args_config = OmegaConf.masked_copy(args_config, [k for k, v in args_config.items() if v is not None])
    config = OmegaConf.merge(config, config.runtime, args_config)

    if args.debug:
        config.train.output_dir = f'debug_{config.train.output_dir}'
    config.train.output_dir = os.path.join(
        config.train.output_path, config.train.output_dir,
    )

    return config


if __name__ == '__main__':

    config = parse_args()
    main(config)
