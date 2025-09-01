#!/usr/bin/env python3

import argparse
import json
import gc
import os
import pathlib
import queue
import traceback
import uuid
import functools
import multiprocessing
import fnmatch
import time
import torch
import torch.distributed as dist
from argparse import Namespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from einops import rearrange
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from diffusers.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from diffusers.training_utils import set_seed
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.logging import get_logger
from transformers import T5EncoderModel, T5Tokenizer

import decord  # isort:skip

from orv.dataset.dataset import RobotDataset, MultiViewRobotDataset  # isort:skip
from orv.utils import CONSOLE


decord.bridge.set_bridge('torch')

logger = get_logger(__name__)

DTYPE_MAPPING = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def check_height(x: Any) -> int:
    x = int(x)
    if x % 16 != 0:
        raise argparse.ArgumentTypeError(
            f'`--height_buckets` must be divisible by 16, but got {x} which does not fit criteria.'
        )
    return x


def check_width(x: Any) -> int:
    x = int(x)
    if x % 16 != 0:
        raise argparse.ArgumentTypeError(
            f'`--width_buckets` must be divisible by 16, but got {x} which does not fit criteria.'
        )
    return x


def check_frames(x: Any) -> int:
    x = int(x)
    if x % 4 != 0 and x % 4 != 1:
        raise argparse.ArgumentTypeError(
            f'`--frames_buckets` must be of form `4 * k` or `4 * k + 1`, but got {x} which does not fit criteria.'
        )
    return x


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id',
        type=str,
        default='THUDM/CogVideoX-2b',  # we 3D VAE
        help='Hugging Face model ID to use for tokenizer, text encoder and VAE.',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='bridgev2',
        help='Dataset type.'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to where training data is located.'
    )
    parser.add_argument('--multiview', action='store_true')
    parser.add_argument('--split', type=str, required=True, help='Split of the dataset.')
    parser.add_argument('--use_cond', action='store_true')
    parser.add_argument('--filter_by_cond', action='store_true')
    parser.add_argument('--load_condGT', action='store_true')
    parser.add_argument('--slice', action='store_true')
    parser.add_argument('--cond_data_root', type=str, required=False)
    parser.add_argument(
        '--dataset_file', type=str, default=None, help='Path to CSV file containing metadata about training data.'
    )
    parser.add_argument(
        '--caption_column',
        type=str,
        default='prompt',
        help='If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the captions. If using the folder structure format for data loading, this should be the name of the file containing line-separated captions (the file should be located in `--data_root`).',
    )
    parser.add_argument(
        '--video_column',
        type=str,
        default='video',
        help='If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the video paths. If using the folder structure format for data loading, this should be the name of the file containing line-separated video paths (the file should be located in `--data_root`).',
    )
    parser.add_argument(
        '--id_token',
        type=str,
        default=None,
        help='Identifier token appended to the start of each prompt if provided.',
    )
    parser.add_argument(
        '--set_uuid',
        action='store_true',
        help='Identifier token appended to the start of each prompt if provided.',
    )
    parser.add_argument(
        '--dataloader_num_workers',
        type=int,
        default=0,
        help='Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.',
    )
    parser.add_argument(
        '--ref_num',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--pin_memory',
        action='store_true',
        help='Whether or not to use the pinned memory setting in pytorch dataloader.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to output directory where preprocessed videos/latents/embeddings will be saved.',
    )
    parser.add_argument(
        '--max_sequence_length', type=int, default=226, help='Max sequence length of prompt embeddings.'
    )
    parser.add_argument('--target_fps', type=int, default=8, help='Frame rate of output videos.')
    parser.add_argument(
        '--use_slicing',
        action='store_true',
        help='Whether to enable sliced encoding/decoding in the VAE. Only used if `--save_latents_and_embeddings` is also used.',
    )
    parser.add_argument(
        '--use_tiling',
        action='store_true',
        help='Whether to enable tiled encoding/decoding in the VAE. Only used if `--save_latents_and_embeddings` is also used.',
    )
    parser.add_argument('--batch_size', type=int, default=1, help='Number of videos to process at once in the VAE.')
    parser.add_argument(
        '--num_decode_threads',
        type=int,
        default=0,
        help='Number of decoding threads for `decord` to use. The default `0` means to automatically determine required number of threads.',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        choices=['fp32', 'fp16', 'bf16'],
        default='fp32',
        help='Data type to use when generating latents and prompt embeddings.',
    )
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility.')
    parser.add_argument(
        '--num_artifact_workers', type=int, default=4, help='Number of worker threads for serializing artifacts.'
    )
    return parser.parse_args()


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt',
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError('`text_input_ids` must be provided when the tokenizer is not specified.')

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompts: List[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompts,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompts,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


to_pil_image = transforms.ToPILImage(mode='RGB')


def save_image(image: torch.Tensor, path: pathlib.Path) -> None:
    save_image = to_pil_image(image[0])

    if (num_frame := image.size(0)) > 1:
        width, height = save_image.size
        merged = Image.new('RGB', (width * num_frame, height))
        merged.paste(save_image, (0, 0))
        for i in range(1, num_frame):
            pil_image = to_pil_image(image[i])
            merged.paste(pil_image, (width * i, 0))
        save_image = merged

    save_image.save(path)


def save_depth(depth: torch.Tensor, path: pathlib.Path) -> None:
    pass


def save_video(video: torch.Tensor, path: pathlib.Path, fps: int = 8) -> None:
    video = [to_pil_image(frame) for frame in video]
    export_to_video(video, path, fps=fps)


def save_prompt(prompt: str, path: pathlib.Path) -> None:
    with open(path, 'w', encoding='utf-8') as file:
        file.write(prompt)


def save_metadata(metadata: Dict[str, Any], path: pathlib.Path) -> None:
    with open(path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(metadata))


@torch.no_grad()
def serialize_artifacts(
    batch_size: int,
    fps: int,
    n_view: int,
    is_multiview: bool,
    set_uuid: bool = False,
    ids: Optional[List[int]] = None,
    start_frame_idxs: Optional[List[int]] = None,
    num_frames: Optional[List[int]] = None,
    images_dir: Optional[pathlib.Path] = None,
    image_latents_dir: Optional[pathlib.Path] = None,
    videos_dir: Optional[pathlib.Path] = None,
    video_latents_dir: Optional[pathlib.Path] = None,
    depths_dir: Optional[pathlib.Path] = None,
    depth_latents_dir: Optional[pathlib.Path] = None,
    depthGT_latents_dir: Optional[pathlib.Path] = None,
    labels_dir: Optional[pathlib.Path] = None,
    label_latents_dir: Optional[pathlib.Path] = None,
    labelGT_latents_dir: Optional[pathlib.Path] = None,
    prompts_dir: Optional[pathlib.Path] = None,
    prompt_embeds_dir: Optional[pathlib.Path] = None,
    images: Optional[torch.Tensor] = None,
    image_latents: Optional[torch.Tensor] = None,  # [b, f, c, h, w]
    videos: Optional[torch.Tensor] = None,
    video_latents: Optional[torch.Tensor] = None,  # [b, f, c, h, w]
    depths: Optional[torch.Tensor] = None,
    depth_latents: Optional[torch.Tensor] = None,
    depthGT_latents: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    label_latents: Optional[torch.Tensor] = None,
    labelGT_latents: Optional[torch.Tensor] = None,
    prompts: Optional[List[str]] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
) -> None:

    data_folder_mapper_list = [
        (images, images_dir, lambda img, path: save_image(img, path), 'png'),
        (image_latents, image_latents_dir, torch.save, 'pt'),
        (videos, videos_dir, functools.partial(save_video, fps=fps), 'mp4'),
        (video_latents, video_latents_dir, torch.save, 'pt'),
        (depths, depths_dir, lambda depth, path: save_depth(depth, path), 'png'),
        (depth_latents, depth_latents_dir, torch.save, 'pt'),
        (depthGT_latents, depthGT_latents_dir, torch.save, 'pt'),
        (labels, labels_dir, lambda label, path: save_image(label, path), 'png'),
        (label_latents, label_latents_dir, torch.save, 'pt'),
        (labelGT_latents, labelGT_latents_dir, torch.save, 'pt'),
        (prompts, prompts_dir, save_prompt, 'txt'),
        (prompt_embeds, prompt_embeds_dir, torch.save, 'pt'),
    ]

    if set_uuid:
        filenames = [uuid.uuid4() for _ in range(batch_size)]
    else:
        filenames = [f'{id:05d}_{start_frame_idx:02d}_{num_frame:02d}'
                     for id, start_frame_idx, num_frame in zip(ids, start_frame_idxs, num_frames)]

    for data, folder, save_fn, extension in data_folder_mapper_list:
        if data is None or not os.path.exists(folder):
            continue
        for slice, filename in zip(data, filenames):
            if isinstance(slice, torch.Tensor):
                slice = slice.clone().to('cpu')
            if slice.ndim == 4:
                if 'latents' not in str(folder):
                    # for images/videos, dimension is [f c h w]
                    slice = rearrange(slice, '(v f) c h w -> v f c h w', v=n_view)
                    for i in range(n_view):
                        filename_i = filename
                        if is_multiview:
                            filename_i = f'{filename}_{i}'
                        path = folder.joinpath(f'{filename_i}.{extension}')
                        save_fn(slice[i, ...], path)
                elif 'latents' in str(folder):
                    # for latents, dimension is [c f h w]
                    # CONSOLE.log(f'{slice.shape=}, {folder=}')
                    slice = rearrange(slice, 'c (v f) h w -> c v f h w', v=n_view)
                    for i in range(n_view):
                        filename_i = filename
                        if is_multiview:
                            filename_i = f'{filename}_{i}'
                        path = folder.joinpath(f'{filename_i}.{extension}')
                        save_fn(slice[:, i, ...], path)
            else:
                path = folder.joinpath(f'{filename}.{extension}')
                save_fn(slice, path)
            del slice
            gc.collect()
            torch.cuda.empty_cache()


def save_intermediates(output_queue: queue.Queue) -> None:
    while True:
        try:
            item = output_queue.get(timeout=30)
            if item is None:
                break
            serialize_artifacts(**item)

        except queue.Empty:
            continue

        except Exception as e:
            print(f'Error: {e}')
            break


@torch.no_grad()
def main():

    process_keys = [
        'images',
        'image_latents',
        'videos',
        'video_latents',
        # 'depths',
        # 'depth_latents',
        # 'depthsGT',
        # 'depthGT_latents',
        # 'labels',
        # 'labelsGT',
        # 'labelGT_latents',
        # 'label_latents',
        # 'prompts',
        # 'prompt_embeds',
    ]
    CONSOLE.log(f'Processing keys: {process_keys}')

    is_process = lambda key: key in process_keys

    args = get_args()
    set_seed(args.seed)

    output_dir = pathlib.Path(args.output_dir) / args.split
    tmp_dir = output_dir.joinpath('tmp')

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create task queue for non-blocking serializing of artifacts
    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=args.num_artifact_workers * 2)
    save_future = save_thread.submit(save_intermediates, output_queue)

    # Initialize distributed processing
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        CONSOLE.log(f'Initialized process {rank} of {world_size}.')
    else:
        # Single GPU
        local_rank = 0
        world_size = 1
        rank = 0
        torch.cuda.set_device(rank)
        CONSOLE.log('Initialized with single process.')

    ref_num = args.ref_num

    # Create folders where intermediate tensors from each rank will be saved
    images_dir = tmp_dir.joinpath(f'images{ref_num}/{rank}')
    image_latents_dir = tmp_dir.joinpath(f'image{ref_num}_latents/{rank}')
    videos_dir = tmp_dir.joinpath(f'videos/{rank}')
    video_latents_dir = tmp_dir.joinpath(f'video_latents/{rank}')
    depths_dir = tmp_dir.joinpath(f'depths/{rank}')
    depth_latents_dir = tmp_dir.joinpath(f'depth_latents/{rank}')
    depthGT_latents_dir = tmp_dir.joinpath(f'depthGT_latents/{rank}')
    labels_dir = tmp_dir.joinpath(f'labels/{rank}')
    label_latents_dir = tmp_dir.joinpath(f'label_latents/{rank}')
    labelGT_latents_dir = tmp_dir.joinpath(f'labelGT_latents/{rank}')
    label_latents_dir = tmp_dir.joinpath(f'label_latents/{rank}')
    prompts_dir = tmp_dir.joinpath(f'prompts/{rank}')
    prompt_embeds_dir = tmp_dir.joinpath(f'prompt_embeds/{rank}')
    if is_process('images'):
        images_dir.mkdir(parents=True, exist_ok=True)
    if is_process('image_latents'):
        image_latents_dir.mkdir(parents=True, exist_ok=True)
    if is_process('videos'):
        videos_dir.mkdir(parents=True, exist_ok=True)
    if is_process('video_latents'):
        video_latents_dir.mkdir(parents=True, exist_ok=True)
    if is_process('depths'):
        assert args.use_cond, f'Need to set `use_cond` if need to process depth maps!'
        depths_dir.mkdir(parents=True, exist_ok=True)
    if is_process('depth_latents'):
        assert args.use_cond, f'Need to set `use_cond` if need to process depth maps!'
        depth_latents_dir.mkdir(parents=True, exist_ok=True)
    if is_process('depthGT_latents'):
        assert args.use_cond, f'Need to set `use_cond` if need to process depth maps!'
        depthGT_latents_dir.mkdir(parents=True, exist_ok=True)
    if is_process('labels'):
        assert args.use_cond, f'Need to set `use_cond` if need to process label maps!'
        labels_dir.mkdir(parents=True, exist_ok=True)
    if is_process('label_latents'):
        assert args.use_cond, f'Need to set `use_cond` if need to process label maps!'
        label_latents_dir.mkdir(parents=True, exist_ok=True)
    if is_process('labelGT_latents'):
        assert args.use_cond, f'Need to set `use_cond` if need to process label maps!'
        labelGT_latents_dir.mkdir(parents=True, exist_ok=True)
    if is_process('label_latents'):
        assert args.use_cond, f'Need to set `use_cond` if need to process label maps!'
        label_latents_dir.mkdir(parents=True, exist_ok=True)
    if is_process('prompts'):
        prompts_dir.mkdir(parents=True, exist_ok=True)
    if is_process('prompt_embeds'):
        prompt_embeds_dir.mkdir(parents=True, exist_ok=True)

    weight_dtype = DTYPE_MAPPING[args.dtype]
    target_fps = args.target_fps
    set_uuid = args.set_uuid
    is_multiview = False

    os.environ['DEBUG'] = '0'

    # 1. Dataset
    if args.dataset == 'bridgev2':

        if not args.multiview:
            dataset = RobotDataset(
                data_root=args.data_root,
                split=args.split,
                use_cond=args.use_cond,
                filter_by_cond=args.filter_by_cond,
                ref_num=ref_num,
                control_keys=['depth', 'label'],
                load_actions=False,
                load_tensor=False,
                load_condGT=args.load_condGT,
                use_3dvae=True,
                slice_frame=args.slice,
                video_size = [480, 640],
                ori_size = [480, 640],
            )
        else:
            is_multiview = True
            dataset = MultiViewRobotDataset(
                data_root=args.data_root,
                split=args.split,
                use_cond=args.use_cond,
                filter_by_cond=args.filter_by_cond,
                n_view=3,
                camera_ids=[0, 1, 2],
                sequence_interval=1,
                sequence_length=16,
                start_frame_interval={
                    'train': 4, 'val': 16, 'test': 16,
                },
                video_size=[320, 480],
                ori_size=[256, 320],
                caption_column='texts',
                ref_num=ref_num,
                control_keys=['depth', 'label'],
                load_actions=False,
                load_tensor=False,
                load_condGT=args.load_condGT,
                use_3dvae=True,
                slice_frame=args.slice,
                num_samples=-1,
                train=False,
            )

    elif args.dataset == 'droid':

        limit_size = {'train': 120000, 'val': 2600}
        dataset = MultiViewRobotDataset(
            data_root=args.data_root,
            split=args.split,
            use_cond=args.use_cond,
            filter_by_cond=args.filter_by_cond,
            n_view=2,
            camera_ids=[0, 1],
            sequence_interval=3,
            sequence_length=28,
            start_frame_interval={
                'train': 16, 'val': 72, 'test': 72,
            },
            video_size=[256, 384],
            ori_size=[176, 320],
            caption_column='language_instruction',
            ref_num=ref_num,
            load_actions=False,
            load_tensor=False,
            load_condGT=args.load_condGT,
            use_3dvae=True,
            slice_frame=args.slice,
            # limit the dataset size!
            num_samples=limit_size[args.split],
            sample_mode='drop_last',
        )

    elif args.dataset == 'rt1':

        limit_size = {'train': 120000, 'val': 2600}
        dataset = RobotDataset(
            data_root=args.data_root,
            split=args.split,
            use_cond=args.use_cond,
            filter_by_cond=args.filter_by_cond,
            sequence_interval=2,
            ref_num=ref_num,
            control_keys=['label', 'depth'],
            load_actions=False,
            load_tensor=False,
            load_condGT=args.load_condGT,
            use_3dvae=True,
            slice_frame=args.slice,
            start_frame_interval={
                'train': 6,'val': 16,'test': 16
            },
            # limit the dataset size !
            num_samples=limit_size[args.split],
            sample_mode='drop_last',
        )

    else:
        raise ValueError(f'Invalid dataset {args.dataset}.')

    n_view = dataset.config.get('n_view', 1)
    original_dataset_size = len(dataset)
    CONSOLE.log(f'Loaded {dataset.__class__.__name__} with length of {original_dataset_size}.')

    # Split data among GPUs
    if world_size > 1:
        samples_per_gpu = original_dataset_size // world_size
        start_index = rank * samples_per_gpu
        end_index = start_index + samples_per_gpu
        if rank == world_size - 1:
            end_index = original_dataset_size  # Make sure the last GPU gets the remaining data
        CONSOLE.log(f'Process {rank} will host {end_index - start_index} samples from {start_index} to {end_index}.')

        # Slice the data
        dataset.samples = dataset.samples[start_index : end_index]

    rank_dataset_size = len(dataset)

    # 2. Dataloader
    def collate_fn(data):
        prompts = [x['prompt'] for x in data]

        images = None
        if is_process('image_latents') or is_process('images'):
            images = [x['image'] for x in data]
            images = torch.stack(images).to(dtype=weight_dtype, non_blocking=True)

        videos = None
        if is_process('video_latents') or is_process('videos'):
            videos = [x['videos'] for x in data]
            videos = torch.stack(videos).to(dtype=weight_dtype, non_blocking=True)

        depths = None
        if 'depths' in data[0]:
            depths = [x['depths'] for x in data]
            depths = torch.stack(depths).to(dtype=weight_dtype, non_blocking=True)

        labels = None
        if 'labels' in data[0]:
            labels = [x['labels'] for x in data]
            labels = torch.stack(labels).to(dtype=weight_dtype, non_blocking=True)

        episode_ids = [int(x['metainfo']['episode_id']) for x in data]
        start_frame_idxs = [int(x['metainfo']['start_frame_idx']) for x in data]
        num_frames = [int(x['metainfo']['num_frame']) for x in data]
        num_views = [int(x['metainfo']['num_view']) for x in data]

        return {
            'images': images,
            'videos': videos,
            'depths': depths,
            'labels': labels,
            'prompts': prompts,
            'episode_ids': episode_ids,
            'start_frame_idxs': start_frame_idxs,
            'num_frames': num_frames,
            'num_views': num_views
        }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        # sampler=BucketSampler(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False),
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )

    # 3. Prepare models
    device = f'cuda:{rank}'

    if is_process('prompt_embeds'):
        tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder='tokenizer')
        text_encoder = T5EncoderModel.from_pretrained(
            args.model_id, subfolder='text_encoder', torch_dtype=weight_dtype
        )
        text_encoder = text_encoder.to(device)

    if is_process('image_latents') or is_process('video_latents') or \
        is_process('depth_latents') or is_process('depthGT_latents') or \
        is_process('label_latents') or is_process('labelGT_latents'):

        vae = AutoencoderKLCogVideoX.from_pretrained(args.model_id, subfolder='vae', torch_dtype=weight_dtype)
        vae = vae.to(device)

        if args.use_slicing:
            vae.enable_slicing()
        if args.use_tiling:
            vae.enable_tiling()

    # 4. Compute latents and embeddings and save
    if rank == 0:
        iterator = tqdm(
            dataloader, desc='Encoding', total=(rank_dataset_size + args.batch_size - 1) // args.batch_size
        )
    else:
        iterator = dataloader

    for step, batch in enumerate(iterator):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        CONSOLE.log(f'{torch.cuda.memory_allocated() / 1024**3}GB')
        CONSOLE.log(f'{torch.cuda.memory_reserved() / 1024**3}GB')
        n_view = batch['num_views'][0]
        CONSOLE.log(f'Current batch has {n_view=}')
        while torch.cuda.memory_allocated() / 1024**3 > 70:
            time.sleep(1)
            CONSOLE.log(f'Waiting for the saving task thread ...')
        try:
            images = None
            image_latents = None
            videos = None
            video_latents = None
            depths = None
            depth_latents = None
            depthsGT = None
            depthGT_latents = None
            labels = None
            label_latents = None
            labelsGT = None
            labelGT_latents = None
            prompt_embeds = None

            if is_process('images') or is_process('image_latents'):
                images = batch['images'].to(device, non_blocking=True)
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            if is_process('videos') or is_process('video_latents'):
                videos = batch['videos'].to(device, non_blocking=True)
                videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            if is_process('depths') or is_process('depth_latents'):
                depths = batch['depths'].to(device, non_blocking=True)
                depths = depths.repeat(1, 1, 3, 1, 1)  # expand to 3 channels
                depths = depths.permute(0, 2, 1, 3, 4)

            if is_process('depthsGT') or is_process('depthGT_latents'):
                assert args.load_condGT, f'Should set `load_condGT`!'
                depthsGT = batch['depths'].to(device, non_blocking=True)
                depthsGT = depthsGT.repeat(1, 1, 3, 1, 1)
                depthsGT = depthsGT.permute(0, 2, 1, 3, 4)

            if is_process('labels') or is_process('label_latents'):
                labels = batch['labels'].to(device, non_blocking=True)
                labels = labels.permute(0, 2, 1, 3, 4)

            if is_process('labelsGT') or is_process('labelGT_latents'):
                assert args.load_condGT, f'Should set `load_condGT`!'
                labelsGT = batch['labels'].to(device, non_blocking=True)
                labelsGT = labelsGT.permute(0, 2, 1, 3, 4)

            prompts = batch['prompts']

            # ! Encode images
            if is_process('image_latents'):

                if args.use_slicing:
                    encoded_slices = [vae._encode(image_slice) for image_slice in images.split(1)]  # type: ignore
                    image_latents = torch.cat(encoded_slices)
                else:
                    images = rearrange(images, 'b c (v f) h w -> b c v f h w', v=n_view)
                    image_latents = torch.stack(
                        [
                            vae._encode(images[:, :, i, ...])  # type: ignore
                            for i in range(n_view)
                        ],
                        dim=2,
                    )  # [b, c, v, f, h, w]
                    images = rearrange(images, 'b c v f h w -> b c (v f) h w')
                    image_latents = rearrange(image_latents, 'b c v f h w -> b c (v f) h w')

                image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

            # ! Encode videos
            if is_process('video_latents'):

                if args.use_slicing:
                    if n_view > 1:
                        raise NotImplementedError
                    encoded_slices = [vae._encode(video_slice) for video_slice in videos.split(1)]  # type: ignore
                    video_latents = torch.cat(encoded_slices)
                else:
                    videos = rearrange(videos, 'b c (v f) h w -> b c v f h w', v=n_view)
                    video_latents = torch.stack(
                        [
                            vae._encode(videos[:, :, i, ...])  # type: ignore
                            for i in range(n_view)
                        ],
                        dim=2,
                    )  # [b, c, v, f, h, w]
                    videos = rearrange(videos, 'b c v f h w -> b c (v f) h w')
                    video_latents = rearrange(video_latents, 'b c v f h w -> b c (v f) h w')

                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

            # ! Encode depths
            if is_process('depth_latents'):

                if args.use_slicing:
                    if n_view > 1:
                        raise NotImplementedError
                    encoded_slices = [vae._encode(depth_slice) for depth_slice in depths.split(1)]  # type: ignore
                    depth_latents = torch.cat(encoded_slices)
                else:
                    depths = rearrange(depths, 'b c (v f) h w -> b c v f h w', v=n_view)
                    depth_latents = torch.stack(
                        [
                            vae._encode(depths[:, :, i, ...])  # type: ignore
                            for i in range(n_view)
                        ],
                        dim=2,
                    )  # [b, c, v, f, h, w]
                    depths = rearrange(depths, 'b c v f h w -> b c (v f) h w')
                    depth_latents = rearrange(depth_latents, 'b c v f h w -> b c (v f) h w')

                depth_latents = depth_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

            # ! Encode GT depths
            if is_process('depthGT_latents'):

                if args.use_slicing:
                    encoded_slices = [vae._encode(depth_slice) for depth_slice in depthsGT.split(1)]  # type: ignore
                    depthGT_latents = torch.cat(encoded_slices)
                else:
                    depthGT_latents = vae._encode(depthsGT)  # type: ignore

                depthGT_latents = depthGT_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

            # ! Encode labels
            if is_process('label_latents'):

                if args.use_slicing:
                    if n_view > 1:
                        raise NotImplementedError
                    encoded_slices = [vae._encode(label_slice) for label_slice in labels.split(1)]  # type: ignore
                    label_latents = torch.cat(encoded_slices)
                else:
                    labels = rearrange(labels, 'b c (v f) h w -> b c v f h w', v=n_view)
                    label_latents = torch.stack(
                        [
                            vae._encode(labels[:, :, i, ...])  # type: ignore
                            for i in range(n_view)
                        ],
                        dim=2,
                    )  # [b, c, v, f, h, w]
                    labels = rearrange(labels, 'b c v f h w -> b c (v f) h w')
                    label_latents = rearrange(label_latents, 'b c v f h w -> b c (v f) h w')

                label_latents = label_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

            # ! Encode GT labels
            if is_process('labelGT_latents'):

                if args.use_slicing:
                    encoded_slices = [vae._encode(label_slice) for label_slice in labelsGT.split(1)]  # type: ignore
                    labelGT_latents = torch.cat(encoded_slices)
                else:
                    labelGT_latents = vae._encode(labelsGT)  # type: ignore

                labelGT_latents = labelGT_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

            # ! Encode prompts
            if is_process('prompt_embeds'):
                prompt_embeds = compute_prompt_embeddings(
                    tokenizer,  # type: ignore
                    text_encoder,  # type: ignore
                    prompts,
                    args.max_sequence_length,
                    device,
                    weight_dtype,
                    requires_grad=False,
                )

            # ! Prepare images
            if images is not None:
                images = (images.permute(0, 2, 1, 3, 4) + 1) / 2  # [b, f, c, h, w]

            # ! Prepare videos
            if videos is not None:
                videos = (videos.permute(0, 2, 1, 3, 4) + 1) / 2  # [-1, 1] -> [0, 1]

            # ! Prepare depths
            if depths is not None:
                depths = depths.permute(0, 2, 1, 3, 4)[:, :, :1, ...]  # back to 1 channel

            # ! Prepare labels
            if labels is not None:
                labels = (labels.permute(0, 2, 1, 3, 4) + 1) / 2

            output_queue.put(
                {
                    'batch_size': len(prompts),
                    'fps': target_fps,
                    'n_view': n_view,
                    'is_multiview': is_multiview,
                    'images_dir': images_dir,
                    'image_latents_dir': image_latents_dir,
                    'videos_dir': videos_dir,
                    'set_uuid': set_uuid,
                    'ids': batch['episode_ids'],
                    'start_frame_idxs': batch['start_frame_idxs'],
                    'num_frames': batch['num_frames'],
                    'video_latents_dir': video_latents_dir,
                    'depths_dir': depths_dir,
                    'depth_latents_dir': depth_latents_dir,
                    'depthGT_latents_dir': depthGT_latents_dir,
                    'labels_dir': labels_dir,
                    'label_latents_dir': label_latents_dir,
                    'labelGT_latents_dir': labelGT_latents_dir,
                    'prompts_dir': prompts_dir,
                    'prompt_embeds_dir': prompt_embeds_dir,
                    'images': images,
                    'image_latents': image_latents,
                    'videos': videos,
                    'video_latents': video_latents,
                    'depths': depths,
                    'depth_latents': depth_latents,
                    'depthGT_latents': depthGT_latents,
                    'labels': labels,
                    'label_latents': label_latents,
                    'labelGT_latents': labelGT_latents,
                    'prompts': prompts,
                    'prompt_embeds': prompt_embeds,
                }
            )

            del images
            del image_latents
            del videos
            del video_latents
            del depths
            del depth_latents
            del depthsGT
            del depthGT_latents
            del labels
            del label_latents
            del labelsGT
            del labelGT_latents
            del prompt_embeds

        except Exception:
            print('-------------------------')
            print(f'An exception occurred while processing data: {rank=}, {world_size=}, {step=}')
            traceback.print_exc()
            print('-------------------------')

    # 5. Complete distributed processing
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    output_queue.put(None)
    save_thread.shutdown(wait=True)
    save_future.result()

    # 6. Combine results from each rank
    if rank == 0:
        print(
            f'Completed preprocessing latents and embeddings. Temporary files from all ranks saved to `{tmp_dir.as_posix()}`'
        )

        # Move files from each rank to common directory
        for subfolder, extension in [
            (f'images{ref_num}', 'png'),
            (f'image{ref_num}_latents', 'pt'),
            ('videos', 'mp4'),
            ('video_latents', 'pt'),
            ('depth_latents', 'pt'),
            ('depthGT_latents', 'pt'),
            ('label_latents', 'pt'),
            ('labelGT_latents', 'pt'),
            ('prompts', 'txt'),
            ('prompt_embeds', 'pt'),
            ('videos', 'txt'),
        ]:
            clean_subfolder = subfolder
            if fnmatch.fnmatch(subfolder, 'images*'):
                clean_subfolder = 'images'
            if fnmatch.fnmatch(subfolder, 'image*latents'):
                clean_subfolder = 'image_latents'
            if not is_process(clean_subfolder):
                continue
            tmp_subfolder = tmp_dir.joinpath(subfolder)
            combined_subfolder = output_dir.joinpath(subfolder)
            combined_subfolder.mkdir(parents=True, exist_ok=True)
            pattern = f'*.{extension}'

            for file in tqdm(tmp_subfolder.rglob(pattern)):
                file.replace(combined_subfolder / file.name)

        # Remove temporary directories
        def rmdir_recursive(dir: pathlib.Path) -> None:
            for child in dir.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    rmdir_recursive(child)
            dir.rmdir()

        rmdir_recursive(tmp_dir)

        print(f'Completed preprocessing. All files saved to `{output_dir.as_posix()}`')


@torch.no_grad()
def encode_empty_prompt():
    args = get_args()
    set_seed(args.seed)

    output_path = pathlib.Path(args.output_dir) / 'empty_prompt.pt'

    weight_dtype = DTYPE_MAPPING[args.dtype]
    device = 'cuda'

    tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder='tokenizer')
    text_encoder = T5EncoderModel.from_pretrained(
        args.model_id, subfolder='text_encoder', torch_dtype=weight_dtype
    )
    text_encoder = text_encoder.to(device)

    prompt_embeds = compute_prompt_embeddings(
        tokenizer,
        text_encoder,
        '',
        args.max_sequence_length,
        device,
        weight_dtype,
        requires_grad=False,
    ).cpu()

    torch.save(prompt_embeds, output_path)

    print(f'Completed!')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()

    # encode_empty_prompt()
