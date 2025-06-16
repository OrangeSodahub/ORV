import argparse
from concurrent.futures import ThreadPoolExecutor
import traceback
import torch
import os
import queue
import pathlib
import multiprocessing
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler
from diffusers.configuration_utils import FrozenDict
from diffusers.utils.export_utils import export_to_video

from models.cogvideox_control import CogVideoXImageToVideoPipelineTraj, CogVideoXTransformer3DModelTraj
from dataset.dataset import RobotDataset, MultiViewRobotDataset, CollateFunctionControl, BucketSampler
from pipelines.utils import CONSOLE


@torch.no_grad()
def serialize_artifacts(
    gifs_folder: Path,
    mp4s_folder: Path,
    data_infos: dict,
    num_frames: int,
    num_views: int,
    image_size: tuple,
    videos: list[list[Image.Image]],
):

    W, H = image_size

    num_batches = len(videos) // num_views
    for i_batch in range(num_batches):

        batch_videos = videos[i_batch * num_views : (i_batch + 1) * num_views]
        data_info = data_infos[i_batch]

        if num_views > 1:
            video = []
            for i_frame in range(num_frames):
                canvas = Image.new('RGB', (W * num_views, H))
                for i_view in range(num_views):
                    canvas.paste(batch_videos[i_view][i_frame], (W * i_view, 0))
                video.append(canvas)
        else:
            video = batch_videos[0]

        sample_name = data_info['sample_name']
        output_mp4_path = os.path.join(str(mp4s_folder), f'eval_{sample_name}.mp4')
        export_to_video(video, output_mp4_path, fps=10)

        output_gif_path = os.path.join(str(gifs_folder), f'eval_{sample_name}.gif')
        video[0].save(output_gif_path, save_all=True, append_images=video[1:], duration=100, loop=0)
        CONSOLE.log(f'Exported GIF to [bold yellow]{output_gif_path}[/]')


def save_results(output_queue: queue.Queue) -> None:
    while True:
        try:
            item = output_queue.get(timeout=30)
            if item is None:
                break
            serialize_artifacts(**item)

        except queue.Empty:
            continue

        except Exception as e:
            CONSOLE.log(f'[bold red]Error: {e}!')
            break


def main(config):

    # Initialize distributed processing
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        CONSOLE.log(f'Initialized process {rank} of {world_size}.')
    else:
        # Single GPU
        local_rank = 0
        world_size = 1
        rank = 0
        torch.cuda.set_device(rank)
        CONSOLE.log('Initialized with single process.')

    eval_config = config.evaluation
    dataset_config = config.dataset
    model_config = config.transformer

    if eval_config.output_dir is None:
        if 'checkpoint' in os.path.basename(model_config.transformer_model_name_or_path):
            eval_config.output_dir = f'eval_{Path(model_config.transformer_model_name_or_path).parent.name}'
        else:
            eval_config.output_dir = f'eval_{Path(model_config.transformer_model_name_or_path).name}'
    output_folder = os.path.join(
        eval_config.output_path, eval_config.output_dir,
    )
    CONSOLE.log(f'[bold yellow] Results will be saved to {output_folder}.')
    dtype = torch.float16 if model_config.dtype == "float16" else torch.bfloat16

    output_folder = pathlib.Path(output_folder)
    tmp_folder = output_folder.joinpath('tmp')
    output_folder.mkdir(parents=True, exist_ok=True)
    tmp_folder.mkdir(parents=True, exist_ok=True)

    gifs_folder = tmp_folder.joinpath(f'gifs/{rank}')
    mp4s_folder = tmp_folder.joinpath(f'mp4s/{rank}')
    gifs_folder.mkdir(parents=True, exist_ok=True)
    mp4s_folder.mkdir(parents=True, exist_ok=True)

    # 0. create task queue for save results
    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=8)
    save_future = save_thread.submit(save_results, output_queue)

    # 1. setup eval dataset
    CONSOLE.log(f'Setting up Eval Dataset of {dataset_config.split} split ...')
    if not dataset_config.slice_frame and eval_config.batch_size > 1:
        # TODO: use bucket tools!
        eval_config.batch_size = 1
        CONSOLE.log(f"[on red]You're trying inference with slice_frame=False, then batch_size will be set to 1.")

    dataset_init_kwargs = dict(
        data_root=dataset_config.data_root,
        split=dataset_config.split,
        renderings_folder=dataset_config.renderings_folder,
        embeddings_folder=dataset_config.embeddings_folder,
        seed=dataset_config.seed,
        num_samples=dataset_config.num_samples,
        use_cond=dataset_config.use_cond,
        camera_ids=dataset_config.camera_ids,
        action_dim=dataset_config.action_dim,
        sequence_interval=dataset_config.sequence_interval,
        sequence_length=dataset_config.sequence_length,
        start_frame_interval=dataset_config.start_frame_interval,
        video_size=dataset_config.video_size,
        ori_size=dataset_config.ori_size,
        control_keys=dataset_config.control_keys,
        caption_column=dataset_config.caption_column,
        ref_num=dataset_config.num_observation,
        load_actions=dataset_config.load_actions,
        load_tensor=dataset_config.load_tensors,
        load_video=dataset_config.load_video,
        empty_prompt=dataset_config.empty_prompt,
        load_condGT=dataset_config.load_condGT,
        slice_frame=dataset_config.slice_frame,
        use_3dvae=dataset_config.use_3dvae,
        test_mode=True,
        no_normalize=dataset_config.no_normalize,
        train=False,
    )

    if not dataset_config.multiview:
        dataset = RobotDataset(**dataset_init_kwargs)
    else:
        n_view = len(dataset_config.camera_ids)
        dataset = MultiViewRobotDataset(n_view=n_view, **dataset_init_kwargs)

    use_bucket_sampler = dataset.num_refs > 1 or dataset_config.multiview

    # ! Split data among GPUs
    if world_size > 1:
        original_dataset_size = len(dataset)
        samples_per_gpu = original_dataset_size // world_size
        start_index = rank * samples_per_gpu
        end_index = start_index + samples_per_gpu
        if rank == world_size - 1:
            end_index = original_dataset_size  # Make sure the last GPU gets the remaining data
        CONSOLE.log(f'Process {rank} will host {end_index - start_index} samples from {start_index} to {end_index}.')

        # Slice the data
        dataset.samples = dataset.samples[start_index : end_index]

    dataloader = DataLoader(
        dataset,
        batch_size=eval_config.batch_size,
        sampler=(
            BucketSampler(dataset,
                          batch_size=eval_config.batch_size,
                          shuffle=False,
                          drop_last=True,
                          train=False,
                        )
            if use_bucket_sampler else None
        ),
        collate_fn=CollateFunctionControl(dtype, load_tensors=dataset_config.load_tensors),
        # num_workers=eval_config.num_workers if world_size == 1 else 0,
        num_workers=0,
        pin_memory=eval_config.pin_memory,
    )

    # 2. setup inference pipeline
    device = 'cuda' if world_size == 1 else f'cuda:{rank}'

    if 'model_index.json' in os.listdir(model_config.transformer_model_name_or_path):
        pipe = CogVideoXImageToVideoPipelineTraj.from_pretrained(
            model_config.transformer_model_name_or_path, torch_dtype=dtype,
        )
    else:
        CONSOLE.log(f'[yellow] Not found model_index.json, will load transformer only ...')
        transformer = CogVideoXTransformer3DModelTraj.from_pretrained(
            model_config.transformer_model_name_or_path,
            subfolder='transformer',
            torch_dtype=dtype,
        )
        pipe = CogVideoXImageToVideoPipelineTraj.from_pretrained(
            model_config.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=dtype,
        )
    CONSOLE.log(f'Loaded pipeline {pipe.__class__.__name__}')

    for param in pipe.transformer.parameters():
        param.requires_grad = False

    # 2.1 Set Scheduler.
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')

    # 2.2 Other configs
    CONSOLE.log(f'Configurating pipeline ...')
    pipe.to(device, dtype=dtype)
    # pipe.enable_sequential_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    pipe.transformer.gradient_checkpointing = False

    # ! IMPORTANT: only pretrained CogVideoX should set `invert_scale_latents`!
    # TODO: fix this term when using CogVideoX's settings!
    pipe.vae._internal_dict = FrozenDict(**(pipe.vae.config | dict(invert_scale_latents=False)))

    mode = eval_config.mode  # 'traj-image', 'traj-image-depth', 'traj-image-label'
    CONSOLE.log(f'Running mode : [bold yellow]{mode}[/] with control keys of dataset: [bold yellow]{dataset_config.control_keys}[/]')

    for batch in tqdm(dataloader):

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:

            # 3. prepare inputs
            # if load preprocessed tensors, images has dimensions of
            # 'b c (v f) c h w', where v is num_views, f is ref_num;
            # if not, pil_images is a list with length equal to
            # n_batch * n_view * ref_num, where each element is an image.
            if dataset_config.load_tensors:
                assert 'images' in batch, 'Missing keys: `images`!'
                images = batch['images']  # [B, C, F, H, W]
                fh, fw = images.shape[-2:]
                height = int(fh * 8)
                width = int(fw * 8)
            else:
                assert 'pil_images' in batch, 'Missing keys: `pil_images`!'
                images = batch['pil_images']
                width, height = images[0].size

            pipeline_args = {
                'image': images,
                'prompt': batch['prompts'],
                'prompt_embeds': batch['prompt_embeds'].to(device),
                'negative_prompt': 'The video is not of a high quality, it has a low resolution. Strange body and strange trajectory. Distortion.',
                'controls_or_guidances': {
                    'actions': (
                        batch['controls']['actions'].to(device, dtype)  # [B, F, D]
                        if 'traj' in mode else None
                    ),
                    'depths': (
                        batch['controls']['latents_depth'].to(device, dtype)  # [B, C, F, H, W]
                        if 'depth' in mode else None
                    ),
                    'labels': (
                        batch['controls']['latents_label'].to(device, dtype)  # [B, C, F, H, W]
                        if 'label' in mode else None
                    ),
                },
                'num_frames': batch['num_frames'],
                'num_views': batch['num_views'],
                'height': height,
                'width': width,
                'guidance_scale': eval_config.guidance_scale,
                'use_dynamic_cfg': eval_config.use_dynamic_cfg,
            }

            # 4. forward pipeline
            generator = torch.Generator().manual_seed(eval_config.seed)
            with torch.no_grad():
                videos = pipe(**pipeline_args, generator=generator, output_type='pil').frames

            # 5. save results
            output_queue.put(
                {
                    'gifs_folder': gifs_folder,
                    'mp4s_folder': mp4s_folder,
                    'data_infos': batch['metainfos'],
                    'num_frames': batch['num_frames'],
                    'num_views': batch['num_views'],
                    'image_size': (width, height),
                    'videos': videos,
                }
            )

        except Exception:
            CONSOLE.log('[bold red]-------------------------')
            CONSOLE.log(f"[bold red]An exception occurred while processing sample {batch['metainfos']['sample_name']}")
            traceback.print_exc()
            CONSOLE.log('[bold red]-------------------------')

    # end
    output_queue.put(None)
    save_thread.shutdown(wait=True)
    save_future.result()

    # combine distributed results
    if rank == 0:
        print(
            f"Completed preprocessing latents and embeddings. Temporary files from all ranks saved to `{tmp_folder.as_posix()}`"
        )

        # Move files from each rank to common directory
        for subfolder, extension in [
            ('mp4s', 'mp4'),
            ('gifs', 'gif'),
        ]:
            tmp_subfolder = tmp_folder.joinpath(subfolder)
            pattern = f"*.{extension}"

            for file in tqdm(tmp_subfolder.rglob(pattern)):
                file.replace(output_folder / file.name)

        # Remove temporary directories
        def rmdir_recursive(dir: pathlib.Path) -> None:
            for child in dir.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    rmdir_recursive(child)
            dir.rmdir()

        rmdir_recursive(tmp_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument(
        "--base_config",
        type=str,
        default='./config/base_eval.yaml',
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
    args = parser.parse_args()
    multiprocessing.set_start_method('spawn', force=True)

    CONSOLE.log(f'Loading configs from {args.config} ...')
    # TODO: fix this!
    base_config = OmegaConf.load(args.base_config)
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(base_config, config)
    config.dataset = OmegaConf.merge(config.dataset, config['dataset'][args.dataset_type])
    args_config = OmegaConf.create(vars(args))
    args_config = OmegaConf.masked_copy(args_config, [k for k, v in args_config.items() if v is not None])
    config = OmegaConf.merge(config, config.runtime, args_config)

    main(config)
