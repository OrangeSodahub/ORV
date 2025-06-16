import argparse
import os
import sys
import torch
from tqdm import tqdm

from diffusers.utils.export_utils import export_to_video
from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from dataset.dataset import DemoRobotDataset
from models.cogvideox_control import CogVideoXTransformer3DModelTraj, CogVideoXImageToVideoPipelineTraj
from pipelines.utils import CONSOLE


def generate_video(
    data_root: str,
    episode_id: str,
    ref_num: int,
    prompt: str,
    slice_frame: bool,
    pretrained_model_path: str,
    model_path: str,
    output_folder: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    mode: str = 'traj-image',  # i2v: image to video, i2vo: original CogVideoX-5b-I2V
    fps: int = 10,
    seed: int = 42,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - tracking_path (str): The path of the tracking maps to be used.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """

    # 1. Setup Dataloader in test mode
    dataset = DemoRobotDataset(
        data_root=data_root,
        split='test',
        use_cond=False,
        filter_by_cond=True,
        ori_size=[320, 480],
        ref_num=ref_num,
        load_tensor=False,
        use_3dvae=True,
        slice_frame=slice_frame,
        test_mode=True,
    )

    # 2. Setup inference pipeline.
    # add device_map='balanced' in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'model_index.json' in os.listdir(model_path):
        pipe = CogVideoXImageToVideoPipelineTraj.from_pretrained(model_path, torch_dtype=dtype)
    else:
        CONSOLE.log(f'Not found model_index.json, will load transformer only ...')
        transformer = CogVideoXTransformer3DModelTraj.from_pretrained(
            model_path,
            subfolder='transformer',
            torch_dtype=dtype,
        )
        pipe = CogVideoXImageToVideoPipelineTraj.from_pretrained(
            pretrained_model_path,
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
    # TODO: fix this term when using CogVideoX's settings!!!!!
    pipe.vae._internal_dict = FrozenDict(**(pipe.vae.config | dict(invert_scale_latents=False)))

    # 3. Load data
    index = dataset.fetch_episode(episode_id=str(int(episode_id)))
    for i in tqdm(index):

        data = dataset.fetch_index(index=i)
        CONSOLE.log(f'Loaded episode {episode_id=} {i=}')

        image = data['pil_image'][0][0]  # n_view -> n_ref_frame
        width, height = image.size  # TODO: support ref_num > 1

        sample_name = data['metainfo']['sample_name']

        pipeline_args = {
            'image': image,
            'prompt': data['prompt'],
            'negative_prompt': 'The video is not of a high quality, it has a low resolution. Strange body and strange trajectory. Distortion.',
            'controls_or_guidances': {
                'actions': (
                    data['actions'].unsqueeze(0).to(device, dtype)  # [B, F, D]
                    if 'traj' in mode else None
                ),
                'depths': (
                    data['latents_depth'].transpose(0, 1).unsqueeze(0).to(device, dtype)  # [B, C, F, H, W]
                    if 'depth' in mode else None
                ),
            },
            'num_frames': data['actions'].size(0) + 1,
            'height': height,
            'width': width,
            'guidance_scale': guidance_scale,
            'use_dynamic_cfg': True if 'text' in mode else False,
        }

        # 4. Forward.
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            video = pipe(**pipeline_args, generator=generator, output_type='pil').frames[0]

        # 5. Save results.
        os.makedirs(output_folder, exist_ok=True)
        output_mp4_path = os.path.join(output_folder, f'demo_{sample_name}.mp4')
        export_to_video(video, output_mp4_path, fps=fps)

        output_gif_path = os.path.join(output_folder, f'demo_{sample_name}.gif')
        video[0].save(output_gif_path, save_all=True, append_images=video[1:], duration=100, loop=0)
        CONSOLE.log(f'Exported GIF to [bold yellow]{output_gif_path}[/]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a video from images and actions.')
    parser.add_argument(
        '--prompt', type=str, required=False, default='', help='The description of the video to be generated'
    )
    parser.add_argument('--data_root', type=str, required=False, default='./data/demo')
    parser.add_argument('--episode_id', type=str, required=False, default='00000')
    parser.add_argument('--ref_num', type=int, required=False, default=1)
    parser.add_argument(
        '--slice_frame',
        action='store_true',
        help='If True, will load segments with fixed length'
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default='THUDM/CogVideoX-2b',
        help='The path of the pre-trained model to be used',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./outputs_old/cirasim_bridge_traj-image_480-320_finetune_2b_30k/checkpoint',
        # default='./outputs/cirasim_bridge_traj-image_480-320_finetune_2b_30k/checkpoint',
        help='The path of the pre-trained model to be used',
    )
    parser.add_argument(
        '--output_folder', type=str, default='./outputs/demos', help='The path where the generated video will be saved'
    )
    parser.add_argument(
        '--guidance_scale', type=float, default=1.0, help='The scale for classifier-free guidance'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=50, help='Number of steps for the inference process'
    )
    parser.add_argument('--num_videos_per_prompt', type=int, default=1, help='Number of videos to generate per prompt')
    parser.add_argument(
        '--mode', type=str, default='traj-image', help="The type of video generation (e.g., 'traj-image', 'text-image', 'traj-image-depth')"
    )
    parser.add_argument(
        '--dtype', type=str, default='bfloat16', help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument('--seed', type=int, default=42, help='The seed for reproducibility')

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16
    generate_video(
        data_root=args.data_root,
        episode_id=args.episode_id,
        ref_num=args.ref_num,
        prompt=args.prompt,
        slice_frame=args.slice_frame,
        pretrained_model_path=args.pretrained_model_path,
        model_path=args.model_path,
        output_folder=args.output_folder,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        mode=args.mode,
        dtype=dtype,
        seed=args.seed,
    )