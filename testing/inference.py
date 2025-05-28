import argparse
from typing import Literal
import os
import sys

import torch
from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
)

from diffusers.utils import export_to_video, load_image, load_video

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from models.cogvideox_tracking import CogVideoXImageToVideoPipelineTracking, CogVideoXPipelineTracking, CogVideoXVideoToVideoPipelineTracking
from models.cogvideox_tracking import CogVideoXTransformer3DModelTracking

def generate_video(
    prompt: str,
    model_path: str,
    tracking_path: str = None,
    tracking_video: torch.Tensor = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v"],  # i2v: image to video, i2vo: original CogVideoX-5b-I2V
    fps: int = 24,
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

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transformer = CogVideoXTransformer3DModelTracking.from_pretrained(
    #     model_path,
    #     subfolder="transformer",
    #     torch_dtype=dtype
    # )

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipelineTracking.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
        height, width = image.height, image.width
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
        height, width = image.height, image.width

    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    for param in pipe.transformer.parameters():
        param.requires_grad = False

    pipe.transformer.gradient_checkpointing = False

    # Convert tracking maps from list of PIL Images to tensor
    if tracking_path is not None:
        tracking_maps = load_video(tracking_path)
        # Convert list of PIL Images to tensor [T, C, H, W]
        tracking_maps = torch.stack([
            torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0 
            for frame in tracking_maps
        ])
        tracking_maps = tracking_maps.to(device=device, dtype=dtype)
        tracking_first_frame = tracking_maps[0:1]  # Get first frame as [1, C, H, W]
        height, width = tracking_first_frame.shape[2], tracking_first_frame.shape[3]
    elif tracking_video is not None:
        tracking_maps = tracking_video.float() / 255.0 # [T, C, H, W]
        tracking_maps = tracking_maps.to(device=device, dtype=dtype)
        tracking_first_frame = tracking_maps[0:1]  # Get first frame as [1, C, H, W]
        height, width = tracking_first_frame.shape[2], tracking_first_frame.shape[3]
    else:
        tracking_maps = None
        tracking_first_frame = None

    # 2. Set Scheduler.
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.to(device, dtype=dtype)
    # pipe.enable_sequential_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    pipe.transformer.gradient_checkpointing = False
    
    if tracking_maps is not None and generate_type == "i2v":
        print("Encoding tracking maps")
        tracking_maps = tracking_maps.unsqueeze(0) # [B, T, C, H, W]
        tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        with torch.no_grad():
            tracking_latent_dist = pipe.vae.encode(tracking_maps).latent_dist
            tracking_maps = tracking_latent_dist.sample() * pipe.vae.config.scaling_factor
            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    else:
        tracking_maps = None
        tracking_first_frame = None

    # 4. Generate the video frames based on the prompt.
    if generate_type == "i2v":
        with torch.no_grad():
            video_generate = pipe(
                prompt=prompt,
                negative_prompt="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                image=image,
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=49,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
                tracking_maps=tracking_maps,
                tracking_image=tracking_first_frame,
                height=height,
                width=width,
            ).frames[0]
    else:
        with torch.no_grad():
            video_generate = pipe(
                prompt=prompt,
                negative_prompt="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                image=image,
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=49,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
            ).frames[0]
    # 5. Export the generated frames to a video file. fps must be 8 for original video.
    output_path = output_path if output_path else f"{generate_type}_img[{os.path.splitext(os.path.basename(image_or_video_path))[0]}]_txt[{prompt}].mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--tracking_path", type=str, default=None, help="The path of the tracking maps to be used")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        tracking_path=args.tracking_path,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
    )