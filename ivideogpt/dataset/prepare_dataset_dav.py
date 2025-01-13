import sys
sys.path.append('.')
sys.path.append('thirdparty/DepthAnyVideo')

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL.ImageOps import exif_transpose

import argparse
import logging
import os
import random

from easydict import EasyDict
import numpy as np
import torch
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler

from thirdparty.DepthAnyVideo.dav.utils import img_utils
from thirdparty.DepthAnyVideo.dav.pipelines import DAVPipeline
from thirdparty.DepthAnyVideo.dav.models import UNetSpatioTemporalRopeConditionModel


def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_infer(data_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    traj_data = np.load(data_path)
    image = traj_data['image']
    num_frames = len(image)

    image = img_utils.imresize_max(image, cfg.max_resolution)
    image = img_utils.imcrop_multi(image)
    image_tensor = np.ascontiguousarray(
        [_img.transpose(2, 0, 1) / 255.0 for _img in image]
    )
    image_tensor = torch.from_numpy(image_tensor).to(device)

    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16):
        pipe_out = pipe(
            image_tensor,
            num_frames=num_frames,
            num_overlap_frames=cfg.num_overlap_frames,
            num_interp_frames=cfg.num_interp_frames,
            decode_chunk_size=cfg.decode_chunk_size,
            num_inference_steps=cfg.denoise_steps,
        )

    disparity = pipe_out.disparity
    disparity_colored = pipe_out.disparity_colored
    image = pipe_out.image
    # (N, H, 2 * W, 3)
    merged = np.concatenate(
        [
            image,
            disparity_colored,
        ],
        axis=2,
    )

    for i in range(num_frames):
        img_utils.write_image(
            os.path.join(output_dir, f"frame_{i:04d}.png"),
            disparity_colored[i],
        )
        np.save(
            os.path.join(output_dir, f"frame_{i:04d}.npy"),
            disparity[i],
        )
    from PIL import Image
    pil_images = [Image.fromarray(disparity.astype(np.uint8)) for disparity in disparity_colored]
    pil_images[0].save(os.path.join(output_dir, '_depth_map.gif'), save_all=True, append_images=pil_images[1:], duration=100, loop=0)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run video depth estimation using Depth Any Video."
    )

    parser.add_argument(
        "--model_base",
        type=str,
        default="hhyangcs/depth-any-video",
        help="Checkpoint path or hub name.",
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=3,
        help="Denoising steps, 1-3 steps work fine.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32,
        help="Number of frames to infer per forward",
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=16,
        help="Number of frames to decode per forward",
    )
    parser.add_argument(
        "--num_interp_frames",
        type=int,
        default=16,
        help="Number of frames for inpaint inference",
    )
    parser.add_argument(
        "--num_overlap_frames",
        type=int,
        default=6,
        help="Number of frames to overlap between windows",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=1024,  # decrease for faster inference and lower memory usage
        help="Maximum resolution for inference.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()
    cfg = EasyDict(vars(args))

    if cfg.seed is None:
        import time

        cfg.seed = int(time.time())
    seed_all(cfg.seed)

    device_type = "cuda"
    device = torch.device(device_type)

    vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.model_base, subfolder="vae")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model_base, subfolder="scheduler"
    )
    unet = UNetSpatioTemporalRopeConditionModel.from_pretrained(
        cfg.model_base, subfolder="unet"
    )
    unet_interp = UNetSpatioTemporalRopeConditionModel.from_pretrained(
        cfg.model_base, subfolder="unet_interp"
    )
    pipe = DAVPipeline(
        vae=vae,
        unet=unet,
        unet_interp=unet_interp,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)

    data_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/data/robonet_preprocessed'
    save_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/data/occ_robonet_dav/point'
    splits = ['train', 'test']

    for split in tqdm(splits, "Processing split"):
        split_dir = os.path.join(data_dir, split)
        traj_files = os.listdir(split_dir)
        for traj_file in (
            pbar := tqdm(traj_files)
        ):
            traj_id = traj_file.removesuffix('.npz')
            save_folder = os.path.join(save_dir, split, traj_id)
            pbar.set_description(f"Processing {traj_id}")
        
            if os.path.exists(save_folder):
                tqdm.write(f"Skipped {split}-{traj_id}")
                continue

            try:
                # Call the function with default parameters
                data_path = os.path.join(split_dir, traj_file)
                run_infer(data_path, save_folder)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    exit(1)
                continue
