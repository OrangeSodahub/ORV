import sys
sys.path.append('.')
sys.path.append('thirdparty/DepthAnythingV2')

import os
import numpy as np
import matplotlib
from tqdm import tqdm
from PIL import Image

import argparse
import logging
import os
import random
import cv2

import numpy as np
import torch

from thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from thirdparty.DepthAnyVideo.dav.utils import img_utils


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

    depth_colored = []

    for i, _image in enumerate(image):
        _depth = depth_anything.infer_image(_image)
        
        # _depth_colored = (_depth - _depth.min()) / (_depth.max() - _depth.min()) * 255.0
        # _depth_colored = _depth.astype(np.uint8)
        _depth_colored = (cmap(_depth / _depth.max())[:, :, :3] * 255).astype(np.uint8)

        img_utils.write_image(
            os.path.join(output_dir, f"frame_{i:04d}.png"),
            _depth_colored,
        )
        np.save(
            os.path.join(output_dir, f"frame_{i:04d}.npy"),
            _depth,
        )

        depth_colored.append(_depth_colored)

    from PIL import Image
    pil_images = [Image.fromarray(_depth_colored.astype(np.uint8)) for _depth_colored in depth_colored]
    pil_images[0].save(os.path.join(output_dir, '_depth_map.gif'), save_all=True, append_images=pil_images[1:], duration=100, loop=0)


if __name__ == '__main__':
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(
        os.path.join('/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/thirdparty/DepthAnythingV2', f'checkpoints/depth_anything_v2_{encoder}.pth'), map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    data_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/data/robonet_preprocessed'
    save_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/data/occ_robonet_dav2/point'
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
