import sys
sys.path.append('.')
sys.path.append('thirdparty/MoGe')

import torch
import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

from thirdparty.MoGe.moge.utils.vis import colorize_depth
from thirdparty.MoGe.moge.model import MoGeModel


if __name__ == '__main__':

    data_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/data/robonet_preprocessed'
    save_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/data/occ_robonet_moge'
    splits = ['train', 'test']

    device = torch.device("cuda")
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)                             

    for split in tqdm(splits, "Processing split", leave=False):
        split_dir = os.path.join(data_dir, split)
        traj_files = os.listdir(split_dir)
        for traj_file in (
            pbar := tqdm(traj_files, leave=False)
        ):
            traj_id = traj_file.removesuffix('.npz')
            traj_data = np.load(os.path.join(split_dir, traj_file))
            images = traj_data['image']
            save_folder = os.path.join(save_dir, split, traj_id)
            pbar.set_description(f"Processing {traj_id}")
        
            if os.path.exists(save_folder) and len(os.listdir(save_folder)) == images.shape[0] + 2:
                tqdm.write(f"Skipped {split}-{traj_id}")
                continue

            try:
                os.makedirs(save_folder)
                depth_maps = []
                seq_data = {'points': [], 'depths': [], 'masks': [], 'intrins': []}
                for i, image in enumerate(tqdm(images, leave=False)):
                    input_image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
                    # `output` has keys "points", "depth", "mask" and "intrinsics",
                    output = model.infer(input_image)
                    depth_map = Image.fromarray(colorize_depth(output['depth'].cpu().numpy()).astype(np.uint8))
                    depth_map.save(os.path.join(save_folder, f'{i:04d}.png'))
                    depth_maps.append(depth_map)
                    seq_data['points'].append(output['points'])
                    seq_data['depths'].append(output['depth'])
                    seq_data['masks'].append(output['mask'])
                    seq_data['intrins'].append(output['intrinsics'])
                # save depth gif
                depth_maps[0].save(os.path.join(save_folder, 'traj_depth.gif'), save_all=True, append_images=depth_maps[1:], duration=100, loop=0)
                # save output
                seq_data = {k: torch.stack(v).cpu().numpy() for k, v in seq_data.items()}
                with open(os.path.join(save_folder, 'traj_data.pkl'), 'wb') as f:
                    pickle.dump(seq_data, f)
            except Exception as e:
                print(e)
                continue
