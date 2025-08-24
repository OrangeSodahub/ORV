import decord
import os
import numpy as np
import fnmatch
from tqdm import tqdm
from diffusers.utils import export_to_video
from PIL import Image


video_path = "assets/videos/occupancy"
save_path = "assets/videos/occupancy/gifs"
os.makedirs(save_path, exist_ok=True)

video_files = list(sorted(fnmatch.filter(os.listdir(video_path), '*.mp4')))
for file in tqdm(video_files):
    vr = decord.VideoReader(os.path.join(video_path, file))
    video = vr.get_batch(range(len(vr))).asnumpy()  # [N, H, W, 3]

    video = [Image.fromarray(frame.astype(np.uint8)) for frame in video]
    # video[0].save(
    #     os.path.join(
    #         save_path, file.replace('.mp4', '.gif')),
    #     save_all=True,
    #     append_images=video[1:],
    #     duration=100,
    #     loop=0
    # )
    print(file, len(video))