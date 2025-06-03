import decord
import os
import numpy as np
from tqdm import tqdm
from diffusers.utils import export_to_video
from PIL import Image
from torchvision import transforms


video_path = "assets/videos/occupancy"
save_path = "assets/videos/new_occupancy"
os.makedirs(save_path, exist_ok=True)

video_files = list(sorted(os.listdir(video_path)))
for file in tqdm(video_files):
    vr = decord.VideoReader(os.path.join(video_path, file))
    video = vr.get_batch(range(len(vr))).asnumpy()  # [N, H, W, 3]

    H, W = video.shape[1:3]
    new_H = int(H * 0.8)
    new_W = int(W * 0.8)

    T = transforms.CenterCrop((new_H, new_W))

    new_video = []
    for frame in tqdm(video, leave=False):
        _frame = Image.fromarray(frame.astype(np.uint8))
        _frame = T(_frame)
        new_video.append(_frame)

    export_to_video(new_video, os.path.join(save_path, file), fps=8)