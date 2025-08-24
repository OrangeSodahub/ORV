#!/usr/bin/env python3
import os
import subprocess
import math

input_dir = "."
rgb_dir = "rgbs"
output_dir = "concat"

os.makedirs(output_dir, exist_ok=True)

video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

for vf in video_files:
    input_path = os.path.join(input_dir, vf)
    rgb_path = os.path.join(rgb_dir, vf)
    output_path = os.path.join(output_dir, vf)

    if not os.path.exists(rgb_path):
        print(f"Skipping {vf}, no corresponding file in {rgb_dir}")
        continue

    def get_dimensions(video_path):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path
        ]
        out = subprocess.check_output(cmd).decode().strip()
        w, h = map(int, out.split(","))
        return w, h

    w1, h1 = get_dimensions(input_path)
    w2, h2 = get_dimensions(rgb_path)
    target_height = max(h1, h2)

    # scaled right video height 80% of target
    scaled_height = math.floor(target_height * 0.8)

    # ffmpeg filter: scale video, then pad to center on canvas
    filter_complex = (
        f"[0:v]scale=-2:{target_height}[v0]; "
        f"[1:v]scale=-2:{scaled_height},pad=ceil(iw*1.25):{target_height}:(ow-iw)/2:(oh-ih)/2:color=white[v1]; "
        f"[v0][v1]hstack=inputs=2"
    )

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-i", rgb_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-y",
        output_path
    ]

    subprocess.run(cmd)

print("All videos concatenated with centered white border on right side!")