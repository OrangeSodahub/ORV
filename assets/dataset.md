## Dataset Format

### Prompt Dataset Requirements

Create a `prompt.txt` file, which should contain prompts separated by lines. Please note that the prompts must be in English, and it is recommended to use the [prompt refinement script](https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py) for better prompts. Alternatively, you can use [CogVideo-caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption) for data annotation:

```
A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.
A black and white animated sequence on a ship’s deck features a bulldog character, named Bully Bulldoger, showcasing exaggerated facial expressions and body language...
...
```

### Video Dataset Requirements

The framework supports resolutions and frame counts that meet the following conditions:

- **Supported Resolutions (Width * Height)**:
    - We only support `720 * 480`.

- **Supported Frame Counts (Frames)**:
    - For our ckpt, the frame count must be 49.
    - If you want to train on your own ckpt, the frame count must be `` or `4 * k + 1` (example: 16, 32, 49, 81)

It is recommended to place all videos in a single folder.

Next, create a `videos.txt` file. The `videos.txt` file should contain the video file paths, separated by lines. Please note that the paths must be relative to the `--data_root` directory. The format is as follows:

```
videos/00000.mp4
videos/00001.mp4
...
```

Next, create a `trackings.txt` file. The `trackings.txt` file should contain the tracking file paths, and also separated by lines.


#### You need to use `accelerate_tracking.py` or `batch_tracking.py` to generate the tracking video as follows. Our bash script `scripts/generate_TrackingVideo.sh` is also provided to generate the tracking video. Please provide your dataset path to the script.

```bash
# for multiple GPUs
accelerate launch accelerate_tracking.py --data_root <videos_root> --output_dir <output_dir>
# for single GPU
python batch_tracking.py --data_root <videos_root> --output_dir <output_dir>
```


The paths in `trackings.txt` must be relative to the `--data_root` directory. The format is as follows:

```
tracking/00000_tracking.mp4
tracking/00001_tracking.mp4
...
```

Next, create a `images.txt` file. The `images.txt` file should contain the image file paths, separated by lines. These images are the images extracted from the video frames. You can use [FLUX.1 Depth](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Depth-dev) to repaint it.

The paths must be relative to the `--data_root` directory. The format is as follows:

```
images/00000.png
images/00001.png
...
```


### Dataset Structure

Your dataset structure should look like this. Running the `tree` command, you should see:

```
dataset
├── prompt.txt
├── videos.txt
├── trackings.txt
├── images.txt
├── images
    ├── images/00000.png
    ├── images/00001.png
    ├── ...
├── tracking
    ├── tracking/00000_tracking.mp4
    ├── tracking/00001_tracking.mp4
    ├── ...
├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...
```


