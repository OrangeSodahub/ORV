# %%

#-------- import the base packages -------------
import sys
import os

import torch
import torch.nn.functional as F
from base64 import b64encode
import numpy as np
from PIL import Image
import cv2
import argparse
from moviepy.editor import ImageSequenceClip
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.cuda

#-------- import spatialtracker -------------
from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer, read_video_from_path

#-------- import Depth Estimator -------------
from PIL import Image
from image_gen_aux import DepthPreprocessor

# set the arguments
parser = argparse.ArgumentParser()
# add the video and segmentation
parser.add_argument('--root', type=str, default='./assets', help='path to the video folder')
# set the gpu
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
# set the downsample factor
parser.add_argument('--downsample', type=float, default=0.8, help='downsample factor')
parser.add_argument('--grid_size', type=int, default=70, help='grid size')
# set the results outdir
parser.add_argument('--outdir', type=str, default='./vis_results', help='output directory')
# set the fps
parser.add_argument('--fps', type=float, default=1, help='fps')
# draw the track length
parser.add_argument('--len_track', type=int, default=0, help='len_track')
parser.add_argument('--output_fps', type=int, default=24, help='Output video fps and total frames')
# crop the video
parser.add_argument('--crop', action='store_true', help='whether to crop the video')
parser.add_argument('--crop_factor', type=float, default=1, help='whether to crop the video')
# backward tracking
parser.add_argument('--backward', action='store_true', help='whether to backward the tracking')
# if visualize the support points
parser.add_argument('--vis_support', action='store_true', help='whether to visualize the support points')
# query frame
parser.add_argument('--query_frame', type=int, default=0, help='query frame')
# set the visualized point size
parser.add_argument('--point_size', type=int, default=10, help='point size')
# take the RGBD as input
parser.add_argument('--rgbd', action='store_true', help='whether to take the RGBD as input')

args = parser.parse_args()

# set input
root_dir = args.root
outdir = args.outdir
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
# set the paras
grid_size = args.grid_size
downsample = args.downsample
# set the gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

def get_available_gpus():
    return list(range(torch.cuda.device_count()))

from accelerate import Accelerator

def process_video(args, model, MonoDEst_M, vid_name, root_dir, outdir):
    device = next(model.parameters()).device
    vid_dir = os.path.join(root_dir, vid_name)
    vid_name_without_ext = os.path.splitext(vid_name)[0]

    # Read original video fps
    cap = cv2.VideoCapture(vid_dir)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Set fps_vis to original fps
    fps_vis = original_fps

    # Read video
    try:
        video = read_video_from_path(vid_dir)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    except:
        print(f"Error reading video {vid_name}")
        return
    
    transform = transforms.Compose([
    transforms.CenterCrop((int(384*args.crop_factor),
                            int(576*args.crop_factor))),  
    ])
    _, T, _, H, W = video.shape
    segm_mask = np.ones((H, W), dtype=np.uint8)
    print(f"Processing {vid_name}. Computing tracks in whole image.")
    if args.crop:
        video = transform(video)
        segm_mask = transform(torch.from_numpy(segm_mask[None, None]))[0,0].numpy()
    
    _, _, _, H, W = video.shape
    target_h, target_w = 480, 720

    # Calculate scaling factor
    scale_h = target_h / H
    scale_w = target_w / W
    scale = max(scale_h, scale_w)  # Choose the larger scaling factor to ensure coverage of the target size

    # Scale the video
    if scale != 1.0:
        video = F.interpolate(video[0], scale_factor=scale, mode='bilinear', align_corners=True)[None]

    # Center crop to target size
    _, _, _, new_H, new_W = video.shape
    start_h = (new_H - target_h) // 2
    start_w = (new_W - target_w) // 2
    video = video[:, :, :, start_h:start_h+target_h, start_w:start_w+target_w]

    print(f"Video scaled and cropped to {target_h}x{target_w}")

    # Update segmentation mask
    if 'segm_mask' in locals():
        segm_mask = cv2.resize(segm_mask, (new_W, new_H))
        segm_mask = segm_mask[start_h:start_h+target_h, start_w:start_w+target_w]

    video = video.to(device)

    if not args.rgbd:
        video_depths = []
        for i in range(video.shape[1]):
            frame = (video[0, i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            depth = depth_preprocessor(Image.fromarray(frame))[0]
            depth_tensor = transforms.ToTensor()(depth)  # [1, H, W]
            video_depths.append(depth_tensor)
        depths = torch.stack(video_depths, dim=0).cuda()  # [T, 1, H, W]
        print("Depth maps shape:", depths.shape)
    else:
        depths = None

    # Use accelerator.unwrap_model() to get original model
    unwrapped_model = accelerator.unwrap_model(model)

    pred_tracks, pred_visibility, T_Firsts = (
        unwrapped_model(video, video_depth=depths,
              grid_size=args.grid_size, backward_tracking=args.backward,
              depth_predictor=None, grid_query_frame=args.query_frame,
              segm_mask=torch.from_numpy(segm_mask)[None, None].to(device), wind_length=12,
              progressive_tracking=False)  # Enable progressive tracking
    )

    vis = Visualizer(save_dir=outdir, grayscale=False,
                 fps=args.output_fps, pad_value=0, linewidth=args.point_size,
                 tracks_leave_trace=args.len_track)
    msk_query = (T_Firsts == args.query_frame)
    pred_tracks = pred_tracks[:,:,msk_query.squeeze()]
    pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
    video_vis = vis.visualize(video=video, tracks=pred_tracks,
                              visibility=pred_visibility,
                              filename=f"{vid_name_without_ext}")

    tracks_vis = pred_tracks.detach().cpu().numpy()
    visbility_vis = pred_visibility.detach().cpu().numpy()
    combined_data = {"tracks": tracks_vis, "visibility": visbility_vis}
    if not os.path.exists(os.path.join(outdir, "tracks")):
        os.makedirs(os.path.join(outdir, "tracks"))
    np.save(os.path.join(outdir, "tracks", f'{vid_name_without_ext}_tracks.npy'), combined_data)

    print(f"Processed {vid_name}. Results saved in {outdir}")

if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device

    args = parser.parse_args()

    # Init model
    S_lenth = 12
    model = SpaTrackerPredictor(
        checkpoint=os.path.join('./checkpoints/spaT_final.pth'),
        interp_shape=(384, 576),
        seq_length=S_lenth
    )
    
    if not args.rgbd:
        depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        depth_preprocessor.to(device)
    else:
        depth_preprocessor = None

    # Use Accelerator to wrap models
    model, depth_preprocessor = accelerator.prepare(model, depth_preprocessor)

    # Get all video files
    video_files = [f for f in os.listdir(args.root) if f.endswith('.mp4')]
    
    # Split video files based on process count
    num_processes = accelerator.num_processes
    videos_per_process = len(video_files) // num_processes
    process_id = accelerator.process_index
    start_idx = process_id * videos_per_process
    end_idx = start_idx + videos_per_process if process_id < num_processes - 1 else len(video_files)
    
    # Each process only handles its assigned videos
    for vid_name in video_files[start_idx:end_idx]:
        accelerator.print(f"Process {process_id} starts processing {vid_name}")
        process_video(args, model, depth_preprocessor, vid_name, args.root, args.outdir)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        print("All videos processed")
