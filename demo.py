import os
import sys
import argparse
from PIL import Image
project_root = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.append(os.path.join(project_root, "submodules/MoGe"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except:
    print("Warning: MoGe not found, motion transfer will not be applied")
    
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
from diffusers.utils import load_image, load_video

from models.pipelines import DiffusionAsShaderPipeline, FirstFrameRepainter, CameraMotionGenerator, ObjectMotionGenerator
from submodules.MoGe.moge.model import MoGeModel


def load_media(media_path, max_frames=49, transform=None):
    """Load video or image frames and convert to tensor
    
    Args:
        media_path (str): Path to video or image file
        max_frames (int): Maximum number of frames to load
        transform (callable): Transform to apply to frames
        
    Returns:
        Tuple[torch.Tensor, float]: Video tensor [T,C,H,W] and FPS
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((480, 720)),
            transforms.ToTensor()
        ])
    
    # Determine if input is video or image based on extension
    ext = os.path.splitext(media_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov']
    
    if is_video:
        frames = load_video(media_path)
        fps = len(frames) / VideoFileClip(media_path).duration
    else:
        # Handle image as single frame
        image = load_image(media_path)
        frames = [image]
        fps = 8  # Default fps for images
    
    # Ensure we have exactly max_frames
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    elif len(frames) < max_frames:
        last_frame = frames[-1]
        while len(frames) < max_frames:
            frames.append(last_frame.copy())
            
    # Convert frames to tensor
    video_tensor = torch.stack([transform(frame) for frame in frames])
    
    return video_tensor, fps, is_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None, help='Path to input video/image')
    parser.add_argument('--prompt', type=str, required=True, help='Repaint prompt')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--depth_path', type=str, default=None, help='Path to depth image')
    parser.add_argument('--tracking_path', type=str, default=None, help='Path to tracking video, if provided, camera motion and object manipulation will not be applied')
    parser.add_argument('--repaint', type=str, default=None, 
                       help='Path to repainted image, or "true" to perform repainting, if not provided use original frame')
    parser.add_argument('--camera_motion', type=str, default=None, 
                    help='Camera motion mode: "trans <dx> <dy> <dz>" or "rot <axis> <angle>" or "spiral <radius>"')
    parser.add_argument('--object_motion', type=str, default=None, help='Object motion mode: up/down/left/right')
    parser.add_argument('--object_mask', type=str, default=None, help='Path to object mask image (binary image)')
    parser.add_argument('--tracking_method', type=str, default="spatracker", 
                        help='default tracking method for image input: moge/spatracker, if \'moge\' method will extract first frame for video input')
    args = parser.parse_args()
    
    # Load input video/image
    video_tensor, fps, is_video = load_media(args.input_path)
    if not is_video:
        args.tracking_method = "moge"
        print("Image input detected, using MoGe for tracking video generation.")

    # Initialize pipeline
    das = DiffusionAsShaderPipeline(gpu_id=args.gpu, output_dir=args.output_dir)
    if args.tracking_method == "moge" and args.tracking_path is None:
        moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(das.device)
    
    # Repaint first frame if requested
    repaint_img_tensor = None
    if args.repaint:
        if args.repaint.lower() == "true":
            repainter = FirstFrameRepainter(gpu_id=args.gpu, output_dir=args.output_dir)
            repaint_img_tensor = repainter.repaint(
                video_tensor[0], 
                prompt=args.prompt,
                depth_path=args.depth_path
            )
        else:
            repaint_img_tensor, _, _ = load_media(args.repaint)
            repaint_img_tensor = repaint_img_tensor[0]  # Take first frame

    # Generate tracking if not provided
    tracking_tensor = None
    pred_tracks = None
    cam_motion = CameraMotionGenerator(args.camera_motion)

    if args.tracking_path:
        tracking_tensor, _, _ = load_media(args.tracking_path)
        
    elif args.tracking_method == "moge":
        # Use the first frame from previously loaded video_tensor
        infer_result = moge.infer(video_tensor[0].to(das.device))  # [C, H, W] in range [0,1]
        H, W = infer_result["points"].shape[0:2]
        pred_tracks = infer_result["points"].unsqueeze(0).repeat(49, 1, 1, 1) #[T, H, W, 3]
        cam_motion.set_intr(infer_result["intrinsics"])

        # Apply object motion if specified
        if args.object_motion:
            if args.object_mask is None:
                raise ValueError("Object motion specified but no mask provided. Please provide a mask image with --object_mask")
                
            # Load mask image
            mask_image = Image.open(args.object_mask).convert('L')  # Convert to grayscale
            mask_image = transforms.Resize((480, 720))(mask_image)  # Resize to match video size
            # Convert to binary mask
            mask = torch.from_numpy(np.array(mask_image) > 127)  # Threshold at 127
            
            motion_generator = ObjectMotionGenerator(device=das.device)

            pred_tracks = motion_generator.apply_motion(
                pred_tracks=pred_tracks,
                mask=mask,
                motion_type=args.object_motion,
                distance=50,
                num_frames=49,
                tracking_method="moge"
            )
            print("Object motion applied")

        # Apply camera motion if specified
        if args.camera_motion:
            poses = cam_motion.get_default_motion() # shape: [49, 4, 4]
            print("Camera motion applied")
        else:
            # no poses
            poses = torch.eye(4).unsqueeze(0).repeat(49, 1, 1)
        # change pred_tracks into screen coordinate
        pred_tracks_flatten = pred_tracks.reshape(video_tensor.shape[0], H*W, 3)
        pred_tracks = cam_motion.w2s(pred_tracks_flatten, poses).reshape([video_tensor.shape[0], H, W, 3]) # [T, H, W, 3]
        _, tracking_tensor = das.visualize_tracking_moge(
            pred_tracks.cpu().numpy(), 
            infer_result["mask"].cpu().numpy()
        )
        print('export tracking video via MoGe.')

    else:
        # Generate tracking points
        pred_tracks, pred_visibility, T_Firsts = das.generate_tracking_spatracker(video_tensor)

        # Apply camera motion if specified
        if args.camera_motion:
            poses = cam_motion.get_default_motion() # shape: [49, 4, 4]
            pred_tracks = cam_motion.apply_motion_on_pts(pred_tracks, poses)
            print("Camera motion applied")
        
        # Apply object motion if specified
        if args.object_motion:
            if args.object_mask is None:
                raise ValueError("Object motion specified but no mask provided. Please provide a mask image with --object_mask")
                
            # Load mask image
            mask_image = Image.open(args.object_mask).convert('L')  # Convert to grayscale
            mask_image = transforms.Resize((480, 720))(mask_image)  # Resize to match video size
            # Convert to binary mask
            mask = torch.from_numpy(np.array(mask_image) > 127)  # Threshold at 127
            
            motion_generator = ObjectMotionGenerator(device=das.device)
            
            pred_tracks = motion_generator.apply_motion(
                pred_tracks=pred_tracks.squeeze(),
                mask=mask,
                motion_type=args.object_motion,
                distance=50,
                num_frames=49,
                tracking_method="spatracker"
            ).unsqueeze(0)
            print(f"Object motion '{args.object_motion}' applied using mask from {args.object_mask}")
    
        # Generate tracking tensor from modified tracks
        _, tracking_tensor = das.visualize_tracking_spatracker(video_tensor, pred_tracks, pred_visibility, T_Firsts)
    
    das.apply_tracking(
        video_tensor=video_tensor,
        fps=8,
        tracking_tensor=tracking_tensor,
        img_cond_tensor=repaint_img_tensor,
        prompt=args.prompt,
        checkpoint_path=args.checkpoint_path
    )
