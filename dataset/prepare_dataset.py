from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import matplotlib
import os
os.environ['EVO_PLOT_BACKEND'] = 'Agg'
matplotlib.use('Agg')
os.environ['MPLBACKEND'] = 'Agg'
# os.environ['DISPLAY'] = 'localhost:10.0'
import queue
import random
import re
import sys
import colorsys
sys.path.append('.')
sys.path.append('thirdparty/monst3r')
sys.path.append('thirdparty/monst3r/thirdparty')

from matplotlib import pyplot as plt
from functools import partial
import torch
import numpy as np
import signal
import open3d as o3d
import trimesh
import shutil
import fnmatch
import multiprocessing
import cv2
import argparse
from torch import Tensor
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
from multiprocessing import Pool, Process, Queue, Event
from tqdm import tqdm
from copy import deepcopy
from rich.console import Console
from numpy import typing as npt
from typing import Callable, Literal, List, Optional
from scipy.spatial.transform import Rotation

CONSOLE = Console(width=120)

# monst3r
try:
    from thirdparty.monst3r.demo import *
    from thirdparty.monst3r.dust3r.utils.viz_demo import *
    from thirdparty.monst3r.dust3r.utils.image import crop_img, ImgNorm, ToTensor, rgb
except Exception as e:
    CONSOLE.log(f'[bold red]Import necessary packages form Monst3R failed! {e}')
    raise
try:
    import nksr
except Exception as e:
    CONSOLE.log(f'[bold red]Import necessary packages form NKSR failed! {e}')

# vggt
try:
    sys.path.append('thirdparty/vggt/')
    from thirdparty.vggt.vggt.models.vggt import VGGT
    from thirdparty.vggt.vggt.utils.load_fn import load_and_preprocess_images
    from thirdparty.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from thirdparty.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
except Exception as e:
    CONSOLE.log(f'[bold red]Import necessary packages form VGGT failed! {e}')

# grounded sam2
try:
    import supervision as sv
    from supervision.annotators.utils import resolve_color
    from supervision.draw.color import Color, ColorPalette
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
    from thirdparty.grounded_sam_2.sam2.build_sam import build_sam2_video_predictor, build_sam2
    from thirdparty.grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor
    from thirdparty.grounded_sam_2.sam2.sam2_video_predictor import SAM2VideoPredictor
    from thirdparty.grounded_sam_2.utils.track_utils import sample_points_from_masks
except Exception as e:
    CONSOLE.log(f'[bold red]Import necessary packages form GoundedSAM2 failed! {e}')

# gaussian render
try:
    from gs_render import render, create_full_center_coords, apply_depth_colormap
except Exception as e:
    CONSOLE.log(f'[bold red]Import necessary packages form GS failed! {e}')


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip
decord.bridge.set_bridge("torch")
from decord import VideoReader, cpu


def read_mp4(path: str) -> npt.NDArray:
    assert path.endswith('.mp4'), f'Invalid path which should be in *.mp4 format, got {path}.'
    vr = VideoReader(path, ctx=cpu(0))
    frames = vr.get_batch(range(len(vr))).numpy()  # [N, H, W, 3]
    return frames


def depths_to_points(depth_map: npt.NDArray | Tensor,
                     intrin: npt.NDArray | Tensor,
                     mask: npt.NDArray | Tensor | None = None,
                     rgb_map: npt.NDArray | Tensor | None = None) -> npt.NDArray | Tensor:
    """
    Convert depth map to 3D points using camera intrinsics.

    Args:
        depth_map (npt.NDArray): Depth map of shape (H, W) containing depth values
        intrin (npt.NDArray): Camera intrinsics matrix of shape (3, 3)

    Returns:
        npt.NDArray: 3D points of shape (N, 3) or (N, 6) where N is number of valid depth pixels
    """
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    depth = depth_map.flatten()

    rgb = None
    if rgb_map is not None:
        rgb = rgb_map.reshape(-1, 3)

    uv1 = np.vstack((u, v, np.ones_like(u)))  # [3, N]
    xyz = np.linalg.inv(intrin) @ uv1
    xyz *= depth

    points = xyz.T
    if rgb is not None:
        points = np.concatenate([points, rgb], axis=-1)

    return points  # [N, 3] or [N, 6]


def points_to_voxels(points: npt.NDArray | Tensor,
                     voxel_size: list = [0.2, 0.2, 0.2],
                     labels: npt.NDArray | Tensor | None = None,
                     max_num_points: int = -1,
                     point_cloud_range: npt.NDArray | Tensor | List[Tensor] | None = None,
                     device: torch.device = torch.device('cuda'),
                     determinstic: bool = True):
    try:
        from ivideogpt.ops.voxelize.voxelization import voxelization
    except:
        raise ImportError

    if isinstance(points, np.ndarray):
        points = torch.tensor(points, device=device, dtype=torch.float32)

    if labels is None:
        labels = torch.zeros_like(points[:, 0])
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels.astype(np.int32), device=points.device, dtype=torch.float32)
    points = torch.cat([points[:, :3], labels[..., None].float()], dim=1)

    # only use x y z label
    points = points[:, :4]
    # will add zero points in hard voxelization
    points[:, -1] = points[:, -1] + 1
    max_voxels = 1e5
    max_num_points = int(1e2)# / voxel_size[0]
    # remove nan
    points = points[~(torch.isnan(points[:, 0]) | torch.isnan(points[:, 1]) | torch.isnan(points[:, 2]))]
    # NOTE: need to add min range to transform voxel to original position
    if point_cloud_range is None:
        point_cloud_range = [points[:, 0].min(), points[:, 1].min(), points[:, 2].min(),
                             points[:, 0].max(), points[:, 1].max(), points[:, 2].max()]

    voxels = voxelization(points, voxel_size, point_cloud_range, int(max_num_points), int(max_voxels), determinstic)
    voxels = [e.cpu() for e in voxels] if not isinstance(voxels, Tensor) else voxels.cpu()

    # hard voxelization
    if max_num_points != -1 and max_voxels != -1:

        voxels, coors, _ = voxels

        labels = voxels[..., -1] # [M, N]
        unique_labels, mapped_labels = torch.unique(labels, sorted=True, return_inverse=True)
        label_counts = torch.zeros((len(voxels), len(unique_labels))).to(labels.device).long()
        label_counts.scatter_add_(1, mapped_labels.long(), torch.ones_like(mapped_labels).long())

        indices = torch.argsort(label_counts, dim=-1, descending=True)
        top1_labels = unique_labels[indices[:, 0]]
        if indices.shape[-1] > 1:
            top2_labels = unique_labels[indices[:, 1]]
            top1_labels = torch.where(top1_labels == 0, top2_labels, top1_labels)
        top1_labels = top1_labels - 1

    # TODO: add dynamic voxelization
    else:
        pass

    # note the sequence of coors
    voxels = np.concatenate([coors.numpy()[:, [2, 1, 0]], top1_labels.numpy()[..., np.newaxis]], axis=-1)  # [M, 4]

    return voxels


def convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05, show_cam=True,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False, save_name=None):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    ori_pct = [trimesh.PointCloud(pts3d[i].reshape(-1, 3), colors=imgs[i].reshape(-1, 3)) for i in range(len(imgs))]

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    if show_cam:
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(scene, pose_c2w, camera_edge_color,
                        None if transparent_cams else imgs[i], focals[i],
                        imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if save_name is None: save_name='scene'
    outfile = os.path.join(outdir, save_name+'.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile, ori_pct


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
    scene.min_conf_thr = min_conf_thr
    scene.thr_for_init_conf = thr_for_init_conf
    msk = to_numpy(scene.get_masks())
    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255 * c[0], 255 * c[1], 255 * c[2]) for c in cam_color]
    return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
                                        cam_color=cam_color)


def load_images(traj_file, size, square_ok=False, verbose=False, dynamic_mask_root=None, crop=True):
    """Open and convert all images or videos in a list or folder to proper input format for DUSt3R."""

    numpy_images = read_mp4(
        path=os.path.join(traj_file, 'rgb.mp4'))  # specialized for bridge1

    # numpy_images = read_mp4(
    #     path=os.path.join(traj_file, 'image_0.mp4'))  # specialized for bridge2

    # numpy_images = read_mp4(
    #     path=os.path.join(traj_file, 'exterior_image_1_left.mp4'))  # specialized for droid

    traj_id = os.path.basename(traj_file).removesuffix('.npz')
    root = os.path.dirname(traj_file)

    imgs = []
    # Sort items by their names
    for i, numpy_image in enumerate(numpy_images):
        full_path = os.path.join(root, f'{traj_id}_{i:04d}.png')
        # Process image files
        img = exif_transpose(Image.fromarray(numpy_image.astype(np.uint8))).convert('RGB')
        W1, H1 = img.size
        img = crop_img(img, size, square_ok=square_ok, crop=crop)
        W2, H2 = img.size

        if verbose:
            print(f' - Adding {full_path} with resolution {W1}x{H1} --> {W2}x{H2}')

        single_dict = dict(
            img=ImgNorm(img)[None],
            true_shape=np.int32([img.size[::-1]]),
            idx=len(imgs),
            instance=full_path,
            mask=~(ToTensor(img)[None].sum(1) <= 0.01)
        )
        
        if dynamic_mask_root is not None:
            dynamic_mask_path = os.path.join(dynamic_mask_root, os.path.basename(full_path))
        else:  # Sintel dataset handling
            dynamic_mask_path = full_path.replace('final', 'dynamic_label_perfect').replace('clean', 'dynamic_label_perfect')

        if os.path.exists(dynamic_mask_path):
            dynamic_mask = Image.open(dynamic_mask_path).convert('L')
            dynamic_mask = crop_img(dynamic_mask, size, square_ok=square_ok)
            dynamic_mask = ToTensor(dynamic_mask)[None].sum(1) > 0.99  # "1" means dynamic
            if dynamic_mask.sum() < 0.8 * dynamic_mask.numel():  # Consider static if over 80% is dynamic
                single_dict['dynamic_mask'] = dynamic_mask
            else:
                single_dict['dynamic_mask'] = torch.zeros_like(single_dict['mask'])
        else:
            single_dict['dynamic_mask'] = torch.zeros_like(single_dict['mask'])

        imgs.append(single_dict)

    assert imgs, 'No images found at ' + root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs


def get_reconstructed_scene_realtime(model, device, silent, image_size, traj_file, save_folder, batch_size, scenegraph_type, refid):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    model.eval()

    seq_name = os.path.basename(traj_file).removesuffix('.npz')
    imgs = load_images(traj_file, size=image_size, verbose=not silent)
    assert len(imgs) > 2, f"Too few images input: {len(imgs)}!"

    if scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)
    elif scenegraph_type == "oneref_mid":
        scenegraph_type = "oneref-" + str(len(imgs) // 2)
    else:
        raise ValueError(f"Unknown scenegraph type for realtime mode: {scenegraph_type}")

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=False)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
    os.makedirs(save_folder, exist_ok=True)

    view1, view2, pred1, pred2 = output['view1'], output['view2'], output['pred1'], output['pred2']
    pts1 = pred1['pts3d'].detach().cpu().numpy()
    pts2 = pred2['pts3d_in_other_view'].detach().cpu().numpy()
    points2 = []
    for batch_idx in range(len(view1['img'])):
        # colors1 = rgb(view1['img'][batch_idx])
        colors2 = rgb(view2['img'][batch_idx])
        # xyzrgb1 = np.concatenate([pts1[batch_idx], colors1], axis=-1)   #(H, W, 6)
        xyzrgb2 = np.concatenate([pts2[batch_idx], colors2], axis=-1)
        # np.save(save_folder + '/pts3d1_p' + str(batch_idx) + '.npy', xyzrgb1)
        # np.save(save_folder + '/pts3d2_p' + str(batch_idx) + '.npy', xyzrgb2)
        points2.append(xyzrgb2)

        # conf1 = pred1['conf'][batch_idx].detach().cpu().numpy()
        conf2 = pred2['conf'][batch_idx].detach().cpu().numpy()
        # np.save(save_folder + '/conf1_p' + str(batch_idx) + '.npy', conf1)
        np.save(save_folder + '/conf2_p' + str(batch_idx) + '.npy', conf2)

        # save the imgs of two views
        # img1_rgb = cv2.cvtColor(colors1 * 255, cv2.COLOR_BGR2RGB)
        # img2_rgb = cv2.cvtColor(colors2 * 255, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(save_folder + '/img1_p' + str(batch_idx) + '.png', img1_rgb)
        # cv2.imwrite(save_folder + '/img2_p' + str(batch_idx) + '.png', img2_rgb)

    # save ply files
    points = [trimesh.PointCloud(points2[i][..., :3].reshape(-1, 3), colors=points2[i][..., 3:].reshape(-1, 3)) for i in range(len(imgs))]
    for i, _points in enumerate(points):
        _points.export(f'{save_folder}/frame_{i:04d}.ply')

    return save_folder


def get_reconstructed_scene(model, device, silent, image_size, traj_file, save_folder, batch_size, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            temporal_smoothing_weight, translation_weight, shared_focal, not_batchify,
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    translation_weight = float(translation_weight)
    model.eval()

    seq_name = os.path.basename(traj_file).removesuffix('.npz')
    dynamic_mask_path = f'data/davis/DAVIS/masked_images/480p/{seq_name}'

    imgs = load_images(traj_file, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path)
    assert len(imgs) > 2, f"Too few images input: {len(imgs)}!"

    if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
    mode = GlobalAlignerMode.PointCloudOptimizer  
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                            flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                            num_total_iter=niter, empty_cache=False, batchify=not not_batchify)

    loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=0.01)

    os.makedirs(save_folder, exist_ok=True)
    outfile, points = get_3D_model_from_scene(save_folder, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                              clean_depth, transparent_cams, cam_size, show_cam)

    poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    depth_maps = scene.save_depth_maps(save_folder)
    dynamic_masks = scene.save_dynamic_masks(save_folder)
    conf = scene.save_conf_maps(save_folder)
    init_conf = scene.save_init_conf_maps(save_folder)
    rgbs = scene.save_rgb_imgs(save_folder)
    enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3)
    # save point cloud
    for i, _points in enumerate(points):
        _points.export(f'{save_folder}/frame_{i:04d}.ply')

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    init_confs = to_numpy([c for c in scene.init_conf_maps])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [cmap(d/depths_max) for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]
    init_confs_max = max([d.max() for d in init_confs])
    init_confs = [cmap(d/init_confs_max) for d in init_confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))
        imgs.append(rgb(init_confs[i]))

    # if two images, and the shape is same, we can compute the dynamic mask
    # if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
    #     motion_mask_thre = 0.35
    #     error_map = get_dynamic_mask_from_pairviewer(scene, both_directions=True, output_dir=save_folder, motion_mask_thre=motion_mask_thre)
    #     # imgs.append(rgb(error_map))
    #     # apply threshold on the error map
    #     normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
    #     error_map_max = normalized_error_map.max()
    #     error_map = cmap(normalized_error_map/error_map_max)
    #     imgs.append(rgb(error_map))
    #     binary_error_map = (normalized_error_map > motion_mask_thre).astype(np.uint8)
    #     imgs.append(rgb(binary_error_map * 255))

    return scene, outfile, imgs


def get_sparse_points(data_dir: str, save_dir: str, splits: list[str],
                      shared_sparse_pts_path: Queue,
                      terminate_process: Event, # type: ignore
                      device_index,
                      args,
                      device: torch.device=torch.device('cuda'),
    ) -> None:

    rank = args.rank
    if rank is not None:
        rank, all_ranks = rank.split('/')
        rank = int(rank)
        all_ranks = int(all_ranks)
    else:
        rank = 0
        all_ranks = 1

    def _handle_terminate(signum, frame):
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        CONSOLE.print(f"[on yellow]Step1 Deleted[/] [blue]{split}/{traj_id}[/]")
        terminate_process.set()
        sys.exit()

    signal.signal(signal.SIGTERM, _handle_terminate)

    torch.cuda.set_device(device_index)
    CONSOLE.log(f'Running sparse points on device {device_index}')

    # default arguments for monst3r
    silent = True
    image_size = 512  # choose from [512, 224]
    use_gt_davis_masks = False
    not_batchify = False

    weights_path = 'thirdparty/monst3r/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth'
    if not os.path.exists(weights_path):
        weights_path = 'Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt'
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
    if args.realtime:
        recon_fun = functools.partial(get_reconstructed_scene_realtime, model, device, silent, image_size)
    else:
        recon_fun = functools.partial(get_reconstructed_scene, model, device, silent, image_size)

    for split in tqdm(splits, "Processing split"):
        split_dir = os.path.join(data_dir, split)

        # Please do sortment here!
        traj_files = list(sorted(os.listdir(split_dir)))

        rank_size = len(traj_files) // all_ranks
        rank_start_idx = rank * rank_size
        rank_end_idx = (rank + 1) * rank_size if rank + 1 < all_ranks else -1
        traj_files = traj_files[rank_start_idx : rank_end_idx]
        CONSOLE.log(f'rank {rank} will host {rank_start_idx=} and {rank_end_idx=}')

        for traj_file in (
            pbar := tqdm(traj_files)
        ):
            traj_id = traj_file.removesuffix('.npz')
            save_folder = os.path.join(save_dir, 'points', split, traj_id)
            pbar.set_description(f"Processing {traj_id}")
        
            if os.path.exists(save_folder) and len(os.listdir(save_folder)) > 1:
                CONSOLE.print(f"[on blue]Step1[/] Skipped [blue]{split}/{traj_id}[/]")
                continue

            try:
                if args.realtime:

                    # Call the function with default parameters
                    outfile = recon_fun(
                        traj_file=os.path.join(split_dir, traj_file),
                        save_folder=save_folder,
                        scenegraph_type='oneref_mid',
                        refid=0,
                        batch_size=args.batch_size,
                    )

                else:

                    scene, outfile, imgs = recon_fun(
                        traj_file=os.path.join(split_dir, traj_file),
                        save_folder=save_folder,
                        batch_size=args.batch_size,
                        schedule='linear',
                        niter=300,
                        min_conf_thr=1.1,
                        as_pointcloud=True,
                        mask_sky=False,
                        clean_depth=True,
                        transparent_cams=False,
                        cam_size=0.05,
                        show_cam=True,
                        scenegraph_type='swinstride',
                        winsize=5,
                        refid=0,
                        temporal_smoothing_weight=0.01,
                        not_batchify=not_batchify,
                        translation_weight='1.0',
                        shared_focal=True,
                        flow_loss_weight=0.01,
                        flow_loss_start_iter=0.1,
                        flow_loss_threshold=25,
                        use_gt_mask=use_gt_davis_masks,
                    )

                # add path to buffer
                shared_sparse_pts_path.put(os.path.join(split, traj_id))

            except Exception as e:
                CONSOLE.print(f"[on blue]Step1[/] Failed [blue]{split}/{traj_id}[/] due to [red]{e}[/]")
                if int(os.getenv('DEBUG', 0)):
                    raise
                continue

    terminate_process.set()


@torch.no_grad()
def get_cameras(
    data_dir: str,
    save_dir: str,
    splits: list,
):

    assert n_view > 1, f'Number of views must be greater than 1!'

    # valid sequences must have at least 2 views
    check_view = 2

    device = torch.device('cuda:0')
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    # model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model = VGGT()
    # _URL = 'https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt'
    _URL = '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/thirdparty/vggt/vggt/checkpoints/model.pt'
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.load_state_dict(torch.load(_URL))
    model = model.to(device)
    CONSOLE.log(f'Loaded VGGT mdoel!')

    global_step = 0
    for split in (pbar1 := tqdm(splits, leave=False, desc='Process split ...')):
        pbar1.set_postfix(split=split)

        load_dir = os.path.join(data_dir, split, 'images1')
        save_dir = os.path.join(save_dir, 'cameras', split)
        os.makedirs(save_dir, exist_ok=True)

        all_trajs = set(os.listdir(load_dir))
        num_frames = int(os.listdir(load_dir)[0].removesuffix('.png').split('_')[2])

        unique_trajs = list(sorted(set(map(lambda fn: re.sub(r'_\d+_\d+_\d+\.png$', '', fn), all_trajs))))
        unique_trajs = list(sorted(map(lambda fn: fn.lstrip('0') or '0', unique_trajs)))
        unique_trajs = list(sorted(map(lambda fn: f'{int(fn):05d}', unique_trajs)))
        valid_trajs = list(sorted([
            fn for fn in unique_trajs if
            all(
                f'{fn}_00_{num_frames:02d}_{view_id}.png' in all_trajs
                for view_id in range(check_view)
            )]
        ))
        CONSOLE.log(f'Found {len(valid_trajs)=} for {split=}')

        # load_dir = os.path.join(data_dir, split)
        # save_dir = os.path.join(save_dir, 'cameras', split)
        # os.makedirs(save_dir, exist_ok=True)

        # all_trajs = set(os.listdir(load_dir))
        # valid_trajs = list(sorted(filter(
        #     lambda traj: len(os.listdir(os.path.join(load_dir, traj))) > 1,
        #     all_trajs,
        # )))

        for traj in (pbar2 := tqdm(valid_trajs, leave=False, desc='Process traj ...')):

            image_names = [
                os.path.join(load_dir, f'{traj}_00_{num_frames:02d}_{view_id}.png')
                for view_id in range(n_view)
                if os.path.exists(os.path.join(load_dir, f'{traj}_00_{num_frames:02d}_{view_id}.png'))
            ]
            assert len(image_names) >= 2, f'At least two views requred! But got {len(image_names)=}'
            pbar2.set_postfix(traj=traj, view=len(image_names))

            images = load_and_preprocess_images(image_names).to(device)

            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(images)

            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            # Note that since images are resized so 'intrinsic' needed to be post-processed.
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

            # Predict Point Maps
            point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

            extrinsic = extrinsic[0].float().cpu()  # -> [n_view, 3, 4]
            intrinsic = intrinsic[0].float().cpu()  # -> [n_view, 3, 3]
            depth_map = depth_map[0].float().cpu()  # -> [n_view, H, W]
            depth_conf = depth_conf[0].float().cpu()  # -> [n_view, H, W]
            point_map = point_map[0].float().cpu()  # -> [n_view, H, W, 3]
            point_conf = point_conf[0].float().cpu()  # -> [n_view, H, W, 3]

            # get rgbs for points
            images = images[0].permute(0, 2, 3, 1).cpu()  # -> [n_view, H, W, 3]
            point_map = torch.cat([point_map, images], dim=-1)  # -> [n_view, H, W, 6]

            # save ply files to visualize
            if global_step < 100:
                all_points = point_map.reshape(-1, 6).numpy()
                ply_data = trimesh.PointCloud(all_points[:, :3], colors=all_points[:, 3:])
                ply_data.export(os.path.join(save_dir, f'{traj}.ply'))

            np.savez_compressed(
                os.path.join(save_dir, f'{traj}.npz'),
                extrin=extrinsic.numpy(),
                intrin=intrinsic.numpy(),
                depth_map=depth_map.numpy(),
                depth_conf=depth_conf.numpy(),
                point_map=point_map.numpy(),
                point_conf=point_conf.numpy(),
            )

            global_step += 1

            # Construct 3D Points from Depth Maps and Cameras
            # which usually leads to more accurate 3D points than point map branch
            # point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0),
            #                                                             extrinsic.squeeze(0), 
            #                                                             intrinsic.squeeze(0))

            # Predict Tracks
            # choose your own points to track, with shape (N, 2) for one scene
            # query_points = torch.FloatTensor([[100.0, 200.0], 
            #                                     [60.72, 259.94]]).to(device)
            # track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])


def process_nksr(reconstructor, point_cloud_data, max_nn: int=20, device=torch.device('cuda')):

    def _preprocess_point_cloud(
        pcd,
        max_nn=20,
        normals=True,
    ):

        cloud = deepcopy(pcd)
        if normals:
            params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
            cloud.estimate_normals(params)
            cloud.orient_normals_towards_camera_location()

        return cloud

    if device != torch.device('cpu'):
        torch.cuda.synchronize()

    voxel_size = None  # [.01, .02, .04, None, None, None]
    detail_level = .8  # [None, None, None, .0, .4, .8]

    # point_cloud_original = o3d.geometry.PointCloud()
    # point_cloud_original.points = o3d.utility.Vector3dVector(point_cloud_data.vertices)
    point_cloud_original = point_cloud_data
    with_normal = _preprocess_point_cloud(point_cloud_original, max_nn=max_nn)

    input_xyz = torch.from_numpy(np.asarray(with_normal.points)).to(device).float()
    input_normal = torch.from_numpy(np.asarray(with_normal.normals)).to(device).float()

    # Note that input_xyz and input_normal are torch tensors of shape [N, 3] and [N, 3] respectively.
    field = reconstructor.reconstruct(input_xyz, input_normal, voxel_size=voxel_size, detail_level=detail_level)
    mesh = field.extract_dual_mesh(mise_iter=2)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.v.cpu().numpy())
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.f.cpu().numpy())
    o3d_mesh.paint_uniform_color((0, 1, 1))

    return o3d_mesh


def save_results(output_queue: queue.Queue) -> None:
    while True:
        save_func = None
        try:
            item = output_queue.get(timeout=30)
            if item is None:
                break
            save_args = item['save_args']
            save_func = item['save_func']
            save_func(*save_args)
        except queue.Empty:
            continue
        except Exception as e:
            if save_func is not None:
                CONSOLE.log(f'[on red]Failed to excuet save func {save_func} due to {e}, will skip!!!')


def get_dense_points(data_dir: str,
                     shared_sparse_pts_path: Queue,
                     shared_dense_pts_path: Queue,
                     terminate_process: Event, # type: ignore
                     device_index,
                     args,
                     device: torch.device=torch.device('cuda'),
    ) -> None:

    def _handle_terminate(signum, frame):
        # terminate save thread
        output_queue.put(None)
        save_thread.shutdown(wait=True)
        save_future.result()
        # delete unfinished missions
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        CONSOLE.print(f"[on yellow]Step2 Deleted[/] [blue]{traj_path}[/]")
        sys.exit()

    def _preprocess_points(points_data):

        points = np.asarray(points_data.points)
        colors = np.asarray(points_data.colors)
        mask = points[:, 2] < .6
        # mask = points[:, 2] < .4
        points = points[mask]
        colors = colors[mask]

        points_data = o3d.geometry.PointCloud()
        points_data.points = o3d.utility.Vector3dVector(points)
        points_data.colors = o3d.utility.Vector3dVector(points)

        cl, ind = points_data.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        points_data = points_data.select_by_index(ind)

        return points_data

    signal.signal(signal.SIGTERM, _handle_terminate)

    torch.cuda.set_device(device_index)
    CONSOLE.log(f'Running dense points on device {device_index}')

    reconstructor = nksr.Reconstructor(device)

    # build save thread
    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=4)
    save_future = save_thread.submit(save_results, output_queue)

    while True:
        save_folder = ''
        try:
            traj_path = shared_sparse_pts_path.get()
        except:
            if terminate_process.is_set():
                sys.exit()
            continue
        load_dir = os.path.join(data_dir, 'points', traj_path)
        save_folder = os.path.join(data_dir, 'mesh', traj_path)

        if os.path.exists(save_folder) and len(os.listdir(save_folder)) != 0:
            CONSOLE.print(f"[on blue]Step2[/] Skipped [blue]{traj_path}[/]")
            continue

        os.makedirs(save_folder, exist_ok=True)
        try:
            points_files = list(sorted(fnmatch.filter(os.listdir(load_dir), 'frame_*.ply')))
            for points_file in tqdm(points_files, leave=False):
                # points_data = trimesh.load(os.path.join(load_dir, points_file))
                points_data = o3d.io.read_point_cloud(os.path.join(load_dir, points_file))
                points_data = _preprocess_points(points_data)
                save_path = os.path.join(save_folder, points_file.replace('.ply', '_nksr.ply'))
                mesh = process_nksr(reconstructor, points_data, device=device)

                # put outputs into save buffer
                output_queue.put(
                    {
                        'save_args': (save_path, mesh),
                        'save_func': o3d.io.write_triangle_mesh,
                    }
                )

            # add path to buffer
            shared_dense_pts_path.put(traj_path)
            CONSOLE.print(f"[bold yellow]Step2: successfully processed {traj_path}")

        except Exception as e:
            CONSOLE.print(f"[on blue]Step2[/] Failed [blue]{traj_path}[/] due to [red]{e}[/]")
            continue


def project_3d_to_2d(points_3d, extrin, intrin):
    points_3d_homogeneous = torch.concat((points_3d, torch.ones([*points_3d.shape[:-1], 1]).to(points_3d)), -1)
    projection = intrin @ torch.linalg.inv(extrin)
    points_2d_homogeneous = points_3d_homogeneous @ projection.T
    points_2d = torch.concat([points_2d_homogeneous[..., :2] / points_2d_homogeneous[..., 2:3],
                              points_2d_homogeneous[..., 2:3]], axis=-1)
    return points_2d # [N, 3]


def get_occupancy(data_dir: str,
                  shared_dense_pts_path: Queue,
                  terminate_process: Event, # type: ignore
                  device_index,
                  args,
                  device: torch.device=torch.device('cuda'),
    ) -> None:

    def _handle_terminate(signum, frame):
        # terminate save thread
        output_queue.put(None)
        save_thread.shutdown(wait=True)
        save_future.result()
        # delete unfinished missions
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        CONSOLE.print(f"[on yellow]Step3 Deleted[/] [blue]{traj_path}[/]")
        sys.exit()

    signal.signal(signal.SIGTERM, _handle_terminate)

    torch.cuda.set_device(device_index)
    CONSOLE.log(f'Running occupancy on device {device_index}')

    def _pose_to_transform(pose):
        OPENGL = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

        c2w = np.eye(4)
        xyz, qwxyz = pose[:3], pose[3:]
        c2w[:3, -1] = xyz
        c2w[:3, :3] = Rotation.from_quat(qwxyz[[1, 2, 3, 0]]).as_matrix()

        # rot = np.eye(4)
        # rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        # transform = np.linalg.inv(c2w @ OPENGL @ rot)

        transform = c2w

        return transform

    # build save thread
    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=4)
    save_future = save_thread.submit(save_results, output_queue)

    colors60 = generate_colors()

    while True:
        save_folder = ''
        try:
            traj_path = shared_dense_pts_path.get()
        except:
            if terminate_process.is_set():
                sys.exit()
            continue
        points_dir = os.path.join(data_dir, 'points', traj_path)
        load_dir = os.path.join(data_dir, 'mesh', traj_path)
        save_folder = os.path.join(data_dir, 'occ', traj_path)
        label_folder = os.path.join(data_dir, 'semantics', traj_path)

        if os.path.exists(save_folder) and len(os.listdir(save_folder)) != 0:
            CONSOLE.print(f"[on blue]Step3[/] Skipped [blue]{traj_path}[/]")
            continue

        os.makedirs(save_folder, exist_ok=True)
        try:
            point_cloud_range = [-0.2, -0.2, 0, 0.2, 0.2, 0.4]
            # point_cloud_range = [-0.2, -0.2, 0, 0.2, 0.2, 0.6]
            voxel_size = [0.001] * 3  # TODO
            mesh_files = list(sorted(fnmatch.filter(os.listdir(load_dir), 'frame_*_nksr.ply')))

            # load extrins
            if os.path.exists(pose_file := os.path.join(points_dir, 'pred_traj.txt')):
                extrins = np.loadtxt(pose_file)
                extrins = np.array(list(map(lambda x: _pose_to_transform(x[1:]), extrins))).reshape(-1, 4, 4)
                extrins = torch.from_numpy(extrins).float().to(device)
            else:
                extrins = torch.eye(4).unsqueeze(dim=0).repeat(len(mesh_files), 1, 1).to(device)

            # load intrins
            if os.path.exists(intrin_file := os.path.join(points_dir, 'pred_intrinsics.txt')):
                intrin_ = np.loadtxt(intrin_file)[0].reshape(3, 3)
            else:
                intrin_ = np.loadtxt('/share/project/cwm/xiuyu.yang/work/dev6/outputs3/demos/rt1/train/00000/points/pred_intrinsics.txt')[0].reshape(3, 3)
            intrin_ = torch.from_numpy(intrin_).float().to(device)
            intrin = torch.eye(4).float().to(device)
            intrin[:3, :3] = intrin_

            labels2d_size = (480, 640)
            points3d_size = (384, 512)
            # labels2d_size = (256, 320)
            # points3d_size = (410, 512)
            src_w = points3d_size[1]
            tgt_w = labels2d_size[1]
            scale = tgt_w / src_w
            # intrin[1, 2] = int(intrin[1, 2] + 5)  # FIXME: this is a legacy issue!!!
            intrin[:2, :3] = intrin[:2, :3] * scale

            # get voxelizations
            for mesh_file, extrin in zip(mesh_files, extrins):

                # load occupancy points
                mesh = o3d.io.read_point_cloud(os.path.join(load_dir, mesh_file))
                points = torch.tensor(np.asarray(mesh.points), device=device, dtype=torch.float32)

                # find labels annotations
                label_file = mesh_file.replace('_nksr.ply', '.npz')
                try:
                    labels2d = np.load(os.path.join(label_folder, label_file))['annotated_frame_index']  # (h, w)
                    labels2d = torch.from_numpy(labels2d).long().to(device)
                    labels2d[labels2d == 255] = -1  # the original labels2d are in uint8 data foramt!

                    points2d = project_3d_to_2d(points, extrin=extrin, intrin=intrin)[:, :2].long()  # (n, 3)
                    masks2d = (points2d[:, 0] >= 0) & (points2d[:, 0] < labels2d_size[1]) & (points2d[:, 1] >= 0) & (points2d[:, 1] < labels2d_size[0])
                    points2d = points2d[masks2d]  # to avoid points lie outside the image

                    labels3d = torch.zeros((points.shape[0],)).long().to(device)
                    labels3d[masks2d] = labels2d[points2d[:, 1], points2d[:, 0]]
                    labels3d[labels3d == -1] = len(colors60) - 1  # labels must be positive when input to voxelization function

                except Exception as e:
                    labels3d = None
                    raise

                points = torch.concat([points, torch.ones_like(points[:, -1:])], dim=-1)
                voxels = points_to_voxels(points,
                                          voxel_size=voxel_size,
                                          labels=labels3d,
                                          point_cloud_range=point_cloud_range,
                                          device=device)

                if labels3d is not None:
                    CONSOLE.log(f'Get unique labels for occupancy: {np.bincount(voxels[:, -1].astype(np.uint8)).nonzero()[0]}')

                save_path = os.path.join(save_folder, mesh_file.replace('_nksr.ply', '.npy'))

                # put outputs into save buffer
                output_queue.put(
                    {
                        'save_args': (save_path, voxels),
                        'save_func': np.save,
                    }
                )
            CONSOLE.print(f"[bold yellow]Step3: successfully processed {traj_path}")

        except Exception as e:
            CONSOLE.print(f"[on blue]Step3[/] Failed [blue]{traj_path}[/] due to [red]{e}[/]")
            # if not isinstance(e, FileNotFoundError):
            #     raise
            continue


# docker run -it --rm --gpus='device=0' --userns=host \
# -v /share/project:/share/project \
# -w /share/project/cwm/xiuyu.yang/work/dev6 \
# baai-cwm-cr01-cn-beijing.cr.volces.com/baai-cwm-cr-namespace/dinosam:v2 \
# bash

# PYTHONPATH='.':$PYTHONPATH python ivideogpt/dataset/prepare_dataset.py --action labeling
@torch.no_grad()
def get_labels(data_dir: str, save_dir: str, splits: list, rank: Optional[str] = None):

    import json

    device = torch.device('cuda:0')

    # all_captions_file_path = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridge/renderings/captions/train/all_captions.jsonl'
    # all_labels_file_path = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridge/renderings/captions/train/labels.txt'

    # all_captions_file_path = '/share/project/cwm/xiuyu.yang/work/dev6/data/droid/renderings/captions/train/all_captions.jsonl'
    # all_labels_file_path = '/share/project/cwm/xiuyu.yang/work/dev6/data/droid/renderings/captions/train/labels.txt'

    all_captions_file_path = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/renderings/captions/all_captions.jsonl'
    all_labels_file_path = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/renderings/captions/labels.txt'

    with open(all_captions_file_path, 'r', encoding='utf-8') as f:
        all_captions = [
            json.loads(line.strip()) for line in f
        ]
    with open(all_labels_file_path, 'r', encoding='utf-8') as f:
        all_labels = [line.strip() for line in f]
    all_captions = {
        (episode['episode_id'], episode['split']): {
            'track_labels': episode['track_labels'],
            'label_ids': episode['label_ids'],
        }
        for episode in all_captions
    }
    all_labels.append('black robot gripper')

    # ! Step 1: initialize

    # init sam image predictor and video predictor model
    sam2_checkpoint = "thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-base"

    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    video_transforms = transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR)

    if rank is not None:
        rank, all_ranks = rank.split('/')
        rank = int(rank)
        all_ranks = int(all_ranks)
    else:
        rank = 0
        all_ranks = 1

    global_step = 0
    for split in (pbar1 := tqdm(splits, leave=False, desc='Process split ...')):
        pbar1.set_postfix(split=split)

        load_dir = os.path.join(data_dir, split)
        save_dir = os.path.join(save_dir, 'semantics', split)
        os.makedirs(save_dir, exist_ok=True)

        trajs = list(sorted(os.listdir(load_dir)))

        # get available trajs in points folder
        # points_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/renderings/points/train'
        points_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/renderings/points/val'
        points_trajs = list(sorted(os.listdir(points_dir)))
        trajs = list(sorted(filter(lambda traj: traj in points_trajs, trajs)))
        CONSOLE.log(f'Found {len(trajs)=} in total.')

        rank_size = len(trajs) // all_ranks
        rank_start_idx = rank * rank_size
        rank_end_idx = (rank + 1) * rank_size if rank + 1 < all_ranks else -1
        trajs = trajs[rank_start_idx : rank_end_idx]
        CONSOLE.log(f'rank {rank} will host {rank_start_idx=} and {rank_end_idx=}')

        for traj in (pbar2 := tqdm(trajs, leave=False, desc='Process traj ...')):
            pbar2.set_postfix(traj=traj)

            traj_captions =  all_captions.get((f'{int(traj):05d}', split), None)
            if traj_captions is None:
                CONSOLE.log(f'[bold red]Not found captions for {traj=}')
                continue
            traj_labels = traj_captions['track_labels']
            label_ids = traj_captions['label_ids']
            # ! add extra labels
            # traj_labels.append('black robot gripper')
            traj_labels.append('robot arm')
            label_ids.append(len(all_labels) - 1)
            text = f"{', '.join(traj_labels)}."
            CONSOLE.log(f'Processing {traj=} with {text=}')

            save_subdir = os.path.join(save_dir, traj)
            os.makedirs(save_subdir, exist_ok=True)

            try:
                video_reader = decord.VideoReader(
                    uri=os.path.join(load_dir, traj, 'rgb.mp4'), num_threads=2)
                # video_reader = decord.VideoReader(
                #     uri=os.path.join(load_dir, traj, 'image_0.mp4'), num_threads=2)
                # video_reader = decord.VideoReader(
                #     uri=os.path.join(load_dir, traj, 'exterior_image_1_left.mp4'), num_threads=2)
            except Exception as e:
                CONSOLE.log(f'[bold red]Failed to load video for {traj=} due to {e}!')
                continue

            frames = video_reader.get_batch(range(len(video_reader))).float()
            frames = frames.permute(0, 3, 1, 2).contiguous()  # -> [f, c, h, w]
            # frames = video_transforms(frames)
            frames = frames.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # -> [f, h, w, c]

            if os.path.exists(save_subdir) and len(fnmatch.filter(
                os.listdir(save_subdir), 'frame_*.npz')) == len(frames):
                CONSOLE.log(f'[bold yellow]Skipped {traj=}')
                continue

            try:

                # init video predictor state
                inference_state = video_predictor.init_state(
                    video_path=os.path.join(load_dir, traj, 'rgb.mp4')
                )
                # inference_state = video_predictor.init_state(
                #     video_path=os.path.join(load_dir, traj, 'image_0.mp4')
                # )
                # inference_state = video_predictor.init_state(
                #     video_path=os.path.join(load_dir, traj, 'exterior_image_1_left.mp4')
                # )
                ann_frame_idx = 0  # the frame index we interact with

                # ! Step2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
                init_frame = Image.fromarray(frames[0])

                inputs = processor(images=init_frame, text=text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = grounding_model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.25,
                    text_threshold=0.3,
                    target_sizes=[init_frame.size[::-1]]
                )

                # prompt SAM image predictor to get the mask for the object
                image_predictor.set_image(np.array(init_frame.convert("RGB")))

                # process the detection results
                input_boxes = results[0]["boxes"].cpu().numpy()
                OBJECTS = results[0]["labels"]  # NOTE this term may contains repeations

                # if 'robot arm or gripper' not in OBJECTS:
                #     CONSOLE.log(f'Skipped {traj=} due to no robot arm or gripper founded!')
                #     continue

                valid_indices = []
                for i, object in enumerate(OBJECTS):
                    if object in traj_labels:
                        valid_indices.append(i)
                input_boxes = np.array([input_boxes[i] for i in valid_indices])
                OBJECTS = [OBJECTS[i] for i in valid_indices]

                # ! map object labels back to global id
                global_ids = np.array([label_ids[traj_labels.index(object)] for object in OBJECTS]).astype(np.uint8)
                OBJECTS_TO_GLOBA_IDS = {object: global_id for object, global_id in zip(OBJECTS, global_ids)}

                # prompt SAM 2 image predictor to get the mask for the object
                masks, scores, logits = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                # convert the mask shape to (n, H, W)
                if masks.ndim == 3:
                    masks = masks[None]
                    scores = scores[None]
                    logits = logits[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)

                # ! Step 3: Register each object's positive points to video predictor with seperate add_new_points call

                PROMPT_TYPE_FOR_VIDEO = "box" # or "point"

                assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

                # If you are using point prompts, we uniformly sample positive points based on the mask
                if PROMPT_TYPE_FOR_VIDEO == "point":
                    # sample the positive points from mask for each objects
                    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

                    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                        labels = np.ones((points.shape[0]), dtype=np.int32)
                        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=object_id,
                            points=points,
                            labels=labels,
                        )
                # Using box prompt
                elif PROMPT_TYPE_FOR_VIDEO == "box":
                    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=object_id,
                            box=box,
                        )
                # Using mask prompt is a more straightforward way
                elif PROMPT_TYPE_FOR_VIDEO == "mask":
                    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                        labels = np.ones((1), dtype=np.int32)
                        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=object_id,
                            mask=mask
                        )
                else:
                    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

                # ! Step 4: Propagate the video predictor to get the segmentation results for each frame

                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # ! Step 5: Save and visualize the segment results across the video and save them

                save_gif = float(torch.rand(1)) < .1

                ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
                result_frames = []
                for frame_idx, segments in video_segments.items():
                    img = frames[frame_idx]
                    
                    object_ids = list(segments.keys())
                    masks = list(segments.values())
                    masks = np.concatenate(masks, axis=0)

                    # ! Save the resulted masks and corresponding labels
                    np.savez_compressed(
                        os.path.join(save_subdir, f'frame_{frame_idx:04d}.npz'),
                        masks=masks.astype(np.bool_),
                        track_labels=OBJECTS,
                        object_ids=np.array(object_ids).astype(np.uint8),
                        label_ids=global_ids.astype(np.uint8),
                    )

                    if save_gif or global_step < 10:

                        # ! plot instances
                        detections = sv.Detections(
                            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                            mask=masks, # (n, h, w)
                            class_id=np.array(object_ids, dtype=np.int32),
                        )

                        empty_frame = np.ascontiguousarray(
                            np.zeros_like(img)
                        )
                        annotated_frame = img.copy()
                        labels = [f'{OBJECTS_TO_GLOBA_IDS[object]}:{object}' for object in [ID_TO_OBJECTS[i] for i in object_ids]]  # global_id : object_string
                        # box_annotator = sv.BoxAnnotator()
                        # annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                        label_annotator = sv.LabelAnnotator()
                        empty_frame = label_annotator.annotate(empty_frame, detections=detections, labels=labels)
                        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
                        mask_annotator = sv.MaskAnnotator()
                        empty_frame = mask_annotator.annotate(scene=empty_frame, detections=detections)
                        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

                        # ! plot semantics
                        unique_global_ids = list(set(global_ids))
                        sem_masks = []
                        for unique_global_id in unique_global_ids:
                            sem_indices = np.array(global_ids) == unique_global_id
                            sem_mask = np.any(masks[sem_indices], axis=0)  # (h, w)
                            sem_masks.append(sem_mask)
                        sem_masks = np.array(sem_masks)  # -> (n, h, w)

                        detections = sv.Detections(
                            xyxy=sv.mask_to_xyxy(sem_masks),  # (n, 4)
                            mask=sem_masks, # (n, h, w)
                            class_id=np.array(unique_global_ids, dtype=np.int32),
                        )

                        empty_frame2 = np.ascontiguousarray(
                            np.zeros_like(img)
                        )
                        labels = [f'{global_id}:{all_labels[global_id]}' for global_id in unique_global_ids]  # global_id: class
                        label_annotator = sv.LabelAnnotator()
                        empty_frame2 = label_annotator.annotate(empty_frame2, detections=detections, labels=labels)
                        mask_annotator = sv.MaskAnnotator()
                        empty_frame2 = mask_annotator.annotate(scene=empty_frame2, detections=detections)

                        # ! save results
                        empty_frame = Image.fromarray(empty_frame)
                        empty_frame2 = Image.fromarray(empty_frame2)
                        annotated_frame = Image.fromarray(annotated_frame)

                        W, H = empty_frame.size
                        merge_frame = Image.new('RGB', (W * 3, H))
                        merge_frame.paste(empty_frame, (0, 0))
                        merge_frame.paste(empty_frame2, (W, 0))
                        merge_frame.paste(annotated_frame, (W * 2, 0))
                        merge_frame.save(os.path.join(save_subdir, f'annotated_frame_{frame_idx:05d}.png'))

                        result_frames.append(merge_frame)

                if result_frames:
                    gif_path = os.path.join(save_subdir, 'result.gif')
                    result_frames[0].save(gif_path, save_all=True, append_images=result_frames[1:], duration=100, loop=0)

            except Exception as e:
                CONSOLE.log(f'[on blue]Get labels[/] Failed [blue]{traj}[/] due to [red]{e}[/]')
                if int(os.getenv('DEBUG', 0)):
                    raise
                continue

            global_step += 1


def _postprocess_labels(load_dir: str, mask_annotator: sv.MaskAnnotator, colors60_list: list[tuple[int]], data: tuple):

    traj, global_step = data

    annotated_frames = []
    frames = list(sorted(fnmatch.filter(os.listdir(os.path.join(load_dir, traj)), 'frame_*.npz')))
    mask_indices_sequence = None
    for frame in (pbar3 := tqdm(frames, leave=False, desc='Process frame ...')):
        pbar3.set_postfix(frame=frame)

        file_path = os.path.join(load_dir, traj, frame)
        try:
            labels_data = np.load(file_path, allow_pickle=True)
            labels = dict(labels_data)
            labels_data.close()
        except Exception as e:
            CONSOLE.log(f'[bold red]Skipped {traj=} due to {e}!')
            continue

        masks = labels['masks'].astype(np.bool_)  # (n, h, w)
        label_ids = labels['label_ids']  # (n,)

        if (
            'annotated_frame_color' in labels
            and
            'annotated_frame_index' in labels
        ):
            continue

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(label_ids, dtype=np.int32),
        )

        H, W = masks.shape[-2:]
        annotated_frame = np.zeros((H, W, 3), dtype=np.uint8)  # -> initialized in black (corresponds to the background)

        # start from the first frame, we compute the sequence of all masks;
        # which force all the frames to use the same sequence to avlid the
        # label flickerings when there exists tiny differences in masks'areas
        # across frames.
        if mask_indices_sequence is None:
            mask_indices_sequence = np.flip(np.argsort(detections.area))

        # >>> mask_annotator.annotate(scene=annotated_frame, detections=detections)
        colored_mask = np.array(annotated_frame, copy=True, dtype=np.uint8)
        for detection_idx in mask_indices_sequence:
            color = resolve_color(
                color=mask_annotator.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=mask_annotator.color_lookup
            )
            mask = detections.mask[detection_idx]
            colored_mask[mask] = color.as_bgr()
        cv2.addWeighted(
            colored_mask, mask_annotator.opacity, annotated_frame, 1 - mask_annotator.opacity, 0, dst=annotated_frame
        )
        annotated_frame = annotated_frame.astype(np.uint8)

        # get the label ids for 2d map
        indices_2d = np.zeros((H, W), dtype=np.int32) - 1  # -> initialized in -1 (cooresponds to the background)
        for detection_idx in mask_indices_sequence:
            label_id = label_ids[detection_idx]
            mask = detections.mask[detection_idx]
            indices_2d[mask] = label_id
        indices_2d = indices_2d.astype(np.uint8)  # NOTE: -1 will be converted to 255

        labels['annotated_frame_color'] = annotated_frame
        labels['annotated_frame_index'] = indices_2d
        np.savez_compressed(file_path, **labels)

        annotated_frames.append(annotated_frame)

    if (random.random() < .5 or global_step < 20) and len(annotated_frames) > 0:
        annotated_frames = [Image.fromarray(frame) for frame in annotated_frames]
        annotated_frames[0].save(os.path.join(load_dir, traj, 'result2.gif'), save_all=True, append_images=annotated_frames[1:], duration=100, loop=0)
        CONSOLE.log(f'Saved gif for {traj=}.')


def generate_colors(n=60):
    colors_list = []
    for i in range(n):
        h = i / n
        s, v = 0.75, 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color = (int(r * 255), int(g * 255), int(b * 255))
        colors_list.append(color)
    return colors_list


def postprocess_labels(data_dir: str, splits: list[str]):

    colors60_list = generate_colors(60)
    colors60_list[-1] = (0, 0, 0)
    CONSOLE.log(colors60_list)
    colors60 = ColorPalette([Color(*color) for color in colors60_list])
    mask_annotator = sv.MaskAnnotator(color=colors60, opacity=1.)

    for split in (pbar1 := tqdm(splits, leave=False, desc='Process split ...')):
        pbar1.set_postfix(split=split)

        load_dir = os.path.join(data_dir, 'semantics', split)
        trajs = list(sorted(os.listdir(load_dir)))

        _postprocess_labels_ = partial(_postprocess_labels, load_dir, mask_annotator, colors60_list)

        with Pool(processes=16) as pool:
            list(tqdm(pool.imap_unordered(_postprocess_labels_, list(zip(trajs, range(len(trajs))))), total=len(trajs)))


@torch.no_grad()
def get_captions(data_dir: str,
                 save_dir: str,
                 splits: list,
                 rank: Optional[str] = None,
    ) -> None:

    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig

    device = torch.device('cuda:0')

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True).eval()

    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    results = []

    if rank is not None:
        rank, all_ranks = rank.split('/')
        rank = int(rank)
        all_ranks = int(all_ranks)
    else:
        rank = 0
        all_ranks = 1

    save_dir = os.path.join(save_dir, 'captions')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'rank_{rank}.jsonl')

    global_step = 0
    for split in (pbar1 := tqdm(splits, leave=False, desc='Process split ...')):
        pbar1.set_postfix(split=split)

        load_dir = os.path.join(data_dir, split, 'images1')

        processed_ids = []
        # if os.path.exists(save_path):
        #     with open(save_path, 'r', encoding='utf-8') as f:
        #         results = [
        #             json.loads(line.strip()) for line in f
        #         ]
        #     results = list(sorted(filter(lambda r: r['split'] == split, results)))
        #     processed_ids = list(map(lambda r: r['episode_id'], results))

        trajs = list(sorted(filter(lambda fn: int(fn.removesuffix('.png').split('_')[1]) == 0, os.listdir(load_dir))))
        if len(trajs[0].removesuffix('.png').split('_')) == 4:  # [traj_id, start_frame_id, n_frame, view_id]
            trajs = list(sorted(filter(lambda fn: int(fn.removesuffix('.png').split('_')[-1]) == 0, trajs)))

        rank_size = len(trajs) // all_ranks
        rank_start_idx = rank * rank_size
        rank_end_idx = (rank + 1) * rank_size if rank + 1 < all_ranks else -1
        trajs = trajs[rank_start_idx : rank_end_idx]
        CONSOLE.log(f'rank {rank} will host {rank_start_idx=} and {rank_end_idx=}')
        trajs = list(sorted(set(trajs)))

        for traj in (pbar2 := tqdm(trajs, leave=False, desc='Process traj ...')):
            traj_id = traj.removesuffix('.png').split('_')[0]
            pbar2.set_postfix(traj=traj_id)

            if traj_id in processed_ids:
                continue
            traj_path = os.path.join(load_dir, traj)

            query = tokenizer.from_list_format([
                {'image': traj_path},
                {'text': 'List the main object classes in the image, with only one word for each class (no more than ten):'},
            ])
            response, _ = model.chat(tokenizer, query=query, history=None)
            response = list(set([x.lower() for x in response.strip('.').split(', ')]))
            if len(response) > 10:
                continue

            results.append(
                r := {
                    'episode_id': traj_id,
                    'split': split,
                    'raw_labels': response,
                }
            )
            CONSOLE.log(r)
            processed_ids.append(traj_id)

            global_step += 1
            if global_step % 5e2 == 0:
                results = list(sorted(results, key=lambda x: (x['episode_id'], x['split'])))
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(map(json.dumps, results)))


@torch.no_grad()
def postprocess_captions(data_dir: str):

    import json
    import re
    import random
    from transformers import AutoTokenizer, AutoModel
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0')

    caption_files = list(sorted(fnmatch.filter(os.listdir(data_dir), 'rank*.jsonl')))
    all_captions = []

    # 1. load raw captions
    if len(caption_files) > 0:

        for file in tqdm(caption_files):
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                r = [json.loads(line.strip()) for line in f]
                all_captions.extend(r)
        CONSOLE.log(f'Loaded captions for all {len(all_captions)} episodes.')

    else:

        with open(os.path.join(data_dir, 'all_captions.jsonl'), 'r', encoding='utf-8') as f:
            all_captions = [json.loads(line.strip()) for line in f]

    all_labels = list(chain(*[
        caption['raw_labels'] for caption in tqdm(all_captions)
    ]))
    CONSOLE.log(f'Have {len(all_labels)} raw labels.')

    # 2. ilter out outliers
    all_captions = list(sorted(all_captions, key=lambda x: x['episode_id']))
    all_labels = list(chain(*[
        caption['raw_labels'] for caption in tqdm(all_captions)
    ]))

    pattern = re.compile(f'^[A-Za-z ]+$')
    for i in tqdm(range(len(all_captions)), 'Filter out outliers ...'):
        raw_labels = all_captions[i]['raw_labels']
        track_labels = list(filter(pattern.match, raw_labels))
        all_captions[i]['track_labels'] = track_labels
    all_labels = list(filter(pattern.match, all_labels))
    CONSOLE.log(f'Have {len(all_labels)} labels after filtering out outliers.')

    label_counts = Counter(all_labels)
    frequency = np.array(list(label_counts.values()))
    threshold = max(1, np.percentile(frequency, 10))
    # all_labels = [label for label in all_labels if label_counts[label] > threshold]
 
    # 2.1 filter out repeations
    for i in tqdm(range(len(all_captions)), 'Filter out repeations ...'):
        track_labels = all_captions[i]['track_labels']
        track_labels = list(set(track_labels))
        all_captions[i]['track_labels'] = track_labels
    all_labels = list(set(all_labels))
    all_labels.extend(
        extra_labels := ['gripper', 'countertop', 'otherproperty', 'background']
    )
    CONSOLE.log(f'Have {len(all_labels)} labels after filtering out repeations.')

    raw_labels_file_path = os.path.join(data_dir, 'raw_labels.txt')
    with open(raw_labels_file_path, 'w', encoding='utf-8') as f:
        f.writelines(lbl + '\n' for lbl in all_labels)
    CONSOLE.log(f'Raw labels are saved to {raw_labels_file_path}.')

    # 3. get text embeddings
    # model_name = 'bert-base-uncased'  # same as the tracking model!
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    batch_size = 128
    all_labels = list(sorted(all_labels))  # IMPORTANT!!!
    for i in tqdm(range(0, len(all_labels), batch_size), desc='Get text embeddings ...'):
        batch_labels = all_labels[i : i + batch_size]
        tokens = tokenizer(batch_labels, padding=True, truncation=True, return_tensors='pt').to(device)
        output = model(**tokens)
        # mean pooling - take attention mask into account for correct averaging
        token_embeddings = output[0]
        input_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings.append(
            (torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)).cpu()
        )
    embeddings = torch.cat(embeddings, dim=0).numpy()  # -> [N, D]
    CONSOLE.log(f'Sum of all embeddings: {np.sum(embeddings)=}')

    embeddings = PCA(n_components=128).fit_transform(embeddings)

    # 4. K-means
    num_clusters = 51
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=50)
    cluster_ids = kmeans.fit_predict(embeddings)  # -> [N,]
    CONSOLE.log(f'Sum of all cluster_ids: {sum(cluster_ids)=}')

    # 4.1 get all labels for each cluster
    cluster_to_labels = [[] for _ in range(num_clusters)]
    for label, cluster_id in zip(all_labels, cluster_ids):
        cluster_to_labels[cluster_id].append(label)

    # 4.2 get single top label for each cluster
    top_labels = {}
    top_embeddings = {}
    for cluster_id, cluster_labels in enumerate(tqdm(cluster_to_labels)):
        valid_cluster_labels = list(
            filter(
                lambda lbl: (
                    (label_counts[lbl] > threshold and len(lbl.split(' ')) == 1)
                    or lbl in extra_labels
                ), cluster_labels
            )
        )
        if not valid_cluster_labels:
            continue
        indices = [all_labels.index(lbl) for lbl in valid_cluster_labels]
        cluster_embeds = embeddings[indices]
        center = cluster_embeds.mean(axis=0)
        distances = np.linalg.norm(cluster_embeds - center, axis=1)
        top_embeddings[cluster_id] = cluster_embeds[np.argmin(distances)]
        top_label = valid_cluster_labels[np.argmin(distances)]
        top_labels[cluster_id] = top_label

    # ! plot label embeddings
    plot_clusters = True
    if plot_clusters:
        import seaborn as sns
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(14, 12))
        palette = sns.color_palette('hsv', n_colors=num_clusters)

        unique_cluster_ids = list(set(cluster_ids.tolist()))
        for cid in tqdm(unique_cluster_ids, desc='Plotting ...'):
            idx = cluster_ids == cid
            cluster_points = embeddings_2d[idx]

            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                s=5,
                alpha=.5,
                color=palette[cid],
            )

            cluster_center_points = cluster_points.mean(axis=0)
            plt.scatter(
                cluster_center_points[0],
                cluster_center_points[1],
                label=f'cluster {cid}',
                s=15,
                alpha=.6,
                color=palette[cid],
                marker='o',
                zorder=5,
            )

        plt.legend()
        plt.savefig(os.path.join(save_dir, 'label_embeddings.png'), dpi=200)

    # 4.1 write top_labels to file
    labels = list(top_labels.values())
    if 'backgournd' not in labels:
        labels.append('background')
    top_labels_file_path = os.path.join(data_dir, 'labels.txt')
    with open(top_labels_file_path, 'w', encoding='utf-8') as f:
        f.writelines(lbl + '\n' for lbl in labels)
    CONSOLE.log(f'Finalized top labels are saved to {top_labels_file_path}.')

    # 5. get correspondenc map
    label_map = {}
    for cluster_id, cluster_labels in enumerate(tqdm(cluster_to_labels)):
        top_label = top_labels.get(cluster_id, 'background')
        for lbl in cluster_labels:
            label_map[lbl] = top_label

    # ! save label clusters
    label_clusters: dict[str, list[str]] = defaultdict(list)
    for k, v in label_map.items():
        label_clusters[v].append(k)
    label_clusters_ = [
        {
            k: list(sorted(label_clusters[k]))
        }
        for k in label_clusters.keys()
    ]
    label_clusters_ = list(sorted(label_clusters_, key=lambda x: list(x.keys())[0]))
    with open(os.path.join(data_dir, 'label_clusters.jsonl'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(map(json.dumps, label_clusters_)))

    # 6. update labels for each episodes
    for i in tqdm(range(len(all_captions)), 'Updating labels for each episodes ...'):
        caption = all_captions[i]
        new_labels = [label_map[label] for label in caption['track_labels']]
        caption['labels'] = new_labels
        caption['label_ids'] = [labels.index(label) for label in new_labels]
        all_captions[i] = caption

    all_captions = list(sorted(all_captions, key=lambda x: x['episode_id']))
    with open(os.path.join(data_dir, 'all_captions.jsonl'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(map(json.dumps, all_captions)))


def align_multiview_extrins(data_dir, splits, rank: Optional[str] = None):

    def apply_depth_colormap(gray, minmax=None, cmap=cv2.COLORMAP_JET):
        """
        Input:
            gray: gray image, tensor/numpy, (H, W)
        Output:
            depth: (3, H, W), tensor
        """
        if type(gray) is not np.ndarray:
            gray = gray.detach().cpu().numpy().astype(np.float32)
        gray = gray.squeeze()
        assert len(gray.shape) == 2
        x = np.nan_to_num(gray)  # change nan to 0
        if minmax is None:
            mi = np.min(x)  # get minimum positive value
            ma = np.max(x)
        else:
            mi, ma = minmax
        x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
        x = 1 - x  # reverse the colormap
        x = (255 * x).astype(np.uint8)
        x = Image.fromarray(cv2.applyColorMap(x, cmap))
        x = transforms.ToTensor()(x)  # (3, H, W)
        return x

    def project_3d_to_2d_np(points_3d, extrin, intrin):
        points_3d_homogeneous = np.concatenate((points_3d, np.ones([*points_3d.shape[:-1], 1])), -1)
        projection = intrin @ extrin
        points_2d_homogeneous = points_3d_homogeneous @ projection.T # (projection @ points_3d_homogeneous.T).T
        points_2d = np.concatenate([points_2d_homogeneous[..., :2] / points_2d_homogeneous[..., 2:3],
                                    points_2d_homogeneous[..., 2:3]], axis=-1)
        return points_2d # [N, 3]

    # def compute_scale_and_shift(prediction, target, mask):
    #     # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    #     a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    #     a_01 = torch.sum(mask * prediction, (1, 2))
    #     a_11 = torch.sum(mask, (1, 2))

    #     # right hand side: b = [b_0, b_1]
    #     b_0 = torch.sum(mask * prediction * target, (1, 2))
    #     b_1 = torch.sum(mask * target, (1, 2))

    #     # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    #     x_0 = torch.zeros_like(b_0)
    #     x_1 = torch.zeros_like(b_1)

    #     det = a_00 * a_11 - a_01 * a_01
    #     valid = det.nonzero()

    #     x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    #     x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    #     return x_0, x_1


    def compute_scale_and_shift(prediction, target, mask):
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        b_0 = torch.sum(mask * prediction * target, (1, 2))

        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_0)

        valid = a_00 != 0
        x_0[valid] = b_0[valid] / a_00[valid]

        return x_0, x_1


    def compute_scale_and_shift_np(prediction, target, mask):
        a_00 = np.sum(mask * prediction * prediction, axis=(1, 2))
        b_0 = np.sum(mask * prediction * target, axis=(1, 2))

        x_0 = np.zeros_like(b_0)
        x_1 = np.zeros_like(b_0)

        valid = a_00 != 0
        x_0[valid] = b_0[valid] / a_00[valid]

        return x_0, x_1

    # -------------------------------------------------------------------
    # this is an legacy issue for bridgev2 dataset !!!
    ori_h, ori_w = 256, 320
    video_size = [320, 480]
    ori_aspect_ratio = ori_w / ori_h
    aspect_ratio = video_size[1] / video_size[0]
    if aspect_ratio < ori_aspect_ratio:
        new_w = int(ori_w * (video_size[0] / ori_h))
        new_h = video_size[0]
    else:
        new_w = video_size[1]
        new_h = int(ori_h * (video_size[1] / ori_w))

    ori_h, ori_w = 480, 640
    depth_transforms = transforms.Compose(
        [
            transforms.Resize(ori_h, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(tuple([ori_h, ori_w])),
            transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(tuple(video_size)),
        ]
    )
    # -------------------------------------------------------------------

    if rank is not None:
        rank, all_ranks = rank.split('/')
        rank = int(rank)
        all_ranks = int(all_ranks)
    else:
        rank = 0
        all_ranks = 1

    for split in (pbar1 := tqdm(splits, leave=False, desc='Process split ...')):
        pbar1.set_postfix(split=split)

        points_dir = os.path.join(data_dir, 'points', split)
        load_dir = os.path.join(data_dir, 'cameras', split)
        save_dir = os.path.join(data_dir, 'aligned_cameras', split)
        os.makedirs(save_dir, exist_ok=True)

        trajs = list(sorted(fnmatch.filter(os.listdir(load_dir), '*.npz')))
        trajs = list(sorted(map(lambda fn: fn.removesuffix('.npz'), trajs)))

        rank_size = len(trajs) // all_ranks
        rank_start_idx = rank * rank_size
        rank_end_idx = (rank + 1) * rank_size if rank + 1 < all_ranks else -1
        trajs = trajs[rank_start_idx : rank_end_idx]
        CONSOLE.log(f'rank {rank} will host {rank_start_idx=} and {rank_end_idx=}')

        for traj in (pbar2 := tqdm(trajs, leave=False, desc='Process traj ...')):
            pbar2.set_postfix(traj=traj)

            save_path = os.path.join(save_dir, f'{traj}.npz')
            if os.path.exists(save_path):
                continue

            try:

                depth1_path = os.path.join(points_dir, f'{int(traj)}', 'frame_0000.npy')
                camera_path = os.path.join(load_dir, f'{int(traj):05d}.npz')
                if os.path.exists(camera_path) and os.path.exists(depth1_path):

                    depth1 = np.load(depth1_path)
                    depth1 = depth_transforms(Image.fromarray(depth1))
                    # camera_data: extrin, intrin, depth_map, depth_conf, point_map, point_map_conf
                    camera_data = np.load(camera_path)
                    depth2 = camera_data['depth_map'][0, ..., 0]  # use the first view
                    depth2 = Image.fromarray(depth2)
                    depth2 = depth2.resize((480, 320))

                    # align
                    depth1 = np.array(depth1)
                    depth2 = np.array(depth2)
                    mask = np.ones_like(depth1)
                    scale, shift = compute_scale_and_shift_np(
                        depth2[np.newaxis], depth1[np.newaxis], mask[np.newaxis],
                    )

                    # scale and shift
                    extrin2_ = camera_data['extrin']
                    extrin2 = np.eye(4)[np.newaxis, ...].repeat(len(extrin2_), axis=0)
                    extrin2[:, :3, :4] = extrin2_

                    global_shift = np.array([0., 0., float(shift), 1.])
                    global_shift = np.linalg.inv(extrin2[0]) @ global_shift.T
                    extrin2[..., :3, -1] = extrin2[..., :3, -1] * float(scale) + global_shift[:3]

                    # save results
                    np.savez(
                        save_path,
                        aligned_extrin=extrin2,  # [n_view, 4, 4]
                        intrin=camera_data['intrin'],  # [n_view, 3, 3]
                    )

                    # save visualizations
                    if random.random() > .5:
                        points1_path = os.path.join(points_dir, f'{int(traj)}', 'frame_0000.ply')
                        points1_data = o3d.io.read_point_cloud(points1_path)
                        points1 = np.asarray(points1_data.points)
                        colors1 = np.asarray(points1_data.colors)
                        intrin_ = np.loadtxt(
                            os.path.join(points_dir, f'{int(traj)}', 'pred_intrinsics.txt')
                        )[0].reshape(3, 3)
                        intrin = np.eye(4)
                        intrin[:3, :3] = intrin_

                        n_view = len(extrin2)
                        W = int(intrin[0][2] * 2)
                        H = int(intrin[1][2] * 2)
                        canvas = Image.new('RGB', (W * n_view, H * 2))
                        for i_view in range(n_view):
                            points2d1 = project_3d_to_2d_np(points1, extrin=extrin2[i_view], intrin=intrin)
                            mask2d1 = (points2d1[:, 0] >= 0) & (points2d1[:, 0] < W) & (points2d1[:, 1] >= 0) & (points2d1[:, 1] < H)
                            points2d1 = points2d1[mask2d1]
                            colors2d1 = colors1[mask2d1]

                            points1_xy = points2d1[:, :2].astype(np.int32)
                            image = np.zeros((H, W, 3), dtype=np.uint8)
                            depth = np.full((H, W), np.inf)
                            image[points1_xy[:, 1], points1_xy[:, 0]] = colors2d1 * 255.
                            depth[points1_xy[:, 1], points1_xy[:, 0]] = points2d1[:, 2]
                            depth[np.isinf(depth)] = 0
                            depth = apply_depth_colormap(depth).permute(1, 2, 0).numpy() * 255.

                            image = Image.fromarray(image.astype(np.uint8))
                            depth = Image.fromarray(depth.astype(np.uint8))

                            canvas.paste(image, (W * i_view, 0, W * (i_view + 1), H))
                            canvas.paste(depth, (W * i_view, H, W * (i_view + 1), H * 2))

                        canvas.save(
                            os.path.join(save_dir, f'{traj}.png')
                        )

            except Exception as e:
                if int(os.getenv('DEBUG', 0)):
                    raise
                continue


@torch.no_grad()
def get_render(data_dir,
               splits: list,
               save_color: bool = True,
               rank: Optional[str] = None,
    ) -> None:

    device = torch.device('cuda:0')

    if rank is not None:
        rank, all_ranks = rank.split('/')
        rank = int(rank)
        all_ranks = int(all_ranks)
    else:
        rank = 0
        all_ranks = 1

    colors60_list = generate_colors(n=60)
    colors60_list[-1] = (0, 0, 0)
    colors60 = torch.from_numpy(np.array(colors60_list)).float().to(device)

    def apply_semantic_colormap(semantic):

        max_label = semantic.max()

        x = torch.zeros((3, semantic.shape[0], semantic.shape[1]), dtype=torch.float)
        for i in range(max_label + 1):
            x[0][semantic == i] = colors60[i][0]
            x[1][semantic == i] = colors60[i][1]
            x[2][semantic == i] = colors60[i][2]

        return x / 255.0

    num_channels_language_feature = 12

    # ! must be aligned with occupancy!!!!
    point_cloud_range = [-0.2, -0.2, 0, 0.2, 0.2, 0.4]
    # point_cloud_range = [-0.2, -0.2, 0, 0.2, 0.2, 0.6]
    voxel_size = [0.001] * 3

    base_scale = 0.00023
    exp_scale = 3.7

    # base_scale = 0.00047
    # exp_scale = 3.2

    occ_range = np.array([point_cloud_range[0:3], point_cloud_range[3:6]])
    occ_dim = np.array(voxel_size)
    occ_shape = ((occ_range[1] - occ_range[0]) / occ_dim).astype(np.uint16)

    # ! compute gaussian scales
    depth_bins = torch.arange(occ_shape[-1], device=device) + 1
    depth_bins = (depth_bins - depth_bins.min()) / (depth_bins.max() - depth_bins.min()) + 1
    gs_scales = base_scale * (depth_bins ** exp_scale)
    gs_scales = gs_scales[None, None, ...].expand(*occ_shape).reshape(-1)

    # ! initialize gs attributes
    xyz = create_full_center_coords(range=occ_range, dim=occ_dim).float().view(-1, 3).to(device)  # [N, 3]
    semantics_zero = torch.zeros((*occ_shape, 1)).long()
    rgb = torch.zeros_like(xyz)
    rot = torch.zeros((xyz.shape[0], 4)).to(device).float()
    rot[:, 0] = 1
    scale = torch.ones((xyz.shape[0], 3)).to(device).float() * gs_scales[:, None]
    opacity = torch.ones((xyz.shape[0], 1)).float().to(device)

    for split in (pbar1 := tqdm(splits, leave=False, desc='Process split ...')):
        pbar1.set_postfix(split=split)

        camera_dir = os.path.join(data_dir, 'aligned_cameras', split)
        load_dir = os.path.join(data_dir, 'occ', split)
        save_dir = os.path.join(data_dir, 'render', split)
        os.makedirs(save_dir, exist_ok=True)

        trajs = list(sorted(os.listdir(load_dir)))
        # trajs = ['534', '752'] + trajs[:5]

        rank_size = len(trajs) // all_ranks
        rank_start_idx = rank * rank_size
        rank_end_idx = (rank + 1) * rank_size if rank + 1 < all_ranks else -1
        trajs = trajs[rank_start_idx : rank_end_idx]
        CONSOLE.log(f'rank {rank} will host {rank_start_idx=} and {rank_end_idx=}, num: {rank_end_idx - rank_start_idx}')

        for traj in (pbar2 := tqdm(trajs, leave=False, desc='Process traj...')):
            pbar2.set_postfix(traj=traj)

            save_path = os.path.join(save_dir, f'{traj}.npz')
            if os.path.exists(save_path):
                continue

            try:
                traj_path = os.path.join(load_dir, traj)
                frames = list(sorted(fnmatch.filter(os.listdir(traj_path), 'frame_*.npy')))

                # ! get camera parameters
                points_path = os.path.join(data_dir, 'points', split, traj)
                extrins = torch.eye(4).unsqueeze(dim=0).to(device)

                camera_path = os.path.join(camera_dir, f'{int(traj):05d}.npz')
                if os.path.exists(camera_path):
                    camera_data = np.load(camera_path)
                    aligned_extrin = camera_data['aligned_extrin']  # [n_view, 4, 4]
                    aligned_extrin = torch.from_numpy(aligned_extrin).to(device)
                    aligned_extrin = torch.linalg.inv(aligned_extrin)
                    aligned_extrin[0] = extrins[0]

                    extrins = aligned_extrin

                if os.path.exists(intrin_file := os.path.join(points_path, 'pred_intrinsics.txt')):
                    intrin = np.loadtxt(intrin_file)[0].reshape(3, 3)
                else:
                    raise FileNotFoundError
                    intrin = np.loadtxt('/share/project/cwm/xiuyu.yang/work/dev6/outputs3/demos/rt1/train/00000/points/pred_intrinsics.txt')[0].reshape(3, 3)
                intrin = torch.from_numpy(intrin).float().to(device)

                # intrin[1, 2] = int(intrin[1, 2] + 5)  # FIXME: this is specialized for droid dataset!!!
                W = int(intrin[0, -1] * 2)
                H = int(intrin[1, -1] * 2)
                image_shape = [H, W]

                render_semantics = []
                render_depths = []
                is_labeled = True
                for frame in (pbar3 := tqdm(frames, leave=False, desc='Process frame...')):
                    pbar3.set_postfix(mem=f'{(torch.cuda.memory_allocated() / (1024 ** 3)):.2f}GB')

                    # load occupancy data -> (n, 4)
                    occ_data = np.load(os.path.join(traj_path, frame))
                    occ_data = torch.tensor(occ_data, dtype=torch.int32, device=device) # [x y z label]

                    # get labels3d
                    semantics = semantics_zero.to(device)
                    semantics[occ_data[:, 0], occ_data[:, 1], occ_data[:, 2]] = occ_data[:, -1:].long().to(device)
                    semantics = semantics.reshape(-1, 1)
                    semantics = torch.clamp(semantics, min=0, max=len(colors60) - 1)
                    unique_classes, semantics = torch.unique(semantics, sorted=True, return_inverse=True)
                    feat = torch.nn.functional.one_hot(semantics, num_classes=num_channels_language_feature).float()
                    if len(unique_classes) == 1:
                        is_labeled = False

                    occ_mask = torch.zeros(*occ_shape).bool()
                    occ_mask[occ_data[:, 0], occ_data[:, 1], occ_data[:, 2]] = True
                    occ_mask = occ_mask.reshape(-1)

                    n_view = len(extrins)
                    render_semantics_frame = []
                    render_depths_frame = []
                    for i_view in tqdm(range(n_view), leave=False):

                        extrin = extrins[i_view].float()

                        render_pkg = render(
                            extrin, intrin, image_shape,
                            xyz[occ_mask], rgb[occ_mask], feat[occ_mask], rot[occ_mask], scale[occ_mask], opacity[occ_mask],
                            bg_color=[0, 0, 0]
                        )

                        render_color = render_pkg['render_color']  # (3, H, W)
                        render_semantic = render_pkg['render_feat']  # (N, H, W)
                        render_depth = render_pkg['render_depth']  # 1
                        render_alpha = render_pkg['render_alpha']  # 1

                        # render postprocess
                        none_mask = render_alpha[0] < 0.10
                        none_label = torch.zeros(num_channels_language_feature).cuda()
                        none_label[0] = 1
                        render_semantic[:, none_mask] = none_label[:, None]
                        render_depth[:, none_mask] = 51.2
                        render_depth = torch.clamp(render_depth, min=0.01, max=0.4)  # IMPORTANT!!!
                        # render_depth = torch.clamp(render_depth, min=0.01, max=0.6)  # IMPORTANT!!!

                        # convert feature logits to labels
                        if render_semantic.shape[0] != 1:
                            render_semantic = torch.max(render_semantic, dim=0)[1].squeeze()
                        else:
                            render_semantic = render_semantic.squeeze()

                        # ! convert index_labels back to semantic labels
                        render_semantic = torch.clamp(render_semantic, min=0, max=len(unique_classes) - 1)
                        render_semantic = unique_classes[render_semantic]

                        render_semantics_frame.append(render_semantic.cpu().numpy())
                        render_depths_frame.append(render_depth.cpu().numpy())

                        if save_color:
                            sem_map = apply_semantic_colormap(render_semantic).cpu().permute(1, 2, 0).detach().numpy() * 255
                            sem_map = Image.fromarray(sem_map.astype(np.uint8))

                            depth_map = apply_depth_colormap(render_depth).cpu().permute(1, 2, 0).detach().numpy() * 255
                            depth_map = Image.fromarray(depth_map.astype(np.uint8))

                            W, H = sem_map.size
                            merge = Image.new('RGB', (W * 2, H))
                            merge.paste(sem_map, (0, 0))
                            merge.paste(depth_map, (W, 0))

                            frame_id = frame.removesuffix(f'.npy')
                            os.makedirs(os.path.join(save_dir, traj), exist_ok=True)
                            merge.save(os.path.join(save_dir, traj, f'{frame_id}_{i_view}.png'))

                    render_semantics_frame = np.stack(render_semantics_frame)  # [n_view, h, w]
                    render_depths_frame = np.stack(render_depths_frame)  # [n_view, h, w]

                    render_semantics.append(render_semantics_frame)
                    render_depths.append(render_depths_frame)

                render_semantics = np.stack(render_semantics)  # [n_frame, n_view, h, w]
                render_depths = np.concatenate(render_depths)  # [n_frame, n_view, h, w]

                np.savez_compressed(save_path,
                                    semantics=render_semantics.astype(np.uint8),
                                    depths=render_depths.astype(np.float32),
                                    is_labeled=is_labeled,
                )

            except Exception as e:
                if int(os.getenv('DEBUG', 0)):
                    raise
                CONSOLE.log(f"Failed render {traj} due to {e}")


def reconstruction_multi_task(process_keys, rank: Optional[str] = None):

    CONSOLE.log(f'Process keys: {process_keys}')
    CONSOLE.log(f'Will have {len(process_keys)} processors in total.')
    device_indexs = {key: index for index, key in enumerate(process_keys)}

    is_process = lambda key: key in process_keys

    if rank is not None:
        rank, all_ranks = rank.split('/')
        rank = int(rank)
        all_ranks = int(all_ranks)
    else:
        rank = 0
        all_ranks = 1

    multiprocessing.set_start_method("spawn")
    shared_sparse_pts_path = Queue()
    shared_dense_pts_path = Queue()
    shared_occupancy_pts_path = Queue()
    terminate_process = Event()

    sparse_pts_dir = os.path.join(save_dir, 'points')
    dense_pts_dir = os.path.join(save_dir, 'mesh')
    occupancy_dir = os.path.join(save_dir, 'occ')
    semantics_dir = os.path.join(save_dir, 'semantics')

    for split in tqdm(splits, leave=False):

        os.makedirs(sparse_pts_split_dir := os.path.join(sparse_pts_dir, split), exist_ok=True)
        sparse_points_folers = list(sorted(os.listdir(sparse_pts_split_dir)))
        if is_process('mesh') and process_keys.index('mesh') == 0:

            rank_size = len(sparse_points_folers) // all_ranks
            rank_start_idx = rank * rank_size
            rank_end_idx = (rank + 1) * rank_size if rank + 1 < all_ranks else -1
            sparse_points_folers = sparse_points_folers[rank_start_idx : rank_end_idx]
            CONSOLE.log(f'rank {rank} will host {rank_start_idx=} and {rank_end_idx=}')

        for sparse_points_folder in tqdm(sparse_points_folers, leave=False):
            shared_sparse_pts_path.put(os.path.join(split, sparse_points_folder))

        os.makedirs(dense_pts_split_dir := os.path.join(dense_pts_dir, split), exist_ok=True)
        dense_points_folders = list(sorted(os.listdir(dense_pts_split_dir)))
        if is_process('occupancy') and process_keys.index('occupancy') == 0:

            rank_size = len(sparse_points_folers) // all_ranks
            rank_start_idx = rank * rank_size
            rank_end_idx = (rank + 1) * rank_size if rank + 1 < all_ranks else -1
            dense_points_folders = dense_points_folders[rank_start_idx : rank_end_idx]
            CONSOLE.log(f'rank {rank} will host {rank_start_idx=} and {rank_end_idx=}')

        for dense_points_folder in tqdm(dense_points_folders, leave=False):
            shared_dense_pts_path.put(os.path.join(split, dense_points_folder))

        os.makedirs(occupancy_split_dir := os.path.join(occupancy_dir, split), exist_ok=True)
        occupancy_folders = list(sorted(os.listdir(occupancy_split_dir)))
        if is_process('render') and process_keys.index('render') == 0:

            rank_size = len(sparse_points_folers) // all_ranks
            rank_start_idx = rank * rank_size
            rank_end_idx = (rank + 1) * rank_size if rank + 1 < all_ranks else -1
            occupancy_folders = occupancy_folders[rank_start_idx : rank_end_idx]
            CONSOLE.log(f'rank {rank} will host {rank_start_idx=} and {rank_end_idx=}')

        for occupancy_folder in tqdm(occupancy_folders, leave=False):
            shared_occupancy_pts_path.put(os.path.join(split, occupancy_folder))

    processes = []
    if is_process('points'):
        processes.append(
            Process(target=get_sparse_points,
                    args=(
                                    data_dir,
                                    save_dir,
                                    splits,
                                    shared_sparse_pts_path,
                                    terminate_process,
                                    device_indexs['points'],
                                    args,
                    )),
        )
    if is_process('mesh'):
        processes.append(
            Process(target=get_dense_points,
                    args=(
                                    save_dir,
                                    shared_sparse_pts_path,
                                    shared_dense_pts_path,
                                    terminate_process,
                                    device_indexs['mesh'],
                                    args,
                    )),
        )
        CONSOLE.log(f'Found {shared_sparse_pts_path.qsize()} tasks for dense points.')
    if is_process('occupancy'):
        processes.append(
            Process(target=get_occupancy,
                    args=(
                                    save_dir,
                                    shared_dense_pts_path,
                                    terminate_process,
                                    device_indexs['occupancy'],
                                    args,
                    )),
        )
        CONSOLE.log(f'Found {shared_dense_pts_path.qsize()} tasks for occupancy.')
    if is_process('rendering'):
        processes.append(
            Process(target=get_render,
                    args=(
                                    save_dir,
                                    shared_occupancy_pts_path,
                                    terminate_process,
                                    device_indexs['render'],
                                    args,
                    )),
        )
        CONSOLE.log(f'Found {shared_occupancy_pts_path.qsize()} tasks for render.')

    for i, p in enumerate(processes):
        CONSOLE.print(f"Starting Process {i}...")
        p.start()
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        terminate_process.set()
        for i, p in enumerate(processes):
            CONSOLE.log(f"Terminating process {i}...")
            p.terminate()
        for i, p in enumerate(processes):
            p.join()
            CONSOLE.log(f"Process {i} finished.")


def find_largest_label_id(data_dir):

    splits = ['train', 'va']

    result = -1

    for split in tqdm(splits, leave=False):
        splits_dir = os.path.join(data_dir, split)
        if not os.path.exists(splits_dir):
            continue

        traj_ids = list(sorted(os.listdir(splits_dir)))
        for traj in tqdm(traj_ids, leave=False):

            frame_files = list(sorted(fnmatch.filter(os.listdir(os.path.join(splits_dir, traj)), 'frame_*.npz')))
            for frame in tqdm(frame_files, leave=False):

                try:
                    frame_data = np.load(os.path.join(splits_dir, traj, frame))
                    global_labels = frame_data['annotated_frame_index']
                except Exception as e:
                    continue

                uniuqe_labels = np.unique(global_labels)
                number_of_classes = len(uniuqe_labels)

                result = max(result, number_of_classes)
                CONSOLE.log(f'Current {result=}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store_true')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--start', type=int, default=-1)
    parser.add_argument('--end', type=str, default=-1)
    parser.add_argument('--action', type=str, default=None)
    parser.add_argument('--rank', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    args = parser.parse_args()

    # ------------------- bridgev2-1 --------------------
    # n_view = 1
    # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridge/videos'
    # save_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridge/renderings'

    # ------------------- bridgev2-2 --------------------
    n_view = 3
    data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/videos'
    save_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/renderings'

    # ------------------- droid --------------------
    # n_view = 2
    # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/droid/videos'
    # save_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/droid/renderings'

    # ------------------- rt1 --------------------
    # n_view = 1
    # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/rt1/videos'
    # save_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/rt1/renderings'

    assert args.split in ('train', 'val', 'test'), f'Got invalid split {args.split}'
    splits = [args.split]

    if args.action == 'reconstruction':
        process_keys = [
            # 'points',
            # 'mesh',
            'occupancy',
        ]
        if len(process_keys) > 0:
            reconstruction_multi_task(process_keys, args.rank)

    elif args.action == 'cameras':
        data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/embeddings_full'
        get_cameras(data_dir, save_dir, splits)

    elif args.action == 'align_cameras':
        align_multiview_extrins(save_dir, splits, rank=args.rank)

    elif args.action == 'caption':
        splits = ['train', 'val']
        # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridge/embeddings_full'
        # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/droid/embeddings_full'
        # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/rt1/embeddings_full'
        data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/embeddings_full'
        get_captions(data_dir, save_dir, splits, args.rank)

    elif args.action == 'caption_post_process':
        # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridge/renderings/captions/train/'
        # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/droid/renderings/captions/train/'
        # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/rt1/renderings/captions/train/'
        data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/renderings/captions/'
        postprocess_captions(data_dir)

    elif args.action == 'labeling':
        get_labels(data_dir, save_dir, splits, args.rank)

    elif args.action == 'labels_post_process':
        # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridge/renderings'
        # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/droid/renderings'
        # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/rt1/renderings'
        data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/renderings'
        postprocess_labels(data_dir, splits)

    elif args.action == 'render':
        get_render(save_dir, splits, rank=args.rank)

    # data_dir = '/share/project/cwm/xiuyu.yang/work/dev6/data/bridge/renderings/semantics'
    # find_largest_label_id(data_dir)