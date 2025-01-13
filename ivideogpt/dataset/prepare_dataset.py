import sys
import torch
import numpy as np
import socket
import signal
import open3d as o3d
import trimesh
import shutil
import fnmatch
import multiprocessing
from multiprocessing import Process, Queue, Event
from tqdm import tqdm
from copy import deepcopy
from rich.console import Console
from numpy import typing as npt
from typing import Callable, Literal
from torch import distributed as dist
from torch import multiprocessing as mp

CONSOLE = Console(width=120)
try:
    import nksr
except Exception as e:
    CONSOLE.print(e)



# def _find_free_port() -> str:
#     """Finds a free port."""
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     sock.bind(("", 0))
#     port = sock.getsockname()[1]
#     sock.close()
#     return port


def depths_to_points(depth_map: npt.NDArray | torch.Tensor,
                     intrin: npt.NDArray | torch.Tensor,
                     mask: npt.NDArray | torch.Tensor=None,
                     rgb_map: npt.NDArray | torch.Tensor=None) -> npt.NDArray | torch.Tensor:
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


def points_to_voxels(points: npt.NDArray | torch.Tensor, voxel_size: list = [0.2, 0.2, 0.2],
                     labels: npt.NDArray | torch.Tensor = None, max_num_points: int = -1,
                     point_cloud_range: npt.NDArray | torch.Tensor | None = None,
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
    max_num_points = 1e2# / voxel_size[0]
    # remove nan
    points = points[~(torch.isnan(points[:, 0]) | torch.isnan(points[:, 1]) | torch.isnan(points[:, 2]))]
    # NOTE: need to add min range to transform voxel to original position
    if point_cloud_range is None:
        point_cloud_range = [points[:, 0].min(), points[:, 1].min(), points[:, 2].min(),
                             points[:, 0].max(), points[:, 1].max(), points[:, 2].max()]

    voxels = voxelization(points, voxel_size, point_cloud_range, int(max_num_points), int(max_voxels), determinstic)
    voxels = [e.cpu() for e in voxels] if not isinstance(voxels, torch.Tensor) else voxels.cpu()

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


import sys
sys.path.append('.')
sys.path.append('thirdparty/monst3r')

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL.ImageOps import exif_transpose

from thirdparty.monst3r.demo import *
from thirdparty.monst3r.dust3r.utils.viz_demo import *
from thirdparty.monst3r.dust3r.utils.image import crop_img, ImgNorm, ToTensor


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
    traj_id = os.path.basename(traj_file).removesuffix('.npz')
    numpy_images = np.load(traj_file)['image']
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


def get_reconstructed_scene(model, device, silent, image_size, traj_file, save_folder, schedule, niter, min_conf_thr,
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
                      device: torch.device,
                      shared_sparse_pts_path: Queue,
                      terminate_process: Event) -> None:

    def _handle_terminate(signum, frame):
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        CONSOLE.print(f"[on yellow]Step1 Deleted[/] [blue]{split}/{traj_id}[/]")
        terminate_process.set()
        sys.exit()

    signal.signal(signal.SIGTERM, _handle_terminate)

    # default arguments for monst3r
    silent = True
    server_name = '127.0.0.1'
    image_size = 512  # choose from [512, 224]
    use_gt_davis_masks = False
    not_batchify = False

    weights_path = 'checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth'
    if not os.path.exists(weights_path):
        weights_path = 'Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt'
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
    recon_fun = functools.partial(get_reconstructed_scene, model, device, silent, image_size)

    for split in tqdm(splits, "Processing split"):
        split_dir = os.path.join(data_dir, split)
        traj_files = os.listdir(split_dir)
        for traj_file in (
            pbar := tqdm(traj_files)
        ):
            traj_id = traj_file.removesuffix('.npz')
            save_folder = os.path.join(save_dir, 'points', split, traj_id)
            pbar.set_description(f"Processing {traj_id}")
        
            if os.path.exists(save_folder) and len(os.listdir(save_folder)) != 0:
                CONSOLE.print(f"[on blue]Step1[/] Skipped [blue]{split}/{traj_id}[/]")
                continue

            try:
                # Call the function with default parameters
                scene, outfile, imgs = recon_fun(
                    traj_file=os.path.join(split_dir, traj_file),
                    save_folder=save_folder,
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
                CONSOLE.print(f"[on red]Step1[/] Failed [blue]{split}/{traj_id}[/] due to [red]{e}[/]")
                continue

    terminate_process.set()


def process_nksr(point_cloud_data, save_path, max_nn: int=20):

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

    device = torch.device("cuda:0")
    reconstructor = nksr.Reconstructor(device)

    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(point_cloud_data.vertices)
    with_normal = _preprocess_point_cloud(point_cloud_original, max_nn=max_nn)
    
    input_xyz = torch.from_numpy(np.asarray(with_normal.points)).to(device).float()
    input_normal = torch.from_numpy(np.asarray(with_normal.normals)).to(device).float()

    # Note that input_xyz and input_normal are torch tensors of shape [N, 3] and [N, 3] respectively.
    field = reconstructor.reconstruct(input_xyz, input_normal)
    mesh = field.extract_dual_mesh(mise_iter=2)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.v.cpu().numpy())
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.f.cpu().numpy())
    o3d_mesh.paint_uniform_color((0, 1, 1))
    o3d.io.write_triangle_mesh(save_path, o3d_mesh)


def get_dense_points(data_dir: str,
                     device: torch.device,
                     shared_sparse_pts_path: Queue,
                     shared_dense_pts_path: Queue,
                     terminate_process: Event) -> None:

    def _handle_terminate(signum, frame):
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        CONSOLE.print(f"[on yellow]Step2 Deleted[/] [blue]{traj_path}[/]")
        sys.exit()

    signal.signal(signal.SIGTERM, _handle_terminate)

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
                points_data = trimesh.load(os.path.join(load_dir, points_file))
                save_path = os.path.join(save_folder, points_file.replace('.ply', '_nksr.ply'))
                process_nksr(points_data, save_path)

            # add path to buffer
            shared_dense_pts_path.put(traj_path)

        except Exception as e:
            CONSOLE.print(f"[on red]Step2[/] Failed [blue]{traj_path}[/] due to [red]{e}[/]")
            continue


def get_occupancy(data_dir: str,
                  device: torch.device,
                  shared_dense_pts_path: Queue,
                  terminate_process: Event):

    def _handle_terminate(signum, frame):
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        CONSOLE.print(f"[on yellow]Step3 Deleted[/] [blue]{traj_path}[/]")
        sys.exit()

    signal.signal(signal.SIGTERM, _handle_terminate)

    def _pose_to_transform(pose):
        OPENGL = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

        c2w = np.eye(4)
        xyz, qwxyz = pose[:3], pose[3:]
        c2w[:3, -1] = xyz
        c2w[:3, :3] = Rotation.from_quat(qwxyz[[1, 2, 3, 0]]).as_matrix()

        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        transform = np.linalg.inv(c2w @ OPENGL @ rot)

        transform = torch.tensor(transform, device=device, dtype=torch.float32)
        return transform

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

        if os.path.exists(save_folder) and len(os.listdir(save_folder)) != 0:
            CONSOLE.print(f"[on blue]Step3[/] Skipped [blue]{traj_path}[/]")
            continue

        os.makedirs(save_folder, exist_ok=True)
        try:
            point_cloud_range = [-0.2, -0.2, 0, 0.2, 0.2, 0.4]
            voxel_size = [0.005] * 3
            mesh_files = list(sorted(fnmatch.filter(os.listdir(load_dir), 'frame_*_nksr.ply')))
            poses = np.loadtxt(os.path.join(points_dir, 'pred_traj.txt'))
            for mesh_file, pose in zip(mesh_files, poses):
                transform = _pose_to_transform(pose[1:])
                mesh = trimesh.load(os.path.join(load_dir, mesh_file))
                points = torch.tensor(mesh.vertices, device=device, dtype=torch.float32)
                points = torch.concat([points, torch.ones_like(points[:, -1:])], dim=-1)
                points = (transform @ points.T).T
                voxels = points_to_voxels(points,
                                        voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        device=device)
                np.save(os.path.join(save_folder, mesh_file.replace('_nksr.ply', '.npy')), voxels)
                # mesh.vertices = points[:, :3].cpu().numpy()
                # mesh.export(os.path.join(save_folder, mesh_file.replace('_nksr.ply', '_transformed.ply')))

        except Exception as e:
            CONSOLE.print(f"[on red]Step3[/] Failed [blue]{traj_path}[/] due to [red]{e}[/]")
            continue


# def _distributed_worker(
#     local_rank: int,
#     main_func: Callable,
#     world_size: int,
#     nerf_ranks: list,
#     guide_ranks: list,
#     num_devices_per_machine: int,
#     machine_rank: int,
#     dist_url: str,
#     device_type: Literal["cpu", "cuda", "mps"] = "cuda",
#     ranks: dict[str, list] = None,
#     events: dict[str, mp.Event] = None,
#     buffers: dict[str, mp.Queue] = None,
# ) -> None:
#     assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
#     global_rank = machine_rank * num_devices_per_machine + local_rank

#     dist.init_process_group(
#         backend="nccl" if device_type == "cuda" else "gloo",
#         init_method=dist_url,
#         world_size=world_size,
#         rank=global_rank,
#     )
#     assert comms.LOCAL_PROCESS_GROUP is None
#     # NOTE: here we force the single machine !!!
#     # and `ranks` is for each ranks_group, e.g. `nerf_ranks`, `guide_ranks`.
#     if local_rank in nerf_ranks:
#         group_ranks = nerf_ranks
#     elif local_rank in guide_ranks:
#         group_ranks = guide_ranks
#     else:
#         raise RuntimeError(f"Local rank = {local_rank} not found on current machine, check `num_devices_per_machine`!!")

#     comms.LOCAL_PROCESS_GROUP = dist.new_group(group_ranks)

#     assert num_devices_per_machine <= torch.cuda.device_count()
#     output = main_func(local_rank, world_size, config, global_rank, ranks, events, buffers)

#     comms.synchronize()
#     dist.destroy_process_group()


# def multi_launch(
#     main_func: Callable,
#     world_size: int = 1,
#     machine_rank: int = 0,
#     dist_url: str = "auto",
#     device_type: Literal['cpu', 'cuda', 'mps'] = 'cuda',
# ) -> None:

#     if world_size == 0:
#         raise ValueError("world_size cannot be 0")
#     elif world_size == 1:
#         raise RuntimeError(f"Only support multi-gpu training.")
#         # uses one process
#         try:
#             main_func(local_rank=0, world_size=world_size, config=config)
#         except KeyboardInterrupt:
#             # print the stack trace
#             CONSOLE.print(traceback.format_exc())
#         finally:
#             profiler.flush_profiler(config.logging)

#     elif world_size > 1:
#         # Using multiple gpus with multiple processes.
#         if dist_url == "auto":
#             port = _find_free_port()
#             dist_url = f"tcp://127.0.0.1:{port}"

#         # NOTE: determin the group ranks
#         all_ranks = [i for i in range(world_size)] # all processors
#         nerf_ranks = set(config.pipeline.nerf_ranks) if not config.pipeline.nerf_single_device else set([all_ranks[-1]])
#         guide_ranks = set(all_ranks) - nerf_ranks
#         nerf_ranks = set(all_ranks) - guide_ranks
#         if len(nerf_ranks) == 0:
#             raise RuntimeError(f"Nerf Model has no ranks! Check your devices and settings of `nerf_ranks` in `pipeline` and `torch.cuda.device_count()`.")
#         if len(guide_ranks) == 0:
#             raise RuntimeError(f"Guide pipeline has no ranks! Check your devices and settings of `nerf_ranks` in `pipeline` and `torch.cuda.device.count()`.")
#         ranks = dict(nerf_ranks=nerf_ranks, guide_ranks=guide_ranks)

#         # create communication tools
#         nerf_trainer = mp.get_context("spawn").Event()
#         call_renderer = mp.get_context("spawn").Event()
#         base_buffers = [mp.get_context("spawn").Queue() for _ in range(len(guide_ranks))]
#         guide_buffer = mp.get_context("spawn").Queue()
#         render_indice_buffers = [mp.get_context("spawn").Queue() for _ in range(len(guide_ranks))]
#         step_buffers = dict(nerf_step=mp.get_context("spawn").Value("i"), diffusion_step=mp.get_context("spawn").Value("i"))
#         events = dict(nerf_trainer=nerf_trainer, call_renderer=call_renderer)
#         buffers = dict(base_buffers=base_buffers, guide_buffer=guide_buffer, render_indice_buffers=render_indice_buffers, step_buffers=step_buffers)
#         process_context = mp.spawn(
#             _distributed_worker,
#             nprocs=world_size,
#             join=False,
#             args=(main_func, world_size, list(nerf_ranks), list(guide_ranks), world_size,
#                   machine_rank, dist_url, config, timeout, device_type, ranks, events, buffers),
#         )
#         # process_context won't be None because join=False, so it's okay to assert this
#         # for Pylance reasons
#         assert process_context is not None
#         try:
#             process_context.join()
#         except KeyboardInterrupt:
#             for i, process in enumerate(process_context.processes):
#                 if process.is_alive():
#                     CONSOLE.log(f"Terminating process {i}...")
#                     process.terminate()
#                 process.join()
#                 CONSOLE.log(f"Process {i} finished.")


def single_launch():

    multiprocessing.set_start_method("spawn")
    shared_sparse_pts_path = Queue()
    shared_dense_pts_path = Queue()
    terminate_process = Event()

    sparse_pts_dir = os.path.join(save_dir, 'points')
    dense_pts_dir = os.path.join(save_dir, 'mesh')
    occupancy_dir = os.path.join(save_dir, 'occ')

    for split in tqdm(splits, leave=False):

        os.makedirs(sparse_pts_split_dir := os.path.join(sparse_pts_dir, split), exist_ok=True)
        sparse_points_folers = os.listdir(sparse_pts_split_dir)
        for sparse_points_folder in tqdm(sparse_points_folers, leave=False):
            shared_sparse_pts_path.put(os.path.join(split, sparse_points_folder))

        os.makedirs(dense_pts_split_dir := os.path.join(dense_pts_dir, split), exist_ok=True)
        dense_points_folders = os.listdir(dense_pts_split_dir)
        for dense_points_folder in tqdm(dense_points_folders, leave=False):
            shared_dense_pts_path.put(os.path.join(split, dense_points_folder))

        os.makedirs(os.path.join(occupancy_dir, split), exist_ok=True)

    processes = [
        Process(target=get_sparse_points, args=(
            data_dir, save_dir, splits, torch.device('cuda:0'), shared_sparse_pts_path, terminate_process)),
        Process(target=get_dense_points, args=(
            save_dir, torch.device('cuda:1'), shared_sparse_pts_path, shared_dense_pts_path, terminate_process)),
        Process(target=get_occupancy, args=(
            save_dir, torch.device('cuda:2'), shared_dense_pts_path, terminate_process)),
    ]
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


if __name__ == '__main__':

    data_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/data/robonet_preprocessed'
    save_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/data/occ_robonet_monst3r'
    splits = ['train', 'test']

    single_launch()