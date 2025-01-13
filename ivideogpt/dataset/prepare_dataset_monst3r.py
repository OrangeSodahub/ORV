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


def get_reconstructed_scene(args, model, device, silent, image_size, traj_file, save_folder, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
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
                            num_total_iter=niter, empty_cache=False, batchify=not args.not_batchify)

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
    # points3d = np.stack([np.concatenate([points[i].vertices, points[i].colors[:, :3]], axis=-1)
    #                      for i in range(len(points))])  # [N_img, N_point, 6]
    # np.save(f'{save_folder}/pred_points.npy', points3d)

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
    if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
        motion_mask_thre = 0.35
        error_map = get_dynamic_mask_from_pairviewer(scene, both_directions=True, output_dir=args.output_dir, motion_mask_thre=motion_mask_thre)
        # imgs.append(rgb(error_map))
        # apply threshold on the error map
        normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        error_map_max = normalized_error_map.max()
        error_map = cmap(normalized_error_map/error_map_max)
        imgs.append(rgb(error_map))
        binary_error_map = (normalized_error_map > motion_mask_thre).astype(np.uint8)
        imgs.append(rgb(binary_error_map * 255))

    return scene, outfile, imgs


if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir is not None:
        tmp_path = args.output_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = args.model_name

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    recon_fun = functools.partial(get_reconstructed_scene, args, model, args.device, args.silent, args.image_size)

    data_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/data/robonet_preprocessed'
    save_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xiuyu.yang/work/dev6/data/occ_robonet_monst3r/point'
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
                    new_model_weights=args.weights,
                    temporal_smoothing_weight=0.01,
                    translation_weight='1.0',
                    shared_focal=True,
                    flow_loss_weight=0.01,
                    flow_loss_start_iter=0.1,
                    flow_loss_threshold=25,
                    use_gt_mask=args.use_gt_davis_masks,
                )
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    exit(1)
                continue
