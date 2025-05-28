import sys

sys.path.append('/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/ivideogpt/dataset')
import os
import torch
import argparse
import open3d as o3d
import numpy as np
import fnmatch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from prepare_dataset import CONSOLE, process_nksr
from prepare_dataset import project_3d_to_2d, generate_colors, points_to_voxels
from thirdparty.grounded_sam_2.sam2.build_sam import build_sam2
from thirdparty.grounded_sam_2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# monst3r
try:
    from thirdparty.monst3r.demo import *
    from thirdparty.monst3r.dust3r.utils.viz_demo import *
    from prepare_dataset import (
        crop_img, ImgNorm, ToTensor, rgb,
        get_reconstructed_scene, get_reconstructed_scene_realtime,
    )
except Exception as e:
    CONSOLE.log(f'[bold red]Import necessary packages form Monst3R failed! {e}')

try:
    import nksr
except Exception as e:
    CONSOLE.log(f'[bold red]Import necessary packages form NKSR failed! {e}')


# grounded sam2
try:
    from prepare_dataset import (
        sv, resolve_color, Color, ColorPalette, AutoProcessor, AutoModelForZeroShotObjectDetection,
        build_sam2_video_predictor, build_sam2, SAM2ImagePredictor, SAM2VideoPredictor, sample_points_from_masks,
    )
except Exception as e:
    CONSOLE.log(f'[bold red]Import necessary packages form GoundedSAM2 failed! {e}')

# gaussian render
try:
    from prepare_dataset import render, create_full_center_coords, apply_depth_colormap
except Exception as e:
    CONSOLE.log(f'[bold red]Import necessary packages form GS failed! {e}')


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip
decord.bridge.set_bridge("torch")
from decord import VideoReader, cpu


colors = ColorPalette.DEFAULT


def generate_sparse_points():

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

    traj_file = os.path.join(data_root, dataset, 'videos', split, str(traj_id))

    if os.path.exists(points_path) and len(os.listdir(points_path)) != 0:
        CONSOLE.print(f"[on blue]Step1[/] Skipped [blue]{split}/{traj_id}[/]")
        return

    save_path = points_path

    if args.realtime:

        # Call the function with default parameters
        outfile = recon_fun(
            traj_file=traj_file,
            save_folder=save_path,
            scenegraph_type='oneref_mid',
            refid=0,
            batch_size=args.batch_size,
        )

    else:

        scene, outfile, imgs = recon_fun(
            traj_file=traj_file,
            save_folder=save_path,
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

    CONSOLE.log(f'[bold yellow]Sparse points saved to {save_path}')


def generate_dense_points():

    os.makedirs(mesh_path, exist_ok=True)

    points_file = os.path.join(points_path, f'frame_{frame_ids:04d}.ply')
    if not os.path.exists(points_file):
        CONSOLE.log(f'[bold red]Not found points file {points_file}! Will generate points first!')
        generate_sparse_points()

    def _preprocess_points(points_data):

        points = np.asarray(points_data.points)
        colors = np.asarray(points_data.colors)
        mask = points[:, 2] < .4
        points = points[mask]
        colors = colors[mask]

        points_data = o3d.geometry.PointCloud()
        points_data.points = o3d.utility.Vector3dVector(points)
        points_data.colors = o3d.utility.Vector3dVector(points)

        cl, ind = points_data.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.)
        points_data = points_data.select_by_index(ind)

        return points_data

    reconstructor = nksr.Reconstructor(device)

    points_data = o3d.io.read_point_cloud(points_file)
    points_data = _preprocess_points(points_data)

    mesh = process_nksr(reconstructor, points_data, device=device)

    save_path = os.path.join(mesh_path, f'frame_{frame_ids:04d}_nksr.ply')
    o3d.io.write_triangle_mesh(save_path, mesh)
    CONSOLE.log(f'[bold yellow]Mesh saved to {save_path}')


def generate_occupancy():

    os.makedirs(occupancy_path, exist_ok=True)

    mesh_file = os.path.join(mesh_path, f'frame_{frame_ids:04d}_nksr.ply')
    if not os.path.exists(mesh_file):
        CONSOLE.log(f'[bold red]Not found mesh file {mesh_file}! Will generate mesh file first!')
        generate_dense_points()

    intrin_ = np.loadtxt(os.path.join(points_path, 'pred_intrinsics.txt'))[0].reshape(3, 3)
    intrin_ = torch.from_numpy(intrin_).float().to(device)
    intrin = torch.eye(4).float().to(device)
    intrin[:3, :3] = intrin_

    labels2d_size = (480, 640)
    points3d_size = (384, 512)
    src_h = points3d_size[0]
    tgt_h = labels2d_size[0]
    scale = tgt_h / src_h
    intrin[:2, :3] = intrin[:2, :3] * scale

    extrin = torch.eye(4).float().to(device)

    mesh = o3d.io.read_point_cloud(mesh_file)
    points = torch.tensor(np.asarray(mesh.points), device=device, dtype=torch.float32)

    # find labels annotations
    label_file = os.path.join(label_path, f'frame_{frame_ids:04d}.npz')
    try:
        labels2d = np.load(label_file)['mask2d']  # (h, w)
        labels2d = torch.from_numpy(labels2d).long().to(device)
        labels2d[labels2d == 255] = -1  # the original labels2d are in uint8 data foramt!

        points2d = project_3d_to_2d(points, extrin=extrin, intrin=intrin)[:, :2].long()  # (n, 3)
        masks2d = (points2d[:, 0] >= 0) & (points2d[:, 0] < 640) & (points2d[:, 1] >= 0) & (points2d[:, 1] < 480)
        points2d = points2d[masks2d]  # to avoid points lie outside the image

        labels3d = torch.zeros((points.shape[0],)).long().to(device)
        labels3d[masks2d] = labels2d[points2d[:, 1], points2d[:, 0]]
        # labels3d[labels3d == -1] = len(colors60) - 1  # labels must be positive when input to voxelization function
        labels3d[labels3d == -1] = 0

    except:
        CONSOLE.log(f'[bold red]Failed to load semantic labels for {mesh_file}: {label_file}!')
        labels3d = None
        raise

    # ! configurations for voxelization!
    point_cloud_range = [-0.2, -0.2, 0, 0.2, 0.2, 0.4]
    voxel_size = [0.001] * 3  # TODO

    points = torch.concat([points, torch.ones_like(points[:, -1:])], dim=-1)
    voxels = points_to_voxels(
                    points,
                    voxel_size=voxel_size,
                    labels=labels3d,
                    point_cloud_range=point_cloud_range,
                    device=device,
            )

    label_indices = voxels[:, -1] % len(colors.colors)
    colors_array = np.array(list(map(lambda color: np.array(color.as_rgb()), colors.colors)))  # -> (n, 3)
    voxels_color = colors_array[label_indices.astype(np.uint8)]  # -> [N, 3]
    voxels = np.concatenate([voxels, voxels_color], axis=-1)  # -> [N, 7]

    if labels3d is not None:
        CONSOLE.log(f'Get unique labels for occupancy: {np.bincount(voxels[:, 3].astype(np.uint8)).nonzero()[0]}')

    save_path = os.path.join(occupancy_path, f'frame_{frame_ids:04d}.npy')
    np.save(save_path, voxels)
    CONSOLE.log(f'[bold yellow]Occupancy saved to {save_path}')


def generate_labels():

    os.makedirs(label_path, exist_ok=True)

    # init sam image predictor and video predictor model
    sam2_checkpoint = "thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    traj_file = os.path.join(data_root, dataset, 'videos', split, str(traj_id))
    inference_state = video_predictor.init_state(
        video_path=os.path.join(traj_file, 'rgb.mp4')
    )

    # load videos
    video_reader = decord.VideoReader(
        uri=os.path.join(traj_file, 'rgb.mp4'), num_threads=2)

    frames = video_reader.get_batch(range(len(video_reader))).float()
    frames = frames.permute(0, 3, 1, 2).contiguous()  # -> [f, c, h, w]
    frames = frames.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # -> [f, h, w, c]

    start_frame_index = 16
    init_frame = frames[start_frame_index]
    H, W = init_frame.shape[:2]

    ann_frame_idx = start_frame_index  # the frame index we interact with

    if False:
        # prompt SAM 2 image predictor to get the mask for the object
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(sam2)
        # mask_generator = SAM2AutomaticMaskGenerator(
        #     model=sam2,
        #     points_per_side=64,
        #     points_per_batch=128,
        #     pred_iou_thresh=0.7,
        #     stability_score_thresh=0.92,
        #     stability_score_offset=0.7,
        #     crop_n_layers=1,
        #     box_nms_thresh=0.7,
        #     crop_n_points_downscale_factor=2,
        #     min_mask_region_area=25.0,
        #     use_m2m=True,
        # )
        _masks = mask_generator.generate(init_frame)

        masks = []  # -> (n, H, W)
        for _mask in _masks:
            masks.append(_mask['segmentation'])
        masks = np.stack(masks)

        # add masks
        for object_id, mask in enumerate(masks, start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask,
            )

    else:

        init_frame = Image.fromarray(init_frame)

        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        image_predictor = SAM2ImagePredictor(sam2_image_model)

        # init grounding dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-base"

        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        # text = 'black robot gripper, countertop, towel, bowl, knife, orange shrimp.'
        # text = 'black robot gripper, countertop, spoon, fork, towel, pan, jar.'
        text = 'black robot gripper, countertop, spoon, orange plastic bowel, jar, green toy.'
        traj_labels = text.strip('.').split(', ')
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

        # process the detection results
        input_boxes = results[0]["boxes"].cpu().numpy()
        OBJECTS = results[0]["labels"]  # NOTE this term may contains repeations

        valid_indices = []
        for i, object in enumerate(OBJECTS):
            if object in traj_labels:
                valid_indices.append(i)
        input_boxes = np.array([input_boxes[i] for i in valid_indices])
        OBJECTS = [OBJECTS[i] for i in valid_indices]

        # ! map object labels back to global id
        global_ids = np.array([traj_labels.index(object) + 1 for object in OBJECTS]).astype(np.uint8)
        OBJECTS_TO_GLOBA_IDS = {object: global_id for object, global_id in zip(OBJECTS, global_ids)}

        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(np.array(init_frame.convert("RGB")))

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

        PROMPT_TYPE_FOR_VIDEO = "box" # or "point"

        assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if PROMPT_TYPE_FOR_VIDEO == "mask":
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

    # SAM2 tracking
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    result_frames = []
    for frame_idx, segments in video_segments.items():
        img = frames[frame_idx]

        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
       
        mask2d = np.zeros((H, W), dtype=np.int32) - 1
        for index, mask in enumerate(masks):
            mask2d[mask] = global_ids[index]

        # save segment results
        np.savez(
            os.path.join(label_path, f'frame_{frame_idx:04d}.npz'),
            masks=masks.astype(np.bool_),
            mask2d=mask2d.astype(np.uint8),
            object_ids=np.array(object_ids).astype(np.uint8),
        )

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
        label_annotator = sv.LabelAnnotator()
        labels = [f'{OBJECTS_TO_GLOBA_IDS[object]}:{object}' for object in [ID_TO_OBJECTS[i] for i in object_ids]]  # global_id : object_string
        empty_frame = label_annotator.annotate(empty_frame, detections=detections, labels=labels)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        mask_annotator = sv.MaskAnnotator()
        empty_frame = mask_annotator.annotate(scene=empty_frame, detections=detections)
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        # ! save results
        empty_frame = Image.fromarray(empty_frame)
        annotated_frame = Image.fromarray(annotated_frame)

        W, H = empty_frame.size
        merge_frame = Image.new('RGB', (W * 2, H))
        merge_frame.paste(empty_frame, (0, 0))
        merge_frame.paste(annotated_frame, (W, 0))
        merge_frame.save(os.path.join(label_path, f'annotated_frame_{frame_idx:05d}.png'))

        result_frames.append(merge_frame)

    gif_path = os.path.join(label_path, 'result.gif')
    result_frames[0].save(gif_path, save_all=True, append_images=result_frames[1:], duration=100, loop=0)
    CONSOLE.log(f'[bold yellow]Tracking results saved to {gif_path}!')


def generate_renderings():

    os.makedirs(render_path, exist_ok=True)

    def apply_semantic_colormap(semantic):

        max_label = semantic.max()

        x = torch.zeros((3, semantic.shape[0], semantic.shape[1]), dtype=torch.float)
        # for i in range(max_label + 1):
        #     x[0][semantic == i] = colors60[i][0]
        #     x[1][semantic == i] = colors60[i][1]
        #     x[2][semantic == i] = colors60[i][2]

        CONSOLE.log(f'{max_label=}')
        for i in range(max_label + 1):
            print(i, colors.by_idx(i))
            x[0][semantic == i] = colors.by_idx(i).as_rgb()[0]
            x[1][semantic == i] = colors.by_idx(i).as_rgb()[1]
            x[2][semantic == i] = colors.by_idx(i).as_rgb()[2]

        # i = 0
        # x[0][semantic == i] = colors.by_idx(i).as_rgb()[0]
        # x[1][semantic == i] = colors.by_idx(i).as_rgb()[1]
        # x[2][semantic == i] = colors.by_idx(i).as_rgb()[2]

        return x / 255.0

    if not os.path.exists(os.path.join(occupancy_path, f'frame_{frame_ids:04d}.npy')):
        CONSOLE.log(f'[bold red]Not found occupancy data for {frame_ids=}!')
        generate_occupancy()

    num_channels_language_feature = 12

    # ! must be aligned with occupancy!!!!
    point_cloud_range = [-0.2, -0.2, 0, 0.2, 0.2, 0.4]
    voxel_size = [0.001] * 3

    base_scale = 0.00025
    exp_scale = 3.7

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

    frames = list(sorted(fnmatch.filter(os.listdir(occupancy_path), 'frame_*.npy')))

    # ! get camera parameters
    extrin = torch.eye(4).to(device)
    intrin = torch.from_numpy(np.loadtxt(os.path.join(points_path, 'pred_intrinsics.txt')))[0].reshape(3, 3).to(device)
    W = int(intrin[0, -1] * 2)
    H = int(intrin[1, -1] * 2)
    image_shape = [H, W]

    render_results = []
    for frame in (pbar3 := tqdm(frames, leave=False, desc='Process frame...')):
        pbar3.set_postfix(mem=f'{(torch.cuda.memory_allocated() / (1024 ** 3)):.2f}GB')

        # load occupancy data -> (n, 4)
        occ_data = np.load(os.path.join(occupancy_path, frame))
        occ_data = torch.tensor(occ_data, dtype=torch.int32, device=device) # [x y z label r g b]
        occ_data = occ_data[:, :4]

        # get labels3d
        semantics = semantics_zero.to(device)
        semantics[occ_data[:, 0], occ_data[:, 1], occ_data[:, 2]] = occ_data[:, -1:].long().to(device)
        semantics = semantics.reshape(-1, 1)
        semantics = torch.clamp(semantics, min=0, max=len(colors.colors) - 1)
        unique_classes, semantics = torch.unique(semantics, sorted=True, return_inverse=True)
        feat = torch.nn.functional.one_hot(semantics, num_classes=num_channels_language_feature).float()
        if len(unique_classes) == 1:
            is_labeled = False

        occ_mask = torch.zeros(*occ_shape).bool()
        occ_mask[occ_data[:, 0], occ_data[:, 1], occ_data[:, 2]] = True
        occ_mask = occ_mask.reshape(-1)

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

        # convert feature logits to labels
        if render_semantic.shape[0] != 1:
            render_semantic = torch.max(render_semantic, dim=0)[1].squeeze()
        else:
            render_semantic = render_semantic.squeeze()

        # ! convert index_labels back to semantic labels
        render_semantic = torch.clamp(render_semantic, min=0, max=len(unique_classes) - 1)
        render_semantic = unique_classes[render_semantic]

        # ! save render results
        sem_map = apply_semantic_colormap(render_semantic).cpu().permute(1, 2, 0).detach().numpy() * 255
        sem_map = Image.fromarray(sem_map.astype(np.uint8))

        depth_map = apply_depth_colormap(render_depth).cpu().permute(1, 2, 0).detach().numpy() * 255
        depth_map = Image.fromarray(depth_map.astype(np.uint8))

        W, H = sem_map.size
        merge = Image.new('RGB', (W * 2, H))
        merge.paste(sem_map, (0, 0))
        merge.paste(depth_map, (W, 0))

        sem_map.save(os.path.join(render_path, frame.replace('.npy', '_label.png')))
        depth_map.save(os.path.join(render_path, frame.replace('.npy', '_depth.png')))

        merge.save(os.path.join(render_path, frame.replace('.npy', '.png')))
        render_results.append(merge)

    gif_path = os.path.join(render_path, 'result.gif')
    render_results[0].save(gif_path, save_all=True, append_images=render_results[1:], duration=100, loop=0)
    CONSOLE.log(f'[bold yellow]Rendering results saved to {gif_path}!')


def extract_mp4_to_images():

    video_transforms = transforms.Compose(
        [
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(tuple([320, 480])),
        ]
    )
    n_view = 1


    mp4_paths = [
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00039_16_17.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/bridge/embeddings_full/val/videos/00039_16_17.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00381_16_17.mp4'
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00777_16_17.mp4'
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00226_216_29_0.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/droid/embeddings_full/val/videos/00226_216_29_0.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/data/bridge/embeddings_full/val/videos/00667_16_17.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/dev6_2/IRASim/results/05/07/val_bridge_frame_ada-debug006/checkpoints/0300000/val_sample_videos/39_0_16.mp4'
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/dev6_2/IRASim/results/05/07/val_bridge_frame_ada-debug006/checkpoints/0300000/val_sample_videos/14_0_16.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/bridge/embeddings_full/val/videos/00014_16_17.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00014_16_17.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_01334_16_17.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_02295_16_17.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00426_00_29_0.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge2_traj-image_480-320_multiview_20k/eval_00032_16_17.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs/eval_cirasim_droid_traj-image_384-256_multiview_30k/eval_20112_00_29.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs/eval_cirasim_bridge_traj-image_480-320_finetune_2b_30k/eval_00667_16_17.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge2_traj-image_480-320_multiview_20k/eval_00059_16_17.mp4',
        # '/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs2/eval_cirasim_bridge2_traj-image_480-320_multiview_20k/eval_00079_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image_320-480_finetune_2b_30k/eval_30130_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/rt1/embeddings_full/val/videos/30130_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/dev6_2/IRASim/results/02/28/val_bridge_frame_ada-debug/checkpoints/0300000/val_sample_videos/447_0_16.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_bridge_traj-image-condGT_480-320_finetune_2b_20k/eval_00447_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data_old/bridge/embeddings_320_480_sliced_full/val/videos/00447_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/dev6_2/IRASim/results/02/28/val_bridge_frame_ada-debug/checkpoints/0300000/val_sample_videos/447_0_16.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/dev6_2/IRASim/results/02/28/val_bridge_frame_ada-debug/checkpoints/0300000/val_sample_videos/323_0_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_bridge_traj-image-condGT_480-320_finetune_2b_20k/eval_00323_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data_old/bridge/embeddings_320_480_sliced_full/val/videos/00323_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/dev6_2/IRASim/results/02/28/val_bridge_frame_ada-debug/checkpoints/0300000/val_sample_videos/323_0_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs/demos/demo_00005_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs/demos/demo_00006_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_multiview_30k/eval_02676_00_29.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs/demos/demo_00005_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/train_results/CIRASIM3/results/02/28/val_bridge_frame_ada_control-debug/checkpoints/0282000/val_sample_videos/82_0_16.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_bridge_traj-image-condGT_480-320_finetune_2b_20k/eval_00082_16_17.mp4'
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data_old/bridge/embeddings_320_480_sliced_full/val/videos/00082_16_17.mp4'
        # '/share/project/cwm/xiuyu.yang/work/dev6/data/bridgev2/embeddings_full/train/videos/00010_00_17_2.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_bridge2_traj-image_480-320_multiview_condfull_20k/eval_00752_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/bridgev2/embeddings_full/val/videos/00752_00_17_2.mp4'
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge2_traj-image_480-320_multiview_20k/eval_00752_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge2_traj-image_480-320_multiview_20k/eval_00534_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_bridge2_traj-image_480-320_multiview_condfull_20k/eval_00534_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/bridgev2/embeddings_full/val/videos/00534_16_17_2.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_bridge_traj-image_480-320_finetune_2b_30k/eval_00295_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-label_480-320_finetune_2b_20k/eval_00295_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data_old/bridge/embeddings_320_480_sliced_full/val/videos/00295_16_17.mp4',
        '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00145_00_29_0.mp4',
    ]

    # mp4_paths = [
        # bridge1
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00002_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00002_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00004_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00004_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00037_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00039_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00089_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00092_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00108_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00253_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00430_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_00457_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_03416_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_03167_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_02756_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_02722_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_02702_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_02575_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_02231_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_02227_32_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_02224_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_02056_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_01979_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_01844_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_01731_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_01544_32_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_01544_16_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old2/eval_cirasim_bridge_traj-image-condfull_480-320_finetune_2b_20k/eval_01372_16_17.mp4',
        # droid
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00031_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00039_648_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00226_216_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00261_1080_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00267_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00272_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00307_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00309_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00330_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00360_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00062_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00128_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00426_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00602_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00611_1080_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00611_1728_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00641_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00721_216_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_00730_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_01509_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_01615_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_02092_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_02102_216_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_02256_432_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_02676_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_03289_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_03435_00_29_0.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs_old/eval_cirasim_droid_traj-image_384-256_finetune_2b_30k/eval_06004_216_29_0.mp4',
        # rt1
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_00025_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_00422_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_00690_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_00818_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_00825_32_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_02218_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_02883_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_03515_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_03557_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_03603_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_03984_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_04189_32_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_05655_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_06003_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_06218_32_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_07494_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_08257_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_10296_32_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_11087_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_11950_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_12865_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_13890_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_15637_32_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_17548_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_20816_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_21690_32_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_23895_00_17.mp4',
        # '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/outputs3/eval_cirasim_rt1_traj-image-condfull_480-320_finetune_2b_20k/eval_67370_00_17.mp4',
    # ]
    bridge1_gt_folder = '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data_old/bridge/embeddings_320_480_sliced_full/val/videos/'
    # droid_gt_folder = '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data_old/droid/embeddings_320_480_sliced_full/val/videos/'
    # rt1_gt_folder = '/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/rt1/embeddings_320_480_sliced_full/val/videos/'

    for mp4_path in tqdm(mp4_paths):

        episode_id = os.path.basename(mp4_path).lstrip('eval_')
        # mp4_path = os.path.join(bridge1_gt_folder, episode_id)
        # mp4_path = os.path.join(droid_gt_folder, episode_id)
        # mp4_path = os.path.join(rt1_gt_folder, episode_id)

        if not os.path.exists(mp4_path):
            CONSOLE.log(f'[bold red]Not found video {mp4_path}!')
            continue

        # load videos
        video_reader = decord.VideoReader(uri=mp4_path, num_threads=2)
        frames = video_reader.get_batch(range(len(video_reader))).float()
        frames = video_transforms(
            frames.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        frames = frames.numpy().astype(np.uint8)  # -> [f, h, w, c]

        video_path = Path(mp4_path)
        save_folder = Path('outputs2/demos2_gt/') / f'{video_path.parent.name}_{video_path.stem}'
        save_folder.mkdir(parents=True, exist_ok=True)

        for i in range(0, len(frames), 2):

            frame = frames[i]

            frame_image = Image.fromarray(frame)
            if n_view > 1:
                W, H = frame_image.size
                _W = W // n_view
                frame_image_views = Image.new('RGB', (_W, H * n_view))
                for j in range(n_view):
                    frame_image_view = frame_image.crop((_W * j, 0, _W * (j + 1), H))
                    frame_image_views.paste(frame_image_view, (0, H * j, _W, H * (j + 1)))
                # frame_image_views = Image.new('RGB', (_W, H * 2))
                # frame_image_views.paste(
                #     frame_image.crop((0, 0, _W, H)), (0, 0, _W, H)
                # )
                # frame_image_views.paste(
                #     frame_image.crop((_W * 2, 0, _W * 3, H)), (0, H, _W, H * 2)
                # )
                frame_image = frame_image_views

            frame_path = save_folder / f'frame_{i:04d}.png'
            frame_image.save(frame_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    args = parser.parse_args()

    os.environ['HF_HOME'] = '/share/project/cwm/xiuyu.yang/.cache/huggingface/'
    os.environ['TORCH_HOME'] = '/share/project/cwm/xiuyu.yang/.cache/torch/'

    # run script
    device = torch.device('cuda:0')

    data_root = '/share/project/cwm/xiuyu.yang/work/dev6/data_old/'

    # data parameters
    # traj_ids (bridge): 667, 1334
    # traj_ids (rt1)
    dataset = 'bridge'
    split = 'val'
    traj_id = 792
    frame_ids_list = list(range(16, 40, 1))

    save_folder = '/share/project/cwm/xiuyu.yang/work/dev6/outputs3/demos/'

    traj_path = os.path.join(save_folder, dataset, split, f'{traj_id:05d}')
    points_path = os.path.join(traj_path, 'points')
    mesh_path = os.path.join(traj_path, 'mesh')
    occupancy_path = os.path.join(traj_path, 'occupancy')
    label_path = os.path.join(traj_path, 'labels')
    render_path = os.path.join(traj_path, 'render')

    # colors60
    colors60_list = generate_colors(n=60)
    colors60_list[-1] = (0, 0, 0)
    colors60 = torch.tensor(colors60_list, device=device, dtype=torch.float32)


    # generate_sparse_points()
    # generate_labels()

    for frame_ids in frame_ids_list:
        generate_occupancy()
        # generate_renderings()

    # extract_mp4_to_images()




# (bs, frame, h, w, 3)

# from diffusers.utils.export_utils import export_to_video


# gt_folder = 'folder_gt'
# pred_folder = 'folder_pred'
# os.makedirs(gt_folder, exist_ok=True)
# os.makedirs(pred_folder, exist_ok=True)

# bs = gt_frames_numpy.shape[0]
# for i in range(bs):

#     gt_video = gt_frames_numpy[i]  # (frame, h, w, 3)
#     pred_video = pred_frames_numpy[i]

#     gt_video = [Image.fromarray(gt_image) for gt_image in gt_video]
#     pred_video = [Image.fromarray(pred_image) for pred_image in pred_video]

#     export_to_video(
#         gt_video,
#         os.path.join(gt_folder, f'{batch_idx}_{i}.mp4'),
#         fps=10,
#     )
#     export_to_video(
#         pred_video,
#         os.path.join(gt_folder, f'{batch_idx}_{i}_eval.mp4'),
#         fps=10,
#     )