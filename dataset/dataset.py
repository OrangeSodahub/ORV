from itertools import chain
import random
import os
import json
import cv2
import fnmatch
import warnings
import traceback
import math
import numpy as np
import torch
from functools import partial
import colorsys
from pathlib import Path
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig, OmegaConf, ListConfig
from numpy import typing as npt
from torch import Tensor
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from diffusers.utils.loading_utils import load_image
from diffusers.configuration_utils import ConfigMixin, register_to_config, __version__

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip
decord.bridge.set_bridge("torch")
from decord import VideoReader, cpu

from training.utils import CONSOLE


def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    return rotm


def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    return rotm


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c),  np.cos(c), 0],
        [0, 0, 1]
    ])
    return rotm


# TODO: improve this function!!!
def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm


def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2euler(R):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert isRotm(R)
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    # (-pi , pi]
    while x > np.pi:
        x -= (2 * np.pi)
    while x <= -np.pi:
        x += (2 * np.pi)
    while y > np.pi:
        y -= (2 * np.pi)
    while y <= -np.pi:
        y += (2 * np.pi)
    while z > np.pi:
        z -= (2 * np.pi)
    while z <= -np.pi:
        z += (2 * np.pi)
    return np.array([x, y, z])


def read_mp4(path: str) -> npt.NDArray:

    assert path.endswith('.mp4'), f'Invalid path which should be in *.mp4 format, got {path}.'

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot process trajectory: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    numpy_images = np.array(frames)

    return numpy_images


def generate_colors(n=60):
    colors_list = []
    for i in range(n):
        h = i / n
        s, v = 0.75, 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color = (int(r * 255), int(g * 255), int(b * 255))
        colors_list.append(color)
    return colors_list


class RobotDataset(Dataset, ConfigMixin):

    config_name = 'dataset_config.json'

    @register_to_config
    def __init__(
        self,
        # disk information
        data_root: str,
        split: str = 'train',
        renderings_folder: str = 'renderings',
        embeddings_folder: str = 'embeddings_full',
        # configurations
        seed: int = 42,
        num_samples: int = -1,
        sample_mode: str = 'drop_last',
        use_cond: bool = True,
        filter_by_cond: bool = False,  # experiment argument!!!
        camera_ids: List[str] = ['0'],
        action_dim: int = 7,  # ee xyz (3) + ee euler (3) + gripper(1)
        sequence_interval: int = 1,
        sequence_length: int = 16,
        sample_frames: int = 17,
        start_frame_interval: dict[str, int] | int = {
            'train': 4, 'val': 16, 'test': 16,
        },
        video_size: list[int] = [320, 480],
        ori_size: list[int] = [256, 320],
        # column names
        control_keys: list[str] = ['depth', 'label'],
        caption_column: str = 'texts',
        video_column: str = 'video',
        latent_column: str = 'latent',
        depth_column: str = 'depth',
        semantic_column: str = 'semantic',
        ref_num: List[int] | int = [1],
        # loadings
        load_actions: bool = True,
        load_tensor: bool = True,
        load_video: bool = False,
        empty_prompt: bool = True,
        load_condGT: bool = False,  # from reconstruction; not renderings
        # others
        slice_frame: bool = True,
        use_3dvae: bool = True,
        vae_has_first_single_frame: bool = True,
        test_mode: bool = False,
        drop_last: bool = True,
        no_normalize: bool = False,
        train: bool = True,
    ) -> None:

        super().__init__()
        CONSOLE.log(f'[bold yellow] Setting up `{self.__class__.__name__}` in {test_mode=} ...')

        self.anno_data_path = os.path.join(data_root, 'annotation', split)
        self.video_data_path = os.path.join(data_root, 'videos', split)
        self.recon_data_path = os.path.join(
            data_root, renderings_folder, 'points', split
        )
        self.label_data_path = os.path.join(
            data_root, renderings_folder, 'semantics', split
        )
        self.render_data_path = os.path.join(
            data_root, renderings_folder, 'render', split
            )
        self.embeddings_data_path = os.path.join(
            data_root, embeddings_folder, split,
        )

        ori_h, ori_w = ori_size
        ori_aspect_ratio = ori_w / ori_h
        aspect_ratio = video_size[1] / video_size[0]
        if aspect_ratio < ori_aspect_ratio:
            new_w = int(ori_w * (video_size[0] / ori_h))
            new_h = video_size[0]
        else:
            new_w = video_size[1]
            new_h = int(ori_h * (video_size[1] / ori_w))

        self.start_frame_interval = start_frame_interval
        if isinstance(start_frame_interval, (dict, DictConfig)):
            self.start_frame_interval = start_frame_interval[split]
        self.accumulate_action = False

        self.c_act_scaler = np.array(
            [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0], dtype=float)

        colors60_list = generate_colors(n=60)
        colors60_list[-1] = (0, 0, 0)
        self.colors60 = torch.from_numpy(np.array(colors60_list)).float()

        # initialize
        self._init_annos()
        self._init_sequences()
        CONSOLE.log(f'Loaded {len(self.ann_files)} trajectories from {split} split.')
        CONSOLE.log(f'Loaded {len(self.samples)} samples from {split} split.')

        # process transformations
        random_flip = False
        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x / 255.),
                # transforms.Lambda(lambda x:
                #         torch.nn.functional.interpolate(x, list(self.video_size), mode='bilinear')),
                transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(tuple(video_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                if not no_normalize
                else transforms.Lambda(lambda x: x),
            ]
        )
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(tuple(video_size)),
            ]
        )

        if 'bridge' in data_root:
            # FIXME: this is a legacy issue!!!
            ori_h, ori_w = 480, 640
        self.depth_transforms = transforms.Compose(
            [
                transforms.Resize(ori_h, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(tuple([ori_h, ori_w])),
                transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(tuple(video_size)),
            ]
        )
        self.label_transforms = transforms.Compose(
            [
                transforms.Resize(ori_h, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(tuple([ori_h, ori_w])),
                transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(tuple(video_size)),
            ]
        )

        CONSOLE.log(self)

    def _init_annos(self):

        ann_files = list(sorted(fnmatch.filter(os.listdir(self.anno_data_path), '*.json')))
        self.ann_files = list(map(lambda fn: os.path.join(self.anno_data_path, fn), ann_files))

        self.recon_files = [None] * len(self.ann_files)
        self.label_files = [None] * (len(self.ann_files))
        self.render_files = [None] * len(self.ann_files)
        if self.config.use_cond or self.config.filter_by_cond:

            # ! we will load conditions from renderings
            if not self.config.load_condGT:

                if not self.config.load_tensor:

                    render_files = list(sorted(fnmatch.filter(os.listdir(self.render_data_path), '*.npz')))

                    filtered_render_files = []
                    with ThreadPoolExecutor(32) as executor:
                        future_to_render_file = [executor.submit(self._check_render_data, render_file) for render_file in render_files]
                        for future in tqdm(as_completed(future_to_render_file), total=len(render_files)):
                            filtered_render_files.extend(future.result())
                    filtered_render_files = list(sorted(filtered_render_files))

                    render_ids = list(map(lambda fn: fn.removesuffix('.npz'), filtered_render_files))
                    self.render_files = list(map(lambda fn: os.path.join(self.render_data_path, fn), filtered_render_files))

                else:

                    depth_latents_files = list(sorted(os.listdir(
                        os.path.join(self.embeddings_data_path, 'depth_latents')
                    )))
                    depth_latents_files = list(sorted(
                        map(lambda fn: fn.removesuffix('.pt').split('_')[0].lstrip('0'), depth_latents_files)
                    ))
                    label_latents_files = list(sorted(os.listdir(
                        os.path.join(self.embeddings_data_path, 'label_latents')
                    )))
                    label_latents_files = list(sorted(
                        map(lambda fn: fn.removesuffix('.pt').split('_')[0].lstrip('0'), label_latents_files)
                    ))

                    render_ids = list(sorted(set(depth_latents_files) & set(label_latents_files)))
                    if 'depth' in self.config.control_keys and 'label' not in self.config.control_keys:
                        render_ids = depth_latents_files
                    if 'label' in self.config.control_keys and 'depth' not in self.config.control_keys:
                        render_ids = label_latents_files

            # ! we will load conditions from reconstructions
            else:

                if not self.config.load_tensor:

                    recon_files = list(sorted(os.listdir(self.recon_data_path)))
                    recon_files = list(sorted(filter(lambda fn:
                            len(list(sorted(fnmatch.filter(os.listdir(os.path.join(self.recon_data_path, fn)), 'frame_*.npy')))) > 0,
                            recon_files,
                    )))
                    label_files = list(sorted(os.listdir(self.label_data_path)))
                    label_files = list(sorted(filter(lambda fn:
                            len(list(sorted(fnmatch.filter(os.listdir(os.path.join(self.label_data_path, fn)), 'frame_*.npz')))) > 0,
                            label_files,
                    )))
                    render_ids = list(sorted(set(recon_files) & set(label_files)))
                    if 'depth' in self.config.control_keys and 'label' not in self.config.control_keys:
                        render_ids = recon_files
                    if 'label' in self.config.control_keys and 'depth' not in self.config.control_keys:
                        render_ids = label_files
                    self.recon_files = list(map(lambda fn: os.path.join(self.recon_data_path, fn), render_ids))
                    self.label_files = list(map(lambda fn: os.path.join(self.label_data_path, fn), render_ids))

                else:

                    depth_latents_files = list(sorted(os.listdir(
                        os.path.join(self.embeddings_data_path, 'depthGT_latents')
                    )))
                    depth_latents_files = list(sorted(
                        map(lambda fn: fn.removesuffix('.pt').split('_')[0].lstrip('0'), depth_latents_files)
                    ))
                    label_latents_files = list(sorted(os.listdir(
                        os.path.join(self.embeddings_data_path, 'labelGT_latents')
                    )))
                    label_latents_files = list(sorted(
                        map(lambda fn: fn.removesuffix('.pt').split('_')[0].lstrip('0'), label_latents_files)
                    ))

                    render_ids = list(sorted(set(depth_latents_files) & set(label_latents_files)))
                    if 'depth' in self.config.control_keys and 'label' not in self.config.control_keys:
                        render_ids = depth_latents_files
                    if 'label' in self.config.control_keys and 'depth' not in self.config.control_keys:
                        render_ids = label_latents_files

            # ! need to filter the original data according to the condition data
            if 'rt1' in self.config.data_root:
                # FIXME: this is a legacy issue caused by irasim!!!

                # self.ann_files = list(sorted(filter(
                #     lambda ann_file: int(json.load(open(ann_file, 'r'))['episode_id']) in render_ids, self.ann_files
                # )))

                render_ids_set = set(render_ids)

                def _fetch_episode_ids(ann_file, valid_ids):
                    try:
                        with open(ann_file, 'r') as f:
                            data = json.load(f)
                        if data['episode_id'].lstrip('0') in valid_ids:
                            return [ann_file]
                        else:
                            return []
                    except Exception:
                        return []

                valid_ann_files = []
                with ThreadPoolExecutor(max_workers=32) as executor:
                    futures = [
                        executor.submit(_fetch_episode_ids, ann_file, render_ids_set)
                        for ann_file in self.ann_files
                    ]
                    for future in tqdm(as_completed(futures), total=len(self.ann_files)):
                        valid_ann_files.extend(future.result())

                self.ann_files = list(sorted(valid_ann_files))

            else:
                self.ann_files = list(sorted(filter(lambda ann_file: os.path.basename(ann_file).removesuffix('.json') in render_ids, self.ann_files)))

            self.render_ids = list(sorted(render_ids))

    def _check_render_data(self, render_file):

        returns = []

        render_data = np.load(os.path.join(self.render_data_path, render_file))
        depth_ok = 'depths' in render_data
        label_ok = bool(render_data['is_labeled'])
        data_ok = {'depth': depth_ok, 'label': label_ok}

        file_ok = all([data_ok[key] for key in self.config.control_keys])
        if file_ok:
            returns.append(render_file)

        return returns

    def _init_sequences(self):

        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = [executor.submit(self._load_and_process_ann_file, i) for i in range(len(self.ann_files))]
            for future in tqdm(as_completed(future_to_ann_file), total=len(self.ann_files)):
                samples.extend(future.result())

        samples = list(sorted(samples, key=lambda x: (int(x['episode_id']), int(x['start_frame_idx']))))
        if self.config.num_samples > 0:
            if self.config.num_samples >= len(samples):
                CONSOLE.log(f'[bold yellow]Given {self.config.num_samples=} which exceeds the existing samples {len(samples)=}. Will use all of them.')
            else:
                if self.config.sample_mode == 'random':
                    random.seed(self.config.seed)
                    selected_indices = random.sample(range(len(samples)), int(self.config.num_samples))
                    samples = [sample for i, sample in enumerate(samples) if i in selected_indices]
                if self.config.sample_mode == 'drop_last':
                    samples = samples[:int(self.config.num_samples)]

        self.samples = samples
        self.episode_ids = list(map(lambda sample: str(int(sample['episode_id'])), samples))

    def _load_and_process_ann_file(self, i):

        ann_file = self.ann_files[i]
        recon_file = self.recon_files[i]
        label_file = self.label_files[i]
        render_file = self.render_files[i]

        samples = []
        try:
            with open(ann_file, "r") as f:
                ann = json.load(f)
        except:
            CONSOLE.log(f'Failed to load ann {ann_file}, will skip it!')
            return samples

        n_frames = len(ann['state'])  # `state` shape [n_frames, 7]
        episode_id = ann['episode_id']
        if self.config.use_cond or self.config.filter_by_cond:
            if (episode_id.lstrip('0') or '0') not in self.render_ids:
                raise RuntimeError(f'Episode id {episode_id} not found in render_ids!')

        # If slice frames, we will extract samples with fixed length and at an preset interval
        if self.config.slice_frame:

            start_frame = 0 if not self.config.vae_has_first_single_frame else self.config.sequence_interval
            for frame_i in range(
                start_frame,
                n_frames,
                self.start_frame_interval * self.config.sequence_interval
            ):

                sample = dict(
                    episode_id=ann['episode_id'],
                    ann_file=ann_file,
                    recon_file=recon_file,
                    label_file=label_file,
                    render_file=render_file,
                    prompt=ann[self.config.caption_column][0],
                )

                frame_ids = []
                curr_frame_i = frame_i
                while True:
                    if curr_frame_i > (n_frames - 1) or len(frame_ids) == self.config.sequence_length:
                        break
                    frame_ids.append(curr_frame_i)
                    curr_frame_i += self.config.sequence_interval

                # make sure there are sequence_length number of frames
                if len(frame_ids) == self.config.sequence_length:

                    # to satify the (8n+1) frames
                    if self.config.vae_has_first_single_frame:
                        frame_ids.insert(0, frame_i - self.config.sequence_interval)

                    sample['frame_ids'] = frame_ids
                    sample['start_frame_idx'] = frame_ids[0]
                    sample['num_frame'] = len(frame_ids)
                    sample['is_sliced'] = True

                    # ! other attributes
                    if 'n_view' in self.config:  # specialized for 'MultiViewRobotDataset'
                        for i_view in range(self.config.n_view):
                            sample[f'has_image_{i_view}'] = ann.get(f'has_image_{i_view}', True)
                            sample[f'use_image_{i_view}'] = sample[f'has_image_{i_view}']

                    # ! check if conditions exist
                    sample_ok = True
                    if (self.config.use_cond or self.config.filter_by_cond) and self.config.load_tensor:
                        sample_n_view = sum([sample[f'has_image_{i_view}'] for i_view in range(self.config.n_view)])
                        sample_name = f'{int(episode_id):05d}_{frame_ids[0]:02d}_{len(frame_ids):02d}'
                        depth_ok = all([
                            os.path.exists(
                                os.path.join(
                                    self.config.data_root,
                                    self.config.embeddings_folder,
                                    self.config.split,
                                    'depth_latents',
                                    f'{sample_name}_{j}.pt'
                                )
                            )
                            for j in range(sample_n_view)
                        ])
                        label_ok =  all([
                            os.path.exists(
                                os.path.join(
                                    self.config.data_root,
                                    self.config.embeddings_folder,
                                    self.config.split,
                                    'label_latents',
                                    f'{sample_name}_{j}.pt'
                                )
                            )
                            for j in range(sample_n_view)
                        ])
                        data_ok = {'depth': depth_ok, 'label': label_ok}
                        sample_ok = all([data_ok[key] for key in self.config.control_keys])

                    if sample_ok:
                        samples.append(sample)

        # When not slice frames, the output episodes will have various length;
        # However, we maybe cut them to satisfy the (8n+1) rule.
        else:

            if self.config.drop_last:
                crop_n_frames = (n_frames // 8) * 8 + 1
                if crop_n_frames > n_frames:
                    crop_n_frames = (n_frames // 8 - 1) * 8 + 1
                n_frames = crop_n_frames
            frame_ids = list(range(n_frames))

            samples.append(
                dict(
                    episode_id=ann['episode_id'],
                    ann_file=ann_file,
                    recon_file=recon_file,
                    label_file=label_file,
                    render_file=render_file,
                    prompt=ann[self.config.caption_column][0],
                    frame_ids=frame_ids,
                    start_frame_idx=frame_ids[0],
                    num_frame=len(frame_ids),
                    is_sliced=False,
                )
            )

        return samples

    def __len__(self):
        return len(self.samples)

    """ __getitem__ """

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert ((0 <= np.array(frame_ids)).all() and (np.array(frame_ids) < len(vr)).all())
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).numpy()
        return frame_data

    @property
    def num_refs(self):
        ref_nums = self.config.ref_num
        num_refs = None
        if isinstance(ref_nums, ListConfig):
            ref_nums = OmegaConf.to_container(ref_nums)
        if isinstance(ref_nums, list):
            num_refs = len(ref_nums)
        if isinstance(ref_nums, int):
            num_refs = 1
        if num_refs is None:
            raise ValueError(f'Invalid {self.config.ref_num=}!')
        return int(num_refs)

    @property
    def ref_num(self):
        ref_nums = self.config.ref_num
        ref_num = None
        if isinstance(ref_nums, ListConfig):
            ref_nums = OmegaConf.to_container(ref_nums)
        if isinstance(ref_nums, list):
            assert all([isinstance(ref_num, int) for ref_num in ref_nums]), f'Invalid type of ref_num!'
            ref_num = random.choice(ref_nums)
        if isinstance(ref_nums, int):
            ref_num = ref_nums
        if ref_num is None:
            raise ValueError(f'Invalid {self.config.ref_num=}!')
        return int(ref_num)

    def get_ref_nums_for_all_samples(self):
        return [self.ref_num for _ in range(len(self))]

    def get_n_views_for_all_samples(self):
        return [1 for _ in range(len(self))]

    def _apply_semantic_colormap(self, semantic):
        # 'semantic': [F, H, W]

        max_label = int(semantic.max())

        x = torch.zeros((3, *semantic.shape), dtype=torch.float)
        for i in range(max_label + 1):
            x[:, semantic == i] = self.colors60[i][:, None]

        x = x.permute(1, 0, 2, 3)  # -> [F, 3, H, W]

        return x / 255.0

    def _get_frames(
        self,
        frame_ids: List[int],
        video_path: Optional[str] = None,
        latent_video_path: Optional[str] = None,
        latent_ref_path: Optional[str] = None,
        is_sliced: Optional[bool] = True,
        image_path: Optional[str] = None,
        sliced_video_path: Optional[str] = None,
    ) -> Dict[str, Tensor]:

        returns = dict()

        # ! load latents for training
        if (not self.config.test_mode and self.config.load_tensor):

            assert latent_video_path is not None, f'Invalid {latent_video_path=}.'
            assert latent_ref_path is not None, f'Invalid {latent_ref_path=}.'
            with open(os.path.join(self.config.data_root, latent_video_path), 'rb') as f:
                latents_video = torch.load(f, weights_only=True)
            with open(os.path.join(self.config.data_root, latent_ref_path), 'rb') as f:
                latents_ref = torch.load(f, weights_only=True)

            # if use 2d latents, shape of `latents` is [F, C, H, W]
            # if use 3d latents, shape of `latents` is [C, F, H, W]
            if self.config.use_3dvae:
                latents_video = latents_video.permute(1, 0, 2, 3)
                latents_ref = latents_ref.permute(1, 0, 2, 3)
                frame_ids = list(sorted(set([frame_id // 4 for frame_id in frame_ids])))

            # the video latents are sliced
            if is_sliced:
                frame_ids = list(range(latents_video.size(0)))

            # sanity check
            if latents_video.shape[0] <= max(frame_ids):
                raise RuntimeError(f'Got mismatched latent video and frame ids: {latents_video.shape} v.s. {frame_ids}, path: {latent_video_path}.')

            returns['latents'] = latents_video[frame_ids]  # [n_frame, n_channel, fH, fW]
            returns['image'] = latents_ref  # [n_frame, n_channel, fH, fW]

        # ! load raws for training/preprocessing
        if (not self.config.test_mode and not self.config.load_tensor) or self.config.load_video:
            assert video_path is not None, f'Invalid video_path: {video_path}.'
            video_reader = decord.VideoReader(
                uri=os.path.join(self.config.data_root, video_path), num_threads=2)

            if not self.config.slice_frame and not self.config.drop_last:
                assert frame_ids == list(range(len(video_reader))), f'Got invalid `frame_ids`: {frame_ids=} v.s. {len(video_reader)=}.'

            frames = video_reader.get_batch(frame_ids).float()
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [f, c, h, w]
            frames = self.video_transforms(frames)

            refs = frames[:self.ref_num].clone()

            returns['videos'] = frames
            returns['image'] = refs

        # ! load ref frame for testing/evaluation
        if self.config.test_mode:

            # ! load ref latents.
            if self.config.load_tensor:

                assert latent_ref_path is not None, f'Invalid {latent_ref_path=}.'
                with open(os.path.join(self.config.data_root, latent_ref_path), 'rb') as f:
                    latents_ref = torch.load(f, weights_only=True)

                    if self.config.use_3dvae:
                        latents_ref = latents_ref.permute(1, 0, 2, 3)

                    returns['image'] = latents_ref  # [n_frame, n_channel, fH, fW]

            else:

                # ! load ref pil images.
                CONSOLE.log(f'[on red]Will not load preprocessed tensors!')
                try:
                    _pil_image = load_image(
                        os.path.join(self.config.data_root, image_path)
                    )
                except Exception as e:
                    CONSOLE.log(f'[red]Failed to load image from {image_path} due to {e}!')
                    raise

                pil_image = []

                if self.ref_num > 1:
                    w, h = _pil_image.size
                    assert w % self.ref_num == 0, f'Invalid width {w=}.'
                    subw = w // self.ref_num

                    for i in range(self.ref_num):
                        sub_pil_image = (
                            _pil_image.crop(
                                (
                                    i * subw,
                                    0,
                                    (i + 1) * subw,
                                    h
                                )
                            )
                        )

                        if sub_pil_image.size != (self.config.video_size[1], self.config.video_size[0]):
                            sub_pil_image = self.image_transforms(sub_pil_image)

                        pil_image.append(sub_pil_image)

                else:
                    pil_image.append(_pil_image)

                # the returned `pil_image` here loads all reference frames but only 1 view;
                # all views (namely multiview, if possible) will be handled by `MultiViewRobotDataset`.
                returns['pil_image'] = [pil_image]  # n_view -> n_ref_frame (n_view=1)

                # try:
                #     pil_image = load_image(
                #         os.path.join(self.config.data_root, image_path)
                #     )
                # except Exception as e:
                #     CONSOLE.log(f'[red]Failed to load image from {image_path} due to {e}!')
                #     raise

                # if pil_image.size != (self.config.video_size[1], self.config.video_size[0]):
                #     pil_image = self.image_transforms(pil_image)
                # returns['pil_image'] = [[pil_image]]  # n_view -> n_frame (n_view=1, n_frame=1)

                # if sliced_video_path is not None:

                #     video_reader = decord.VideoReader(
                #         uri=os.path.join(self.config.data_root, video_path), num_threads=2)
                #     frames = video_reader.get_batch(list(range(len(video_reader))))  # [f, h, w, c]

                #     returns['sliced_frames'] = frames

        return returns

    def _get_cond_frames(
        self,
        frame_ids: List[int],
        recon_file_path: Optional[str] = None,
        label_file_path: Optional[str] = None,
        render_file_path: Optional[str] = None,
        latent_depth_paths: Optional[list[str] | str] = None,
        latent_label_paths: Optional[list[str] | str] = None,
        view_ids: list[int] = [0],
        num_view: int = 1,
    ):
        returns = dict()

        if latent_depth_paths is not None and not isinstance(latent_depth_paths, list):
            latent_depth_paths = [latent_depth_paths]
        if latent_label_paths is not None and not isinstance(latent_label_paths, list):
            latent_label_paths = [latent_label_paths]

        # ! load latents for training
        # if (not self.config.test_mode and self.config.load_tensor):  # TODO: fix this term!!!
        if self.config.load_tensor:

            if 'depth' in self.config.control_keys:

                assert latent_depth_paths is not None, f'Invalid {latent_depth_paths=}.'

                latents_depth = []
                for latent_depth_path in latent_depth_paths:

                    with open(os.path.join(self.config.data_root, latent_depth_path), 'rb') as f:
                        latents_depth_view = torch.load(f, weights_only=True)

                    if self.config.use_3dvae:
                        # if use 2d latents, shape of `latents` is [F, C, H, W]
                        # if use 3d latents, shape of `latents` is [C, F, H, W]
                        latents_depth_view = latents_depth_view.permute(1, 0, 2, 3)
                        frame_ids = list(sorted(set([frame_id // 4 for frame_id in frame_ids])))

                    latents_depth.append(latents_depth_view)

                latents_depth = torch.stack(latents_depth)  # -> [n_view, n_frame, n_channel, fH, fW]
                latents_depth = latents_depth.flatten(0, 1)
                returns['latents_depth'] = latents_depth  # [n_frame, n_channel, fH, fW]

            if 'label' in self.config.control_keys:

                assert latent_label_paths is not None, f'Invalid {latent_label_paths=}.'

                latents_label = []
                for latent_label_path in latent_label_paths:

                    with open(os.path.join(self.config.data_root, latent_label_path), 'rb') as f:
                        latents_label_view = torch.load(f, weights_only=True)

                    if self.config.use_3dvae:
                        latents_label_view = latents_label_view.permute(1, 0, 2, 3)  # -> [F, C, H, W]
                        frame_ids = list(sorted(set([frame_id // 4 for frame_id in frame_ids])))

                    latents_label.append(latents_label_view)

                latents_label = torch.stack(latents_label)  # -> [n_view, n_frame, n_channel, fH, fW]
                latents_label = latents_label.flatten(0, 1)
                returns['latents_label'] = latents_label

        # ! load raws for training/preprocessing
        if (not self.config.test_mode and not self.config.load_tensor):

            depths = labels = None

            if not self.config.load_condGT:

                render_data = np.load(render_file_path)

                if 'depth' in self.config.control_keys:
                    # depths = torch.from_numpy(render_data['depths'])[frame_ids]  # [n_frame, n_view, h, w]
                    # --------------------------------------------------------------------------
                    depths = torch.from_numpy(render_data['depths'])  # [n_frame, n_view, h, w]
                    if depths.shape[1] != num_view:  # FIXME: this is a legacy issue!!!
                        _, _, h, w = depths.shape
                        depths = depths.reshape(-1, num_view, h, w)
                    depths = depths[frame_ids]
                    # -------------------------------------------------------------------------
                    depths = torch.stack([depths[:, view_id] for view_id in view_ids], dim=1)
                    depths = depths.transpose(0, 1).flatten(0, 1)  # -> [v * f, h, w]

                    depths = torch.stack([
                        self.depth_transforms(depth) for depth in depths[:, None]
                    ])  # [F, C, H, W]

                    depths = torch.clamp(depths, min=0.01, max=0.4) * 2.5  # TODO: improve this function!!!

                if 'label' in self.config.control_keys:
                    is_labeled = render_data['is_labeled']
                    if is_labeled:
                        labels = torch.from_numpy(render_data['semantics'][frame_ids]).float()  # [n_frame, h, w] or [n_frame, 3, h, w]
                        labels = torch.stack([labels[:, view_id] for view_id in view_ids], dim=1)
                        labels = labels.transpose(0, 1).flatten(0, 1)  # -> [v * f, h, w]

                        if labels.ndim == 3:
                            labels = self._apply_semantic_colormap(labels)  # [F, C, H, W]

                        labels = torch.stack([
                            self.label_transforms(label) for label in labels
                        ])  # [F, C, H, W]

            else:

                # load depths
                if 'depth' in self.config.control_keys:
                    recon_files = list(sorted(fnmatch.filter(os.listdir(recon_file_path), 'frame_*.npy')))
                    recon_files = [recon_files[i] for i in frame_ids]
                    depths = torch.from_numpy(
                        np.array([np.load(os.path.join(recon_file_path, recon_file)) for recon_file in recon_files])
                    )  # [n_frame, h, w]

                    depths = torch.stack([
                        self.depth_transforms(depth) for depth in depths[:, None]
                    ])  # [F, C, H, W]

                # load semantics
                if 'label' in self.config.control_keys:
                    label_files = list(sorted(fnmatch.filter(os.listdir(label_file_path), 'frame_*.npz')))
                    label_files = [label_files[i] for i in frame_ids]
                    labels = torch.from_numpy(
                        np.array([np.load(os.path.join(label_file_path, label_file))['annotated_frame_color'] for label_file in label_files])
                    ).permute(0, 3, 1, 2)  # [F, C, H, W]

                    labels = torch.stack([
                        self.label_transforms(label) for label in labels
                    ])  # [F, C, H, W]

            if depths is not None:
                returns['depths'] = depths
            if labels is not None:
                returns['labels'] = labels

        # ! load ref frame for testing
        if self.config.test_mode:
            pass

        return returns

    def _get_robot_states(self, label, frame_ids):

        all_states = np.array(label['state'])
        all_cont_gripper_states = np.array(label['continuous_gripper_state'])
        arm_states = all_states[frame_ids, :6]
        cont_gripper_states = all_cont_gripper_states[frame_ids]

        if self.config.vae_has_first_single_frame:
            assert self.config.use_3dvae, 'Only when using 3d vae, vae has first single frame!'

        # sannity check
        assert (
            ((
                arm_states.shape[0] == self.config.sequence_length
                and not self.config.vae_has_first_single_frame) or
             (
                 arm_states.shape[0] - 1 == self.config.sequence_length
                 and self.config.vae_has_first_single_frame)
        ) and self.config.slice_frame) or not self.config.slice_frame, \
            f'{arm_states.shape=} v.s. {self.config.sequence_length=} with {frame_ids=}!'

        assert (
            ((
                cont_gripper_states.shape[0] == self.config.sequence_length
                and not self.config.vae_has_first_single_frame) or
             (
                 cont_gripper_states.shape[0] - 1 == self.config.sequence_length
                 and self.config.vae_has_first_single_frame)
        ) and self.config.slice_frame) or not self.config.slice_frame, \
            f'{cont_gripper_states.shape=} v.s. {self.config.sequence_length=} with {frame_ids=}!'

        return arm_states, cont_gripper_states

    def _get_actions(self, arm_states, gripper_states):

        sequence_length = arm_states.shape[0]
        action = np.zeros((sequence_length - 1, self.config.action_dim))

        if self.accumulate_action:

            first_xyz = arm_states[0, 0:3]
            first_rpy = arm_states[0, 3:6]
            first_rotm = euler2rotm(first_rpy)

            for k in range(1, sequence_length):

                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper

        else:

            for k in range(1, sequence_length):

                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = euler2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper

        return torch.from_numpy(action)  # (l - 1, act_dim)

    def fetch_index(self, index: int):
        return self.__getitem__(index, raise_error=True)

    def fetch_episode(self, episode_id: str):

        if episode_id not in self.episode_ids:
            raise ValueError(f'Not found {episode_id=} in dataset with {self.episode_ids[:20]=}.')

        return [i for i, x in enumerate(self.episode_ids) if x == episode_id]

    def __getitem__(
        self,
        index_or_res: int | tuple,
        camera_index: Optional[int] = 0,
        return_video = False,
        raise_error: bool = False,
    ) -> Dict[str, Any]:

        if not isinstance(index_or_res, int):
            index, ref_num = index_or_res
        else:
            index = index_or_res
            ref_num = self.ref_num

        data = dict()

        sample = self.samples[index]
        camera_id = self.config.camera_ids[camera_index]

        ann_file = sample['ann_file']
        prompt = sample['prompt']
        frame_ids = sample['frame_ids']
        start_frame_idx = int(sample['start_frame_idx'])
        num_frame = int(sample['num_frame'])
        episode_id = int(sample['episode_id'])

        data['prompt'] = prompt if not self.config.empty_prompt else ''

        try:

            with open(ann_file, 'r') as f:
                label = json.load(f)

            sample_name = f'{episode_id:05d}_{start_frame_idx:02d}_{num_frame:02d}'
            # for those datasets with len(camera_ids) > 1, their sample_name includes the camera_id.
            # if you setup the `RobotDataset`, it will only use the first camera;
            # to use multiple cameras, you should setup the `MultiViewRobotDataset`.
            if len(self.config.camera_ids) > 1:
                sample_name = f'{sample_name}_0'  # default camera 0

            # ! load prompt embeddings
            if self.config.load_tensor:
                if self.config.empty_prompt:
                    data['prompt_embeds'] = torch.load(
                        os.path.join(self.config.data_root, self.config.embeddings_folder, 'empty_prompt.pt'), weights_only=True
                    )[0]  # [seq_len, embed_dim] do not have `batch_size` dimension
                else:
                    prompt_embeds_path = os.path.join(
                        self.config.data_root, self.config.embeddings_folder, self.config.split, 'prompt_embeds', f'{sample_name}.pt'
                    )
                    data['prompt_embeds'] = torch.load(prompt_embeds_path, weights_only=True)  # [seq_len, embed_dim]

            # ! load states and actions
            if self.config.load_actions:
                arm_states, gripper_states = self._get_robot_states(label, frame_ids)
                actions = self._get_actions(arm_states, gripper_states)
                actions *= self.c_act_scaler

                data['actions'] = actions.float()

            # ! load frames
            video_path = label['videos'][camera_index]['video_path']
            # latent_video_path = label['latent_videos'][camera_index]['latent_video_path']  # FIXME: original 2d latents

            image_path = os.path.join(
                self.config.embeddings_folder, self.config.split, f'images{ref_num}', f"{sample_name}.png")
            latent_ref_path = os.path.join(
                self.config.embeddings_folder, self.config.split, f'image{ref_num}_latents', f'{sample_name}.pt')
            sliced_video_path = (
                os.path.join(
                    self.config.embeddings_folder, self.config.split, 'videos', f'{sample_name}.mp4')
                if self.config.slice_frame else None
            )
            latent_video_path = os.path.join(
                self.config.embeddings_folder, self.config.split, 'video_latents', f"{sample_name}.pt")

            # ------------------------------------------------------------------------------------------------
            # TODO: this is a legacy issue, need to fix it!!!
            if not self.config.slice_frame:  # num_frame != 17:
                if self.config.load_tensor:
                    CONSOLE.log(f"[on red]Warning: you're tring to load latents with non-fixed frames!")
                # legacy_sample_name = f'{episode_id:05d}_{start_frame_idx:02d}_{17:02d}'
                legacy_sample_name = f'{episode_id:05d}_{start_frame_idx:02d}_{16:02d}'
                image_path = os.path.join(
                    self.config.embeddings_folder, self.config.split, f'images{ref_num}', f"{legacy_sample_name}.png")
                latent_ref_path = os.path.join(
                    self.config.embeddings_folder, self.config.split, f'image{ref_num}_latents', f'{legacy_sample_name}.pt')
            # ------------------------------------------------------------------------------------------------

            data.update(
                self._get_frames(
                    frame_ids,
                    image_path=image_path,
                    video_path=video_path,
                    latent_video_path=latent_video_path,
                    latent_ref_path=latent_ref_path,
                    sliced_video_path=sliced_video_path,
                    is_sliced=sample['is_sliced'],
                )
            )

            # ! load conditions
            if self.config.use_cond:

                # TODO: finish this sanity check
                recon_file = sample['recon_file']
                label_file = sample['label_file']
                render_file = sample['render_file']
                # assert not (recon_file is None and label_file is None and render_file is None), f'Please set `render_file` if use_cond!'

                latent_depth_path = (
                    os.path.join(
                        self.config.embeddings_folder, self.config.split, 'depth_latents', f'{sample_name}.pt')
                    if not self.config.load_condGT else
                    os.path.join(
                        self.config.embeddings_folder, self.config.split, 'depthGT_latents', f'{sample_name}.pt')
                )
                latent_label_path = (
                    os.path.join(
                        self.config.embeddings_folder, self.config.split, 'label_latents', f'{sample_name}.pt')
                    if not self.config.load_condGT else
                    os.path.join(
                        self.config.embeddings_folder, self.config.split, 'labelGT_latents', f'{sample_name}.pt')
                )

                data.update(
                    self._get_cond_frames(
                        frame_ids=frame_ids,
                        recon_file_path=recon_file,
                        label_file_path=label_file,
                        render_file_path=render_file,
                        latent_depth_path=latent_depth_path,
                        latent_label_path=latent_label_path,
                    )
                )

            data['metainfo'] = {
                'episode_id': label['episode_id'],
                'camera_id': camera_id,
                'frame_ids': frame_ids,
                'ref_num': ref_num,
                'start_frame_idx': start_frame_idx,
                'num_frame': num_frame,
                'num_view': 1,
                'sample_name': sample_name,
            }

            return data

        except Exception as e:
            warnings.warn(f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                          f"(by randomly sampling another sample in the same dataset).")
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())

            if int(os.getenv('DEBUG', 0)) or raise_error:
                raise

            return self[(np.random.randint(len(self.samples)), ref_num)]

    def to_json_string(self) -> str:
        """
        Serializes the configuration instance to a JSON string.

        Returns:
            `str`:
                String containing all the attributes that make up the configuration instance in JSON format.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict["_class_name"] = self.__class__.__name__
        config_dict["_diffusers_version"] = __version__

        def to_json_saveable(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, Path):
                value = value.as_posix()
            elif isinstance(value, (ListConfig, DictConfig)):
                value = OmegaConf.to_container(value)
            return value

        if "quantization_config" in config_dict:
            config_dict["quantization_config"] = (
                config_dict.quantization_config.to_dict()
                if not isinstance(config_dict.quantization_config, dict)
                else config_dict.quantization_config
            )

        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
        # Don't save "_ignore_files" or "_use_default_values"
        config_dict.pop("_ignore_files", None)
        config_dict.pop("_use_default_values", None)
        # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
        _ = config_dict.pop("_pre_quantization_dtype", None)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @staticmethod
    def save_gif(frames: Tensor | npt.NDArray | List[Image.Image], path: os.PathLike) -> None:

        if isinstance(frames, Tensor):
            frames = frames.cpu().numpy()

        if isinstance(frames, np.ndarray):
            frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]

        frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    @staticmethod
    def save_mp4(frames: Tensor | npt.NDArray | List[Image.Image], path: os.PathLike) -> None:
        pass


class DemoRobotDataset(RobotDataset):

    r"""Data structure
    DATA_ROOT
        |----0000 // episode_id
        |      |----rgb
        |      |----anotations.json
        |----0001
        |      |----rgb
        |      |----anotations.json
        ...
    """

    @register_to_config
    def __init__(
        self,
        data_root: str,
        split: str = 'test',
        video_size: list[int] = [320, 480],
        ori_size: list[int] = [256, 320],
        ref_num: int = 1,
        load_actions: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(
            data_root=data_root,
            split=split,
            ref_num=ref_num,
            load_actions=load_actions,
            **kwargs)
        CONSOLE.log(f'[bold yellow] Setting up `{self.__class__.__name__}` ...')

        ori_h, ori_w = ori_size
        new_h = int(ori_h * (video_size[1] / ori_w))

        self.c_act_scaler = np.array(
            [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0], dtype=float)

        # initialize
        self._init_annos()
        self._init_sequences()

        # process transformations
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(new_h, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(tuple(video_size)),
            ]
        )

    def _init_annos(self):

        episodes = list(sorted(os.listdir(self.config.data_root)))
        self.ann_files = list(map(lambda fn: os.path.join(self.config.data_root, fn, 'annotations.json'), episodes))

        self.render_files = [None] * len(self.ann_files)

    def _load_and_process_ann_file(self, i):

        ann_file = self.ann_files[i]
        render_file = self.render_files[i]

        samples = []
        try:
            with open(ann_file, "r") as f:
                ann = json.load(f)
        except:
            CONSOLE.log(f'Failed to load ann {ann_file}, will skip it!')
            return samples

        n_frames = len(ann['state'])

        # If slice frames, we will extract samples with fixed length and at an preset interval
        if self.config.slice_frame:

            start_frame = 0 if not self.config.use_3dvae else 1
            for frame_i in range(start_frame, n_frames, self.start_frame_interval):

                sample = dict(
                    episode_id=ann['episode_id'], ann_file=ann_file, render_file=render_file, prompt=ann['texts'][0],
                    )

                frame_ids = []
                curr_frame_i = frame_i
                while True:
                    if curr_frame_i > (n_frames - 1) or len(frame_ids) == self.config.sequence_length:
                        break
                    frame_ids.append(curr_frame_i)
                    curr_frame_i += self.config.sequence_interval

                # make sure there are sequence_length number of frames
                if len(frame_ids) == self.config.sequence_length:

                    # to satify the (8n+1) frames
                    if self.config.use_3dvae:
                        frame_ids.insert(0, frame_i - 1)

                    sample['frame_ids'] = frame_ids
                    sample['start_frame_idx'] = frame_ids[0]
                    sample['num_frame'] = len(frame_ids)
                    sample['is_sliced'] = True
                    samples.append(sample)

        # When not slice frames, the output episodes will have various length;
        # However, we maybe crop them to satisfy the (8n+1) length.
        else:

            if self.config.drop_last:
                crop_n_frames = (n_frames // 8) * 8 + 1
                if crop_n_frames > n_frames:
                    crop_n_frames = (n_frames // 8 - 1) * 8 + 1
                n_frames = crop_n_frames
            frame_ids = list(range(n_frames))

            samples.append(
                dict(
                    episode_id=ann['episode_id'],
                    ann_file=ann_file,
                    render_file=render_file,
                    prompt=ann['texts'][0],
                    frame_ids=frame_ids,
                    start_frame_idx=frame_ids[0],
                    num_frame=len(frame_ids),
                    is_sliced=False,
                )
            )

        return samples

    def __getitem__(self, index: int, camera_index: Optional[int] = 0, raise_error: bool = False):
        data = dict()

        sample = self.samples[index]
        camera_id = self.config.camera_ids[camera_index]

        ann_file = sample['ann_file']
        prompt = sample['prompt']
        frame_ids = sample['frame_ids']
        episode_id = sample['episode_id']
        start_frame_idx = int(sample['start_frame_idx'])
        num_frame = int(sample['num_frame'])

        data['prompt'] = prompt if not self.config.empty_prompt else ''

        try:

            with open(ann_file, 'r') as f:
                label = json.load(f)

            sample_name = f'{int(episode_id):05d}_{start_frame_idx:02d}_{num_frame:02d}'

            # ! load states and actions
            if self.config.load_actions:
                arm_states, gripper_states = self._get_robot_states(label, frame_ids)
                actions = self._get_actions(arm_states, gripper_states)
                actions *= self.c_act_scaler

                data['actions'] = actions.float()

            image_path = os.path.join(
                f'{int(episode_id):04d}', 'rgb', f'{start_frame_idx:05d}.png'
            )

            # ! load frames
            data.update(
                self._get_frames(
                    frame_ids,
                    image_path=image_path,
                )
            )

            # ! load conditions
            if self.config.use_cond:
                render_file = sample['render_file']
                assert render_file is not None, f'Please set `render_file` if use_cond!'

            data['metainfo'] = {
                'episode_id': label['episode_id'],
                'camera_id': camera_id,
                'frame_ids': frame_ids,
                'ref_num': 1,
                'start_frame_idx': start_frame_idx,
                'num_frame': num_frame,
                'num_view': 1,
                'sample_name': sample_name,
            }

            return data

        except Exception as e:
            warnings.warn(f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                          f"(by randomly sampling another sample in the same dataset).")
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())

            if int(os.getenv('DEBUG', 0)) or raise_error:
                raise

            return self[np.random.randint(len(self.samples))]


class MultiViewRobotDataset(RobotDataset):

    @register_to_config
    def __init__(
        self,
        n_view: int,
        camera_ids: List[str | int],
        **kwargs,
    ) -> None:
        assert n_view == len(camera_ids), f'Mismatched {n_view=} and {camera_ids=}!'
        for i in range(n_view):
            if isinstance(camera_ids[i], int):
                camera_ids[i] = str(camera_ids[i])
        super().__init__(camera_ids=camera_ids, **kwargs)

        ref_num = self.config.ref_num
        ref_nums = [ref_num] if isinstance(ref_num, int) else ref_num
        self.resolutions = [
            (ref_num, n_view) for ref_num in ref_nums for n_view in range(1, self.config.n_view + 1)
        ]

        n_views_for_all_samples = self.get_n_views_for_all_samples(train=False)
        n_views_for_all_samples = np.array(n_views_for_all_samples)
        view_counts = np.bincount(n_views_for_all_samples, minlength=n_view + 1)  # 0, 1, ..., n_view, n_view + 1
        view_counts_dict = {
            f'n_view={i}': f'{(view_counts[i] / len(self) * 1e2):.2f}%, {view_counts[i]}/{len(self)}'
            for i in range(1, n_view + 1)
        }
        CONSOLE.log(f'Number of views of all samples: {view_counts_dict}')

        # ! we rearrange the multiview data here
        if np.sum(view_counts > 1) > 1 and kwargs.get('train', True):
            selected = np.zeros_like(n_views_for_all_samples)
            # TODO: should provide configs!!! --------------------
            view_data_ratio = {1: 0.4, 2: -1, 3: -1}
            # ----------------------------------------------------
            for i in range(1, n_view + 1):
                view_indices = np.where(n_views_for_all_samples == i)[0]
                view_selected = view_indices.copy()
                if view_data_ratio[i] > 0 and view_counts[i] / len(self) > view_data_ratio[i]:
                    CONSOLE.log(f'Will downsample n_view={i} data from {(view_counts[i] / len(self)):.2f} to {view_data_ratio[i]:.2f}')
                    view_data_size = min(view_counts[i], int(view_data_ratio[i] * len(self)))
                    view_selected = random.sample(view_indices.tolist(), view_data_size)
                    view_selected = np.array(view_selected)
                selected[view_selected] = 1
            CONSOLE.log(f'All downsampled data {selected.sum()}/{len(self)}, ratio: {(selected.sum()/len(self)):.2f}')
            self.samples = [self.samples[i] for i in range(len(self)) if bool(selected[i])]

    def get_n_views_for_all_samples(self, train: bool = True):
        n_views_for_all_samples = [self.config.n_view for _ in range(len(self))]

        for i in tqdm(range(len(self.samples))):
            sample = self.samples[i]
            is_exist = [int(sample[f'has_image_{i_view}']) for i_view in range(self.config.n_view)]
            n_view = sum(is_exist)

            # ! training called by dataloader
            if train and n_view > 1:
                use_n_view = random.randint(2, n_view)
                n_view = use_n_view

            n_views_for_all_samples[i] = n_view

        return n_views_for_all_samples

    def _aggregate_multivew(self, data: list[dict]) -> dict:
        """
        Args:
            data (list[dict]): len(data) = n_view.
        """

        returns = dict()

        for key in data[0].keys():

            returns[key] = [data[i][key] for i in range(len(data))]

            # perform 'v f c h w -> (v f) c h w' in torch, np, list formats.
            if isinstance(data[0][key], torch.Tensor):
                returns[key] = torch.concat(returns[key])
            if isinstance(data[0][key], np.ndarray):
                returns[key] = np.concatenate(returns[key])
            if key == 'pil_image':
                # Note here [0] is because we force n_view=1 in `RobotDataset.get_frames()`
                returns[key] = [image_list[0] for image_list in returns[key]]  # n_view -> n_ref_frame

        return returns

    def __getitem__(self, index_or_res: int | tuple) -> Dict[str, Any]:

        if not isinstance(index_or_res, int):
            index, _, use_n_view = index_or_res  # (index, ref_num, n_view)
        else:
            index = index_or_res
            use_n_view = self.config.n_view

        data = dict()

        sample = self.samples[index]

        ann_file = sample['ann_file']
        prompt = sample['prompt']
        frame_ids = sample['frame_ids']
        start_frame_idx = int(sample['start_frame_idx'])
        num_frame = int(sample['num_frame'])
        episode_id = int(sample['episode_id'])

        data['prompt'] = prompt if not self.config.empty_prompt else ''

        # initialize 'view_ids'
        view_ids = [i_view for i_view in range(self.config.n_view) if bool(sample[f'has_image_{i_view}'])]
        n_view = len(view_ids)
        has_n_view =len(view_ids)

        try:

            with open(ann_file, 'r') as f:
                label = json.load(f)

            # ! update view_ids if possible
            # Droid multivew data have fixed 2 vidws;
            # Bridgev2 multiview data have various 1~3 views;
            if use_n_view < self.config.n_view:
                n_view = use_n_view
                view_ids = random.sample(view_ids, n_view)

            sample_name = f'{episode_id:05d}_{start_frame_idx:02d}_{num_frame:02d}'

            # ! load prompt embeddings
            if self.config.load_tensor:

                empty_prompt_embeds_path = os.path.join(
                    self.config.data_root, self.config.embeddings_folder, 'empty_prompt.pt')
                prompt_embeds_path = os.path.join(
                    self.config.data_root, self.config.embeddings_folder, self.config.split, 'prompt_embeds', f'{sample_name}.pt'
                )

                if self.config.empty_prompt:
                    if os.path.exists(empty_prompt_embeds_path):
                        data['prompt_embeds'] = torch.load(
                            empty_prompt_embeds_path, weights_only=True
                        )[0]  # [seq_len, embed_dim] do not have `batch_size` dimension
                    else:
                        CONSOLE.log(f"[red]You're trying to load prompt embeddings but file not found"
                                    f"{empty_prompt_embeds_path=}.")
                else:
                    if os.path.exists(prompt_embeds_path):
                        data['prompt_embeds'] = torch.load(prompt_embeds_path, weights_only=True)  # [seq_len, embed_dim]
                    else:
                        CONSOLE.log(f"[red]You're trying to load prompt embeddings but file not found"
                                    f"{prompt_embeds_path=}.")

            # ! load states and actions
            if self.config.load_actions:
                arm_states, gripper_states = self._get_robot_states(label, frame_ids)
                actions = self._get_actions(arm_states, gripper_states)
                actions *= self.c_act_scaler

                data['actions'] = actions.float()

            # ! load frames
            video_paths = [label['videos'][i]['video_path'] for i in view_ids]

            image_paths = [
                os.path.join(
                    self.config.embeddings_folder, self.config.split, f'images{self.ref_num}', f"{sample_name}_{i}.png")
                for i in view_ids
            ]
            latent_ref_paths = [
                os.path.join(
                    self.config.embeddings_folder, self.config.split, f'image{self.ref_num}_latents', f'{sample_name}_{i}.pt')
                for i in view_ids
            ]
            # sliced_video_path = (
            #     os.path.join(
            #         self.config.embeddings_folder, self.config.split, 'videos', f'{sample_name}.mp4')
            #     if self.config.slice_frame else None
            # )
            latent_video_paths = [
                os.path.join(
                    self.config.embeddings_folder, self.config.split, 'video_latents', f"{sample_name}_{i}.pt")
                for i in view_ids
            ]

            data.update(
                self._aggregate_multivew(
                    [
                        self._get_frames(
                            frame_ids,
                            image_path=image_paths[i],
                            video_path=video_paths[i],
                            latent_video_path=latent_video_paths[i],
                            latent_ref_path=latent_ref_paths[i],
                            # sliced_video_path=sliced_video_path,
                            is_sliced=sample['is_sliced'],
                        )
                        for i in range(n_view)
                    ]
                )
            )

            # ! load conditions
            if self.config.use_cond:

                # TODO: finish this sanity check
                recon_file = sample['recon_file']
                label_file = sample['label_file']
                render_file = sample['render_file']
                # assert not (recon_file is None and label_file is None and render_file is None), f'Please set `render_file` if use_cond!'

                latent_depth_paths = [
                    os.path.join(
                        self.config.embeddings_folder, self.config.split, 'depth_latents', f'{sample_name}_{i}.pt')
                    for i in view_ids
                ]
                latent_label_paths = [
                    os.path.join(
                        self.config.embeddings_folder, self.config.split, 'label_latents', f'{sample_name}_{i}.pt')
                    for i in view_ids
                ]

                data.update(
                    self._get_cond_frames(
                        frame_ids=frame_ids,
                        recon_file_path=recon_file,
                        label_file_path=label_file,
                        render_file_path=render_file,
                        latent_depth_paths=latent_depth_paths,
                        latent_label_paths=latent_label_paths,
                        view_ids=view_ids,
                        num_view=has_n_view,
                    )
                )

            data['metainfo'] = {
                'episode_id': label['episode_id'],
                'camera_id': self.config.camera_ids,
                'frame_ids': frame_ids,
                'start_frame_idx': start_frame_idx,
                'num_frame': num_frame,
                'num_view': len(view_ids),  # <=len(camera_ids)
                'sample_name': sample_name,
            }

            return data

        except Exception as e:
            warnings.warn(f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                          f"(by randomly sampling another sample in the same dataset).")
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())

            if int(os.getenv('DEBUG', 0)):
                raise

            return self[np.random.randint(len(self.samples))]


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by reference frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self,
        data_source: RobotDataset | MultiViewRobotDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        train: bool = True,
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.train = train

        self.buckets = {resolution: [] for resolution in data_source.resolutions}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            CONSOLE.log(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        all_indices = []

        pairs = [(index, ref_num, n_view) for index, (ref_num, n_view) in enumerate(
                    zip(
                        self.data_source.get_ref_nums_for_all_samples(),
                        self.data_source.get_n_views_for_all_samples(train=self.train),
                    )
                )]
        if self.shuffle:
            random.shuffle(pairs)

        for pair in pairs:
            index, ref_num, n_view = pair
            self.buckets[(ref_num, n_view)].append((index, ref_num, n_view))
            if len(self.buckets[(ref_num, n_view)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(ref_num, n_view)])
                all_indices.extend(self.buckets[(ref_num, n_view)])
                del self.buckets[(ref_num, n_view)]
                self.buckets[(ref_num, n_view)] = []

        if not self.drop_last:

            for res, bucket in list(self.buckets.items()):
                if len(bucket) == 0:
                    continue
                if self.shuffle:
                    random.shuffle(bucket)
                    all_indices.extend(bucket)
                    del self.buckets[res]
                    self.buckets[res] = []

        CONSOLE.log(f'Bucket Sampler: {len(all_indices)=}, {all_indices[:10]=}')
        yield from all_indices


class CollateFunctionControl:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
            data keys: `action`, `latent`, `depths`, `semantics`, `metainfo`
            Returns:
                `actions`: [batch_size, n_frame - 1, 7]
                `latents`: [batch_size, n_frame, 4, fH, fW]
                `prompts`: List[str]
                `depths`:  [batch_size, n_frame, 1, H, W]
                `metainfos`: List[Dict[str, Any]]
        """

        returns = dict(
            controls=dict()
        )
        data_keys = data[0].keys()

        prompts = [x["prompt"] for x in data]
        returns["prompts"] = prompts

        if "prompt_embeds" in data_keys:
            prompt_embeds = [x["prompt_embeds"] for x in data]
            prompt_embeds = torch.stack(prompt_embeds).to(dtype=self.weight_dtype, non_blocking=True)
            returns["prompt_embeds"] = prompt_embeds

        if "actions" in data_keys:
            actions = [x["actions"] for x in data]
            actions = torch.stack(actions).to(dtype=self.weight_dtype, non_blocking=True)
            returns["controls"]["actions"] = actions

        if "videos" in data_keys:
            videos = [x["videos"] for x in data]
            videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)
            returns["videos"] = videos.permute(0, 2, 1, 3, 4)  # -> [B, C, F, H, W]

        if "latents" in data_keys:
            latents = [x["latents"] for x in data]
            latents = torch.stack(latents).to(dtype=self.weight_dtype, non_blocking=True)
            returns["latents"] = latents.permute(0, 2, 1, 3, 4)  # -> [B, C, F, H, W]

        if "image" in data_keys:
            images = [x["image"] for x in data]
            images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)
            returns['images'] = images.permute(0, 2, 1, 3, 4)  # -> [B, C, F, H, W] where F == n_view * dataset.ref_num

            fh, fw = images.shape[-2:]
            width = int(fw * 8)
            height = int(fh * 8)
            returns['image_width'] = width
            returns['image_height'] = height

        if "depths" in data_keys:
            depths = [x["depths"] for x in data]
            depths = torch.stack(depths).to(dtype=self.weight_dtype, non_blocking=True)
            returns["controls"]["depths"] = depths.permute(0, 2, 1, 3, 4)  # -> [B, C, F, H, W] where F == n_view * dataset.ref_num

        if "labels" in data_keys:
            labels = [x["labels"] for x in data]
            labels = torch.stack(labels).to(dtype=self.weight_dtype, non_blocking=True)
            returns["controls"]["labels"] = labels.permute(0, 2, 1, 3, 4)  # -> [B, C, F, H, W] where F == n_view * dataset.ref_num

        if 'latents_depth' in data_keys:
            latents_depth = [x['latents_depth'] for x in data]
            latents_depth = torch.stack(latents_depth).to(dtype=self.weight_dtype, non_blocking=True)
            returns['controls']['latents_depth'] = latents_depth.permute(0, 2, 1, 3, 4)  # -> [B, C, F, H, W]

        if 'latents_label' in data_keys:
            latents_label = [x['latents_label'] for x in data]
            latents_label = torch.stack(latents_label).to(dtype=self.weight_dtype, non_blocking=True)
            returns['controls']['latents_label'] = latents_label.permute(0, 2, 1, 3, 4)  # -> [B, C, F, H, W]

        # test mode
        if "pil_image" in data[0]:
            pil_images = [x["pil_image"] for x in data]  # n_batch -> n_view -> n_frame

            # we flatten the batched list [n_batch, n_view, n_frame] here.
            # then the processed image tensors have dimension [n_batch * n_view * n_frame, c, h ,w].
            pil_images = list(chain(*(chain(*pil_images))))
            returns["pil_images"] = pil_images

            width, height = pil_images[0].size  # n_batch * n_view * n_frame
            returns["image_width"] = width
            returns["image_height"] = height

            if 'sliced_frames' in data[0]:
                sliced_frames = [x['sliced_frames'] for x in data]
                sliced_frames = torch.stack(sliced_frames).to(dtype=self.weight_dtype, non_blocking=True)
                returns['sliced_frames'] = sliced_frames

        returns["metainfos"] = [x["metainfo"] for x in data]
        returns['num_views'] = returns['metainfos'][0]['num_view']
        returns['num_frames'] = returns['metainfos'][0]['num_frame']

        return returns


if __name__ == "__main__":
    import argparse
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_cond", action="store_true")
    parser.add_argument("--filter_by_cond", action="store_true")
    parser.add_argument("--ref_num", type=int, default=1)
    parser.add_argument("--load_tensor", action="store_true")
    parser.add_argument("--load_action", action="store_true")
    parser.add_argument("--slice_frame", action="store_true")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Whether or not to use the pinned memory setting in pytorch dataloader.",
    )
    args = parser.parse_args()
    weight_dtype = torch.float32
    os.environ['DEBUG'] = '0'

    # ================ RobotDataset =================
    # dataset_init_kwargs = {
    #     'data_root': './data/rt1',
    #     'use_cond': args.use_cond,
    #     'filter_by_cond': args.filter_by_cond,
    #     'ref_num': args.ref_num,
    #     'load_actions': args.load_action,
    #     'load_tensor': args.load_tensor,
    #     'load_condGT': False,
    #     'sequence_interval': 2,
    #     'control_keys': ['label', 'depth'],
    #     'slice_frame': args.slice_frame,
    #     'start_frame_interval': { 'train': 6,'val': 16,'test': 16 },
    #     'use_3dvae': True,
    #     'empty_prompt': True,
    #     'test_mode': args.test_mode,
    # }
    # dataset = RobotDataset(split=args.split, **dataset_init_kwargs)
    # breakpoint()

    # ================= MultiviewDataset =================
    dataset_init_kwargs = {
        'data_root': './data/bridgev2',
        'use_cond': args.use_cond,
        'filter_by_cond': args.filter_by_cond,
        'load_actions': args.load_action,
        'load_tensor': args.load_tensor,
        'slice_frame': args.slice_frame,
        'use_3dvae': True,
        'empty_prompt': True,
        'test_mode': args.test_mode,
        'n_view': 3,
        'camera_ids': [0, 1, 2],
    }
    dataset = MultiViewRobotDataset(split=args.split, **dataset_init_kwargs)
    data = dataset[0]

    collate_fn_control = CollateFunctionControl(weight_dtype, load_tensors=False)

    batch_size = args.batch_size
    # if not slice frame then different videos (batchs) have mismatched length!
    if not args.slice_frame:
        batch_size = 1

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=BucketSampler(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True),
        collate_fn=collate_fn_control,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )

    for data in tqdm(dataloader):
        controls = data['controls']
        tqdm.write(f"{data['num_frames']=}, {data['num_views']=}")
        if 'prompt_embeds' in data:
            tqdm.write(f"prompt_embeds: {data['prompt_embeds'].shape}")
        if 'actions' in data["controls"]:
            tqdm.write(f"actions: {data['controls']['actions'].shape}")
        if 'latents' in data:
            tqdm.write(f"latents: {data['latents'].shape}")
        if 'videos' in data:
            tqdm.write(f"videos: {data['videos'].shape}")
        if 'depths' in controls:
            tqdm.write(f"depths: {controls['depths'].shape}, {controls['depths'].min()}, {controls['depths'].max()}, {controls['depths'].mean()}")
        if 'labels' in controls:
            tqdm.write(f"labels: {controls['labels'].shape}, {controls['labels'].min()}, {controls['labels'].max()}, {controls['labels'].mean()}")
        if 'latents_depth' in controls:
            tqdm.write(f"latents_depth: {controls['latents_depth'].shape}")
        if 'latents_label' in controls:
            tqdm.write(f"latents_label: {controls['latents_label'].shape}")
        if 'pil_images' in data:
            tqdm.write(f"length of pil images: {len(data['pil_images'])}")
            tqdm.write(f"pil image size: {data['image_height']=}, {data['image_width']=}")
