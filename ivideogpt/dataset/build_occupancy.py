import torch
import numpy as np
from numpy import typing as npt


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
                     determinstic: bool = True):
    try:
        from ivideogpt.ops.voxelize.voxelization import voxelization
    except:
        raise ImportError

    if isinstance(points, np.ndarray):
        points = torch.tensor(points, device='cuda', dtype=torch.float32)

    if labels is None:
        labels = torch.zeros_like(points[:, 0])
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels.astype(np.int32), device='cuda', dtype=torch.float32)
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