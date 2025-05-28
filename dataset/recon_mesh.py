import os
import trimesh
import open3d as o3d
import numpy as np
import torch
import argparse
import fnmatch
import shutil
from tqdm import tqdm
from copy import deepcopy


config = {'depth': 10,
          'min_density': 0.1,
          'n_threads': -1,
          'downsample': False,
          'voxel_size': 0.5,
          'max_nn': 20,
          'pc_range':  [-50, -50, -5, 50, 50, 3],
          'occ_size':  [200, 200, 16],
          'self_range': [3.0, 3.0, 3.0]}


def preprocess_cloud(
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


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=n_threads
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities


def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original= None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)


def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def process_single_poisson(point_cloud_data, save_path):
    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(point_cloud_data.vertices)
    with_normal2 = o3d.geometry.PointCloud()
    with_normal = preprocess_cloud(point_cloud_original, max_nn=config['max_nn'])
    with_normal2.points = with_normal.points
    with_normal2.normals = with_normal.normals
    mesh, _ = create_mesh_from_map(buffer=None, depth=config['depth'], n_threads=config['n_threads'], min_density=config['min_density'], point_cloud_original=with_normal2)
    point_cloud_recon = np.asarray(mesh.vertices, dtype=float)

    mesh.paint_uniform_color((0, 1, 1))
    o3d.io.write_triangle_mesh(save_path, mesh)


def process_single_nksr(point_cloud_data, save_path):

    device = torch.device("cuda:0")
    reconstructor = nksr.Reconstructor(device)

    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(point_cloud_data.vertices)
    with_normal = preprocess_cloud(point_cloud_original, max_nn=config['max_nn'])
    
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='nksr')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    process_func = {'poisson': process_single_poisson, 'nksr': process_single_nksr}
    if args.mode not in process_func.keys():
        raise ValueError(f'Unsupported {args.mode}')
    if args.mode == 'nksr':
        try:
            import nksr
        except Exception as e:
            print(e)
            exit(1)

    data_dir = '/share/project/algorithm/xiuyu.yang/work/dev6/data/occ_robonet_monst3r/point'
    save_dir = '/share/project/algorithm/xiuyu.yang/work/dev6/data/occ_robonet_monst3r/mesh'
    splits = ['train', 'test']

    for split in tqdm(splits, "Processing split", leave=False):

        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue

        traj_files = os.listdir(split_dir)
        if args.demo:
            traj_files = ['berkeley_sawyer_traj26603']

        for traj_file in (
            pbar := tqdm(traj_files, leave=False)
        ):
            load_dir = os.path.join(data_dir, split, traj_file)
            save_folder = os.path.join(save_dir, split, traj_file)
            pbar.set_description(f"Processing {traj_file}")
        
            if os.path.exists(save_folder):
                mode = os.listdir(save_folder)[0].split('.')[0].split('_')[-1]
                if mode == args.mode:
                    tqdm.write(f"Skipped {split}-{traj_file}")
                    continue

            os.makedirs(save_folder, exist_ok=True)

            try:
                point_cloud_files = list(map(lambda fn: os.path.join(load_dir, fn),
                                             fnmatch.filter(os.listdir(load_dir), 'frame_*.ply')))
                for i, point_cloud_file in enumerate(tqdm(point_cloud_files, leave=False)):
                    point_cloud_data = trimesh.load(point_cloud_file)
                    save_path = os.path.join(save_folder, f'frame_{i:04d}_{args.mode}.ply')
                    process_func[args.mode](point_cloud_data, save_path)

            except Exception as e:
                print(e)
                if os.path.exists(save_folder):
                    shutil.rmtree(save_folder)
                if isinstance(e, KeyboardInterrupt):
                    exit(1)
                continue
