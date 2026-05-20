import numpy as np
import torch
from utils.sh_utils import RGB2SH
import open3d as o3d
from plyfile import PlyData, PlyElement
import os
from kornia.geometry import conversions


def write_gaussian_ply(data, path):
    xyz = data[:, :3]
    normals = np.zeros_like(xyz)
    f_dc = data[:, 3:6]
    f_rest = data[:, 6:51]
    opacities = data[:,51:52]
    scale = data[:,52:54]
    rotation = data[:,54:58]

    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(2):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l

    write_path = path
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(write_path)

def create_rotation_matrix_from_direction_vector_batch(direction_vectors):
    # Normalize the batch of direction vectors
    direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True)
    # Create a batch of arbitrary vectors that are not collinear with the direction vectors
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).to(direction_vectors.device).expand(direction_vectors.shape[0], -1).clone()
    is_collinear = torch.all(torch.abs(direction_vectors - v1) < 1e-5, dim=-1)
    v1[is_collinear] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).to(direction_vectors.device)
    
    # Calculate the first orthogonal vectors
    v1 = torch.cross(direction_vectors, v1)
    v1_norm = torch.norm(v1, dim=-1, keepdim=True)
    v1 = torch.where(v1_norm > 1e-8, v1 / v1_norm, v1)

    # Calculate the second orthogonal vectors by taking the cross product
    v2 = torch.cross(direction_vectors, v1)
    v2_norm = torch.norm(v2, dim=-1, keepdim=True)
    v2 = torch.where(v2_norm > 1e-8, v2 / v2_norm, v2)

    # Create the batch of rotation matrices with the direction vectors as the last columns
    rotation_matrices = torch.stack((v1, v2, direction_vectors), dim=-1)
    return rotation_matrices

def quaternion_to_normal(quaternions):
    rotation_matrices = conversions.quaternion_to_rotation_matrix(
        quaternions,
        order=conversions.QuaternionCoeffOrder.WXYZ
    )

    direction_vectors = rotation_matrices[:, :, 2]
    
    direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True)
    return direction_vectors


def convert_ply(input_dir):
    pcd = o3d.io.read_point_cloud(os.path.join(input_dir, "example.ply"))

    point_cloud = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    colors = np.zeros([point_cloud.shape[0], 3])
    colors[:, 0]  = 0.6
    colors[:, 1]  = 0.6
    colors[:, 2]  = 0.6

    gaussian = np.zeros([point_cloud.shape[0], 58])
    gaussian[:, :3] = point_cloud
    fused_color = RGB2SH(np.asarray(colors))

    features = np.zeros((fused_color.shape[0], 3, (3 + 1) ** 2))
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0


    features_dc = features[:,:,0:1].reshape(features.shape[0], -1)
    features_rest = features[:,:,1:].reshape(features.shape[0], -1)
    gaussian[:,3:6] = features_dc
    gaussian[:, 6:51] = features_rest
    gaussian[:,51] = 5
    gaussian[:,52:54] = -5.5 # 2.753644934974715785741109710242

    rotations = create_rotation_matrix_from_direction_vector_batch(torch.from_numpy(normals).float())
    quaternions = conversions.rotation_matrix_to_quaternion(rotations,eps=1e-5, order=conversions.QuaternionCoeffOrder.WXYZ)

    gaussian[:,54:58] = quaternions.numpy()

    write_gaussian_ply(gaussian, os.path.join(input_dir, 'point_cloud.ply'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Convert example.ply to point_cloud.ply.")
    parser.add_argument("input_dir", help="Directory containing example.ply.")
    args = parser.parse_args()
    convert_ply(args.input_dir)
    
