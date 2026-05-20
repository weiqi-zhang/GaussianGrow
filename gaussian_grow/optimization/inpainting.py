import torch
from scipy.spatial import cKDTree
import copy
from torch import nn
import numpy as np
from kornia.geometry import conversions
import torch.nn.functional as F


def quaternion_to_normal(quaternions):
    rotation_matrices = conversions.quaternion_to_rotation_matrix(
        quaternions,
        order=conversions.QuaternionCoeffOrder.WXYZ
    )
    direction_vectors = rotation_matrices[:, :, 2]
    direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True)
    return direction_vectors

def f_normal_score(noramls1, normals2):
    with torch.no_grad():
        cos_sim = F.cosine_similarity(noramls1.unsqueeze(1), normals2, dim=2)
        cos_sim = torch.where((cos_sim >= -1) & (cos_sim < 0.5), torch.tensor(1e-8, device=cos_sim.device), cos_sim)
        cos_sim = torch.where((cos_sim >= 0.9) & (cos_sim <= 1), torch.tensor(2.0, device=cos_sim.device), cos_sim)

    return cos_sim
    
def dis_score(distance):
    inverse_distance = 1 / (distance + 1e-8)
    sum_inverse_distance = inverse_distance.sum(dim=1, keepdim=True)
    weights = inverse_distance / sum_inverse_distance
    return weights

def update_colored_points(gaussians, gaussian_dir):
    update_tensor = gaussians._update.clone().bool()
    no_opt_gaussian = copy.deepcopy(gaussians)
    no_opt_gaussian.select_gaussian(~update_tensor.bool())
    opt_gaussian = copy.deepcopy(gaussians)
    opt_gaussian.select_gaussian(update_tensor.bool())

    ptree = cKDTree(opt_gaussian._xyz.clone().detach().cpu().numpy())

    xyz_array = no_opt_gaussian._xyz.clone().detach().cpu().numpy()
    rotation_array = no_opt_gaussian._rotation.clone().detach().cpu().numpy()
    combined_array = np.concatenate([xyz_array, rotation_array], axis=-1)

    splits = np.array_split(combined_array, 1, axis=0)
    for split_idx, p in enumerate(splits):
        no_opt_xyz = p[:, :3]
        no_opt_rot = p[:, 3:7]
        distances, indices = ptree.query(no_opt_xyz, 91) # [num_points, N_neighbours]
        
        distances = torch.from_numpy(distances).cuda()
        indices = torch.from_numpy(indices).cuda()

        xyz = opt_gaussian._xyz[indices]
        features_dc = opt_gaussian._features_dc[indices]
        features_rest = opt_gaussian._features_rest[indices]
        opacity = opt_gaussian._opacity[indices]
        scaling = opt_gaussian._scaling[indices]
        rotation = opt_gaussian._rotation[indices]

        dis_weight = dis_score(distances)

        no_opt_gaussian_normals = quaternion_to_normal(torch.from_numpy(no_opt_rot).cuda())
        batch, num = rotation.shape[0], rotation.shape[1]
        opt_gaussian_normals = quaternion_to_normal(rotation.reshape(-1, 4))
        opt_gaussian_normals = opt_gaussian_normals.reshape(batch, num, 3)

        normal_weight = f_normal_score(no_opt_gaussian_normals, opt_gaussian_normals)
        
        weight = dis_weight * normal_weight

        no_opt_gaussian._features_dc = (weight.unsqueeze(-1).unsqueeze(-1) * features_dc).sum(dim=1).float()
        no_opt_gaussian._features_rest = (weight.unsqueeze(-1).unsqueeze(-1) * features_rest).sum(dim=1).float()

    gaussians._xyz = nn.Parameter(torch.cat((no_opt_gaussian._xyz, opt_gaussian._xyz), dim=0).requires_grad_(True))
    gaussians._features_dc = nn.Parameter(torch.cat((no_opt_gaussian._features_dc, opt_gaussian._features_dc), dim=0).requires_grad_(True))
    gaussians._features_rest = nn.Parameter(torch.cat((no_opt_gaussian._features_rest, opt_gaussian._features_rest), dim=0).requires_grad_(True))
    gaussians._opacity = nn.Parameter(torch.cat((no_opt_gaussian._opacity, opt_gaussian._opacity), dim=0).requires_grad_(True))
    gaussians._scaling = nn.Parameter(torch.cat((no_opt_gaussian._scaling, opt_gaussian._scaling), dim=0).requires_grad_(True))
    gaussians._rotation = nn.Parameter(torch.cat((no_opt_gaussian._rotation, opt_gaussian._rotation), dim=0).requires_grad_(True))
    gaussians._update[~update_tensor] = 1
    
    return gaussians
