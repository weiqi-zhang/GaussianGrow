import torch
from torch import nn

import numpy as np

from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)
import torch.nn.functional as F

from gaussian_renderer import gs_render
from scene import Scene
from types import SimpleNamespace
from gaussian_grow.core.constants import VIEWPOINTS
from tqdm import tqdm 
import torchvision
from depthid_render import get_depth_with_id 
import os
from gaussian_grow.core.render import gen_rays_at
# ---------------- UTILS ----------------------

def degree_to_radian(d):
    return d * np.pi / 180

def radian_to_degree(r):
    return 180 * r / np.pi

def polar_to_xyz(theta, phi, dist):
    """ assume y-axis is the up axis """

    theta = degree_to_radian(theta)
    phi = degree_to_radian(phi)

    x = np.sin(phi) * np.sin(theta) * dist
    y = np.cos(phi) * dist
    z = np.sin(phi) * np.cos(theta) * dist

    return [x, y, z]

# ---------------- VIEWPOINTS ----------------------

def init_viewpoints(mode, sample_space, init_dist, init_elev, principle_directions, 
    use_principle=True, use_shapenet=False, use_objaverse=False):

    if mode == "predefined":

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list
        ) = init_predefined_viewpoints(sample_space, init_dist, init_elev)

    elif mode == "hemisphere":

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list
        ) = init_hemisphere_viewpoints(sample_space, init_dist)

    else:
        raise NotImplementedError()

    # punishments for views -> in case always selecting the same view
    view_punishments = [1 for _ in range(len(dist_list))]

    if use_principle:

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list,
            view_punishments,
            length
        ) = init_principle_viewpoints(
            principle_directions, 
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list,
            view_punishments,
            use_shapenet,
            use_objaverse
        )

    return dist_list, elev_list, azim_list, sector_list, view_punishments, length

def init_principle_viewpoints(
    principle_directions, 
    dist_list, 
    elev_list, 
    azim_list, 
    sector_list,
    view_punishments,
    use_shapenet=False,
    use_objaverse=False
):

    if use_shapenet:
        key = "shapenet"

        pre_elev_list = [v for v in VIEWPOINTS[key]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[key]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[key]["sector"]]

        num_principle = 10
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]

    elif use_objaverse:
        key = "objaverse"

        pre_elev_list = [v for v in VIEWPOINTS[key]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[key]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[key]["sector"]]

        num_principle = len(pre_azim_list)
        pre_dist_list = [1.0 for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]
    else:
        num_principle = 6
        pre_elev_list = [v for v in VIEWPOINTS[num_principle]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[num_principle]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[num_principle]["sector"]]
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]

    dist_list = pre_dist_list + dist_list
    elev_list = pre_elev_list + elev_list
    azim_list = pre_azim_list + azim_list
    sector_list = pre_sector_list + sector_list
    view_punishments = pre_view_punishments + view_punishments

    return dist_list, elev_list, azim_list, sector_list, view_punishments, len(pre_azim_list)

def init_predefined_viewpoints(sample_space, init_dist, init_elev):
    
    viewpoints = VIEWPOINTS[sample_space]

    assert sample_space == len(viewpoints["sector"])

    dist_list = [init_dist for _ in range(sample_space)] # always the same dist
    elev_list = [viewpoints["elev"][i] for i in range(sample_space)]
    azim_list = [viewpoints["azim"][i] for i in range(sample_space)]
    sector_list = [viewpoints["sector"][i] for i in range(sample_space)]
    return dist_list, elev_list, azim_list, sector_list

def init_hemisphere_viewpoints(sample_space, init_dist):
    """
        y is up-axis
    """

    num_points = 2 * sample_space
    ga = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    flags = []
    elev_list = [] # degree
    azim_list = [] # degree

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1

        # only take the north hemisphere
        if y >= 0: 
            flags.append(True)
        else:
            flags.append(False)

        theta = ga * i  # golden angle increment

        elev_list.append(radian_to_degree(np.arcsin(y)))
        azim_list.append(radian_to_degree(theta))

        radius = np.sqrt(1 - y * y)  # radius at y
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

    elev_list = [elev_list[i] for i in range(len(elev_list)) if flags[i]]
    azim_list = [azim_list[i] for i in range(len(azim_list)) if flags[i]]

    dist_list = [init_dist for _ in elev_list]
    sector_list = ["good" for _ in elev_list]

    return dist_list, elev_list, azim_list, sector_list

# ---------------- CAMERAS ----------------------

def init_camera(dist, elev, azim, image_size, device):
    R, T = look_at_view_transform(dist, elev, azim, device)
    image_size = torch.tensor([image_size, image_size]).unsqueeze(0)
    cameras = PerspectiveCameras(R=R, T=T, device=device, image_size=image_size)

    return cameras

def convert_camera_from_pytorch3d_to_colmap(
    p3d_cameras,
    height,
    width,
    device='cuda',
):
    """From a pytorch3d-compatible camera object and its camera matrices R, T, K, and width, height,
    outputs Gaussian Splatting camera parameters.

    Args:
        p3d_cameras (P3DCameras): R matrices should have shape (N, 3, 3),
            T matrices should have shape (N, 3, 1),
            K matrices should have shape (N, 3, 3).
        height (float): _description_
        width (float): _description_
        device (_type_, optional): _description_. Defaults to 'cuda'.
    """

    N = p3d_cameras.R.shape[0]
    if device is None:
        device = p3d_cameras.device

    if type(height) == torch.Tensor:
        height = int(torch.Tensor([[height.item()]]).to(device))
        width = int(torch.Tensor([[width.item()]]).to(device))
    else:
        height = int(height)
        width = int(width)

    # Inverse extrinsics
    R_inv = (p3d_cameras.R * torch.Tensor([-1.0, 1.0, -1]).to(device)).transpose(-1, -2)
    T_inv = (p3d_cameras.T * torch.Tensor([-1.0, 1.0, -1]).to(device)).unsqueeze(-1)
    world2cam_inv = torch.cat([R_inv, T_inv], dim=-1)
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    world2cam_inv = torch.cat([world2cam_inv, line], dim=-2)
    cam2world_inv = world2cam_inv.inverse()
    camera_to_worlds_inv = cam2world_inv[:, :3]
    
    for cam_idx in range(N):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = camera_to_worlds_inv[cam_idx]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0).cpu().numpy() #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = w2c[:3,:3]
        T = w2c[:3, 3]

    return R, T

def convert_camera_from_pytorch3d_to_colmap_torch(
    p3d_cameras,
    height,
    width,
    device='cuda',
):
    """From a pytorch3d-compatible camera object and its camera matrices R, T, K, and width, height,
    outputs Gaussian Splatting camera parameters.

    Args:
        p3d_cameras (P3DCameras): R matrices should have shape (N, 3, 3),
            T matrices should have shape (N, 3, 1),
            K matrices should have shape (N, 3, 3).
        height (float): _description_
        width (float): _description_
        device (_type_, optional): _description_. Defaults to 'cuda'.
    """

    N = p3d_cameras.R.shape[0]
    if device is None:
        device = p3d_cameras.device

    if type(height) == torch.Tensor:
        height = int(torch.Tensor([[height.item()]]).to(device))
        width = int(torch.Tensor([[width.item()]]).to(device))
    else:
        height = int(height)
        width = int(width)

    # Inverse extrinsics
    R_inv = (p3d_cameras.R * torch.Tensor([-1.0, 1.0, -1]).to(device)).transpose(-1, -2)
    T_inv = (p3d_cameras.T * torch.Tensor([-1.0, 1.0, -1]).to(device)).unsqueeze(-1)
    world2cam_inv = torch.cat([R_inv, T_inv], dim=-1)
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    world2cam_inv = torch.cat([world2cam_inv, line], dim=-2)
    cam2world_inv = world2cam_inv.inverse()
    camera_to_worlds_inv = cam2world_inv[:, :3]
    
    for cam_idx in range(N):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = camera_to_worlds_inv[cam_idx]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0) #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3,:3]
        T = w2c[:3, 3]

    return R, T

def convert_camera_from_pytorch3d_to_gs(
    p3d_cameras,
    height,
    width,
    device='cuda',
):
    """From a pytorch3d-compatible camera object and its camera matrices R, T, K, and width, height,
    outputs Gaussian Splatting camera parameters.

    Args:
        p3d_cameras (P3DCameras): R matrices should have shape (N, 3, 3),
            T matrices should have shape (N, 3, 1),
            K matrices should have shape (N, 3, 3).
        height (float): _description_
        width (float): _description_
        device (_type_, optional): _description_. Defaults to 'cuda'.
    """

    N = p3d_cameras.R.shape[0]
    if device is None:
        device = p3d_cameras.device

    if type(height) == torch.Tensor:
        height = int(torch.Tensor([[height.item()]]).to(device))
        width = int(torch.Tensor([[width.item()]]).to(device))
    else:
        height = int(height)
        width = int(width)

    # Inverse extrinsics
    R_inv = (p3d_cameras.R * torch.Tensor([-1.0, 1.0, -1]).to(device)).transpose(-1, -2)
    T_inv = (p3d_cameras.T * torch.Tensor([-1.0, 1.0, -1]).to(device)).unsqueeze(-1)
    world2cam_inv = torch.cat([R_inv, T_inv], dim=-1)
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    world2cam_inv = torch.cat([world2cam_inv, line], dim=-2)
    cam2world_inv = world2cam_inv.inverse()
    camera_to_worlds_inv = cam2world_inv[:, :3]
    
    for cam_idx in range(N):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = camera_to_worlds_inv[cam_idx]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0).cpu().numpy() #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]

    return R, T

def optimize_camera(gaussians, update_tensor, DEVICE, view_idx, camera_dir, image_size=256, threshold=0.5,  sample_ratio=0.2):
    
    dist = 1
    elev = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True))
    azim = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, requires_grad=True))

    xyz = gaussians._xyz

    l = [
        {'params': [elev], 'lr': 1e-3, "name": "elev"},
        {'params': [azim], 'lr': 1e-3, "name": "azim"},
    ]
   
    optimizer = torch.optim.Adam(l, lr=1e-3, eps=1e-15)

    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    bg_color = [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    camera_dir = os.path.join(camera_dir, f"view_idx_{view_idx}")
    os.makedirs(camera_dir, exist_ok=True)

    M = gaussians._xyz[update_tensor].shape[0]
    K = len(gaussians._xyz) - M

    sampled_M = max(1, int(M * 0.3))
    sampled_K = max(1, int(K * 0.15))
    idx_m = torch.randperm(M)[:sampled_M]
    idx_k = torch.randperm(K)[:sampled_K]

    pbar = tqdm(range(400), desc="Optimizing camera", unit="step")
    for step in pbar:
        elev_1 = elev * 360
        azim_1 = azim * 360
        
        optimizer.zero_grad()

        ones = torch.ones((xyz.shape[0], 1)).cuda()      
        points_w = torch.cat((xyz, ones), dim=1).permute(1, 0).double()
        camera = init_camera(dist, elev_1, azim_1, image_size, DEVICE)
        R, T = convert_camera_from_pytorch3d_to_colmap_torch(camera, image_size, image_size)
        Rt = torch.cat([
            torch.cat([R, T.unsqueeze(-1)], dim=1),  # R: (3,3), T: (3,1)
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=DEVICE)
        ], dim=0)

        world_view_transform = Rt.double()
        points_c = world_view_transform @ points_w
        points_c_norm =  points_c / points_c[3:, :]

        K = torch.zeros([3,4]).double().cuda()
        focal = 0.5 * image_size / np.tan(0.5 * 1)
        K[0,0] = image_size / 2
        K[1,1] = image_size / 2
        K[0,2] = image_size / 2
        K[1,2] = image_size / 2
        K[2,2] = 1
        points_pixel = K @ points_c
        points_pixel = points_pixel / points_pixel[2:, :]
        pc_pixel = points_pixel[:2, :]

        points_c_norm = points_c_norm.permute(1, 0)
        pc_pixel = pc_pixel.permute(1, 0)
        uv = pc_pixel
        points = points_c_norm
        mask = update_tensor

        mask_points = points[mask]      # [M, 3]
        mask_uv = uv[mask]              # [M, 2]
        non_mask_points = points[~mask] # [K, 3]
        non_mask_uv = uv[~mask]         # [K, 2]

        M, K = len(mask_points), len(non_mask_points)
    
        if M == 0 or K == 0:
            continue
        

        mask_z_sample = mask_points[idx_m, 2]
        mask_uv_sample = mask_uv[idx_m]

        non_mask_z_sample = non_mask_points[idx_k, 2]
        non_mask_uv_sample = non_mask_uv[idx_k]
        

        z_diff = mask_z_sample[:, None] - non_mask_z_sample[None, :]
        uv_diff = mask_uv_sample[:, None, :] - non_mask_uv_sample[None, :, :]
        distance = torch.norm(uv_diff, dim=-1)
        

        depth_cond = z_diff > 0
        overlap_cond = distance < threshold
        occ_loss = torch.sum(torch.sigmoid(z_diff) * (depth_cond & overlap_cond).float())
        occ_loss = occ_loss

        mask_z = points_c_norm[update_tensor, 2]
        z_loss = torch.mean(mask_z)

        loss = 0.0 * z_loss + occ_loss
        loss.backward()

        optimizer.step()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "elev": f"{elev_1.item():.1f}",
            "azim": f"{azim_1.item():.1f}"
        })

        with torch.no_grad():
            if step % 100 == 0:
                camera = init_camera(dist, elev_1, azim_1, image_size, DEVICE)
                R, T = convert_camera_from_pytorch3d_to_gs(camera, image_size, image_size)
                scene = Scene([R], [T], gaussians, image_size=1024)
                views = scene.getTrainCameras()
                view = views[0]
                render_pkg = gs_render(view, gaussians, pipeline, background)
                torchvision.utils.save_image(render_pkg['render'], f'{camera_dir}/learning_camera_vis_{step}.png')

    return elev.clone().detach() * 360, azim.clone().detach() * 360

def optimize_camera_v2(gaussians, update_tensor, DEVICE, view_idx_tmp, camera_dir, image_size=256, threshold=20,  sample_ratio=0.2):
    azim_list = [
            45, 135, 225, 315,
            45, 135, 225, 315,
            45, 135, 225, 315,
    ]

    elev_list = [
            30, 30, 30, 30,
            0, 0, 0, 0,
            -15, -15, -15, -15,
    ]

    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    bg_color = [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    max_num = -1
    max_idx = -1
    for view_idx in range(len(azim_list)):
        dist, elev, azim,  = 1, elev_list[view_idx], azim_list[view_idx]
        camera = init_camera(dist, elev, azim, 1024, DEVICE)
        

        R, T = convert_camera_from_pytorch3d_to_colmap(camera, 1024, 1024)
        points_w = gaussians._xyz.clone()
        ones = torch.ones(gaussians._xyz.shape[0], 1).cuda()
        points_w = torch.cat((points_w, ones), dim=1).permute(1, 0)
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0
        world_view_transform = torch.from_numpy(Rt).cuda().float()
        points_c = world_view_transform @ points_w
        points_c_norm =  points_c / points_c[3:, :]
        K = torch.zeros([3,4]).cuda()
        focal = 0.5 * 1024 / np.tan(0.5 * 1)
        K[0,0] = 1024 / 2
        K[1,1] = 1024 / 2
        K[0,2] = 1024 / 2
        K[1,2] = 1024 / 2
        K[2,2] = 1
        points_pixel = K @ points_c
        
        points_pixel = points_pixel / points_pixel[2:, :]
        pc_pixel = points_pixel[:2, :]
        
        x_grid = np.arange(1024)
        y_grid = np.arange(1024)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        grid_x = torch.from_numpy(grid_x).cuda()
        grid_y = torch.from_numpy(grid_y).cuda()

        R_gs, T_gs = convert_camera_from_pytorch3d_to_gs(camera, 1024,1024)
        scene = Scene([R_gs], [T_gs], gaussians, image_size=1024)
        view = scene.getTrainCameras()[0]

        rendering_results = gs_render(view, gaussians, pipeline, background)

        radii = rendering_results['radii']
    

        pick_tensor = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device=DEVICE)

        pix_depth, pix_id = get_depth_with_id(pc_pixel, radii.float(), points_c_norm[2,:], 1024, 1024)
        
        pix_id = pix_id.permute(1, 0, 2)

        pix_depth = pix_depth.permute(1, 0, 2)
        

        first_elements = pix_depth[:, :, 0].unsqueeze(-1)
        mask_tmp = pix_depth >= (first_elements + 0.01)
        pix_depth[mask_tmp] = -1
        pix_id[mask_tmp] = -1
        pix_id[:, :, 990:] = -1

        pix_id = pix_id.cpu()
        
        mask_tmp = (pix_id != -1)
        valid_ids = pix_id[mask_tmp].cuda()
        pick_tensor[valid_ids.long()] = True

        pick_tensor = torch.logical_and(pick_tensor, update_tensor)

        num = pick_tensor.sum()
        if num > max_num:
            max_num = num
            max_idx = view_idx

    dist = 1
    elev = torch.tensor(elev_list[max_idx], dtype=torch.float32)
    azim = nn.Parameter(torch.tensor(azim_list[max_idx], dtype=torch.float32, requires_grad=True))

    xyz = gaussians._xyz

    l = [
        {'params': [azim], 'lr': 1e-1, "name": "azim"},
    ]
   
    optimizer = torch.optim.Adam(l, lr=1e-1, eps=1e-15)

    camera_dir = os.path.join(camera_dir, f"view_idx_{view_idx_tmp}")
    os.makedirs(camera_dir, exist_ok=True)

    M = gaussians._xyz[update_tensor].shape[0]
    K = len(gaussians._xyz) - M

    sampled_M = min(2000, max(1, int(M * 0.5)))
    sampled_K = min(50000, max(1, int(K * 0.5)))

    idx_m = torch.randperm(M)[:sampled_M]
    idx_k = torch.randperm(K)[:sampled_K]

    pbar = tqdm(range(100), desc="Optimizing camera", unit="step")
    for step in pbar:
        
        optimizer.zero_grad()

        ones = torch.ones((xyz.shape[0], 1)).cuda()      
        points_w = torch.cat((xyz, ones), dim=1).permute(1, 0).double()
        camera = init_camera(dist, elev, azim, image_size, DEVICE)
        R, T = convert_camera_from_pytorch3d_to_colmap_torch(camera, image_size, image_size)
        Rt = torch.cat([
            torch.cat([R, T.unsqueeze(-1)], dim=1),  # R: (3,3), T: (3,1)
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=DEVICE)
        ], dim=0)

        world_view_transform = Rt.double()
        points_c = world_view_transform @ points_w
        points_c_norm =  points_c / points_c[3:, :]

        K = torch.zeros([3,4]).double().cuda()
        focal = 0.5 * image_size / np.tan(0.5 * 1)
        K[0,0] = image_size / 2
        K[1,1] = image_size / 2
        K[0,2] = image_size / 2
        K[1,2] = image_size / 2
        K[2,2] = 1
        points_pixel = K @ points_c
        points_pixel = points_pixel / points_pixel[2:, :]
        pc_pixel = points_pixel[:2, :]

        points_c_norm = points_c_norm.permute(1, 0)
        pc_pixel = pc_pixel.permute(1, 0)
        uv = pc_pixel
        points = points_c_norm
        mask = update_tensor

        mask_points = points[mask]      # [M, 3]
        mask_uv = uv[mask]              # [M, 2]
        non_mask_points = points[~mask] # [K, 3]
        non_mask_uv = uv[~mask]         # [K, 2]
        

        mask_z_sample = mask_points[idx_m, 2]
        mask_uv_sample = mask_uv[idx_m]

        non_mask_z_sample = non_mask_points[idx_k, 2]
        non_mask_uv_sample = non_mask_uv[idx_k]
        

        z_diff = mask_z_sample[:, None] - non_mask_z_sample[None, :]

        uv_diff = mask_uv_sample[:, None, :] - non_mask_uv_sample[None, :, :] # torch.Size([2000, 50000, 2])

        distance = torch.norm(uv_diff, dim=-1) ## torch.Size([2000, 50000])

        t = 100.0

        depth_prob = torch.sigmoid(z_diff * t) # torch.Size([2000, 50000])
        overlap_prob = torch.sigmoid((threshold - distance) * t) # torch.Size([2000, 50000])

        occ_matrix = depth_prob * overlap_prob

        occ_loss = torch.sum(occ_matrix)

        loss = occ_loss
        loss.backward()

        optimizer.step()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "azim": f"{azim.item():.1f}"
        })

        with torch.no_grad():
            if step % 100 == 0:
                camera = init_camera(dist, elev, azim, image_size, DEVICE)
                R, T = convert_camera_from_pytorch3d_to_gs(camera, image_size, image_size)
                scene = Scene([R], [T], gaussians, image_size=1024)
                views = scene.getTrainCameras()
                view = views[0]
                render_pkg = gs_render(view, gaussians, pipeline, background)
                torchvision.utils.save_image(render_pkg['render'], f'{camera_dir}/learning_camera_vis_{step}.png')

    with torch.no_grad():            
        camera = init_camera(dist, elev, azim, image_size, DEVICE)
        R, T = convert_camera_from_pytorch3d_to_gs(camera, image_size, image_size)
        scene = Scene([R], [T], gaussians, image_size=1024)
        views = scene.getTrainCameras()
        view = views[0]
        render_pkg = gs_render(view, gaussians, pipeline, background)
        torchvision.utils.save_image(render_pkg['render'], f'{camera_dir}/final_vis_.png')

    return elev.clone().detach(), azim.clone().detach()

def optimize_camera_cos(xyz, normals, azim_raw, image_size=512, DEVICE='cuda'):

    azim = nn.Parameter(torch.tensor(azim_raw, dtype=torch.float32, requires_grad=True, device=DEVICE))

    l = [
        {'params': [azim], 'lr': 1e-1, "name": "azim"},
    ]
    
    optimizer = torch.optim.Adam(l, lr=1e-1, eps=1e-15)
    pbar = tqdm(range(1000), desc="Optimizing camera", unit="step")
    
    for step in pbar:
        optimizer.zero_grad()
    
        camera = init_camera(1, 0, azim, image_size, DEVICE)

        R, T = convert_camera_from_pytorch3d_to_colmap_torch(camera, image_size, image_size)
        points_w = xyz
        ones = torch.ones(xyz.shape[0], 1).cuda()
        points_w = torch.cat((points_w, ones), dim=1).permute(1, 0)
        Rt = torch.cat([
            torch.cat([R, T.unsqueeze(-1)], dim=1),  # R: (3,3), T: (3,1)
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=DEVICE)
        ], dim=0)
        world_view_transform = Rt
        points_c = world_view_transform @ points_w
        K = torch.zeros([3,4]).cuda()
        focal = 0.5 * image_size / np.tan(0.5 * 1)
        K[0,0] = image_size / 2
        K[1,1] = image_size / 2
        K[0,2] = image_size / 2
        K[1,2] = image_size / 2
        K[2,2] = 1
        points_pixel = K @ points_c
        
        points_pixel = points_pixel / points_pixel[2:, :]
        pc_pixel = points_pixel[:2, :].permute(1, 0)

        pose = torch.eye(4, dtype=torch.float32, device=DEVICE)
        pose[:3, :3] = R
        pose[:3, 3] = T
        pose = torch.linalg.inv(pose)
        K_3 = torch.zeros([3,3]).cuda()
        focal = 0.5 * image_size / np.tan(0.5 * 1)
        K_3[0,0] = image_size / 2 
        K_3[1,1] = image_size / 2
        K_3[0,2] = image_size / 2
        K_3[1,2] = image_size / 2
        K_3[2,2] = 1
        intrinsics_inv = torch.linalg.inv(K_3)
        rays_o, rays_v = gen_rays_at(image_size, image_size, pose, intrinsics_inv)
        rays_v = rays_v.permute(2, 0, 1).unsqueeze(0)

        loss = -1 * compute_cosine_similarity(pc_pixel, normals, rays_v)
        loss.backward()
        optimizer.step()   

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "azim": f"{azim.item():.5f}"
        })

    return azim

def compute_cosine_similarity(
    points_uv: torch.Tensor,
    normals: torch.Tensor,
    ray_direction_field: torch.Tensor
) -> torch.Tensor:
    """
    Compute differentiable cosine similarity between point normals and camera rays.
    
    Args:
        points_uv: Projected UV coordinates in pixel space.
        normals: 3D point normals.
        ray_direction_field: Unit ray direction field with shape [1, 3, H, W].
    
    Returns:
        Cosine similarity values with shape [N].
    """
    H, W = ray_direction_field.shape[2], ray_direction_field.shape[3]
    N = points_uv.shape[0]
    

    grid_x = (points_uv[:, 0] / (W - 1)) * 2 - 1
    grid_y = (points_uv[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=1).view(1, N, 1, 2)
    

    sampled_directions = F.grid_sample(
        ray_direction_field, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    sampled_directions = sampled_directions.permute(0, 2, 3, 1).view(N, 3)
    

    sampled_directions = F.normalize(sampled_directions, p=2, dim=1, eps=1e-8)
    normals = F.normalize(normals, p=2, dim=1, eps=1e-8)
    

    cosine_similarities = torch.abs((sampled_directions * normals).sum(dim=1)).mean()
    return cosine_similarities

def visibility_loss(A_points, B_points, world2cam, image_size, k=100, eps=1e-6):
    """
    A_points: Tensor of shape (N, 3) in world coordinates.
    B_points: Tensor of shape (M, 3) in world coordinates.
    world2cam: Tensor of shape (4, 4), world2cam matrix.
    image_size: (width, height) of the image.
    """

    A_points = A_points
    B_points = B_points
    world2cam = world2cam
    A_cam = (world2cam[:3, :3] @ A_points.T + world2cam[:3, 3:]).T  # (N,3)
    B_cam = (world2cam[:3, :3] @ B_points.T + world2cam[:3, 3:]).T  # (M,3)
    

    u = A_cam[:, 0] / (A_cam[:, 2] + eps)
    v = A_cam[:, 1] / (A_cam[:, 2] + eps)
    W, H = image_size
    u_norm = (u + 1) / 2
    v_norm = (v + 1) / 2
    

    

    dA = torch.norm(A_cam, dim=1)  # (N,)
    

    vis_loss = 0.0
    for i in range(len(A_cam)):
        vi = A_cam[i] / (dA[i] + eps)
        

        vj = B_cam / (torch.norm(B_cam, dim=1, keepdim=True) + eps)  # (M,3)
        cos_theta = (vi * vj).sum(dim=1)  # (M,)
        w_dir = torch.sigmoid(10.0 * (cos_theta - 0.99))
        
        dB = torch.norm(B_cam, dim=1)  # (M,)
        w_depth = torch.sigmoid(10.0 * (dA[i] - dB + 0.1))
        
        w_total = w_dir * w_depth
        sum_w = w_total.sum()
        if sum_w > eps:
            d_occlusion = (w_total * dB).sum() / sum_w
        else:
            d_occlusion = dA[i] + 1e3
            

        p_occ = torch.sigmoid(10.0 * (d_occlusion - dA[i]))
        visibility = (1 - p_occ)
        vis_loss -= visibility
    
    return vis_loss / len(A_cam)
