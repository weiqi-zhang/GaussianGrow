import os
import copy
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from types import SimpleNamespace
from torchvision import transforms

from pytorch3d.renderer import TexturesUV
from pytorch3d.ops import interpolate_face_attributes
from scipy.ndimage import binary_dilation

import find_max_in_circles
from depthid_render import get_depth_with_id

from gaussian_renderer import gs_render
from gaussian_grow.core.camera_helper import init_camera, convert_camera_from_pytorch3d_to_colmap
from gaussian_grow.core.render import gen_rays_at, ray_marching
from gaussian_grow.core.render_helper import init_renderer, render
from gaussian_grow.core.shading_helper import (
    BlendParams,
    init_soft_phong_shader,
    init_flat_texel_shader,
)
from gaussian_grow.core.vis_helper import visualize_quad_mask
from gaussian_grow.core.constants import *

# Depth-occlusion tolerance: pixel depths farther than (first + EPS) are treated
# as occluded. depthid_render allocates 1000 slots per pixel; the tail 9 are
# unreliable padding (slot index 991+), so we mask them out.
OCCLUSION_DEPTH_EPS = 0.01
DEPTH_SLOT_LIMIT = 991
middle = float(np.log(np.sqrt(0.0000008)))

def get_all_4_locations(values_y, values_x):
    y_0 = torch.floor(values_y)
    y_1 = torch.ceil(values_y)
    x_0 = torch.floor(values_x)
    x_1 = torch.ceil(values_x)

    return torch.cat([y_0, y_0, y_1, y_1], 0).long(), torch.cat([x_0, x_1, x_0, x_1], 0).long()

def compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device):
    """
        compose quad mask:
            -> 0: background
            -> 1: old
            -> 2: update
            -> 3: new
    """

    new_mask_tensor = transforms.ToTensor()(new_mask_image).to(device)
    update_mask_tensor = transforms.ToTensor()(update_mask_image).to(device)
    old_mask_tensor = transforms.ToTensor()(old_mask_image).to(device)

    all_mask_tensor = new_mask_tensor + update_mask_tensor + old_mask_tensor

    quad_mask_tensor = torch.zeros_like(all_mask_tensor)
    quad_mask_tensor[old_mask_tensor == 1] = 1
    quad_mask_tensor[update_mask_tensor == 1] = 2
    quad_mask_tensor[new_mask_tensor == 1] = 3

    return old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor

def compute_view_heat(similarity_tensor, quad_mask_tensor):
    num_total_pixels = quad_mask_tensor.reshape(-1).shape[0]
    heat = 0
    for idx in QUAD_WEIGHTS:
        heat += (quad_mask_tensor == idx).sum() * QUAD_WEIGHTS[idx] / num_total_pixels

    return heat

@torch.no_grad()
def build_backproject_mask(mesh, faces, verts_uvs, 
    cameras, reference_image, faces_per_pixel, 
    image_size, uv_size, device):
    # construct pixel UVs
    renderer_scaled = init_renderer(cameras,
        shader=init_soft_phong_shader(
            camera=cameras,
            blend_params=BlendParams(),
            device=device),
        image_size=image_size, 
        faces_per_pixel=faces_per_pixel
    )
    fragments_scaled = renderer_scaled.rasterizer(mesh)

    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces.textures_idx]

    pixel_uvs = interpolate_face_attributes(
        fragments_scaled.pix_to_face, fragments_scaled.bary_coords, faces_verts_uvs
    )  # NxHsxWsxKx2
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(-1, 2)

    texture_locations_y, texture_locations_x = get_all_4_locations(
        (1 - pixel_uvs[:, 1]).reshape(-1) * (uv_size - 1),
        pixel_uvs[:, 0].reshape(-1) * (uv_size - 1)
    )
    K = faces_per_pixel

    texture_values = torch.from_numpy(np.array(reference_image.resize((image_size, image_size)))).float() / 255.
    texture_values = texture_values.to(device).unsqueeze(0).expand([4, -1, -1, -1]).unsqueeze(0).expand([K, -1, -1, -1, -1])

    # texture
    texture_tensor = torch.zeros(uv_size, uv_size, 3).to(device)
    texture_locations_y[texture_locations_y < 0] = 0
    texture_locations_y[texture_locations_y > 999] = 999
    texture_locations_x[texture_locations_x < 0] = 0
    texture_locations_x[texture_locations_x > 999] = 999
    texture_tensor[texture_locations_y, texture_locations_x, :] = texture_values.reshape(-1, 3)

    return texture_tensor[:, :, 0]

@torch.no_grad()
def build_diffusion_mask(mesh_stuff, 
    renderer, exist_texture, similarity_texture_cache, target_value, device, image_size, 
    smooth_mask=False, view_threshold=0.01):

    mesh, faces, verts_uvs = mesh_stuff
    # Rendering masks swaps texture maps, so avoid mutating the caller's mesh.
    mask_mesh = mesh.clone()

    # visible mask => the whole region
    exist_texture_expand = exist_texture.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device)
    mask_mesh.textures = TexturesUV(
        maps=torch.ones_like(exist_texture_expand),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    visible_mask_tensor, _, similarity_map_tensor, *_ = render(mask_mesh, renderer)

    # faces that are too rotated away from the viewpoint will be treated as invisible
    valid_mask_tensor = (similarity_map_tensor >= view_threshold).float()
    visible_mask_tensor *= valid_mask_tensor

    # nonexist mask <=> new mask
    exist_texture_expand = exist_texture.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device)
    mask_mesh.textures = TexturesUV(
        maps=1 - exist_texture_expand,
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    new_mask_tensor, *_ = render(mask_mesh, renderer)
    new_mask_tensor *= valid_mask_tensor

    # exist mask => visible mask - new mask
    exist_mask_tensor = visible_mask_tensor - new_mask_tensor
    exist_mask_tensor[exist_mask_tensor < 0] = 0

    # all update mask
    mask_mesh.textures = TexturesUV(
        maps=(
            similarity_texture_cache.argmax(0) == target_value
        ).float().unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    all_update_mask_tensor, *_ = render(mask_mesh, renderer)

    # current update mask => intersection between all update mask and exist mask
    update_mask_tensor = exist_mask_tensor * all_update_mask_tensor

    # keep mask => exist mask - update mask
    old_mask_tensor = exist_mask_tensor - update_mask_tensor

    # convert
    new_mask = new_mask_tensor[0].cpu().float().permute(2, 0, 1)
    new_mask = transforms.ToPILImage()(new_mask).convert("L")

    update_mask = update_mask_tensor[0].cpu().float().permute(2, 0, 1)
    update_mask = transforms.ToPILImage()(update_mask).convert("L")

    old_mask = old_mask_tensor[0].cpu().float().permute(2, 0, 1)
    old_mask = transforms.ToPILImage()(old_mask).convert("L")

    exist_mask = exist_mask_tensor[0].cpu().float().permute(2, 0, 1)
    exist_mask = transforms.ToPILImage()(exist_mask).convert("L")

    return new_mask, update_mask, old_mask, exist_mask

@torch.no_grad()
def build_diffusion_mask_gaussian(
    similarity_view_cache, target_value, device, image_size, similarity_map_tensor, visible_mask_tensor, exist_mask_tensor, camera, gaussians, views, R_list, T_list, select_tensor, new_gaussian,
    smooth_mask=False, view_threshold=0.0001, second=False):
    view = views[target_value]

    # visible mask => the whole region
    visible_mask_tensor = torch.where(visible_mask_tensor > 0.0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    visible_mask_tensor = visible_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
    # faces that are too rotated away from the viewpoint will be treated as invisible
    valid_mask_tensor = (similarity_map_tensor >= 0.0).float()
    if second:
        valid_mask_tensor = valid_mask_tensor.squeeze(-1)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        image = valid_mask_tensor.unsqueeze(0)

        edge_x = F.conv2d(image, sobel_kernel_x, padding=1)
        edge_y = F.conv2d(image, sobel_kernel_y, padding=1)

        edges = torch.sqrt(edge_x**2 + edge_y**2)

        threshold = 0.2
        edges = edges.squeeze()
        image_no_edges = image.squeeze()
        image_no_edges[edges > threshold] = 0

        image_no_edges = image_no_edges[1:-1, 1:-1]
        image_no_edges_padded = F.pad(image_no_edges.unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0).squeeze(0)
        valid_mask_tensor = image_no_edges_padded.unsqueeze(-1).unsqueeze(0)

    visible_mask_tensor *= valid_mask_tensor

    # exist mask => visible mask - new mask
    exist_mask_tensor = torch.where(exist_mask_tensor > 0.1, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    exist_mask_tensor = exist_mask_tensor * visible_mask_tensor

    # nonexist mask <=> new mask
    new_mask_tensor = visible_mask_tensor - exist_mask_tensor
    new_mask_tensor[new_mask_tensor < 0] = 0

    if new_gaussian:
        similarity_idx_cache = torch.zeros([len(R_list), gaussians._xyz.shape[0]]).to(device)
        for idx in range(len(R_list)):
            R = R_list[idx]
            T = T_list[idx]
            tmp_view = views[idx]

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
            focal = 0.5 * image_size / np.tan(0.5 * 1)
            K[0,0] = image_size / 2
            K[1,1] = image_size / 2
            K[0,2] = image_size / 2
            K[1,2] = image_size / 2
            K[2,2] = 1
            points_pixel = K @ points_c
            points_pixel = points_pixel / points_pixel[2:, :]
            pc_pixel = points_pixel[:2, :]        

            pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
            bg_color = [1,1,1]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            rendering_results = gs_render(tmp_view, gaussians, pipeline, background)
            radii = rendering_results['radii']

            cos_similarity = similarity_view_cache[idx]
            results = find_max_in_circles.find_max_in_circles(cos_similarity, pc_pixel, radii.float())

            pix_depth, pix_id = get_depth_with_id(pc_pixel, radii.float(), points_c_norm[2,:], image_size, image_size)

            pix_id = pix_id.permute(1, 0, 2)

            pix_depth = pix_depth.permute(1, 0, 2)

            first_elements = pix_depth[:, :, 0].unsqueeze(-1)
            mask_tmp = pix_depth >= (first_elements + OCCLUSION_DEPTH_EPS)
            pix_depth[mask_tmp] = -1
            pix_id[mask_tmp] = -1
            pix_id[:, :, DEPTH_SLOT_LIMIT:] = -1

            mask_tmp = (pix_id != -1)
            valid_ids = pix_id[mask_tmp]
            update_tensor = torch.zeros([gaussians._xyz.shape[0]], dtype=torch.bool).cuda()
            update_tensor[valid_ids.long()] = True
            update_tensor[~gaussians._update.bool()] = False
            results[~update_tensor.bool()] = -1
            similarity_idx_cache[idx] = results
        # select_tenosr 118624
        all_update_idx_tensor_0 = (similarity_idx_cache[target_value] >= 0.8).bool()
        all_update_idx_tensor = (similarity_idx_cache.argmax(0) == target_value).bool()
        sel_gaussian = copy.deepcopy(gaussians)
        all_update_idx_tensor = torch.logical_and(all_update_idx_tensor.bool(), select_tensor.bool())
        sel_gaussian.select_gaussian(all_update_idx_tensor.bool())
        res = gs_render(view, sel_gaussian, pipeline, background)
        all_update_mask_tensor = res['rendered_alpha'].unsqueeze(-1).repeat(1, 1, 1, 3)
        all_update_mask_tensor = torch.where(all_update_mask_tensor > 0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())    
    else:
        all_update_mask_tensor = torch.zeros([1, image_size, image_size, 3]).cuda()
    # current update mask => intersection between all update mask and exist mask
    update_mask_tensor = exist_mask_tensor * all_update_mask_tensor
    # keep mask => exist mask - update mask
    old_mask_tensor = exist_mask_tensor - update_mask_tensor

    # convert
    new_mask = new_mask_tensor[0].cpu().float().permute(2, 0, 1)
    new_mask = transforms.ToPILImage()(new_mask).convert("L")

    update_mask = update_mask_tensor[0].cpu().float().permute(2, 0, 1)
    update_mask = transforms.ToPILImage()(update_mask).convert("L")

    old_mask = old_mask_tensor[0].cpu().float().permute(2, 0, 1)
    old_mask = transforms.ToPILImage()(old_mask).convert("L")

    exist_mask = exist_mask_tensor[0].cpu().float().permute(2, 0, 1)
    exist_mask = transforms.ToPILImage()(exist_mask).convert("L")

    return new_mask, update_mask, old_mask, exist_mask

@torch.no_grad()
def build_diffusion_mask_gaussian_repaint(
    visible_mask_tensor, exist_mask_tensor):

    # visible mask => the whole region
    visible_mask_tensor = torch.where(visible_mask_tensor > 0.0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    visible_mask_tensor = visible_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)

    # exist mask => visible mask - new mask
    visible_mask_tensor = erode_mask(visible_mask_tensor, erosion_pixels=5)
    exist_mask_tensor = torch.where(exist_mask_tensor > 0.01, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    exist_mask_tensor = exist_mask_tensor * visible_mask_tensor

    # nonexist mask <=> new mask
    new_mask_tensor = visible_mask_tensor - exist_mask_tensor
    new_mask_tensor[new_mask_tensor < 0] = 0

    # convert
    new_mask = new_mask_tensor[0].cpu().float().permute(2, 0, 1)
    new_mask = transforms.ToPILImage()(new_mask).convert("L")

    exist_mask = exist_mask_tensor[0].cpu().float().permute(2, 0, 1)
    exist_mask = transforms.ToPILImage()(exist_mask).convert("L")

    return new_mask, exist_mask

@torch.no_grad()
def render_one_view(
    dist, elev, azim,
    image_size, faces_per_pixel,
    device):

    # render the view
    cameras = init_camera(
        dist, elev, azim,
        image_size, device
    )
    return cameras

@torch.no_grad()
def build_similarity_gaussian_cache_for_all_views(mesh, faces, verts_uvs,
    dist_list, elev_list, azim_list,
    image_size, image_size_scaled, uv_size, faces_per_pixel,
    device):

    num_candidate_views = len(dist_list)
    similarity_texture_cache = torch.zeros(num_candidate_views, uv_size, uv_size).to(device)
    similarity_view_cache = torch.zeros(num_candidate_views, image_size, image_size).to(device)

    print("=> building similarity gaussian cache for all views...")
    for i in tqdm(range(num_candidate_views)):
        cameras, _, _, _, similarity_tensor, _, _ = render_one_view(mesh,
            dist_list[i], elev_list[i], azim_list[i],
            image_size, faces_per_pixel, device)
        similarity_view_cache[i] = similarity_tensor.reshape(image_size, image_size)

        similarity_texture_cache[i] = build_backproject_mask(mesh, faces, verts_uvs, 
            cameras, transforms.ToPILImage()(similarity_tensor[0, :, :, 0]).convert("RGB"), faces_per_pixel,
            image_size_scaled, uv_size, device)

    return similarity_texture_cache, similarity_view_cache

def build_similarity_gaussian_cache_for_all_views_gaussian(
    dist_list, elev_list, azim_list,
    image_size, image_size_scaled, uv_size, faces_per_pixel,
    device, udf_network):

    num_candidate_views = len(dist_list)
    similarity_texture_cache = torch.zeros(num_candidate_views, uv_size, uv_size).to(device)
    similarity_view_cache = torch.zeros(num_candidate_views, image_size, image_size).to(device)

    print("=> building similarity gaussian cache for all views...")
    for i in tqdm(range(num_candidate_views)):
        cameras = render_one_view(
            dist_list[i], elev_list[i], azim_list[i],
            image_size, faces_per_pixel, device)
        R, T = convert_camera_from_pytorch3d_to_colmap(cameras, image_size, image_size)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = T
        pose = np.linalg.inv(pose)
        K = np.eye(3, dtype=np.float32)
        K[0,0] = image_size / 2
        K[1,1] = image_size / 2
        K[0,2] = image_size / 2
        K[1,2] = image_size / 2
        intrinsics_inv = np.linalg.inv(K)
        rays_o, rays_v = gen_rays_at(image_size, image_size, torch.from_numpy(pose), torch.from_numpy(intrinsics_inv))
        d_pred_out = ray_marching(rays_o.cuda().reshape(1, -1, 3), rays_v.cuda().reshape(1, -1, 3), udf_network, tau=0.001)
        d_pred_out = d_pred_out.reshape(image_size , image_size, 1)
        point = rays_o.cuda() + d_pred_out * rays_v.cuda()
        mask = ~(torch.isnan(point) | torch.isinf(point))
        valid_points = point[mask].reshape(-1, 3)
        gradient = udf_network.gradient(valid_points).squeeze(1)
        norm = torch.norm(gradient, p=2, dim=1, keepdim=True)
        gradient_normalized = gradient / norm

        normal_map = torch.zeros_like(point).cuda()
        normal_map[mask] = gradient_normalized.reshape(-1)
        cosine_similarity = torch.abs(torch.nn.CosineSimilarity(dim=2)(rays_v.cuda(), normal_map.cuda()))

        cosine_similarity = cosine_similarity.detach().cpu().unsqueeze(-1).repeat(1, 1, 3)
        cosine_similarity[~mask] = 0
        similarity_view_cache[i] = cosine_similarity[:,:,0].reshape(image_size, image_size)

    return similarity_view_cache

def build_similarity_gaussian_cache_for_all_views_gaussian_2(
    dist_list, elev_list, azim_list,
    image_size, image_size_scaled, uv_size, faces_per_pixel,
    gaussians, scene,
    device, udf_network):

    num_candidate_views = len(dist_list)
    views = scene.getTrainCameras()
    bg_color = [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    similarity_view_cache = torch.zeros(num_candidate_views, image_size, image_size).to(device)

    print("=> building similarity gaussian cache for all views...")
    for i in tqdm(range(num_candidate_views)):
        cameras = render_one_view(
            dist_list[i], elev_list[i], azim_list[i],
            image_size, faces_per_pixel, device)
        R, T = convert_camera_from_pytorch3d_to_colmap(cameras, image_size, image_size)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = T
        pose = np.linalg.inv(pose)
        K = np.eye(3, dtype=np.float32)
        K[0,0] = image_size / 2
        K[1,1] = image_size / 2
        K[0,2] = image_size / 2
        K[1,2] = image_size / 2
        intrinsics_inv = np.linalg.inv(K)
        _, rays_v = gen_rays_at(image_size, image_size, torch.from_numpy(pose), torch.from_numpy(intrinsics_inv))
        view = views[i]
        rendering_results = gs_render(view, gaussians, pipeline, background)
        normal_map = rendering_results['rend_normal']
        depth_map = rendering_results['surf_depth']
        normal_map = normal_map.permute(1, 2, 0)

        cosine_similarity = torch.abs(torch.nn.CosineSimilarity(dim=2)(rays_v.cuda(), normal_map.cuda()))

        cosine_similarity = cosine_similarity.detach().cpu().unsqueeze(-1).repeat(1, 1, 3)
        mask = ~(torch.isnan(cosine_similarity) | torch.isinf(cosine_similarity))
        cosine_similarity[~mask] = 0
        similarity_view_cache[i] = cosine_similarity[:,:,0].reshape(image_size, image_size)

    return similarity_view_cache

@torch.no_grad()
def render_one_view_and_build_masks(dist, elev, azim, 
    selected_view_idx, view_idx, view_punishments,
    similarity_texture_cache, exist_texture,
    mesh, faces, verts_uvs,
    image_size, faces_per_pixel,
    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
    device, save_intermediate=False, smooth_mask=False, view_threshold=0.01):
    
    # render the view
    (
        cameras, renderer,
        init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments
    ) = render_one_view(mesh,
        dist, elev, azim,
        image_size, faces_per_pixel,
        device
    )
    
    init_image = init_images_tensor[0].cpu()
    init_image = init_image.permute(2, 0, 1)
    init_image = transforms.ToPILImage()(init_image).convert("RGB")

    normal_map = normal_maps_tensor[0].cpu()
    normal_map = normal_map.permute(2, 0, 1)
    normal_map = transforms.ToPILImage()(normal_map).convert("RGB")

    depth_map = depth_maps_tensor[0].cpu().numpy()
    depth_map = Image.fromarray(depth_map).convert("L")

    similarity_map = similarity_tensor[0, :, :, 0].cpu()
    similarity_map = transforms.ToPILImage()(similarity_map).convert("L")

    flat_renderer = init_renderer(cameras,
        shader=init_flat_texel_shader(
            camera=cameras,
            device=device),
        image_size=image_size,
        faces_per_pixel=faces_per_pixel
    )
    new_mask_image, update_mask_image, old_mask_image, exist_mask_image = build_diffusion_mask(
        (mesh, faces, verts_uvs), 
        flat_renderer, exist_texture, similarity_texture_cache, selected_view_idx, device, image_size, 
        smooth_mask=smooth_mask, view_threshold=view_threshold
    )
    # selected_view_idx is the sample-space index used by similarity_texture_cache.

    (
        old_mask_tensor, 
        update_mask_tensor, 
        new_mask_tensor, 
        all_mask_tensor, 
        quad_mask_tensor
    ) = compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device)

    view_heat = compute_view_heat(similarity_tensor, quad_mask_tensor)
    view_heat *= view_punishments[selected_view_idx]

    # save intermediate results
    if save_intermediate:
        init_image.save(os.path.join(init_image_dir, "{}.png".format(view_idx)))
        normal_map.save(os.path.join(normal_map_dir, "{}.png".format(view_idx)))
        depth_map.save(os.path.join(depth_map_dir, "{}.png".format(view_idx)))
        similarity_map.save(os.path.join(similarity_map_dir, "{}.png".format(view_idx)))

        new_mask_image.save(os.path.join(mask_image_dir, "{}_new.png".format(view_idx)))
        update_mask_image.save(os.path.join(mask_image_dir, "{}_update.png".format(view_idx)))
        old_mask_image.save(os.path.join(mask_image_dir, "{}_old.png".format(view_idx)))
        exist_mask_image.save(os.path.join(mask_image_dir, "{}_exist.png".format(view_idx)))

        visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_heat, device)

    return (
        view_heat,
        renderer, cameras, fragments,
        init_image, normal_map, depth_map, 
        init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
        old_mask_image, update_mask_image, new_mask_image, 
        old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor
    )

def render_normal_and_position(dists, elevs, azims, image_size, faces_per_pixel, device, udf_network,save_dir):
    posi_dir=os.path.join(save_dir,'position_map')
    normal_dir=os.path.join(save_dir,'normal_map')
    os.makedirs(posi_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    
    length = len(dists)
    for idx in range(length):
        dist = dists[idx]
        elev = elevs[idx]
        azim = azims[idx]
        cameras = render_one_view(
            dist, elev, azim,
            image_size, faces_per_pixel,
            device
        )
        R, T = convert_camera_from_pytorch3d_to_colmap(cameras, image_size, image_size)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = T
        pose = np.linalg.inv(pose)
        K = np.eye(3, dtype=np.float32)
        focal = 0.5 * image_size / np.tan(0.5 * 1)
        K[0,0] = image_size / 2
        K[1,1] = image_size / 2
        K[0,2] = image_size / 2
        K[1,2] = image_size / 2
        intrinsics_inv = np.linalg.inv(K)
        rays_o, rays_v = gen_rays_at(image_size, image_size, torch.from_numpy(pose), torch.from_numpy(intrinsics_inv))

        d_pred_out = ray_marching(rays_o.cuda().reshape(1, -1, 3), rays_v.cuda().reshape(1, -1, 3), udf_network, n_steps=[999, 1000], tau=0.008)
        d_pred_out = d_pred_out.reshape(image_size , image_size, 1)
        point = rays_o.cuda() + d_pred_out * rays_v.cuda()
        mask = ~(torch.isnan(point) | torch.isinf(point))
        valid_points = point[mask].reshape(-1, 3)
        gradient = udf_network.gradient(valid_points).squeeze(1)
        with torch.no_grad():
            gradient[: , 2] = -1 * gradient[: , 2]
            gradient = gradient[: , [0,2,1]]
            norm = torch.norm(gradient, p=2, dim=1, keepdim=True)
            gradient_normalized = gradient / norm
            normal_map = torch.ones_like(point).cuda()
            normal_map[mask] = gradient_normalized.reshape(-1)
            normal_maps_tensor = normal_map.reshape(image_size, image_size, 3).unsqueeze(0)
            normal_map = normal_maps_tensor.squeeze(0)

            normal_map = (normal_map + 1)* 0.5
            normal_map = normal_map.detach().cpu() * 255
            normal_map = normal_map.clamp(0, 255).numpy()
            normal_map = normal_map.astype(np.uint8)
            normal_map = Image.fromarray(normal_map)
            normal_map.save(os.path.join(normal_dir,"normal_map_{}.png".format(idx)))

            position_map = torch.ones_like(point)
            scale_factor = 1.15
            max_bb = (valid_points - 0).max(0)[0]
            min_bb = (valid_points - 0).min(0)[0]
            center = (max_bb + min_bb) / 2
            scale = torch.norm(valid_points - center, dim=1).max() * 2.0
            
            valid_points[:, 0] = -1 * valid_points[:, 0]
            valid_points[:, 1] = -1 * valid_points[:, 1]
            eps=1e-4
            position_map[mask] = 0.5 - ((valid_points - center) * (scale_factor / (float(scale)+eps))).reshape(-1) / scale_factor
            position_map = position_map[:, :, [0, 2, 1]]

            position_map = position_map.detach().cpu().numpy() * 255
            position_map = position_map.astype(np.uint8)
            position_map = Image.fromarray(position_map)
            position_map.save(os.path.join(posi_dir,"position_map_{}.png".format(idx)))
        
def render_one_view_and_build_masks_gaussian(dist, elev, azim, 
    selected_view_idx, view_idx, view_punishments,
    similarity_view_cache,
    image_size, faces_per_pixel,
    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, gaussian_dir,
    device, 
    scene,
    udf_network, new_gaussian, R_list, T_list, gaussians,
    save_intermediate=False, smooth_mask=False, view_threshold=0.01, second=False):
    
    # render the view
    cameras = render_one_view(
        dist, elev, azim,
        image_size, faces_per_pixel,
        device
    )

    views = scene.getTrainCameras()
    view = views[0]
    bg_color = [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    rendering_results = gs_render(view, scene.gaussians, pipeline, background)
    rend_normal = rendering_results['rend_normal']
    torchvision.utils.save_image(rend_normal, os.path.join(normal_map_dir, "{}_rend_normal.png".format(view_idx)))
    surf_normal = rendering_results['surf_normal']
    torchvision.utils.save_image(surf_normal, os.path.join(normal_map_dir, "{}_surf_normal.png".format(view_idx)))
    rend_dist = rendering_results['rend_dist']
    torchvision.utils.save_image(rend_dist, os.path.join(depth_map_dir, "{}_rend_dist.png".format(view_idx)))
    surf_depth = rendering_results['surf_depth']

    torchvision.utils.save_image(rendering_results["render"], os.path.join(init_image_dir, "gs-{}.png".format(view_idx)))
    init_image = Image.open(os.path.join(init_image_dir, "gs-{}.png".format(view_idx))).convert("RGB")
    init_image.save(os.path.join(init_image_dir, "raw-{}.png".format(view_idx)))
    image_array = np.array(init_image)
    init_images_tensor = torch.from_numpy(image_array).unsqueeze(0).cuda() / 255.0
    init_image = Image.fromarray(image_array, 'RGB')

    R, T = convert_camera_from_pytorch3d_to_colmap(cameras, image_size, image_size)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = T
    pose = np.linalg.inv(pose)
    K = np.eye(3, dtype=np.float32)
    K[0,0] = image_size / 2
    K[1,1] = image_size / 2
    K[0,2] = image_size / 2
    K[1,2] = image_size / 2
    intrinsics_inv = np.linalg.inv(K)
    rays_o, rays_v = gen_rays_at(image_size, image_size, torch.from_numpy(pose), torch.from_numpy(intrinsics_inv))

    d_pred_out = ray_marching(rays_o.cuda().reshape(1, -1, 3), rays_v.cuda().reshape(1, -1, 3), udf_network, tau=0.01, n_steps=[299, 300])
    d_pred_out = d_pred_out.reshape(image_size , image_size, 1)
    normal_map=rend_normal
    normal_maps_tensor = normal_map.permute(1 ,2 ,0).unsqueeze(0)
    cosine_similarity = torch.abs(torch.nn.CosineSimilarity(dim=2)(rays_v.cuda(), normal_map.permute(1, 2, 0).cuda()))
    normal_map = transforms.ToPILImage()(normal_map).convert("RGB")

    similarity_tensor = cosine_similarity.unsqueeze(0).unsqueeze(-1)
    non_zero_similarity = (similarity_tensor > 0).float()
    non_zero_similarity = (non_zero_similarity * 255.).cpu().numpy().astype(np.uint8)[0]
    non_zero_similarity = cv2.erode(non_zero_similarity, kernel=np.ones((3, 3), np.uint8), iterations=2)
    non_zero_similarity = torch.from_numpy(non_zero_similarity).to(similarity_tensor.device).unsqueeze(0) / 255.
    similarity_tensor = non_zero_similarity.unsqueeze(-1) * similarity_tensor

    similarity_map = similarity_tensor[0, :, :, 0].cpu()
    similarity_map = transforms.ToPILImage()(similarity_map).convert("L")

    depth = d_pred_out.reshape(-1)
    depth = torch.where(torch.isnan(depth), torch.tensor(10.0).cuda(), depth)
    depth = torch.where(torch.isinf(depth), torch.tensor(10.0).cuda(), depth)
    no_depth = 8
    pad_value = 0
    depth_min, depth_max = depth[depth < no_depth].min(), depth[depth < no_depth].max()
    target_min, target_max = 15, 255
    depth_value = depth[depth < no_depth]
    depth_value = depth_max - depth_value # reverse values
    depth_value /= (depth_max - depth_min)
    depth_value = depth_value * (target_max - target_min) + target_min
    depth_maps_tensor = depth.clone()
    depth_maps_tensor[depth < no_depth] = depth_value
    depth_maps_tensor[depth >= no_depth] = pad_value
    depth_maps_tensor = depth_maps_tensor.reshape(1, image_size, image_size)
    depth_map = depth_maps_tensor[0].cpu().numpy()
    depth_map = Image.fromarray(depth_map).convert("L")

    

    radii = rendering_results['radii']
    mid = int((radii.max() + radii.min()) / 2)
    visibility_filter = radii >= mid
    
    rendered_alpha = rendering_results['rendered_alpha'] # torch.Size([1, 1024, 1024]) 
    rendered_alpha = rendered_alpha.reshape(1, image_size, image_size)

    if new_gaussian == None:
        exist_mask_tensor = torch.zeros([1, image_size, image_size, 3]).cuda()
        update_tensor = None
    else:
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
        focal = 0.5 * image_size / np.tan(0.5 * 1)
        K[0,0] = image_size / 2
        K[1,1] = image_size / 2
        K[0,2] = image_size / 2
        K[1,2] = image_size / 2
        K[2,2] = 1
        points_pixel = K @ points_c
        
        points_pixel = points_pixel / points_pixel[2:, :]
        pc_pixel = points_pixel[:2, :]

        
        x_grid = np.arange(image_size)
        y_grid = np.arange(image_size)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        grid_x = torch.from_numpy(grid_x).cuda()
        grid_y = torch.from_numpy(grid_y).cuda()

        rendering_results = gs_render(view, gaussians, pipeline, background)
        radii = rendering_results['radii']
        update_tensor = torch.zeros([gaussians._xyz.shape[0]], dtype=torch.bool).cuda()
        pix_depth, pix_id = get_depth_with_id(pc_pixel, radii.float(), points_c_norm[2,:], image_size, image_size)

        pix_id = pix_id.permute(1, 0, 2)

        pix_depth = pix_depth.permute(1, 0, 2)

        first_elements = pix_depth[:, :, 0].unsqueeze(-1)
        mask_tmp = pix_depth >= (first_elements + OCCLUSION_DEPTH_EPS)
        pix_depth[mask_tmp] = -1
        pix_id[mask_tmp] = -1
        pix_id[:, :, DEPTH_SLOT_LIMIT:] = -1
        
        mask_tmp = (pix_id != -1)
        valid_ids = pix_id[mask_tmp]
        update_tensor[valid_ids.long()] = True

        update_tensor[~gaussians._update.bool()] = False
        opt_gaussian = copy.deepcopy(gaussians)
        opt_gaussian.select_gaussian(update_tensor)

        render_res = gs_render(view, opt_gaussian, pipeline, background)
        exist_mask_tensor = render_res['rendered_alpha']
        exist_mask_tensor = exist_mask_tensor.reshape(1, image_size, image_size).unsqueeze(-1).repeat(1, 1, 1, 3)
        exist_mask_tensor[exist_mask_tensor != 0] = 1

        exist_mask_tensor = exist_mask_tensor.permute(0, 3, 1, 2)

        gray = exist_mask_tensor.mean(dim=1, keepdim=True)

        binary = (gray > 0.5).float()

        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32).to(exist_mask_tensor.device)
        dilated = F.conv2d(binary, kernel, padding=1)
        eroded = F.conv2d(binary, kernel, padding=1)
        edges = dilated - eroded

        kernel_5 = torch.ones((1, 1, 5, 5), dtype=torch.float32).to(exist_mask_tensor.device)
        eroded_5 = F.conv2d(binary, kernel_5, padding=2)
        eroded_5 = (eroded_5 >= 25).float()
        exist_mask_tensor = eroded_5.permute(0, 2, 3, 1)

    new_mask_image, update_mask_image, old_mask_image, exist_mask_image = build_diffusion_mask_gaussian(
        similarity_view_cache, selected_view_idx, device, image_size, similarity_tensor, depth_maps_tensor, exist_mask_tensor, cameras, gaussians, views, R_list, T_list, update_tensor, new_gaussian,
        smooth_mask=smooth_mask, view_threshold=view_threshold, second=second
    )
    # selected_view_idx is the sample-space index used by similarity_view_cache.

    (
        old_mask_tensor, 
        update_mask_tensor, 
        new_mask_tensor, 
        all_mask_tensor, 
        quad_mask_tensor
    ) = compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device)

    view_heat = compute_view_heat(similarity_tensor, quad_mask_tensor)
    view_heat *= view_punishments[selected_view_idx]

    # save intermediate results
    if save_intermediate:
        init_image.save(os.path.join(init_image_dir, "{}.png".format(view_idx)))
        normal_map.save(os.path.join(normal_map_dir, "{}.png".format(view_idx)))
        depth_map.save(os.path.join(depth_map_dir, "{}.png".format(view_idx)))
        similarity_map.save(os.path.join(similarity_map_dir, "{}.png".format(view_idx)))

        new_mask_image.save(os.path.join(mask_image_dir, "{}_new.png".format(view_idx)))
        update_mask_image.save(os.path.join(mask_image_dir, "{}_update.png".format(view_idx)))
        old_mask_image.save(os.path.join(mask_image_dir, "{}_old.png".format(view_idx)))
        exist_mask_image.save(os.path.join(mask_image_dir, "{}_exist.png".format(view_idx)))

        visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_heat, device)

    return (
        view_heat,
        cameras,
        init_image, normal_map, depth_map, 
        init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
        old_mask_image, update_mask_image, new_mask_image, 
        old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor, visibility_filter
    )

def render_one_view_and_build_masks_gaussian_repaint(dist, elev, azim, 
    selected_view_idx, view_idx,
    image_size, faces_per_pixel,
    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, gaussian_dir,
    device, 
    scene,
    udf_network, gaussians,
    save_intermediate=False, second=True):
    
    # render the view
    cameras = render_one_view(
        dist, elev, azim,
        image_size, faces_per_pixel,
        device
    )

    views = scene.getTrainCameras()
    if second:
        view = views[0]
    else:
        view = views[view_idx]
    bg_color = [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    rendering_results = gs_render(view, scene.gaussians, pipeline, background)
    rend_normal = rendering_results['rend_normal']
    torchvision.utils.save_image(rend_normal, os.path.join(normal_map_dir, "{}_rend_normal.png".format(view_idx)))
    surf_normal = rendering_results['surf_normal']
    torchvision.utils.save_image(surf_normal, os.path.join(normal_map_dir, "{}_surf_normal.png".format(view_idx)))
    rend_dist = rendering_results['rend_dist']
    torchvision.utils.save_image(rend_dist, os.path.join(depth_map_dir, "{}_rend_dist.png".format(view_idx)))

    torchvision.utils.save_image(rendering_results["render"], os.path.join(init_image_dir, "gs-{}.png".format(view_idx)))
    init_image = Image.open(os.path.join(init_image_dir, "gs-{}.png".format(view_idx))).convert("RGB")
    init_image.save(os.path.join(init_image_dir, "raw-{}.png".format(view_idx)))
    image_array = np.array(init_image)
    init_images_tensor = torch.from_numpy(image_array).unsqueeze(0).cuda() / 255.0
    init_image = Image.fromarray(image_array, 'RGB')

    R, T = convert_camera_from_pytorch3d_to_colmap(cameras, image_size, image_size)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = T
    pose = np.linalg.inv(pose)
    K = np.eye(3, dtype=np.float32)
    focal = 0.5 * image_size / np.tan(0.5 * 1)
    K[0,0] = image_size / 2
    K[1,1] = image_size / 2
    K[0,2] = image_size / 2
    K[1,2] = image_size / 2
    intrinsics_inv = np.linalg.inv(K)
    rays_o, rays_v = gen_rays_at(image_size, image_size, torch.from_numpy(pose), torch.from_numpy(intrinsics_inv))

    d_pred_out = ray_marching(rays_o.cuda().reshape(1, -1, 3), rays_v.cuda().reshape(1, -1, 3), udf_network, tau=0.01, n_steps=[199, 200])
    d_pred_out = d_pred_out.reshape(image_size , image_size, 1)
    depth = d_pred_out.reshape(-1)
    depth = torch.where(torch.isnan(depth), torch.tensor(10.0).cuda(), depth)
    depth = torch.where(torch.isinf(depth), torch.tensor(10.0).cuda(), depth)
    no_depth = 8
    pad_value = 0
    depth_min, depth_max = depth[depth < no_depth].min(), depth[depth < no_depth].max()
    target_min, target_max = 15, 255
    depth_value = depth[depth < no_depth]
    depth_value = depth_max - depth_value # reverse values
    depth_value /= (depth_max - depth_min)
    depth_value = depth_value * (target_max - target_min) + target_min
    depth_maps_tensor = depth.clone()
    depth_maps_tensor[depth < no_depth] = depth_value
    depth_maps_tensor[depth >= no_depth] = pad_value
    depth_maps_tensor = depth_maps_tensor.reshape(1, image_size, image_size)
    depth_map = depth_maps_tensor[0].cpu().numpy()
    depth_map = Image.fromarray(depth_map).convert("L")

    R, T = convert_camera_from_pytorch3d_to_colmap(cameras, image_size, image_size)
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
    focal = 0.5 * image_size / np.tan(0.5 * 1)
    K[0,0] = image_size / 2
    K[1,1] = image_size / 2
    K[0,2] = image_size / 2
    K[1,2] = image_size / 2
    K[2,2] = 1
    points_pixel = K @ points_c
    
    points_pixel = points_pixel / points_pixel[2:, :]
    pc_pixel = points_pixel[:2, :]

    
    x_grid = np.arange(image_size)
    y_grid = np.arange(image_size)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_x = torch.from_numpy(grid_x).cuda()
    grid_y = torch.from_numpy(grid_y).cuda()

    rendering_results = gs_render(view, gaussians, pipeline, background)
    radii = rendering_results['radii']
    update_tensor = torch.zeros([gaussians._xyz.shape[0]], dtype=torch.bool).cuda()
    pix_depth, pix_id = get_depth_with_id(pc_pixel, radii.float(), points_c_norm[2,:], image_size, image_size)

    pix_id = pix_id.permute(1, 0, 2)

    pix_depth = pix_depth.permute(1, 0, 2)

    first_elements = pix_depth[:, :, 0].unsqueeze(-1)
    mask_tmp = pix_depth >= (first_elements + OCCLUSION_DEPTH_EPS)
    pix_depth[mask_tmp] = -1
    pix_id[mask_tmp] = -1
    pix_id[:, :, DEPTH_SLOT_LIMIT:] = -1
    
    mask_tmp = (pix_id != -1)
    valid_ids = pix_id[mask_tmp]
    update_tensor[valid_ids.long()] = True

    update_tensor[~gaussians._update.bool()] = False
    opt_gaussian = copy.deepcopy(gaussians)
    opt_gaussian.select_gaussian(update_tensor)

    render_res = gs_render(view, opt_gaussian, pipeline, background)
    exist_mask_tensor = render_res['rendered_alpha']
    exist_mask_tensor = exist_mask_tensor.reshape(1, image_size, image_size).unsqueeze(-1).repeat(1, 1, 1, 3)
    exist_mask_tensor[exist_mask_tensor != 0] = 1
    exist_mask_tensor_raw = exist_mask_tensor.clone() # [1, 1024, 1024, 3]

    new_mask_image, exist_mask_image = build_diffusion_mask_gaussian_repaint(
        depth_maps_tensor, exist_mask_tensor
    )
    mask_array = np.array(new_mask_image)

    binary_mask = mask_array > 0
    

    dilated_mask = binary_dilation(binary_mask, iterations=5)

    new_mask_image = Image.fromarray(dilated_mask.astype(np.uint8) * 255)

    new_mask_image_raw, exist_mask_image_raw = build_diffusion_mask_gaussian_repaint(
        depth_maps_tensor, exist_mask_tensor_raw
    )

    # selected_view_idx is the sample-space index used by similarity_view_cache.

    new_mask_tensor = transforms.ToTensor()(new_mask_image).to(device)
    new_mask_tensor_raw = transforms.ToTensor()(new_mask_image_raw).to(device)

    # save intermediate results
    if save_intermediate:
        init_image.save(os.path.join(init_image_dir, "{}.png".format(view_idx)))
        depth_map.save(os.path.join(depth_map_dir, "{}.png".format(view_idx)))

        new_mask_image.save(os.path.join(mask_image_dir, "{}_new.png".format(view_idx)))
        new_mask_image_raw.save(os.path.join(mask_image_dir, "{}_new_raw.png".format(view_idx)))
        exist_mask_image.save(os.path.join(mask_image_dir, "{}_exist.png".format(view_idx)))

    return (
        cameras,
        init_image, depth_map, 
        init_images_tensor, depth_maps_tensor, 
        new_mask_image, exist_mask_image,
        new_mask_tensor, exist_mask_tensor,
        new_mask_image_raw, exist_mask_image_raw,
        new_mask_tensor_raw, exist_mask_tensor_raw,
    )

def erode_mask(mask, erosion_pixels=7):
    """
    Erode a binary NHWC mask tensor.
    
    Args:
        mask: Binary mask tensor with shape [1, H, W, 3].
        erosion_pixels: Number of pixels to erode.
    
    Returns:
        Eroded mask with the same shape as the input.
    """

    mask_nchw = mask.permute(0, 3, 1, 2).float()
    

    kernel_size = 2 * erosion_pixels + 1
    padding = erosion_pixels
    

    kernel = torch.ones((3, 1, kernel_size, kernel_size), 
                        dtype=torch.float32, 
                        device=mask.device)
    

    convolved = F.conv2d(mask_nchw, kernel, padding=padding, groups=3)
    

    eroded = (convolved == (kernel_size**2))
    

    eroded_nhwc = eroded.permute(0, 2, 3, 1).float()
    
    return eroded_nhwc
    

    
