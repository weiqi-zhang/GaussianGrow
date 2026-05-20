import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from gaussian_grow.release_utils import add_capudf_args, ensure_capudf_checkpoint, require_cuda

from gaussian_renderer import gs_render
from arguments import OptimizationParams
from depthid_render import get_depth_with_id
from gaussian_grow.core.camera_helper import (
    convert_camera_from_pytorch3d_to_colmap,
    convert_camera_from_pytorch3d_to_gs,
    init_camera,
    init_viewpoints,
    optimize_camera_cos,
)
from gaussian_grow.core.io_helper import save_args
from gaussian_grow.core.projection_helper import render_normal_and_position
from gaussian_grow.core.render import gen_rays_at, ray_marching
from gaussian_grow.point_cloud_conversion import convert_ply, quaternion_to_normal
from scene import Scene
from scene.fields import CAPUDFNetwork
from scene.gaussian_model import GaussianModel

# depthid_render allocates 1000 depth slots per pixel; the last 10 are unreliable
# padding from kernel writes and must be cleared before downstream use.
MAX_DEPTH_SLOTS = 990


def init_args():
    print("=> initializing input arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--obj_name", type=str, required=True)
    parser.add_argument("--obj_file", type=str, required=True)
    parser.add_argument("--a_prompt", type=str, default="best quality, high quality, extremely detailed, good geometry, pure color, less strip")
    parser.add_argument("--n_prompt", type=str, default="deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke")
    parser.add_argument("--new_strength", type=float, default=1)
    parser.add_argument("--new_second_strength", type=float, default=0.3)
    parser.add_argument("--update_strength", type=float, default=0.5)
    parser.add_argument("--update_second_strength", type=float, default=0.3)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=10)
    parser.add_argument("--output_scale", type=float, default=1)
    parser.add_argument("--view_threshold", type=float, default=0.1)
    parser.add_argument("--num_viewpoints", type=int, default=8)
    parser.add_argument("--viewpoint_mode", type=str, default="predefined", choices=["predefined", "hemisphere"])
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_shapenet", action="store_true", help="operate on ShapeNet objects")
    parser.add_argument("--use_objaverse", action="store_true", help="operate on Objaverse objects")
    parser.add_argument("--no_repaint", action="store_true", help="do NOT apply repaint")

    parser.add_argument("--device", type=str, choices=["a6000", "2080"], default="a6000")

    parser.add_argument("--dist", type=float, default=1,
        help="distance to the camera from the object")
    parser.add_argument("--elev", type=float, default=0,
        help="the angle between the vector from the object to the camera and the horizontal plane")
    parser.add_argument("--azim", type=float, default=180,
        help="the angle between the vector from the object to the camera and the vertical plane")
    add_capudf_args(parser)

    args = parser.parse_args()
    op = OptimizationParams(parser)

    if args.device == "a6000":
        setattr(args, "render_simple_factor", 12)
        setattr(args, "fragment_k", 1)
        setattr(args, "image_size", 512)
        setattr(args, "uv_size", 3000)
    else:
        setattr(args, "render_simple_factor", 4)
        setattr(args, "fragment_k", 1)
        setattr(args, "image_size", 512)
        setattr(args, "uv_size", 1000)

    return args, op

if __name__ == "__main__":
    args, op = init_args()
    DEVICE = require_cuda()

    # Resolve output directory.
    output_dir = args.output_dir
    if args.no_repaint:
        output_dir += "-norepaint"

    os.makedirs(output_dir, exist_ok=True)
    print("=> OUTPUT_DIR:", output_dir)
    name=args.obj_name
    raw = os.path.join(args.input_dir, args.obj_file)
    des = os.path.join(output_dir, 'example.ply')

    pcd = o3d.io.read_point_cloud(raw)
    points = np.asarray(pcd.points)
    num_points = len(points)
    print(f"Loaded point cloud with {num_points} points.")
    
    if num_points > 100000:
        print("Point cloud has more than 100,000 points; downsampling.")
        sampling_ratio = 100000 / num_points
        pcd = pcd.random_down_sample(sampling_ratio)
        print(f"Downsampled point cloud to {len(np.asarray(pcd.points))} points.")

    print("Estimating point-cloud normals.")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    print("Finished normal estimation.")
    
    o3d.io.write_point_cloud(des, pcd)
    print(f"Saved point cloud to: {des}")

    ckpt_path = ensure_capudf_checkpoint(args, name, output_dir)
    convert_ply(output_dir)

    gaussians = GaussianModel(0)
    gaussians.load_ply(os.path.join(output_dir, 'point_cloud.ply'))

    with torch.no_grad():
        gaussians.set_scale()
    udf_network = CAPUDFNetwork(
            d_out=1,
            d_in=3,
            d_hidden=256,
            n_layers=8,
            skip_in=[4],
            multires=0,
            bias=0.5,
            scale=1.0,
            geometric_init=True,
            weight_norm=True
        ).cuda()

    udf_network.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda'))["udf_network_fine"])
    udf_network = udf_network.eval()
    # Stage 1 needs the principal views plus four optimized transition views.
    principle_directions = None
    (
        dist_list, 
        elev_list, 
        azim_list, 
        sector_list,
        view_punishments,
        length
    ) = init_viewpoints(args.viewpoint_mode, args.num_viewpoints, args.dist, args.elev, principle_directions, 
                            use_principle=True, 
                            use_shapenet=args.use_shapenet,
                            use_objaverse=args.use_objaverse)
    
    save_args(args, output_dir)
    generate_dir = os.path.join(output_dir, "generate")
    os.makedirs(generate_dir, exist_ok=True)

    update_dir = os.path.join(output_dir, "update")
    os.makedirs(update_dir, exist_ok=True)

    init_image_dir = os.path.join(generate_dir, "rendering")
    os.makedirs(init_image_dir, exist_ok=True)

    normal_map_dir = os.path.join(generate_dir, "normal")
    os.makedirs(normal_map_dir, exist_ok=True)

    mask_image_dir = os.path.join(generate_dir, "mask")
    os.makedirs(mask_image_dir, exist_ok=True)

    depth_map_dir = os.path.join(generate_dir, "depth")
    os.makedirs(depth_map_dir, exist_ok=True)

    similarity_map_dir = os.path.join(generate_dir, "similarity")
    os.makedirs(similarity_map_dir, exist_ok=True)

    inpainted_image_dir = os.path.join(generate_dir, "inpainted")
    os.makedirs(inpainted_image_dir, exist_ok=True)

    mesh_dir = os.path.join(generate_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    interm_dir = os.path.join(generate_dir, "intermediate")
    os.makedirs(interm_dir, exist_ok=True)

    gaussian_dir = os.path.join(generate_dir, "gaussian")
    os.makedirs(gaussian_dir, exist_ok=True)

    # Prepare viewpoints and camera cache.
    NUM_PRINCIPLE = length

    pre_dist_list = dist_list[:NUM_PRINCIPLE]
    pre_elev_list = elev_list[:NUM_PRINCIPLE]
    pre_azim_list = azim_list[:NUM_PRINCIPLE]
    pre_sector_list = sector_list[:NUM_PRINCIPLE]
    pre_view_punishments = view_punishments[:NUM_PRINCIPLE]

    R_list = []
    T_list = []
    R_raw_list = []
    T_raw_list = []
    for view_idx in range(NUM_PRINCIPLE):
        dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx]
        camera = init_camera(dist, elev, azim, args.image_size, DEVICE)
        R, T = convert_camera_from_pytorch3d_to_gs(camera, args.image_size, args.image_size)
        R_list.append(R)
        T_list.append(T.reshape(3))

        R_c, T_c = convert_camera_from_pytorch3d_to_colmap(camera, args.image_size, args.image_size)
        R_raw_list.append(R_c)
        T_raw_list.append(T_c)

    scene = Scene(R_list, T_list, gaussians, image_size=args.image_size)
    views = scene.getTestCameras()
    bg_color = [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)

    select_tensors_list = []

    for i in range(NUM_PRINCIPLE):
        dist = pre_dist_list[i]
        elev = pre_elev_list[i]
        azim = pre_azim_list[i]
        camera = init_camera(dist, elev, azim, args.image_size, DEVICE)
        R, T = convert_camera_from_pytorch3d_to_colmap(camera, args.image_size, args.image_size)
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
        focal = 0.5 * args.image_size / np.tan(0.5 * 1)
        K[0,0] = args.image_size / 2
        K[1,1] = args.image_size / 2
        K[0,2] = args.image_size / 2
        K[1,2] = args.image_size / 2
        K[2,2] = 1
        points_pixel = K @ points_c
        
        points_pixel = points_pixel / points_pixel[2:, :]
        pc_pixel = points_pixel[:2, :]

        image_size = (args.image_size, args.image_size)
        image = torch.zeros(image_size).cuda()
        
        x_grid = np.arange(args.image_size)
        y_grid = np.arange(args.image_size)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        grid_x = torch.from_numpy(grid_x).cuda()
        grid_y = torch.from_numpy(grid_y).cuda()

        
        view = views[i]
        rendering_results = gs_render(view, gaussians, pipeline, background)
        radii = rendering_results['radii']

        update_tensor = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool)

        pix_depth, pix_id = get_depth_with_id(pc_pixel, radii.float(), points_c_norm[2,:], args.image_size, args.image_size)
        
        pix_id = pix_id.permute(1, 0, 2)

        pix_depth = pix_depth.permute(1, 0, 2)
        

        first_elements = pix_depth[:, :, 0].unsqueeze(-1)
        mask_tmp = pix_depth >= (first_elements + 0.01)
        pix_depth[mask_tmp] = -1
        pix_id[mask_tmp] = -1
        pix_id[:, :, MAX_DEPTH_SLOTS:] = -1

        if i == 0 or i == 2:
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R
            pose[:3, 3] = T
            pose = np.linalg.inv(pose)
            K = np.eye(3, dtype=np.float32)
            focal = 0.5 * args.image_size / np.tan(0.5 * 1)
            K[0,0] = args.image_size / 2
            K[1,1] = args.image_size / 2
            K[0,2] = args.image_size / 2
            K[1,2] = args.image_size / 2
            intrinsics_inv = np.linalg.inv(K)
            rays_o, rays_v = gen_rays_at(args.image_size, args.image_size, torch.from_numpy(pose), torch.from_numpy(intrinsics_inv))

            d_pred_out = ray_marching(rays_o.cuda().reshape(1, -1, 3), rays_v.cuda().reshape(1, -1, 3), udf_network, tau=0.01, n_steps=[199, 200])
            d_pred_out = d_pred_out.reshape(args.image_size, args.image_size, 1)
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
            depth_maps_tensor = depth_maps_tensor.reshape(1, args.image_size, args.image_size)
            depth_map = depth_maps_tensor[0].cpu().numpy()
            depth_map = Image.fromarray(depth_map).convert("L")
            depth_maps_tensor = depth_maps_tensor > 0
            edge = cv2.Canny(np.uint8(depth_maps_tensor.clone().detach().squeeze(0).cpu().numpy() * 255), threshold1=100, threshold2=200)
            dilated = cv2.dilate(edge, np.ones((1,1), np.uint8), iterations=1)
            _, thresholded = cv2.threshold(np.uint8(depth_maps_tensor.clone().detach().squeeze(0).cpu().numpy() * 255), 200, 255, cv2.THRESH_BINARY)
            edges = cv2.bitwise_and(thresholded, dilated)
            edges = torch.from_numpy(edges.astype(bool)).bool().cuda().unsqueeze(-1).repeat(1, 1, pix_id.shape[2])
            pix_id[~edges] = -1

        mask_tmp = (pix_id != -1)
        valid_ids = pix_id[mask_tmp].cuda()
        update_tensor[valid_ids.long()] = True

        select_tensors_list.append(update_tensor)

    azim_appendix_list = []

    intersection_0_1 = torch.logical_and(select_tensors_list[0], select_tensors_list[1])
    xyz_0_1 = gaussians._xyz[intersection_0_1]
    normals_0_1 = quaternion_to_normal(gaussians._rotation[intersection_0_1])
    azim = optimize_camera_cos(xyz_0_1.clone().detach(), normals_0_1.clone().detach(), 45)
    azim_appendix_list.append(float(azim.item()))
    
    intersection_0_3 = torch.logical_and(select_tensors_list[0], select_tensors_list[3])
    xyz_0_3 = gaussians._xyz[intersection_0_3]
    normals_0_3 = quaternion_to_normal(gaussians._rotation[intersection_0_3])
    azim = optimize_camera_cos(xyz_0_3.clone().detach(), normals_0_3.clone().detach(), 315)
    azim_appendix_list.append(float(azim.item()))

    intersection_2_1 = torch.logical_and(select_tensors_list[2], select_tensors_list[1])
    xyz_2_1 = gaussians._xyz[intersection_2_1]
    normals_2_1 = quaternion_to_normal(gaussians._rotation[intersection_2_1])
    azim = optimize_camera_cos(xyz_2_1.clone().detach(), normals_2_1.clone().detach(), 135)
    azim_appendix_list.append(float(azim.item()))

    intersection_2_3 = torch.logical_and(select_tensors_list[2], select_tensors_list[3])
    xyz_2_3 = gaussians._xyz[intersection_2_3]
    normals_2_3 = quaternion_to_normal(gaussians._rotation[intersection_2_3])
    azim = optimize_camera_cos(xyz_2_3.clone().detach(), normals_2_3.clone().detach(), 225)
    azim_appendix_list.append(float(azim.item()))
    pre_dist_list = pre_dist_list + [1.0 for i in azim_appendix_list]
    pre_elev_list = pre_elev_list + [0.0 for i in azim_appendix_list]
    pre_azim_list = pre_azim_list + azim_appendix_list
    
    print(azim_appendix_list)
    with open(os.path.join(output_dir,"add_azim.json"), "w") as f:
        json.dump(azim_appendix_list, f)
    render_normal_and_position(pre_dist_list, pre_elev_list, pre_azim_list, args.image_size, args.fragment_k, DEVICE, udf_network,output_dir)
    gaussians.save_ply(os.path.join(output_dir, "gs.ply"))
    print("Stage 1 finished.")
    raise SystemExit(0)
