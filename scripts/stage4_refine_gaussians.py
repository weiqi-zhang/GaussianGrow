import os
import argparse
import json
import time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

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
    optimize_camera_v2,
)
from gaussian_grow.core.diffusion_helper import apply_controlnet_depth, get_controlnet_depth
from gaussian_grow.core.io_helper import save_args
from gaussian_grow.core.projection_helper import render_one_view_and_build_masks_gaussian_repaint
from gaussian_grow.core.render import gen_rays_at, ray_marching
from gaussian_grow.core.vis_helper import visualize_principle_viewpoints
from models.delight.dehighlight_utils import Light_Shadow_Remover
from gaussian_grow.optimization.opt_gaussian import (
    opt_gaussian_from_one_view_generate, 
    opt_gaussian_from_one_view_v2,
    opt_gaussian_from_one_view_overlap
)
from gaussian_grow.optimization.inpainting import update_colored_points
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
    parser.add_argument("--prompt", type=str, required=True)
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
    parser.add_argument(
        "--delight_model_path",
        type=str,
        default=os.environ.get("HUNYUAN3D_DELIGHT_MODEL_PATH"),
        help="Optional local path to hunyuan3d-delight-v2-0. Defaults to <HUNYUAN3D_MODEL_PATH>/hunyuan3d-delight-v2-0 when set.",
    )
    add_capudf_args(parser)

    args = parser.parse_args()
    op = OptimizationParams(parser)

    if args.device == "a6000":
        setattr(args, "render_simple_factor", 12)
        setattr(args, "fragment_k", 1)
        setattr(args, "image_size", 1024)
        setattr(args, "uv_size", 3000)
    else:
        setattr(args, "render_simple_factor", 4)
        setattr(args, "fragment_k", 1)
        setattr(args, "image_size", 1024)
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
    ckpt_path = ensure_capudf_checkpoint(args, name, output_dir)
    gaussians = GaussianModel(0)
    gaussians.load_ply(os.path.join(output_dir, 'gs.ply'))

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
    # Stage 4 reuses the views generated by the Hunyuan3D texture stage.
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

    camera_dir = os.path.join(generate_dir, "camera")
    os.makedirs(camera_dir, exist_ok=True)

        
    # Prepare viewpoints and camera cache.
    NUM_PRINCIPLE = 10
    pre_dist_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    pre_elev_list = []
    pre_azim_list = []
    with open(os.path.join(output_dir,"azim.json"), "r") as f:
        pre_azim_list= json.load(f)
    with open(os.path.join(output_dir,"elev.json"), "r") as f:
        pre_elev_list= json.load(f)
    print("pre_azim_list:", pre_azim_list)
    print("pre_elev_list:", pre_elev_list)
    pre_sector_list = ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
    pre_view_punishments = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

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
    gaussians.training_setup(op, scene.cameras_extent)

    with torch.no_grad():
        gaussians.set_scale()

    update_views = []
    masks = []
    visibilitys = []
    
    init_images = []
    for i in range(NUM_PRINCIPLE):
        image_path = os.path.join(output_dir, "view",f'view_{i}.png')
        image = Image.open(image_path)
        image = image.resize((args.image_size, args.image_size))
        image_tensor = torch.from_numpy(np.array(image)).cuda() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)
        init_images.append(image_tensor)
    
    views = scene.getTrainCameras()
    generate_mask_tensors = []
    for i in range(NUM_PRINCIPLE):
        view = views[i]
        bg_color = [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        rendering_results = gs_render(view, scene.gaussians, pipeline, background)
        rend_normal = rendering_results['surf_depth']
        generate_mask_tensor = (rend_normal.sum(dim=0) > 0).float().unsqueeze(0)
        generate_mask_tensors.append(generate_mask_tensor)

    # Generate or refine view-conditioned images.
    print("=> start generating texture...")
    start_time = time.time()
    new_gaussian= None
    view_list = [5, 4, 3, 1, 2, 0]
    for view_idx in view_list:
        print("=> processing view {}...".format(view_idx))

        dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx]      

        generate_image_tensor = init_images[view_idx]
        generate_mask_tensor = generate_mask_tensors[view_idx]

        update_view, mask, update, new_gaussian = opt_gaussian_from_one_view_v2(gaussians, scene, view_idx, init_images[view_idx], generate_mask_tensor, op, dist, elev, azim, DEVICE, udf_network, new_gaussian, gaussian_dir)
        
        gaussians.save_ply(os.path.join(gaussian_dir, "{}_generate.ply".format(view_idx)))
        views = scene.getTrainCameras()
        view = views[view_idx]
        bg_color = [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        render_pkg = gs_render(view, gaussians, pipeline, background)
        image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        torchvision.utils.save_image(image, os.path.join(gaussian_dir, "{}_generate.png".format(view_idx)))

    update_tensor = gaussians._update.clone().detach().cpu()
    if not (update_tensor == 0).all():
        torch.save(update_tensor, f'{gaussian_dir}/update_tensor.pt')

    select_tensors_list = []
    overlap_mask_list = []
    for i in range(4):
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

        start_time = time.time()
    

        update_tensor = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool)

        start_time = time.time()
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
            dilated = cv2.dilate(edge, np.ones((5,5), np.uint8), iterations=1)
            _, thresholded = cv2.threshold(np.uint8(depth_maps_tensor.clone().detach().squeeze(0).cpu().numpy() * 255), 200, 255, cv2.THRESH_BINARY)
            edges = cv2.bitwise_and(thresholded, dilated)
            edges = torch.from_numpy(edges.astype(bool)).bool().cuda().unsqueeze(-1).repeat(1, 1, pix_id.shape[2])

            pix_id[~edges] = -1

        mask_tmp = (pix_id != -1)
        valid_ids = pix_id[mask_tmp].cuda()
        update_tensor[valid_ids.long()] = True

        select_tensors_list.append(update_tensor)

    for i in range(4):
        if i == 0:
            intersection = torch.logical_and(select_tensors_list[0], select_tensors_list[1])
        elif i == 1:
            intersection = torch.logical_and(select_tensors_list[0], select_tensors_list[3])
        elif i == 2:
            intersection = torch.logical_and(select_tensors_list[2], select_tensors_list[1])
        elif i ==3:
            intersection = torch.logical_and(select_tensors_list[2], select_tensors_list[3])

        dist = pre_dist_list[i + 6]
        elev = pre_elev_list[i + 6]
        azim = pre_azim_list[i + 6]
        view = views[i]
        bg_color = [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        render_pkg = gs_render(view, gaussians, pipeline, background)
        radii = render_pkg['radii']
        xyz = gaussians._xyz[intersection]
        ones = torch.ones((xyz.shape[0], 1)).cuda()      
        points_w = torch.cat((xyz, ones), dim=1).permute(1, 0)
        camera = init_camera(dist, elev, azim, args.image_size, DEVICE)
        R, T = convert_camera_from_pytorch3d_to_colmap(camera, args.image_size, args.image_size)
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0
        world_view_transform = torch.from_numpy(Rt).cuda().float()
        points_c = world_view_transform @ points_w
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
        rendering_results = gs_render(view, gaussians, pipeline, background)
        tmp_r = radii[intersection]

        x_grid = np.arange(args.image_size)
        y_grid = np.arange(args.image_size)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        grid_x = torch.from_numpy(grid_x).cuda()
        grid_y = torch.from_numpy(grid_y).cuda()
        image_size = (args.image_size, args.image_size)
        overlap_mask = torch.zeros(image_size).cuda()
        for (cx, cy), radius in zip(pc_pixel.T, tmp_r):
            square_dist = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            overlap_mask += (square_dist <= radius**2)

        overlap_mask = torch.clip(overlap_mask, 0, 1)
        overlap_mask_list.append(overlap_mask.unsqueeze(0))
    
    for view_idx in range(6, 10):
        print("=> processing view {}...".format(view_idx))
           
        overlap_mask = overlap_mask_list[view_idx - 6]
        dist, elev, azim = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx] 

        update = opt_gaussian_from_one_view_overlap(gaussians, scene, view_idx, init_images[view_idx], overlap_mask, op, dist, elev, azim, sector, DEVICE, udf_network, new_gaussian, gaussian_dir)

        gaussians.save_ply(os.path.join(gaussian_dir, "{}_overlap.ply".format(view_idx)))
        views = scene.getTrainCameras()
        view = views[view_idx]
        bg_color = [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        render_pkg = gs_render(view, gaussians, pipeline, background)
        image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        torchvision.utils.save_image(image, os.path.join(gaussian_dir, "{}_overlap.png".format(view_idx)))

        (
            cameras,
            init_image, depth_map, 
            init_images_tensor, depth_maps_tensor, 
            generate_mask_image, keep_mask_image,
            generate_mask_tensor, keep_mask_tenosr,
            generate_mask_image_raw, exist_mask_image_raw,
            generate_mask_tensor_raw, keep_mask_tenosr_raw,
        ) = render_one_view_and_build_masks_gaussian_repaint(
            dist, elev, azim, 
            view_idx, view_idx,
            args.image_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, gaussian_dir,
            DEVICE, 
            scene,
            udf_network, gaussians,
            save_intermediate=True, second=False
        )
        update, new_gaussian = opt_gaussian_from_one_view_generate(gaussians, scene, view_idx, init_images[view_idx].permute(1, 2, 0)*255.0, generate_mask_tensor_raw, op, init_images_tensor.squeeze(0), dist, elev, azim, sector, DEVICE, udf_network, new_gaussian, gaussian_dir, second=False)
        
        gaussians.save_ply(os.path.join(gaussian_dir, "{}_generate.ply".format(view_idx)))
        views = scene.getTrainCameras()
        view = views[view_idx]
        bg_color = [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        render_pkg = gs_render(view, gaussians, pipeline, background)
        image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        torchvision.utils.save_image(image, os.path.join(gaussian_dir, "{}_generate.png".format(view_idx)))
    
    init_image_dir = os.path.join(update_dir, "rendering")
    os.makedirs(init_image_dir, exist_ok=True)

    normal_map_dir = os.path.join(update_dir, "normal")
    os.makedirs(normal_map_dir, exist_ok=True)

    mask_image_dir = os.path.join(update_dir, "mask")
    os.makedirs(mask_image_dir, exist_ok=True)

    depth_map_dir = os.path.join(update_dir, "depth")
    os.makedirs(depth_map_dir, exist_ok=True)

    similarity_map_dir = os.path.join(update_dir, "similarity")
    os.makedirs(similarity_map_dir, exist_ok=True)

    inpainted_image_dir = os.path.join(update_dir, "inpainted")
    os.makedirs(inpainted_image_dir, exist_ok=True)

    mesh_dir = os.path.join(update_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    interm_dir = os.path.join(update_dir, "intermediate")
    os.makedirs(interm_dir, exist_ok=True)

    gaussian_dir = os.path.join(update_dir, "gaussian")
    os.makedirs(gaussian_dir, exist_ok=True)

    camera_dir = os.path.join(update_dir, "camera")
    os.makedirs(camera_dir, exist_ok=True)
    
    print("=> starting 2D inpainting...")

    controlnet, ddim_sampler = get_controlnet_depth()

    transform_tensor = transforms.ToTensor()

    NUM_INPAINT_ITERATIONS = 6
    for view_idx in range(NUM_INPAINT_ITERATIONS):
        print("=> processing view {}...".format(view_idx))

        update_tensor = gaussians._update
        update_tensor = ~(update_tensor.cuda().bool())
            
        elev, azim = optimize_camera_v2(gaussians, update_tensor, DEVICE, view_idx, camera_dir, image_size=256, threshold=0.1,  sample_ratio=0.2)
        camera = init_camera(dist, elev, azim, args.image_size, DEVICE)
        R, T = convert_camera_from_pytorch3d_to_gs(camera, args.image_size, args.image_size)
        scene = Scene([R], [T], gaussians, image_size=args.image_size)
        torch.cuda.empty_cache()

        (
            cameras,
            init_image, depth_map, 
            init_images_tensor, depth_maps_tensor, 
            generate_mask_image, keep_mask_image,
            generate_mask_tensor, keep_mask_tenosr,
            generate_mask_image_raw, exist_mask_image_raw,
            generate_mask_tensor_raw, keep_mask_tenosr_raw,
        ) = render_one_view_and_build_masks_gaussian_repaint(
            dist, elev, azim, 
            view_idx, view_idx,
            args.image_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, gaussian_dir,
            DEVICE, 
            scene,
            udf_network, gaussians,
            save_intermediate=True, second=True
        ) # (generate_mask_tensor.shape) torch.Size([1, 1024, 1024])
        
        if generate_mask_tensor_raw.sum()<500:continue

        generate_image, generate_image_before, generate_image_after, generate_image_tensor = apply_controlnet_depth(
            controlnet, ddim_sampler, 
            init_image.convert("RGBA"), args.prompt, args.new_strength, args.ddim_steps,
            generate_mask_image, keep_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
            args.a_prompt, args.n_prompt, args.guidance_scale, args.seed, args.eta, 1, DEVICE, args.blend
        ) 
        delight_model_path = args.delight_model_path
        if delight_model_path is None and os.environ.get("HUNYUAN3D_MODEL_PATH"):
            delight_model_path = os.path.join(os.environ["HUNYUAN3D_MODEL_PATH"], "hunyuan3d-delight-v2-0")
        if delight_model_path:
            light_model = Light_Shadow_Remover(delight_model_path)
            generate_image = light_model(generate_image)
            del light_model
        generate_image.save(os.path.join(inpainted_image_dir, "{}_delight.png".format(view_idx)))
        generate_image_before.save(os.path.join(inpainted_image_dir, "{}_before.png".format(view_idx)))
        generate_image_after.save(os.path.join(inpainted_image_dir, "{}_after.png".format(view_idx)))
        generate_image_tensor = transform_tensor(generate_image).cuda()

        generate_image_tensor = F.interpolate(generate_image_tensor.unsqueeze(0), size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)

        # Remove batch dimension.
        generate_image_tensor = generate_image_tensor.squeeze(0).permute(1, 2, 0) * 255.0

        update, new_gaussian = opt_gaussian_from_one_view_generate(gaussians, scene, view_idx, generate_image_tensor, generate_mask_tensor_raw, op, init_images_tensor.squeeze(0), dist, elev, azim, sector, DEVICE, udf_network, new_gaussian, gaussian_dir, second=True)

        gaussians.save_ply(os.path.join(gaussian_dir, "{}_generate.ply".format(view_idx)))
        views = scene.getTrainCameras()
        view = views[0]
        bg_color = [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        render_pkg = gs_render(view, gaussians, pipeline, background)
        image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        torchvision.utils.save_image(image, os.path.join(gaussian_dir, "{}_generate.png".format(view_idx)))

    
    print("=> starting 3D inpainting...")
    gaussians = update_colored_points(gaussians, gaussian_dir)
    gaussians.save_ply(os.path.join(gaussian_dir, "final.ply".format(view_idx)))
    print("=> total generate time: {} s".format(time.time() - start_time))
        
    visualize_principle_viewpoints(output_dir, pre_dist_list, pre_elev_list, pre_azim_list)
    print("Stage 4 finished.")
