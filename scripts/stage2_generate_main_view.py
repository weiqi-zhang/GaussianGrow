import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import argparse
import time
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from gaussian_grow.release_utils import add_capudf_args, ensure_capudf_checkpoint, require_cuda

from gaussian_renderer import gs_render
from arguments import OptimizationParams
from gaussian_grow.core.camera_helper import (
    convert_camera_from_pytorch3d_to_colmap,
    convert_camera_from_pytorch3d_to_gs,
    init_camera,
    init_viewpoints,
)
from gaussian_grow.core.diffusion_helper import (
    get_controlnet_depth,
    apply_controlnet_depth,
)
from gaussian_grow.core.io_helper import save_args
from gaussian_grow.core.projection_helper import (
    render_one_view_and_build_masks_gaussian,
    build_similarity_gaussian_cache_for_all_views_gaussian_2,
)
from scene import Scene
from scene.fields import CAPUDFNetwork
from scene.gaussian_model import GaussianModel

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
    parser.add_argument("--smooth_mask", action="store_true", help="smooth the diffusion mask")
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
    
    update_tensor = torch.zeros([gaussians._xyz.shape[0]], dtype=torch.bool).cuda()
    # Stage 2 only synthesizes the conditioning image for the main view.
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
    controlnet, ddim_sampler = get_controlnet_depth()
    generate_dir = os.path.join(output_dir, "generate")
    os.makedirs(generate_dir, exist_ok=True)

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
    
    gaussians.training_setup(op, scene.cameras_extent)

    similarity_view_cache = build_similarity_gaussian_cache_for_all_views_gaussian_2(
        pre_dist_list, pre_elev_list, pre_azim_list,
        args.image_size, args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
        gaussians, scene,
        DEVICE, udf_network
    )
    with torch.no_grad():
        gaussians.set_scale()
    update_views = []
    masks = []
    visibilitys = []
    prompts = [args.prompt]
    with torch.no_grad():
        for view_idx in range(NUM_PRINCIPLE):
            print(f"view {view_idx}")
            dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx]
            views = scene.getTrainCameras()
            view = views[view_idx]
            bg_color = [0,0,0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
            rendering_results = gs_render(view, scene.gaussians, pipeline, background)
            radii = rendering_results['radii']

            rendered_alpha = rendering_results['rendered_alpha'] # torch.Size([1, 1024, 1024]) 
            rendered_alpha = rendered_alpha.reshape(1, args.image_size, args.image_size)
            masks.append(rendered_alpha)

    # Generate or refine view-conditioned images.
    print("=> start generating texture...")
    start_time = time.time()
    new_gaussian= None
    for view_idx in range(1):

        print("=> processing view {}...".format(view_idx))
        print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx] 
        prompt = " the {} view of {}".format(sector, prompts[view_idx])

        # 1.1. render and build masks
        (
            view_score,
            cameras,
            init_image, normal_map, depth_map, 
            init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
            keep_mask_image, update_mask_image, generate_mask_image, 
            keep_mask_tensor, update_mask_tensor, generate_mask_tensor, all_mask_tensor, quad_mask_tensor, visibility_filter
        ) = render_one_view_and_build_masks_gaussian(dist, elev, azim, 
            view_idx, view_idx, view_punishments,
            similarity_view_cache,
            args.image_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, gaussian_dir,
            DEVICE, 
            scene,
            udf_network, new_gaussian, R_raw_list, T_raw_list, gaussians,
            save_intermediate=True, smooth_mask=args.smooth_mask, view_threshold=args.view_threshold
        )
   
        visibilitys.append(visibility_filter)
        torch.cuda.empty_cache()
        print("=> generating image for prompt: {}...".format(prompt))
        
        if args.no_repaint and view_idx != 0:
            actual_generate_mask_image = Image.fromarray((np.ones_like(np.array(generate_mask_image)) * 255.).astype(np.uint8))
        else:
            actual_generate_mask_image = generate_mask_image
        init_image.save(os.path.join(output_dir, "init_image.png"))
        actual_generate_mask_image.save(os.path.join(output_dir, "generate_mask.png"))
        keep_mask_image.save(os.path.join(output_dir, "keep_mask.png"))
        print("=> generate for view {}".format(view_idx))
        generate_image, generate_image_before, generate_image_after, generate_image_tensor = apply_controlnet_depth(controlnet, ddim_sampler, 
            init_image.convert("RGBA"), prompt, args.new_strength, args.ddim_steps,
            actual_generate_mask_image, keep_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
            args.a_prompt, args.n_prompt, args.guidance_scale, args.seed, args.eta, 1, DEVICE, args.blend)        

        
        generate_image.save(os.path.join(output_dir, "main_view.png"))
        print("Stage 2 finished.")
        raise SystemExit(0)
