import time
import copy
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from math import exp
from PIL import Image
from tqdm import tqdm
from types import SimpleNamespace

import trimesh

from depthid_render import get_depth_with_id
from depthid_render_mask_control_v2 import get_depth_with_id as get_depth_with_id_2

from gaussian_renderer import gs_render
from gaussian_grow.core.camera_helper import init_camera, convert_camera_from_pytorch3d_to_colmap

# Depth-occlusion tolerance for opt_gaussian rendering: pixels with depth
# farther than (first + EPS) are treated as occluded. depthid_render returns
# 1000 slots per pixel; tail 9 are unreliable padding (index 991+).
OCCLUSION_DEPTH_EPS = 0.005
DEPTH_SLOT_LIMIT = 991
edge_num = 10
middle = float(np.log(np.sqrt(0.000002)))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def _ssim_weight(img1, img2, window, window_size, channel, size_average=True, weight_mask=None):

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if weight_mask is not None:

        weight_mask = weight_mask.to(ssim_map.device).type_as(ssim_map)
        while weight_mask.dim() < 4:
            weight_mask = weight_mask.unsqueeze(0)
        

        weighted_sum = (ssim_map * weight_mask).sum()
        mask_sum = weight_mask.sum()
        return weighted_sum / (mask_sum + 1e-8)
        
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def l1_loss(network_output, gt, weight=None):
    if weight is not None:
        return torch.abs((network_output - gt) * weight).mean()
    else:
        return torch.abs((network_output - gt)).mean()

def opt_gaussian_from_one_view(gaussians, scene, view_idx, generate_image, generate_mask_image, opt, init_image, keep_mask_tensor, update_mask_tensor, alpha, visibility_filter, dist, elev, azim, sector, DEVICE, udf_network, new_gaussian=None, gaussian_dir=None):
    views = scene.getTrainCameras()
    view = views[view_idx]
    bg_color = [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    print("optim gaussian")

    init_image = init_image[:,:,:3].permute(2, 0, 1)

    gt_image = (generate_image.permute(2, 0, 1) / 255.0)
    
    keep = init_image * torch.logical_or(keep_mask_tensor, update_mask_tensor)
    gt_image = gt_image * generate_mask_image
    torchvision.utils.save_image(gt_image, f'{gaussian_dir}/gt_{view_idx}.png')
    mask = torch.logical_or(torch.logical_or(keep_mask_tensor, update_mask_tensor), generate_mask_image).float()

    mask_image = Image.fromarray(generate_mask_image.clone().detach().squeeze(0).cpu().byte().numpy() * 255)

    mask_image.save(f'{gaussian_dir}/mask_{view_idx}.png')
    
    camera = init_camera(dist, elev, azim, init_image.shape[1], DEVICE)
    R, T = convert_camera_from_pytorch3d_to_colmap(camera, init_image.shape[1], init_image.shape[1])
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
    focal = 0.5 * init_image.shape[1] / 2 / np.tan(0.5 * 1)
    K[0,0] = init_image.shape[1] / 2
    K[1,1] = init_image.shape[1] / 2
    K[0,2] = init_image.shape[1] / 2
    K[1,2] = init_image.shape[1] / 2
    K[2,2] = 1
    points_pixel = K @ points_c
    
    points_pixel = points_pixel / points_pixel[2:, :]
    pc_pixel = points_pixel[:2, :]

    image_size = (init_image.shape[1], init_image.shape[1])
    image = torch.zeros(image_size).cuda()
    
    x_grid = np.arange(init_image.shape[1])
    y_grid = np.arange(init_image.shape[1])
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_x = torch.from_numpy(grid_x).cuda()
    grid_y = torch.from_numpy(grid_y).cuda()

    rendering_results = gs_render(view, gaussians, pipeline, background)
    radii = rendering_results['radii']
    update_tensor = torch.zeros(visibility_filter.shape, dtype=torch.bool).cuda()

    start_time = time.time()
    pix_depth, pix_id = get_depth_with_id_2(pc_pixel, radii.float(), points_c_norm[2,:], generate_mask_image.int().squeeze(0).permute(1, 0).contiguous(), init_image.shape[1], init_image.shape[1])

    pix_id = pix_id.permute(1, 0, 2)

    pix_depth = pix_depth.permute(1, 0, 2)
    
    edge = cv2.Canny(np.uint8(generate_mask_image.clone().detach().squeeze(0).cpu().numpy() * 255), threshold1=100, threshold2=200)
    dilated = cv2.dilate(edge, np.ones((5,5), np.uint8), iterations=2)
    _, thresholded = cv2.threshold(np.uint8(generate_mask_image.clone().detach().squeeze(0).cpu().numpy() * 255), 200, 255, cv2.THRESH_BINARY)
    edges = cv2.bitwise_and(thresholded, dilated)
    edges = torch.from_numpy(edges.astype(bool)).bool().cuda().unsqueeze(-1)
    torchvision.utils.save_image(edges.squeeze(-1).unsqueeze(0).float(), f'{gaussian_dir}/generate_{view_idx}_edge.png')

    

    pix_id = pix_id.cpu()
    pix_depth = pix_depth.cpu()

    first_elements = pix_depth[:, :, 0].unsqueeze(-1)
    mask_tmp = (pix_depth >= (first_elements + OCCLUSION_DEPTH_EPS))
    pix_depth[mask_tmp] = -1
    pix_id[mask_tmp] = -1
    
    pix_id[:, :, DEPTH_SLOT_LIMIT:] = -1
    
    mask_tmp = (pix_id != -1)
    valid_ids = pix_id[mask_tmp]
    update_tensor[valid_ids.long().cuda()] = True

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

    update_tensor1 = torch.zeros(visibility_filter.shape, dtype=torch.bool).cuda()

    start_time = time.time()
    pix_depth, pix_id = get_depth_with_id(pc_pixel, radii.float(), points_c_norm[2,:], init_image.shape[1], init_image.shape[1])
    
    pix_id = pix_id.permute(1, 0, 2)

    pix_depth = pix_depth.permute(1, 0, 2)
    
    edge = cv2.Canny(np.uint8(generate_mask_image.clone().detach().squeeze(0).cpu().numpy() * 255), threshold1=100, threshold2=200)
    dilated = cv2.dilate(edge, np.ones((5,5), np.uint8), iterations=2)
    _, thresholded = cv2.threshold(np.uint8(generate_mask_image.clone().detach().squeeze(0).cpu().numpy() * 255), 200, 255, cv2.THRESH_BINARY)
    edges = cv2.bitwise_and(thresholded, dilated)
    edges = torch.from_numpy(edges.astype(bool)).bool().cuda().unsqueeze(-1)
    
    indices = torch.arange(pix_id.shape[2]).expand_as(pix_id).cuda()

    first_elements = pix_depth[:, :, 0].unsqueeze(-1)
    mask_tmp = pix_depth >= (first_elements + OCCLUSION_DEPTH_EPS)
    pix_depth[mask_tmp] = -1
    pix_id[mask_tmp] = -1
    pix_id[:, :, DEPTH_SLOT_LIMIT:] = -1
    pix_id[edges & (indices >= edge_num)] = -1
    pix_id[(~generate_mask_image.bool()).squeeze(0).unsqueeze(-1) & (indices >= 0)] = -1
    
    mask_tmp = (pix_id != -1)
    valid_ids = pix_id[mask_tmp]
    update_tensor1[valid_ids.long()] = True

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    update_tensor = torch.logical_and(update_tensor, update_tensor1)

    xyz = gaussians._xyz[update_tensor].clone().detach().cpu().numpy()
    point_cloud = trimesh.Trimesh(xyz)
    point_cloud.export(f"{gaussian_dir}/generate_{view_idx}.obj")
    xyz = gaussians._xyz[~update_tensor].clone().detach().cpu().numpy()
    point_cloud = trimesh.Trimesh(xyz)
    point_cloud.export(f"{gaussian_dir}/generate_inver_{view_idx}.obj")
    xyz = gaussians._xyz[update_tensor]
    ones = torch.ones((xyz.shape[0], 1)).cuda()      
    points_w = torch.cat((xyz, ones), dim=1).permute(1, 0)
    camera = init_camera(dist, elev, azim, init_image.shape[1], DEVICE)
    R, T = convert_camera_from_pytorch3d_to_colmap(camera, init_image.shape[1], init_image.shape[1])
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    world_view_transform = torch.from_numpy(Rt).cuda().float()
    points_c = world_view_transform @ points_w
    K = torch.zeros([3,4]).cuda()
    focal = 0.5 * init_image.shape[1] / np.tan(0.5 * 1)
    K[0,0] = init_image.shape[1] / 2
    K[1,1] = init_image.shape[1] / 2
    K[0,2] = init_image.shape[1] / 2
    K[1,2] = init_image.shape[1] / 2
    K[2,2] = 1
    points_pixel = K @ points_c
    points_pixel = points_pixel / points_pixel[2:, :]
    pc_pixel = points_pixel[:2, :]
    rendering_results = gs_render(view, gaussians, pipeline, background)
    radii = rendering_results['radii']
    tmp_r = radii[update_tensor]

    x_grid = np.arange(init_image.shape[1])
    y_grid = np.arange(init_image.shape[1])
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_x = torch.from_numpy(grid_x).cuda()
    grid_y = torch.from_numpy(grid_y).cuda()
    image_size = (init_image.shape[1], init_image.shape[1])
    image = torch.zeros(image_size).cuda()
    for (cx, cy), radius in zip(pc_pixel.T, tmp_r):
        square_dist = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        image += (square_dist <= radius**2)

    image = torch.clip(image, 0, 1)
    torchvision.utils.save_image(image, f'{gaussian_dir}/generate_mask_{view_idx}.png')
    
    with torch.no_grad():
        opt_gaussian = copy.deepcopy(gaussians)
        opt_gaussian.select_gaussian(update_tensor)

    opt.iterations = 1500
    opt.position_lr_max_steps = 1500
    opt_gaussian.training_setup(opt, scene.cameras_extent)

    opt_num = 0
    
    save_loss = []
    for iteration in tqdm(range(opt.iterations)):
        opt_gaussian.update_learning_rate(iteration)
        background = torch.rand(3).cuda()

        
        render_pkg = gs_render(view, opt_gaussian, pipeline, background)
        image, viewspace_point_tensor, visibility_filter, radii, render_mask = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['rendered_alpha']
        image = image * generate_mask_image
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        rend_dist = render_pkg["rend_dist"]
        Ll1 = l1_loss(image, gt_image)
        Ll1_mask = l1_loss(generate_mask_image, render_mask)
        scale_loss = torch.mean(torch.square(torch.clamp(opt_gaussian._scaling, max=0.9*middle) - opt_gaussian._scaling))
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.0 * Ll1_mask + 20.0 * scale_loss
        
        loss.backward()
        if (iteration + 1) % 100 == 0:
            save_loss.append((1.0 - ssim(image, gt_image)).clone().detach().cpu().numpy())

        if iteration % 100 == 0:
            print((1.0 - ssim(image, gt_image)).item())
        
        with torch.no_grad():
            

            

                
            opt_gaussian.optimizer.step()
            opt_gaussian.optimizer.zero_grad(set_to_none = True)
    
    with torch.no_grad():
        gaussians.num_no_opt = gaussians._xyz[~update_tensor].shape[0]
        gaussians._xyz = nn.Parameter(torch.cat((gaussians._xyz[~update_tensor], opt_gaussian._xyz), dim=0).requires_grad_(True))
        gaussians._features_dc = nn.Parameter(torch.cat((gaussians._features_dc[~update_tensor], opt_gaussian._features_dc), dim=0).requires_grad_(True))
        gaussians._features_rest = nn.Parameter(torch.cat((gaussians._features_rest[~update_tensor], opt_gaussian._features_rest), dim=0).requires_grad_(True))
        gaussians._opacity = nn.Parameter(torch.cat((gaussians._opacity[~update_tensor], opt_gaussian._opacity), dim=0).requires_grad_(True))
        gaussians._scaling = nn.Parameter(torch.cat((gaussians._scaling[~update_tensor], opt_gaussian._scaling), dim=0).requires_grad_(True))
        gaussians._rotation = nn.Parameter(torch.cat((gaussians._rotation[~update_tensor], opt_gaussian._rotation), dim=0).requires_grad_(True))
        gaussians._update[update_tensor] = 1
        gaussians._update = gaussians._update[~update_tensor.bool()]
        gaussians._update = torch.cat((gaussians._update, torch.ones(opt_gaussian._xyz.shape[0]).cuda()),dim=0)

        
        if view_idx == 0 and new_gaussian == None:
            new_gaussian = opt_gaussian
        else:
            new_gaussian._xyz = nn.Parameter(torch.cat((new_gaussian._xyz, opt_gaussian._xyz), dim=0).requires_grad_(True))
            new_gaussian._features_dc = nn.Parameter(torch.cat((new_gaussian._features_dc, opt_gaussian._features_dc), dim=0).requires_grad_(True))
            new_gaussian._features_rest = nn.Parameter(torch.cat((new_gaussian._features_rest, opt_gaussian._features_rest), dim=0).requires_grad_(True))
            new_gaussian._opacity = nn.Parameter(torch.cat((new_gaussian._opacity, opt_gaussian._opacity), dim=0).requires_grad_(True))
            new_gaussian._scaling = nn.Parameter(torch.cat((new_gaussian._scaling, opt_gaussian._scaling), dim=0).requires_grad_(True))
            new_gaussian._rotation = nn.Parameter(torch.cat((new_gaussian._rotation, opt_gaussian._rotation), dim=0).requires_grad_(True))

    gt_image = gt_image * generate_mask_image + keep
    with torch.no_grad():
        background = torch.tensor([0.0,0.0,0.0]).cuda()
        render_pkg = gs_render(view, opt_gaussian, pipeline, background)
        epochs = range(len(save_loss))
        plt.plot(epochs, save_loss, marker='o', color='b', label='Loss')

        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.savefig(f'{gaussian_dir}/generate_loss_{view_idx}.png')
        plt.clf()
        torchvision.utils.save_image(render_pkg["render"], f'{gaussian_dir}/semi-generate_{view_idx}.png')

    gt_image = gt_image * generate_mask_image + keep
    return gt_image, mask, update_tensor, new_gaussian

def opt_gaussian_from_one_view_v2(gaussians, scene, view_idx, generate_image, generate_mask_image, opt, dist, elev, azim, DEVICE, udf_network, new_gaussian=None, gaussian_dir=None):
    views = scene.getTrainCameras()
    view = views[view_idx]
    bg_color = [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    print("optim gaussian")

    gt_image = generate_image * generate_mask_image
    torchvision.utils.save_image(gt_image, f'{gaussian_dir}/gt_{view_idx}.png')
    mask = generate_mask_image
    mask_image = Image.fromarray(mask.clone().detach().squeeze(0).cpu().byte().numpy() * 255)

    mask_image.save(f'{gaussian_dir}/mask_{view_idx}.png')
    
    camera = init_camera(dist, elev, azim, generate_image.shape[1], DEVICE)
    R, T = convert_camera_from_pytorch3d_to_colmap(camera, generate_image.shape[1], generate_image.shape[1])
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
    focal = 0.5 * generate_image.shape[1] / np.tan(0.5 * 1)
    K[0,0] = generate_image.shape[1] / 2
    K[1,1] = generate_image.shape[1] / 2
    K[0,2] = generate_image.shape[1] / 2
    K[1,2] = generate_image.shape[1] / 2
    K[2,2] = 1
    points_pixel = K @ points_c
    

    points_pixel = points_pixel / points_pixel[2:, :]
    pc_pixel = points_pixel[:2, :]

    image_size = (generate_image.shape[1], generate_image.shape[1])
    image = torch.zeros(image_size).cuda()
    
    x_grid = np.arange(generate_image.shape[1])
    y_grid = np.arange(generate_image.shape[1])
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_x = torch.from_numpy(grid_x).cuda()
    grid_y = torch.from_numpy(grid_y).cuda()

    rendering_results = gs_render(view, gaussians, pipeline, background)
    radii = rendering_results['radii']
    update_tensor = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool).cuda()

    start_time = time.time()
    pix_depth, pix_id = get_depth_with_id_2(pc_pixel, radii.float(), points_c_norm[2,:], generate_mask_image.int().squeeze(0).permute(1, 0).contiguous(), generate_image.shape[1], generate_image.shape[1])

    pix_id = pix_id.permute(1, 0, 2)

    pix_depth = pix_depth.permute(1, 0, 2)
    
    edge = cv2.Canny(np.uint8(generate_mask_image.clone().detach().squeeze(0).cpu().numpy() * 255), threshold1=100, threshold2=200)
    dilated = cv2.dilate(edge, np.ones((5,5), np.uint8), iterations=1)
    _, thresholded = cv2.threshold(np.uint8(generate_mask_image.clone().detach().squeeze(0).cpu().numpy() * 255), 200, 255, cv2.THRESH_BINARY)
    edges = cv2.bitwise_and(thresholded, dilated)
    edges = torch.from_numpy(edges.astype(bool)).bool().cuda().unsqueeze(-1)
    torchvision.utils.save_image(edges.squeeze(-1).unsqueeze(0).float(), f'{gaussian_dir}/generate_{view_idx}_edge.png')

    

    pix_id = pix_id.cpu()
    pix_depth = pix_depth.cpu()

    first_elements = pix_depth[:, :, 0].unsqueeze(-1)
    mask_tmp = (pix_depth >= (first_elements + OCCLUSION_DEPTH_EPS))
    pix_depth[mask_tmp] = -1
    pix_id[mask_tmp] = -1
    
    pix_id[:, :, DEPTH_SLOT_LIMIT:] = -1
    
    mask_tmp = (pix_id != -1)
    valid_ids = pix_id[mask_tmp]
    update_tensor[valid_ids.long().cuda()] = True

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

    update_tensor1 = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool).cuda()

    start_time = time.time()
    pix_depth, pix_id = get_depth_with_id(pc_pixel, radii.float(), points_c_norm[2,:], generate_image.shape[1], generate_image.shape[1])
    
    pix_id = pix_id.permute(1, 0, 2)

    pix_depth = pix_depth.permute(1, 0, 2)
    
    edge = cv2.Canny(np.uint8(generate_mask_image.clone().detach().squeeze(0).cpu().numpy() * 255), threshold1=100, threshold2=200)
    dilated = cv2.dilate(edge, np.ones((5,5), np.uint8), iterations=1)
    _, thresholded = cv2.threshold(np.uint8(generate_mask_image.clone().detach().squeeze(0).cpu().numpy() * 255), 200, 255, cv2.THRESH_BINARY)
    edges = cv2.bitwise_and(thresholded, dilated)
    edges = torch.from_numpy(edges.astype(bool)).bool().cuda().unsqueeze(-1)
    
    indices = torch.arange(pix_id.shape[2]).expand_as(pix_id).cuda()

    first_elements = pix_depth[:, :, 0].unsqueeze(-1)
    mask_tmp = pix_depth >= (first_elements + OCCLUSION_DEPTH_EPS)
    pix_depth[mask_tmp] = -1
    pix_id[mask_tmp] = -1
    pix_id[:, :, DEPTH_SLOT_LIMIT:] = -1
    pix_id[edges & (indices >= edge_num)] = -1
    pix_id[(~generate_mask_image.bool()).squeeze(0).unsqueeze(-1) & (indices >= 0)] = -1
    
    mask_tmp = (pix_id != -1)
    valid_ids = pix_id[mask_tmp]
    update_tensor1[valid_ids.long()] = True

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    update_tensor = torch.logical_and(update_tensor, update_tensor1)
    
    with torch.no_grad():
        opt_gaussian = copy.deepcopy(gaussians)
        opt_gaussian.select_gaussian(update_tensor)
        opt_gaussian.reset_color()

    xyz = gaussians._xyz[update_tensor.bool()]
    pc = trimesh.Trimesh(xyz.clone().detach().cpu().numpy())
    pc.export(f'{gaussian_dir}/generate_{view_idx}.obj')
    xyz = gaussians._xyz[~update_tensor.bool()]
    pc = trimesh.Trimesh(xyz.clone().detach().cpu().numpy())
    pc.export(f'{gaussian_dir}/generate_inverse_{view_idx}.obj')
    opt.iterations = 1500
    opt.position_lr_max_steps = 1500
    opt_gaussian.training_setup(opt, scene.cameras_extent)

    opt_num = 0
    
    save_loss = []
    for iteration in tqdm(range(opt.iterations)):
        opt_gaussian.update_learning_rate(iteration)
        background = torch.rand(3).cuda()

        view.full_proj_transform.requires_grad = True
        view.world_view_transform.requires_grad = True
        render_pkg = gs_render(view, opt_gaussian, pipeline, background)
        image, viewspace_point_tensor, visibility_filter, radii, render_mask = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['rendered_alpha']
        image = image * generate_mask_image
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        rend_dist = render_pkg["rend_dist"]
        Ll1 = l1_loss(image, gt_image)
        Ll1_mask = l1_loss(generate_mask_image, render_mask)
        scale_loss = torch.mean(torch.square(torch.clamp(opt_gaussian._scaling, max=0.9*middle) - opt_gaussian._scaling))
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        
        udf = udf_network(opt_gaussian._xyz).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.0 * Ll1_mask + 10.0 * scale_loss + 50 * udf 
        
        loss.backward()
        if (iteration + 1) % 100 == 0:
            save_loss.append((1.0 - ssim(image, gt_image)).clone().detach().cpu().numpy())

        if iteration % 100 == 0:
            print((1.0 - ssim(image, gt_image)).item())

        with torch.no_grad():
            if iteration < 400:
                opt_gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration % 100 == 0 and iteration > 0:
                    size_threshold = 20
                    opt_num += opt_gaussian.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
            
            if iteration >= 400 and opt_gaussian._xyz.grad is not None:
                grad_mask = torch.ones_like(opt_gaussian._xyz.grad)
                grad_mask[:,:] = 0
                opt_gaussian._xyz.grad *= grad_mask

            if opt_gaussian._xyz.grad is not None:
                grad_mask = torch.ones_like(opt_gaussian._xyz.grad)
                grad_mask[opt_num:,:] = 0
                opt_gaussian._xyz.grad *= grad_mask

                        
            opt_gaussian.optimizer.step()
            opt_gaussian.optimizer.zero_grad(set_to_none = True)
    
    with torch.no_grad():
        gaussians.num_no_opt = gaussians._xyz[~update_tensor].shape[0]
        gaussians._xyz = nn.Parameter(torch.cat((gaussians._xyz[~update_tensor], opt_gaussian._xyz), dim=0).requires_grad_(True))
        gaussians._features_dc = nn.Parameter(torch.cat((gaussians._features_dc[~update_tensor], opt_gaussian._features_dc), dim=0).requires_grad_(True))
        gaussians._features_rest = nn.Parameter(torch.cat((gaussians._features_rest[~update_tensor], opt_gaussian._features_rest), dim=0).requires_grad_(True))
        gaussians._opacity = nn.Parameter(torch.cat((gaussians._opacity[~update_tensor], opt_gaussian._opacity), dim=0).requires_grad_(True))
        gaussians._scaling = nn.Parameter(torch.cat((gaussians._scaling[~update_tensor], opt_gaussian._scaling), dim=0).requires_grad_(True))
        gaussians._rotation = nn.Parameter(torch.cat((gaussians._rotation[~update_tensor], opt_gaussian._rotation), dim=0).requires_grad_(True))
        gaussians._update[update_tensor] = 1
        gaussians._update = gaussians._update[~update_tensor.bool()]
        gaussians._update = torch.cat((gaussians._update, torch.ones(opt_gaussian._xyz.shape[0]).cuda()),dim=0)
        
        if view_idx == 0 or new_gaussian == None:
            new_gaussian = opt_gaussian
        else:
            new_gaussian._xyz = nn.Parameter(torch.cat((new_gaussian._xyz, opt_gaussian._xyz), dim=0).requires_grad_(True))
            new_gaussian._features_dc = nn.Parameter(torch.cat((new_gaussian._features_dc, opt_gaussian._features_dc), dim=0).requires_grad_(True))
            new_gaussian._features_rest = nn.Parameter(torch.cat((new_gaussian._features_rest, opt_gaussian._features_rest), dim=0).requires_grad_(True))
            new_gaussian._opacity = nn.Parameter(torch.cat((new_gaussian._opacity, opt_gaussian._opacity), dim=0).requires_grad_(True))
            new_gaussian._scaling = nn.Parameter(torch.cat((new_gaussian._scaling, opt_gaussian._scaling), dim=0).requires_grad_(True))
            new_gaussian._rotation = nn.Parameter(torch.cat((new_gaussian._rotation, opt_gaussian._rotation), dim=0).requires_grad_(True))

    with torch.no_grad():
        background = torch.tensor([0.0,0.0,0.0]).cuda()
        render_pkg = gs_render(view, opt_gaussian, pipeline, background)
        epochs = range(len(save_loss))
        plt.plot(epochs, save_loss, marker='o', color='b', label='Loss')

        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.savefig(f'{gaussian_dir}/generate_loss_{view_idx}.png')
        plt.clf()
        torchvision.utils.save_image(render_pkg["render"], f'{gaussian_dir}/semi-generate_{view_idx}.png')

    return gt_image, generate_mask_image, update_tensor, new_gaussian

def opt_gaussian_from_one_view_generate(gaussians, scene, view_idx, generate_image, generate_mask_image, opt, init_image, dist, elev, azim, sector, DEVICE, udf_network, new_gaussian=None, gaussian_dir=None, second=True):
    
    views = scene.getTrainCameras()
    if second:
        view = views[0]
    else:
        view = views[view_idx]
    bg_color = [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    print("optim gaussian")

    init_image = init_image[:,:,:3].permute(2, 0, 1)

    gt_image = (generate_image.permute(2, 0, 1) / 255.0)

    mask = generate_mask_image.unsqueeze(0)

    dilate_pixels = 3
    kernel_size = 2 * dilate_pixels + 1
    kernel = torch.ones((1, 1, kernel_size, kernel_size)).cuda()

    dilated = F.conv2d(mask, kernel, padding=dilate_pixels)
    dilated = (dilated > 0).float()
    generate_mask_image = dilated.squeeze(0).float()

    gt_image = gt_image * generate_mask_image
    torchvision.utils.save_image(gt_image, f'{gaussian_dir}/gt_{view_idx}.png')

    mask_image = Image.fromarray(generate_mask_image.clone().detach().squeeze(0).cpu().byte().numpy() * 255)

    mask_image.save(f'{gaussian_dir}/mask_{view_idx}.png')
    
    camera = init_camera(dist, elev, azim, init_image.shape[1], DEVICE)
    R, T = convert_camera_from_pytorch3d_to_colmap(camera, init_image.shape[1], init_image.shape[1])
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
    focal = 0.5 * init_image.shape[1] / np.tan(0.5 * 1)
    K[0,0] = init_image.shape[1] / 2
    K[1,1] = init_image.shape[1] / 2
    K[0,2] = init_image.shape[1] / 2
    K[1,2] = init_image.shape[1] / 2
    K[2,2] = 1
    points_pixel = K @ points_c
    
    points_pixel = points_pixel / points_pixel[2:, :]
    pc_pixel = points_pixel[:2, :]

    image_size = (init_image.shape[1], init_image.shape[1])
    image = torch.zeros(image_size).cuda()
    
    x_grid = np.arange(init_image.shape[1])
    y_grid = np.arange(init_image.shape[1])
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_x = torch.from_numpy(grid_x).cuda()
    grid_y = torch.from_numpy(grid_y).cuda()

    rendering_results = gs_render(view, gaussians, pipeline, background)
    radii = rendering_results['radii']

    start_time = time.time()
   

    update_tensor = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool)

    start_time = time.time()
    pix_depth, pix_id = get_depth_with_id(pc_pixel, radii.float(), points_c_norm[2,:], init_image.shape[1], init_image.shape[1])
    
    pix_id = pix_id.permute(1, 0, 2)

    pix_depth = pix_depth.permute(1, 0, 2)
    

    first_elements = pix_depth[:, :, 0].unsqueeze(-1)
    mask_tmp = pix_depth >= (first_elements + OCCLUSION_DEPTH_EPS)
    pix_depth[mask_tmp] = -1
    pix_id[mask_tmp] = -1
    pix_id[:, :, DEPTH_SLOT_LIMIT:] = -1

    pix_id = pix_id.cpu()
    indices = torch.arange(pix_id.shape[2]).expand_as(pix_id)
    pix_id[(~generate_mask_image.bool()).squeeze(0).unsqueeze(-1).cpu() & (indices >= 0)] = -1
    
    mask_tmp = (pix_id != -1)
    valid_ids = pix_id[mask_tmp].cuda()
    update_tensor[valid_ids.long()] = True

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

    xyz = gaussians._xyz[update_tensor].clone().detach().cpu().numpy()
    point_cloud = trimesh.Trimesh(xyz)
    point_cloud.export(f"{gaussian_dir}/generate_{view_idx}.obj")
    xyz = gaussians._xyz[~update_tensor].clone().detach().cpu().numpy()
    point_cloud = trimesh.Trimesh(xyz)
    point_cloud.export(f"{gaussian_dir}/generate_inver_{view_idx}.obj")
    xyz = gaussians._xyz[update_tensor]
    ones = torch.ones((xyz.shape[0], 1)).cuda()      
    points_w = torch.cat((xyz, ones), dim=1).permute(1, 0)
    camera = init_camera(dist, elev, azim, init_image.shape[1], DEVICE)
    R, T = convert_camera_from_pytorch3d_to_colmap(camera, init_image.shape[1], init_image.shape[1])
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    world_view_transform = torch.from_numpy(Rt).cuda().float()
    points_c = world_view_transform @ points_w
    K = torch.zeros([3,4]).cuda()
    focal = 0.5 * init_image.shape[1] / np.tan(0.5 * 1)
    K[0,0] = init_image.shape[1] / 2
    K[1,1] = init_image.shape[1] / 2
    K[0,2] = init_image.shape[1] / 2
    K[1,2] = init_image.shape[1] / 2
    K[2,2] = 1
    points_pixel = K @ points_c
    points_pixel = points_pixel / points_pixel[2:, :]
    pc_pixel = points_pixel[:2, :]
    rendering_results = gs_render(view, gaussians, pipeline, background)
    radii = rendering_results['radii']
    tmp_r = radii[update_tensor]

    x_grid = np.arange(init_image.shape[1])
    y_grid = np.arange(init_image.shape[1])
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_x = torch.from_numpy(grid_x).cuda()
    grid_y = torch.from_numpy(grid_y).cuda()
    image_size = (init_image.shape[1], init_image.shape[1])
    image = torch.zeros(image_size).cuda()
    for (cx, cy), radius in zip(pc_pixel.T, tmp_r):
        square_dist = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        image += (square_dist <= radius**2)

    image = torch.clip(image, 0, 1)
    torchvision.utils.save_image(image, f'{gaussian_dir}/generate_mask_{view_idx}.png')
    
    with torch.no_grad():
        opt_gaussian = copy.deepcopy(gaussians)
        opt_gaussian.select_gaussian(update_tensor)

    opt.iterations = 1000
    opt.position_lr_max_steps = 1000
    opt_gaussian.training_setup(opt, scene.cameras_extent)

    opt_num = 0
    
    save_loss = []

    for iteration in tqdm(range(opt.iterations)):
        opt_gaussian.update_learning_rate(iteration)
        background = torch.rand(3).cuda()

        
        render_pkg = gs_render(view, opt_gaussian, pipeline, background)
        image, viewspace_point_tensor, visibility_filter, radii, render_mask = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['rendered_alpha']
        image = image * generate_mask_image
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        rend_dist = render_pkg["rend_dist"]
        Ll1 = l1_loss(image, gt_image)
        Ll1_mask = l1_loss(generate_mask_image, render_mask)
        scale_loss = torch.mean(torch.square(torch.clamp(opt_gaussian._scaling, max=0.9*middle) - opt_gaussian._scaling))
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        udf = udf_network(opt_gaussian._xyz).mean()

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.0 * Ll1_mask + 20.0 * scale_loss + 50.0 * udf
        
        loss.backward()
        if (iteration + 1) % 100 == 0:
            save_loss.append((1.0 - ssim(image, gt_image)).clone().detach().cpu().numpy())

        if iteration % 100 == 0:
            print((1.0 - ssim(image, gt_image)).item())
        
        with torch.no_grad():
            if iteration < 800:
                opt_gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration % 100 == 0 and iteration > 0:
                size_threshold = 20
                opt_num += opt_gaussian.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
            
            if iteration >= 1000:
                grad_mask = torch.ones_like(opt_gaussian._xyz.grad)
                grad_mask[:,:] = 0
                opt_gaussian._xyz.grad *= grad_mask

            if opt_gaussian._xyz.grad is not None:
                grad_mask = torch.ones_like(opt_gaussian._xyz.grad)
                grad_mask[opt_num:,:] = 0
                opt_gaussian._xyz.grad *= grad_mask
            

                
            opt_gaussian.optimizer.step()
            opt_gaussian.optimizer.zero_grad(set_to_none = True)
    
    with torch.no_grad():
        gaussians.num_no_opt = gaussians._xyz[~update_tensor].shape[0]
        gaussians._xyz = nn.Parameter(torch.cat((gaussians._xyz[~update_tensor], opt_gaussian._xyz), dim=0).requires_grad_(True))
        gaussians._features_dc = nn.Parameter(torch.cat((gaussians._features_dc[~update_tensor], opt_gaussian._features_dc), dim=0).requires_grad_(True))
        gaussians._features_rest = nn.Parameter(torch.cat((gaussians._features_rest[~update_tensor], opt_gaussian._features_rest), dim=0).requires_grad_(True))
        gaussians._opacity = nn.Parameter(torch.cat((gaussians._opacity[~update_tensor], opt_gaussian._opacity), dim=0).requires_grad_(True))
        gaussians._scaling = nn.Parameter(torch.cat((gaussians._scaling[~update_tensor], opt_gaussian._scaling), dim=0).requires_grad_(True))
        gaussians._rotation = nn.Parameter(torch.cat((gaussians._rotation[~update_tensor], opt_gaussian._rotation), dim=0).requires_grad_(True))
        gaussians._update[update_tensor] = 1
        gaussians._update = gaussians._update[~update_tensor.bool()]
        gaussians._update = torch.cat((gaussians._update, torch.ones(opt_gaussian._xyz.shape[0]).cuda()),dim=0)

    with torch.no_grad():
        background = torch.tensor([0.0,0.0,0.0]).cuda()
        render_pkg = gs_render(view, opt_gaussian, pipeline, background)
        epochs = range(len(save_loss))
        plt.plot(epochs, save_loss, marker='o', color='b', label='Loss')

        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.savefig(f'{gaussian_dir}/generate_loss_{view_idx}.png')
        plt.clf()
        torchvision.utils.save_image(render_pkg["render"], f'{gaussian_dir}/semi-generate_{view_idx}.png')

    return update_tensor, new_gaussian

def opt_gaussian_from_one_view_overlap(gaussians, scene, view_idx, generate_image, generate_mask_image, opt, dist, elev, azim, sector, DEVICE, udf_network, new_gaussian=None, gaussian_dir=None):
    views = scene.getTrainCameras()
    view = views[view_idx]
    bg_color = [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    print("optim gaussian")

    gt_image = generate_image * generate_mask_image
    torchvision.utils.save_image(gt_image, f'{gaussian_dir}/gt_overlap_{view_idx}.png')

    mask_image = Image.fromarray(generate_mask_image.clone().detach().squeeze(0).cpu().byte().numpy() * 255)

    mask_image.save(f'{gaussian_dir}/mask_overlap_{view_idx}.png')
    

    camera = init_camera(dist, elev, azim, generate_image.shape[1], DEVICE)
    R, T = convert_camera_from_pytorch3d_to_colmap(camera, generate_image.shape[1], generate_image.shape[1])
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
    focal = 0.5 * generate_image.shape[1] / np.tan(0.5 * 1)
    K[0,0] = generate_image.shape[1] / 2
    K[1,1] = generate_image.shape[1] / 2
    K[0,2] = generate_image.shape[1] / 2
    K[1,2] = generate_image.shape[1] / 2
    K[2,2] = 1
    points_pixel = K @ points_c
    
    points_pixel = points_pixel / points_pixel[2:, :]
    pc_pixel = points_pixel[:2, :]

    image_size = (generate_image.shape[1], generate_image.shape[1])
    image = torch.zeros(image_size).cuda()
    
    x_grid = np.arange(generate_image.shape[1])
    y_grid = np.arange(generate_image.shape[1])
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_x = torch.from_numpy(grid_x).cuda()
    grid_y = torch.from_numpy(grid_y).cuda()

    rendering_results = gs_render(view, gaussians, pipeline, background)
    radii = rendering_results['radii']

    start_time = time.time()
   
    update_tensor_1 = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool)

    start_time = time.time()
    pix_depth, pix_id = get_depth_with_id(pc_pixel, radii.float(), points_c_norm[2,:], generate_image.shape[1], generate_image.shape[1])
    
    pix_id = pix_id.permute(1, 0, 2).cpu()

    pix_depth = pix_depth.permute(1, 0, 2)
    
    
    first_elements = pix_depth[:, :, 0].unsqueeze(-1)
    mask_tmp = pix_depth >= (first_elements + OCCLUSION_DEPTH_EPS)
    pix_depth[mask_tmp] = -1
    pix_id[mask_tmp] = -1
    pix_id[:, :, DEPTH_SLOT_LIMIT:] = -1
    
    pix_id = pix_id.cpu()
    indices = torch.arange(pix_id.shape[2]).expand_as(pix_id)
    pix_id[(~generate_mask_image.bool()).squeeze(0).unsqueeze(-1).cpu() & (indices >= 0)] = -1
    
    mask_tmp = (pix_id != -1)
    valid_ids = pix_id[mask_tmp].cuda()
    update_tensor_1[valid_ids.long()] = True

    update_tensor = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool)
    pix_depth, pix_id = get_depth_with_id_2(pc_pixel, radii.float(), points_c_norm[2,:], generate_mask_image.int().squeeze(0).permute(1, 0).contiguous(), generate_image.shape[1], generate_image.shape[1])

    pix_id = pix_id.permute(1, 0, 2).cpu()
    pix_depth = pix_depth.permute(1, 0, 2)
    
    edge = cv2.Canny(np.uint8(generate_mask_image.clone().detach().squeeze(0).cpu().numpy() * 255), threshold1=100, threshold2=200)
    dilated = cv2.dilate(edge, np.ones((5,5), np.uint8), iterations=2)
    _, thresholded = cv2.threshold(np.uint8(generate_mask_image.clone().detach().squeeze(0).cpu().numpy() * 255), 200, 255, cv2.THRESH_BINARY)
    edges = cv2.bitwise_and(thresholded, dilated)
    edges = torch.from_numpy(edges.astype(bool)).bool().cuda().unsqueeze(-1)
    
    indices = torch.arange(pix_id.shape[2]).expand_as(pix_id)

    first_elements = pix_depth[:, :, 0].unsqueeze(-1)
    mask_tmp = (pix_depth >= (first_elements + OCCLUSION_DEPTH_EPS))
    pix_depth[mask_tmp] = -1
    pix_id[mask_tmp] = -1
    
    pix_id[:, :, DEPTH_SLOT_LIMIT:] = -1

    
    
    mask_tmp = (pix_id != -1)
    valid_ids = pix_id[mask_tmp]
    update_tensor[valid_ids.long()] = True

    update_tensor = torch.logical_and(update_tensor.cuda(), update_tensor_1.cuda())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
    with torch.no_grad():
        opt_gaussian = copy.deepcopy(gaussians)
        opt_gaussian.select_gaussian(update_tensor)

    opt.iterations = 1000
    opt.position_lr_max_steps = 1000
    opt_gaussian.training_setup(opt, scene.cameras_extent)

    opt_num = 0
    
    save_loss = []

    for iteration in tqdm(range(opt.iterations)):
        opt_gaussian.update_learning_rate(iteration)
        background = torch.rand(3).cuda()

        
        render_pkg = gs_render(view, opt_gaussian, pipeline, background)
        image, viewspace_point_tensor, visibility_filter, radii, render_mask = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['rendered_alpha']
        image = image * generate_mask_image
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        rend_dist = render_pkg["rend_dist"]
        Ll1 = l1_loss(image, gt_image)
        Ll1_mask = l1_loss(generate_mask_image, render_mask)
        scale_loss = torch.mean(torch.square(torch.clamp(opt_gaussian._scaling, max=0.94*middle) - opt_gaussian._scaling))
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        udf = udf_network(opt_gaussian._xyz).mean()

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.0 * Ll1_mask + 20.0 * scale_loss + 50.0 * udf
        
        loss.backward()
        if (iteration + 1) % 100 == 0:
            save_loss.append((1.0 - ssim(image, gt_image)).clone().detach().cpu().numpy())

        if iteration % 100 == 0:
            print((1.0 - ssim(image, gt_image)).item())
        
        with torch.no_grad():
            
            if iteration >= 1000:
                grad_mask = torch.ones_like(opt_gaussian._xyz.grad)
                grad_mask[:,:] = 0
                opt_gaussian._xyz.grad *= grad_mask

            if opt_gaussian._xyz.grad is not None:
                grad_mask = torch.ones_like(opt_gaussian._xyz.grad)
                grad_mask[opt_num:,:] = 0
                opt_gaussian._xyz.grad *= grad_mask
            

                
            opt_gaussian.optimizer.step()
            opt_gaussian.optimizer.zero_grad(set_to_none = True)
    
    with torch.no_grad():
        gaussians.num_no_opt = gaussians._xyz[~update_tensor].shape[0]
        gaussians._xyz = nn.Parameter(torch.cat((gaussians._xyz[~update_tensor], opt_gaussian._xyz), dim=0).requires_grad_(True))
        gaussians._features_dc = nn.Parameter(torch.cat((gaussians._features_dc[~update_tensor], opt_gaussian._features_dc), dim=0).requires_grad_(True))
        gaussians._features_rest = nn.Parameter(torch.cat((gaussians._features_rest[~update_tensor], opt_gaussian._features_rest), dim=0).requires_grad_(True))
        gaussians._opacity = nn.Parameter(torch.cat((gaussians._opacity[~update_tensor], opt_gaussian._opacity), dim=0).requires_grad_(True))
        gaussians._scaling = nn.Parameter(torch.cat((gaussians._scaling[~update_tensor], opt_gaussian._scaling), dim=0).requires_grad_(True))
        gaussians._rotation = nn.Parameter(torch.cat((gaussians._rotation[~update_tensor], opt_gaussian._rotation), dim=0).requires_grad_(True))
        gaussians._update[update_tensor] = 1
        gaussians._update = gaussians._update[~update_tensor.bool()]
        gaussians._update = torch.cat((gaussians._update, torch.ones(opt_gaussian._xyz.shape[0]).cuda()),dim=0)

    
    with torch.no_grad():
        background = torch.tensor([0.0,0.0,0.0]).cuda()
        render_pkg = gs_render(view, opt_gaussian, pipeline, background)
        epochs = range(len(save_loss))
        plt.plot(epochs, save_loss, marker='o', color='b', label='Loss')

        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.savefig(f'{gaussian_dir}/generate_loss_{view_idx}.png')
        plt.clf()
        torchvision.utils.save_image(render_pkg["render"], f'{gaussian_dir}/semi-generate_{view_idx}.png')

    return update_tensor
