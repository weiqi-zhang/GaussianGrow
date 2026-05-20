import torch

import cv2
import numpy as np

from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from models.ControlNet.gradio_depth2image import init_model, process

def detect_and_extend_edges_with_mask(mask, extension=3):
    """
    Dilate mask edges while preserving the original mask body.
    
    Args:
        mask: Binary mask tensor with shape [1, H, W].
        extension: Number of dilation iterations.
    
    Returns:
        Combined binary mask with shape [1, H, W].
    """

    mask = (mask > 0).float().unsqueeze(0)

    sobel_kernel_x = torch.tensor([[ -1, 0, 1],
                                   [ -2, 0, 2],
                                   [ -1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[ -1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_kernel_x = sobel_kernel_x.to(mask.device)
    sobel_kernel_y = sobel_kernel_y.to(mask.device)

    grad_x = F.conv2d(mask, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_kernel_y, padding=1)

    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    edge = (grad > 0).float()

    dilation_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32).to(mask.device)

    dilated_edge = edge
    for _ in range(extension):
        dilated_edge = F.conv2d(dilated_edge, dilation_kernel, padding=1)
        dilated_edge = (dilated_edge > 0).float()

    extended_edge = dilated_edge - edge
    extended_edge = torch.clamp(extended_edge, min=0)

    original_mask = mask.squeeze(0)
    combined_mask = torch.clamp(original_mask + extended_edge, max=1.0)

    combined_mask = combined_mask.squeeze(0)

    return combined_mask

def detect_and_extend_edges_with_depth(depth_image, extension=3):
    depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    _, binary = cv2.threshold(depth_gray, 0, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 100, 200)
    kernel = np.ones((3,3), np.uint8)

    dilated_edges = cv2.dilate(edges, kernel, iterations=extension)
    mask = dilated_edges > 0
    edge_depth = depth_gray.copy()
    edge_depth[edges == 0] = 0
    extended_depth = depth_gray.copy()
    extended_depth[mask] = edge_depth[mask]
    extended_depth = np.where((mask) & (depth_gray == 0), edge_depth, depth_gray)
    extended_depth = cv2.cvtColor(extended_depth, cv2.COLOR_GRAY2BGR)

    return extended_depth

def get_controlnet_depth():
    print("=> initializing ControlNet Depth...")
    model, ddim_sampler = init_model()

    return model, ddim_sampler

@torch.no_grad()
def apply_controlnet_depth(model, ddim_sampler, 
    init_image, prompt, strength, ddim_steps,
    generate_mask_image, keep_mask_image, depth_map_np, 
    a_prompt, n_prompt, guidance_scale, seed, eta, num_samples,
    device, blend=0, save_memory=False):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_np: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """

    print("=> generating ControlNet Depth RePaint image...")

    # ControlNet expects PIL inputs. White mask pixels are inpainted; black pixels are preserved.
    diffused_image_np = process(
        model, ddim_sampler,
        np.array(init_image), prompt, a_prompt, n_prompt, num_samples,
        ddim_steps, guidance_scale, seed, eta, 
        strength=strength, detected_map=depth_map_np, unknown_mask=np.array(generate_mask_image), save_memory=save_memory
    )[0]

    init_image = init_image.convert("RGB")
    diffused_image = Image.fromarray(diffused_image_np).convert("RGB")

    if blend > 0 and transforms.ToTensor()(keep_mask_image).sum() > 0:
        print("=> blending the generated region...")
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        keep_image_np = np.array(init_image).astype(np.uint8)
        keep_image_np_dilate = cv2.dilate(keep_image_np, kernel, iterations=1)

        keep_mask_np = np.array(keep_mask_image).astype(np.uint8)
        keep_mask_np_dilate = cv2.dilate(keep_mask_np, kernel, iterations=1)

        generate_image_np = np.array(diffused_image).astype(np.uint8)

        overlap_mask_np = np.array(generate_mask_image).astype(np.uint8)
        overlap_mask_np *= keep_mask_np_dilate
        print("=> blending {} pixels...".format(np.sum(overlap_mask_np)))

        overlap_keep = keep_image_np_dilate[overlap_mask_np == 1]
        overlap_generate = generate_image_np[overlap_mask_np == 1]

        overlap_np = overlap_keep * blend + overlap_generate * (1 - blend)

        generate_image_np[overlap_mask_np == 1] = overlap_np

        diffused_image = Image.fromarray(generate_image_np.astype(np.uint8)).convert("RGB")

        init_image_masked = init_image
        diffused_image_masked = diffused_image
        return diffused_image, init_image_masked, diffused_image_masked, torch.from_numpy(generate_image_np).cuda()

    init_image_masked = init_image
    diffused_image_masked = diffused_image

    return diffused_image, init_image_masked, diffused_image_masked, torch.from_numpy(diffused_image_np).cuda()

