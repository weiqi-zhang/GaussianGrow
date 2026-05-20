import os
import torch

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from PIL import Image

from gaussian_grow.core.constants import *
from gaussian_grow.core.camera_helper import polar_to_xyz

def visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_score, device):
    quad_mask_tensor = quad_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
    quad_mask_image_tensor = torch.zeros_like(quad_mask_tensor)
    
    for idx in PALETTE:
        selected = quad_mask_tensor[quad_mask_tensor == idx].reshape(-1, 3)
        selected = torch.FloatTensor(PALETTE[idx]).to(device).unsqueeze(0).repeat(selected.shape[0], 1)

        quad_mask_image_tensor[quad_mask_tensor == idx] = selected.reshape(-1)

    quad_mask_image_np = quad_mask_image_tensor[0].cpu().numpy().astype(np.uint8)
    quad_mask_image = Image.fromarray(quad_mask_image_np).convert("RGB")
    quad_mask_image.save(os.path.join(mask_image_dir, "{}_quad_{:.5f}.png".format(view_idx, view_score)))


def visualize_principle_viewpoints(output_dir, dist_list, elev_list, azim_list):
    theta_list = [e for e in azim_list]
    phi_list = [90 - e for e in elev_list]
    DIST = dist_list[0]

    xyz_list = [polar_to_xyz(theta, phi, DIST) for theta, phi in zip(theta_list, phi_list)]

    xyz_np = np.array(xyz_list)
    color_np = np.array([[0, 0, 0]]).repeat(xyz_np.shape[0], 0)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    SCALE = 0.8
    ax.set_xlim((-DIST, DIST))
    ax.set_ylim((-DIST, DIST))
    ax.set_zlim((-SCALE * DIST, SCALE * DIST))

    ax.scatter(xyz_np[:, 0], xyz_np[:, 2], xyz_np[:, 1], s=100, c=color_np, depthshade=True, label="Principle views")
    ax.scatter([0], [0], [0], c=[[1, 0, 0]], s=100, depthshade=True, label="Object center")

    # draw hemisphere
    # theta inclination angle
    # phi azimuthal angle
    n_theta = 50    # number of values for theta
    n_phi = 200     # number of values for phi
    r = DIST        #radius of sphere

    # theta, phi = np.mgrid[0.0:0.5*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]
    theta, phi = np.mgrid[0.0:1*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.25, linewidth=1)

    # Make the grid
    ax.quiver(
        xyz_np[:, 0], 
        xyz_np[:, 2], 
        xyz_np[:, 1], 
        -xyz_np[:, 0], 
        -xyz_np[:, 2], 
        -xyz_np[:, 1],
        normalize=True,
        length=0.3
    )

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    ax.view_init(30, 35)
    ax.legend()

    plt.show()

    plt.savefig(os.path.join(output_dir, "principle_viewpoints.png"))



