# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import argparse
import os
import sys
from pathlib import Path

import trimesh
from PIL import Image

HUNYUAN_ROOT = Path(__file__).resolve().parent
if str(HUNYUAN_ROOT) not in sys.path:
    sys.path.insert(0, str(HUNYUAN_ROOT))

from hy3dgen.rembg import BackgroundRemover


def texture_mesh(
    save_path='./outputs/test',
    mesh_path=None,
    model_path=None,
    image_name='main_view.png',
    output_name='texture.glb',
):
    if mesh_path is None:
        raise ValueError("--mesh_path is required for texturing an existing mesh.")

    save_path = Path(save_path)
    image_path = save_path / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Missing generated main-view image: {image_path}")
    if not Path(mesh_path).exists():
        raise FileNotFoundError(f"Missing mesh file: {mesh_path}")

    model_path = model_path or os.environ.get("HUNYUAN3D_MODEL_PATH", "tencent/Hunyuan3D-2")
    rembg = BackgroundRemover()
    image = Image.open(image_path)

    if image.mode == 'RGB':
        image = rembg(image)

    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
    mesh = trimesh.load_mesh(mesh_path)
    mesh = pipeline(mesh, image=image, save_path=str(save_path))
    image.save(save_path / "input_paint.png")
    mesh.export(save_path / output_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="stage3")
    parser.add_argument("--save_path", type=str, default="./outputs/test", help="Path to save the rendered images.")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the input mesh to texture.")
    parser.add_argument("--model_path", type=str, default=None, help="Local Hunyuan3D model path or Hugging Face repo id.")
    parser.add_argument("--image_name", type=str, default="main_view.png", help="Generated condition image name inside save_path.")
    parser.add_argument("--output_name", type=str, default="texture.glb", help="Output textured mesh name inside save_path.")
    args = parser.parse_args()

    texture_mesh(
        save_path=args.save_path,
        mesh_path=args.mesh_path,
        model_path=args.model_path,
        image_name=args.image_name,
        output_name=args.output_name,
    )
    print("Stage 3 finished.")
