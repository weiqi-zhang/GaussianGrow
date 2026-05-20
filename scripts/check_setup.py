#!/usr/bin/env python
"""Preflight checks for a GaussianGrow runtime environment."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PYTHON_IMPORTS = [
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("pytorch3d", "PyTorch3D"),
    ("cv2", "opencv-python"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("trimesh", "trimesh"),
    ("open3d", "Open3D"),
    ("diffusers", "diffusers"),
    ("kornia", "kornia"),
    ("omegaconf", "OmegaConf"),
    ("pytorch_lightning", "PyTorch Lightning"),
    ("einops", "einops"),
    ("rembg", "rembg"),
    ("xatlas", "xatlas"),
    ("pygltflib", "pygltflib"),
    ("plyfile", "plyfile"),
    ("scipy", "SciPy"),
    ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"),
    ("imageio", "imageio"),
]

CUDA_EXTENSION_IMPORTS = [
    ("depthid_render", "local extension: depthid_render"),
    ("depthid_render_mask_control_v2", "local extension: depthid_render_mask_control_v2"),
    ("find_max_in_circles", "local extension: find_max_in_circles"),
    ("custom_rasterizer", "local extension: Hunyuan3D custom_rasterizer"),
    ("custom_rasterizer_kernel", "local extension: Hunyuan3D custom_rasterizer_kernel"),
    ("diff_surfel_rasterization", "external extension: diff_surfel_rasterization"),
    ("simple_knn._C", "external extension: simple_knn"),
]


def _status(ok: bool, message: str) -> None:
    tag = "OK" if ok else "FAIL"
    print(f"[{tag}] {message}")


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _try_import(module_name: str, label: str) -> str | None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - preflight should report any import failure.
        return f"{label} import failed ({module_name}): {exc}"
    _status(True, f"{label} import")
    return None


def _check_path(path: Path, label: str) -> str | None:
    if path.exists():
        _status(True, f"{label}: {path}")
        return None
    return f"Missing {label}: {path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether GaussianGrow can run in this environment.")
    parser.add_argument("--allow-no-cuda", action="store_true", help="Do not fail when CUDA is unavailable.")
    parser.add_argument("--allow-missing-assets", action="store_true", help="Do not fail on missing model/assets paths.")
    parser.add_argument("--skip-imports", action="store_true", help="Only check paths and CUDA.")
    args = parser.parse_args()

    failures: list[str] = []

    sys.path.insert(0, str(ROOT))

    if not args.skip_imports:
        for module_name, label in PYTHON_IMPORTS:
            failure = _try_import(module_name, label)
            if failure:
                failures.append(failure)

        for module_name, label in CUDA_EXTENSION_IMPORTS:
            failure = _try_import(module_name, label)
            if failure:
                failures.append(failure)

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            _status(True, f"CUDA available: {name}")
        elif args.allow_no_cuda:
            _warn("CUDA is not available; full pipeline execution requires CUDA.")
        else:
            failures.append("CUDA is not available; set up a CUDA PyTorch build and CUDA_VISIBLE_DEVICES.")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Unable to check CUDA because PyTorch failed to import: {exc}")

    input_dir = Path(os.environ.get("INPUT_DIR", ROOT / "examples" / "fire_dragon"))
    obj_file = os.environ.get("OBJ_FILE", "dragon_normalized.obj")
    maybe_input = _check_path(input_dir / obj_file, "input mesh")
    if maybe_input:
        failures.append(maybe_input)

    controlnet_weight = Path(
        os.environ.get(
            "CONTROLNET_DEPTH_CKPT",
            ROOT / "models" / "ControlNet" / "models" / "control_sd15_depth.pth",
        )
    ).expanduser()
    maybe_controlnet = _check_path(controlnet_weight, "ControlNet depth checkpoint")
    if maybe_controlnet:
        if args.allow_missing_assets:
            _warn(maybe_controlnet)
        else:
            failures.append(maybe_controlnet)

    controlnet_config = ROOT / "models" / "ControlNet" / "models" / "cldm_v15.yaml"
    maybe_controlnet_config = _check_path(controlnet_config, "ControlNet config")
    if maybe_controlnet_config:
        failures.append(maybe_controlnet_config)

    capudf_root = os.environ.get("CAPUDF_ROOT")
    if capudf_root:
        maybe_capudf = _check_path(Path(capudf_root).expanduser() / "run.py", "CAP-UDF run.py")
        if maybe_capudf:
            if args.allow_missing_assets:
                _warn(maybe_capudf)
            else:
                failures.append(maybe_capudf)
    else:
        if args.allow_missing_assets:
            _warn("CAPUDF_ROOT is not set.")
        else:
            failures.append("CAPUDF_ROOT is not set.")

    hunyuan_path = os.environ.get("HUNYUAN3D_MODEL_PATH")
    if hunyuan_path:
        maybe_hunyuan = _check_path(Path(hunyuan_path).expanduser(), "Hunyuan3D model path")
        if maybe_hunyuan:
            if args.allow_missing_assets:
                _warn(maybe_hunyuan)
            else:
                failures.append(maybe_hunyuan)
    else:
        _warn("HUNYUAN3D_MODEL_PATH is not set; Hunyuan3D may try to download weights from Hugging Face.")

    if failures:
        print("\nPreflight failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
