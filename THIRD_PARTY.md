# Third-Party Components

GaussianGrow is released under the MIT License (see `LICENSE`). It bundles or
depends on code from several upstream research projects, each governed by its
own license. By using this repository you also agree to the corresponding
upstream terms.

## Bundled Source Trees

| Path | Project | Upstream | Upstream License | Notes |
|------|---------|----------|------------------|-------|
| `Hunyuan3D/`, `Hunyuan3D/hy3dgen/` | Hunyuan3D 2.0 (texture pipeline) | https://github.com/Tencent/Hunyuan3D-2 | TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT | License headers are preserved in the bundled source files (see `Hunyuan3D/minimal_demo.py`). Non-commercial use only. |
| `models/ControlNet/` | ControlNet | https://github.com/lllyasviel/ControlNet | Apache License 2.0 | Original LICENSE / NOTICE files should be obtained from the upstream repository when distributing. |
| `models/ControlNet/annotator/uniformer/` | mmcv / mmsegmentation | https://github.com/open-mmlab/mmcv, https://github.com/open-mmlab/mmsegmentation | Apache License 2.0 | Re-vendored as part of the ControlNet release. |
| `models/delight/` | Hunyuan3D Delight helper | https://github.com/Tencent/Hunyuan3D-2 | TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT | Same Tencent terms as Hunyuan3D. |
| `scene/`, `gaussian_renderer/`, `arguments/`, `utils/` | 3D Gaussian Splatting / 2D Gaussian Splatting | https://github.com/graphdeco-inria/gaussian-splatting, https://github.com/hbb1/2d-gaussian-splatting | Non-commercial research license (Inria GRAPHDECO) | Original license headers are preserved in the copied files (see `scene/__init__.py`). |

## Runtime Dependencies (Not Bundled)

These are installed by the user, not shipped in this repository:

| Component | Source | Notes |
|-----------|--------|-------|
| `diff_surfel_rasterization` | https://github.com/hbb1/diff-surfel-rasterization | 2D Gaussian Splatting rasterizer. |
| `simple_knn` | https://github.com/graphdeco-inria/gaussian-splatting (submodule) | KNN helper from 3DGS. |
| CAP-UDF | https://github.com/junshengzhou/CAP-UDF | Required at runtime via `CAPUDF_ROOT`. |
| ControlNet depth checkpoint | Released with ControlNet (`control_sd15_depth.pth`) | Required at runtime via `CONTROLNET_DEPTH_CKPT`. |
| Hunyuan3D-2 model weights | https://huggingface.co/tencent/Hunyuan3D-2 | Required at runtime via `HUNYUAN3D_MODEL_PATH`. |

## What Is Intentionally Excluded From The Release

To keep the source tree small and avoid redistributing third-party weights, the
repository does **not** ship:

- Large model checkpoints (ControlNet, Hunyuan3D, Stable Diffusion, etc.).
- Generated pipeline outputs (`outputs/`).
- Compiled CUDA artifacts (`*.so`, `*.o`, `build/`).
- Local conda environments and `.cache` directories.
