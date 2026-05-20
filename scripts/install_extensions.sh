#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install -e extensions/depthid_render
python -m pip install -e extensions/depthid_render_mask_control_v2
python -m pip install -e extensions/find_max_in_circles
python -m pip install -e Hunyuan3D/hy3dgen/texgen/custom_rasterizer

echo "Local CUDA extensions installed."
echo "Also install diff_surfel_rasterization and simple_knn; see README.md for source links."
