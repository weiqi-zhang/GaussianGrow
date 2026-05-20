#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_DIR="${INPUT_DIR:-$ROOT/examples/fire_dragon}"
OBJ_NAME="${OBJ_NAME:-dragon_normalized}"
OBJ_FILE="${OBJ_FILE:-dragon_normalized.obj}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/fire_dragon}"
PROMPT="${PROMPT:-A cartoon-style small green dragon with a pure and vibrant green body and big expressive eyes.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-malformed, extra limbs, poorly drawn anatomy, badly drawn, extra legs, low resolution, blurry, watermark, text, censored, deformed, bad anatomy}"
DEVICE="${DEVICE:-2080}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

if [[ -z "${CAPUDF_ROOT:-}" ]]; then
  echo "CAPUDF_ROOT is required. Set it to your CAP-UDF checkout path." >&2
  exit 2
fi

COMMON_ARGS=(
  --input_dir "$INPUT_DIR"
  --output_dir "$OUTPUT_DIR"
  --obj_name "$OBJ_NAME"
  --obj_file "$OBJ_FILE"
  --n_prompt "$NEGATIVE_PROMPT"
  --ddim_steps 50
  --new_strength 1
  --new_second_strength 1
  --update_strength 0.6
  --update_second_strength 0.3
  --view_threshold 0.1
  --blend 0
  --dist 1
  --viewpoint_mode predefined
  --seed 47
  --device "$DEVICE"
  --use_objaverse
  --capudf_root "$CAPUDF_ROOT"
)

if [[ -n "${CAPUDF_PYTHON:-}" ]]; then
  COMMON_ARGS+=(--capudf_python "$CAPUDF_PYTHON")
fi

HUNYUAN_ARGS=()
if [[ -n "${HUNYUAN3D_MODEL_PATH:-}" ]]; then
  HUNYUAN_ARGS+=(--model_path "$HUNYUAN3D_MODEL_PATH")
fi

DELIGHT_ARGS=()
if [[ -n "${HUNYUAN3D_DELIGHT_MODEL_PATH:-}" ]]; then
  DELIGHT_ARGS+=(--delight_model_path "$HUNYUAN3D_DELIGHT_MODEL_PATH")
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python scripts/check_setup.py
fi

python scripts/stage1_initialize_gaussians.py "${COMMON_ARGS[@]}" --num_viewpoints 13
python scripts/stage2_generate_main_view.py "${COMMON_ARGS[@]}" --num_viewpoints 13 --prompt "$PROMPT"
python scripts/stage3_texture_mesh.py --save_path "$OUTPUT_DIR" --mesh_path "$INPUT_DIR/$OBJ_FILE" "${HUNYUAN_ARGS[@]}"
python scripts/stage4_refine_gaussians.py "${COMMON_ARGS[@]}" --num_viewpoints 40 --prompt "$PROMPT" "${DELIGHT_ARGS[@]}"

echo "Pipeline finished. Outputs are in: $OUTPUT_DIR"
