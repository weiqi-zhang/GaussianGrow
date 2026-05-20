import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Hunyuan3D.minimal_demo import texture_mesh


def main():
    parser = argparse.ArgumentParser(description="Stage 3: texture the input mesh with Hunyuan3D.")
    parser.add_argument("--save_path", type=str, default="./outputs/test", help="Directory containing the generated main view.")
    parser.add_argument("--mesh_path", type=str, required=True, help="Input mesh path.")
    parser.add_argument("--model_path", type=str, default=None, help="Local Hunyuan3D model path or Hugging Face repo id.")
    parser.add_argument("--image_name", type=str, default="main_view.png", help="Condition image name inside save_path.")
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


if __name__ == "__main__":
    main()
