import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def add_project_root(anchor_file: str) -> Path:
    root = Path(anchor_file).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def add_capudf_args(parser):
    parser.add_argument(
        "--capudf_root",
        type=str,
        default=os.environ.get("CAPUDF_ROOT"),
        help="Path to a CAP-UDF checkout. Can also be set with CAPUDF_ROOT.",
    )
    parser.add_argument(
        "--capudf_python",
        type=str,
        default=os.environ.get("CAPUDF_PYTHON", sys.executable),
        help="Python executable used to run CAP-UDF when the checkpoint is missing.",
    )
    parser.add_argument(
        "--skip_capudf_run",
        action="store_true",
        help="Require the CAP-UDF checkpoint to already exist instead of launching CAP-UDF.",
    )


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GaussianGrow requires a CUDA GPU. Check your PyTorch CUDA build "
            "and CUDA_VISIBLE_DEVICES before running the pipeline."
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def ensure_capudf_checkpoint(args, name: str, output_dir: str) -> str:
    if not args.capudf_root:
        raise RuntimeError(
            "CAP-UDF checkpoint is required. Set CAPUDF_ROOT or pass --capudf_root; "
            "expected checkpoint: <CAPUDF_ROOT>/outs/<obj_name>/checkpoints/ckpt_060000.pth"
        )

    capudf_root = Path(args.capudf_root).expanduser().resolve()
    ckpt_path = capudf_root / "outs" / name / "checkpoints" / "ckpt_060000.pth"
    if ckpt_path.exists():
        return str(ckpt_path)

    if args.skip_capudf_run:
        raise FileNotFoundError(f"Missing CAP-UDF checkpoint: {ckpt_path}")

    source_ply = Path(output_dir) / "example.ply"
    if not source_ply.exists():
        raise FileNotFoundError(f"Missing point cloud for CAP-UDF: {source_ply}")

    data_dir = capudf_root / "data" / "text2tex" / name
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_ply, data_dir / f"{name}.ply")

    run_py = capudf_root / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"Missing CAP-UDF entry script: {run_py}")

    subprocess.run(
        [args.capudf_python, str(run_py), "--dataname", name, "--dir", name],
        check=True,
        cwd=str(capudf_root),
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CAP-UDF finished but checkpoint was not created: {ckpt_path}")
    return str(ckpt_path)
