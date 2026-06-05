"""Upload only trained HW3 Task1 model weights to ModelScope.

ModelScope is reserved for reusable trained weights. Run metadata, configs,
renders, reports, proxy meshes, and debugging artifacts stay in Git or local
outputs and are intentionally rejected by this uploader.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MODEL_ID = "youngchen/CS60003"
DEFAULT_REMOTE_ROOT = "hw3/task1"

WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".npz"}
GAUSSIAN_WEIGHT_FILENAMES = {"point_cloud.ply", "splat.ply", "gaussian_splat.ply"}
GAUSSIAN_WEIGHT_DIR_HINTS = {"exports", "nerfstudio", "splats", "gaussians"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Task1 output run directory.")
    parser.add_argument("--remote-subdir", default=None, help="Remote subdir under hw3/task1/.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ModelScope model repo id.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected weights without uploading.")
    return parser.parse_args()


def main() -> None:
    """Upload existing trained model weights to ModelScope."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    weight_files = _find_weight_files(run_dir)
    if not weight_files:
        raise RuntimeError(
            "No trained model weight files found. ModelScope upload is skipped by policy."
        )
    if args.dry_run:
        for path in weight_files:
            print(path.relative_to(run_dir).as_posix(), flush=True)
        return

    token = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("MODELSCOPE_API_TOKEN is required for ModelScope upload.")

    try:
        from modelscope.hub.api import HubApi
    except ImportError as exc:
        raise RuntimeError("Install modelscope before uploading Task1 artifacts.") from exc

    api = HubApi()
    api.login(token)
    remote_subdir = args.remote_subdir or run_dir.name
    for path in weight_files:
        rel_path = path.relative_to(run_dir).as_posix()
        remote_path = f"{DEFAULT_REMOTE_ROOT}/{remote_subdir}/{rel_path}"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote_path,
            repo_id=args.model_id,
            repo_type="model",
            commit_message=f"Upload HW3 Task1 trained weight {rel_path}",
        )
        print(f"uploaded {remote_path}", flush=True)


def _find_weight_files(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.rglob("*") if path.is_file() and _is_weight_file(path, run_dir))


def _is_weight_file(path: Path, run_dir: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in WEIGHT_SUFFIXES:
        return True
    if suffix != ".ply" or path.name not in GAUSSIAN_WEIGHT_FILENAMES:
        return False
    rel_parts = set(path.relative_to(run_dir).parts[:-1])
    return bool(rel_parts & GAUSSIAN_WEIGHT_DIR_HINTS)


if __name__ == "__main__":
    main()
