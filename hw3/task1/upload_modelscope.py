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
EXCLUDED_UPLOAD_DIRS = {
    "colmap",
    "eval",
    "preprocessed",
    "processed",
    "renders",
    "scripts",
    "swanlab",
}


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
    weights = {
        path
        for path in run_dir.rglob("*")
        if path.is_file() and _is_gaussian_weight(path, run_dir)
    }
    weights.update(run_dir.glob("nerfstudio/**/nerfstudio_models/*.ckpt"))
    for root_name in ["object_b_threestudio", "object_c_zero123"]:
        latest = _latest_checkpoint_dir(run_dir / root_name)
        if latest and (latest / "last.ckpt").is_file():
            weights.add(latest / "last.ckpt")
    weights.update(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and _is_regular_weight(path, run_dir)
    )
    return sorted(weights)


def _is_weight_file(path: Path, run_dir: Path) -> bool:
    return _is_gaussian_weight(path, run_dir) or _is_regular_weight(path, run_dir)


def _is_gaussian_weight(path: Path, run_dir: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix != ".ply" or path.name not in GAUSSIAN_WEIGHT_FILENAMES:
        return False
    rel_parts = set(path.relative_to(run_dir).parts[:-1])
    return bool(rel_parts & GAUSSIAN_WEIGHT_DIR_HINTS)


def _is_regular_weight(path: Path, run_dir: Path) -> bool:
    suffix = path.suffix.lower()
    rel_parts = set(path.relative_to(run_dir).parts[:-1])
    if rel_parts & EXCLUDED_UPLOAD_DIRS:
        return False
    if "ckpts" in rel_parts:
        return False
    if "nerfstudio_models" in rel_parts and suffix == ".ckpt":
        return True
    return suffix in {".pt", ".pth", ".safetensors", ".onnx", ".npz"}


def _latest_checkpoint_dir(root: Path) -> Path | None:
    checkpoint_dirs = [path for path in root.glob("**/ckpts") if path.is_dir()]
    if not checkpoint_dirs:
        return None
    return max(checkpoint_dirs, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    main()
