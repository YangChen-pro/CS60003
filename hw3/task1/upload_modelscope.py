"""Upload the six reusable Task 1 weights to ModelScope."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MODEL_ID = "youngchen/CS60003"
DEFAULT_REMOTE_ROOT = "hw3/task1"

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Task1 output run directory.")
    parser.add_argument("--remote-subdir", default=None, help="Remote subdir under hw3/task1/.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ModelScope model repo id.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected weights without uploading.")
    parser.add_argument(
        "--replace-remote-subdir",
        action="store_true",
        help="After uploading, remove stale files below the selected remote subdir.",
    )
    return parser.parse_args()


def main() -> None:
    """Upload existing trained model weights to ModelScope."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    weight_files = _find_weight_files(run_dir)
    remote_subdir = _normalize_remote_subdir(args.remote_subdir or run_dir.name)
    if args.dry_run:
        for path in weight_files:
            print(path.relative_to(run_dir).as_posix(), flush=True)
        return

    token = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("MODELSCOPE_API_TOKEN is required for ModelScope upload.")

    try:
        from modelscope_hub import HubApi
    except ImportError as exc:
        try:
            from modelscope.hub.api import HubApi
        except ImportError:
            raise RuntimeError("Install modelscope-hub or modelscope before uploading.") from exc

    api = HubApi()
    api.login(token)
    uploaded_paths: set[str] = set()
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
        uploaded_paths.add(remote_path)
        print(f"uploaded {remote_path}", flush=True)
    if args.replace_remote_subdir:
        deleted = _prune_remote_subdir(
            api,
            model_id=args.model_id,
            remote_subdir=remote_subdir,
            keep_paths=uploaded_paths,
        )
        for path in deleted:
            print(f"deleted stale file {path}", flush=True)


def _find_weight_files(run_dir: Path) -> list[Path]:
    candidates = [
        run_dir / "exports/background/splat/splat.ply",
        run_dir / "exports/object_a/splat/splat.ply",
        _latest_file(
            run_dir / "nerfstudio/background",
            "**/nerfstudio_models/step-000029999.ckpt",
        ),
        _latest_file(
            run_dir / "nerfstudio/object_a",
            "**/nerfstudio_models/step-000029999.ckpt",
        ),
        _latest_file(run_dir / "object_b_threestudio", "**/ckpts/last.ckpt"),
        _latest_file(run_dir / "object_c_zero123", "**/ckpts/last.ckpt"),
    ]
    missing = [str(path) for path in candidates if path is None or not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing Task 1 weight files: {missing}")
    return sorted(path for path in candidates if path is not None)


def _latest_file(root: Path, pattern: str) -> Path | None:
    files = [path for path in root.glob(pattern) if path.is_file()]
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def _normalize_remote_subdir(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid remote subdir: {value}")
    normalized = path.as_posix().strip("/")
    if not normalized or normalized == ".":
        raise ValueError("Remote subdir must not be empty.")
    return normalized


def _prune_remote_subdir(
    api: object,
    *,
    model_id: str,
    remote_subdir: str,
    keep_paths: set[str],
) -> list[str]:
    prefix = f"{DEFAULT_REMOTE_ROOT}/{_normalize_remote_subdir(remote_subdir)}/"
    entries = api.list_repo_files(
        repo_id=model_id,
        repo_type="model",
        revision="master",
        recursive=True,
    )
    paths = sorted(
        path
        for entry in entries
        if (path := _entry_path(entry)).startswith(prefix)
        and path not in keep_paths
        and _entry_type(entry).lower() not in {"tree", "directory", "dir"}
    )
    if not paths:
        return []
    result = api.delete_files(
        model_id,
        "model",
        paths,
        revision="master",
    )
    failed = list(result.get("failed_files", [])) if isinstance(result, dict) else []
    if failed:
        raise RuntimeError(f"ModelScope failed to delete: {failed}")
    return paths


def _entry_path(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("Path") or entry.get("path") or "")
    return str(getattr(entry, "path", ""))


def _entry_type(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("Type") or entry.get("type") or "blob")
    return str(getattr(entry, "type", "blob"))


if __name__ == "__main__":
    main()
