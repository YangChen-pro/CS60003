"""Upload HW3 Task1 run artifacts to ModelScope.

This mirrors the HW2 habit of keeping final checkpoints/artifacts under the
`youngchen/CS60003` ModelScope model repository. The token is read only from
`MODELSCOPE_API_TOKEN`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MODEL_ID = "youngchen/CS60003"
DEFAULT_REMOTE_ROOT = "hw3/task1"

UPLOAD_CANDIDATES = [
    "source_config.yaml",
    "config.json",
    "manifest.json",
    "image_stats.csv",
    "pairwise_yaw_diffs.csv",
    "contact_sheet.png",
    "summary.json",
    "env.json",
    "metrics.json",
    "best.pt",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Task1 output run directory.")
    parser.add_argument("--remote-subdir", default=None, help="Remote subdir under hw3/task1/.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ModelScope model repo id.")
    return parser.parse_args()


def main() -> None:
    """Upload existing run artifacts to ModelScope."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
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
    for name in UPLOAD_CANDIDATES:
        path = run_dir / name
        if not path.is_file():
            continue
        remote_path = f"{DEFAULT_REMOTE_ROOT}/{remote_subdir}/{name}"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote_path,
            repo_id=args.model_id,
            repo_type="model",
            commit_message=f"Upload HW3 Task1 {remote_subdir} {name}",
        )
        print(f"uploaded {remote_path}", flush=True)


if __name__ == "__main__":
    main()
