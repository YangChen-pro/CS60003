"""Validate HW3 Task1 smoke-test outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "source_config.yaml",
    "config.json",
    "manifest.json",
    "image_stats.csv",
    "pairwise_yaw_diffs.csv",
    "contact_sheet.png",
    "summary.json",
]

FORMAL_REQUIRED_FILES = [
    "source_config.yaml",
    "config.json",
    "asset_manifest.json",
    "metrics.csv",
    "fused_scene.ply",
    "renders/fused_scene_preview.png",
    "renders/fused_scene_turntable.gif",
    "summary.json",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Task1 smoke-test run directory.")
    return parser.parse_args()


def main() -> None:
    """Check required smoke-test outputs and summary fields."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Task1 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stage = str(summary.get("stage", "smoke_assets"))
    required_files = FORMAL_REQUIRED_FILES if stage == "formal_ai_chain" else REQUIRED_FILES
    missing = [name for name in required_files if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task1 output files: {missing}")
    if summary.get("status") != "PASS":
        raise ValueError(f"Unexpected summary status: {summary.get('status')}")
    if stage == "smoke_assets" and int(summary.get("image_count", 0)) != 9:
        raise ValueError("Task1 smoke test must validate exactly 9 images.")
    if stage == "formal_ai_chain" and int(summary.get("asset_count", 0)) != 4:
        raise ValueError("Task1 formal chain must include object A/B/C and one background.")
    print(json.dumps({"status": "PASS", "run_dir": run_dir.as_posix()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
