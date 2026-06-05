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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Task1 smoke-test run directory.")
    return parser.parse_args()


def main() -> None:
    """Check required smoke-test outputs and summary fields."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task1 output files: {missing}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ValueError(f"Unexpected summary status: {summary.get('status')}")
    if int(summary.get("image_count", 0)) != 9:
        raise ValueError("Task1 smoke test must validate exactly 9 images.")
    print(json.dumps({"status": "PASS", "run_dir": run_dir.as_posix()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
