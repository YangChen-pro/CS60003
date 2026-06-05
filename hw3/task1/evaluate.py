"""Validate maintained HW3 Task1 real high-quality chain outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REAL_REQUIRED_FILES = [
    "source_config.yaml",
    "config.json",
    "summary.json",
    "scripts/00_check_tools.sh",
    "scripts/01_object_a_splatfacto.sh",
    "scripts/02_background_splatfacto.sh",
    "scripts/03_object_b_threestudio.sh",
    "scripts/04_object_c_triposr.sh",
    "scripts/05_export_geometry.sh",
    "scripts/06_render_blender.sh",
]

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Task1 real-chain run directory.")
    return parser.parse_args()


def main() -> None:
    """Check required real-chain outputs and summary fields."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Task1 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stage = str(summary.get("stage", ""))
    if stage != "real_high_quality":
        raise ValueError(f"Unsupported Task1 stage in maintained evaluator: {stage}")
    missing = [name for name in REAL_REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task1 output files: {missing}")
    if summary.get("status") not in {"PASS", "READY", "NEEDS_INPUTS"}:
        raise ValueError(f"Unexpected summary status: {summary.get('status')}")
    if int(summary.get("script_count", 0)) != 7:
        raise ValueError("Task1 real high-quality chain must generate 7 orchestration scripts.")
    print(json.dumps({"status": "PASS", "run_dir": run_dir.as_posix()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
