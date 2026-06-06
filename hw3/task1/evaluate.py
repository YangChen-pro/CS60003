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
    "scripts/04_object_c_zero123.sh",
    "scripts/05_export_geometry.sh",
    "scripts/06_render_blender.sh",
]

STRICT_REQUIRED_FILES = [
    "exports/object_a/splat/splat.ply",
    "exports/background/splat/splat.ply",
    "exports/object_b/mesh/model.obj",
    "exports/object_c/mesh/model.obj",
    "renders/final_3dgs_backplate/fused_scene.mp4",
    "renders/final_3dgs_backplate/fused_scene_manifest.json",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Task1 real-chain run directory.")
    parser.add_argument("--strict-real-outputs", action="store_true", help="Validate trained artifacts and final video.")
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
    if args.strict_real_outputs:
        validate_strict_real_outputs(run_dir)
    print(
        json.dumps(
            {"status": "PASS", "run_dir": run_dir.as_posix(), "strict_real_outputs": args.strict_real_outputs},
            ensure_ascii=False,
        ),
        flush=True,
    )


def validate_strict_real_outputs(run_dir: Path) -> None:
    """Validate trained Task1 artifacts, final render, and SwanLab links."""
    missing = [name for name in STRICT_REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing strict Task1 real outputs: {missing}")
    min_sizes = {
        "exports/object_a/splat/splat.ply": 1024 * 1024,
        "exports/background/splat/splat.ply": 1024 * 1024,
        "exports/object_b/mesh/model.obj": 1024 * 1024,
        "exports/object_c/mesh/model.obj": 1024 * 1024,
        "renders/final_3dgs_backplate/fused_scene.mp4": 1024 * 1024,
        "renders/final_3dgs_backplate/fused_scene_manifest.json": 128,
    }
    small = [name for name, size in min_sizes.items() if (run_dir / name).stat().st_size <= size]
    if small:
        raise ValueError(f"Strict Task1 outputs are unexpectedly small: {small}")
    if not list((run_dir / "object_b_threestudio").glob("**/ckpts/last.ckpt")):
        raise FileNotFoundError("Missing object B SDS checkpoint: object_b_threestudio/**/ckpts/last.ckpt")
    if not list((run_dir / "object_c_zero123").glob("**/ckpts/last.ckpt")):
        raise FileNotFoundError("Missing object C Zero123 checkpoint: object_c_zero123/**/ckpts/last.ckpt")
    manifest = json.loads((run_dir / "renders/final_3dgs_backplate/fused_scene_manifest.json").read_text(encoding="utf-8"))
    expected = {"width": 1920, "height": 1080, "frames": 144, "fps": 24}
    if {key: manifest.get(key) for key in expected} != expected:
        raise ValueError(f"Unexpected final video manifest fields: {manifest}")
    swanlab_runs = Path(__file__).with_name("SWANLAB_RUNS.md").read_text(encoding="utf-8")
    if swanlab_runs.count("https://swanlab.cn/@youngchen/cs60003-hw3-task1/runs/") < 4:
        raise ValueError("SWANLAB_RUNS.md must record four real Task1 SwanLab runs.")


if __name__ == "__main__":
    main()
