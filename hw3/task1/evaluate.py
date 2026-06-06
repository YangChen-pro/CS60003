"""Validate maintained HW3 Task1 real high-quality chain outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


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
    "renders/fused_splats/fused_scene.mp4",
    "renders/fused_splats/fused_scene_manifest.json",
]

STRICT_ASSET_NAMES = {"background", "object_a", "object_b", "object_c"}
REQUIRED_RENDERER = "gsplat fused splat renderer"
REQUIRED_PIPELINE_MODE = "fused_splats"
BANNED_COMPOSITION_SOURCES = [
    "object_a_cutouts",
    "object_b_test_renders",
    "object_c_test_renders",
    "composite_sprite",
    "test_renders",
    "panel",
]


def _is_under_root(child: Path, root: Path) -> bool:
    try:
        return child.resolve().is_relative_to(root.resolve())
    except AttributeError:
        child_resolved = child.resolve()
        root_resolved = root.resolve()
        return child_resolved == root_resolved or root_resolved in child_resolved.parents


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
    allowed_status = {"PASS"} if args.strict_real_outputs else {"PASS", "READY", "NEEDS_INPUTS"}
    if summary.get("status") not in allowed_status:
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
        "renders/fused_splats/fused_scene.mp4": 1024 * 1024,
        "renders/fused_splats/fused_scene_manifest.json": 128,
    }
    small = [name for name, size in min_sizes.items() if (run_dir / name).stat().st_size <= size]
    if small:
        raise ValueError(f"Strict Task1 outputs are unexpectedly small: {small}")
    if not list((run_dir / "object_b_threestudio").glob("**/ckpts/last.ckpt")):
        raise FileNotFoundError("Missing object B SDS checkpoint: object_b_threestudio/**/ckpts/last.ckpt")
    if not list((run_dir / "object_c_zero123").glob("**/ckpts/last.ckpt")):
        raise FileNotFoundError("Missing object C Zero123 checkpoint: object_c_zero123/**/ckpts/last.ckpt")
    manifest = json.loads((run_dir / "renders/fused_splats/fused_scene_manifest.json").read_text(encoding="utf-8"))
    expected = {"width": 1920, "height": 1080, "frames": 144, "fps": 24}
    if {key: manifest.get(key) for key in expected} != expected:
        raise ValueError(f"Unexpected final video manifest fields: {manifest}")
    if manifest.get("renderer") != REQUIRED_RENDERER:
        raise ValueError(f"Strict final render must use renderer '{REQUIRED_RENDERER}', got: {manifest.get('renderer')}")
    if manifest.get("pipeline_mode") != REQUIRED_PIPELINE_MODE:
        raise ValueError(
            f"Strict final render must declare pipeline mode '{REQUIRED_PIPELINE_MODE}', got: {manifest.get('pipeline_mode')}"
        )
    validate_unified_3d_manifest(run_dir, manifest)
    validate_camera_payload(manifest)
    swanlab_runs = Path(__file__).with_name("SWANLAB_RUNS.md").read_text(encoding="utf-8")
    if swanlab_runs.count("https://swanlab.cn/@youngchen/cs60003-hw3-task1/runs/") < 4:
        raise ValueError("SWANLAB_RUNS.md must record four real Task1 SwanLab runs.")


def validate_unified_3d_manifest(run_dir: Path, manifest: dict) -> None:
    """Ensure strict final video comes from one unified 3D renderer."""
    source_mode = str(manifest.get("source_mode", "")).lower()
    if source_mode != "unified_3d_assets":
        raise ValueError(f"Strict manifest source_mode must be 'unified_3d_assets', got: {manifest.get('source_mode')}")
    source_text = json.dumps(manifest.get("assets", []), ensure_ascii=False)
    if any(name in source_text.lower() for name in BANNED_COMPOSITION_SOURCES):
        raise ValueError("Strict final render cannot use cutout/test composite source modes.")
    assets = manifest.get("assets", [])
    names = {asset.get("name") for asset in assets if isinstance(asset, dict)}
    if names != STRICT_ASSET_NAMES or len(assets) != len(STRICT_ASSET_NAMES):
        raise ValueError(f"Strict final render assets mismatch: {names}")
    if len(names) != len(assets):
        raise ValueError(f"Strict final render assets must be unique; got duplicate names in {names}")
    missing_sources = []
    for asset in assets:
        source = asset.get("source", "") if isinstance(asset, dict) else ""
        if not source:
            missing_sources.append(str(asset))
            continue
        path = Path(source)
        candidates = [path]
        if not path.is_absolute():
            run_dir_candidate = run_dir / source
            if run_dir_candidate.resolve() != path.resolve():
                candidates.append(run_dir_candidate)
        existing = [candidate for candidate in candidates if candidate.exists()]
        if not existing:
            missing_sources.append(source)
        if not any(_is_under_root(candidate.resolve(), run_dir) for candidate in existing):
            missing_sources.append(f"{source}(source outside run_dir)")
        source_lower = source.lower()
        if any(name in source_lower for name in BANNED_COMPOSITION_SOURCES):
            missing_sources.append(f"{source}(forbidden source path)")
        # Optional hardening: ensure each source points to expected real asset classes for strict 3D.
        if source_lower.startswith("object_a") and "object_a" not in str(asset.get("name", "")):
            missing_sources.append(f"{source}(asset-name mismatch)")
        if source_lower.startswith("object_b") and "object_b" not in str(asset.get("name", "")):
            missing_sources.append(f"{source}(asset-name mismatch)")
        if source_lower.startswith("object_c") and "object_c" not in str(asset.get("name", "")):
            missing_sources.append(f"{source}(asset-name mismatch)")
        if source_lower.startswith("background") and asset.get("name") != "background":
            missing_sources.append(f"{source}(asset-name mismatch)")
        if source_lower and asset.get("name") == "object_a" and "object_a" not in source_lower and "splat.ply" not in source_lower:
            missing_sources.append(f"{source}(object_a source should be object_a splat)")
        if asset.get("name") == "background" and "background" not in source_lower and "splat.ply" not in source_lower:
            missing_sources.append(f"{source}(background source should be background splat)")
        if asset.get("name") == "object_b" and "obj" not in source_lower and "mesh" not in source_lower:
            missing_sources.append(f"{source}(object_b source should be mesh obj)")
        if asset.get("name") == "object_c" and "obj" not in source_lower and "mesh" not in source_lower:
            missing_sources.append(f"{source}(object_c source should be mesh obj)")
    if missing_sources:
        raise FileNotFoundError(f"Strict final render source files missing: {missing_sources}")


def validate_camera_payload(manifest: dict) -> None:
    """Validate camera metadata for strict outputs."""
    camera = manifest.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("Strict final render must include camera metadata in manifest.")
    mode = str(camera.get("name") or camera.get("mode") or "").lower()
    if mode not in {"orbit", "stabilized_orbit", "background_trajectory"}:
        raise ValueError(f"Strict final render must use a valid camera mode, got: {camera.get('name')}")
    if mode == "background_trajectory":
        if not (camera.get("centers") and isinstance(camera.get("centers"), list)):
            raise ValueError("Background trajectory mode must record camera centers.")
    required = {"center", "radius", "focal_scale"}
    missing = [field for field in required if field not in camera]
    if missing:
        raise ValueError(f"Strict final render camera metadata missing required fields: {missing}")
    focal_scale = float(camera.get("focal_scale", 0.0))
    if focal_scale <= 0 or not np.isfinite(focal_scale):
        raise ValueError(f"Strict final render camera focal_scale must be a positive finite number, got: {camera.get('focal_scale')}")


if __name__ == "__main__":
    main()
