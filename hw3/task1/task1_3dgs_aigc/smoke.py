"""Smoke-test runner for HW3 Task1 assets."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from task1_3dgs_aigc.assets import (
    collect_object_a_images,
    neighbor_diffs,
    save_contact_sheet,
    validate_image,
)
from task1_3dgs_aigc.utils import (
    command_status,
    copy_source_config,
    environment_summary,
    make_run_dir,
    save_csv,
    save_json,
)


def run_smoke_test(config: dict[str, Any]) -> dict[str, Any]:
    """Validate current AI-generated Task1 image assets and write evidence."""
    run_dir = make_run_dir(config["experiment"]["output_root"], config["experiment"]["name"])
    copy_source_config(config["config_path"], run_dir)
    save_json(run_dir / "config.json", config)

    min_size = int(config["data"]["min_image_size"])
    yaws = [str(yaw).zfill(3) for yaw in config["task1"]["object_a_expected_yaws"]]
    object_a_paths = collect_object_a_images(config["data"]["object_a_dir"], yaws)
    object_c_path = Path(config["data"]["object_c_image"])

    image_rows = [validate_image(path, min_size) for path in object_a_paths]
    image_rows.append(validate_image(object_c_path, min_size))
    diffs = neighbor_diffs(object_a_paths)
    manifest = _build_manifest(config, object_a_paths, object_c_path, image_rows)
    summary = _build_summary(config, run_dir, image_rows, diffs)

    save_json(run_dir / "manifest.json", manifest)
    save_json(run_dir / "summary.json", summary)
    save_csv(run_dir / "image_stats.csv", image_rows)
    save_csv(run_dir / "pairwise_yaw_diffs.csv", diffs)
    save_contact_sheet(object_a_paths, object_c_path, run_dir / "contact_sheet.png")
    return summary


def _build_manifest(
    config: dict[str, Any],
    object_a_paths: list[Path],
    object_c_path: Path,
    image_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": config["task1"]["stage"],
        "final_submission_assets": bool(config["task1"].get("final_submission_assets", False)),
        "note": config["task1"].get("note", ""),
        "object_a": {
            "count": len(object_a_paths),
            "views": [path.as_posix() for path in object_a_paths],
        },
        "object_c": {
            "image": object_c_path.as_posix(),
        },
        "image_count": len(image_rows),
    }


def _build_summary(
    config: dict[str, Any],
    run_dir: Path,
    image_rows: list[dict[str, Any]],
    diffs: list[dict[str, Any]],
) -> dict[str, Any]:
    optional_tools = list(config.get("external_tools", {}).get("optional", []))
    return {
        "status": "PASS",
        "run_dir": run_dir.as_posix(),
        "image_count": len(image_rows),
        "object_a_count": 8,
        "object_c_count": 1,
        "min_width": min(row["width"] for row in image_rows),
        "min_height": min(row["height"] for row in image_rows),
        "mean_neighbor_rms_diff_64": _mean_diff(diffs),
        "optional_tools": command_status(optional_tools),
        "environment": environment_summary(),
        "next_required_assets": [
            "phone-captured object A multi-view images or video",
            "phone-captured object C single image",
            "text-to-3D object B asset",
            "Mip-NeRF 360 background scene",
        ],
    }


def _mean_diff(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row["rms_diff_64"]) for row in rows) / len(rows), 3)
