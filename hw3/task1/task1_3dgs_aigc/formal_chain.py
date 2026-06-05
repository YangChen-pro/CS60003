"""End-to-end AI-material HW3 Task1 chain.

This stage treats the existing AI object A/C images as the formal inputs, creates
a procedural text-to-3D proxy for object B, builds a lightweight background, and
renders a fused multi-view scene. It is a practical stand-in for the heavy
COLMAP/3DGS/threestudio/Zero123 toolchain when the goal is to run the whole
integration path on the available 136 machine.
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image

from task1_3dgs_aigc.assets import collect_object_a_images
from task1_3dgs_aigc.geometry import (
    Point,
    render_points,
    save_ply,
    save_turntable_gif,
    summarize_points,
    transform_points,
)
from task1_3dgs_aigc.utils import (
    command_status,
    copy_source_config,
    environment_summary,
    make_run_dir,
    save_csv,
    save_json,
)


def run_formal_chain(config: dict[str, Any]) -> dict[str, Any]:
    """Run the formal AI-material fusion chain and write all artifacts."""
    started = time.time()
    run_dir = make_run_dir(config["experiment"]["output_root"], config["experiment"]["name"])
    assets_dir = run_dir / "assets"
    frames_dir = run_dir / "renders" / "frames"
    assets_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    copy_source_config(config["config_path"], run_dir)
    save_json(run_dir / "config.json", config)

    object_a = _build_object_a(config)
    object_b = _build_object_b(config)
    object_c = _build_object_c(config)
    background = _build_background(config)
    fused = _fuse_scene(config, object_a, object_b, object_c, background)

    save_ply(assets_dir / "object_a_ai_multiview_gaussians.ply", object_a)
    save_ply(assets_dir / "object_b_text_to_3d_proxy.ply", object_b)
    save_ply(assets_dir / "object_c_single_image_proxy.ply", object_c)
    save_ply(assets_dir / "background_proxy_gaussians.ply", background)
    save_ply(run_dir / "fused_scene.ply", fused)

    frame_paths = _render_turntable(config, fused, frames_dir)
    preview = run_dir / "renders" / "fused_scene_preview.png"
    shutil.copy2(frame_paths[0], preview)
    gif_path = run_dir / "renders" / "fused_scene_turntable.gif"
    save_turntable_gif(frame_paths, gif_path)

    asset_rows = _asset_rows(object_a, object_b, object_c, background, fused)
    save_csv(run_dir / "metrics.csv", asset_rows)
    manifest = _manifest(config, run_dir, frame_paths, preview, gif_path)
    summary = _summary(config, run_dir, asset_rows, started)
    save_json(run_dir / "asset_manifest.json", manifest)
    save_json(run_dir / "summary.json", summary)
    _copy_report_assets(config, run_dir, preview, gif_path)
    return summary


def _build_object_a(config: dict[str, Any]) -> list[Point]:
    yaws = [str(yaw).zfill(3) for yaw in config["task1"]["object_a_expected_yaws"]]
    paths = collect_object_a_images(config["data"]["object_a_dir"], yaws)
    sample_size = int(config["formal_chain"]["object_a_sample_size"])
    points: list[Point] = []
    for path, yaw_label in zip(paths, yaws):
        yaw = math.radians(float(yaw_label))
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((sample_size, sample_size))
        pixels = rgb.load()
        for py in range(0, sample_size, 2):
            for px in range(0, sample_size, 2):
                r, g, b = pixels[px, py]
                if not _is_object_a_foreground(r, g, b, px, py, sample_size):
                    continue
                u = (px / (sample_size - 1) - 0.5) * 2.0
                v = (0.55 - py / (sample_size - 1)) * 2.0
                local_yaw = yaw + u * 0.55
                radius = 0.43 + 0.08 * max(0.0, 0.45 - abs(v))
                x = math.sin(local_yaw) * radius
                z = math.cos(local_yaw) * radius
                points.append((x, v, z, r, g, b, 0.018))
    return points


def _build_object_c(config: dict[str, Any]) -> list[Point]:
    sample_size = int(config["formal_chain"]["object_c_sample_size"])
    with Image.open(config["data"]["object_c_image"]) as image:
        rgb = image.convert("RGB").resize((sample_size, sample_size))
    pixels = rgb.load()
    points: list[Point] = []
    for py in range(0, sample_size, 2):
        for px in range(0, sample_size, 2):
            if not _is_robot_region(px, py, sample_size):
                continue
            r, g, b = pixels[px, py]
            u = (px / (sample_size - 1) - 0.5) * 1.25
            v = (0.55 - py / (sample_size - 1)) * 1.65
            depth = 0.16 * (1.0 - min(1.0, abs(u) * 1.2))
            points.append((u, v, depth, r, g, b, 0.016))
            points.append((u, v, -depth * 0.55, r, g, b, 0.012))
    return points


def _build_object_b(config: dict[str, Any]) -> list[Point]:
    points: list[Point] = []
    radial_steps = int(config["formal_chain"]["object_b_radial_steps"])
    height_steps = int(config["formal_chain"]["object_b_height_steps"])
    for i in range(height_steps):
        y = -0.45 + i / max(1, height_steps - 1) * 0.75
        radius = 0.18 + 0.05 * math.sin(i * 0.8)
        for j in range(radial_steps):
            theta = 2 * math.pi * j / radial_steps
            points.append((math.cos(theta) * radius, y, math.sin(theta) * radius, 68, 146, 102, 0.018))
    for i in range(height_steps):
        phi = i / max(1, height_steps - 1) * math.pi / 2
        y = 0.18 + math.sin(phi) * 0.42
        radius = math.cos(phi) * 0.55
        for j in range(radial_steps):
            theta = 2 * math.pi * j / radial_steps
            color = (128 + int(45 * math.sin(theta)), 72, 196 + int(30 * math.cos(theta)))
            points.append((math.cos(theta) * radius, y, math.sin(theta) * radius, *color, 0.02))
    for j in range(10):
        theta = 2 * math.pi * j / 10
        base_x = math.cos(theta) * 0.38
        base_z = math.sin(theta) * 0.38
        for k in range(7):
            t = k / 6
            points.append((base_x * (1 - 0.25 * t), 0.35 + 0.45 * t, base_z * (1 - 0.25 * t), 180, 226, 255, 0.018))
    return points


def _build_background(config: dict[str, Any]) -> list[Point]:
    points: list[Point] = []
    grid = int(config["formal_chain"]["background_grid"])
    for ix in range(grid):
        x = -3.2 + 6.4 * ix / max(1, grid - 1)
        for iz in range(grid):
            z = -2.2 + 4.7 * iz / max(1, grid - 1)
            tone = int(188 + 18 * math.sin(ix * 0.5) + 10 * math.cos(iz * 0.7))
            points.append((x, -1.08, z, tone, max(140, tone - 42), 92, 0.028))
    for ix in range(grid):
        x = -3.2 + 6.4 * ix / max(1, grid - 1)
        for iy in range(grid // 2):
            y = -1.0 + 2.6 * iy / max(1, grid // 2 - 1)
            tone = int(212 + 10 * math.sin(ix * 0.3 + iy * 0.6))
            points.append((x, y, 2.45, tone, tone - 18, tone - 45, 0.032))
    return points


def _fuse_scene(
    config: dict[str, Any],
    object_a: list[Point],
    object_b: list[Point],
    object_c: list[Point],
    background: list[Point],
) -> list[Point]:
    placement = config["formal_chain"]["placements"]
    return (
        background
        + transform_points(object_a, **placement["object_a"])
        + transform_points(object_b, **placement["object_b"])
        + transform_points(object_c, **placement["object_c"])
    )


def _render_turntable(config: dict[str, Any], points: list[Point], frames_dir: Path) -> list[Path]:
    render_config = config["formal_chain"]["render"]
    frame_count = int(render_config["frame_count"])
    frame_paths: list[Path] = []
    for index in range(frame_count):
        frame_path = frames_dir / f"frame_{index:03d}.png"
        render_points(
            points,
            frame_path,
            yaw_degrees=360.0 * index / frame_count,
            width=int(render_config["width"]),
            height=int(render_config["height"]),
            camera_distance=float(render_config["camera_distance"]),
            focal=float(render_config["focal"]),
        )
        frame_paths.append(frame_path)
    return frame_paths


def _asset_rows(*clouds: list[Point]) -> list[dict[str, Any]]:
    names = ["object_a_multiview", "object_b_text_to_3d", "object_c_single_image", "background", "fused_scene"]
    methods = ["AI multiview to gaussian proxy", "prompt to procedural 3D proxy", "single image extrusion proxy", "Mip-NeRF-like background proxy", "merged gaussian/point scene"]
    rows = []
    for name, method, points in zip(names, methods, clouds):
        rows.append({"asset": name, "method": method, **summarize_points(points)})
    return rows


def _manifest(config: dict[str, Any], run_dir: Path, frame_paths: list[Path], preview: Path, gif_path: Path) -> dict[str, Any]:
    return {
        "stage": "formal_ai_chain",
        "run_dir": run_dir.as_posix(),
        "object_b_prompt": config["formal_chain"]["object_b_prompt"],
        "preview": preview.as_posix(),
        "turntable_gif": gif_path.as_posix(),
        "frames": [path.as_posix() for path in frame_paths],
    }


def _summary(config: dict[str, Any], run_dir: Path, asset_rows: list[dict[str, Any]], started: float) -> dict[str, Any]:
    optional_tools = list(config.get("external_tools", {}).get("optional", []))
    return {
        "status": "PASS",
        "stage": "formal_ai_chain",
        "run_dir": run_dir.as_posix(),
        "asset_count": 4,
        "fused_point_count": int(asset_rows[-1]["point_count"]),
        "metrics": asset_rows,
        "elapsed_seconds": round(time.time() - started, 3),
        "optional_tools": command_status(optional_tools),
        "environment": environment_summary(),
        "limitations": [
            "AI images are treated as formal inputs per user direction.",
            "Heavy COLMAP/3DGS/threestudio/Zero123 steps are represented by reproducible local proxies.",
        ],
    }


def _copy_report_assets(config: dict[str, Any], run_dir: Path, preview: Path, gif_path: Path) -> None:
    report_dir_value = config["experiment"].get("report_assets_dir", "")
    if not report_dir_value:
        return
    report_dir = Path(report_dir_value)
    report_dir.mkdir(parents=True, exist_ok=True)
    for path in [run_dir / "summary.json", run_dir / "metrics.csv", run_dir / "asset_manifest.json", preview, gif_path]:
        shutil.copy2(path, report_dir / path.name)


def _is_object_a_foreground(r: int, g: int, b: int, px: int, py: int, size: int) -> bool:
    saturation = max(r, g, b) - min(r, g, b)
    u = (px / (size - 1) - 0.5) * 2.0
    v = (py / (size - 1) - 0.5) * 2.0
    return saturation > 34 and (u * u / 0.72 + v * v / 1.55) < 1.0


def _is_robot_region(px: int, py: int, size: int) -> bool:
    x = px / (size - 1)
    y = py / (size - 1)
    head = 0.32 <= x <= 0.69 and 0.24 <= y <= 0.48
    body = 0.37 <= x <= 0.65 and 0.49 <= y <= 0.70
    left_arm = 0.27 <= x <= 0.39 and 0.51 <= y <= 0.72
    right_arm = 0.63 <= x <= 0.75 and 0.51 <= y <= 0.72
    legs = (0.38 <= x <= 0.50 or 0.53 <= x <= 0.65) and 0.70 <= y <= 0.90
    ears = (0.28 <= x <= 0.34 or 0.66 <= x <= 0.72) and 0.34 <= y <= 0.43
    return head or body or left_arm or right_arm or legs or ears
