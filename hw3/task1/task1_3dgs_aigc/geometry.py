"""Lightweight 3D point/mesh helpers for HW3 Task1.

The formal HW3 chain writes PLY point clouds and CPU-rendered turntable frames.
These helpers intentionally avoid heavyweight dependencies so the same code can
run on the local Mac and on the 136 `qwen14b` environment.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

Point = tuple[float, float, float, int, int, int, float]


def save_ply(path: str | Path, points: Iterable[Point]) -> None:
    """Save colored points with a scalar radius as an ASCII PLY file."""
    rows = list(points)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(rows)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float radius",
        "end_header",
    ]
    body = [
        f"{x:.5f} {y:.5f} {z:.5f} {r:d} {g:d} {b:d} {radius:.5f}"
        for x, y, z, r, g, b, radius in rows
    ]
    Path(path).write_text("\n".join(header + body) + "\n", encoding="utf-8")


def transform_points(
    points: Iterable[Point],
    *,
    scale: float = 1.0,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_degrees: float = 0.0,
) -> list[Point]:
    """Apply uniform scale, Y-axis rotation, and translation."""
    yaw = math.radians(yaw_degrees)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    tx, ty, tz = translate
    transformed: list[Point] = []
    for x, y, z, r, g, b, radius in points:
        sx, sy, sz = x * scale, y * scale, z * scale
        rx = cos_yaw * sx + sin_yaw * sz
        rz = -sin_yaw * sx + cos_yaw * sz
        transformed.append((rx + tx, sy + ty, rz + tz, r, g, b, radius * scale))
    return transformed


def summarize_points(points: list[Point]) -> dict[str, float | int]:
    """Return count and axis-aligned bounds for a point cloud."""
    if not points:
        return {"point_count": 0}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    return {
        "point_count": len(points),
        "x_min": round(min(xs), 4),
        "x_max": round(max(xs), 4),
        "y_min": round(min(ys), 4),
        "y_max": round(max(ys), 4),
        "z_min": round(min(zs), 4),
        "z_max": round(max(zs), 4),
    }


def render_points(
    points: list[Point],
    output_path: str | Path,
    *,
    yaw_degrees: float,
    width: int = 960,
    height: int = 640,
    camera_distance: float = 5.0,
    focal: float = 420.0,
) -> None:
    """Render a simple perspective point-splat view."""
    image = Image.new("RGB", (width, height), (232, 222, 205))
    draw = ImageDraw.Draw(image)
    projected = []
    yaw = math.radians(yaw_degrees)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    for x, y, z, r, g, b, radius in points:
        cam_x = cos_yaw * x - sin_yaw * z
        cam_z = sin_yaw * x + cos_yaw * z + camera_distance
        if cam_z <= 0.1:
            continue
        screen_x = int(width / 2 + focal * cam_x / cam_z)
        screen_y = int(height / 2 - focal * (y - 0.08) / cam_z)
        splat = max(1, int(focal * max(radius, 0.004) / cam_z))
        projected.append((cam_z, screen_x, screen_y, splat, r, g, b))

    for _, screen_x, screen_y, splat, r, g, b in sorted(projected, reverse=True):
        draw.ellipse(
            (screen_x - splat, screen_y - splat, screen_x + splat, screen_y + splat),
            fill=(r, g, b),
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_turntable_gif(frame_paths: list[Path], output_path: str | Path, duration_ms: int = 120) -> None:
    """Save rendered frames as a looping GIF."""
    frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    if not frames:
        raise ValueError("No frames were provided for GIF export.")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
