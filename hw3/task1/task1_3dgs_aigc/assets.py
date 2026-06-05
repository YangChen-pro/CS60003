"""Image asset validation and smoke-test statistics for HW3 Task1."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageStat

YAW_PATTERN = re.compile(r"yaw_(\d{3})\.png$")


def collect_object_a_images(object_a_dir: str | Path, expected_yaws: list[str]) -> list[Path]:
    """Collect object A views ordered by expected yaw labels."""
    directory = Path(object_a_dir)
    images = {match.group(1): path for path in directory.glob("*.png") if (match := YAW_PATTERN.search(path.name))}
    missing = [yaw for yaw in expected_yaws if yaw not in images]
    if missing:
        raise FileNotFoundError(f"Missing object A yaw views: {missing}")
    return [images[yaw] for yaw in expected_yaws]


def validate_image(path: str | Path, min_size: int) -> dict[str, Any]:
    """Validate one image and return basic statistics."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))
    with Image.open(image_path) as image:
        width, height = image.size
        if min(width, height) < min_size:
            raise ValueError(f"Image is smaller than {min_size}: {image_path} {image.size}")
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        return {
            "path": image_path.as_posix(),
            "file_name": image_path.name,
            "width": width,
            "height": height,
            "mode": image.mode,
            "mean_r": round(stat.mean[0], 3),
            "mean_g": round(stat.mean[1], 3),
            "mean_b": round(stat.mean[2], 3),
            "std_r": round(stat.stddev[0], 3),
            "std_g": round(stat.stddev[1], 3),
            "std_b": round(stat.stddev[2], 3),
            "edge_score": round(edge_score(rgb), 3),
        }


def edge_score(image: Image.Image) -> float:
    """Compute a lightweight edge-detail score from a PIL image."""
    grayscale = image.convert("L").resize((256, 256))
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).stddev[0])


def neighbor_diffs(paths: list[Path]) -> list[dict[str, Any]]:
    """Compute simple RMS differences between neighboring yaw views."""
    rows: list[dict[str, Any]] = []
    if len(paths) < 2:
        return rows
    wrapped = paths + [paths[0]]
    for left, right in zip(wrapped, wrapped[1:]):
        rows.append(
            {
                "left": left.name,
                "right": right.name,
                "rms_diff_64": round(rms_diff(left, right), 3),
            }
        )
    return rows


def rms_diff(left_path: Path, right_path: Path) -> float:
    """Compute RMS difference after resizing images to 64 x 64."""
    with Image.open(left_path) as left, Image.open(right_path) as right:
        left_rgb = left.convert("RGB").resize((64, 64))
        right_rgb = right.convert("RGB").resize((64, 64))
        diff = ImageChops.difference(left_rgb, right_rgb)
        squares = [value * value for value in ImageStat.Stat(diff).mean]
        return math.sqrt(sum(squares) / len(squares))


def save_contact_sheet(paths: list[Path], object_c_path: Path, output_path: str | Path) -> None:
    """Save a compact contact sheet for visual smoke-test evidence."""
    thumbs = [_thumbnail(path) for path in paths]
    thumbs.append(_thumbnail(object_c_path))
    sheet = Image.new("RGB", (3 * 256, 3 * 256), (245, 245, 240))
    for index, image in enumerate(thumbs):
        x = (index % 3) * 256
        y = (index // 3) * 256
        sheet.paste(image, (x, y))
    sheet.save(output_path)


def _thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as image:
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail((244, 244))
        canvas = Image.new("RGB", (256, 256), (245, 245, 240))
        offset = ((256 - thumbnail.width) // 2, (256 - thumbnail.height) // 2)
        canvas.paste(thumbnail, offset)
        return canvas
