"""Generate small synthetic HW3 image assets for early pipeline tests.

The generated files are placeholders for smoke tests only. They are not meant to
replace the final phone captures required by the assignment.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter


CANVAS = 1024
SCALE = 265
CENTER = (CANVAS // 2, 580)
OUT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "synthetic_test"


Color = tuple[int, int, int]
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class Sprite3D:
    point: Point3
    radius: int
    color: Color
    depth_bias: float = 0.0


def blend(color: Color, factor: float) -> Color:
    return tuple(max(0, min(255, round(channel * factor))) for channel in color)


def project(point: Point3, yaw_deg: float) -> tuple[float, float, float]:
    x, y, z = point
    yaw = math.radians(yaw_deg)
    xr = x * math.cos(yaw) + z * math.sin(yaw)
    zr = -x * math.sin(yaw) + z * math.cos(yaw)
    sx = CENTER[0] + SCALE * xr
    sy = CENTER[1] - SCALE * y
    return sx, sy, zr


def textured_background(seed: int) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGB", (CANVAS, CANVAS), (232, 230, 222))
    pixels = image.load()
    for y in range(CANVAS):
        for x in range(CANVAS):
            base = 226 + int(10 * y / CANVAS)
            noise = rng.randint(-7, 7)
            pixels[x, y] = (base + noise, base + noise, base - 5 + noise)

    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 720, CANVAS, CANVAS), fill=(210, 207, 196, 210))
    draw.ellipse((255, 690, 769, 835), fill=(198, 196, 188, 230), outline=(160, 158, 150, 180), width=3)
    draw.ellipse((315, 715, 705, 810), fill=(221, 219, 210, 235), outline=(176, 174, 164, 190), width=2)
    for _ in range(120):
        x = rng.randint(70, 954)
        y = rng.randint(70, 690)
        r = rng.choice((1, 1, 2))
        tone = rng.randint(145, 185)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(tone, tone, tone, 45))
    return image.filter(ImageFilter.GaussianBlur(0.25))


def cylinder_points(
    radius: float,
    y_min: float,
    y_max: float,
    color: Color,
    yaw_deg: float,
    theta_steps: int = 112,
    y_steps: int = 64,
) -> list[Sprite3D]:
    sprites: list[Sprite3D] = []
    for yi in range(y_steps):
        y = y_min + (y_max - y_min) * yi / (y_steps - 1)
        for ti in range(theta_steps):
            theta = 2 * math.pi * ti / theta_steps
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            _, _, depth = project((x, y, z), yaw_deg)
            normal_light = 0.72 + 0.28 * max(0.0, math.sin(theta + math.radians(yaw_deg)) * 0.7 + 0.3)
            stripe = 0.92 + 0.06 * math.sin(8 * theta + 5 * y)
            sprites.append(Sprite3D((x, y, z), 4, blend(color, normal_light * stripe), depth))
    return sprites


def cone_points(
    radius: float,
    y_min: float,
    y_max: float,
    color: Color,
    yaw_deg: float,
    theta_steps: int = 112,
    y_steps: int = 36,
) -> list[Sprite3D]:
    sprites: list[Sprite3D] = []
    for yi in range(y_steps):
        alpha = yi / (y_steps - 1)
        y = y_min + (y_max - y_min) * alpha
        current_radius = radius * (1 - alpha)
        for ti in range(theta_steps):
            theta = 2 * math.pi * ti / theta_steps
            x = current_radius * math.cos(theta)
            z = current_radius * math.sin(theta)
            _, _, depth = project((x, y, z), yaw_deg)
            normal_light = 0.78 + 0.24 * max(0.0, math.sin(theta + math.radians(yaw_deg)))
            sprites.append(Sprite3D((x, y, z), 4, blend(color, normal_light), depth))
    return sprites


def marker_on_cylinder(theta_deg: float, y: float, color: Color, radius: int = 13) -> Sprite3D:
    theta = math.radians(theta_deg)
    return Sprite3D((0.423 * math.cos(theta), y, 0.423 * math.sin(theta)), radius, color, 0.08)


def rocket_sprites(yaw_deg: float) -> list[Sprite3D]:
    sprites = []
    sprites.extend(cylinder_points(0.42, -0.82, 0.58, (42, 150, 156), yaw_deg))
    sprites.extend(cone_points(0.42, 0.58, 1.02, (204, 83, 56), yaw_deg))

    # Body rings.
    for y in (-0.52, -0.12, 0.34):
        for ti in range(96):
            theta = 2 * math.pi * ti / 96
            sprites.append(Sprite3D((0.426 * math.cos(theta), y, 0.426 * math.sin(theta)), 5, (238, 218, 122), 0.06))

    # Fixed decorative marks create feature points for early matching experiments.
    marks = [
        marker_on_cylinder(0, 0.18, (238, 238, 224), 22),
        marker_on_cylinder(0, 0.18, (58, 90, 135), 10),
        marker_on_cylinder(45, -0.25, (235, 112, 79), 11),
        marker_on_cylinder(115, 0.05, (248, 242, 202), 10),
        marker_on_cylinder(210, -0.35, (238, 112, 84), 10),
        marker_on_cylinder(285, 0.28, (248, 242, 202), 10),
    ]
    sprites.extend(marks)

    # Three stabilizer fins as dense point sprites.
    for theta_deg, color in ((0, (222, 128, 72)), (125, (212, 116, 68)), (245, (212, 116, 68))):
        theta = math.radians(theta_deg)
        tangent = (-math.sin(theta), 0.0, math.cos(theta))
        radial = (math.cos(theta), 0.0, math.sin(theta))
        for u in range(28):
            for v in range(24 - u // 2):
                height = -0.84 + 0.45 * v / 24
                width = 0.04 + 0.33 * u / 27
                x = radial[0] * (0.42 + width) + tangent[0] * 0.01
                z = radial[2] * (0.42 + width) + tangent[2] * 0.01
                sprites.append(Sprite3D((x, height, z), 5, color, 0.03))

    return sprites


def draw_sprites(base: Image.Image, sprites: Iterable[Sprite3D], yaw_deg: float) -> Image.Image:
    draw = ImageDraw.Draw(base, "RGBA")
    projected = []
    for sprite in sprites:
        sx, sy, depth = project(sprite.point, yaw_deg)
        projected.append((depth + sprite.depth_bias, sx, sy, sprite.radius, sprite.color))

    for _, sx, sy, radius, color in sorted(projected, key=lambda item: item[0]):
        alpha = 238
        draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=(*color, alpha))

    # Soft shadow and high-gloss highlight.
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.ellipse((335, 748, 690, 840), fill=(0, 0, 0, 65))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    base.alpha_composite(shadow)
    draw = ImageDraw.Draw(base, "RGBA")
    draw.arc((398, 165, 604, 745), 205, 255, fill=(255, 255, 255, 75), width=5)
    return base


def render_object_a_views() -> list[Path]:
    out_dir = OUT_ROOT / "object_a_multiview"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, yaw in enumerate(range(0, 360, 45), start=1):
        image = textured_background(60003 + index).convert("RGBA")
        image = draw_sprites(image, rocket_sprites(yaw), yaw)
        path = out_dir / f"object_a_view_{index:02d}_yaw_{yaw:03d}.png"
        image.convert("RGB").save(path, quality=95)
        paths.append(path)
    return paths


def render_object_c_single() -> Path:
    out_dir = OUT_ROOT / "object_c_single"
    out_dir.mkdir(parents=True, exist_ok=True)
    image = textured_background(70003).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")

    # Clean single-view placeholder: a wooden cube robot with simple geometry.
    draw.ellipse((320, 735, 705, 835), fill=(0, 0, 0, 45))
    draw.rounded_rectangle((350, 300, 670, 520), radius=32, fill=(176, 122, 76, 255), outline=(102, 72, 48, 255), width=4)
    draw.rounded_rectangle((400, 520, 620, 705), radius=28, fill=(154, 98, 61, 255), outline=(96, 66, 45, 255), width=4)
    draw.rounded_rectangle((255, 535, 390, 605), radius=22, fill=(166, 108, 68, 255), outline=(96, 66, 45, 255), width=4)
    draw.rounded_rectangle((630, 535, 765, 605), radius=22, fill=(166, 108, 68, 255), outline=(96, 66, 45, 255), width=4)
    draw.rounded_rectangle((425, 700, 488, 785), radius=18, fill=(130, 82, 55, 255), outline=(80, 55, 38, 255), width=3)
    draw.rounded_rectangle((532, 700, 595, 785), radius=18, fill=(130, 82, 55, 255), outline=(80, 55, 38, 255), width=3)
    draw.ellipse((430, 370, 475, 415), fill=(32, 43, 57, 255))
    draw.ellipse((545, 370, 590, 415), fill=(32, 43, 57, 255))
    draw.arc((445, 418, 575, 480), 12, 168, fill=(50, 44, 38, 255), width=7)
    for x in (395, 500, 605):
        draw.line((x, 310, x - 25, 512), fill=(215, 160, 98, 88), width=4)

    path = out_dir / "object_c_single_front.png"
    image.convert("RGB").save(path, quality=95)
    return path


def main() -> None:
    paths = render_object_a_views()
    paths.append(render_object_c_single())
    metadata = {
        "purpose": "Synthetic placeholders for HW3 smoke tests only; replace with phone captures for final submission.",
        "object_a": {
            "description": "8 generated multi-view PNGs of a stylized rocket figurine.",
            "views": [path.relative_to(OUT_ROOT).as_posix() for path in paths[:-1]],
        },
        "object_c": {
            "description": "1 generated clean single-view PNG of a wooden cube robot.",
            "image": paths[-1].relative_to(OUT_ROOT).as_posix(),
        },
        "count": len(paths),
    }
    (OUT_ROOT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generated": [str(path) for path in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
