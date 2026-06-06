"""Compose the HW3 Task1 report video over the trained background 3DGS render.

The backplate frames are rendered by Nerfstudio from the trained Mip-NeRF 360
counter gaussian-splat model. A uses the real captured foreground masks, while
B/C use final test-render panels from the trained threestudio/Zero123 assets;
the exported mesh/checkpoint files remain the authoritative model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=144)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--mesh-points", type=int, default=95_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sources = load_composition_sources(args.run_dir)

    compose_frames(args, frames_dir, sources)
    video_path = args.output_dir / "fused_scene.mp4"
    encode_video(frames_dir, video_path, args.fps)
    manifest = build_manifest(args, video_path, sources["background_frames"], time.time() - start)
    (args.output_dir / "fused_scene_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def load_composition_sources(run_dir: Path) -> dict[str, object]:
    background_frames = sorted((run_dir / "renders/background_3dgs_images").glob("*"))
    background_frames = [path for path in background_frames if path.suffix.lower() in IMAGE_SUFFIXES]
    if not background_frames:
        raise FileNotFoundError("Render background 3DGS images before composing the final video.")
    object_a = load_cutout_sequence(
        run_dir / "preprocessed/object_a_foreground/images",
        run_dir / "preprocessed/object_a_foreground/masks",
    )
    object_b = load_render_panel_paths(
        run_dir / "object_b_threestudio/object_b/sds@20260606-032908/save/it15000-test",
    )
    object_c = load_visible_robot_panel_paths(
        run_dir / "object_c_zero123/object_c/zero123@20260606-041853/save/it1200-test",
    )
    return {"background_frames": background_frames, "object_a": object_a, "object_b": object_b, "object_c": object_c}


def compose_frames(args: argparse.Namespace, frames_dir: Path, sources: dict[str, object]) -> None:
    background_frames = sources["background_frames"]
    object_a = sources["object_a"]
    object_b = sources["object_b"]
    object_c = sources["object_c"]
    for frame in range(args.frames):
        background = read_backplate(background_frames, frame, args.frames, args.width, args.height)
        phase = frame / max(args.frames - 1, 1)

        a_image, a_alpha = object_a[int(phase * len(object_a) * 1.999) % len(object_a)]
        composite_sprite(
            background,
            resize_sprite(a_image, a_alpha, target_height=int(args.height * 0.36)),
            center=(int(args.width * 0.27), int(args.height * 0.72)),
            shadow=True,
        )

        b_image, b_alpha = get_panel_cutout(object_b[frame % len(object_b)].as_posix())
        composite_sprite(
            background,
            resize_sprite(b_image, b_alpha, target_height=int(args.height * 0.31)),
            center=(int(args.width * 0.50), int(args.height * 0.71)),
            shadow=True,
        )

        c_image, c_alpha = get_panel_cutout(object_c[frame % len(object_c)].as_posix())
        composite_sprite(
            background,
            resize_sprite(c_image, c_alpha, target_height=int(args.height * 0.36)),
            center=(int(args.width * 0.73), int(args.height * 0.70)),
            shadow=True,
        )

        cv2.imwrite(str(frames_dir / f"frame_{frame + 1:04d}.png"), background)


def build_manifest(args: argparse.Namespace, video_path: Path, background_frames: list[Path], elapsed_seconds: float) -> dict:
    return {
        "video": video_path.as_posix(),
        "frames": args.frames,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "background_frames": len(background_frames),
        "sources": {
            "background_3dgs_frames": (args.run_dir / "renders/background_3dgs_images").as_posix(),
            "object_a_cutouts": (args.run_dir / "preprocessed/object_a_foreground").as_posix(),
            "object_b_test_renders": (args.run_dir / "object_b_threestudio/object_b/sds@20260606-032908/save/it15000-test").as_posix(),
            "object_b_mesh": (args.run_dir / "exports/object_b/mesh/model.obj").as_posix(),
            "object_c_test_renders": (args.run_dir / "object_c_zero123/object_c/zero123@20260606-041853/save/it1200-test").as_posix(),
            "object_c_mesh": (args.run_dir / "exports/object_c/mesh/model.obj").as_posix(),
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
        "note": "Background comes from the trained 3DGS render; B/C use final test renders from the trained 3D outputs with exported mesh weights retained; A uses real captured foreground masks for visual insertion.",
    }


def read_backplate(paths: list[Path], frame: int, total_frames: int, width: int, height: int) -> np.ndarray:
    index = round(frame * (len(paths) - 1) / max(total_frames - 1, 1))
    image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read background frame: {paths[index]}")
    return resize_cover(image, width, height)


def resize_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(image, (int(round(src_w * scale)), int(round(src_h * scale))), interpolation=cv2.INTER_CUBIC)
    y0 = max((resized.shape[0] - height) // 2, 0)
    x0 = max((resized.shape[1] - width) // 2, 0)
    return resized[y0 : y0 + height, x0 : x0 + width].copy()


def load_cutout_sequence(image_dir: Path, mask_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No object A cutout images: {image_dir}")
    cutouts: list[tuple[np.ndarray, np.ndarray]] = []
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_dir / f"{image_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        cutouts.append(crop_alpha(image, feather(mask)))
    if not cutouts:
        raise RuntimeError("No usable object A cutouts.")
    return cutouts


def load_render_panel_paths(directory: Path) -> list[Path]:
    images = sorted(
        (path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if not images:
        raise FileNotFoundError(f"No rendered object panels: {directory}")
    return images


def load_visible_robot_panel_paths(directory: Path) -> list[Path]:
    images = load_render_panel_paths(directory)
    visible: list[Path] = []
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        panel = select_panel(image, "left")
        mask = robot_object_mask(panel)
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        area = int((mask > 0).sum())
        dark = int(((gray < 90) & (mask > 0)).sum())
        if area > int(panel.shape[0] * panel.shape[1] * 0.015) and dark > 80:
            visible.append(image_path)
    return visible or images[:1]


@lru_cache(maxsize=256)
def get_panel_cutout(path_text: str) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(path_text, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read object render panel: {path_text}")
    panel_image = select_panel(image, "left")
    if "object_b_threestudio" in path_text:
        mask = green_object_mask(panel_image)
    elif "object_c_zero123" in path_text:
        mask = robot_object_mask(panel_image)
    else:
        mask = grabcut_center_mask(panel_image)
    return crop_alpha(panel_image, feather(mask))


def select_panel(image: np.ndarray, panel: str) -> np.ndarray:
    if panel != "left":
        raise ValueError(f"Unsupported panel: {panel}")
    width = min(image.shape[0], image.shape[1])
    return image[:, :width].copy()


def grabcut_center_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    rect = (int(width * 0.16), int(height * 0.10), int(width * 0.68), int(height * 0.78))
    mask = np.zeros((height, width), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return largest_component(foreground)


def green_object_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    green = (hue >= 35) & (hue <= 95) & (saturation > 45) & (value > 25)
    dark_green_base = (hue >= 30) & (hue <= 105) & (saturation > 25) & (value > 8)
    mask = np.where(green | dark_green_base, 255, 0).astype(np.uint8)
    height, width = mask.shape
    prior = np.zeros_like(mask)
    cv2.ellipse(prior, (width // 2, int(height * 0.56)), (int(width * 0.33), int(height * 0.42)), 0, 0, 360, 255, -1)
    mask = cv2.bitwise_and(mask, prior)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.dilate(largest_component(mask), np.ones((5, 5), np.uint8), iterations=1)
    return mask


def fast_center_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate(
        [
            lab[: max(2, height // 12), :, :].reshape(-1, 3),
            lab[-max(2, height // 12) :, :, :].reshape(-1, 3),
            lab[:, : max(2, width // 12), :].reshape(-1, 3),
            lab[:, -max(2, width // 12) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background[None, None, :], axis=2)
    mask = np.where(distance > 18, 255, 0).astype(np.uint8)
    prior = np.zeros((height, width), np.uint8)
    cv2.ellipse(prior, (width // 2, int(height * 0.56)), (int(width * 0.36), int(height * 0.40)), 0, 0, 360, 255, -1)
    mask = cv2.bitwise_and(mask, prior)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    mask = cv2.dilate(largest_component(mask), np.ones((5, 5), np.uint8), iterations=1)
    return mask


def robot_object_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate(
        [
            lab[: max(2, height // 12), :, :].reshape(-1, 3),
            lab[-max(2, height // 12) :, :, :].reshape(-1, 3),
            lab[:, : max(2, width // 12), :].reshape(-1, 3),
            lab[:, -max(2, width // 12) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background[None, None, :], axis=2)
    mask = np.where(distance > 16, 255, 0).astype(np.uint8)
    prior = np.zeros((height, width), np.uint8)
    center = (int(width * 0.39), int(height * 0.55))
    axes = (int(width * 0.26), int(height * 0.43))
    cv2.ellipse(prior, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.bitwise_and(mask, prior)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(largest_component(mask), np.ones((5, 5), np.uint8), iterations=1)
    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return np.where(labels == label, 255, 0).astype(np.uint8)


def crop_alpha(image: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = cv2.findNonZero((alpha > 8).astype(np.uint8))
    if coords is None:
        return image, alpha
    x, y, w, h = cv2.boundingRect(coords)
    pad = max(8, int(0.03 * max(w, h)))
    y0, y1 = max(y - pad, 0), min(y + h + pad, image.shape[0])
    x0, x1 = max(x - pad, 0), min(x + w + pad, image.shape[1])
    return image[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


def feather(mask: np.ndarray) -> np.ndarray:
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
    return np.clip(mask, 0, 255).astype(np.uint8)


def resize_sprite(image: np.ndarray, alpha: np.ndarray, *, target_height: int) -> tuple[np.ndarray, np.ndarray]:
    scale = target_height / image.shape[0]
    size = (max(1, int(round(image.shape[1] * scale))), target_height)
    return (
        cv2.resize(image, size, interpolation=cv2.INTER_CUBIC),
        cv2.resize(alpha, size, interpolation=cv2.INTER_CUBIC),
    )


def composite_sprite(
    base: np.ndarray,
    sprite: tuple[np.ndarray, np.ndarray],
    *,
    center: tuple[int, int],
    shadow: bool,
) -> None:
    image, alpha = sprite
    h, w = image.shape[:2]
    x0 = center[0] - w // 2
    y0 = center[1] - h // 2
    if shadow:
        add_shadow(base, center=(center[0], y0 + h - 8), width=int(w * 0.65), height=max(18, int(h * 0.09)))
    x1, y1 = x0 + w, y0 + h
    bx0, by0 = max(x0, 0), max(y0, 0)
    bx1, by1 = min(x1, base.shape[1]), min(y1, base.shape[0])
    if bx0 >= bx1 or by0 >= by1:
        return
    sx0, sy0 = bx0 - x0, by0 - y0
    sx1, sy1 = sx0 + (bx1 - bx0), sy0 + (by1 - by0)
    a = (alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[:, :, None]
    base[by0:by1, bx0:bx1] = (
        image[sy0:sy1, sx0:sx1].astype(np.float32) * a + base[by0:by1, bx0:bx1].astype(np.float32) * (1.0 - a)
    ).astype(np.uint8)


def add_shadow(base: np.ndarray, *, center: tuple[int, int], width: int, height: int) -> None:
    overlay = np.zeros(base.shape[:2], dtype=np.uint8)
    cv2.ellipse(overlay, center, (width // 2, height // 2), 0, 0, 360, 120, -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), max(6, height / 2))
    shadow = overlay.astype(np.float32)[:, :, None] / 255.0
    base[:] = (base.astype(np.float32) * (1.0 - shadow * 0.32)).astype(np.uint8)


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            str(output_path),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
