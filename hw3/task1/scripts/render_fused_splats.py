"""Render HW3 Task1 fusion with the real trained 3D artifacts.

This renderer avoids Blender point-cloud proxies for A/background. It renders
the exported Nerfstudio gaussian-splat PLYs directly with gsplat, and converts
the B/C exported colored OBJ meshes into small surface splats so all four
assets can be composited in a single camera path.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat.rendering import rasterization

SH_C0 = 0.28209479177387814


@dataclass
class SplatAsset:
    name: str
    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=144)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--background-max", type=int, default=650_000)
    parser.add_argument("--object-a-max", type=int, default=360_000)
    parser.add_argument("--mesh-max", type=int, default=260_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    assets = build_assets(args)
    tensors = concat_assets(assets, args.device)
    render_sequence(args, frames_dir, tensors)
    video_path = args.output_dir / "fused_scene.mp4"
    encode_video(frames_dir, video_path, args.fps)
    manifest = build_manifest(args, assets, video_path, time.time() - start)
    (args.output_dir / "fused_scene_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def build_assets(args: argparse.Namespace) -> list[SplatAsset]:
    """Load and normalize the four trained Task1 assets."""
    specs = [
        _gaussian_spec("background", "exports/background/splat/splat.ply", args.background_max, 0.35, (1.5, 98.5), 4.4, (0.0, 0.0), -0.10, (3, 97), 0.22, 0.016),
        _gaussian_spec("object_a", "exports/object_a/splat/splat.ply", args.object_a_max, 0.45, (2, 98), 0.88, (-1.15, 0.10), 0.03, (5, 95), 0.35, 0.022),
        _mesh_spec("object_b", "exports/object_b/mesh/model.obj", args.mesh_max, 0.010, 0.62, (0.0, -0.04), 0.03),
        _mesh_spec("object_c", "exports/object_c/mesh/model.obj", args.mesh_max, 0.010, 0.76, (1.15, 0.08), 0.03),
    ]
    return [_load_normalized_asset(args.run_dir, spec) for spec in specs]


def _gaussian_spec(
    name: str, relative_path: str, max_points: int, opacity_quantile: float, crop_percentile: tuple[float, float],
    target_height: float, xy: tuple[float, float], ground_z: float, robust_percentile: tuple[float, float],
    scale_multiplier: float, scale_max: float,
) -> dict:
    return locals()


def _mesh_spec(
    name: str, relative_path: str, max_points: int, base_scale: float,
    target_height: float, xy: tuple[float, float], ground_z: float,
) -> dict:
    spec = locals()
    spec.update({"robust_percentile": (0, 100), "scale_multiplier": 1.0, "scale_max": 0.014})
    return spec


def _load_normalized_asset(run_dir: Path, spec: dict) -> SplatAsset:
    path = run_dir / spec["relative_path"]
    if "base_scale" in spec:
        asset = load_obj_as_splats(path, spec["name"], max_points=spec["max_points"], base_scale=spec["base_scale"])
    else:
        asset = load_gaussian_ply(
            path, spec["name"], max_points=spec["max_points"],
            opacity_quantile=spec["opacity_quantile"], crop_percentile=spec["crop_percentile"],
        )
    return normalize_asset(
        asset, target_height=spec["target_height"], xy=spec["xy"], ground_z=spec["ground_z"],
        robust_percentile=spec["robust_percentile"], scale_multiplier=spec["scale_multiplier"], scale_max=spec["scale_max"],
    )


def render_sequence(args: argparse.Namespace, frames_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    for frame in range(args.frames):
        image = render_frame(frame, args.frames, args.width, args.height, args.device, **tensors)
        path = frames_dir / f"frame_{frame + 1:04d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def build_manifest(
    args: argparse.Namespace,
    assets: list[SplatAsset],
    video_path: Path,
    elapsed_seconds: float,
) -> dict:
    return {
        "renderer": "gsplat fused splat renderer",
        "run_dir": args.run_dir.as_posix(),
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "assets": [
            {
                "name": asset.name,
                "source": asset.source,
                "splats": int(asset.means.shape[0]),
            }
            for asset in assets
        ],
        "video": video_path.as_posix(),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def load_gaussian_ply(
    path: Path,
    name: str,
    *,
    max_points: int,
    opacity_quantile: float,
    crop_percentile: tuple[float, float],
) -> SplatAsset:
    count, properties, offset = read_ply_header(path)
    data = np.memmap(path, dtype=np.dtype([(prop, "<f4") for prop in properties]), mode="r", offset=offset, shape=(count,))
    means = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    finite = np.isfinite(means).all(axis=1)
    raw_opacity = np.asarray(data["opacity"], dtype=np.float32)
    finite &= np.isfinite(raw_opacity)
    indices = np.flatnonzero(finite)
    lower, upper = crop_percentile
    bounds_min = np.percentile(means[indices], lower, axis=0)
    bounds_max = np.percentile(means[indices], upper, axis=0)
    inside = np.all((means[indices] >= bounds_min) & (means[indices] <= bounds_max), axis=1)
    indices = indices[inside]
    threshold = np.quantile(raw_opacity[indices], opacity_quantile)
    indices = indices[raw_opacity[indices] >= threshold]
    indices = deterministic_sample(indices, max_points, seed=hash(name) & 0xFFFF)

    means = means[indices]
    colors = np.column_stack([data["f_dc_0"][indices], data["f_dc_1"][indices], data["f_dc_2"][indices]])
    colors = np.clip(0.5 + SH_C0 * colors, 0.0, 1.0).astype(np.float32)
    scales = np.exp(np.column_stack([data["scale_0"][indices], data["scale_1"][indices], data["scale_2"][indices]])).astype(np.float32)
    quats = np.column_stack([data["rot_0"][indices], data["rot_1"][indices], data["rot_2"][indices], data["rot_3"][indices]]).astype(np.float32)
    opacities = sigmoid(raw_opacity[indices]).astype(np.float32)
    return SplatAsset(name=name, means=means, quats=quats, scales=scales, opacities=opacities, colors=colors, source=path.as_posix())


def read_ply_header(path: Path) -> tuple[int, list[str], int]:
    count = 0
    properties: list[str] = []
    in_vertex = False
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Invalid PLY header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("format ") and "binary_little_endian" not in text:
                raise RuntimeError(f"Unsupported PLY format: {text}")
            if text.startswith("element vertex "):
                count = int(text.split()[-1])
                in_vertex = True
            elif text.startswith("element ") and not text.startswith("element vertex "):
                in_vertex = False
            elif in_vertex and text.startswith("property "):
                properties.append(text.split()[-1])
            elif text == "end_header":
                return count, properties, handle.tell()


def load_obj_as_splats(path: Path, name: str, *, max_points: int, base_scale: float) -> SplatAsset:
    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if not raw.startswith("v "):
                continue
            values = [float(value) for value in raw.split()[1:]]
            vertices.append((values[0], values[1], values[2]))
            if len(values) >= 6:
                colors.append(tuple(np.clip(values[3:6], 0.0, 1.0)))
            else:
                colors.append((0.72, 0.72, 0.72))
    if not vertices:
        raise RuntimeError(f"OBJ has no vertices: {path}")
    indices = deterministic_sample(np.arange(len(vertices)), max_points, seed=hash(name) & 0xFFFF)
    means = np.asarray(vertices, dtype=np.float32)[indices]
    rgb = np.asarray(colors, dtype=np.float32)[indices]
    scales = np.full((indices.shape[0], 3), base_scale, dtype=np.float32)
    quats = np.zeros((indices.shape[0], 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = np.full(indices.shape[0], 0.92, dtype=np.float32)
    return SplatAsset(name=name, means=means, quats=quats, scales=scales, opacities=opacities, colors=rgb, source=path.as_posix())


def deterministic_sample(indices: np.ndarray, max_points: int, *, seed: int) -> np.ndarray:
    if indices.shape[0] <= max_points:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def normalize_asset(
    asset: SplatAsset,
    *,
    target_height: float,
    xy: tuple[float, float],
    ground_z: float,
    robust_percentile: tuple[float, float],
    scale_multiplier: float,
    scale_max: float,
) -> SplatAsset:
    lower, upper = robust_percentile
    mins = np.percentile(asset.means, lower, axis=0)
    maxs = np.percentile(asset.means, upper, axis=0)
    center = (mins + maxs) * 0.5
    height = max(maxs[2] - mins[2], 1e-6)
    scale = target_height / height
    means = (asset.means - center) * scale
    robust_ground = (mins[2] - center[2]) * scale
    means[:, 0] += xy[0]
    means[:, 1] += xy[1]
    means[:, 2] += ground_z - robust_ground
    scales = np.clip(asset.scales * scale * scale_multiplier, 0.0025, scale_max)
    return SplatAsset(
        name=asset.name,
        means=means.astype(np.float32),
        quats=normalize_quats(asset.quats),
        scales=scales.astype(np.float32),
        opacities=asset.opacities.astype(np.float32),
        colors=asset.colors.astype(np.float32),
        source=asset.source,
    )


def normalize_quats(quats: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm = np.where(norm < 1e-6, 1.0, norm)
    return (quats / norm).astype(np.float32)


def concat_assets(assets: list[SplatAsset], device: str) -> dict[str, torch.Tensor]:
    means = np.concatenate([asset.means for asset in assets], axis=0)
    quats = np.concatenate([asset.quats for asset in assets], axis=0)
    scales = np.concatenate([asset.scales for asset in assets], axis=0)
    opacities = np.concatenate([asset.opacities for asset in assets], axis=0)
    colors = np.concatenate([asset.colors for asset in assets], axis=0)
    return {
        "means": torch.from_numpy(means).to(device),
        "quats": torch.from_numpy(quats).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
        "colors": torch.from_numpy(colors).to(device),
    }


def render_frame(
    frame: int,
    total_frames: int,
    width: int,
    height: int,
    device: str,
    *,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
) -> np.ndarray:
    phase = 2.0 * math.pi * frame / max(total_frames, 1)
    radius = 4.4
    eye = torch.tensor([math.sin(phase) * radius, -math.cos(phase) * radius, 1.85], device=device)
    target = torch.tensor([0.0, 0.02, 0.62], device=device)
    c2w = camera_to_world(eye, target)
    viewmat = nerfstudio_viewmat(c2w).unsqueeze(0)
    focal = 0.92 * width
    intrinsics = torch.tensor(
        [[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        render, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=True,
            near_plane=0.02,
            far_plane=30.0,
            render_mode="RGB",
            sh_degree=None,
            rasterize_mode="antialiased",
            radius_clip=0.2,
        )
    background = torch.tensor([0.70, 0.68, 0.62], device=device, dtype=torch.float32)
    image = render[0, :, :, :3] + (1.0 - alpha[0, :, :, :1]) * background
    image = torch.clamp(image, 0.0, 1.0)
    return (image.detach().cpu().numpy() * 255).astype(np.uint8)


def camera_to_world(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = torch.nn.functional.normalize(target - eye, dim=0)
    up = torch.tensor([0.0, 0.0, 1.0], device=eye.device)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    true_up = torch.nn.functional.normalize(torch.cross(right, forward, dim=0), dim=0)
    c2w = torch.eye(4, device=eye.device, dtype=torch.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def nerfstudio_viewmat(c2w: torch.Tensor) -> torch.Tensor:
    rotation = c2w[:3, :3] * torch.tensor([[1.0, -1.0, -1.0]], device=c2w.device)
    translation = c2w[:3, 3:4]
    rotation_inverse = rotation.transpose(0, 1)
    translation_inverse = -rotation_inverse @ translation
    viewmat = torch.eye(4, device=c2w.device, dtype=torch.float32)
    viewmat[:3, :3] = rotation_inverse
    viewmat[:3, 3:4] = translation_inverse
    return viewmat


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    command = [
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
        "18",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


if __name__ == "__main__":
    main()
