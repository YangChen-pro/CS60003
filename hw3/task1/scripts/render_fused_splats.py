"""Render HW3 Task1 fusion with the real trained 3D artifacts.

This renderer avoids Blender point-cloud proxies for A/background. It renders
the exported Nerfstudio gaussian-splat PLYs directly with gsplat, and converts
the B/C exported colored OBJ meshes into small surface splats so all four
assets can be composited in a single camera path.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
from pathlib import Path
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None
try:
    from PIL import Image
except Exception:
    Image = None
import numpy as np
import torch
from gsplat.rendering import rasterization

SH_C0 = 0.28209479177387814
RENDERER_NAME = "gsplat fused splat renderer"
PIPELINE_MODE = "fused_splats"
DEFAULT_LIGHT_DIR = np.array([0.38, -0.20, 0.90], dtype=np.float32)
DEFAULT_LIGHT_DIR = DEFAULT_LIGHT_DIR / np.linalg.norm(DEFAULT_LIGHT_DIR)
AMBIENT = 0.36
DIFFUSE = 0.74
FOCAL_SCALE_MIN = 0.7
FOCAL_SCALE_MAX = 1.5
KEYFRAME_INDICES = (0, 47, 143)
PALETTE_COLORS = {
    "object_a": np.array([0.92, 0.89, 0.86], dtype=np.float32),
    "object_b": np.array([0.69, 0.53, 0.93], dtype=np.float32),
    "object_c": np.array([0.72, 0.66, 0.56], dtype=np.float32),
    "background": np.array([0.75, 0.74, 0.73], dtype=np.float32),
}


@dataclass
class SplatAsset:
    name: str
    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    normals: Optional[np.ndarray]
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
    parser.add_argument("--object-a-max", type=int, default=450_000)
    parser.add_argument("--mesh-max", type=int, default=260_000)
    parser.add_argument("--object-a-opacity-quantile", type=float, default=0.35)
    parser.add_argument("--object-a-scale-mult", type=float, default=0.44)
    parser.add_argument("--object-b-max", type=int, default=260_000)
    parser.add_argument("--object-c-max", type=int, default=300_000)
    parser.add_argument("--object-b-scale-max", type=float, default=0.0130)
    parser.add_argument("--object-b-base-scale", type=float, default=0.0038)
    parser.add_argument("--object-c-scale-max", type=float, default=0.0110)
    parser.add_argument("--object-c-base-scale", type=float, default=0.0032)
    parser.add_argument("--camera-radius-scale", type=float, default=0.82)
    parser.add_argument("--camera-height-scale", type=float, default=0.86)
    parser.add_argument("--radius-clip", type=float, default=0.085)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--camera-orbit-eccentricity", type=float, default=0.85)
    parser.add_argument("--camera-radial-sweep", type=float, default=0.10)
    parser.add_argument("--camera-height-sweep", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    assets = build_assets(args, rng)
    camera_cfg = build_camera_config(
        args.run_dir,
        assets,
        total_frames=args.frames,
        radius_scale=args.camera_radius_scale,
        height_scale=args.camera_height_scale,
        orbit_eccentricity=args.camera_orbit_eccentricity,
        radial_sweep=args.camera_radial_sweep,
        height_sweep=args.camera_height_sweep,
    )
    camera_cfg["radius_clip"] = float(args.radius_clip)
    camera_cfg["gamma"] = float(args.gamma)
    tensors = concat_assets(assets, args.device)
    render_sequence(args, frames_dir, tensors, camera_cfg)
    video_path = args.output_dir / "fused_scene.mp4"
    encode_video(frames_dir, video_path, args.fps)
    export_strict_keyframes(frames_dir, args.output_dir)
    manifest = build_manifest(args, assets, video_path, time.time() - start, camera_cfg)
    (args.output_dir / "fused_scene_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def build_assets(args: argparse.Namespace, rng: np.random.Generator) -> list[SplatAsset]:
    """Load and normalize the four trained Task1 assets."""
    background_spec = _gaussian_spec(
        name="background",
        relative_path="exports/background/splat/splat.ply",
        max_points=args.background_max,
        opacity_quantile=0.35,
        crop_percentile=(1.5, 98.5),
        target_height=4.6,
        xy=(0.0, 0.0),
        ground_z=0.0,
        robust_percentile=(3, 97),
        scale_multiplier=0.22,
        scale_max=0.018,
        apply_dataparser=True,
        dataparser_target="background",
        seed=int(rng.integers(0, 2**31)),
    )
    background = _load_normalized_asset(args.run_dir, background_spec)
    background_ground = float(np.percentile(background.means, 2, axis=0)[2])
    background_center = np.percentile(background.means, 50, axis=0)

    # Place foreground assets on the reconstructed desk plane derived from the background scene.
    obj_specs = [
        _gaussian_spec(
            name="object_a",
            relative_path="exports/object_a/splat/splat.ply",
            max_points=args.object_a_max,
            opacity_quantile=args.object_a_opacity_quantile,
            crop_percentile=(2, 98),
            target_height=1.0,
            xy=(background_center[0] - 0.95, background_center[1] - 0.03),
            ground_z=background_ground + 0.01,
            robust_percentile=(5, 95),
            scale_multiplier=args.object_a_scale_mult,
            scale_max=0.015,
            apply_dataparser=True,
            dataparser_target="object_a",
            seed=int(rng.integers(0, 2**31)),
        ),
        _mesh_spec(
            name="object_b",
            relative_path="exports/object_b/mesh/model.obj",
            max_points=args.object_b_max,
            base_scale=args.object_b_base_scale,
            scale_cap=0.013,
            scale_max=args.object_b_scale_max,
            target_height=0.78,
            xy=(background_center[0], background_center[1] - 0.01),
            ground_z=background_ground + 0.01,
            seed=int(rng.integers(0, 2**31)),
        ),
        _mesh_spec(
            name="object_c",
            relative_path="exports/object_c/mesh/model.obj",
            max_points=args.object_c_max,
            base_scale=args.object_c_base_scale,
            scale_cap=0.011,
            scale_max=args.object_c_scale_max,
            target_height=0.84,
            xy=(background_center[0] + 0.95, background_center[1] + 0.02),
            ground_z=background_ground + 0.03,
            seed=int(rng.integers(0, 2**31)),
        ),
    ]
    return [background] + [_load_normalized_asset(args.run_dir, spec, reference=background) for spec in obj_specs]



def _gaussian_spec(
    name: str,
    relative_path: str,
    max_points: int,
    opacity_quantile: float,
    crop_percentile: tuple[float, float],
    target_height: float,
    xy: tuple[float, float],
    ground_z: float,
    robust_percentile: tuple[float, float],
    scale_multiplier: float,
    scale_max: float,
    *,
    apply_dataparser: bool = False,
    dataparser_target: str = "",
    seed: int = 0,
) -> dict[str, Any]:
    return locals()


def _mesh_spec(
    name: str,
    relative_path: str,
    max_points: int,
    base_scale: float,
    scale_max: float,
    scale_cap: float,
    target_height: float,
    xy: tuple[float, float],
    ground_z: float,
    seed: int = 0,
) -> dict[str, Any]:
    spec = locals()
    spec.update({"robust_percentile": (0, 100), "scale_multiplier": 1.0})
    spec["apply_dataparser"] = False
    spec["dataparser_target"] = ""
    return spec


def _load_normalized_asset(run_dir: Path, spec: dict, reference: SplatAsset | None = None) -> SplatAsset:
    path = run_dir / spec["relative_path"]
    if "base_scale" in spec:
        asset = load_obj_as_splats(
            path,
            spec["name"],
            max_points=spec["max_points"],
            base_scale=spec["base_scale"],
            scale_cap=spec["scale_cap"],
            seed=spec.get("seed", 0),
        )
    else:
        asset = load_gaussian_ply(
            path, spec["name"], max_points=spec["max_points"],
            opacity_quantile=spec["opacity_quantile"], crop_percentile=spec["crop_percentile"],
        )
    if spec.get("apply_dataparser", False):
        asset = apply_dataparser_transform(
            run_dir,
            asset,
            dataparser_target=spec.get("dataparser_target", ""),
        )
    return normalize_asset(
        asset,
        target_height=spec["target_height"],
        xy=spec["xy"],
        ground_z=spec["ground_z"],
        robust_percentile=spec["robust_percentile"],
        scale_multiplier=spec["scale_multiplier"],
        scale_max=spec["scale_max"],
        reference=reference,
    )


def apply_dataparser_transform(run_dir: Path, asset: SplatAsset, *, dataparser_target: str) -> SplatAsset:
    """Apply nerfstudio dataparser normalization transform and scale to raw asset points."""
    if not dataparser_target:
        return asset
    files = sorted((run_dir / "nerfstudio" / dataparser_target).rglob("dataparser_transforms.json"))
    if not files:
        return asset
    transform_path = files[-1]
    data = json.loads(transform_path.read_text(encoding="utf-8"))
    scale = float(data.get("scale", 1.0))
    matrix = np.array(
        data.get(
            "transform",
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ),
        dtype=np.float32,
    )
    if matrix.shape != (3, 4):
        return asset
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :4] = matrix
    homogeneous = np.concatenate([asset.means * scale, np.ones((asset.means.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = (homogeneous @ transform.T)[:, :3]
    return SplatAsset(
        name=asset.name,
        means=transformed.astype(np.float32),
        quats=asset.quats,
        scales=asset.scales * scale,
        opacities=asset.opacities,
        colors=asset.colors,
        normals=asset.normals,
        source=asset.source,
    )




def _asset_sha256(path: str) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_sequence(
    args: argparse.Namespace,
    frames_dir: Path,
    tensors: dict[str, torch.Tensor],
    camera_cfg: dict[str, Any],
) -> None:
    for frame in range(args.frames):
        image = render_frame(
            frame,
            args.frames,
            args.width,
            args.height,
            args.device,
            camera_cfg=camera_cfg,
            **tensors,
        )
        path = frames_dir / f"frame_{frame + 1:04d}.png"
        write_frame(path, image)


def write_frame(path: Path, image: np.ndarray) -> None:
    if cv2 is not None:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return
    if Image is None:
        raise RuntimeError("No image writer available: install opencv-python or pillow.")
    Image.fromarray(image).save(path)


def export_strict_keyframes(frames_dir: Path, output_dir: Path) -> None:
    for idx, frame_index in enumerate(KEYFRAME_INDICES, start=1):
        source = frames_dir / f"frame_{frame_index + 1:04d}.png"
        if not source.exists():
            continue
        target = output_dir / f"strict_keyframe_{idx:03d}.png"
        target.write_bytes(source.read_bytes())


def build_manifest(
    args: argparse.Namespace,
    assets: list[SplatAsset],
    video_path: Path,
    elapsed_seconds: float,
    camera_cfg: dict[str, Any],
) -> dict:
    focal_scale = float(camera_cfg.get("focal_scale", 0.9))
    if not math.isfinite(focal_scale) or focal_scale <= 0:
        focal_scale = 0.9
    center = camera_cfg.get("center")
    if center is None and camera_cfg.get("centers"):
        center = np.asarray(camera_cfg["centers"], dtype=np.float32).mean(axis=0).tolist()
    if center is None:
        center = [0.0, 0.0, 0.0]
    asset_hashes = {}
    for asset in assets:
        source_path = args.run_dir / asset.source if not Path(asset.source).is_absolute() else Path(asset.source)
        if not source_path.exists():
            cwd_path = Path.cwd() / asset.source
            if cwd_path.exists():
                source_path = cwd_path
        if not source_path.exists():
            alt_path = Path(asset.source.lstrip("/"))
            if alt_path.exists():
                source_path = alt_path
        asset_hashes[asset.name] = _asset_sha256(source_path.as_posix())
    return {
        "renderer": RENDERER_NAME,
        "run_dir": args.run_dir.as_posix(),
        "pipeline_mode": PIPELINE_MODE,
        "source_mode": "unified_3d_assets",
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "seed": args.seed,
        "camera": {
            "name": camera_cfg["mode"],
            "center": center,
            "radius": camera_cfg.get("radius"),
            "height": camera_cfg.get("height"),
            "target_z": camera_cfg.get("target_z"),
            "focal_scale": focal_scale,
            "orbit_eccentricity": camera_cfg.get("orbit_eccentricity"),
            "radial_sweep": camera_cfg.get("radial_sweep"),
            "height_sweep": camera_cfg.get("height_sweep"),
            "radius_clip": camera_cfg.get("radius_clip"),
            "gamma": camera_cfg.get("gamma"),
            "centers": camera_cfg.get("centers"),
            "targets": camera_cfg.get("targets"),
        },
        "assets": [
            {
                "name": asset.name,
                "source": asset.source,
                "splats": int(asset.means.shape[0]),
            }
            for asset in assets
        ],
        "asset_hashes": asset_hashes,
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
    quats = normalize_quats(quats)
    normals = quats_to_normals(quats)
    if normals.shape[0] != means.shape[0]:
        normals = None
    return SplatAsset(
        name=name,
        means=means.astype(np.float32),
        quats=quats.astype(np.float32),
        scales=scales.astype(np.float32),
        opacities=opacities.astype(np.float32),
        colors=colors.astype(np.float32),
        normals=normals,
        source=path.as_posix(),
    )




def load_background_camera_trajectory(
    run_dir: Path,
    background: SplatAsset,
    total_frames: int,
    target_width: int,
) -> dict[str, Any] | None:
    path = run_dir / "processed" / "background" / "transforms.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data.get("frames", [])
    except Exception:
        return None
    if not frames:
        return None
    mats = []
    for frame in frames:
        matrix = frame.get("transform_matrix")
        if not matrix:
            continue
        mats.append(np.array(matrix, dtype=np.float32))
    if not mats:
        return None
    mats_np = np.array(mats, dtype=np.float32)
    centers = mats_np[:, :3, 3]
    if not np.all(np.isfinite(centers)):
        return None
    target = np.percentile(background.means, 50, axis=0)
    span = np.percentile(background.means, 98, axis=0) - np.percentile(background.means, 2, axis=0)
    indices = np.linspace(0, len(centers) - 1, num=total_frames).astype(int)
    centers = centers[indices]
    targets = np.repeat(target[None, :], len(centers), axis=0)
    extent_x = max(float(span[0]), 1e-6)
    extent_y = max(float(span[1]), 1e-6)
    xy_radius = float(np.clip(0.42 * (extent_x + extent_y), 1.0, 4.8))
    focal_scale = float(_estimate_focal_scale(background.means, target_width=float(target_width)))
    return {
        "mode": "background_trajectory",
        "centers": centers.tolist(),
        "targets": targets.tolist(),
        "extent_x": extent_x,
        "extent_y": extent_y,
        "center": target.tolist(),
        "radius": xy_radius,
        "height": float(0.26 * (span[2] + 1.0) + 0.8),
        "target_z": float(span[2] * 0.04),
        "focal_scale": focal_scale,
    }


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


def load_obj_as_splats(
    path: Path,
    name: str,
    *,
    max_points: int,
    base_scale: float,
    scale_cap: float,
    seed: int,
) -> SplatAsset:
    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_normals: list[tuple[float, float, float]] = []
    face_normals: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.startswith("v "):
                values = [float(value) for value in raw.split()[1:]]
                vertices.append((values[0], values[1], values[2]))
                if len(values) >= 6:
                    colors.append(tuple(np.clip(values[3:6], 0.0, 1.0)))
                else:
                    colors.append((0.72, 0.72, 0.72))
            elif raw.startswith("vn "):
                values = [float(value) for value in raw.split()[1:]]
                if len(values) >= 3:
                    normal = np.asarray(values[:3], dtype=np.float32)
                    norm = float(np.linalg.norm(normal))
                    if norm > 1e-6:
                        normal = normal / norm
                    vertex_normals.append(tuple(normal))
            elif raw.startswith("f "):
                tokens = [int(token.split("/")[0]) - 1 for token in raw.split()[1:] if token.split("/")[0]]
                if len(tokens) < 3:
                    continue
                normal_indices: list[int] = []
                for token in raw.split()[1:]:
                    parts = token.split("/")
                    if len(parts) >= 3 and parts[2]:
                        normal_indices.append(int(parts[2]) - 1)
                first = tokens[0]
                for idx in range(1, len(tokens) - 1):
                    faces.append((first, tokens[idx], tokens[idx + 1]))
                    if len(normal_indices) >= 3:
                        face_normals.append((normal_indices[0], normal_indices[idx], normal_indices[idx + 1]))
    if not vertices:
        raise RuntimeError(f"OBJ has no vertices: {path}")
    vertices_np = np.asarray(vertices, dtype=np.float32)
    colors_np = np.asarray(colors, dtype=np.float32)
    vertex_normals_np = np.asarray(vertex_normals, dtype=np.float32) if vertex_normals else np.zeros_like(vertices_np, dtype=np.float32)
    if vertex_normals_np.size and vertex_normals_np.shape[0] >= vertices_np.shape[0]:
        vertex_normals_np = vertex_normals_np[: vertices_np.shape[0]]
    if faces:
        means, rgb, sampled_normals = sample_obj_surface(
            vertices_np,
            colors_np,
            np.asarray(face_normals, dtype=np.int64) if face_normals else np.asarray(faces, dtype=np.int64),
            vertex_normals=np.asarray(vertex_normals_np, dtype=np.float32),
            max_points=max_points,
            seed=seed,
        )
    else:
        indices = deterministic_sample(np.arange(len(vertices_np)), max_points, seed=hash(name) & 0xFFFF)
        means = vertices_np[indices]
        rgb = colors_np[indices]
        sampled_normals = (
            vertex_normals_np[indices]
            if vertex_normals_np.size
            else np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(indices), 1))
        )
    means, rgb = prune_outliers(means, rgb, percentile=0.999)
    if vertex_normals_np.size:
        sampled_normals = sampled_normals[: len(means)]
    scales = adaptive_scales(
        means,
        base_scale=base_scale,
        scale_cap=scale_cap,
        target_count=max_points,
        rng_seed=seed,
    )
    quats = np.zeros((means.shape[0], 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = np.full(means.shape[0], 0.96, dtype=np.float32)
    if not np.isfinite(sampled_normals).all():
        sampled_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32).repeat(len(means), axis=0)
    sampled_normals = np.asarray(sampled_normals, dtype=np.float32)
    norm = np.linalg.norm(sampled_normals, axis=1, keepdims=True)
    norm = np.where(norm < 1e-6, 1.0, norm)
    sampled_normals = sampled_normals / norm
    return SplatAsset(
        name=name,
        means=means.astype(np.float32),
        quats=quats.astype(np.float32),
        scales=scales.astype(np.float32),
        opacities=opacities.astype(np.float32),
        colors=enhance_obj_colors(name, rgb, sampled_normals),
        normals=sampled_normals.astype(np.float32),
        source=path.as_posix(),
    )


def sample_obj_surface(
    vertices: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    faces_arr = np.asarray(faces, dtype=np.int64)
    if faces_arr.size == 0:
        return vertices, colors, vertex_normals
    rng = np.random.default_rng(seed)
    a = vertices[faces_arr[:, 0]]
    b = vertices[faces_arr[:, 1]]
    c = vertices[faces_arr[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    area = np.where(area > 0, area, 1e-12)
    probs = area / area.sum()
    indices = rng.choice(len(faces_arr), size=max_points, replace=True, p=probs)
    triangles = faces_arr[indices]
    tri_a, tri_b, tri_c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    va, vb, vc = vertices[tri_a], vertices[tri_b], vertices[tri_c]
    ca, cb, cc = colors[tri_a], colors[tri_b], colors[tri_c]
    u = rng.random(size=max_points)
    v = rng.random(size=max_points)
    outside = u + v >= 1.0
    u[outside] = 1.0 - u[outside]
    v[outside] = 1.0 - v[outside]
    w = 1.0 - u - v
    points = w[:, None] * va + u[:, None] * vb + v[:, None] * vc
    sampled_colors = w[:, None] * ca + u[:, None] * cb + v[:, None] * cc
    if vertex_normals.shape[0] >= vertices.shape[0]:
        na, nb, nc = vertex_normals[tri_a], vertex_normals[tri_b], vertex_normals[tri_c]
        sampled_normals = w[:, None] * na + u[:, None] * nb + v[:, None] * nc
    else:
        sampled_normals = np.zeros_like(points, dtype=np.float32)
        sampled_normals[:, 2] = 1.0
    sampled_normals = np.where(
        np.linalg.norm(sampled_normals, axis=1, keepdims=True) < 1e-6,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        sampled_normals,
    )
    return (
        points.astype(np.float32),
        np.clip(sampled_colors.astype(np.float32), 0.0, 1.0),
        sampled_normals.astype(np.float32),
    )


def enhance_obj_colors(name: str, colors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Enhance OBJ colors and compensate missing or weak texture.

    Many OBJ exports from Zero123/threestudio have weak color channels; fallback to
    palette + normal shading for stable, plausible appearance.
    """
    if colors.size == 0:
        palette = PALETTE_COLORS.get(name, np.array([0.8, 0.8, 0.8], dtype=np.float32))
        return np.repeat(palette[None, :], repeats=normals.shape[0], axis=0)
    base = colors.astype(np.float32)
    cmin, cmax = float(base.min()), float(base.max())
    if cmax - cmin < 1e-5:
        palette = PALETTE_COLORS.get(name, np.array([0.8, 0.8, 0.8], dtype=np.float32))
        base = np.repeat(palette[None, :], repeats=base.shape[0], axis=0)
    base = np.clip(base, 0.0, 1.0)
    normal_factor = (normals @ DEFAULT_LIGHT_DIR).astype(np.float32)
    normal_factor = np.clip(normal_factor, 0.25, 1.0)
    shading = (AMBIENT + DIFFUSE * normal_factor)[:, None]
    shaded = base * shading
    palette = PALETTE_COLORS.get(name, np.array([0.8, 0.8, 0.8], dtype=np.float32))
    fallback = palette * (
        AMBIENT + DIFFUSE * np.linspace(0.7, 1.0, num=base.shape[0], dtype=np.float32)[:, None]
    )
    blend = 0.5 + 0.5 * normal_factor[:, None]
    return np.clip(blend * shaded + (1.0 - blend) * fallback, 0.0, 1.0)


def adaptive_scales(
    means: np.ndarray,
    *,
    base_scale: float,
    scale_cap: float,
    target_count: int,
    rng_seed: int,
) -> np.ndarray:
    del target_count, rng_seed
    mins = np.percentile(means, 0.2, axis=0)
    maxs = np.percentile(means, 99.8, axis=0)
    diag = np.linalg.norm(maxs - mins) + 1e-6
    target = np.full((means.shape[0], 3), base_scale * (diag / 0.9), dtype=np.float32)
    return np.clip(target, 0.0020, scale_cap)


def prune_outliers(points: np.ndarray, colors: np.ndarray, *, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points, colors
    z = np.linalg.norm(points, axis=1)
    limit = np.quantile(z, percentile)
    keep = z <= limit
    if not np.any(keep):
        return points, colors
    return points[keep], colors[keep]


def build_camera_config(
    run_dir: Path,
    assets: list[SplatAsset],
    total_frames: int,
    *,
    radius_scale: float = 1.0,
    height_scale: float = 1.0,
    orbit_eccentricity: float = 0.85,
    radial_sweep: float = 0.10,
    height_sweep: float = 0.18,
) -> dict[str, Any]:
    background = next(asset for asset in assets if asset.name == "background")
    trajectory = load_background_camera_trajectory(
        run_dir,
        background,
        total_frames=total_frames,
        target_width=1920,
    )
    if trajectory is not None:
        trajectory["orbit_eccentricity"] = float(orbit_eccentricity)
        trajectory["radial_sweep"] = float(radial_sweep)
        trajectory["height_sweep"] = float(height_sweep)
        return trajectory
    mins = np.percentile(background.means, 2, axis=0)
    maxs = np.percentile(background.means, 98, axis=0)
    center = ((mins + maxs) * 0.5).astype(float).tolist()
    span = (maxs - mins).astype(float)
    xy_radius = max(span[0] + span[1], 1e-6) * 0.35
    return {
        "mode": "orbit",
        "center": center,
        "radius": float(np.clip(xy_radius * float(radius_scale), 1.0, 4.5)),
        "height": float(max(1.2, (0.22 * span[2] + 1.6) * float(height_scale))),
        "target_z": float(span[2] * 0.05),
        "focal_scale": _estimate_focal_scale(background.means, target_width=1920),
        "orbit_eccentricity": float(orbit_eccentricity),
        "radial_sweep": float(radial_sweep),
        "height_sweep": float(height_sweep),
    }


def _estimate_focal_scale(points: np.ndarray, target_width: float) -> float:
    mins = np.percentile(points, 4, axis=0)
    maxs = np.percentile(points, 96, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    dominant = float(max(span[0], span[1], 1e-6))
    if not np.isfinite(dominant) or dominant <= 0:
        return 0.9
    return float(np.clip(target_width / (3.4 * dominant), FOCAL_SCALE_MIN, FOCAL_SCALE_MAX))


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
    reference: SplatAsset | None = None,
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
    if reference is not None:
        bg_bottom = np.percentile(reference.means[:, 2], 2)
        bg_top = np.percentile(reference.means[:, 2], 98)
        means[:, 2] += (bg_bottom + 0.015 * (bg_top - bg_bottom)) - np.percentile(means[:, 2], 2)
    return SplatAsset(
        name=asset.name,
        means=means.astype(np.float32),
        quats=normalize_quats(asset.quats),
        scales=scales.astype(np.float32),
        opacities=asset.opacities.astype(np.float32),
        colors=asset.colors.astype(np.float32),
        normals=asset.normals,
        source=asset.source,
    )


def normalize_quats(quats: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm = np.where(norm < 1e-6, 1.0, norm)
    return (quats / norm).astype(np.float32)


def quats_to_normals(quats: np.ndarray) -> np.ndarray:
    """Approximate forward normals from unit quaternions in [x,y,z,w] format."""
    q = normalize_quats(quats)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    normals = np.empty((q.shape[0], 3), dtype=np.float32)
    normals[:, 0] = 2.0 * (x * z + w * y)
    normals[:, 1] = 2.0 * (y * z - w * x)
    normals[:, 2] = 1.0 - 2.0 * (x * x + y * y)
    return np.where(np.linalg.norm(normals, axis=1, keepdims=True) > 1e-6, normals, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def concat_assets(assets: list[SplatAsset], device: str) -> dict[str, torch.Tensor]:
    means = np.concatenate([asset.means for asset in assets], axis=0)
    quats = np.concatenate([asset.quats for asset in assets], axis=0)
    scales = np.concatenate([asset.scales for asset in assets], axis=0)
    opacities = np.concatenate([asset.opacities for asset in assets], axis=0)
    colors = np.concatenate([asset.colors for asset in assets], axis=0)
    normals = np.concatenate(
        [np.zeros_like(asset.means) if asset.normals is None else asset.normals for asset in assets],
        axis=0,
    )
    return {
        "means": torch.from_numpy(means).to(device),
        "quats": torch.from_numpy(quats).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
        "colors": torch.from_numpy(colors).to(device),
        "normals": torch.from_numpy(normals).to(device),
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
    normals: torch.Tensor,
    camera_cfg: dict[str, Any],
) -> np.ndarray:
    phase = 2.0 * math.pi * frame / max(total_frames, 1)
    mode = camera_cfg.get("mode")
    orbit_eccentricity = float(camera_cfg.get("orbit_eccentricity", 0.85))
    radial_sweep = float(camera_cfg.get("radial_sweep", 0.10))
    height_sweep = float(camera_cfg.get("height_sweep", 0.18))
    if mode == "background_trajectory":
        centers = camera_cfg.get("centers", [])
        targets = camera_cfg.get("targets", [])
        if centers:
            center_np = np.asarray(centers[frame % len(centers)], dtype=np.float32)
            target_np = np.asarray(targets[frame % len(targets)], dtype=np.float32)
        else:
            center_np = np.asarray(camera_cfg["center"], dtype=np.float32)
            target_np = np.asarray([camera_cfg["center"][0], camera_cfg["center"][1], camera_cfg["center"][2] + camera_cfg.get("target_z", 0.0)], dtype=np.float32)
        center = torch.tensor(center_np, device=device, dtype=torch.float32)
        target = torch.tensor(target_np, device=device, dtype=torch.float32)
        radius = float(camera_cfg.get("radius", 2.5))
        eye = center + torch.tensor(
            [
                radius * math.sin(phase),
                radius * math.cos(phase) * orbit_eccentricity,
                radius * 0.16 * math.cos(1.3 * phase),
            ],
            device=device,
            dtype=torch.float32,
        )
    else:
        center = torch.tensor(camera_cfg["center"], device=device, dtype=torch.float32)
        radius = float(camera_cfg["radius"]) * (1.0 + float(radial_sweep) * math.sin(2.0 * phase))
        eye = torch.tensor(
            [
                center[0] + math.sin(phase) * radius,
                center[1] + math.cos(phase) * radius,
                center[2] + float(camera_cfg["height"]) + height_sweep * math.cos(1.7 * phase),
            ],
            device=device,
            dtype=torch.float32,
        )
        target = center + torch.tensor([0.0, 0.0, float(camera_cfg["target_z"])], device=device, dtype=torch.float32)
    c2w = camera_to_world(eye, target)
    viewmat = nerfstudio_viewmat(c2w).unsqueeze(0)
    focal = float(camera_cfg.get("focal_scale", 0.92)) * width
    # apply light shading for all points before rasterization
    light_vec = torch.tensor(DEFAULT_LIGHT_DIR, device=device, dtype=torch.float32)
    normal_factor = (normals @ light_vec).clamp(-0.25, 1.0)
    view_factor = torch.nn.functional.normalize(eye[None, :] - means, dim=1)
    view_factor = torch.clamp((normals * view_factor).sum(dim=1), 0.0, 1.0)
    shade = (
        (AMBIENT + DIFFUSE * normal_factor) * 0.78
        + 0.22 * view_factor
    ).clamp(0.0, 1.2).unsqueeze(1)
    lit_colors = torch.clamp(colors * shade, 0.0, 1.0)
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
            colors=lit_colors,
            viewmats=viewmat,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=True,
            near_plane=0.01,
            far_plane=30.0,
            render_mode="RGB",
            sh_degree=None,
            rasterize_mode="antialiased",
            radius_clip=float(camera_cfg.get("radius_clip", 0.085)),
        )
    background = torch.tensor([0.72, 0.70, 0.67], device=device, dtype=torch.float32)
    image = render[0, :, :, :3] + (1.0 - alpha[0, :, :, :1]) * background
    # tone mapping for image contrast and gamma control.
    image = torch.clamp(image, 0.0, 1.0)
    gamma = float(camera_cfg.get("gamma", 1.0))
    if gamma > 0 and abs(gamma - 1.0) > 1e-6:
        image = torch.pow(image, 1.0 / gamma)
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
