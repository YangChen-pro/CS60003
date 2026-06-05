"""Blender renderer for HW3 Task1 real high-quality chain.

Usage inside Blender:
  blender -b -P render_real_chain_blender.py -- RUN_DIR OUTPUT_DIR
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy


def main() -> None:
    run_dir, output_dir = _args()
    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_scene()
    _import_first(run_dir / "exports" / "background" / "mesh", "background")
    _import_first(run_dir / "exports" / "object_a" / "mesh", "object_a", location=(-1.2, 0.0, 0.0))
    _import_first(run_dir / "object_b_threestudio", "object_b", location=(0.0, 0.0, 0.0))
    _import_first(run_dir / "object_c_triposr", "object_c", location=(1.2, 0.0, 0.0))
    _setup_lighting()
    _setup_camera()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.frame_start = 1
    scene.frame_end = 96
    _animate_camera()
    scene.render.filepath = str(output_dir / "fused_scene.mp4")
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    bpy.ops.render.render(animation=True)


def _args() -> tuple[Path, Path]:
    argv = sys.argv
    marker = argv.index("--") if "--" in argv else len(argv) - 2
    return Path(argv[marker + 1]), Path(argv[marker + 2])


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_first(directory: Path, label: str, location=(0.0, 0.0, 0.0)) -> None:
    candidates = list(directory.rglob("*.obj")) + list(directory.rglob("*.ply")) + list(directory.rglob("*.glb"))
    if not candidates:
        raise FileNotFoundError(f"missing mesh for {label}: {directory}")
    path = candidates[0]
    if path.suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif path.suffix == ".ply":
        _import_ply(path)
    else:
        bpy.ops.import_scene.gltf(filepath=str(path))
    for obj in bpy.context.selected_objects:
        obj.name = f"{label}_{obj.name}"
        obj.location.x += location[0]
        obj.location.y += location[1]
        obj.location.z += location[2]


def _import_ply(path: Path) -> None:
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))


def _setup_lighting() -> None:
    bpy.ops.object.light_add(type="AREA", location=(0, -3, 5))
    light = bpy.context.object
    light.data.energy = 650
    light.data.size = 5


def _setup_camera() -> None:
    bpy.ops.object.camera_add(location=(0, -5.0, 2.1), rotation=(math.radians(68), 0, 0))
    bpy.context.scene.camera = bpy.context.object


def _animate_camera() -> None:
    camera = bpy.context.scene.camera
    for frame, angle in [(1, 0), (48, math.pi), (96, 2 * math.pi)]:
        camera.location = (math.sin(angle) * 4.5, -math.cos(angle) * 4.5, 2.1)
        camera.rotation_euler = (math.radians(68), 0, angle)
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)


if __name__ == "__main__":
    main()
