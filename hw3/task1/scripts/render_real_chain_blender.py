"""Blender renderer for the HW3 Task1 real high-quality chain.

The authoritative A/background artifacts are Nerfstudio gaussian-splat PLY
files. Blender 2.82 cannot render those splats natively, so this script samples
them into small colored proxy octahedra for the report video while keeping the
full splat PLYs as the saved model artifacts.

Usage inside Blender:
  blender -b -P render_real_chain_blender.py -- RUN_DIR OUTPUT_DIR
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

import bpy
import numpy as np
from mathutils import Vector

SH_C0 = 0.28209479177387814


def main() -> None:
    run_dir, output_dir = _args()
    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_scene()

    background = _import_asset(run_dir, "background", max_points=70000, radius=0.018)
    _fit_and_place(background, target_height=4.5, ground_z=-0.05, xy=(0.0, 0.0), robust_percentile=(2, 98))

    object_a = _import_asset(run_dir, "object_a", max_points=60000, radius=0.025)
    _fit_and_place(object_a, target_height=0.92, ground_z=0.02, xy=(-1.15, 0.05), robust_percentile=(2, 98))

    object_b = _import_asset(run_dir, "object_b", max_points=0, radius=0.0)
    _fit_and_place(object_b, target_height=0.62, ground_z=0.02, xy=(0.0, -0.04))

    object_c = _import_asset(run_dir, "object_c", max_points=0, radius=0.0)
    _fit_and_place(object_c, target_height=0.72, ground_z=0.02, xy=(1.15, 0.05))

    _setup_lighting()
    _setup_camera()
    _setup_render_settings(output_dir)
    bpy.ops.render.render(animation=True)


def _args() -> tuple[Path, Path]:
    argv = sys.argv
    marker = argv.index("--") if "--" in argv else len(argv) - 2
    return Path(argv[marker + 1]), Path(argv[marker + 2])


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_asset(run_dir: Path, label: str, *, max_points: int, radius: float) -> list[bpy.types.Object]:
    mesh_dir = run_dir / "exports" / label / "mesh"
    mesh = _first_existing(mesh_dir, ("*.obj", "*.glb", "*.ply"))
    if mesh and mesh.name != "SKIPPED_TSDF.json":
        return _import_mesh(mesh, label)

    splat = _first_existing(run_dir / "exports" / label / "splat", ("*.ply",))
    if splat and max_points > 0:
        return [_import_splat_proxy(splat, label, max_points=max_points, radius=radius)]

    raise FileNotFoundError(f"missing renderable asset for {label} under {run_dir / 'exports' / label}")


def _first_existing(directory: Path, patterns: tuple[str, ...]) -> Optional[Path]:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in directory.rglob(pattern) if path.is_file())
    candidates = [path for path in candidates if not path.name.startswith("SKIPPED_")]
    return sorted(candidates)[0] if candidates else None


def _import_mesh(path: Path, label: str) -> list[bpy.types.Object]:
    if path.suffix.lower() == ".obj":
        return [_import_colored_obj(path, label)]
    if path.suffix.lower() == ".ply":
        return [_import_plain_ply(path, label)]

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    for obj in imported:
        obj.name = f"{label}_{obj.name}"
    return imported


def _import_colored_obj(path: Path, label: str) -> bpy.types.Object:
    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.startswith("v "):
                parts = raw.split()
                values = [float(value) for value in parts[1:]]
                vertices.append((values[0], values[1], values[2]))
                colors.append(tuple(_clamp_color(values[3:6])) if len(values) >= 6 else (0.72, 0.72, 0.72))
            elif raw.startswith("f "):
                face = [_obj_index(part, len(vertices)) for part in raw.split()[1:]]
                if len(face) >= 3:
                    faces.append(face)

    mesh = bpy.data.meshes.new(f"{label}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{label}_colored_obj", mesh)
    bpy.context.collection.objects.link(obj)
    _assign_loop_colors(mesh, colors)
    obj.data.materials.append(_vertex_color_material(f"{label}_vertex_color"))
    return obj


def _import_plain_ply(path: Path, label: str) -> bpy.types.Object:
    before = set(bpy.data.objects)
    if hasattr(bpy.ops, "import_mesh"):
        bpy.ops.import_mesh.ply(filepath=str(path))
    else:
        raise RuntimeError("This Blender build has no PLY importer.")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError(f"PLY import produced no objects: {path}")
    obj = imported[0]
    obj.name = f"{label}_{obj.name}"
    return obj


def _import_splat_proxy(path: Path, label: str, *, max_points: int, radius: float) -> bpy.types.Object:
    points, colors = _sample_gaussian_splat(path, max_points)
    vertices, faces, loop_colors = _octahedra(points, colors, radius)
    mesh = bpy.data.meshes.new(f"{label}_splat_proxy_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{label}_splat_proxy", mesh)
    bpy.context.collection.objects.link(obj)
    _assign_face_loop_colors(mesh, loop_colors)
    obj.data.materials.append(_vertex_color_material(f"{label}_splat_color"))
    return obj


def _sample_gaussian_splat(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    count, properties, offset = _read_ply_header(path)
    dtype = np.dtype([(name, "<f4") for name in properties])
    data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))

    xyz = np.column_stack([data["x"], data["y"], data["z"]])
    finite = np.isfinite(xyz).all(axis=1)
    opacity = np.asarray(data["opacity"]) if "opacity" in data.dtype.names else np.ones(count, dtype=np.float32)
    finite &= np.isfinite(opacity)
    candidates = np.flatnonzero(finite)
    if candidates.size == 0:
        raise RuntimeError(f"no finite gaussian splat points: {path}")

    if candidates.size > max_points * 4:
        threshold = np.quantile(opacity[candidates], 0.70)
        strong = candidates[opacity[candidates] >= threshold]
        candidates = strong if strong.size >= max_points else candidates

    rng = np.random.default_rng(42)
    if candidates.size > max_points:
        candidates = rng.choice(candidates, size=max_points, replace=False)
    candidates = np.sort(candidates)

    points = xyz[candidates].astype(np.float32)
    colors = np.column_stack([data["f_dc_0"][candidates], data["f_dc_1"][candidates], data["f_dc_2"][candidates]])
    colors = np.clip(0.5 + SH_C0 * colors, 0.0, 1.0).astype(np.float32)
    return points, colors


def _read_ply_header(path: Path) -> tuple[int, list[str], int]:
    count = 0
    properties: list[str] = []
    in_vertex = False
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"invalid PLY header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("format ") and "binary_little_endian" not in text:
                raise RuntimeError(f"unsupported PLY format for {path}: {text}")
            if text.startswith("element vertex "):
                count = int(text.split()[-1])
                in_vertex = True
            elif text.startswith("element ") and not text.startswith("element vertex "):
                in_vertex = False
            elif in_vertex and text.startswith("property "):
                properties.append(text.split()[-1])
            elif text == "end_header":
                return count, properties, handle.tell()


def _octahedra(points: np.ndarray, colors: np.ndarray, radius: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[tuple[float, float, float]]]:
    offsets = np.array(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        dtype=np.float32,
    ) * radius
    local_faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    loop_colors: list[tuple[float, float, float]] = []
    for point, color in zip(points, colors):
        base = len(vertices)
        vertices.extend(tuple(row) for row in (point + offsets))
        faces.extend(tuple(base + index for index in face) for face in local_faces)
        loop_colors.extend([tuple(color)] * len(local_faces))
    return vertices, faces, loop_colors


def _assign_loop_colors(mesh: bpy.types.Mesh, vertex_colors: list[tuple[float, float, float]]) -> None:
    color_layer = mesh.vertex_colors.new(name="Col")
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            rgb = vertex_colors[vertex_index]
            color_layer.data[loop_index].color = (rgb[0], rgb[1], rgb[2], 1.0)


def _assign_face_loop_colors(mesh: bpy.types.Mesh, face_colors: list[tuple[float, float, float]]) -> None:
    color_layer = mesh.vertex_colors.new(name="Col")
    for poly, rgb in zip(mesh.polygons, face_colors):
        for loop_index in poly.loop_indices:
            color_layer.data[loop_index].color = (rgb[0], rgb[1], rgb[2], 1.0)


def _vertex_color_material(name: str) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    attribute = nodes.new(type="ShaderNodeAttribute")
    attribute.attribute_name = "Col"
    material.node_tree.links.new(attribute.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.56
    return material


def _obj_index(token: str, vertex_count: int) -> int:
    raw = token.split("/")[0]
    index = int(raw)
    return index - 1 if index > 0 else vertex_count + index


def _clamp_color(values: list[float]) -> tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, value)) for value in values[:3])


def _add_anchor_plane() -> None:
    bpy.ops.mesh.primitive_plane_add(size=4.2, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object
    plane.name = "subtle_contact_plane"
    material = bpy.data.materials.new("warm_counter_contact_surface")
    material.diffuse_color = (0.42, 0.36, 0.28, 1.0)
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.42, 0.36, 0.28, 1.0)
    material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.72
    plane.data.materials.append(material)


def _setup_lighting() -> None:
    bpy.ops.object.light_add(type="AREA", location=(0, -3.2, 4.8))
    key = bpy.context.object
    key.data.energy = 720
    key.data.size = 5.8
    bpy.ops.object.light_add(type="AREA", location=(-3.0, 2.6, 2.4))
    rim = bpy.context.object
    rim.data.energy = 95
    rim.data.size = 3.0


def _setup_camera() -> None:
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.65))
    target = bpy.context.object
    target.name = "camera_target"
    bpy.ops.object.camera_add(location=(0, -4.8, 2.0))
    camera = bpy.context.object
    camera.data.lens = 42
    bpy.context.scene.camera = camera
    for frame, angle in [(1, 0.0), (72, math.pi), (144, 2 * math.pi)]:
        camera.location = (math.sin(angle) * 4.4, -math.cos(angle) * 4.4, 2.05)
        direction = target.location - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)


def _setup_render_settings(output_dir: Path) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(os.environ.get("HW3_RENDER_SAMPLES", "96"))
    scene.render.resolution_x = int(os.environ.get("HW3_RENDER_WIDTH", "1920"))
    scene.render.resolution_y = int(os.environ.get("HW3_RENDER_HEIGHT", "1080"))
    scene.render.fps = 24
    scene.frame_start = 1
    scene.frame_end = int(os.environ.get("HW3_RENDER_FRAME_END", "144"))
    scene.render.filepath = str(output_dir / "fused_scene.mp4")
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    _set_color_management(scene)


def _fit_and_place(
    objects: list[bpy.types.Object],
    *,
    target_height: float,
    ground_z: float,
    xy: tuple[float, float],
    robust_percentile: Optional[tuple[int, int]] = None,
) -> None:
    if not objects:
        return
    mins, maxs = _bounds(objects, robust_percentile=robust_percentile)
    center = (mins + maxs) / 2
    height = max(maxs.z - mins.z, 1e-6)
    scale = target_height / height
    for obj in objects:
        obj.location = (obj.location - center) * scale
        obj.scale *= scale
    mins, _ = _bounds(objects, robust_percentile=robust_percentile)
    delta = Vector((xy[0], xy[1], ground_z - mins.z))
    for obj in objects:
        obj.location += delta


def _bounds(objects: list[bpy.types.Object], *, robust_percentile: Optional[tuple[int, int]] = None) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    coords: list[Vector] = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        if robust_percentile and getattr(evaluated.data, "vertices", None):
            coords.extend(evaluated.matrix_world @ vertex.co for vertex in evaluated.data.vertices)
        else:
            coords.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    if robust_percentile:
        array = np.array([[v.x, v.y, v.z] for v in coords], dtype=np.float32)
        low, high = robust_percentile
        mins_arr = np.percentile(array, low, axis=0)
        maxs_arr = np.percentile(array, high, axis=0)
        return Vector(tuple(mins_arr)), Vector(tuple(maxs_arr))
    mins = Vector((min(v.x for v in coords), min(v.y for v in coords), min(v.z for v in coords)))
    maxs = Vector((max(v.x for v in coords), max(v.y for v in coords), max(v.z for v in coords)))
    return mins, maxs


def _set_color_management(scene: bpy.types.Scene) -> None:
    for view_transform in ("Filmic", "Standard"):
        try:
            scene.view_settings.view_transform = view_transform
            break
        except TypeError:
            continue
    for look in ("Medium High Contrast", "Filmic - Medium High Contrast", "Filmic - High Contrast", "None"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


if __name__ == "__main__":
    main()
