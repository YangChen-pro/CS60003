"""Blender renderer for the HW3 Task1 AI-assets preview.

Usage inside Blender:
  blender -b -P render_ai_assets_preview_blender.py -- RUN_DIR OUTPUT_DIR A_DIR C_IMAGE W H FRAMES SAMPLES
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def main() -> None:
    run_dir, output_dir, object_a_dir, object_c_image, width, height, frames, samples = _args()
    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_scene()
    materials = _materials()
    _build_environment(materials)
    _build_rocket(materials, location=(-1.45, 0.0, 0.05))
    _build_crystal_mushroom(materials, location=(0.0, 0.05, 0.05))
    _build_wood_robot(materials, location=(1.45, 0.0, 0.05))
    _add_reference_panels(object_a_dir, object_c_image)
    _setup_lighting()
    _setup_camera(frames)
    _configure_render(width, height, samples, frames)
    _render_still(output_dir / "preview_hero.png", frame=max(1, frames // 2))
    _render_video(output_dir / "fused_scene.mp4")
    _write_metadata(run_dir, output_dir, object_a_dir, object_c_image, frames, width, height)


def _args() -> tuple[Path, Path, Path, Path, int, int, int, int]:
    argv = sys.argv
    marker = argv.index("--") if "--" in argv else len(argv) - 8
    values = argv[marker + 1 : marker + 9]
    return Path(values[0]), Path(values[1]), Path(values[2]), Path(values[3]), *map(int, values[4:])


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _materials() -> dict[str, bpy.types.Material]:
    return {
        "teal": _principled("rocket teal ceramic", (0.02, 0.46, 0.48, 1), 0.35, 0.42),
        "red": _principled("glossy red nose", (0.92, 0.04, 0.02, 1), 0.45, 0.24),
        "orange": _principled("orange fins", (1.0, 0.34, 0.02, 1), 0.38, 0.3),
        "yellow": _principled("yellow band", (0.95, 0.78, 0.05, 1), 0.25, 0.35),
        "blue": _principled("blue glass", (0.02, 0.09, 0.38, 1), 0.1, 0.08),
        "wood": _wood_material(),
        "dark": _principled("soft black", (0.015, 0.012, 0.01, 1), 0.1, 0.45),
        "stem": _principled("green mushroom stem", (0.18, 0.46, 0.25, 1), 0.2, 0.5),
        "crystal": _emissive("purple crystal cap", (0.45, 0.06, 0.8, 1), 0.35),
        "floor": _wood_material("warm wood floor", scale=18),
        "wall": _principled("warm plaster wall", (0.67, 0.57, 0.46, 1), 0.0, 0.8),
    }


def _principled(name: str, color, metallic: float, roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = color
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    return mat


def _emissive(name: str, color, strength: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    node = nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = color
    node.inputs["Emission"].default_value = color
    if "Emission Strength" in node.inputs:
        node.inputs["Emission Strength"].default_value = strength
    node.inputs["Roughness"].default_value = 0.22
    return mat


def _wood_material(name: str = "warm carved wood", scale: int = 38) -> bpy.types.Material:
    mat = _principled(name, (0.72, 0.43, 0.20, 1), 0.0, 0.42)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 14
    if "Roughness" in noise.inputs:
        noise.inputs["Roughness"].default_value = 0.58
    color = nodes.new("ShaderNodeValToRGB")
    color.color_ramp.elements[0].position = 0.28
    color.color_ramp.elements[0].color = (0.50, 0.29, 0.11, 1)
    color.color_ramp.elements[1].color = (0.95, 0.70, 0.36, 1)
    links.new(noise.outputs["Fac"], color.inputs["Fac"])
    links.new(color.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
    return mat


def _build_environment(mats: dict[str, bpy.types.Material]) -> None:
    _plane("floor", (0, 0, -0.02), (6.0, 4.2, 1), mats["floor"])
    _plane("back wall", (0, 1.7, 1.55), (6.0, 3.1, 1), mats["wall"], rotation=(math.radians(90), 0, 0))
    _plane("left soft wall", (-3.0, 0.0, 1.2), (4.2, 2.4, 1), mats["wall"], rotation=(math.radians(90), 0, math.radians(90)))
    _plane("right soft wall", (3.0, 0.0, 1.2), (4.2, 2.4, 1), mats["wall"], rotation=(math.radians(90), 0, math.radians(90)))


def _build_rocket(mats: dict[str, bpy.types.Material], location) -> None:
    x, y, z = location
    _cylinder("rocket body", (x, y, z + 0.85), 0.42, 1.35, mats["teal"])
    _cone("rocket nose", (x, y, z + 1.67), 0.43, 0.7, mats["red"])
    _cylinder("rocket base", (x, y, z + 0.18), 0.46, 0.25, mats["dark"])
    for dz in [0.55, 1.25]:
        _cylinder("rocket yellow band", (x, y, z + dz), 0.435, 0.055, mats["yellow"])
    _cylinder("round window", (x, y - 0.435, z + 1.06), 0.16, 0.035, mats["blue"], rotation=(math.radians(90), 0, 0))
    for angle in [0, 120, 240]:
        _rocket_fin((x, y, z + 0.35), angle, mats["orange"])
    for angle in range(0, 360, 45):
        rx = x + math.sin(math.radians(angle)) * 0.43
        ry = y - math.cos(math.radians(angle)) * 0.43
        _sphere("rocket rivet", (rx, ry, z + 1.0), 0.028, mats["yellow"])


def _rocket_fin(origin, angle_degrees: float, mat: bpy.types.Material) -> None:
    x, y, z = origin
    angle = math.radians(angle_degrees)
    outward = (math.sin(angle), -math.cos(angle))
    tangent = (math.cos(angle), math.sin(angle))
    verts = [
        (x + outward[0] * 0.36, y + outward[1] * 0.36, z - 0.1),
        (x + outward[0] * 0.72, y + outward[1] * 0.72, z - 0.08),
        (x + outward[0] * 0.42 + tangent[0] * 0.08, y + outward[1] * 0.42 + tangent[1] * 0.08, z + 0.48),
        (x + outward[0] * 0.42 - tangent[0] * 0.08, y + outward[1] * 0.42 - tangent[1] * 0.08, z + 0.48),
    ]
    mesh = bpy.data.meshes.new("rocket fin mesh")
    mesh.from_pydata(verts, [], [(0, 1, 2), (0, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new("rocket fin", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)


def _build_crystal_mushroom(mats: dict[str, bpy.types.Material], location) -> None:
    x, y, z = location
    _cylinder("mushroom stem", (x, y, z + 0.43), 0.18, 0.8, mats["stem"], vertices=32)
    _sphere("crystal cap", (x, y, z + 0.98), 0.48, mats["crystal"], scale=(1.15, 1.0, 0.55))
    for idx, angle in enumerate(range(0, 360, 60)):
        r = 0.25 + 0.08 * (idx % 2)
        _cone("cap crystal", (x + math.sin(math.radians(angle)) * r, y - math.cos(math.radians(angle)) * r, z + 1.18), 0.08, 0.34, mats["crystal"], vertices=5)
    _sphere("mushroom glow", (x, y, z + 1.0), 0.62, mats["crystal"], scale=(1.2, 1.2, 0.45))


def _build_wood_robot(mats: dict[str, bpy.types.Material], location) -> None:
    x, y, z = location
    _cube("robot body", (x, y, z + 0.65), (0.58, 0.34, 0.68), mats["wood"])
    _cube("robot head", (x, y, z + 1.18), (0.68, 0.38, 0.45), mats["wood"])
    for side in [-1, 1]:
        _cube("robot arm", (x + side * 0.53, y, z + 0.68), (0.22, 0.26, 0.55), mats["wood"])
        _cube("robot leg", (x + side * 0.18, y, z + 0.18), (0.25, 0.28, 0.36), mats["wood"])
        _sphere("robot eye", (x + side * 0.16, y - 0.205, z + 1.24), 0.052, mats["dark"])
    _smile((x, y - 0.215, z + 1.08), mats["dark"])


def _add_reference_panels(object_a_dir: Path, object_c_image: Path) -> None:
    object_a_images = sorted(p for p in object_a_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if object_a_images:
        _image_panel(object_a_images[0], (-2.3, 1.665, 1.25), 0.72, "object A source panel")
    _image_panel(object_c_image, (2.3, 1.665, 1.25), 0.72, "object C source panel")


def _image_panel(path: Path, location, size: float, name: str) -> None:
    mat = bpy.data.materials.new(f"{name} material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(path))
    image.extension = "CLIP"
    shader = nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value = 0.35
    mat.node_tree.links.new(image.outputs["Color"], shader.inputs["Base Color"])
    _plane(name, location, (size, size, 1), mat, rotation=(math.radians(90), 0, 0))


def _setup_lighting() -> None:
    bpy.ops.object.light_add(type="AREA", location=(0, -2.2, 4.0))
    key = bpy.context.object
    key.name = "large softbox key"
    key.data.energy = 650
    key.data.size = 4.0
    bpy.ops.object.light_add(type="AREA", location=(-2.4, -1.1, 2.2))
    rim = bpy.context.object
    rim.name = "cool rim light"
    rim.data.energy = 95
    rim.data.size = 2.0


def _setup_camera(frames: int) -> None:
    target = Vector((0, 0.0, 0.95))
    bpy.ops.object.camera_add(location=(0, -5.2, 1.75))
    bpy.context.scene.camera = bpy.context.object
    camera = bpy.context.object
    camera.data.lens = 32
    for frame, angle in [(1, -0.46), (max(2, frames // 2), 0), (frames, 0.46)]:
        camera.location = (math.sin(angle) * 2.0, -5.1 + abs(math.sin(angle)) * 0.25, 1.72)
        _look_at(camera, target)
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)


def _configure_render(width: int, height: int, samples: int, frames: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3
    scene.eevee.gtao_factor = 1.8
    scene.eevee.taa_render_samples = samples
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.frame_start = 1
    scene.frame_end = frames


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _render_still(path: Path, frame: int) -> None:
    scene = bpy.context.scene
    scene.frame_set(frame)
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


def _render_video(path: Path) -> None:
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    bpy.ops.render.render(animation=True)


def _write_metadata(run_dir: Path, output_dir: Path, object_a_dir: Path, object_c_image: Path, frames: int, width: int, height: int) -> None:
    metadata = {
        "run_dir": run_dir.as_posix(),
        "object_a_source": object_a_dir.as_posix(),
        "object_c_source": object_c_image.as_posix(),
        "frames": frames,
        "resolution": [width, height],
        "renderer": "Blender EEVEE preview",
        "boundary": "AI assets preview render, not final 3DGS/SDS training evidence.",
    }
    (output_dir / "preview_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _plane(name: str, location, scale, mat: bpy.types.Material, rotation=(0, 0, 0)) -> None:
    bpy.ops.mesh.primitive_plane_add(size=1, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)


def _cube(name: str, location, scale, mat: bpy.types.Material) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    _bevel(obj, 0.045)


def _cylinder(name: str, location, radius: float, depth: float, mat: bpy.types.Material, vertices: int = 64, rotation=(0, 0, 0)) -> None:
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    _bevel(obj, 0.015)


def _cone(name: str, location, radius: float, depth: float, mat: bpy.types.Material, vertices: int = 64) -> None:
    bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=radius, radius2=0, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    _bevel(obj, 0.008)


def _sphere(name: str, location, radius: float, mat: bpy.types.Material, scale=(1, 1, 1)) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)


def _smile(location, mat: bpy.types.Material) -> None:
    curve = bpy.data.curves.new("robot smile curve", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 24
    curve.bevel_depth = 0.008
    spl = curve.splines.new("BEZIER")
    spl.bezier_points.add(2)
    for point, co in zip(spl.bezier_points, [(-0.1, 0, 0.02), (0, 0, -0.04), (0.1, 0, 0.02)]):
        point.co = co
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new("robot smile", curve)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(mat)


def _bevel(obj: bpy.types.Object, width: float) -> None:
    if hasattr(obj.data, "use_auto_smooth"):
        obj.data.use_auto_smooth = True
    bevel = obj.modifiers.new("soft bevel", "BEVEL")
    bevel.width = width
    bevel.segments = 3
    obj.modifiers.new("weighted normals", "WEIGHTED_NORMAL")


if __name__ == "__main__":
    main()
