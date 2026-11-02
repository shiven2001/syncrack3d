"""Render the foreground prop types in one labelled grid.

One example of each factory in ``occlusion._PROP_CATALOG``, sat on the floor
and framed on its measured bounds. Writes a tile per type and an unlabelled
``outputs/prop_grid/grid.png``.

    PYTHONPATH=. /home/shiven/anaconda3/envs/infinigen_v2/bin/python tools/render_prop_grid.py
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from syncrack3d.occlusion import _PROP_CATALOG, _load, _world_bounds  # noqa: E402
from syncrack3d.render import configure_cycles  # noqa: E402

OUT = REPO / "outputs" / "prop_grid"
TILE = 768
SAMPLES = 192
SEED = 4
COLUMNS = 3
LENS_MM = 50.0
SENSOR_MM = 36.0

LABELS = {
    "CellShelfFactory": "cell shelf",
    "LargeShelfFactory": "large shelf",
    "FloorLampFactory": "floor lamp",
    "TableDiningFactory": "table",
    "BottleFactory": "bottle",
    "FoodBoxFactory": "food box",
}


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _mat(name: str, color: tuple[float, float, float], roughness: float = 0.85) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.15
    return mat


def _studio() -> bpy.types.Object:
    """Soft product studio: mid-grey floor, pale wall, three-point area lights.

    Energies stay modest so white props (shelves, marble tables) do not blow out
    before auto-exposure lands; relative ratios are fixed and scaled in `_frame`.
    """
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("CWG_World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.70, 0.72, 0.76, 1.0)
        bg.inputs["Strength"].default_value = 0.22

    floor_mesh = bpy.data.meshes.new("CWG_Floor")
    floor_mesh.from_pydata(
        [(x, y, 0.0) for x in (-12.0, 12.0) for y in (-12.0, 12.0)],
        [],
        [(0, 2, 3, 1)],
    )
    floor = bpy.data.objects.new("CWG_Floor", floor_mesh)
    bpy.context.scene.collection.objects.link(floor)
    floor.data.materials.append(_mat("CWG_Floor", (0.48, 0.47, 0.45), 0.8))

    wall_mesh = bpy.data.meshes.new("CWG_Wall")
    wall_mesh.from_pydata(
        [(x, 4.5, z) for x in (-12.0, 12.0) for z in (0.0, 8.0)],
        [],
        [(0, 1, 3, 2)],
    )
    wall = bpy.data.objects.new("CWG_Wall", wall_mesh)
    bpy.context.scene.collection.objects.link(wall)
    wall.data.materials.append(_mat("CWG_Wall", (0.82, 0.83, 0.85), 0.9))

    for name, energy, size, location, color in (
        ("CWG_Key", 320.0, 2.6, (3.2, -4.0, 3.6), (1.00, 0.97, 0.93)),
        ("CWG_Fill", 90.0, 4.2, (-3.8, -2.2, 2.2), (0.90, 0.94, 1.00)),
        ("CWG_Rim", 140.0, 2.0, (-1.8, 3.4, 3.0), (0.94, 0.96, 1.00)),
        ("CWG_Top", 60.0, 3.4, (0.2, -0.8, 4.6), (1.00, 1.00, 1.00)),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy = energy
        light.size = size
        light.spread = 160.0
        light.color = color
        obj = bpy.data.objects.new(name, light)
        obj.location = location
        bpy.context.scene.collection.objects.link(obj)
        aim = obj.constraints.new("TRACK_TO")
        aim.target = floor
        aim.track_axis = "TRACK_NEGATIVE_Z"
        aim.up_axis = "UP_Y"
    return floor


def _render(path: Path) -> None:
    from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
    from cwg_infinigen.core.rendering.sensor_response import apply_sensor_response

    scene = bpy.context.scene
    scene.view_settings.exposure = 0.0
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    # Auto-exposure keeps dark props and white shelves in a similar display range.
    rgb = apply_sensor_response(
        NodeWrangler(tree), layers.outputs["Image"], enabled=True,
        auto_exposure=True, ev_compensation=0.1, noise=False,
    )
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = str(path.parent)
    out.format.file_format = "PNG"
    out.format.color_mode = "RGB"
    out.file_slots[0].path = path.stem
    tree.links.new(rgb, out.inputs[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in path.parent.glob(f"{path.stem}*.png"):
        old.unlink()
    bpy.ops.render.render(write_still=False)
    written = sorted(path.parent.glob(f"{path.stem}*.png"), key=lambda item: item.stat().st_mtime)
    if not written:
        raise RuntimeError(f"Cycles wrote no PNG for {path.stem}")
    written[-1].replace(path)


def _fit_distance(center, points, direction, fov: float, fill: float) -> float:
    """Camera distance so `points` fill `fill` of a square frame."""
    forward = np.asarray(direction, dtype=float)
    forward = forward / np.linalg.norm(forward)
    # Blender camera looks down local -Z with local +Y up, so local +Z
    # points from the target back toward the camera.
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, forward)
    norm = float(np.linalg.norm(right))
    right = np.array([1.0, 0.0, 0.0]) if norm < 1e-8 else right / norm
    up = np.cross(forward, right)
    limit = math.tan(fov * 0.5) * fill
    distance = 0.05
    for point in points:
        rel = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
        span = max(abs(float(np.dot(rel, right))), abs(float(np.dot(rel, up))))
        distance = max(distance, float(np.dot(rel, forward)) + span / limit)
    return distance


def _oriented_corners(obj) -> np.ndarray:
    """World-space corners of the object's local bounding box."""
    return np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box], dtype=float)


def _frame(obj) -> None:
    """Sit the prop on the floor at the origin and look at it from a three-quarter view."""
    # Peek at unrotated extents so wide furniture (tables) can share the shelf pose.
    bpy.context.view_layer.update()
    raw = _world_bounds(obj)
    if raw is None:
        raise RuntimeError("prop has no mesh bounds")
    raw_size = raw[1] - raw[0]
    flat = float(raw_size[2]) < 0.25 * max(float(raw_size[0]), float(raw_size[1]))
    low = float(raw_size[2]) < 0.55 * max(float(raw_size[0]), float(raw_size[1]))
    # Shelves use a shallow three-quarter. A long table needs more yaw so we
    # see the corner, not a thin side elevation.
    obj.rotation_euler = (0.0, 0.0, math.radians(50.0 if (low and not flat) else 22.0))
    bpy.context.view_layer.update()
    bounds = _world_bounds(obj)
    if bounds is None:
        raise RuntimeError("prop has no mesh bounds")
    lo, hi = bounds
    center = (lo + hi) * 0.5
    obj.location = tuple(float(v) for v in (obj.location - Vector((center[0], center[1], lo[2]))))
    bpy.context.view_layer.update()
    bounds = _world_bounds(obj)
    lo, hi = bounds
    center = (lo + hi) * 0.5
    size = hi - lo
    radius = max(0.05, 0.5 * float(np.linalg.norm(size)))
    fov = 2.0 * math.atan(SENSOR_MM / (2.0 * LENS_MM))
    # A long low table has a large bounding sphere, so sphere-fit leaves it
    # small in frame. Fit those on the projected box instead, from a higher
    # three-quarter so the top reads the way a shelf front does.
    short = float(size[2]) < 0.55 * max(float(size[0]), float(size[1]))
    if flat:
        eye_dir = np.array([0.2, -0.55, 1.0])
    elif short:
        eye_dir = np.array([0.95, -0.42, 0.62])
    else:
        eye_dir = np.array([1.0, -0.42, 0.34])
    if short or flat:
        # Fit the mesh corners, not the inflated axis-aligned box around a yawed table.
        distance = _fit_distance(center, _oriented_corners(obj), eye_dir, fov, fill=0.86)
    else:
        distance = radius / math.sin(fov * 0.5) * 1.28
    eye = eye_dir / np.linalg.norm(eye_dir) * distance + center

    aim = bpy.data.objects.new("CWG_Aim", None)
    aim.location = tuple(float(v) for v in center)
    bpy.context.scene.collection.objects.link(aim)
    cam_data = bpy.data.cameras.new("CWG_Cam")
    cam_data.lens = LENS_MM
    cam_data.clip_start = 0.01
    cam_data.clip_end = max(100.0, distance * 8.0)
    cam = bpy.data.objects.new("CWG_Cam", cam_data)
    cam.location = tuple(float(v) for v in eye)
    track = cam.constraints.new("TRACK_TO")
    track.target = aim
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    # Keep illuminance roughly stable across prop size.
    scale = max(radius, 0.25)
    size_mul = {
        "CWG_Key": 1.4,
        "CWG_Fill": 2.4,
        "CWG_Rim": 1.2,
        "CWG_Top": 2.0,
    }
    for light in bpy.data.objects:
        if light.type != "LIGHT":
            continue
        light.data.energy *= scale * scale
        light.data.size = max(0.7, radius * size_mul.get(light.name, 1.5))
        for constraint in light.constraints:
            if constraint.type == "TRACK_TO":
                constraint.target = aim


def _blank(message: str) -> Image.Image:
    image = Image.new("RGB", (TILE, TILE), (236, 236, 238))
    draw = ImageDraw.Draw(image)
    draw.multiline_text((24, 24), message, fill=(40, 40, 42), font=_font(22))
    return image


def _one(module_name: str, class_name: str, path: Path) -> Image.Image:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_cycles(resolution=TILE, samples=SAMPLES, denoise=True, dicing_rate=1.0)
    _studio()
    factory_cls = _load(module_name, class_name)
    if factory_cls is None:
        return _blank(f"{class_name}\nfailed to import")
    obj = factory_cls(SEED).spawn_asset(i=0)
    obj.parent = None
    _frame(obj)
    _render(path)
    return Image.open(path).convert("RGB")


def _grid(tiles: list[tuple[str, Image.Image]], path: Path) -> None:
    gap = 8
    rows = math.ceil(len(tiles) / COLUMNS)
    sheet_w = COLUMNS * TILE + (COLUMNS + 1) * gap
    sheet_h = rows * TILE + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    for index, (_, image) in enumerate(tiles):
        col, row = index % COLUMNS, index // COLUMNS
        x = gap + col * (TILE + gap)
        y = gap + row * (TILE + gap)
        sheet.paste(image.resize((TILE, TILE), Image.Resampling.LANCZOS), (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    tiles = []
    for module_name, class_name in _PROP_CATALOG:
        label = LABELS.get(class_name, class_name)
        path = OUT / "tiles" / f"{label.replace(' ', '_')}.png"
        print(f"rendering {label}", flush=True)
        try:
            image = _one(module_name, class_name, path)
        except Exception:
            traceback.print_exc()
            image = _blank(f"{label}\n{traceback.format_exc().strip().splitlines()[-1]}")
        tiles.append((label, image))
    grid = OUT / "grid.png"
    _grid(tiles, grid)
    print(grid, flush=True)


if __name__ == "__main__":
    main()
