"""Render the seven foreground prop types in one labelled grid.

One example of each factory in ``occlusion._PROP_CATALOG``, sat on the floor
and framed on its measured bounds. Writes a tile per type and
``outputs/prop_grid/grid.png``.

    PYTHONPATH=. .venv/bin/python tools/render_prop_grid.py
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
TILE = 640
SAMPLES = 32
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
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _studio() -> bpy.types.Object:
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("CWG_World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.62, 0.63, 0.65, 1.0)
        bg.inputs["Strength"].default_value = 0.35

    mesh = bpy.data.meshes.new("CWG_Floor")
    mesh.from_pydata(
        [(x, y, 0.0) for x in (-20.0, 20.0) for y in (-20.0, 20.0)],
        [],
        [(0, 2, 3, 1)],
    )
    floor = bpy.data.objects.new("CWG_Floor", mesh)
    bpy.context.scene.collection.objects.link(floor)
    mat = bpy.data.materials.new("CWG_Floor")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.45, 0.46, 0.47, 1.0)
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.7
    floor.data.materials.append(mat)

    for name, energy, location in (
        ("CWG_Key", 800.0, (2.2, -3.4, 3.2)),
        ("CWG_Fill", 220.0, (-2.6, -1.6, 2.2)),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy = energy
        light.size = 2.4
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
    rgb = apply_sensor_response(
        NodeWrangler(tree), layers.outputs["Image"], enabled=True,
        auto_exposure=True, ev_compensation=0.0, noise=False,
    )
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = str(path.parent)
    out.format.file_format = "PNG"
    out.format.color_mode = "RGB"
    out.file_slots[0].path = path.stem
    tree.links.new(rgb, out.inputs[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=False)
    written = sorted(path.parent.glob(f"{path.stem}*.png"), key=lambda item: item.stat().st_mtime)
    frame = next(item for item in written if item != path)
    frame.replace(path)


def _frame(obj) -> None:
    """Sit the prop on the floor at the origin and look at it from a three-quarter view."""
    obj.rotation_euler = (0.0, 0.0, math.radians(18.0))
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
    distance = radius / math.sin(fov * 0.5) * 1.35
    # A flat prop reads from above. A shelf's open face is the
    # wider vertical face, so the camera sits on the thinner horizontal axis.
    if size[2] < 0.25 * max(size[0], size[1]):
        eye = np.array([0.2, -0.55, 1.0])
    elif size[0] <= size[1]:
        eye = np.array([1.0, -0.38, 0.36])
    else:
        eye = np.array([0.38, -1.0, 0.36])
    eye = eye / np.linalg.norm(eye) * distance + center

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

    scale = max(radius, 0.2)
    for light in bpy.data.objects:
        if light.type != "LIGHT":
            continue
        light.data.energy *= scale * scale
        light.data.size = max(0.4, radius * 1.6)
        for constraint in light.constraints:
            if constraint.type == "TRACK_TO":
                constraint.target = aim


def _blank(message: str) -> Image.Image:
    image = Image.new("RGB", (TILE, TILE), (40, 40, 42))
    draw = ImageDraw.Draw(image)
    draw.multiline_text((24, 24), message, fill=(230, 210, 180), font=_font(22))
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
    label_h = 44
    rows = math.ceil(len(tiles) / COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * TILE, rows * (TILE + label_h)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    font = _font(26)
    for index, (label, image) in enumerate(tiles):
        col, row = index % COLUMNS, index // COLUMNS
        x, y = col * TILE, row * (TILE + label_h)
        sheet.paste(image.resize((TILE, TILE), Image.Resampling.LANCZOS), (x, y))
        text = label
        width = draw.textlength(text, font=font)
        draw.text((x + (TILE - width) / 2, y + TILE + 8), text, fill=(235, 235, 235), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    tiles = []
    for module_name, class_name, _footprint, _on_floor in _PROP_CATALOG:
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
