"""Close-up grids of the wall, floor and ceiling finishes.

Walls are the 15 unified palettes (one seed-4 sample; marble picks one of its
three shaders). Floors are the 7 matched-shell generators. Ceilings are the 3
flat colours. Each tile is filled by the surface. Writes

    outputs/surface_grids/walls.png
    outputs/surface_grids/floors.png
    outputs/surface_grids/ceilings.png

    PYTHONPATH=. /home/shiven/anaconda3/envs/infinigen_v2/bin/python tools/render_surface_grids.py
"""

from __future__ import annotations

import importlib
import math
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from syncrack3d.material_diversity import UNIFIED_CATEGORIES, apply_diverse_wall_material  # noqa: E402
from syncrack3d.materials import enable_true_displacement  # noqa: E402
from syncrack3d.render import configure_cycles  # noqa: E402
from syncrack3d.room_shell import (  # noqa: E402
    FLOOR_GENERATORS, _CEILING_COLORS, _metric_uv_xy, ceiling_plaster_material,
)

OUT = REPO / "outputs" / "surface_grids"
TILE = 640
SAMPLES = 96
SEED = 4
LENS_MM = 50.0

WALLS = (
    ("plaster", "plaster"),
    ("stucco", "stucco"),
    ("concrete", "concrete"),
    ("brick", "brick"),
    ("tile", "tile"),
    ("large_tile", "large tile"),
    ("marble", "marble"),
    ("stone_block", "stone block"),
    ("cobble_wall", "cobble"),
    ("rubble_stone", "rubble"),
    ("sandstone", "sandstone"),
)
FLOORS = (
    ("HardwoodFloor", "hardwood"),
    ("SquareWoodTile", "square wood"),
    ("StaggeredWoodTile", "staggered wood"),
    ("CrossedWoodTile", "crossed wood"),
    ("HexagonWoodTile", "hexagon wood"),
    ("NonWoodTile", "non-wood tile"),
    ("Concrete", "concrete"),
)
CEILINGS = (
    (_CEILING_COLORS[0], "light"),
    (_CEILING_COLORS[1], "mid"),
    (_CEILING_COLORS[2], "warm"),
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _light(name: str, energy: float, size: float, location, color) -> None:
    light = bpy.data.lights.new(name, "AREA")
    light.energy = energy
    light.size = size
    light.spread = 140.0
    light.color = color
    obj = bpy.data.objects.new(name, light)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)


def _look(eye, target) -> None:
    aim = bpy.data.objects.new("CWG_Aim", None)
    aim.location = target
    bpy.context.scene.collection.objects.link(aim)
    cam_data = bpy.data.cameras.new("CWG_Cam")
    cam_data.lens = LENS_MM
    cam_data.clip_start = 0.02
    cam_data.clip_end = 40.0
    cam = bpy.data.objects.new("CWG_Cam", cam_data)
    cam.location = eye
    track = cam.constraints.new("TRACK_TO")
    track.target = aim
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam


def _world() -> None:
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("CWG_World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.72, 0.74, 0.78, 1.0)
        bg.inputs["Strength"].default_value = 0.18


def _box(name: str, dimensions, location):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def _seed_numpy(salt: str) -> None:
    from cwg_infinigen.core.util.math import int_hash

    np.random.seed(int_hash((SEED, salt)) % (2**32))


def _render(path: Path, auto_exposure: bool = True, exposure: float = 0.05) -> Image.Image:
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
        auto_exposure=auto_exposure, ev_compensation=exposure, noise=False,
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
    return Image.open(path).convert("RGB")


def _wall(category: str, path: Path) -> Image.Image:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_cycles(resolution=TILE, samples=SAMPLES, denoise=True, dicing_rate=1.5)
    _world()
    # Front face at y = 0, normal -Y, same frame as the inspection wall.
    wall = _box("CWG_Swatch", (4.0, 0.12, 3.2), (0.0, 0.06, 1.6))
    _seed_numpy(f"wall-{category}")
    cls, _, record = apply_diverse_wall_material(wall, SEED, forced=category, palette="unified")
    enable_true_displacement(wall, 1.5)
    print(f"  {category}: {cls}", flush=True)
    _light("CWG_Key", 180.0, 1.6, (1.3, -1.6, 2.4), (1.0, 0.97, 0.93))
    _light("CWG_Fill", 70.0, 2.2, (-1.5, -1.3, 1.8), (0.90, 0.94, 1.0))
    _light("CWG_Rake", 40.0, 0.8, (0.2, -0.7, 2.6), (1.0, 0.98, 0.95))
    _look((0.28, -1.05, 1.85), (0.0, 0.0, 1.55))
    del record
    return _render(path)


def _floor(class_name: str, path: Path) -> Image.Image:
    from cwg_infinigen.core import surface

    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_cycles(resolution=TILE, samples=SAMPLES, denoise=True, dicing_rate=2.0)
    _world()
    # Top face at z = 0. Local XY is the floor, matching the room shell.
    floor = _box("CWG_Swatch", (4.0, 4.0, 0.08), (0.0, 0.0, -0.04))
    module = next(mod for mod, cls in FLOOR_GENERATORS if cls == class_name)
    _seed_numpy(f"floor-{class_name}")
    generator = getattr(importlib.import_module(f"cwg_infinigen.assets.materials.{module}"), class_name)()
    material = generator.generate()
    material.displacement_method = "BUMP"
    floor.data.materials.clear()
    surface.assign_material(floor, material)
    _metric_uv_xy(floor)
    _light("CWG_Key", 160.0, 1.8, (1.2, -1.4, 2.2), (1.0, 0.97, 0.93))
    _light("CWG_Fill", 80.0, 2.4, (-1.4, -0.6, 2.0), (0.90, 0.94, 1.0))
    _look((0.18, -0.62, 1.35), (0.0, 0.0, 0.0))
    return _render(path)


def _ceiling(color, path: Path) -> Image.Image:
    """One plaster material. The coat is the ceiling's light, mid, or warm off-white."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_cycles(resolution=TILE, samples=96, denoise=True, dicing_rate=2.0)
    _world()
    slab = _box("CWG_Swatch", (4.0, 4.0, 0.08), (0.0, 0.0, 0.04))
    label = path.stem.rsplit("_", 1)[-1]
    slab.data.materials.append(ceiling_plaster_material(color, seed={"light": 1, "mid": 2, "warm": 3}.get(label, 0)))
    _metric_uv_xy(slab)
    _light("CWG_Key", 55.0, 1.1, (1.15, -0.55, 0.75), (1.0, 0.98, 0.95))
    _light("CWG_Fill", 18.0, 1.8, (-0.7, -0.8, 1.2), (0.92, 0.95, 1.0))
    _look((0.22, -0.72, 0.92), (0.0, 0.0, 0.0))
    return _render(path, auto_exposure=False, exposure=-1.15)


def _grid(images: list[tuple[Image.Image, str]], columns: int, path: Path) -> None:
    gap = 8
    caption = 36
    rows = math.ceil(len(images) / columns)
    sheet = Image.new(
        "RGB",
        (columns * TILE + (columns + 1) * gap, rows * (TILE + caption) + (rows + 1) * gap),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(22)
    for index, (image, label) in enumerate(images):
        col, row = index % columns, index // columns
        x = gap + col * (TILE + gap)
        y = gap + row * (TILE + caption + gap)
        sheet.paste(image.resize((TILE, TILE), Image.Resampling.LANCZOS), (x, y))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (TILE - (box[2] - box[0])) / 2, y + TILE + 6), label, fill=(40, 40, 42), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(path, flush=True)


def _run(kind: str, items, render, columns: int) -> None:
    images = []
    for key, label in items:
        path = OUT / "tiles" / f"{kind}_{key if isinstance(key, str) else label}.png"
        print(f"rendering {kind} {label}", flush=True)
        try:
            images.append((render(key, path), label))
        except Exception:
            traceback.print_exc()
            images.append((Image.new("RGB", (TILE, TILE), (236, 236, 238)), label))
    _grid(images, columns, OUT / f"{kind}.png")


def main() -> None:
    missing = set(UNIFIED_CATEGORIES) - {key for key, _ in WALLS}
    if missing:
        raise SystemExit(f"wall grid is missing {sorted(missing)}")
    import cwg_infinigen.core.constraints.example_solver.room.decorate  # noqa: F401

    which = set(sys.argv[1:] or ("walls", "floors", "ceilings"))
    if "walls" in which:
        _run("walls", WALLS, _wall, 5)
    if "floors" in which:
        _run("floors", FLOORS, _floor, 4)
    if "ceilings" in which:
        _run("ceilings", CEILINGS, _ceiling, 3)


if __name__ == "__main__":
    main()
