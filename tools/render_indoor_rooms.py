"""One indoor room per unified wall material.

Same room, ceiling plaster, hardwood floor, and side walls in that wall's
finish. The inspection wall is real displacement. Writes
``outputs/indoor_rooms/grid.png``.

    PYTHONPATH=. /home/shiven/anaconda3/envs/infinigen_v2/bin/python tools/render_indoor_rooms.py
"""

from __future__ import annotations

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

from syncrack3d.geometry import build_scene  # noqa: E402
from syncrack3d.lighting import LightSpec, LightingPlan, apply_lighting  # noqa: E402
from syncrack3d.material_diversity import UNIFIED_CATEGORIES, _metric_uv, apply_diverse_wall_material  # noqa: E402
from syncrack3d.materials import enable_true_displacement  # noqa: E402
from syncrack3d.render import configure_cycles  # noqa: E402
from syncrack3d.room_shell import ShellPlan, _face_frame, _metric_uv_xy, build_room_shell  # noqa: E402
from syncrack3d.scene_setup import configure_capture_scene  # noqa: E402
from syncrack3d.spec import sample_scene_spec  # noqa: E402

OUT = REPO / "outputs" / "indoor_rooms"
SEED = 4
TILE_W, TILE_H = 720, 540
COLUMNS = 4

ROOMS = (
    ("plaster", "plaster"),
    ("stucco", "stucco"),
    ("concrete", "concrete"),
    ("brick", "brick"),
    ("tile", "tile"),
    ("large_tile", "large tile"),
    ("marble", "marble"),
    ("stone_block", "stone block"),
    ("cobble_wall", "cobble"),
    ("sandstone", "sandstone"),
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _seed_numpy(salt: str) -> None:
    from cwg_infinigen.core.util.math import int_hash

    np.random.seed(int_hash((SEED, salt)) % (2**32))


def _hardwood(floor) -> None:
    from cwg_infinigen.assets.materials.wood.hardwood_floor import HardwoodFloor
    from cwg_infinigen.core import surface

    state = np.random.get_state()
    try:
        _seed_numpy("room-floor")
        material = HardwoodFloor().generate()
        material.displacement_method = "BUMP"
        floor.data.materials.clear()
        surface.assign_material(floor, material)
        _metric_uv_xy(floor)
    finally:
        np.random.set_state(state)


def _match_returns(shell, wall) -> None:
    material = wall.data.materials[0].copy()
    material.displacement_method = "BUMP"
    for obj in shell:
        if not obj.name.startswith("CWG_Shell_Return_"):
            continue
        _face_frame(obj, math.pi / 2 if obj.name.endswith("left") else -math.pi / 2)
        _metric_uv(obj)
        obj.data.materials.clear()
        obj.data.materials.append(material)


def _camera() -> None:
    aim = bpy.data.objects.new("CWG_Room_Aim", None)
    aim.location = (0.0, -0.1, 1.2)
    bpy.context.scene.collection.objects.link(aim)
    cam_data = bpy.data.cameras.new("CWG_Room_Cam")
    cam_data.lens = 24.0
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.05
    cam_data.clip_end = 40.0
    cam = bpy.data.objects.new("CWG_Room_Cam", cam_data)
    cam.location = (0.55, -3.15, 1.4)
    track = cam.constraints.new("TRACK_TO")
    track.target = aim
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam


def _render(path: Path) -> Image.Image:
    from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
    from cwg_infinigen.core.rendering.sensor_response import apply_sensor_response

    scene = bpy.context.scene
    scene.render.resolution_x = TILE_W
    scene.render.resolution_y = TILE_H
    scene.view_settings.exposure = 0.0
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
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
    return Image.open(path).convert("RGB")


def _one(category: str, path: Path) -> Image.Image:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_cycles(resolution=TILE_H, samples=64, denoise=True, dicing_rate=2.0)
    spec = sample_scene_spec(SEED, width_m=4.2, height_m=2.7, thickness_m=0.16, skirting=False)
    wall = build_scene(spec).hosts[0]
    _seed_numpy(f"room-{category}")
    apply_diverse_wall_material(wall, SEED, forced=category, palette="unified")
    enable_true_displacement(wall, 2.0)
    shell = build_room_shell(spec, ShellPlan(
        depth_m=4.4,
        has_ceiling=True,
        returns=("left", "right"),
        floor_color=(0.42, 0.40, 0.38, 1.0),
        ceiling_color=(0.91, 0.90, 0.88, 1.0),
        return_color=(0.74, 0.74, 0.72, 1.0),
    ), SEED)
    _match_returns(shell, wall)
    _hardwood(next(obj for obj in shell if obj.name == "CWG_Shell_Floor"))
    configure_capture_scene(spec, bpy.data.objects["CWG_ROOT"])
    apply_lighting(LightingPlan(
        regime="ceiling_diffuse",
        world_strength=0.35,
        lights=(
            LightSpec("key", "ceiling_diffuse", 4300.0, 90.0, 2.2, 28.0, 15.0,
                      (0.6, -1.6, 2.35), target=(0.0, -0.2, 1.3)),
            LightSpec("fill", "window_soft", 5800.0, 35.0, 1.6, 42.0, 210.0,
                      (-1.8, -2.4, 1.7), target=(0.2, -0.2, 1.1)),
        ),
    ), spec.height_m, bpy.context.scene.collection)
    _camera()
    return _render(path)


def _grid(images: list[tuple[Image.Image, str]], path: Path) -> None:
    gap, caption = 8, 32
    rows = math.ceil(len(images) / COLUMNS)
    sheet = Image.new(
        "RGB",
        (COLUMNS * TILE_W + (COLUMNS + 1) * gap, rows * (TILE_H + caption) + (rows + 1) * gap),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(20)
    for index, (image, label) in enumerate(images):
        col, row = index % COLUMNS, index // COLUMNS
        x = gap + col * (TILE_W + gap)
        y = gap + row * (TILE_H + caption + gap)
        sheet.paste(image, (x, y))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (TILE_W - (box[2] - box[0])) / 2, y + TILE_H + 6), label, fill=(40, 40, 42), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(path, flush=True)


def main() -> None:
    missing = set(UNIFIED_CATEGORIES) - {key for key, _ in ROOMS}
    if missing:
        raise SystemExit(f"room grid is missing {sorted(missing)}")
    import cwg_infinigen.core.constraints.example_solver.room.decorate  # noqa: F401

    images = []
    for category, label in ROOMS:
        path = OUT / "tiles" / f"{category}.png"
        print(f"rendering {label}", flush=True)
        try:
            images.append((_one(category, path), label))
        except Exception:
            traceback.print_exc()
            images.append((Image.new("RGB", (TILE_W, TILE_H), (236, 236, 238)), label))
    _grid(images, OUT / "grid.png")


if __name__ == "__main__":
    main()
