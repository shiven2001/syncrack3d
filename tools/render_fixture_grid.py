"""Render one example of each fixture kind in an unlabelled grid.

Vent and the access panel are each one kind. A wire, conduit, pipe or plumbing
run draws its own direction, down the wall or across it. Each assembly is built
on a short wall patch and framed on its measured bounds. Writes
``outputs/fixture_grid/grid.png``.

    PYTHONPATH=. /home/shiven/anaconda3/envs/infinigen_v2/bin/python tools/render_fixture_grid.py
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from syncrack3d.occlusion import _world_bounds  # noqa: E402
from syncrack3d.render import configure_cycles  # noqa: E402
from syncrack3d.structures import SPAN_KINDS, SPAN_SECTION, build_structures, sample_structure_plan  # noqa: E402

OUT = REPO / "outputs" / "fixture_grid"
TILE = 640
SAMPLES = 128
SEED = 4
COLUMNS = 4
LENS_MM = 50.0
SENSOR_MM = 36.0

# One tile per kind. Run direction is drawn inside the kind, not shown twice.
KINDS = (
    "utility_box", "vent", "window", "door",
    "access_panel", "trunking", "wall_cable", "cable",
    "conduit", "downpipe", "plumbing", "panel_seam",
)
NAMES = {
    "utility_box": "utility box",
    "vent": "vent",
    "window": "window",
    "door": "door",
    "access_panel": "access panel",
    "trunking": "trunking",
    "wall_cable": "wires",
    "cable": "cable",
    "conduit": "conduit",
    "downpipe": "pipe",
    "plumbing": "plumbing",
    "panel_seam": "panel seam",
}

# Host wall (width, height) so a full-width trim is a short visible run.
WALL = {
    "utility_box": (1.1, 1.3),
    "vent": (1.4, 1.3),
    "window": (2.0, 1.7),
    "door": (1.8, 2.5),
    "access_panel": (1.4, 1.4),
    "wall_cable": (2.2, 2.2),
    "cable": (2.4, 1.4),
    "conduit": (2.2, 2.2),
    "trunking": (2.2, 2.2),
    "plumbing": (2.2, 2.2),
    "downpipe": (2.2, 2.2),
    "panel_seam": (2.2, 2.2),
}


def _fit_distance(center, points, direction, fov: float, fill: float) -> float:
    forward = np.asarray(direction, dtype=float)
    forward = forward / np.linalg.norm(forward)
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


def _box_corners(lo, hi) -> np.ndarray:
    return np.array([
        [x, y, z]
        for x in (float(lo[0]), float(hi[0]))
        for y in (float(lo[1]), float(hi[1]))
        for z in (float(lo[2]), float(hi[2]))
    ])


def _mat(name: str, color: tuple[float, float, float], roughness: float = 0.85) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.12
    return mat


def _plane(name: str, corners: list[tuple[float, float, float]], color, roughness: float) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(corners, [], [(0, 1, 3, 2)])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(_mat(name, color, roughness))
    return obj


def _studio(center) -> None:
    """Pale wall in the XZ plane (fixtures stand out toward -Y) and a floor."""
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("CWG_World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.70, 0.72, 0.76, 1.0)
        bg.inputs["Strength"].default_value = 0.22

    cx, cz = float(center[0]), float(center[2])
    _plane(
        "CWG_Wall",
        [(cx + x, 0.02, cz + z) for x in (-6.0, 6.0) for z in (-4.0, 6.0)],
        (0.78, 0.76, 0.72),
        0.9,
    )
    _plane(
        "CWG_Floor",
        [(cx + x, y, 0.0) for x in (-6.0, 6.0) for y in (0.4, -6.0)],
        (0.50, 0.49, 0.47),
        0.82,
    )
    for name, energy, size, location, color in (
        ("CWG_Key", 280.0, 1.8, (cx + 1.4, -2.4, cz + 1.6), (1.00, 0.97, 0.93)),
        ("CWG_Fill", 90.0, 2.6, (cx - 1.8, -1.8, cz + 0.8), (0.90, 0.94, 1.00)),
        ("CWG_Rim", 70.0, 1.4, (cx - 0.6, 1.2, cz + 1.4), (0.94, 0.96, 1.00)),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy = energy
        light.size = size
        light.spread = 160.0
        light.color = color
        obj = bpy.data.objects.new(name, light)
        obj.location = location
        bpy.context.scene.collection.objects.link(obj)


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


def _frame(root, edge_to_edge: bool = False, margin_frac: float | None = None, fill: float = 0.92,
           view_span: float | None = None, crop_x: float | None = None, focus=None) -> None:
    """Look at the fixture from a slight three-quarter in front of the wall."""
    bpy.context.view_layer.update()
    bounds = _world_bounds(root)
    if bounds is None:
        raise RuntimeError("fixture has no mesh bounds")
    lo, hi = np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float)
    if focus is not None:
        lo, hi = np.array(focus[0], dtype=float), np.array(focus[1], dtype=float)
    if crop_x is not None:
        # A short length of a horizontal run, so the wires fill the tile.
        mid = (float(lo[0]) + float(hi[0])) * 0.5
        lo[0], hi[0] = mid - crop_x * 0.5, mid + crop_x * 0.5
    elif view_span is not None:
        # Crop to a length of the run so clips and joints stay readable,
        # while the run still leaves the top and bottom of the tile.
        lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
        mid = (float(lo[2]) + float(hi[2])) * 0.5
        lo[2], hi[2] = mid - view_span * 0.5, mid + view_span * 0.5
        context = max(0.06, view_span * 0.22 - (float(hi[0]) - float(lo[0])) * 0.5)
        lo[0] -= context
        hi[0] += context
    center = (lo + hi) * 0.5
    size = hi - lo
    # A hairline joint or a full-width trim needs wall around it. A socket,
    # window or door should fill the tile. A full-length run is framed on its
    # own ends, so it reads as reaching the wall edges.
    span = max(float(size[0]), float(size[2]))
    narrow = min(float(size[0]), float(size[2]))
    hairline = crop_x is None and not edge_to_edge and (narrow < 0.045 or span / max(narrow, 1e-3) > 8.0)
    frac = 0.04 if edge_to_edge else 0.28 if hairline else 0.14
    if margin_frac is not None:
        frac = margin_frac
    margin = np.maximum(size * frac, 0.0 if edge_to_edge or margin_frac == 0.0 else 0.03)
    if hairline and float(size[2]) < 0.2:
        margin[2] = max(float(margin[2]), 0.07)
    if hairline and float(size[0]) < 0.2:
        margin[0] = max(float(margin[0]), 0.07)
    frame_lo = lo - margin
    frame_hi = hi + margin
    frame_center = (frame_lo + frame_hi) * 0.5
    eye_dir = np.array([0.42, -1.0, 0.22])
    fov = 2.0 * math.atan(SENSOR_MM / (2.0 * LENS_MM))
    distance = _fit_distance(frame_center, _box_corners(frame_lo, frame_hi), eye_dir, fov, fill=fill)
    eye = eye_dir / np.linalg.norm(eye_dir) * distance + frame_center

    aim = bpy.data.objects.new("CWG_Aim", None)
    aim.location = tuple(float(v) for v in frame_center)
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

    radius = max(0.15, 0.5 * float(np.linalg.norm(size)))
    for light in bpy.data.objects:
        if light.type != "LIGHT":
            continue
        light.data.energy *= radius * radius
        light.data.size = max(0.45, radius * 1.8)
        constraint = light.constraints.new("TRACK_TO")
        constraint.target = aim
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _blank() -> Image.Image:
    return Image.new("RGB", (TILE, TILE), (236, 236, 238))


def _dress_window(seed: int) -> None:
    """Sill and head take a real inspection-wall material, displaced as geometry."""
    import numpy as np

    from cwg_infinigen.core.util.math import int_hash
    from syncrack3d.materials import enable_true_displacement
    from syncrack3d.material_diversity import apply_diverse_wall_material
    from syncrack3d.structures import bind_window_masonry

    mesh = bpy.data.meshes.new("CWG_SillHost")
    host = bpy.data.objects.new("CWG_SillHost", mesh)
    bpy.context.scene.collection.objects.link(host)
    np.random.seed(int_hash((seed, "window_sill")) % (2**32))
    apply_diverse_wall_material(host, seed, palette="unified")
    enable_true_displacement(host, 1.5)
    bind_window_masonry(host, 1.5)
    host.hide_render = True


def _one(kind: str, path: Path, seed: int = SEED) -> Image.Image:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_cycles(resolution=TILE, samples=SAMPLES, denoise=True, dicing_rate=1.0)
    width, height = WALL[kind]
    spec = SimpleNamespace(width_m=width, height_m=height, skirting=False)
    plans = sample_structure_plan(seed, spec, include_kinds=[kind])
    if not plans:
        raise RuntimeError(f"no placement for {kind}")
    if kind in SPAN_KINDS:
        # The grid shows every run standing vertical. The generator still draws either axis.
        from dataclasses import replace
        plans = [replace(plans[0], u_m=0.0, v_m=height * 0.5,
                         width_m=SPAN_SECTION[kind], height_m=height)]
    records = build_structures(plans, host_height_m=height)
    if not records:
        raise RuntimeError(f"no geometry for {kind}")
    if kind == "window":
        _dress_window(seed)
    root = bpy.data.objects[records[0]["object_name"]]
    bpy.context.view_layer.update()
    bounds = _world_bounds(root)
    if bounds is None:
        raise RuntimeError(f"{kind} has no mesh bounds")
    _studio((bounds[0] + bounds[1]) * 0.5)
    focus = None
    if kind == "utility_box":
        # Frame the cabinet. The feed still leaves the top of the tile.
        plan = plans[0]
        focus = (
            (plan.u_m - plan.width_m * 0.5, -0.2, plan.v_m - plan.height_m * 0.5),
            (plan.u_m + plan.width_m * 0.5, 0.02, plan.v_m + plan.height_m * 0.5 + 0.07),
        )
    _frame(root, edge_to_edge=kind in SPAN_KINDS,
           margin_frac=0.05 if kind == "cable" else 0.08 if kind == "utility_box" else 0.0 if kind in SPAN_KINDS else None,
           fill=0.92 if kind == "utility_box" else 0.9,
           view_span=0.62 if kind in SPAN_KINDS else None,
           crop_x=0.42 if kind == "cable" else None,
           focus=focus)
    _render(path)
    return Image.open(path).convert("RGB")


def _grid(images: list[Image.Image], names: list[str], path: Path, columns: int = COLUMNS) -> None:
    gap = 8
    label_h = 36
    rows = math.ceil(len(images) / columns)
    cell_h = TILE + label_h
    sheet = Image.new(
        "RGB",
        (columns * TILE + (columns + 1) * gap, rows * cell_h + (rows + 1) * gap),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(22)
    for index, image in enumerate(images):
        col, row = index % columns, index // columns
        x = gap + col * (TILE + gap)
        y = gap + row * (cell_h + gap)
        sheet.paste(image.resize((TILE, TILE), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + TILE / 2, y + TILE + label_h / 2), names[index], font=font,
                  fill=(30, 30, 30), anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


VARIED = (
    "utility_box", "window", "access_panel", "vent", "trunking", "wall_cable", "cable",
    "conduit", "downpipe", "plumbing", "panel_seam", "door",
)
VARIATION_SEEDS = (4, 18, 31)


def main() -> None:
    images = []
    for kind in KINDS:
        path = OUT / "tiles" / f"{kind}.png"
        print(f"rendering {kind}", flush=True)
        try:
            images.append(_one(kind, path))
        except Exception:
            traceback.print_exc()
            images.append(_blank())
    grid = OUT / "grid.png"
    _grid(images, [NAMES[kind] for kind in KINDS], grid)
    print(grid, flush=True)

    varied, names = [], []
    for kind in VARIED:
        for seed in VARIATION_SEEDS:
            path = OUT / "tiles" / f"{kind}_{seed}.png"
            print(f"rendering {kind} seed {seed}", flush=True)
            try:
                varied.append(_one(kind, path, seed))
            except Exception:
                traceback.print_exc()
                varied.append(_blank())
            names.append(f"{NAMES[kind]}  {seed}")
    sheet = OUT / "variation.png"
    _grid(varied, names, sheet, columns=len(VARIATION_SEEDS))
    print(sheet, flush=True)


if __name__ == "__main__":
    main()
