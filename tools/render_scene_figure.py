"""The paper's scene-layout figure, one column wide (\\columnwidth, 600 dpi).

One wall built by the unified recipe, seen from inside the room. Only the camera
and the lights are the figure's own: a sampled rig can stand outside the room
shell, which leaves hard wedges of light on the side walls and floor and hides
the layout this figure exists to show. Outlines come from the render's object
index pass and from the room's own corners, never from drawing by hand.

    # candidates at low resolution -> outputs/scene_figure/previews/sheet.png
    .venv/bin/python tools/render_scene_figure.py preview --seeds 0-60

    # final render of FIGURE_SEED -> outputs/scene_figure/seed_<seed>/
    .venv/bin/python tools/render_scene_figure.py render

    # outlines and labels only, no Blender work -> outputs/scene_figure/Figure_2.png
    .venv/bin/python tools/render_scene_figure.py annotate

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "scene_figure"
FIGURE = OUT / "Figure_2.png"
FIGURE_SEED = 274

COLUMN_WIDTH_IN = 3.297  # \columnwidth of cas-dc: 238.25 pt
FIGURE_DPI = 600
FINAL_SIZE = (1980, 1485)  # the column width at FIGURE_DPI
FINAL_SAMPLES = 128
PREVIEW_SIZE = (640, 480)
PREVIEW_SAMPLES = 32
LARGE_PROPS = {"CellShelfFactory", "LargeShelfFactory", "FloorLampFactory", "TableDiningFactory"}
MIN_DEPTH_M = 3.2  # a shallower room needs a lens too wide to look natural
BACK_WALL_SHARE = 0.60  # share of the frame width the inspection wall fills
TARGET_MEDIAN = 0.25  # scene-linear median the metering aims for

# Object index pass values. Fixtures and props take a base plus their index.
ID_FLOOR, ID_CEILING, ID_LEFT, ID_RIGHT, ID_WALL, ID_SKIRTING = 1, 2, 3, 4, 5, 6
ID_FIXTURE, ID_PROP = 100, 200

KIND_LABEL = {
    "cable": "Cable", "wall_cable": "Wall cable", "conduit": "Conduit", "downpipe": "Downpipe",
    "utility_box": "Utility box", "window": "Window", "panel_seam": "Panel seam", "door": "Door",
    "vent": "Vent", "access_panel": "Access panel", "trunking": "Trunking", "plumbing": "Plumbing",
}
PROP_LABEL = {
    "CellShelfFactory": "Shelf", "LargeShelfFactory": "Shelf", "FloorLampFactory": "Floor lamp",
    "TableDiningFactory": "Table", "BottleFactory": "Bottle", "FoodBoxFactory": "Box",
}


# ------------------------------------------------------------------------- scene
def _build(seeds: str, output: Path, dicing: float, on_built) -> None:
    """Run the unified recipe without its renders, calling on_built after each wall."""
    import syncrack3d.build_dataset as bd

    real = bd.build_one

    def wrap(seed, args, out_root):
        record = real(seed, args, out_root)
        on_built(seed, record)
        return record

    bd.build_one = wrap
    sys.argv = [
        "build_dataset", "--recipe", "unified", "--seeds", seeds, "--output", str(output),
        "--no-render", "--dicing-rate", str(dicing), "--keep-going",
        "--closeups-per-defect", "1", "--context-view-fraction", "0",
    ]
    try:
        bd.main()
    finally:
        bd.build_one = real


def _room(record: dict) -> tuple[float, float, float]:
    spec = record["spec"]
    return float(spec["width_m"]), float(spec["height_m"]), float(record["room_shell"]["depth_m"])


def _camera_pose(record: dict) -> tuple[tuple, tuple, float]:
    """Location, aim and lens (36 mm sensor) of the view from inside the room."""
    width, height, depth = _room(record)
    distance = depth - 0.15
    lens = BACK_WALL_SHARE * distance * 36.0 / width
    return (width * 0.07, -distance, min(1.4, height * 0.5)), (0.0, 0.0, height * 0.47), lens


def _set_camera(record: dict) -> None:
    import bpy
    from mathutils import Vector

    location, aim, lens = _camera_pose(record)
    data = bpy.data.cameras.new("CWG_Figure_Cam")
    data.sensor_width = 36.0
    data.lens = lens
    data.clip_start = 0.05
    data.clip_end = 40.0
    camera = bpy.data.objects.new("CWG_Figure_Cam", data)
    camera.location = location
    camera.rotation_euler = (Vector(aim) - Vector(location)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()


def _set_lights(record: dict) -> None:
    """A ceiling panel and soft daylight from the room's open side, all inside the shell."""
    import bpy

    from syncrack3d.lighting import LightSpec, LightingPlan, apply_lighting

    width, height, depth = _room(record)
    floor_area = width * depth
    ceiling = (0.0, -depth * 0.42, height - 0.04)
    daylight = (-width * 0.30, -depth + 0.10, height * 0.62)
    plan = LightingPlan(
        regime="ceiling_diffuse",
        world_strength=0.10,
        lights=(
            LightSpec("key", "ceiling_diffuse", 4300.0, 12.0 * floor_area, 1.8, 0.0, 0.0,
                      ceiling, target=(0.0, -depth * 0.42, 0.0)),
            LightSpec("fill", "window_soft", 5600.0, 15.0 * floor_area, 1.2, 40.0, 180.0,
                      daylight, target=(width * 0.12, 0.0, height * 0.42)),
        ),
    )
    collection = bpy.context.scene.collection
    for light in apply_lighting(plan, height, collection):
        # Otherwise the disks show up in the lens and as mirror images in the window glass.
        light.visible_camera = False
        light.visible_glossy = False


def _tag_objects(record: dict) -> dict:
    """Give every visible object a pass index; return what each index stands for."""
    import bpy

    named = {"CWG_Shell_Floor": ID_FLOOR, "CWG_Shell_Ceiling": ID_CEILING,
             "CWG_Shell_Return_left": ID_LEFT, "CWG_Shell_Return_right": ID_RIGHT,
             "CWG_Skirting": ID_SKIRTING}
    for name, index in named.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.pass_index = index
    for obj in bpy.data.objects:
        if obj.get("cwg_host"):
            obj.pass_index = ID_WALL
        elif match := re.match(r"CWG_Structure_(\d+)_", obj.name):
            # By name: joined parts such as a window's head masonry lose their index property.
            obj.pass_index = ID_FIXTURE + int(match[1])
    fixtures = {ID_FIXTURE + int(s["index"]): s["kind"] for s in record["structures"]}
    props = {}
    for prop in record["props"]:
        root = bpy.data.objects[prop["object_name"]]
        index = ID_PROP + int(prop["index"])
        for part in (root, *root.children_recursive):
            part.pass_index = index
        props[index] = prop["factory"]
    untagged = [o.name for o in bpy.data.objects
                if o.type in ("MESH", "CURVE") and not o.pass_index and not o.hide_render]
    if untagged:
        print(f"untagged objects: {untagged}", flush=True)
    return {"fixtures": fixtures, "props": props}


def _load_exr(path: Path) -> np.ndarray:
    import bpy

    image = bpy.data.images.load(str(path))
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(height, width, 4)[::-1]


def _render_passes(folder: Path, size: tuple[int, int], samples: int, exposure: float) -> dict[str, Path]:
    """Render once; return the display PNG plus linear EXRs of the beauty and index passes."""
    import bpy

    scene = bpy.context.scene
    folder.mkdir(parents=True, exist_ok=True)
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.use_persistent_data = False
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = exposure
    bpy.context.view_layer.use_pass_object_index = True

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    paths = {}
    for name, socket in (("linear", "Image"), ("ids", "IndexOB")):
        out = tree.nodes.new("CompositorNodeOutputFile")
        out.base_path = str(folder)
        out.format.file_format = "OPEN_EXR"
        out.format.color_depth = "32"
        out.format.exr_codec = "ZIP"
        out.file_slots[0].path = f"{name}_"
        tree.links.new(layers.outputs[socket], out.inputs[0])
        paths[name] = folder / f"{name}_{scene.frame_current:04d}.exr"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.filepath = str(folder / "render.png")
    bpy.ops.render.render(write_still=True)
    paths["render"] = folder / "render.png"
    return paths


def _metered_exposure(folder: Path) -> float:
    """Exposure in stops that puts the room's median radiance on TARGET_MEDIAN."""
    paths = _render_passes(folder / "meter", (320, 240), 16, 0.0)
    linear = _load_exr(paths["linear"])[..., :3]
    median = float(np.median(linear @ np.array([0.2126, 0.7152, 0.0722])))
    return round(math.log2(TARGET_MEDIAN / max(median, 1e-4)), 3)


def _camera_matrix(record: dict, size: tuple[int, int]) -> list:
    """World to pixel projection (pixel centre convention, y down) of the figure camera."""
    import bpy

    camera = bpy.context.scene.camera
    world_to_cam = np.array(camera.matrix_world.inverted())
    lens, sensor = camera.data.lens, camera.data.sensor_width
    focal = lens / sensor * size[0]
    intrinsics = np.array([[focal, 0, size[0] / 2], [0, -focal, size[1] / 2], [0, 0, 1]])
    return (intrinsics @ np.diag([1, 1, -1, 1])[:3] @ world_to_cam).tolist()


# ------------------------------------------------------------------------- stages
def preview(seeds: str) -> None:
    from PIL import Image, ImageDraw

    folder = OUT / "previews"
    tiles = []

    def on_built(seed, record):
        props = [p["factory"] for p in record["props"]]
        width, height, depth = _room(record)
        kinds = [s["kind"] for s in record["structures"]]
        material = record["wall_material"]["diverse"]["category"]
        floor = (record.get("shell_finish") or {}).get("floor")
        print(f"seed {seed}: {material}, {floor}, depth {depth:.1f}, {kinds}, {props}", flush=True)
        if depth < MIN_DEPTH_M or not LARGE_PROPS.intersection(props):
            return
        _set_lights(record)
        _set_camera(record)
        _tag_objects(record)
        exposure = _metered_exposure(folder / f"seed_{seed:05d}")
        paths = _render_passes(folder / f"seed_{seed:05d}", PREVIEW_SIZE, PREVIEW_SAMPLES, exposure)
        text = f"{seed} {material} {floor} | {','.join(kinds)} | {','.join(p.removesuffix('Factory') for p in props)}"
        tiles.append((Image.open(paths["render"]).convert("RGB"), text))

    _build(seeds, folder / "scenes", 3.0, on_built)
    columns = 4
    rows = max(1, math.ceil(len(tiles) / columns))
    width, height = PREVIEW_SIZE
    sheet = Image.new("RGB", (columns * width, rows * (height + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, text) in enumerate(tiles):
        x, y = (index % columns) * width, (index // columns) * (height + 22)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + height + 4), text[:105], fill="black")
    sheet.save(folder / "sheet.png")
    print(folder / "sheet.png", flush=True)


def render(seed: int, size: tuple[int, int], samples: int, dicing: float) -> None:
    folder = OUT / f"seed_{seed:05d}"

    def on_built(_, record):
        _set_lights(record)
        _set_camera(record)
        tags = _tag_objects(record)
        exposure = _metered_exposure(folder)
        paths = _render_passes(folder, size, samples, exposure)
        ids = np.rint(_load_exr(paths["ids"])[..., 0]).astype(np.int16)
        np.save(folder / "ids.npy", ids)
        width, height, depth = _room(record)
        (folder / "layout.json").write_text(json.dumps({
            "seed": seed, "size": list(size), "exposure_ev": exposure,
            "width_m": width, "height_m": height, "depth_m": depth,
            "projection": _camera_matrix(record, size),
            "structures": [{k: s[k] for k in ("index", "kind", "u_m", "v_m", "width_m", "height_m")}
                           for s in record["structures"]],
            "fixtures": tags["fixtures"], "props": tags["props"],
            "wall_material": record["wall_material"]["diverse"]["category"],
            "floor": (record.get("shell_finish") or {}).get("floor"),
        }, indent=2) + "\n")
        print(f"visible ids: {sorted(int(v) for v in np.unique(ids))}", flush=True)

    _build(str(seed), folder / "scene", dicing, on_built)


# ------------------------------------------------------------------------- annotation
# Okabe-Ito hues, the colour-blind safe set the earlier version of this figure used.
WALL_COLOR = "#D55E00"
SIDE_COLOR = "#0072B2"
FLOOR_COLOR = "#E69F00"
CEILING_COLOR = "#7F7F7F"
FIXTURE_COLOR = "#CC3D94"
PROP_COLOR = "#F0E442"
FONT_PT = 6.5
HALO = "white"

# Positions are fractions of the image: (text position, leader target or None, colour, weight).
LABELS = (
    ("Ceiling", (0.64, 0.095), None, CEILING_COLOR, "normal"),
    ("Side wall", (0.105, 0.33), None, SIDE_COLOR, "normal"),
    ("Side wall", (0.905, 0.33), None, SIDE_COLOR, "normal"),
    ("Floor", (0.50, 0.925), None, FLOOR_COLOR, "normal"),
    ("Inspection wall", (0.40, 0.70), None, WALL_COLOR, "normal"),
    ("Access panel", (0.368, 0.115), (0.368, 0.203), FIXTURE_COLOR, "normal"),
    ("Window", (0.388, 0.605), (0.388, 0.548), FIXTURE_COLOR, "normal"),
    ("Utility box", (0.575, 0.475), (0.565, 0.405), FIXTURE_COLOR, "normal"),
    ("Plumbing", (0.60, 0.61), (0.684, 0.565), FIXTURE_COLOR, "normal"),
    ("Floor lamp", (0.105, 0.60), (0.232, 0.555), PROP_COLOR, "normal"),
)


def _project(projection: np.ndarray, points) -> np.ndarray:
    points = np.column_stack([np.asarray(points, dtype=float), np.ones(len(points))])
    image = points @ projection.T
    return image[:, :2] / image[:, 2:]


def _surfaces(layout: dict) -> list[tuple[np.ndarray, str, float]]:
    """(image polygon, colour, line width) of each room surface, from the room's corners."""
    width, height, depth = layout["width_m"], layout["height_m"], layout["depth_m"]
    projection = np.array(layout["projection"])
    x, y = width / 2, -(depth - 0.35)  # the near edge stays in front of the camera, out of frame
    corners = {
        "ceiling": [(-x, 0, height), (x, 0, height), (x, y, height), (-x, y, height)],
        "floor": [(-x, 0, 0), (x, 0, 0), (x, y, 0), (-x, y, 0)],
        "left": [(-x, 0, 0), (-x, 0, height), (-x, y, height), (-x, y, 0)],
        "right": [(x, 0, 0), (x, 0, height), (x, y, height), (x, y, 0)],
        "wall": [(-x, 0, 0), (x, 0, 0), (x, 0, height), (-x, 0, height)],
    }
    styles = {"ceiling": (CEILING_COLOR, 0.5), "floor": (FLOOR_COLOR, 0.55), "left": (SIDE_COLOR, 0.55),
              "right": (SIDE_COLOR, 0.55), "wall": (WALL_COLOR, 0.85)}
    return [(_project(projection, corners[name]), *styles[name]) for name in corners]


def _object_outlines(ids: np.ndarray, value: int) -> list[np.ndarray]:
    """Smooth outer outlines of one object, a little outside its visible pixels."""
    from scipy import ndimage
    from skimage import measure, morphology

    radius = max(2, round(ids.shape[1] * 0.002))
    mask = ndimage.binary_closing(ids == value, morphology.disk(radius * 2))
    mask = ndimage.binary_fill_holes(ndimage.binary_dilation(mask, morphology.disk(radius)))
    soft = ndimage.gaussian_filter(mask.astype(float), radius * 0.6)
    return [c[:, ::-1] for c in measure.find_contours(np.pad(soft, 1), 0.5) if len(c) > 40]


def annotate(seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import patheffects
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch, Polygon
    from matplotlib.path import Path as MplPath
    from matplotlib.transforms import Bbox
    from PIL import Image
    from scipy import ndimage

    folder = OUT / f"seed_{seed:05d}"
    layout = json.loads((folder / "layout.json").read_text())
    rgb = np.asarray(Image.open(folder / "render.png").convert("RGB")) / 255.0
    ids = np.load(folder / "ids.npy")
    height, width = ids.shape
    if rgb.shape[:2] != ids.shape:
        raise SystemExit(f"render {rgb.shape[:2]} and index pass {ids.shape} differ")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_PT})
    legend_in = 0.27
    image_in = COLUMN_WIDTH_IN * height / width
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, image_in + legend_in))
    ax = fig.add_axes((0, legend_in / (image_in + legend_in), 1, image_in / (image_in + legend_in)))
    ax.imshow(rgb, interpolation="none")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    # A tint on the visible pixels of the crack host, and on nothing else.
    tint = np.zeros((height, width, 4))
    tint[..., :3] = matplotlib.colors.to_rgb(WALL_COLOR)
    tint[..., 3] = 0.11 * ndimage.gaussian_filter((ids == ID_WALL).astype(float), 0.7)
    ax.imshow(tint, interpolation="none", zorder=2)

    # Inner strokes: a polygon clipped by itself draws only inside its own edge, so neighbouring
    # surfaces keep separate colours along a shared corner. The frame edge gets no stroke.
    frame = Bbox.from_extents(0, 0, width, height)
    surfaces = _surfaces(layout)
    for polygon, color, line_pt in surfaces:
        clip = MplPath(np.vstack([polygon, polygon[:1]]), closed=True).clip_to_bbox(frame, inside=True)
        clip_patch = Polygon(clip.vertices, closed=True, transform=ax.transData, fill=False, ec="none")
        ax.add_patch(clip_patch)
        for edge, stroke, z in ((HALO, line_pt + 0.3, 3), (color, line_pt, 4)):
            outline = Polygon(polygon, closed=True, fill=False, ec=edge, lw=2 * stroke, joinstyle="miter",
                              zorder=z, alpha=0.75 if edge == HALO else 1.0, clip_on=True)
            ax.add_patch(outline)
            outline.set_clip_path(clip_patch)

    # Fixtures and props stand in front of those edges: put their pixels back, then outline them.
    objects = ids >= ID_FIXTURE
    front = np.dstack([rgb, ndimage.gaussian_filter(objects.astype(float), 0.7)])
    ax.imshow(front, interpolation="none", zorder=5)
    for value in (int(v) for v in np.unique(ids[objects])):
        color = PROP_COLOR if value >= ID_PROP else FIXTURE_COLOR
        halo = "#3A3A3A" if value >= ID_PROP else HALO
        for contour in _object_outlines(ids, value):
            ax.plot(*(contour - 1).T, color=color, lw=0.6, zorder=6, solid_joinstyle="round",
                    path_effects=[patheffects.withStroke(linewidth=1.0, foreground=halo)])

    for text, at, target, color, weight in LABELS:
        xy = (at[0] * width, at[1] * height)
        box = dict(boxstyle="round,pad=0.2,rounding_size=0.25", fc=(1, 1, 1, 0.94), ec=color, lw=0.6)
        if target is None:
            ax.text(*xy, text, ha="center", va="center", bbox=box, zorder=9, weight=weight, color="#1A1A1A")
        else:
            ax.annotate(text, xy=(target[0] * width, target[1] * height), xytext=xy, ha="center", va="center",
                        bbox=box, zorder=9, weight=weight, color="#1A1A1A",
                        arrowprops=dict(arrowstyle="-", color="#1A1A1A", lw=0.55, shrinkA=0, shrinkB=0))

    handles = [
        Patch(fc=(*matplotlib.colors.to_rgb(WALL_COLOR), 0.25), ec=WALL_COLOR, lw=0.8, label="Inspection wall"),
        Patch(fc=SIDE_COLOR, label="Side walls"),
        Patch(fc=FLOOR_COLOR, label="Floor"),
        Patch(fc=CEILING_COLOR, label="Ceiling"),
        Patch(fc=FIXTURE_COLOR, label="Fixtures"),
        Patch(fc=PROP_COLOR, ec="#3A3A3A", lw=0.3, label="Foreground prop"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=FONT_PT,
               bbox_to_anchor=(0.5, 0.0), borderaxespad=0.15, borderpad=0.1, labelspacing=0.28,
               columnspacing=2.0, handlelength=1.5, handleheight=0.8, handletextpad=0.5)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    print(FIGURE, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("preview", "render", "annotate"))
    parser.add_argument("--seeds", default="0-40", help="preview: seeds to try")
    parser.add_argument("--seed", type=int, default=FIGURE_SEED, help="render, annotate: the figure's wall")
    parser.add_argument("--samples", type=int, default=FINAL_SAMPLES)
    parser.add_argument("--width", type=int, default=FINAL_SIZE[0])
    parser.add_argument("--height", type=int, default=FINAL_SIZE[1])
    parser.add_argument("--dicing-rate", type=float, default=1.5)
    args = parser.parse_args()
    if args.stage == "preview":
        preview(args.seeds)
    elif args.stage == "render":
        render(args.seed, (args.width, args.height), args.samples, args.dicing_rate)
    else:
        annotate(args.seed)


if __name__ == "__main__":
    main()
