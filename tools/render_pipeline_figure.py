"""The paper's pipeline figure, full text width (\\textwidth, 600 dpi).

One wall of the unified recipe followed through the five stages. Nothing is drawn by
hand: the room and the close-up are the wall's own renders (the room with the camera
and lights of the scene-layout figure, so the whole room shows), the placement and
capture diagrams are plotted from its scene.json, the crack field is read from the
shader sockets of the generator's own injector, and the label and box come from the
exporter.

It is the annotation figure's wall, and shares that wall's render and export.

    # the wall -> outputs/annotation_figure/wall/
    PYTHONPATH=. .venv/bin/python tools/render_pipeline_figure.py render

    # the real exporter -> outputs/annotation_figure/pack/
    PYTHONPATH=. .venv/bin/python tools/render_pipeline_figure.py export

    # the crack's field, face-on -> outputs/pipeline_figure/seed_00709/field/
    PYTHONPATH=. .venv/bin/python tools/render_pipeline_figure.py field

    # the whole room, from inside it -> outputs/pipeline_figure/seed_00709/room/
    PYTHONPATH=. .venv/bin/python tools/render_pipeline_figure.py room

    # the figure, no Blender -> outputs/pipeline_figure/gen_pipeline.png
    PYTHONPATH=. .venv/bin/python tools/render_pipeline_figure.py plot

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "pipeline_figure"
WALL = REPO / "outputs" / "annotation_figure" / "wall"
PACK = REPO / "outputs" / "annotation_figure" / "pack"
FIELD = OUT / "seed_00709" / "field"
ROOM = OUT / "seed_00709" / "room"
FIGURE = OUT / "gen_pipeline.png"

# Neutral stucco. Token 2 leaves the door's corner and has the material-aware edge: a lip and chips along it.
SEED = 709
CATEGORY = "stucco"
TOKEN = 2
JUNCTION_LABEL_INDEX = 2  # the vent's junction, which has room around it for the word
# Text positions, metres on the panel: (u, v) in stage 2, (x, y) in stage 4; a label is (index, dx, dy, ha).
TOKEN_LABEL = (0, 0.14, -0.3, "center")
JUNCTION_LABEL = (-0.3, 0.72)
CLOSEUPS_TEXT = (-1.8, -0.80)  # under the left close-ups
KEY_TEXT = (0.37, -2.00)  # above the light's line, where there is room
CONTEXT_LABEL = (6, 0.0, -0.60, "center")
PLAN_TOP_M = 0.56
ROOM_SIZE_PX = 1000
ROOM_SAMPLES = 128
CLOSE_RIG = 2
FIELD_WINDOW_M = 0.42  # rendered wide enough that the rotated crop has no empty corners
FIELD_CROP_M = 0.24

TEXT_WIDTH_IN = 6.842  # \textwidth of cas-dc: 494.5 pt
FIGURE_DPI = 600
FONT_PT = 6.5
SMALL_PT = 6.0
INK = "#1A1A1A"
MUTED = "#555555"
BORDER = "#8A8A8A"
ACCENT = "#0072B2"
WALL_COLOR, FIXTURE_COLOR, RUN_COLOR = "#D55E00", "#CC3D94", "#8C8C8C"
JUNCTION_COLOR = "#009E73"
IGNORE_COLOR, BACKGROUND_COLOR = "#F0E442", "#ECECEC"
LIGHT_COLOR = "#0072B2"
LARGE_FIXTURES = {"window", "door", "vent", "access_panel", "utility_box"}

STAGES = (
    (
        "Scene layout",
        "Full room with an inspection wall, side walls, ceiling and floor, plus fixtures and props.",
    ),
    (
        "Crack placement",
        "Mesh tokens on the wall, with fixture keep-out zones and junction anchors.",
    ),
    (
        "Crack field",
        "9 crack types, with geometric cut into the wall and material-aware generation.",
    ),
    (
        "Cameras and lighting",
        "Close-up views and wider-context views, with 5 different lighting regimes.",
    ),
    (
        "Rendering, annotation,\nand export",
        "Rendering using path tracing, with annotation labels after evidence and "
        "visibility checks. Export of RGB, mask, token maps, boxes and JSON.",
    ),
)


def _wrap(text: str, width_in: float, size_pt: float) -> list[str]:
    """Greedy line breaks, measured with the figure's own font."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    font = FontProperties(family="DejaVu Sans")
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if (
            TextPath((0, 0), trial, size=size_pt, prop=font).get_extents().width / 72.0
            > width_in
            and line
        ):
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line]


def _sample(rig: int) -> str:
    return f"seed_{SEED:05d}_rig{rig}_resample0_frame0001_camera_0"


def render() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "syncrack3d.build_dataset",
            "--recipe",
            "unified",
            "--material-category",
            CATEGORY,
            "--seeds",
            str(SEED),
            "--output",
            str(WALL),
        ],
        cwd=REPO,
        check=True,
    )


def export() -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "prepare_defect_annotated_dataset.py"),
            "-i",
            str(WALL / "frames"),
            "-o",
            str(PACK),
            "--aov-masks",
            "on",
            "--instance-masks",
            "on",
            "--with-bboxes",
            "--max-negative-fraction",
            "1",
            "--keep-unusable-closeups",
        ],
        cwd=REPO,
        check=True,
    )


def _instance() -> dict:
    instances = json.loads(
        (WALL / f"seed_{SEED:05d}" / "defect_instances.json").read_text()
    )["instances"]
    return next(i for i in instances if i["instance_id"] == TOKEN)


def field() -> None:
    """The token's own crack, by its own seed and size, face-on, with the real injector."""
    import tools.render_crack_figure as rcf

    inst = _instance()
    rcf.TOKEN_SIZE_M = max(inst["extent_m"])
    case = dict(
        category=CATEGORY,
        wall_seed=SEED,
        crack_seed=inst["seed"],
        settings=rcf.STANDARD,
        window=(0.0, 0.0, FIELD_WINDOW_M),
        size=1400,
        light=rcf.RAKING,
    )
    spec, wall, _, generator, material = rcf._build_wall(case)
    injected = rcf._inject(case, spec, wall, 1.0)
    passes = rcf._capture(case, spec, injected, FIELD, 1.0, 64, 1.0)
    rcf._save("token", case, spec, generator, material, injected, passes, FIELD)


def room() -> None:
    """The wall's own scene seen from inside the room; only the camera and the lights are the figure's."""
    import syncrack3d.build_dataset as bd
    import tools.render_scene_figure as rsf

    wall = json.loads((WALL / f"seed_{SEED:05d}" / "scene.json").read_text())
    real = bd.build_one

    def wrap(seed, args, out_root):
        record = real(seed, args, out_root)
        for key in (
            "spec",
            "structures",
            "defects_placed",
            "props",
            "room_shell",
            "wall_material",
        ):
            if json.dumps(record[key], sort_keys=True, default=str) != json.dumps(
                wall[key], sort_keys=True, default=str
            ):
                raise RuntimeError(f"the room view's {key} differs from the wall's")
        rsf._set_lights(record)
        rsf._set_camera(record)
        rsf._render_passes(
            ROOM,
            (ROOM_SIZE_PX, ROOM_SIZE_PX),
            ROOM_SAMPLES,
            rsf._metered_exposure(ROOM),
        )
        print(
            f"room view matches the wall: props {[prop['factory'] for prop in record['props']]}",
            flush=True,
        )
        return record

    bd.build_one = wrap
    sys.argv = [
        "build_dataset",
        "--recipe",
        "unified",
        "--material-category",
        CATEGORY,
        "--seeds",
        str(SEED),
        "--output",
        str(ROOM / "scene"),
        "--no-render",
        "--dicing-rate",
        "1.5",
    ]
    try:
        bd.main()
    finally:
        bd.build_one = real


def _check_field(case: dict, morphology: dict) -> None:
    """The field must be the rendered wall's crack, not one that merely looks like it."""
    params, metrics = case["params"], case["metrics"]
    for key, a, b in (
        ("w_min", params["w_min"], morphology["crack_width_min_m"]),
        ("w_max", params["w_max"], morphology["crack_width_max_m"]),
        ("depth", metrics["crack_depth_m"], morphology["crack_depth_m"]),
    ):
        if abs(a - b) > 2e-6:
            raise RuntimeError(f"the field's {key} is {a}, the wall's crack has {b}")
    knots_a, knots_b = np.array(params["path_knots"]), np.array(
        morphology["path_knots_m"]
    )
    if knots_a.shape != knots_b.shape or np.abs(knots_a - knots_b).max() > 1e-5:
        raise RuntimeError("the field's path is not the wall's crack")
    print(
        f"field matches the wall's crack: {morphology['style']}, {len(knots_a)} knots",
        flush=True,
    )


def _field_tile() -> tuple[np.ndarray, float]:
    """Height h (mm) of the crack, turned to its direction on the wall and cropped."""
    from scipy import ndimage

    with np.load(FIELD / "passes.npz") as data:
        height = data["height"]
    _check_field(
        json.loads((FIELD / "case.json").read_text()), _instance()["morphology"]
    )
    # The field's long axis is +90 deg in its own frame; a junction crack runs at its recorded angle on the wall.
    record = json.loads((WALL / f"seed_{SEED:05d}" / "scene.json").read_text())
    junction = next(j for j in record["junctions"] if j["index"] == TOKEN - 1)
    # Rotation fills outside the array with wall height, which the crop below leaves out anyway.
    turned = ndimage.rotate(
        height, junction["angle_deg"] - 90.0, reshape=False, order=1, mode="nearest"
    )
    size = height.shape[0]
    keep = int(size * FIELD_CROP_M / FIELD_WINDOW_M) // 2
    mid = size // 2
    return turned[mid - keep : mid + keep, mid - keep : mid + keep], float(height.min())


def _draw_wall_face(ax, record: dict, halo) -> None:
    """Stage 2: the wall seen face-on, from the wall's own record."""
    from matplotlib.patches import Polygon, Rectangle

    width, height = record["spec"]["width_m"], record["spec"]["height_m"]
    ax.add_patch(
        Rectangle(
            (-width / 2, 0),
            width,
            height,
            fc="#FBF1EA",
            ec=WALL_COLOR,
            lw=0.8,
            zorder=1,
        )
    )
    labels = {
        "conduit": "Conduit",
        "vent": "Vent",
        "downpipe": "Downpipe",
        "door": "Door",
    }
    corners = {}
    for item in record["structures"]:
        u, v, w, h = item["u_m"], item["v_m"], item["width_m"], item["height_m"]
        large = item["kind"] in LARGE_FIXTURES
        ax.add_patch(
            Rectangle(
                (u - w / 2, v - h / 2),
                w,
                h,
                fc="#F2D5E6" if large else "#E4E2E2",
                ec=FIXTURE_COLOR if large else RUN_COLOR,
                lw=0.5,
                zorder=2,
            )
        )
        if large:
            corners[item["kind"]] = [
                (u + a * w / 2, v + b * h / 2) for a in (-1, 1) for b in (-1, 1)
            ]
        # A run that spans the wall takes its name above the wall, clear of its own narrow strip.
        text = labels.get(item["kind"], item["kind"])
        style = dict(fontsize=SMALL_PT - 0.5, color=MUTED, zorder=5, path_effects=halo)
        if item["kind"] == "door":
            ax.text(u, v, text, ha="center", va="center", **style)
        elif item["kind"] == "vent":
            ax.text(u, v - h / 2 - 0.06, text, ha="center", va="top", **style)
        else:
            right = (
                u + w / 2 + 0.5 > width / 2 and h > 2 * w
            )  # a run near the wall's edge keeps its name on the panel
            ax.text(
                u + w / 2 if right else u,
                height + 0.05 if h > 2 * w else v + h / 2 + 0.06,
                text,
                ha="right" if right else "center",
                va="bottom",
                **style,
            )
    for token in record["defects_placed"]:
        u, v, size, spin = (
            token["u_m"],
            token["v_m"],
            token["size_m"],
            math.radians(token["spin_deg"]),
        )
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size / 2
        rotation = np.array(
            [[math.cos(spin), -math.sin(spin)], [math.sin(spin), math.cos(spin)]]
        )
        ax.add_patch(
            Polygon(
                square @ rotation.T + (u, v),
                closed=True,
                fc=(0, 0.447, 0.698, 0.18),
                ec=ACCENT,
                lw=0.5,
                zorder=3,
            )
        )
        ax.plot(u, v, "o", ms=2.0, mew=0.3, mec="white", color=ACCENT, zorder=4)
    for junction in record["junctions"]:
        token = next(
            t for t in record["defects_placed"] if t["index"] == junction["index"]
        )
        kind = junction["anchor"].split("_")[0]
        corner = min(
            corners[kind], key=lambda c: math.dist(c, (token["u_m"], token["v_m"]))
        )
        ax.plot(
            [token["u_m"], corner[0]],
            [token["v_m"], corner[1]],
            color=JUNCTION_COLOR,
            lw=0.7,
            zorder=4,
        )
        ax.plot(
            *corner, "D", ms=2.4, mew=0.3, mec="white", color=JUNCTION_COLOR, zorder=6
        )
        if junction["index"] == JUNCTION_LABEL_INDEX:
            ax.text(
                corner[0] + JUNCTION_LABEL[0],
                corner[1] + JUNCTION_LABEL[1],
                "Junction",
                ha="center",
                va="center",
                fontsize=SMALL_PT - 0.5,
                color=JUNCTION_COLOR,
                zorder=6,
                path_effects=halo,
            )
    index, dx, dy, ha = TOKEN_LABEL
    token = next(t for t in record["defects_placed"] if t["index"] == index)
    ax.text(
        token["u_m"] + dx,
        token["v_m"] + dy,
        "Token",
        ha=ha,
        va="center",
        fontsize=SMALL_PT - 0.5,
        color=ACCENT,
        zorder=6,
        path_effects=halo,
    )
    ax.set_xlim(-width / 2 - 0.08, width / 2 + 0.08)
    ax.set_ylim(height / 2 - 0.5 * (width + 0.16), height / 2 + 0.5 * (width + 0.16))


def _camera_aim(camera: dict) -> np.ndarray:
    """The pose equation run backwards: a = p - d [sin(psi) cos(theta), -cos(psi) cos(theta), sin(theta)]."""
    psi, theta = math.radians(camera["yaw_deg"]), math.radians(camera["pitch_deg"])
    return np.array(camera["location"]) - camera["distance_m"] * np.array(
        [
            math.sin(psi) * math.cos(theta),
            -math.cos(psi) * math.cos(theta),
            math.sin(theta),
        ]
    )


def _draw_capture(ax, record: dict, halo) -> None:
    """Stage 4: the wall's own camera poses and key light, in plan."""
    from matplotlib.patches import Rectangle

    width, depth = record["spec"]["width_m"], record["room_shell"]["depth_m"]
    x = width / 2
    ax.add_patch(
        Rectangle(
            (-x, -depth), width, depth, fc="#F4F2EF", ec="#BDBDBD", lw=0.4, zorder=1
        )
    )
    ax.plot([-x, x], [0, 0], color=WALL_COLOR, lw=1.6, solid_capstyle="butt", zorder=3)
    ax.text(
        -x + 0.08,
        0.09,
        "Inspection wall",
        ha="left",
        va="bottom",
        fontsize=SMALL_PT - 0.5,
        color=WALL_COLOR,
        zorder=8,
    )
    for item in record["structures"]:
        u, w = item["u_m"], item["width_m"]
        ax.plot(
            [u - w / 2, u + w / 2],
            [-0.07, -0.07],
            color=FIXTURE_COLOR if item["kind"] in LARGE_FIXTURES else RUN_COLOR,
            lw=1.2,
            solid_capstyle="butt",
            zorder=3,
        )
    key = next(
        light for light in record["lighting"]["lights"] if light["role"] == "key"
    )
    lx, ly = key["location"][:2]
    ax.plot(
        [lx, 0.0], [ly, 0.0], color=LIGHT_COLOR, lw=0.55, ls=(0, (2.5, 1.5)), zorder=4
    )
    ax.plot(lx, ly, "o", ms=5.5, mew=0.4, mec="white", color=LIGHT_COLOR, zorder=7)
    label = dict(
        va="center",
        fontsize=SMALL_PT - 0.5,
        linespacing=1.0,
        zorder=8,
        path_effects=halo,
    )
    ax.text(*KEY_TEXT, "Key light", ha="left", color=LIGHT_COLOR, **label)
    for camera in record["cameras"]:
        close = camera["role"] == "closeup"
        color = INK if close else "#7D7D7D"
        aim = _camera_aim(camera)
        ax.plot(
            [camera["location"][0], aim[0]],
            [camera["location"][1], aim[1]],
            color=color,
            lw=0.5,
            zorder=5,
        )
        ax.plot(
            *camera["location"][:2],
            "o",
            ms=2.6 if close else 3.6,
            mew=0.3,
            mec="white",
            color=color,
            zorder=6,
        )
    ax.text(*CLOSEUPS_TEXT, "Close-up\ncamera", ha="left", **label)
    index, dx, dy, ha = CONTEXT_LABEL
    context = next(camera for camera in record["cameras"] if camera["index"] == index)
    ax.text(
        context["location"][0] + dx,
        context["location"][1] + dy,
        "Context\ncamera",
        ha=ha,
        color="#5F5F5F",
        **label,
    )
    x0, x1 = -x - 0.12, max(x, lx) + 0.2
    span = max(x1 - x0, depth + 0.6)
    ax.set_xlim(x0, x0 + span)
    ax.set_ylim(PLAN_TOP_M - span, PLAN_TOP_M)


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import patheffects
    from matplotlib import pyplot as plt
    from matplotlib.colors import FuncNorm, LinearSegmentedColormap, to_rgb
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
    from PIL import Image

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_PT,
            "mathtext.fontset": "dejavusans",
            "text.color": INK,
        }
    )
    record = json.loads((WALL / f"seed_{SEED:05d}" / "scene.json").read_text())
    halo = [patheffects.withStroke(linewidth=1.0, foreground="white")]

    margin, gap = 0.012, 0.14
    wide_card = 1.68
    narrow_card = (TEXT_WIDTH_IN - 2 * margin - 4 * gap - wide_card) / 4
    pad = 0.05
    header, text_pt, line_in = 0.285, SMALL_PT - 0.2, 0.103
    tile = narrow_card - 2 * pad
    widths = [narrow_card] * 4 + [wide_card]
    wrapped_text = [
        _wrap(text, w - 2 * pad - 0.02, text_pt) for (_, text), w in zip(STAGES, widths)
    ]
    card_h = (
        header
        + tile
        + 0.035
        + max(len(lines) for lines in wrapped_text) * line_in
        + 0.03
    )
    height = 2 * margin + card_h
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, height))
    canvas = fig.add_axes((0, 0, 1, 1), zorder=0)
    canvas.set_xlim(0, TEXT_WIDTH_IN)
    canvas.set_ylim(height, 0)
    canvas.axis("off")

    def axes_at(x, top, w, h):
        ax = fig.add_axes(
            (x / TEXT_WIDTH_IN, 1 - (top + h) / height, w / TEXT_WIDTH_IN, h / height),
            zorder=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#777777")
        return ax

    lefts = [margin + i * (narrow_card + gap) for i in range(5)]
    for i, ((title, _), lines, x, w) in enumerate(
        zip(STAGES, wrapped_text, lefts, widths)
    ):
        canvas.add_patch(
            FancyBboxPatch(
                (x, margin),
                w,
                card_h,
                boxstyle="round,pad=0,rounding_size=0.045",
                fc="white",
                ec=BORDER,
                lw=0.6,
            )
        )
        canvas.add_patch(
            Circle((x + 0.105, margin + 0.142), 0.072, fc=ACCENT, ec="none", zorder=3)
        )
        canvas.text(
            x + 0.105,
            margin + 0.142,
            str(i + 1),
            color="white",
            ha="center",
            va="center_baseline",
            fontsize=FONT_PT,
            weight="bold",
            zorder=4,
        )
        wrapped = title.replace(" and ", " and\n") if i == 3 else title
        canvas.text(
            x + 0.205,
            margin + 0.142,
            wrapped,
            ha="left",
            va="center",
            fontsize=FONT_PT,
            weight="bold",
            linespacing=1.05,
        )
        for k, line in enumerate(lines):
            canvas.text(
                x + pad + 0.01,
                margin + header + tile + 0.035 + k * line_in,
                line,
                ha="left",
                va="top",
                fontsize=text_pt,
                color=INK,
            )
        if i < 4:
            canvas.add_patch(
                FancyArrowPatch(
                    (x + w + 0.012, margin + card_h / 2),
                    (x + w + gap - 0.012, margin + card_h / 2),
                    arrowstyle="-|>",
                    mutation_scale=6,
                    lw=0.8,
                    color=INK,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=3,
                )
            )

    top = margin + header
    # 1: the room
    ax = axes_at(lefts[0] + pad, top, tile, tile)
    ax.imshow(
        np.asarray(Image.open(ROOM / "render.png").convert("RGB")),
        interpolation="antialiased",
    )

    # 2: the wall's plan
    ax = axes_at(lefts[1] + pad, top, tile, tile)
    _draw_wall_face(ax, record, halo)
    ax.set_aspect("equal")

    # 3: the crack's height
    stops, place = (-2.4, -0.5, 0.0, 1.0, 6.0), (0.0, 0.3, 0.5, 0.7, 1.0)
    norm = FuncNorm(
        (lambda v: np.interp(v, stops, place), lambda t: np.interp(t, place, stops)),
        vmin=stops[0],
        vmax=stops[-1],
    )
    cmap = LinearSegmentedColormap.from_list(
        "cut",
        [
            (0.0, "#0B1F3A"),
            (0.3, ACCENT),
            (0.5, "#FFFFFF"),
            (0.7, "#F2A65A"),
            (1.0, "#8A2F00"),
        ],
    )
    crack_h, _ = _field_tile()
    ax = axes_at(lefts[2] + pad, top, tile, tile)
    ax.imshow(crack_h, cmap=cmap, norm=norm, interpolation="antialiased")
    ax.text(
        0.96,
        0.05,
        "Cut",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=SMALL_PT - 0.5,
        color="#0B4F80",
        weight="bold",
        path_effects=halo,
    )
    if crack_h.max() > 0.1:  # a plain groove has no raised edge to name
        ax.text(
            0.04,
            0.05,
            "Raised",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=SMALL_PT - 0.5,
            color="#B25A14",
            weight="bold",
            path_effects=halo,
        )

    # 4: cameras and the key light
    ax = axes_at(lefts[3] + pad, top, tile, tile)
    _draw_capture(ax, record, halo)
    ax.set_aspect("equal")

    # 5: RGB and its exported label
    sample = _sample(CLOSE_RIG)
    rgb = np.asarray(Image.open(PACK / "images" / f"{sample}.png").convert("RGB"))
    crack = np.asarray(Image.open(PACK / "masks" / f"{sample}.png")) > 0
    ignore = np.asarray(Image.open(PACK / "ignore_masks" / f"{sample}.png")) > 0
    box = next(
        b
        for b in json.loads((PACK / "bboxes" / f"{sample}.json").read_text())["boxes"]
        if b["instance_id"] == TOKEN
    )
    label = np.empty((*crack.shape, 3))
    label[:] = to_rgb(BACKGROUND_COLOR)
    label[ignore] = to_rgb(IGNORE_COLOR)
    label[crack] = to_rgb(ACCENT)
    inner = 0.05
    half = (wide_card - 2 * pad - inner) / 2
    ax = axes_at(lefts[4] + pad, top + (tile - half) / 2 - 0.05, half, half)
    ax.imshow(rgb, interpolation="antialiased")
    tag = dict(ha="left", va="bottom", fontsize=SMALL_PT - 0.5, path_effects=halo)
    ax.text(0.04, 0.04, "RGB", transform=ax.transAxes, **tag)
    ax = axes_at(
        lefts[4] + pad + half + inner, top + (tile - half) / 2 - 0.05, half, half
    )
    ax.imshow(label, interpolation="nearest")
    ax.add_patch(
        Rectangle(
            (box["x_min"] - 0.5, box["y_min"] - 0.5),
            box["width"],
            box["height"],
            fill=False,
            ec=INK,
            lw=0.5,
        )
    )
    ax.text(0.04, 0.02, "Label and box", transform=ax.transAxes, **tag)
    key_y = top + (tile - half) / 2 - 0.05 + half + 0.085
    key_x = lefts[4] + pad + 0.01
    for colour, text, step, edge in (
        (ACCENT, "Crack", 0.42, "none"),
        (IGNORE_COLOR, "Ignored", 0.53, "none"),
        (BACKGROUND_COLOR, "Background", 0.0, "#999999"),
    ):
        canvas.add_patch(
            Rectangle((key_x, key_y - 0.03), 0.09, 0.06, fc=colour, ec=edge, lw=0.3)
        )
        canvas.text(
            key_x + 0.115, key_y, text, ha="left", va="center", fontsize=SMALL_PT - 0.5
        )
        key_x += step

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    print(FIGURE, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=("render", "export", "field", "room", "plot"))
    {"render": render, "export": export, "field": field, "room": room, "plot": plot}[
        parser.parse_args().stage
    ]()


if __name__ == "__main__":
    main()
