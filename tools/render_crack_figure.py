"""The paper's two crack field figures, one column wide (\\columnwidth, 600 dpi).

Each case is one wall from the generator's own material code with one crack from
the generator's own injection code, using the unified recipe's settings. Nothing
about the crack is redrawn: a spy on build_crack_field reads the path, radius and
profile from the same shader sockets that drive the displacement, and they are
written out as render passes next to the beauty image.

    # low resolution, to choose cracks, walls and windows
    PYTHONPATH=. .venv/bin/python tools/render_crack_figure.py render --preview

    # final passes -> outputs/crack_figure/final/<case>/passes.npz
    PYTHONPATH=. .venv/bin/python tools/render_crack_figure.py render

    # both figures, no Blender work -> outputs/crack_figure/crack_{shape,edges}.png
    PYTHONPATH=. .venv/bin/python tools/render_crack_figure.py plot

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "crack_figure"

TOKEN_SIZE_M = 0.36
CRACK_SEED = 309
CRAZING_SEED = 2060
EDGE_SEED = 762  # chosen from a contact sheet of seeds for its tall lips and heavy chipping on plaster and stucco

# The unified recipe's values (syncrack3d/recipes.py: COMMON and STYLES).
COMMON = dict(crack_fit_path=True, crack_label_mode="support", crack_path_mode="angular",
              crack_branch_mode="local_curved", layered_fraction=0.7, crack_edge_profile="strong")
STANDARD = dict(COMMON, crack_profile="standard", crack_width_scale=1.3, crack_depth_scale=0.5)
CRAZING = dict(COMMON, crack_profile="crazing", crack_width_scale=2.0, crack_depth_scale=2.0,
               crack_network_grime=(0.8, 1.0), crack_network_edge="organic", crack_network_extent=1.4)

# Case: wall category, wall seed, crack seed, crack settings, window (centre u, centre v, size; m),
# image size, light (incidence, azimuth, size, distance, power, world).
RAKING = (72.0, 0.0, 0.10, 1.2, 60.0, 0.02)
EDGE_WINDOW = (-0.036, 0.0, 0.24)  # most of the crack, with wall around it
CASES = {
    "ordinary": dict(category="concrete", wall_seed=12, crack_seed=CRACK_SEED, settings=STANDARD,
                     window=(0.0, 0.0, 0.32), size=1400, light=RAKING),
    "crazing": dict(category="plaster", wall_seed=12, crack_seed=CRAZING_SEED, settings=CRAZING,
                    window=(0.0, 0.0, 0.45), size=1400, light=RAKING),
    "crazing_zoom": dict(category="plaster", wall_seed=12, crack_seed=CRAZING_SEED, settings=CRAZING,
                         window=(0.0, 0.0, 0.11), size=1000, light=RAKING),
    "edge_plaster": dict(category="plaster", wall_seed=12, crack_seed=EDGE_SEED, settings=STANDARD,
                         window=EDGE_WINDOW, size=1200, light=RAKING),
    "edge_stucco": dict(category="stucco", wall_seed=12, crack_seed=EDGE_SEED, settings=STANDARD,
                        window=EDGE_WINDOW, size=1200, light=RAKING),
    "edge_brick": dict(category="brick", wall_seed=21, crack_seed=EDGE_SEED, settings=STANDARD,
                       window=EDGE_WINDOW, size=1200, light=RAKING),
}
PASSES = ("Image", "Position", "FigProf", "FigHeight", "FigLabel", "FigPath", "FigRadius", "FigAlong")


def _load_exr(path: Path) -> np.ndarray:
    import bpy

    image = bpy.data.images.load(str(path))
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(height, width, 4)[::-1]


def _build_wall(case: dict):
    """The wall as build_one makes it: its seed, its material stream, explicit normals."""
    import random

    import cwg_infinigen.core.constraints.example_solver.room.decorate  # noqa: F401
    from cwg_infinigen.core import surface  # noqa: F401
    from cwg_infinigen.core.util.math import int_hash
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.geometry import build_scene
    from syncrack3d.material_diversity import apply_diverse_wall_material
    from syncrack3d.spec import sample_scene_spec

    seed = case["wall_seed"]
    spec = sample_scene_spec(seed, module="flat", substrate="paint")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    clear_scene()
    build = build_scene(spec)
    wall = build.hosts[0]
    wall["syncrack3d_displacement_sign"] = 1
    np.random.seed(int_hash((seed, "cwg_wall_material")) % (2**32))
    generator, _, record = apply_diverse_wall_material(wall, seed, case["category"], palette="unified")
    return spec, wall, build, generator, record


def _inject(case: dict, spec, wall, dicing: float) -> dict:
    """Inject one crack with the real injector; return its parameters and sockets."""
    from mathutils import Vector

    import cwg_infinigen.assets.defect_injection as di
    from syncrack3d.materials import enable_true_displacement

    captured = {}
    real = di.build_crack_field

    def spy(nt, coord, p, u_idx="X", v_idx="Y", out=None):
        result = real(nt, coord, p, u_idx, v_idx, out)
        if out is not None:
            captured.update(p=p, geo=out, prof=result)
        return result

    di.build_crack_field = spy
    try:
        state = di._host_injection(wall)
        # Same frame _plane_frame gives a card on the front face: u along +x, v along +z.
        frame = (Vector((0.0, 0.0, spec.height_m * 0.5)), Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)))
        _, metrics = di._inject_crack(state, di._local_frame(wall, frame), case["crack_seed"], TOKEN_SIZE_M, 1,
                                      **case["settings"])
    finally:
        di.build_crack_field = real
    nt = state.material.node_tree
    tap = {
        "FigProf": captured["prof"],
        "FigHeight": di._math(nt, "MULTIPLY", state.height_terms[0], 1000.0),
        "FigLabel": state.class_fields[next(iter(state.class_fields))][0],
    }
    geo = captured["geo"]
    if geo:  # a crazing network has no single path
        tap["FigPath"] = di._math(nt, "MULTIPLY", geo["path"], 1000.0)
        tap["FigRadius"] = di._math(nt, "MULTIPLY", geo["w_eff"], 1000.0)
        tap["FigAlong"] = di._math(nt, "MULTIPLY", geo["v"], 1000.0)
    di.finalize_injections(dicing_rate=dicing, enable_subdivision=True)
    enable_true_displacement(wall, dicing)
    return {"p": captured["p"], "metrics": metrics, "tap": tap, "material": state.material}


def _capture(case: dict, spec, injected: dict, folder: Path, scale: float, samples: int, dicing: float) -> dict:
    """Orthographic, face-on render of the window with every pass written to EXR."""
    import bpy

    from syncrack3d.lighting import LightingPlan, LightSpec, _placement, apply_lighting
    from syncrack3d.render import configure_cycles
    from syncrack3d.scene_setup import _make_nishita_world

    size = max(64, int(case["size"] * scale))
    scene = configure_cycles(resolution=size, samples=samples, dicing_rate=dicing)
    scene.world = _make_nishita_world()
    u, v, window = case["window"]
    centre = (u, 0.0, spec.height_m * 0.5 + v)
    data = bpy.data.cameras.new("CWG_Figure_Cam")
    data.type = "ORTHO"
    data.ortho_scale = window
    data.clip_start = 0.01
    data.clip_end = 5.0
    camera = bpy.data.objects.new("CWG_Figure_Cam", data)
    camera.location = (u, -1.0, centre[2])
    camera.rotation_euler = (math.pi / 2, 0.0, 0.0)
    scene.collection.objects.link(camera)
    scene.camera = camera

    inc, azi, light_size, distance, power, world = case["light"]
    location = _placement(spec.width_m, spec.height_m, inc, azi, distance, centre)
    key = LightSpec("key", "figure_raking", 5500.0, power, light_size, inc, azi, tuple(location), target=centre)
    for light in apply_lighting(LightingPlan("figure_raking", world, (key,)), spec.height_m, scene.collection):
        light.visible_camera = False

    layer = bpy.context.view_layer
    layer.use_pass_position = True
    nt = injected["material"].node_tree
    for name, socket in injected["tap"].items():
        if name not in [a.name for a in layer.aovs]:
            aov = layer.aovs.add()
            aov.name, aov.type = name, "VALUE"
        node = nt.nodes.new("ShaderNodeOutputAOV")
        node.aov_name = name
        nt.links.new(socket, node.inputs["Value"])

    folder.mkdir(parents=True, exist_ok=True)
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    paths = {}
    for name in PASSES:
        if name not in layers.outputs or not layers.outputs[name].enabled:
            continue
        out = tree.nodes.new("CompositorNodeOutputFile")
        out.base_path = str(folder)
        out.format.file_format = "OPEN_EXR"
        out.format.color_depth = "32"
        out.format.exr_codec = "ZIP"
        out.file_slots[0].path = f"{name}_"
        tree.links.new(layers.outputs[name], out.inputs[0])
        paths[name] = folder / f"{name}_{scene.frame_current:04d}.exr"
    scene.render.filepath = str(folder / "render.png")
    bpy.ops.render.render(write_still=True)
    return {name: _load_exr(path) for name, path in paths.items()}


def _save(name: str, case: dict, spec, generator, material: dict, injected: dict, passes: dict, folder: Path) -> None:
    arrays = {"image": passes["Image"][..., :3], "surface_mm": -passes["Position"][..., 1] * 1000.0}
    for key, value in passes.items():
        if key.startswith("Fig"):
            arrays[key[3:].lower()] = value[..., 0]
    np.savez_compressed(folder / "passes.npz", **arrays)
    p = {k: v for k, v in injected["p"].items()}
    (folder / "case.json").write_text(json.dumps({
        "case": name, "category": case["category"], "generator": generator, "wall_seed": case["wall_seed"],
        "paint_band": material.get("paint_band"), "crack_seed": case["crack_seed"], "token_size_m": TOKEN_SIZE_M,
        "window": case["window"], "light": case["light"], "width_m": spec.width_m, "height_m": spec.height_m,
        "params": p, "metrics": injected["metrics"],
    }, indent=1, default=str) + "\n")
    print(f"WROTE {folder} h[{arrays['height'].min():.3f},{arrays['height'].max():.3f}] "
          f"surface[{arrays['surface_mm'].min():.3f},{arrays['surface_mm'].max():.3f}]", flush=True)


def render(names: list[str], scale: float, samples: int, dicing: float, tag: str, overrides: dict) -> None:
    for name in names:
        case = dict(CASES[name], **overrides)
        folder = OUT / tag / name
        spec, wall, _, generator, material = _build_wall(case)
        injected = _inject(case, spec, wall, dicing)
        passes = _capture(case, spec, injected, folder, scale, samples, dicing)
        _save(name, case, spec, generator, material, injected, passes, folder)


# ------------------------------------------------------------------------- figures
COLUMN_WIDTH_IN = 3.297  # \columnwidth of cas-dc: 238.25 pt
FIGURE_DPI = 600
FONT_PT = 6.5
INK = "#1A1A1A"
BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"
# Text, text position, then the feature it points at, in the crack's coordinates (mm).
# Text placed on its feature gets no line.
CALLOUTS = {"edge_plaster": (("Chip", (-122.0, 58.0), (-73.7, 57.9)), ("Lip", (32.0, -54.0), (-25.1, -54.1))),
            "edge_stucco": (("Lip", (-125.0, 52.0), (-77.7, 52.1)), ("Chip", (-125.0, -42.0), (-58.7, -42.7)),
                            ("Step", (40.0, 10.0), (40.0, 10.0))),
            "edge_brick": ()}
SHAPE_FIGURE = OUT / "crack_shape.png"
EDGE_FIGURE = OUT / "crack_edges.png"
LEFT_IN, GAP_IN, RIGHT_IN = 0.15, 0.045, 0.012


def _load(name: str, tag: str = "final") -> tuple[dict, dict]:
    folder = OUT / tag / name
    with np.load(folder / "passes.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, json.loads((folder / "case.json").read_text())


def _display(image: np.ndarray, share: float = 0.42) -> np.ndarray:
    """Linear render to sRGB, with the wall's median on one display level."""
    linear = np.clip(image * share / np.median(image.mean(axis=-1)), 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def _extent(case: dict) -> tuple[float, float, float, float]:
    """Image extent in mm of the crack's own coordinates."""
    u, v, size = (x * 1000.0 for x in case["window"])
    return (u - size / 2, u + size / 2, v - size / 2, v + size / 2)


def _tile(ax, image, case, **kwargs) -> None:
    ax.imshow(image, extent=_extent(case), interpolation="antialiased", **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#777777")


def _scale_bar(ax, case, length_mm: float, text: str, color=INK) -> None:
    from matplotlib import patheffects

    x0, x1, y0, y1 = _extent(case)
    x = x0 + 0.07 * (x1 - x0)
    y = y0 + 0.07 * (y1 - y0)
    halo = [patheffects.withStroke(linewidth=1.6, foreground="white")]
    ax.plot([x, x + length_mm], [y, y], color=color, lw=1.1, solid_capstyle="butt", path_effects=halo)
    ax.text(x + length_mm / 2, y + 0.025 * (y1 - y0), text, color=color, ha="center", va="bottom",
            fontsize=FONT_PT - 0.5, path_effects=halo)


def _along_track(arrays: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """eta, c(eta) and w(eta) in mm, per image row, read from the shader's own sockets."""
    return (np.median(arrays["along"], axis=1), np.median(arrays["path"], axis=1), np.median(arrays["radius"], axis=1))


def _section_row(arrays: dict, half_mm: float) -> int:
    """The widest image row where the groove is one clean V, away from tips and branches."""
    eta, centre, radius = _along_track(arrays)
    size_mm = abs(eta[0] - eta[-1]) / (len(eta) - 1)
    best, best_radius = -1, 0.0
    for row in range(len(eta)):
        if abs(eta[row]) > 0.7 * half_mm or radius[row] <= best_radius:
            continue
        rows = range(max(row - 3, 0), min(row + 4, len(eta)))
        if any(abs(centre[r] - centre[row]) > 0.5 * size_mm * 3 for r in rows):
            continue  # the path is running sideways here, so a horizontal cut would not be a clean V
        on = arrays["prof"][row] > 0.02
        if np.count_nonzero(np.diff(on.astype(int)) == 1) == 1:
            best, best_radius = row, radius[row]
    if best < 0:
        raise RuntimeError("no clean section row")
    return best


def _at(arrays: dict, name: str, target: tuple[float, float]) -> tuple[float, float]:
    """Height (mm) and profile at a point in the crack's coordinates (mm)."""
    x0, x1, y0, y1 = _extent(_load(name)[1])
    rows, cols = arrays["height"].shape
    row, col = int((y1 - target[1]) / (y1 - y0) * rows), int((target[0] - x0) / (x1 - x0) * cols)
    return float(arrays["height"][row, col]), float(arrays["prof"][row, col])


def _check(ordinary: dict, crazing: dict, edges: dict) -> None:
    """Fail loudly if the figure would claim something the passes do not show."""
    arrays, case = ordinary
    cut = arrays["prof"] > 0.8
    relief = arrays["surface_mm"] - np.median(arrays["surface_mm"][arrays["prof"] < 0.01])
    print(f"cut from height field {arrays['height'][cut].mean():.3f} mm, measured surface {relief[cut].mean():.3f} mm, "
          f"d_i {case['metrics']['crack_depth_m'] * 1000:.3f} mm", flush=True)
    if abs(arrays["height"][cut].mean() - relief[cut].mean()) > 0.35:
        raise RuntimeError("the measured surface does not follow the height field")
    if case["metrics"]["layered"] is not None:
        raise RuntimeError("the ordinary example should have a plain groove")
    modes = {name: (c["metrics"]["layered"] or {}).get("mode") for name, (_, c) in edges.items()}
    if modes != {"edge_plaster": "paint", "edge_stucco": "stucco", "edge_brick": None}:
        raise RuntimeError(f"unexpected edge modes: {modes}")
    if (crazing[1]["metrics"]["layered"] or {}).get("mode") != "paint":
        raise RuntimeError("the crazing example should have its own edge")
    for name, (other, _) in edges.items():
        for text, _, target in CALLOUTS[name]:
            value, groove = _at(other, name, target)
            if groove > 0.02 or (value > -0.1 if text == "Chip" else value < 0.3):
                raise RuntimeError(f"{name}: the {text} callout points at h {value:.2f} mm, P {groove:.2f}")
    base = edges["edge_brick"][0]
    for name, (other, _) in edges.items():
        path = np.abs(other["path"] - base["path"]).max()
        groove = np.abs(other["prof"] - base["prof"]).mean()
        print(f"{name}: path differs by {path:.4f} mm, profile by {groove:.5f} on average", flush=True)
        if path > 0.05 or groove > 0.01:
            raise RuntimeError(f"{name} is not the same crack")


def plot_shape(ordinary: dict, crazing: dict, zoom: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("profile", ["#FFFFFF", "#56A5D8", BLUE, "#0B1F3A"])
    tile = (COLUMN_WIDTH_IN - LEFT_IN - 2 * GAP_IN - RIGHT_IN) / 3
    title, row_gap, plot_h, bottom = 0.125, 0.015, 0.82, 0.255
    height = 2 * (title + tile) + row_gap + 0.27 + plot_h + bottom
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, height))

    def axes_at(x, top, w, h):
        return fig.add_axes((x / COLUMN_WIDTH_IN, 1 - (top + h) / height, w / COLUMN_WIDTH_IN, h / height))

    columns = [LEFT_IN + i * (tile + GAP_IN) for i in range(3)]
    row1, row2 = title, 2 * title + tile + row_gap
    arrays, case = ordinary
    d_i = case["metrics"]["crack_depth_m"] * 1000
    half = case["params"]["half_v"] * 1000
    knots = np.array(case["params"]["path_knots"]) * 1000
    eta, centre, radius = _along_track(arrays)

    # (a) the path over the crack's footprint
    ax = axes_at(columns[0], row1, tile, tile)
    _tile(ax, np.where(arrays["prof"] > 0.02, 0.76, 1.0), case, cmap="gray", vmin=0, vmax=1)
    ax.text(0.43, 0.33, "Branches", transform=ax.transAxes, ha="right", va="center", fontsize=FONT_PT - 0.5, color="#555555")
    inside = np.abs(eta) <= half
    ax.plot(centre[inside], eta[inside], color=INK, lw=0.5, label=r"$c(\eta)$", zorder=3)
    ax.plot(knots[:, 1], knots[:, 0], color=ORANGE, lw=0.7, ls=(0, (2.2, 1.4)), marker="o", ms=1.5, mew=0,
            label=r"$\ell(\eta)$", zorder=4)
    ax.legend(loc="upper right", frameon=False, handlelength=1.5, labelspacing=0.15, borderaxespad=0.1,
              borderpad=0.1, handletextpad=0.4, fontsize=FONT_PT)
    x0, x1, y0, y1 = _extent(case)
    span = x1 - x0
    origin = (x0 + 0.09 * span, y1 - 0.30 * span)
    for tip, text, offset in (((origin[0] + 0.17 * span, origin[1]), r"$\xi$", (0.035, 0)),
                              ((origin[0], origin[1] + 0.17 * span), r"$\eta$", (0, 0.04))):
        ax.annotate("", xy=tip, xytext=origin, arrowprops=dict(arrowstyle="-|>", lw=0.5, color=INK, mutation_scale=4,
                                                               shrinkA=0, shrinkB=0))
        ax.text(tip[0] + offset[0] * span, tip[1] + offset[1] * span, text, ha="center", va="center")
    ax.set_title(r"(a) Path", fontsize=FONT_PT, pad=1.5)

    # (b) the profile, (c) the render
    ax = axes_at(columns[1], row1, tile, tile)
    _tile(ax, arrays["prof"], case, cmap=cmap, vmin=0, vmax=1)
    _scale_bar(ax, case, 50.0, "5 cm")
    ax.text(0.04, 0.76, rf"$d_i$ = {d_i:.1f} mm", transform=ax.transAxes, ha="left", va="top", fontsize=FONT_PT - 0.5)
    key = ax.inset_axes((0.05, 0.89, 0.34, 0.045))
    key.imshow(np.linspace(0, 1, 128)[None, :], cmap=cmap, vmin=0, vmax=1, aspect="auto", extent=(0, 1, 0, 1))
    key.set_yticks([])
    key.set_xticks([0, 1], ["0", "1"], fontsize=FONT_PT - 1)
    key.tick_params(length=1, width=0.3, pad=0.5)
    for spine in key.spines.values():
        spine.set_linewidth(0.3)
    ax.set_title(r"(b) Profile $P$", fontsize=FONT_PT, pad=1.5)
    ax = axes_at(columns[2], row1, tile, tile)
    _tile(ax, _display(arrays["image"]), case)
    ax.set_title("(c) Render", fontsize=FONT_PT, pad=1.5)

    # (d) the whole network with the zoom marked, (e) and (f) the zoom
    full, full_case = crazing
    fine, fine_case = zoom
    d_c = full_case["metrics"]["crack_depth_m"] * 1000
    ax = axes_at(columns[0], row2, tile, tile)
    _tile(ax, full["prof"], full_case, cmap=cmap, vmin=0, vmax=1)
    _scale_bar(ax, full_case, 100.0, "10 cm")
    zx0, zx1, zy0, zy1 = _extent(fine_case)
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=ORANGE, lw=0.7))
    ax.set_title("(d) $P$, whole patch", fontsize=FONT_PT, pad=1.5)
    ax = axes_at(columns[1], row2, tile, tile)
    edge_cmap = LinearSegmentedColormap.from_list("edge", ["#0B1F3A", BLUE, "#FFFFFF", "#F2A65A", "#8A2F00"])
    _tile(ax, fine["height"], fine_case, cmap=edge_cmap, vmin=-0.6, vmax=0.6)
    _scale_bar(ax, fine_case, 20.0, "2 cm")
    key = ax.inset_axes((0.10, 0.89, 0.34, 0.045))
    key.imshow(np.linspace(-0.6, 0.6, 128)[None, :], cmap=edge_cmap, vmin=-0.6, vmax=0.6, aspect="auto",
               extent=(-0.6, 0.6, 0, 1))
    key.set_yticks([])
    key.set_xticks([-0.6, 0, 0.6], ["-0.6", "0", "+0.6"], fontsize=FONT_PT - 1,
                   path_effects=[patheffects.withStroke(linewidth=1.2, foreground="white")])
    key.tick_params(length=1, width=0.3, pad=0.5)
    for spine in key.spines.values():
        spine.set_linewidth(0.3)
    ax.text(0.96, 0.05, rf"$d_i$ = {d_c:.2f} mm", transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_PT - 0.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_title("(e) $h$ (mm), boxed area", fontsize=FONT_PT, pad=1.5)
    ax = axes_at(columns[2], row2, tile, tile)
    _tile(ax, _display(fine["image"]), fine_case)
    ax.set_title("(f) Render", fontsize=FONT_PT, pad=1.5)
    for top, text in ((row1, "Ordinary crack"), (row2, "Crazing")):
        fig.text(0.075 / COLUMN_WIDTH_IN, 1 - (top + tile / 2) / height, text, rotation=90, ha="center", va="center")

    # (g) the radius along the crack, (h) the cross-section
    plots_top = row2 + tile + 0.27
    plot_left, plot_gap, plot_right = 0.215, 0.30, 0.055
    wide = (COLUMN_WIDTH_IN - plot_left - plot_gap - plot_right) / 2
    ax = axes_at(plot_left, plots_top, wide, plot_h)
    order = np.argsort(eta[inside])
    ax.axvspan(0.80 * half, 0.98 * half, color="#E8E8E8", lw=0)
    ax.axvspan(-0.98 * half, -0.80 * half, color="#E8E8E8", lw=0)
    ax.plot(eta[inside][order], radius[inside][order], color=BLUE, lw=0.7)
    w_min, w_max = case["params"]["w_min"] * 1000, case["params"]["w_max"] * 1000
    for value, label in ((w_min, r"$w_{\min}$"), (w_max, r"$w_{\max}$")):
        ax.axhline(value, color="#555555", lw=0.4, ls=(0, (3, 2)))
        ax.text(0, value + 0.12, label, ha="center", va="bottom", fontsize=FONT_PT - 0.5, color="#333333")
    ax.text(0.89 * half, w_max + 0.12, r"$\tau_o$", ha="center", va="bottom", fontsize=FONT_PT - 0.5, color="#333333")
    ax.set(xlim=(-half, half), ylim=(0, w_max + 1.15), xlabel=r"Along the crack, $\eta$ (mm)",
           xticks=[-100, 0, 100], yticks=[0, 2, 4, 6])
    ax.set_ylabel(r"Radius $w$ (mm)", labelpad=1.5)
    ax.set_title(r"(g) Radius, with pointed ends", fontsize=FONT_PT, pad=1.5)

    row = _section_row(arrays, half)
    x = np.linspace(x0, x1, arrays["height"].shape[1], endpoint=False) + span / arrays["height"].shape[1] / 2 - centre[row]
    w = radius[row]
    near = np.abs(x) <= 1.6 * w
    relief = arrays["surface_mm"][row] - np.median(arrays["surface_mm"][row][(np.abs(x) > 1.6 * w) & (np.abs(x) < 25)])
    ax = axes_at(plot_left + wide + plot_gap, plots_top, wide, plot_h)
    ax.plot(x[near], arrays["height"][row][near], color=BLUE, lw=0.8, label=r"$h$", zorder=3)
    ax.plot(x[near][::2], relief[near][::2], ls="none", marker="o", ms=1.5, mew=0, color=ORANGE, label="Surface", zorder=4)
    ax.axhline(-d_i, color="#555555", lw=0.4, ls=(0, (3, 2)))
    ax.text(1.55 * w, -d_i + 0.08, r"$-d_i$", ha="right", va="bottom", fontsize=FONT_PT - 0.5, color="#333333")
    ax.annotate("", xy=(w, 0.28), xytext=(-w, 0.28), arrowprops=dict(arrowstyle="<|-|>", lw=0.5, color=INK,
                                                                     mutation_scale=4, shrinkA=0, shrinkB=0))
    ax.text(0, 0.34, r"$2w$", ha="center", va="bottom", fontsize=FONT_PT - 0.5)
    ax.set(xlim=(-1.6 * w, 1.6 * w), ylim=(-d_i - 0.45, 1.75), xlabel=r"Across the crack, $\xi-c$ (mm)",
           xticks=[-10, -5, 0, 5, 10], yticks=[-3, -2, -1, 0])
    ax.set_ylabel("Height (mm)", labelpad=1.5)
    ax.legend(loc="upper center", ncols=2, frameon=False, handlelength=1.0, columnspacing=1.0, borderaxespad=0.1,
              borderpad=0.1, fontsize=FONT_PT - 0.5, handletextpad=0.35)
    ax.set_title(f"(h) Section at $\\eta$ = {eta[row]:.0f} mm", fontsize=FONT_PT, pad=1.5)
    print(f"section at eta {eta[row]:.1f} mm, w {w:.2f} mm", flush=True)
    fig.savefig(SHAPE_FIGURE, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    print(SHAPE_FIGURE, flush=True)


def plot_edges(edges: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import FuncNorm, LinearSegmentedColormap

    stops = (-2.4, -0.5, 0.0, 1.0, 6.0)
    place = (0.0, 0.3, 0.5, 0.7, 1.0)
    norm = FuncNorm((lambda v: np.interp(v, stops, place), lambda t: np.interp(t, place, stops)), vmin=stops[0], vmax=stops[-1])
    cmap = LinearSegmentedColormap.from_list("cut", [(0.0, "#0B1F3A"), (0.3, BLUE), (0.5, "#FFFFFF"), (0.7, "#F2A65A"),
                                                     (1.0, "#8A2F00")])
    tile = (COLUMN_WIDTH_IN - LEFT_IN - 2 * GAP_IN - RIGHT_IN) / 3
    title, bar_h, bottom = 0.125, 0.065, 0.235
    height = title + 2 * tile + GAP_IN + 0.05 + bar_h + bottom
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, height))

    def axes_at(x, top, w, h):
        return fig.add_axes((x / COLUMN_WIDTH_IN, 1 - (top + h) / height, w / COLUMN_WIDTH_IN, h / height))

    columns = (("edge_plaster", "Plaster: torn film"), ("edge_stucco", "Stucco: broken coat"), ("edge_brick", "Brick: groove only"))
    for i, (name, label) in enumerate(columns):
        arrays, case = edges[name]
        x = LEFT_IN + i * (tile + GAP_IN)
        ax = axes_at(x, title, tile, tile)
        _tile(ax, _display(arrays["image"]), case)
        ax.set_title(label, fontsize=FONT_PT, pad=1.5)
        ax = axes_at(x, title + tile + GAP_IN, tile, tile)
        _tile(ax, arrays["height"], case, cmap=cmap, norm=norm)
        for text, at, target in CALLOUTS[name]:
            ax.annotate(text, xy=target, xytext=at, ha="center", va="center", fontsize=FONT_PT - 0.5,
                        arrowprops=None if at == target else dict(arrowstyle="-", lw=0.4, color=INK, shrinkA=1, shrinkB=0.5))
        if i == 2:
            _scale_bar(ax, case, 50.0, "5 cm")
    for top, text in ((title, "Render"), (title + tile + GAP_IN, "Height $h$")):
        fig.text(0.075 / COLUMN_WIDTH_IN, 1 - (top + tile / 2) / height, text, rotation=90, ha="center", va="center")
    bar = axes_at(LEFT_IN, title + 2 * tile + GAP_IN + 0.05, COLUMN_WIDTH_IN - LEFT_IN - 0.13, bar_h)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=bar, orientation="horizontal")
    colorbar.set_ticks([-2, -1, -0.5, 0, 1, 2, 3, 4, 5, 6])
    colorbar.set_ticklabels(["-2", "-1", "-0.5", "0", "+1", "+2", "+3", "+4", "+5", "+6"])
    colorbar.minorticks_off()
    colorbar.outline.set_linewidth(0.4)
    colorbar.ax.tick_params(length=1.5, width=0.4, pad=1)
    colorbar.set_label("Height $h$ (mm), nonlinear scale", labelpad=1)
    fig.savefig(EDGE_FIGURE, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    print(EDGE_FIGURE, flush=True)


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_PT, "mathtext.fontset": "dejavusans",
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#777777", "xtick.color": INK,
                         "ytick.color": INK, "axes.linewidth": 0.4, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
                         "xtick.major.size": 1.5, "ytick.major.size": 1.5, "xtick.major.pad": 1, "ytick.major.pad": 1,
                         "axes.labelpad": 1, "axes.spines.top": True, "axes.spines.right": True})
    ordinary, crazing = _load("ordinary"), _load("crazing")
    edges = {name: _load(name) for name in ("edge_plaster", "edge_stucco", "edge_brick")}
    _check(ordinary, crazing, edges)
    plot_shape(ordinary, crazing, _load("crazing_zoom"))
    plot_edges(edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    run = sub.add_parser("render")
    run.add_argument("--cases", default=",".join(CASES))
    run.add_argument("--preview", action="store_true")
    run.add_argument("--crack-seed", type=int)
    run.add_argument("--wall-seed", type=int)
    run.add_argument("--window", type=float, nargs=3)
    run.add_argument("--tag")
    sub.add_parser("plot")
    args = parser.parse_args()
    if args.stage == "plot":
        plot()
        return
    overrides = {}
    if args.crack_seed is not None:
        overrides["crack_seed"] = args.crack_seed
    if args.wall_seed is not None:
        overrides["wall_seed"] = args.wall_seed
    if args.window:
        overrides["window"] = tuple(args.window)
    scale, samples, dicing = (0.5, 32, 1.5) if args.preview else (1.0, 128, 1.0)
    tag = args.tag or ("preview" if args.preview else "final")
    render(args.cases.split(","), scale, samples, dicing, tag, overrides)


if __name__ == "__main__":
    main()
