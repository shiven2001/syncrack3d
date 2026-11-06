"""The paper's two simulator port figures, full page width (\\textwidth, 600 dpi).

Nothing is drawn by hand or generated. The map panels are the files export_room.py and
compose_room.py wrote, the mesh panels are the vertices and triangles read back from wall.usd,
and the Isaac Sim panels are the linear renders of isaac_port/render_relight_views.py. A lit pair
shares one exposure and one display transform, so the difference between its images is the lamp.
The script stops if the mesh does not sit on the stated grid or does not follow the class rule.

    # the room (Run from the repository root)
    python syncrack3d/build_dataset.py --recipe unified --seeds 910021 --output outputs/simulator_figure/scenes \
        --no-render --save-blend
    python isaac_port/export_room.py outputs/simulator_figure/scenes/seed_910021 outputs/simulator_figure/export/seed_910021
    OMNI_KIT_ACCEPT_EULA=YES isaac-sim-5/python isaac_port/compose_room.py outputs/simulator_figure/export/seed_910021

    # the lit views
    OMNI_KIT_ACCEPT_EULA=YES isaac-sim-5/python isaac_port/render_relight_views.py \
        outputs/simulator_figure/export/seed_910021 outputs/simulator_figure/relight/seed_910021 --crack 2 \
        --room-eye -1.237 -2.3 1.6 --room-target 0.3 0 1.1 --room-focal 13

    # -> outputs/simulator_figure/simulator_port[_column], simulator_transfer[_column, _no_ratio] (.png, .pdf), figure_facts.json
    python tools/render_simulator_figure.py

Needs numpy, opencv, matplotlib and pxr (usd-core).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

SEED = 910021
CRACK = 2  # scene_meta.json defects index: the same crack as render_relight_views.py --crack
ROOT = Path("outputs/simulator_figure")
EXPORT = ROOT / "export" / f"seed_{SEED}"
RELIGHT = ROOT / "relight" / f"seed_{SEED}"

TEXT_WIDTH_IN = 6.842  # \textwidth of cas-dc: 494.5 pt
COLUMN_WIDTH_IN = 3.297  # \columnwidth of cas-dc: 238.25 pt
FIGURE_DPI = 600
FONT_PT = 6.5
INK, MUTED = "#0b0b0b", "#52514e"
TAU_P = 0.5  # Table tab:simulator_settings
RELIEF_EXAGGERATION = 15
MAP_CROP_M = 0.44  # wall width of the map panels
TRIANGLE_CROP_M = 0.04  # wall width of the triangle panel
LIT_CROP = (130, 850, 260, 980)  # rows, columns of the 1024 x 1024 crack view: this crack only
STRIP_ASPECT = 0.42  # width / height of the tall crack panels

# Host surfaces are neutral; only the three classes that are objects carry a hue. A map has every
# pair of classes next to each other, and those three are the hues that stay apart for all pairs.
CLASS_COLORS = {"wall": "#d6d5d0", "floor": "#8b8a85", "ceiling": "#f4f3f0",
                "fixture": "#2a78d6", "crack": "#eb6834", "prop": "#1baf7a"}
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])


def to_srgb(lin):
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)


def imread_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def read_mesh():
    """World-space vertices and triangles of each class prim in wall.usd."""
    stage = Usd.Stage.Open(str(EXPORT / "wall.usd"))
    out = {}
    for name in ("wall", "crack"):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/Wall/{name}"))
        out[name] = (np.asarray(mesh.GetPointsAttr().Get(), np.float64),
                     np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3))
    return out


def grid_of_mesh(mesh, W, H):
    """The wall's height on its vertex grid (row 0 = top), checked against the stated spacing."""
    pts = np.concatenate([mesh["wall"][0], mesh["crack"][0]])
    xs, zs = np.unique(np.round(pts[:, 0], 4)), np.unique(np.round(pts[:, 2], 4))
    dx, dz = np.median(np.diff(xs)), np.median(np.diff(zs))
    gx, gz = int(round(W / dx)) + 1, int(round(H / dz)) + 1
    if abs(dx - 0.0025) > 2e-5 or abs(dz - 0.0025) > 2e-5:
        raise SystemExit(f"mesh spacing {dx}, {dz} is not 2.5 mm")
    height = np.full((gz, gx), np.nan)
    cols = np.round((pts[:, 0] + W / 2) / W * (gx - 1)).astype(int)
    rows = np.round((H - pts[:, 2]) / H * (gz - 1)).astype(int)
    height[rows, cols] = -pts[:, 1] * 1000.0
    if np.isnan(height).any():
        raise SystemExit("mesh does not fill its grid")
    return height, float(dx), float(dz)


def check_cell_rule(mesh, crack, ppm, W, H):
    """Every texel at or above the threshold must fall in a crack cell, and each crack cell must hold one."""
    pts, tris = mesh["crack"]
    centres = pts[tris].mean(1)
    gx, gz = int(round(W / 0.0025)), int(round(H / 0.0025))
    cells = {(int((H - z) / H * gz), int((x + W / 2) / W * gx)) for x, z in centres[:, [0, 2]]}
    rows, cols = np.nonzero(np.asarray(crack) >= TAU_P)
    want = set(zip(np.minimum(((rows + 0.5) / crack.shape[0] * gz).astype(int), gz - 1).tolist(),
                   np.minimum(((cols + 0.5) / crack.shape[1] * gx).astype(int), gx - 1).tolist()))
    if cells != want:
        raise SystemExit(f"crack cells {len(cells)} do not match the coverage rule {len(want)}: "
                         f"{len(cells - want)} extra, {len(want - cells)} missing")
    return len(cells)


def panel(ax, title, letter):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_linewidth(0.4)
        side.set_color(MUTED)
    ax.set_title(f"({letter}) {title}", fontsize=FONT_PT, color=INK, loc="left", pad=2.5)


def scale_bar(ax, length_px, label, xy=(0.05, 0.05), color="white"):
    """A bar of length_px data units at an axes-fraction position, with its length written above it."""
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = ax.get_ylim()
    bx, by = x0 + (x1 - x0) * xy[0], y0 + (y1 - y0) * xy[1]
    ax.plot([bx, bx + length_px], [by, by], color=color, lw=1.2, solid_capstyle="butt")
    ax.text(bx + length_px / 2, by + (y1 - y0) * 0.022, label, color=color, ha="center", va="bottom", fontsize=FONT_PT - 0.5)


def inset_colorbar(fig, ax, image, ticks, labels, shift=0.0, bar_width=0.31):
    """A short bar in the panel's lower right corner, on a white backing."""
    ax.add_patch(Rectangle((0.60 - shift, 0.015), 0.385 + shift, 0.125, transform=ax.transAxes, facecolor="white", alpha=0.92, edgecolor="none", zorder=3))
    bar = fig.colorbar(image, cax=ax.inset_axes([0.64 - shift, 0.095, bar_width, 0.03], zorder=4), orientation="horizontal", ticks=ticks)
    bar.ax.set_xticklabels(labels)
    bar.ax.tick_params(length=1.5, width=0.3, pad=1, labelsize=FONT_PT - 1, colors=INK)
    bar.outline.set_linewidth(0.3)


def relief_panel(ax, g, azimuth=-24.0, elevation=50.0):
    """The vertex grid as an axonometric drawing: one flat-shaded quad per mesh cell, far ones first."""
    shade = LightSource(azdeg=315, altdeg=25).hillshade(g, vert_exag=RELIEF_EXAGGERATION, dx=2.5, dy=2.5)
    xx, yy = np.meshgrid(np.arange(g.shape[1]) * 2.5, -np.arange(g.shape[0]) * 2.5)
    az, el = np.radians(azimuth), np.radians(elevation)
    rx = xx * np.cos(az) - yy * np.sin(az)
    ry = xx * np.sin(az) + yy * np.cos(az)
    z = g * RELIEF_EXAGGERATION
    sx, sy = rx, ry * np.sin(el) + z * np.cos(el)
    depth = ry * np.cos(el) - z * np.sin(el)
    corners = [(0, 0), (0, 1), (1, 1), (1, 0)]
    verts = np.stack([np.stack([sx[i:sx.shape[0] - 1 + i, j:sx.shape[1] - 1 + j], sy[i:sx.shape[0] - 1 + i, j:sx.shape[1] - 1 + j]], -1)
                      for i, j in corners], 2).reshape(-1, 4, 2)
    tone = shade[:-1, :-1].ravel()
    order = np.argsort(-depth[:-1, :-1].ravel())
    colors = np.clip(0.28 + 0.8 * tone[order], 0, 1)
    ax.add_collection(PolyCollection(verts[order], facecolors=np.stack([colors] * 3 + [np.ones_like(colors)], -1),
                                     edgecolors="none", linewidths=0, antialiaseds=False, rasterized=True))
    ax.set_xlim(sx.min(), sx.max())
    ax.set_ylim(sy.min(), sy.max())
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="datalim")


def port_figure(meta, facts, column=False, out_name="simulator_port"):
    info = json.loads((EXPORT / "wall" / "wall.json").read_text())
    W, H, ppm = info["width_m"], info["height_m"], info["px_per_m"]
    height = np.load(EXPORT / "wall" / "height_mm.npy", mmap_mode="r")
    crack = np.load(EXPORT / "wall" / "crack.npy", mmap_mode="r")
    color = imread_rgb(EXPORT / "textures" / "wall_color.png")
    normal = imread_rgb(EXPORT / "textures" / "wall_normal.png")

    cx, _, cz = meta["defects"][CRACK]["center_world"]
    s = int(round(MAP_CROP_M * ppm))
    r0 = int(round((H - cz) * ppm - s / 2))
    c0 = int(round((cx + W / 2) * ppm - s / 2))
    window = (r0, r0 + s), (c0, c0 + s)
    crop = lambda im: np.asarray(im[window[0][0]:window[0][1], window[1][0]:window[1][1]])  # noqa: E731
    h_crop = crop(height).astype(np.float32)
    h_median = float(np.median(h_crop))
    cov_crop = crop(crack).astype(np.float32)

    mesh = read_mesh()
    grid, dx, dz = grid_of_mesh(mesh, W, H)
    n_cells = check_cell_rule(mesh, crack, ppm, W, H)
    facts.update(wall_m=[round(W, 3), round(H, 3)], texel_mm=round(1000 / ppm, 3), mesh_spacing_mm=[round(dx * 1000, 3), round(dz * 1000, 3)],
                 triangles=dict(wall=len(mesh["wall"][1]), crack=len(mesh["crack"][1])), crack_cells=n_cells,
                 map_crop_m=MAP_CROP_M, triangle_crop_m=TRIANGLE_CROP_M, relief_exaggeration=RELIEF_EXAGGERATION)

    gap = 0.025  # of a panel's width, between columns
    width = COLUMN_WIDTH_IN if column else TEXT_WIDTH_IN
    side = width * 0.988 / (3 + 2 * gap)
    title, between, foot = 0.15, 0.17 if column else 0.20, 0.17 if column else 0.03  # the column version's legend goes below
    fig_h = title + side + between + side + foot
    fig = plt.figure(figsize=(width, fig_h))
    grid_spec = fig.add_gridspec(2, 3, left=0.006, right=0.994, top=1 - title / fig_h, bottom=foot / fig_h, wspace=gap, hspace=between / side)
    axes = [fig.add_subplot(grid_spec[i, j]) for i in range(2) for j in range(3)]

    ax = axes[0]
    ax.imshow(crop(color), interpolation="antialiased")
    panel(ax, "Base colour", "a")
    scale_bar(ax, 0.1 * ppm, "100 mm", xy=(0.09, 0.05) if column else (0.05, 0.05), color=INK)

    ax = axes[1]
    ax.imshow(crop(normal), interpolation="antialiased")
    panel(ax, "Normal map" if column else "Tangent-space normal", "b")

    ax = axes[2]
    span = 3.0
    im = ax.imshow(h_crop, cmap=DIVERGING, vmin=-span, vmax=span, interpolation="antialiased")
    panel(ax, "Height $h_{\\mathrm{room}}$", "c")
    inset_colorbar(fig, ax, im, [-span, 0, span], ["\u22123", "0", "3 mm"], **(dict(shift=0.33, bar_width=0.52) if column else dict(shift=0.05)))

    ax = axes[3]
    im = ax.imshow(cov_crop, cmap="gray_r", vmin=0, vmax=1, interpolation="antialiased")
    panel(ax, "Crack coverage", "d")
    inset_colorbar(fig, ax, im, [0, TAU_P, 1], ["0", f"{TAU_P}", "1"], **(dict(shift=0.33, bar_width=0.52) if column else {}))

    # (e) the exported vertices, shaded; nothing here comes from the textures
    ax = axes[4]
    k = 0.0025 * ppm
    gr0, gc0 = int(r0 / k), int(c0 / k)
    gs = int(round(MAP_CROP_M / 0.0025))
    relief_panel(ax, grid[gr0:gr0 + gs, gc0:gc0 + gs] - h_median)
    panel(ax, f"Relief mesh, \u00d7{RELIEF_EXAGGERATION}" if column else f"Relief mesh, height \u00d7{RELIEF_EXAGGERATION}", "e")

    # (f) the triangles round one crack texel, with their class and the texel rule's contour
    ax = axes[5]
    rows, cols = np.nonzero(cov_crop >= TAU_P)
    pick = np.argmin((rows - s / 2) ** 2 + (cols - s / 2) ** 2)
    tr, tc = rows[pick] + r0, cols[pick] + c0
    x_c, z_c = tc / ppm - W / 2, H - tr / ppm
    half = TRIANGLE_CROP_M / 2
    for name, face, edge, width in (("wall", CLASS_COLORS["wall"], "#ffffff", 0.3), ("crack", CLASS_COLORS["crack"], "#ffffff", 0.3)):
        pts, tris = mesh[name]
        centres = pts[tris].mean(1)
        keep = (abs(centres[:, 0] - x_c) < half + 0.003) & (abs(centres[:, 2] - z_c) < half + 0.003)
        verts = pts[tris[keep]][:, :, [0, 2]]
        ax.add_collection(PolyCollection(verts, facecolors=face, edgecolors=edge, linewidths=width))
    tex = int(round(half * ppm)) + 2
    local = np.asarray(crack[tr - tex:tr + tex, tc - tex:tc + tex])
    ex = (x_c - tex / ppm + 0.5 / ppm, x_c + tex / ppm - 0.5 / ppm, z_c + tex / ppm - 0.5 / ppm, z_c - tex / ppm + 0.5 / ppm)
    ax.contour(np.linspace(ex[0], ex[1], local.shape[1]), np.linspace(ex[2], ex[3], local.shape[0]), local, levels=[TAU_P],
               colors=INK, linewidths=0.55)
    ax.set_xlim(x_c - half, x_c + half)
    ax.set_ylim(z_c - half, z_c + half)
    ax.set_aspect("equal")
    panel(ax, "Mesh classes" if column else "Mesh classes at 2.5 mm", "f")
    scale_bar(ax, 0.01, "10 mm", xy=(0.05, 0.06), color=INK)
    handles = [Patch(color=CLASS_COLORS["crack"], label="crack triangles"), Patch(color=CLASS_COLORS["wall"], label="wall triangles"),
               plt.Line2D([], [], color=INK, lw=0.55, label=f"coverage = {TAU_P}")]
    style = dict(fontsize=FONT_PT - 1, edgecolor="none", handlelength=1.1, labelcolor=INK)
    if column:
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, borderaxespad=0.1, borderpad=0.1, columnspacing=1.2, **style)
    else:
        ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.92, borderpad=0.3, labelspacing=0.2, **style)

    fig.savefig(ROOT / f"{out_name}.png", dpi=FIGURE_DPI, facecolor="white")
    fig.savefig(ROOT / f"{out_name}.pdf", dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def lit_pair(view):
    """Left and right frames in sRGB under one exposure (the pair's median luminance to 0.18), and the linear frames."""
    left, right = (np.load(RELIGHT / f"{view}_{side}.npy") for side in ("left", "right"))
    lum = np.array([0.2126, 0.7152, 0.0722], np.float32)
    k = 0.18 / float(np.median(np.concatenate([(left @ lum).ravel(), (right @ lum).ravel()])))
    return np.clip(to_srgb(left * k), 0, 1), np.clip(to_srgb(right * k), 0, 1), left @ lum, right @ lum, k


def class_map(semantic, id_to_label):
    h, w = semantic.shape
    out = np.full((h, w, 3), 255, np.uint8)
    present = {}
    for key, label in id_to_label.items():
        if label in CLASS_COLORS:
            mask = semantic == int(key)
            out[mask] = tuple(int(CLASS_COLORS[label][i:i + 2], 16) for i in (1, 3, 5))
            present[label] = mask
    return out, present


def groove_profile(ratio, mask, r0, r1, c0, c1):
    """(L-R)/(L+R) across the groove: per image row, relative to the wall 20-40 px to either side of the crack
    class, then the median over rows where that class is one narrow run (the near-vertical stretches)."""
    rows = []
    for r in range(r0, r1):
        cols = np.nonzero(mask[r, c0:c1])[0]
        if len(cols) == 0 or cols.max() - cols.min() > 16:
            continue
        i = int(round(cols.mean())) + c0
        rows.append(ratio[r, i - 12:i + 13] - np.median(np.r_[ratio[r, i - 40:i - 20], ratio[r, i + 20:i + 40]]))
    profile = np.median(rows, axis=0)
    return dict(rows=len(rows), offsets_px=list(range(-12, 13)), median=np.round(profile, 3).tolist(),
                minimum=round(float(profile.min()), 3), minimum_at_px=int(profile.argmin()) - 12,
                maximum=round(float(profile.max()), 3), maximum_at_px=int(profile.argmax()) - 12)


def transfer_figure(views, facts, with_ratio, name):
    """One row: the room and its classes, then tall crops of the crack under one lamp."""
    room_l, room_r, *_, k_room = lit_pair("room")
    crack_l, crack_r, lum_l, lum_r, k_crack = lit_pair("crack")
    r0, r1, c0, c1 = LIT_CROP
    semantic = np.load(RELIGHT / "crack_semantic.npy")
    crack_ids = [int(k) for k, v in views["crack"]["id_to_label"].items() if v == "crack"]
    mask = np.isin(semantic, crack_ids)
    ratio = (lum_l - lum_r) / np.clip(lum_l + lum_r, 1e-6, None)
    facts["groove_profile"] = groove_profile(ratio, mask, r0, r1, c0, c1)

    # A tall strip of the crop, centred on the crack and holding all of it.
    cols = np.nonzero(mask[r0:r1, c0:c1].any(axis=0))[0] + c0
    strip = int(round((r1 - r0) * STRIP_ASPECT))
    if cols.max() - cols.min() + 20 > strip:
        raise SystemExit(f"crack is {cols.max() - cols.min()} px wide, the strip {strip}")
    s0 = int(np.clip((cols.min() + cols.max()) // 2 - strip // 2, c0, c1 - strip))
    box = (r0, r1, s0, s0 + strip)
    crop = lambda im: im[box[0]:box[1], box[2]:box[3]]  # noqa: E731

    n_strips = 3 if with_ratio else 2
    gap, title, foot = 0.04, 0.15, 0.03  # inches
    height = (TEXT_WIDTH_IN * 0.988 - (1 + n_strips) * gap) / (2 + n_strips * STRIP_ASPECT)
    fig_h = title + height + foot
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, fig_h))
    ratios = [1, 1] + [STRIP_ASPECT] * n_strips
    mean_w = height * (2 + n_strips * STRIP_ASPECT) / (2 + n_strips)
    grid = fig.add_gridspec(1, 2 + n_strips, width_ratios=ratios, left=0.006, right=0.994, top=1 - title / fig_h, bottom=foot / fig_h,
                            wspace=gap / mean_w)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(room_r, interpolation="antialiased")
    panel(ax, "Room under one lamp", "a")

    labels = views["room"]["id_to_label"]
    classes, present = class_map(np.load(RELIGHT / "room_semantic.npy"), labels)
    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(classes, interpolation="nearest")
    panel(ax, "Semantic classes", "b")
    order = ["crack", "wall", "floor", "ceiling", "fixture", "prop"]
    ax.legend(handles=[Patch(facecolor=CLASS_COLORS[n], edgecolor=MUTED, linewidth=0.3, label=n) for n in order], loc="lower left",
              ncol=2, fontsize=FONT_PT - 0.5, frameon=True, framealpha=0.95, edgecolor="none", borderpad=0.35, handlelength=1.0,
              columnspacing=0.8, labelspacing=0.25, handletextpad=0.4, labelcolor=INK)
    facts["room_crack_pixels"] = int(present["crack"].sum())
    facts["room_classes"] = sorted(present)

    mm_per_px = 1000 * json.loads((RELIGHT / "views.json").read_text())["crack_frame_m"] / 1024
    ax = fig.add_subplot(grid[0, 2])
    ax.imshow(crop(crack_r), interpolation="antialiased")
    panel(ax, "Crack", "c")
    scale_bar(ax, 50 / mm_per_px, "50 mm", xy=(0.08, 0.05))

    column, letter = 3, "d"
    if with_ratio:
        ax = fig.add_subplot(grid[0, column])
        im = ax.imshow(crop(ratio), cmap=DIVERGING, vmin=-0.8, vmax=0.8, interpolation="antialiased")
        panel(ax, "$(L-R)/(L+R)$", letter)
        ax.add_patch(Rectangle((0.03, 0.012), 0.94, 0.075, transform=ax.transAxes, facecolor="white", alpha=0.92, edgecolor="none", zorder=3))
        bar = fig.colorbar(im, cax=ax.inset_axes([0.15, 0.058, 0.7, 0.014], zorder=4), orientation="horizontal", ticks=[-0.8, 0, 0.8])
        bar.ax.tick_params(length=1.2, width=0.3, pad=0.8, labelsize=FONT_PT - 1.5, colors=INK)
        bar.outline.set_linewidth(0.3)
        column, letter = 4, "e"

    over = crack_r.copy()
    over[mask] = np.array([0xEB, 0x68, 0x34], np.float32) / 255
    ax = fig.add_subplot(grid[0, column])
    ax.imshow(crop(over), interpolation="nearest")
    panel(ax, "Crack class", letter)

    facts.update(crack_view_mm_per_px=round(mm_per_px, 3), crack_crop_m=[round(strip * mm_per_px / 1000, 3), round((r1 - r0) * mm_per_px / 1000, 3)],
                 crack_mask_pixels_in_crop=int(crop(mask).sum()),
                 clipped_fraction=dict(room=float(np.mean(room_r >= 1)), crack=float(np.mean(crop(crack_r) >= 1))),
                 exposure_scale=dict(room=float(k_room), crack=float(k_crack)))
    fig.savefig(ROOT / f"{name}.png", dpi=FIGURE_DPI, facecolor="white")
    fig.savefig(ROOT / f"{name}.pdf", dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def transfer_column_figure(views, name="simulator_transfer_column"):
    """Two by two, one column wide: the room and its classes, then one crack and its class."""
    _, room_r, *_ = lit_pair("room")
    _, crack_r, *_ = lit_pair("crack")
    r0, r1, c0, c1 = LIT_CROP
    gap, title, between, foot = 0.04, 0.15, 0.17, 0.02  # inches
    side = (COLUMN_WIDTH_IN * 0.985 - gap) / 2
    fig_h = title + side + between + side + foot
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, fig_h))
    grid = fig.add_gridspec(2, 2, left=0.0075, right=0.9925, top=1 - title / fig_h, bottom=foot / fig_h, wspace=gap / side, hspace=between / side)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(room_r, interpolation="antialiased")
    panel(ax, "Room under one lamp", "a")

    classes, _ = class_map(np.load(RELIGHT / "room_semantic.npy"), views["room"]["id_to_label"])
    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(classes, interpolation="nearest")
    panel(ax, "Semantic classes", "b")
    ax.legend(handles=[Patch(facecolor=CLASS_COLORS[n], edgecolor=MUTED, linewidth=0.3, label=n)
                       for n in ("crack", "wall", "floor", "ceiling", "fixture", "prop")],
              loc="lower left", ncol=2, fontsize=FONT_PT - 1, frameon=True, framealpha=0.95, edgecolor="none", borderpad=0.3, handlelength=0.9,
              columnspacing=0.7, labelspacing=0.2, handletextpad=0.35, labelcolor=INK, borderaxespad=0.25)

    mm_per_px = 1000 * json.loads((RELIGHT / "views.json").read_text())["crack_frame_m"] / 1024
    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(crack_r[r0:r1, c0:c1], interpolation="antialiased")
    panel(ax, "Crack under a raking lamp", "c")
    scale_bar(ax, 50 / mm_per_px, "50 mm", xy=(0.06, 0.05))

    semantic = np.load(RELIGHT / "crack_semantic.npy")
    mask = np.isin(semantic, [int(k) for k, v in views["crack"]["id_to_label"].items() if v == "crack"])
    over = crack_r.copy()
    over[mask] = np.array([0xEB, 0x68, 0x34], np.float32) / 255
    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(over[r0:r1, c0:c1], interpolation="nearest")
    panel(ax, "Crack class on (c)", "d")
    fig.savefig(ROOT / f"{name}.png", dpi=FIGURE_DPI, facecolor="white")
    fig.savefig(ROOT / f"{name}.pdf", dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_PT, "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "text.color": INK})
    meta = json.loads((EXPORT / "scene_meta.json").read_text())
    views = json.loads((RELIGHT / "views.json").read_text())
    facts = dict(seed=SEED, crack=CRACK, material=meta["test_wall"]["material"]["diverse"]["category"],
                 crack_size_m=meta["defects"][CRACK]["size_m"], room=meta["room"], spp=views["spp"], res=views["res"],
                 lamps={v: views["views"][v]["lamps"] for v in views["views"]},
                 ambient={v: views["views"][v]["ambient"] for v in views["views"]})
    port_figure(meta, facts)
    port_figure(meta, facts, column=True, out_name="simulator_port_column")
    transfer_figure(views["views"], facts, True, "simulator_transfer")
    transfer_figure(views["views"], facts, False, "simulator_transfer_no_ratio")
    transfer_column_figure(views["views"])
    (ROOT / "figure_facts.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
