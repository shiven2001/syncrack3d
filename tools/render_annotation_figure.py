"""The paper's annotation figure, one column wide (\\columnwidth, 600 dpi).

One crack token from one wall of the unified recipe, in a close-up and in a wide
context view of the same wall, at the recipe's own 640 x 640. Every label, box and
evidence value is read from what tools/prepare_defect_annotated_dataset.py wrote;
nothing is drawn by hand. Before plotting, the labels are derived again from the raw
passes with the paper's equations, and the figure stops if they disagree.

    # the wall -> outputs/annotation_figure/wall/
    PYTHONPATH=. .venv/bin/python tools/render_annotation_figure.py render

    # the real exporter -> outputs/annotation_figure/pack/
    PYTHONPATH=. .venv/bin/python tools/render_annotation_figure.py export

    # the check and the figure, no Blender -> outputs/annotation_figure/annotation_labels.png
    PYTHONPATH=. .venv/bin/python tools/render_annotation_figure.py plot

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "annotation_figure"
WALL = OUT / "wall"
PACK = OUT / "pack"
FIGURE = OUT / "annotation_labels.png"

# Chosen from the exporter's records of 22 walls (137 frames): neutral stucco, and a token that keeps
# both crack and ignored pixels in a sampled wide view, so the frame reaches the dataset's splits.
SEED = 709
CATEGORY = "stucco"
TOKEN = 3
CLOSE_RIG = 5
WIDE_RIG = 7
CLOSE_INSET_PX = 20
ZOOM_PX = 70

TAU_BG, TAU_P = 0.15, 0.5  # UNDER_RESOLUTION_IGNORE_FLOOR, AOV_COVERAGE_THRESHOLD
TAU_MU, TAU_V = 0.05, 0.01  # INSTANCE_ID_TOLERANCE, INSTANCE_VARIANCE_TOLERANCE
E_MIN = 64  # MIN_INSTANCE_LABEL_PX

COLUMN_WIDTH_IN = 3.297  # \columnwidth of cas-dc: 238.25 pt
FIGURE_DPI = 600
FONT_PT = 6.5
INK = "#1A1A1A"
TOKEN_COLOR = "#0072B2"
BAND_COLOR, BACKGROUND_COLOR = "#F0E442", "#ECECEC"


def _sample(rig: int) -> str:
    return f"seed_{SEED:05d}_rig{rig}_resample0_frame0001_camera_0"


def _load_view(rig: int) -> dict:
    """Everything the exporter wrote for one frame, plus the raw passes beside its render."""
    from PIL import Image

    from tools.prepare_defect_annotated_dataset import read_scalar_exr

    sample = _sample(rig)
    stem = f"{rig}_0_0001_0"
    frames = WALL / "frames" / f"seed_{SEED:05d}"

    def raw(name):
        return read_scalar_exr(frames / name / "camera_0" / f"{name}_{stem}.exr").astype(np.float64)

    with np.load(PACK / "instance_masks" / f"{sample}.npz") as data:
        masks = {key: data[key] for key in data.files}
    record = json.loads((PACK / "instances" / f"{sample}.json").read_text())
    boxes = json.loads((PACK / "bboxes" / f"{sample}.json").read_text())["boxes"]
    return {
        "rig": rig,
        "rgb": np.asarray(Image.open(PACK / "images" / f"{sample}.png").convert("RGB")),
        "crack": np.asarray(Image.open(PACK / "masks" / f"{sample}.png")) > 0,
        "ignore": np.asarray(Image.open(PACK / "ignore_masks" / f"{sample}.png")) > 0,
        "class_cov": np.asarray(Image.open(frames / "DefectCrack" / "camera_0" / f"DefectCrack_{stem}.png").convert("L"),
                                dtype=np.float64) / 255.0,
        "Q": np.clip(raw("DefectInstanceCoverage"), 0.0, 1.0),
        "M1": raw("DefectInstance"),
        "M2": raw("DefectInstanceSquared"),
        "hard": masks["instance_id"],
        "soft": masks["soft_instance_id"],
        "boxes": {b["instance_id"]: b for b in boxes},
        "instances": {r["instance_id"]: r for r in record["instances"]},
    }


def _check(view: dict) -> dict:
    """Derive the labels from the raw passes with the paper's equations; compare with the export."""
    Q, ids = view["Q"], set(view["instances"])
    eps = np.finfo(np.float32).eps
    mu = view["M1"] / np.maximum(Q, eps)
    var = np.maximum(view["M2"] / np.maximum(Q, eps) - mu * mu, 0.0)
    nearest = np.rint(mu)
    agree = np.isin(nearest.astype(np.int64), sorted(ids)) & (np.abs(mu - nearest) <= TAU_MU) & (var <= TAU_V)
    hard = np.where((Q >= TAU_P) & agree, nearest, 0).astype(np.int64)
    soft = np.where((Q > 0) & agree, nearest, 0).astype(np.int64)
    ambiguous = (Q >= TAU_P) & ~agree
    marginal = (Q >= TAU_BG) & (Q < TAU_P)

    if (mismatch := int((hard != view["hard"]).sum())):
        raise RuntimeError(f"rig {view['rig']}: hard owner differs from the export at {mismatch} pixels")
    if (mismatch := int((soft != view["soft"]).sum())):
        raise RuntimeError(f"rig {view['rig']}: soft owner differs from the export at {mismatch} pixels")
    if (missing := int(((ambiguous | marginal) & ~view["ignore"]).sum())):
        raise RuntimeError(f"rig {view['rig']}: {missing} ambiguous or marginal pixels are not ignored in the export")
    if (mismatch := int((view["crack"] & view["ignore"]).sum())):
        raise RuntimeError(f"rig {view['rig']}: {mismatch} pixels are both crack and ignored")
    stats = {}
    for token in sorted(view["instances"]):
        area, mass = int((hard == token).sum()), float(Q[soft == token].sum())
        if area == 0:
            continue  # the exporter records evidence only for tokens with a hard pixel
        evidence, recorded = max(area, mass), view["instances"][token]["coverage_mass_px"]
        if abs(evidence - recorded) > 0.05:
            raise RuntimeError(f"rig {view['rig']}: token {token} evidence {evidence:.2f}, exporter recorded {recorded}")
        stats[token] = {"area": area, "mass": mass, "evidence": evidence, "status": view["instances"][token]["label_status"]}
    demoted = view["ignore"] & (hard > 0)
    extra = int((view["ignore"] & ~(ambiguous | marginal | demoted)).sum())
    print(f"rig {view['rig']}: {stats} marginal={int(marginal.sum())} demoted={int(demoted.sum())} ambiguous={int(ambiguous.sum())} "
          f"ignored for another reason={extra}; the export agrees with the equations", flush=True)
    return stats


def _inset_row(view: dict) -> int:
    """The row the close-up's inset centres on: one crossing of the token, of median width, in the
    middle half of the crack and nearest its centre, so the inset shows neither a chip nor a tip."""
    ys, _ = np.where(view["hard"] == TOKEN)
    y_min, y_max = int(ys.min()), int(ys.max())
    centre, span = (y_min + y_max) / 2, y_max - y_min
    candidates = []
    for row in range(y_min, y_max + 1):
        on = (view["soft"][row] == TOKEN) & (view["Q"][row] >= 0.02)
        if abs(row - centre) <= span / 4 and np.count_nonzero(np.diff(on.astype(int)) == 1) == 1 and view["Q"][row][on].max() > 0.99:
            candidates.append((row, int(on.sum())))
    if not candidates:
        raise RuntimeError(f"rig {view['rig']}: no row crosses the token once")
    target = sorted(width for _, width in candidates)[len(candidates) // 2]
    return min((row for row, width in candidates if width == target), key=lambda r: abs(r - centre))


def _label_image(view: dict) -> np.ndarray:
    """The exported classes."""
    from matplotlib.colors import to_rgb

    image = np.empty((*view["crack"].shape, 3))
    image[:] = to_rgb(BACKGROUND_COLOR)
    image[view["ignore"]] = to_rgb(BAND_COLOR)
    image[view["crack"]] = to_rgb(TOKEN_COLOR)
    return image


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import patheffects
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_PT, "mathtext.fontset": "dejavusans",
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#777777", "xtick.color": INK,
                         "ytick.color": INK, "axes.linewidth": 0.4, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
                         "xtick.major.size": 1.5, "ytick.major.size": 1.5, "xtick.major.pad": 1, "ytick.major.pad": 1,
                         "axes.labelpad": 1})
    close, wide = _load_view(CLOSE_RIG), _load_view(WIDE_RIG)
    stats_close, stats_wide = _check(close), _check(wide)
    for name, view, stats in (("close-up", close, stats_close), ("wide view", wide, stats_wide)):
        if stats[TOKEN]["status"] != "visible" or stats[TOKEN]["evidence"] < E_MIN or TOKEN not in view["boxes"]:
            raise RuntimeError(f"the {name} should label and box the token")
    token_px = wide["soft"] == TOKEN
    crack_px, ignored_px = int((wide["crack"] & token_px).sum()), int((wide["ignore"] & token_px).sum())
    if not crack_px or not ignored_px:
        raise RuntimeError(f"the wide view should mix crack and ignored pixels: {crack_px} crack, {ignored_px} ignored")
    split = "".join(path.read_text() for path in (PACK / "splits").glob("*.txt"))
    print(f"wide view, token {TOKEN}: {crack_px} crack, {ignored_px} ignored; contrast {wide['instances'][TOKEN]['contrast_codes']}, "
          f"box {wide['boxes'][TOKEN]['width']}x{wide['boxes'][TOKEN]['height']}; in a split: {_sample(WIDE_RIG) in split}", flush=True)
    if _sample(WIDE_RIG) not in split:
        raise RuntimeError("the wide view should reach the dataset's splits")

    cmap = LinearSegmentedColormap.from_list("coverage", ["#FFFFFF", "#56A5D8", TOKEN_COLOR, "#0B1F3A"])
    halo = [patheffects.withStroke(linewidth=1.2, foreground="white")]

    left, gap, right = 0.15, 0.045, 0.012
    title, row_gap, legend_gap, legend_h = 0.125, 0.015, 0.03, 0.21
    tile = (COLUMN_WIDTH_IN - left - 2 * gap - right) / 3
    small = (COLUMN_WIDTH_IN - left - 3 * gap - right) / 4
    height = 2 * title + tile + row_gap + small + legend_gap + legend_h
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, height))

    def axes_at(x, top, w, h):
        return fig.add_axes((x / COLUMN_WIDTH_IN, 1 - (top + h) / height, w / COLUMN_WIDTH_IN, h / height))

    def style(ax, limits, text=None):
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[3], limits[2])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#777777")
        if text:
            ax.set_title(text, fontsize=FONT_PT, pad=1.5)

    # ---- close-up: render, class coverage, exported labels
    frame = (-0.5, 639.5, -0.5, 639.5)
    row1 = title
    box = close["boxes"][TOKEN]
    for i, (image, text, kwargs) in enumerate((
            (close["rgb"], "(a) RGB", {"interpolation": "antialiased"}),
            (close["class_cov"], r"(b) Class coverage $C_p$", {"cmap": cmap, "vmin": 0, "vmax": 1, "interpolation": "nearest"}),
            (_label_image(close), "(c) Label and box", {"interpolation": "nearest"}))):
        ax = axes_at(left + i * (tile + gap), row1, tile, tile)
        ax.imshow(image, **kwargs)
        style(ax, frame, text)
        if i == 1:
            key = ax.inset_axes((0.05, 0.89, 0.34, 0.045))
            key.imshow(np.linspace(0, 1, 128)[None, :], cmap=cmap, vmin=0, vmax=1, aspect="auto", extent=(0, 1, 0, 1))
            key.set_yticks([])
            key.set_xticks([0, 1], ["0", "1"], fontsize=FONT_PT - 1)
            key.tick_params(length=1, width=0.3, pad=0.5)
            for spine in key.spines.values():
                spine.set_linewidth(0.3)
        if i == 2:
            ax.add_patch(Rectangle((box["x_min"] - 0.5, box["y_min"] - 0.5), box["width"], box["height"], fill=False, ec=INK, lw=0.5))
            # The ignored rim is about a pixel wide here, so it needs its own magnification.
            row = _inset_row(close)
            mx = int(np.flatnonzero((close["soft"][row] == TOKEN) & (close["Q"][row] > 0)).mean())
            mh = CLOSE_INSET_PX // 2
            ax.add_patch(Rectangle((mx - mh - 0.5, row - mh - 0.5), 2 * mh, 2 * mh, fill=False, ec=INK, lw=0.4))
            inset = ax.inset_axes((0.02, 0.74, 0.22, 0.22))  # left of the box, clear of its edge
            inset.imshow(_label_image(close), interpolation="nearest")
            inset.set_xlim(mx - mh - 0.5, mx + mh - 0.5)
            inset.set_ylim(row + mh - 0.5, row - mh - 0.5)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color(INK)

    # ---- wide view: the frame, then one pixel-aligned zoom of the same token
    row2 = 2 * title + tile + row_gap
    ax = axes_at(left, row2, small, small)
    ax.imshow(wide["rgb"], interpolation="antialiased")
    style(ax, frame, "(d) RGB")
    ys, xs = np.where((wide["soft"] == TOKEN) & (wide["Q"] >= TAU_BG))
    zx = (int(xs.min()) + int(xs.max())) // 2 - ZOOM_PX // 2
    zy = (int(ys.min()) + int(ys.max())) // 2 - ZOOM_PX // 2
    ax.add_patch(Rectangle((zx - 0.5, zy - 0.5), ZOOM_PX, ZOOM_PX, fill=False, ec=TOKEN_COLOR, lw=0.8))
    limits = (zx - 0.5, zx + ZOOM_PX - 0.5, zy - 0.5, zy + ZOOM_PX - 0.5)
    for i, (image, text, kwargs) in enumerate((
            (wide["rgb"], "(e) RGB, zoom", {"interpolation": "nearest"}),
            (wide["Q"], r"(f) $Q_p$", {"cmap": cmap, "vmin": 0, "vmax": 1, "interpolation": "nearest"}),
            (_label_image(wide), "(g) Label, box", {"interpolation": "nearest"}))):
        ax = axes_at(left + (i + 1) * (small + gap), row2, small, small)
        ax.imshow(image, **kwargs)
        style(ax, limits, text)
        if i == 2:
            b = wide["boxes"][TOKEN]
            ax.add_patch(Rectangle((b["x_min"] - 0.5, b["y_min"] - 0.5), b["width"], b["height"], fill=False, ec=INK, lw=0.5))
    for top, h, text in ((row1, tile, "Close-up"), (row2, small, "Wide view")):
        fig.text(0.075 / COLUMN_WIDTH_IN, 1 - (top + h / 2) / height, text, rotation=90, ha="center", va="center")

    handles = [Patch(fc=TOKEN_COLOR, label="Crack"), Patch(fc=BACKGROUND_COLOR, ec="#999999", lw=0.3, label="Background"),
               Patch(fc=BAND_COLOR, label=r"Ignored: $\tau_{\mathrm{bg}}\leq Q_p<\tau_p$"),
               Line2D([], [], color=INK, lw=0.6, label="Exported box")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=FONT_PT - 0.5,
               bbox_to_anchor=(0.5, 1 - (row2 + small + legend_gap) / height), borderaxespad=0, borderpad=0, columnspacing=1.0,
               labelspacing=0.2, handlelength=1.2, handleheight=0.7, handletextpad=0.35)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    print(FIGURE, flush=True)


def render() -> None:
    subprocess.run([sys.executable, "-m", "syncrack3d.build_dataset", "--recipe", "unified", "--material-category", CATEGORY,
                    "--seeds", str(SEED), "--output", str(WALL)], cwd=REPO, check=True)


def export() -> None:
    subprocess.run([sys.executable, str(REPO / "tools" / "prepare_defect_annotated_dataset.py"), "-i", str(WALL / "frames"),
                    "-o", str(PACK), "--aov-masks", "on", "--instance-masks", "on", "--with-bboxes",
                    "--max-negative-fraction", "1", "--keep-unusable-closeups"], cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("render", "export", "plot"))
    {"render": render, "export": export, "plot": plot}[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
