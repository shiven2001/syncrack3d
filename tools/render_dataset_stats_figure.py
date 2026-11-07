"""Training-split distributions of a rendered dataset, one column wide, from its manifests and masks.

    python tools/render_dataset_stats_figure.py /mnt/nvme_storage/syncrack3d_v1.30.0_dataset outputs/dataset_stats_figure

(a) training walls per inspection-wall material. (b) mean crack width per cracked view, label
area divided by the length of its skeleton, by crack family. Mask values: 60 crack, 255 ignore.
Also writes stats.json with the numbers 3_results.tex quotes for the training split.
"""

import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHUNKS = {"train_a": (910000, 910499), "train_b": (910500, 910999)}
COLUMN_WIDTH_IN = 3.297
CRACK = 60
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BAR = "#2a78d6"
FAMILIES = {"standard": ("Ordinary", "#2a78d6"), "wide": ("Wide", "#eb6834"), "crazing": ("Crazing", "#1baf7a")}
MATERIALS = {"plaster": "Plaster", "stucco": "Stucco", "brick": "Brick", "marble": "Marble", "large_tile": "Large tile",
             "sandstone": "Sandstone", "cobble_wall": "Cobble", "tile": "Tile", "concrete": "Concrete",
             "stone_block": "Stone block"}
WIDTH_BINS = np.logspace(np.log10(0.8), np.log10(40), 36)  # pixels, log spaced so thin cracks stay readable

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "xtick.minor.width": 0.4, "xtick.minor.size": 1.5})


def view_stats(path):
    """(crack pixels, skeleton pixels, ignore pixels) of one exported view."""
    crack = np.asarray(Image.open(path)) == CRACK
    area = int(crack.sum())
    ignore_path = path.parent.parent / "ignore_masks" / path.name
    ignore = int((np.asarray(Image.open(ignore_path)) > 0).sum()) if ignore_path.exists() else 0
    return area, int(skeletonize(crack).sum()) if area else 0, ignore, crack.size


def style_axes(ax, grid_axis):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(which="both", colors=MUTED, labelcolor=INK)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def main(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    walls, jobs = [], []
    for chunk, (lo, hi) in CHUNKS.items():
        for seed in range(lo, hi + 1):
            wall = json.loads((root / chunk / "walls" / f"seed_{seed}" / "scene.json").read_text())
            walls.append(wall)
            for path in sorted((root / chunk / "pack" / "masks").glob(f"seed_{seed}_*.png")):
                jobs.append((path, wall))
    with Pool(16) as pool:
        measured = pool.map(view_stats, [p for p, _ in jobs], chunksize=64)

    material = Counter(w["wall_material"]["diverse"]["category"] for w in walls)
    style = Counter(w["appearance_controls"]["crack_style"] for w in walls)
    widths = {family: [] for family in FAMILIES}
    fractions, lengths, mean_widths = [], [], []
    roles, lens, cracked_context, ignore_total, crack_total = Counter(), Counter(), 0, 0, 0
    for (path, wall), (area, skeleton, ignore, size) in zip(jobs, measured):
        rig = int(path.stem.split("_rig")[1].split("_")[0])
        camera = wall["cameras"][rig]
        roles[camera["role"]] += 1
        lens[round(camera["lens_equiv_mm"])] += 1
        ignore_total += ignore
        crack_total += area
        if area:
            width = area / max(skeleton, 1)
            widths[wall["appearance_controls"]["crack_style"]].append(width)
            fractions.append(area / size), lengths.append(skeleton), mean_widths.append(width)
            cracked_context += camera["role"] == "context"

    span = [5, 50, 95]
    stats = dict(
        walls=len(walls), views=len(jobs), roles=roles, lens_mm=dict(sorted(lens.items())), material=material, style=style,
        cracked_views=len(mean_widths), cracked_context=cracked_context,
        cracked_views_by_family={k: len(v) for k, v in widths.items()},
        tokens_injected=sum(w["defects_injected"] for w in walls), fixtures=sum(len(w["structures"]) for w in walls),
        props=sum(len(w["props"]) for w in walls),
        crack_percent_p5_p50_p95=np.round(np.percentile(fractions, span) * 100, 3).tolist(),
        centreline_px_p5_p50_p95=np.round(np.percentile(lengths, span), 1).tolist(),
        mean_width_px_p5_p50_p95=np.round(np.percentile(mean_widths, span), 2).tolist(),
        mean_width_px_median_by_family={k: round(float(np.median(v)), 2) for k, v in widths.items()},
        ignore_percent_of_crack_px=round(100 * ignore_total / crack_total, 2),
        views_outside_width_bins=int(sum((np.array(v) < WIDTH_BINS[0]).sum() + (np.array(v) > WIDTH_BINS[-1]).sum()
                                         for v in widths.values())),
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))

    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 3.25), dpi=600)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.18, 1), hspace=0.62, left=0.195, right=0.97, top=0.955, bottom=0.095)

    ax = fig.add_subplot(grid[0])
    items = sorted(material.items(), key=lambda kv: kv[1])
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], height=0.72, color=BAR, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, [MATERIALS[k] for k, _ in items])
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_xlabel("Training rooms", color=MUTED, labelpad=1.5)
    ax.set_title("(a) Inspection-wall material", loc="left", fontsize=7, color=INK, pad=3)
    style_axes(ax, "x")
    ax.tick_params(axis="y", length=0)

    ax = fig.add_subplot(grid[1])
    for family, (name, colour) in FAMILIES.items():
        counts, _ = np.histogram(widths[family], bins=WIDTH_BINS)
        ax.stairs(counts / len(widths[family]), WIDTH_BINS, color=colour, lw=1.1,
                  label=f"{name} ({len(widths[family]):,} views)")
    ax.set_xscale("log")
    ax.set_xlim(WIDTH_BINS[0], WIDTH_BINS[-1])
    ax.set_xticks([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, 0.17)  # headroom so the legend clears the curves
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.set_xlabel("Mean crack width per view (px, area / centreline length)", color=MUTED, labelpad=1.5)
    ax.set_ylabel("Share of cracked views", color=MUTED, labelpad=2)
    ax.set_title("(b) Crack width in the image, by crack family", loc="left", fontsize=7, color=INK, pad=3)
    ax.legend(frameon=False, fontsize=6, handlelength=1.4, loc="upper right", borderaxespad=0.1, labelspacing=0.3)
    style_axes(ax, "y")

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"Figure_dataset_stats.{suffix}", dpi=600)


if __name__ == "__main__":
    main(*sys.argv[1:3])
