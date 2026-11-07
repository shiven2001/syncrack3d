"""Contact sheet of dataset views: one row per crack family, close-ups beside context views.

    python tools/render_contact_sheet.py /mnt/nvme_storage/syncrack3d_v1.30.0_dataset outputs/contact_sheet_figure

The views are chosen by eye for illustration, from every split, so the sheet says what the dataset
can look like, not what a typical view looks like. Each tile is the released RGB with its class mask
inset (white crack, grey ignore). Figure_contact_sheet.json records each tile's source view.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT_WIDTH_IN, DPI = 6.842, 400
ROWS = (("standard", "Ordinary"), ("wide", "Wide"), ("crazing", "Crazing"))
GROUPS = (("closeup", "Close-up"), ("context", "Context"))
CRACK = 60
# (seed, view) per crack family and view type
CHOSEN = {
    ("standard", "closeup"): ((900067, 3), (910278, 1), (910085, 2)),
    ("standard", "context"): ((910925, 6), (900118, 7), (910456, 7)),
    ("wide", "closeup"): ((910866, 1), (900275, 0), (910826, 3)),
    ("wide", "context"): ((910811, 2), (900247, 4), (900210, 2)),
    ("crazing", "closeup"): ((910613, 4), (910493, 2), (910807, 1)),
    ("crazing", "context"): ((900057, 6), (910345, 6), (910350, 6)),
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 5.5, "pdf.fonttype": 42})


def chunk_of(seed):
    if 900000 <= seed <= 900299:
        return "test"
    if 901000 <= seed <= 901199:
        return "val"
    if 910000 <= seed <= 910499:
        return "train_a"
    if 910500 <= seed <= 910999:
        return "train_b"
    raise ValueError(f"seed {seed} is outside the v1.30 dataset")


def tile(pack, name):
    rgb = np.asarray(Image.open(pack / "images" / name).convert("RGB")).astype(float) / 255
    crack = np.asarray(Image.open(pack / "masks" / name)) == CRACK
    ignore_path = pack / "ignore_masks" / name
    ignore = np.asarray(Image.open(ignore_path)) > 0 if ignore_path.exists() else np.zeros_like(crack)
    inset = np.full(crack.shape + (3,), 0.12)
    inset[ignore & ~crack] = 0.5
    inset[crack] = 1.0
    size, border = int(rgb.shape[0] * 0.34), 4
    small = np.asarray(Image.fromarray((inset * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)) / 255
    out = rgb.copy()
    out[-size - 2 * border:, -size - 2 * border:] = 1.0
    out[-size - border:-border, -size - border:-border] = small
    return out, float(crack.mean())


def main(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 3.46), dpi=DPI)
    outer = fig.add_gridspec(1, len(GROUPS), left=0.03, right=0.999, top=0.962, bottom=0.002, wspace=0.03)
    record = {}
    for g, (role, title) in enumerate(GROUPS):
        views_per_row = len(CHOSEN[(ROWS[0][0], role)])
        inner = outer[g].subgridspec(len(ROWS), views_per_row, wspace=0.03, hspace=0.03)
        for r, (style, label) in enumerate(ROWS):
            record[f"{style}/{role}"] = []
            for c, (seed, rig) in enumerate(CHOSEN[(style, role)]):
                chunk = chunk_of(seed)
                wall = json.loads((root / chunk / "walls" / f"seed_{seed}" / "scene.json").read_text())
                if wall["appearance_controls"]["crack_style"] != style or wall["cameras"][rig]["role"] != role:
                    raise SystemExit(f"{seed}/{rig} is not a {style} {role} view")
                image, crack_fraction = tile(root / chunk / "pack", f"seed_{seed}_rig{rig}_resample0_frame0001_camera_0.png")
                if not crack_fraction:
                    raise SystemExit(f"{seed}/{rig} has no crack label")
                record[f"{style}/{role}"].append(dict(split=chunk, seed=seed, view=rig,
                                                      material=wall["wall_material"]["diverse"]["category"],
                                                      crack_percent=round(100 * crack_fraction, 3)))
                ax = fig.add_subplot(inner[r, c])
                ax.set_xticks([]), ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.imshow(image)
                if r == 0 and c == views_per_row // 2:
                    ax.set_title(title, fontsize=7, pad=3)
                if g == 0 and c == 0:
                    ax.set_ylabel(label, fontsize=7, labelpad=3)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"Figure_contact_sheet.{suffix}", dpi=DPI)
    (out / "Figure_contact_sheet.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
