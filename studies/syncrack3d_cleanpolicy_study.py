"""Clean-wall policy study: material-aware vs all-cracked vs uniform plain walls.

The same 200 unified-recipe walls (seeds 820000-820199) are rendered twice:
cracked (--clean-wall-scale 0) and crack-free (--clean-wall-uniform 1). A wall's
material, camera rig, lighting and props do not depend on the policy, so each arm
is assembled by taking, per wall, one of the two renders:

  cleanmat   A, material-aware: crack-free where the unified per-category table
             says so (paint/stucco .15, concrete .35, patterned masonry/tile/stone .60)
  cleannone  B, all cracked
  cleanuni   C, uniform: the same NUMBER of crack-free walls as A, chosen by the
             same per-wall draw regardless of material (the walls with the lowest
             draw), so A vs C differs only in where the plain walls fall

Each arm trains from scratch on its own 200 walls (no v1.x data), with the
four-thousand-step, five-seed budget of the earlier studies.

    python -m studies.syncrack3d_cleanpolicy_study
"""

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/shiven/syndefect3d")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "syncrack3d"))

from syncrack3d.material_diversity import _rng, is_clean_wall  # noqa: E402

WALLS = ROOT / "syncrack3d/outputs/controlled_walls/syncrack3d_cleanpolicy_v1"
DT = ROOT / "syncrack3d/outputs/domain_transfer"
ARMS = ("cleanmat", "cleannone", "cleanuni")
SEEDS = [0, 1, 2, 3, 4]


def walls():
    out = {}
    for f in sorted((WALLS / "cracked").glob("seed_*/scene.json")):
        rec = json.loads(f.read_text())
        seed = int(rec["seed"])
        clean = json.loads((WALLS / "clean" / f.parent.name / "scene.json").read_text())
        assert clean["wall_material"]["diverse"]["category"] == rec["wall_material"]["diverse"]["category"], seed
        out[seed] = rec["wall_material"]["diverse"]["category"]
    return out


def assign(cats):
    """{arm: set of crack-free wall seeds}."""
    mat = {s for s, c in cats.items() if is_clean_wall(s, c, 1.0, "unified")}
    draw = sorted(cats, key=lambda s: _rng(s, 0x434C4E5F).random())
    return {"cleanmat": mat, "cleannone": set(), "cleanuni": set(draw[:len(mat)])}


def write_label(pack, stem, dest):
    raw = cv2.imread(str(pack / "masks" / f"{stem}.png"), 0)
    vals = set(np.unique(raw).tolist())
    if not vals <= {0, 60, 255}:
        raise ValueError((stem, vals))
    lab = np.zeros(raw.shape, np.uint8)
    lab[raw == 60] = 1
    ignore = pack / "ignore_masks" / f"{stem}.png"
    if ignore.exists():
        lab[(cv2.imread(str(ignore), 0) > 0) & (lab == 0)] = 255
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), lab)
    return int((lab == 1).any())


def rows_for(arm, clean, cats):
    out_dir = DT / f"syncrack3d_{arm}_bd3_v1"
    packs = {False: WALLS / "cracked" / "pack", True: WALLS / "clean" / "pack"}
    by_seed = {k: {} for k in packs}
    for k, pack in packs.items():
        for img in (pack / "images").glob("*.png"):
            by_seed[k].setdefault(int(img.stem.split("_")[1]), []).append(img)
    rows, cracked = [], 0
    for seed in sorted(cats):
        is_clean = seed in clean
        for img in sorted(by_seed[is_clean].get(seed, [])):
            dest = out_dir / "masks" / f"{img.stem}.png"
            cracked += write_label(packs[is_clean], img.stem, dest)
            rows.append((f"{arm}__{img.stem}", str(img.resolve()), str(dest.resolve()), False))
    record = dict(n=len(rows), with_crack=cracked, clean_walls=len(clean), walls=len(cats),
                  clean_by_category=dict(sorted(Counter(cats[s] for s in clean).items())),
                  seeds=SEEDS, iters=4000, design=__doc__.split("\n\n")[2].strip())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_list.json").write_text(json.dumps(record, indent=1) + "\n")
    return rows, record


def main():
    import torch

    from domain_transfer.multiclass_seg import train
    from domain_transfer.segmentation import benchmark

    cats = walls()
    if len(cats) != 200:
        raise SystemExit(f"expected 200 paired walls, found {len(cats)}")
    clean = assign(cats)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    b = benchmark()
    for arm in ARMS:
        rows, record = rows_for(arm, clean[arm], cats)
        print(arm, {k: record[k] for k in ("n", "with_crack", "clean_walls")}, flush=True)
        for seed in SEEDS:
            run_dir = DT / f"syncrack3d_{arm}_bd3_v1" / "runs" / f"s{seed}"
            if not (run_dir / "last.pt").exists():
                train(b, rows, run_dir, seed, iters=4000)
            torch.cuda.empty_cache()
        print(arm, "trained", flush=True)


if __name__ == "__main__":
    main()
