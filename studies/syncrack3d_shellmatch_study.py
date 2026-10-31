"""Matched indoor context finishes: plain shell vs side walls/floor with real finishes.

Control is the all-cracked arm of the clean-wall study (syncrack3d_cleannone_bd3_v1):
200 unified-recipe walls (seeds 820000-820199), every wall cracked, plain flat-colour
shell. The treatment re-renders the same walls with --shell-finish matched: indoor
side walls in the test wall's own finish (70%, else a different paint) and an
Infinigen floor material. Cracks, cameras, lights and props are checked identical
wall by wall before training; outdoor facade walls are unchanged by design. Same
four-thousand-step, five-seed budget, trained from scratch on the 200 walls.

    python -m studies.syncrack3d_shellmatch_study
"""

import json
import sys
from pathlib import Path

import cv2

ROOT = Path("/home/shiven/syndefect3d")
sys.path.insert(0, str(ROOT))

from studies.syncrack3d_cleanpolicy_study import write_label  # noqa: E402

CONTROL = ROOT / "syncrack3d/outputs/controlled_walls/syncrack3d_cleanpolicy_v1/cracked"
TREAT = ROOT / "syncrack3d/outputs/controlled_walls/syncrack3d_shellmatch_v1/matched"
OUT = ROOT / "syncrack3d/outputs/domain_transfer/syncrack3d_shellmatch_bd3_v1"
SEEDS = [0, 1, 2, 3, 4]
SAME = ("cameras", "defects_placed", "lighting", "props", "spec")


def check_pairs():
    walls = sorted(TREAT.glob("seed_*/scene.json"))
    if len(walls) != 200:
        raise SystemExit(f"expected 200 treatment walls, found {len(walls)}")
    finishes = {}
    for f in walls:
        a = json.loads((CONTROL / f.parent.name / "scene.json").read_text())
        b = json.loads(f.read_text())
        diff = [k for k in SAME if a[k] != b[k]]
        if diff:
            raise SystemExit(f"{f.parent.name}: treatment differs from control in {diff}")
        sf = b.get("shell_finish") or {}
        key = "outdoor" if b["environment"] == "outdoor" else sf.get("returns", {}).get("source", "no side walls")
        finishes[key] = finishes.get(key, 0) + 1
    return finishes


def main():
    import torch

    from domain_transfer.multiclass_seg import train
    from domain_transfer.segmentation import benchmark

    finishes = check_pairs()
    pack = TREAT / "pack"
    rows, cracked = [], 0
    for img in sorted((pack / "images").glob("*.png")):
        dest = OUT / "masks" / f"{img.stem}.png"
        cracked += write_label(pack, img.stem, dest)
        rows.append((f"shellmatch__{img.stem}", str(img.resolve()), str(dest.resolve()), False))
    record = dict(n=len(rows), with_crack=cracked, side_wall_finishes=finishes, seeds=SEEDS, iters=4000,
                  design="same 200 all-cracked walls as syncrack3d_cleannone_bd3_v1, --shell-finish matched")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_list.json").write_text(json.dumps(record, indent=1) + "\n")
    print(record, flush=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    b = benchmark()
    for seed in SEEDS:
        run_dir = OUT / "runs" / f"s{seed}"
        if not (run_dir / "last.pt").exists():
            train(b, rows, run_dir, seed, iters=4000)
        torch.cuda.empty_cache()
    print("trained", flush=True)


if __name__ == "__main__":
    main()
