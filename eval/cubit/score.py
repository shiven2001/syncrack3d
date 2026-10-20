"""Zero-shot SynCrack3D crack model on CUBIT-InSeg, at 1/3 scale only.

CUBIT-InSeg here: YOLO-seg polygons, class 0 = crack (CR images), class 1 =
spalling with exposed rebar (SP images). The local copy is incomplete: images
and label files are matched by name across all split folders, because the test
folder has no image/label pair and val has 2. Spalling polygons (dilated 15 px
at full resolution, scaled with the image) are IGNORE, not background. Images
without a label file are skipped.

Images are resized to 1/3 (about 1500x1000), where a crack is about 2.5 px,
close to the training renders. Full-resolution tiled inference is not run.
Metrics match BD3: centreline P/R/F1 at 3 px, clDice, crack IoU, plus the share
of spalling photos with predicted crack area above 0.1% outside the ignore region.

    python eval/cubit/score.py --model path/to/checkpoint_dir --out path/to/out
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer import load_model, predict
from metrics import centreline_counts, centreline_scores, cldice_counts

CUBIT = Path("/mnt/nvme_storage/CUBIT-InSeg")
SCALE = 1 / 3
ALARM_FRACTION = 0.001


def pairs(root):
    ims = {p.stem: (s, p) for s in ("train", "val", "test") for p in (root / s / "images").iterdir()}
    lbs = {p.stem: (s, p) for s in ("train", "val", "test") for p in (root / s / "labels").iterdir()}
    return [(k, ims[k][0], ims[k][1], lbs[k][1]) for k in sorted(set(ims) & set(lbs))]


def rasterise(label, h, w):
    crack, ignore = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    for line in label.read_text().split("\n"):
        v = line.split()
        if len(v) < 7:
            continue
        xy = np.round(np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.fillPoly(crack if v[0] == "0" else ignore, [xy], 1)
    k = max(3, int(round(15 * w / 4500)) | 1)
    ignore = cv2.dilate(ignore, np.ones((k, k), np.uint8))
    return crack, ignore > 0


def counts(pred_crack, gt, ignore):
    keep = ~ignore
    p, g = pred_crack & keep, (gt > 0) & keep
    tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
    return np.r_[tp, fp, fn, centreline_counts(p, g, 3), cldice_counts(p, g)]


def metrics(c):
    cl = centreline_scores(c[3:7])
    t = c[7:]
    a, b_ = t[0] / max(1, t[1]), t[2] / max(1, t[3])
    return {
        "centreline_f1_3px": cl["f1"],
        "centreline_precision": cl["precision"],
        "centreline_recall": cl["recall"],
        "cldice": float(2 * a * b_ / max(1e-9, a + b_)),
        "crack_iou": float(c[0] / max(1, c[0] + c[1] + c[2])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=CUBIT)
    ap.add_argument("--model", type=Path, required=True, help="directory containing runs/s*/last.pt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch

    seeds = sorted(args.model.glob("runs/s*/last.pt"))
    if not seeds:
        raise SystemExit(f"no runs/s*/last.pt under {args.model}")
    data = pairs(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, alarm = {}, {}
    for ck in seeds:
        model = load_model(ck)
        seed = ck.parent.name
        rows[seed], alarm[seed] = {}, {}
        for k, _split, imp, lbp in data:
            img = cv2.cvtColor(cv2.imread(str(imp)), cv2.COLOR_BGR2RGB)
            im = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            h, w = im.shape[:2]
            gt, ign = rasterise(lbp, h, w)
            pred = predict(model, im) == 1
            rows[seed][k] = counts(pred, gt, ign)
            alarm[seed][k] = float((pred & ~ign).mean())
        del model
        torch.cuda.empty_cache()
        print("done", seed, flush=True)
    ids = [k for k, *_ in data]
    np.savez_compressed(
        args.out / "counts.npz",
        ids=np.array(ids),
        seeds=np.array([c.parent.name for c in seeds]),
        **{f"third_{c.parent.name}": np.stack([rows[c.parent.name][k] for k in ids]) for c in seeds},
    )
    split_of = {k: s for k, s, *_ in data}
    cr = [k for k in ids if k.startswith("CR")]
    sp = [k for k in ids if k.startswith("SP")]
    per = [metrics(sum(rows[c.parent.name][k] for k in cr)) for c in seeds]
    labelled = [metrics(sum(rows[c.parent.name][k] for k in ids)) for c in seeds]

    def pack(rows_):
        return {
            m: [float(np.mean([p[m] for p in rows_])), float(np.std([p[m] for p in rows_], ddof=1)) if len(rows_) > 1 else 0.0]
            for m in rows_[0]
        }

    res = {
        "model": str(args.model),
        "scale": "1/3",
        "seeds": [c.parent.name for c in seeds],
        "pairs": len(ids),
        "pairs_by_split_folder": {s: sum(split_of[k] == s for k in ids) for s in ("train", "val", "test")},
        "crack_photos": len(cr),
        "spalling_photos": len(sp),
        "results": {
            "crack_photos_CR": pack(per),
            "all_labelled": pack(labelled),
            "spalling_photos_with_false_crack": float(np.mean([
                np.mean([alarm[c.parent.name][k] > ALARM_FRACTION for k in sp]) for c in seeds
            ])),
        },
    }
    (args.out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
