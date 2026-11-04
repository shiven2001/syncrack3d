"""Test-time input scale and crack threshold for a SynCrack3D checkpoint.

No retraining. Each image is resized by `scale`, predicted, and the crack
probability map is resized back. A pixel is crack when p(crack) > p(class 2)
and p(crack) + p(class 2) >= `threshold`; 0.5 is the released rule in
syncrack3d/eval/infer.py. Scored like the BuildCrack and CUBIT scorers:
centreline P/R/F1 pooled at 3 px, mean of five seeds, plus BuildCrack
empty-tile alarms (>0.1% crack). CUBIT is CR crack photos at 1/3 scale times
`scale`, spalling ignored. Use CUBIT to pick a setting and BuildCrack to check
it, or the other way round, never both on one set.

    python -m studies.syncrack3d_testtime_sweep --model path/to/checkpoint_dir
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[1]
EVAL = APP / "eval"
sys.path.insert(0, str(EVAL))
from infer import MEAN, STD, load_model  # noqa: E402
from metrics import centreline_counts, centreline_scores  # noqa: E402

SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
THRESHOLDS = (0.5, 0.35, 0.2)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probs(model, img):
    """(p_crack, p_class2) at the image's own size."""
    import torch

    h, w = img.shape[:2]
    height, width = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
    x = np.pad(img.astype(np.float32), ((0, height - h), (0, width - w), (0, 0)), mode="reflect")
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)[None]).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        p = torch.softmax(model(x).float(), 1)[0, 1:, :h, :w].cpu().numpy()
    return p[0], p[1]


def crack_masks(model, img, scale, thresholds=THRESHOLDS):
    h, w = img.shape[:2]
    im = img if scale == 1 else cv2.resize(img, (round(w * scale), round(h * scale)),
                                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    p1, p2 = probs(model, im)
    if scale != 1:
        p1, p2 = (cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in (p1, p2))
    return {t: (p1 > p2) & (p1 + p2 >= t) for t in thresholds}


def datasets():
    cubit = _module("cubit_score", EVAL / "cubit/score.py")
    build = _module("buildcrack_score", EVAL / "buildcrack/score.py")
    out = {"buildcrack": [], "cubit": []}
    for p in sorted((build.DATA / "images").glob("*.png")):
        gt = cv2.imread(str(build.DATA / "labels" / p.name), 0) > 0
        out["buildcrack"].append((cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), gt, np.ones_like(gt)))
    for k, _split, imp, lbp in cubit.pairs(cubit.CUBIT):
        if not k.startswith("CR"):
            continue
        img = cv2.cvtColor(cv2.imread(str(imp)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(img, None, fx=cubit.SCALE, fy=cubit.SCALE, interpolation=cv2.INTER_AREA)
        gt, ign = cubit.rasterise(lbp, *im.shape[:2])
        out["cubit"].append((im, (gt > 0) & ~ign, ~ign))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=APP / "outputs/domain_transfer/syncrack3d_testtime_sweep_v1.json")
    ap.add_argument("--scales", type=float, nargs="+", default=SCALES)
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    args = ap.parse_args()
    import torch

    data = datasets()
    seeds = sorted(args.model.glob("runs/s*/last.pt"))
    acc = {}  # (dataset, scale, threshold) -> per-seed [counts(4), alarms]
    for ck in seeds:
        model = load_model(ck)
        for ds, items in data.items():
            for scale in args.scales:
                tot = {t: np.zeros(4, np.int64) for t in args.thresholds}
                alarm = {t: [] for t in args.thresholds}
                for img, gt, valid in items:
                    for t, pred in crack_masks(model, img, scale, args.thresholds).items():
                        pred &= valid
                        tot[t] += centreline_counts(pred, gt, 3)
                        if not gt.any():
                            alarm[t].append(pred.mean() > 0.001)
                for t in args.thresholds:
                    acc.setdefault((ds, scale, t), []).append(
                        (centreline_scores(tot[t]), float(np.mean(alarm[t])) if alarm[t] else None))
        del model
        torch.cuda.empty_cache()
        print("done", ck.parent.name, flush=True)
    rows = []
    for (ds, scale, t), v in sorted(acc.items()):
        rows.append({"dataset": ds, "scale": scale, "threshold": t,
                     **{k: float(np.mean([s[k] for s, _ in v])) for k in ("precision", "recall", "f1")},
                     "f1_sd": float(np.std([s["f1"] for s, _ in v], ddof=1)),
                     "empty_alarm": None if v[0][1] is None else float(np.mean([a for _, a in v]))})
    args.out.write_text(json.dumps({"model": str(args.model), "rows": rows}, indent=1) + "\n")
    for ds in data:
        print(f"\n{ds}: scale threshold   precision recall  F1 (sd)        empty-tile alarm")
        for r in rows:
            if r["dataset"] == ds:
                al = "" if r["empty_alarm"] is None else f"{r['empty_alarm']:.3f}"
                print(f"   {r['scale']:5} {r['threshold']:9} {r['precision']:9.3f} {r['recall']:6.3f}  "
                      f"{r['f1']:.3f} ({r['f1_sd']:.3f})  {al}")


if __name__ == "__main__":
    main()
