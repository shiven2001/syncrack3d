"""Score a crack model on the frozen SynCrack3D synthetic evaluation set.

    python eval/synth/score.py --set OUT --model MODEL_DIR [--save SUMMARY.json]

MODEL_DIR holds runs/s*/last.pt; every seed is scored and results are reported
as mean ± sd over seeds. Protocol v2, as for BD3, CUBIT and BuildCrack: 1.5x
test-time upscale (eval/infer.py) and 5 px centreline tolerance
(eval/metrics.py). Ignore masks are excluded except on crack pixels, as in
training.

Per subset (legacy, major, network, facade, negatives), by wall material, and
pooled over the cracked subsets:
  centreline precision / recall / F1 on cracked views (pooled lengths)
  crack IoU on cracked views (pooled pixels)
  false-positive pixels per view farther than 5 px from a labelled crack
  plain-view alarm rate: share of crack-free views whose predicted crack covers
  more than 0.1% of the view (the BD3 plain-wall rule)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL))
from infer import load_model, predict  # noqa: E402
from metrics import TOLERANCE_PX, _grow, centreline_counts, centreline_scores  # noqa: E402

ALARM_FRACTION = 0.001
CRACKED = ("legacy", "major", "network", "facade")

ap = argparse.ArgumentParser()
ap.add_argument("--set", type=Path, required=True, help="eval/synth/build.py output")
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--save", type=Path)
args = ap.parse_args()
manifest = json.loads((args.set / "manifest.json").read_text())
checkpoints = sorted(args.model.glob("runs/s*/last.pt"))
if not checkpoints:
    raise SystemExit(f"no runs/s*/last.pt under {args.model}")


def new():
    return {"cl": np.zeros(4), "iou": np.zeros(3), "fp": [], "alarm": []}


def finish(a):
    s = centreline_scores(a["cl"]) if a["cl"][1] else dict(precision=None, recall=None, f1=None)
    return dict(precision=s["precision"], recall=s["recall"], f1=s["f1"],
                crack_iou=a["iou"][0] / max(1, a["iou"].sum()) if a["iou"].sum() else None,
                fp_px_per_view=float(np.mean(a["fp"])) if a["fp"] else None,
                plain_view_alarm=float(np.mean(a["alarm"])) if a["alarm"] else None)


views = []
for v in manifest["views"]:
    mask = cv2.imread(str(args.set / v["mask"]), 0)
    ign_path = args.set / v["mask"].replace("masks/", "ignore_masks/")
    crack = mask == 60
    ign = (cv2.imread(str(ign_path), 0) > 0) if ign_path.exists() else np.zeros_like(crack)
    views.append((v, crack, ~(ign & ~crack)))

per_seed = []
for ck in checkpoints:
    model = load_model(ck)
    groups = {}
    for v, crack, valid in views:
        img = cv2.cvtColor(cv2.imread(str(args.set / v["image"])), cv2.COLOR_BGR2RGB)
        pred = (predict(model, img) == 1) & valid
        keys = [v["subset"], f"{v['subset']}/{v['material']}"] + (["all_cracked"] if v["subset"] in CRACKED else [])
        far = pred & ~_grow(crack, TOLERANCE_PX)
        for k in keys:
            a = groups.setdefault(k, new())
            a["fp"].append(int(far.sum()))
            if crack.any():
                a["cl"] += centreline_counts(pred, crack, TOLERANCE_PX)
                a["iou"] += [(pred & crack).sum(), (pred & ~crack).sum(), (~pred & crack).sum()]
            else:
                a["alarm"].append(pred.mean() > ALARM_FRACTION)
    per_seed.append({k: finish(a) for k, a in groups.items()})
    print("scored", ck.parent.name, flush=True)

summary = {}
for k in per_seed[0]:
    summary[k] = {}
    for m in per_seed[0][k]:
        vals = [s[k][m] for s in per_seed if s[k][m] is not None]
        summary[k][m] = [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)] if vals else None
    summary[k]["views"] = sum(1 for v, _, _ in views if k in (v["subset"], f"{v['subset']}/{v['material']}")
                              or (k == "all_cracked" and v["subset"] in CRACKED))

print(f"{'group':28} {'views':>5} {'F1':>13} {'recall':>13} {'precision':>13} {'IoU':>13} {'FP px/view':>13} {'plain alarm':>13}")
for k in ["all_cracked", *CRACKED, "negatives", *sorted(x for x in summary if "/" in x)]:
    if k not in summary:
        continue
    s = summary[k]
    cell = lambda m, f=".3f": "-" if s[m] is None else f"{s[m][0]:{f}}±{s[m][1]:{f}}"  # noqa: E731
    print(f"{k:28} {s['views']:>5} {cell('f1'):>13} {cell('recall'):>13} {cell('precision'):>13} "
          f"{cell('crack_iou'):>13} {cell('fp_px_per_view', '.0f'):>13} {cell('plain_view_alarm'):>13}")
result = dict(model=str(args.model), seeds=[c.parent.name for c in checkpoints], set=str(args.set),
              set_commit=manifest["code_commit"], protocol=dict(test_scale="eval/infer.TEST_SCALE",
                                                                tolerance_px=TOLERANCE_PX,
                                                                alarm_fraction=ALARM_FRACTION),
              summary=summary, per_seed=per_seed)
if args.save:
    args.save.write_text(json.dumps(result, indent=1))
