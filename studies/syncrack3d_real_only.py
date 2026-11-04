"""Real-only crack segmentation references for the SynCrack3D zero-shot results (U-Net + ResNet-34).

In-domain 5-fold grouped cross-validation on each real benchmark, on exactly the
images and labels the zero-shot scorers use (syncrack3d/eval/{bd3,cubit,buildcrack}):

- BD3: 487 minor_crack + 355 plain photos, the manifest's frozen grouped folds;
- CUBIT-InSeg: every labelled image at 1/3 scale, spalling (dilated) ignored as in
  the scorer; groups are near-duplicates (studies/near_duplicates.py);
- BuildCrack: 358 tiles; a group is the source photo (scorer's bootstrap unit).

Every image is predicted once by a model that never saw it or its group. Recipe is
seg_train.train (U-Net R34, ImageNet, 3-class head, CE + Dice, AdamW 3e-4,
4000 steps, batch 8, 512 crops, no synthetic envelope or camera simulation, final
checkpoint), seeds 0 1 2. Inference is at the training resolution (scale 1, no
1.5x upscale: that upscale matches synthetic crack widths). Metrics and pooling
come from the scorers themselves (5 px centreline tolerance). Real-only is
in-domain, so it is an upper reference, not a zero-shot result.

    python -m studies.syncrack3d_real_only [--datasets bd3 cubit buildcrack]
"""

import argparse
import collections
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from studies import near_duplicates
from studies.seg_train import train as fit

APP = Path(__file__).resolve().parents[1]

EVAL = APP / "eval"
OUT = APP / "outputs/domain_transfer/syncrack3d_real_only_v1"
CUBIT = Path(os.environ.get("SYNCRACK3D_CUBIT", "/mnt/nvme_storage/CUBIT-InSeg"))
BUILDCRACK = Path(os.environ.get("SYNCRACK3D_BUILDCRACK", APP / "external/BuildCrack"))
FOLDS, SEEDS, SCALE = 5, (0, 1, 2), 1.0
IGNORE = 255
sys.path.insert(0, str(EVAL))
from infer import load_model, predict  # noqa: E402
from metrics import TOLERANCE_PX  # noqa: E402


def scorer(name):
    spec = importlib.util.spec_from_file_location(f"{name}_score", EVAL / name / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.write_text(json.dumps(value, indent=1) + "\n")


def balanced_folds(ids, group_of, positive):
    """Greedy grouped folds: large and positive-heavy groups first, then the emptiest fold."""
    members = collections.defaultdict(list)
    for i in ids:
        members[group_of[i]].append(i)
    order = sorted(members, key=lambda g: (-sum(positive[i] for i in members[g]), -len(members[g]), g))
    folds, pos, rng = [[] for _ in range(FOLDS)], [0] * FOLDS, random.Random(0)
    for g in order:
        k = min(range(FOLDS), key=lambda k: (len(folds[k]), pos[k], rng.random()))
        folds[k] += members[g]
        pos[k] += sum(positive[i] for i in members[g])
    return {i: k for k, f in enumerate(folds) for i in f}


def prepare(name, out):
    """Write <out>/data/{images,gt}/<id>.png (gt 0/1/255) and return {id: (image, gt, fold, positive)}."""
    frozen = out / "split.json"
    if frozen.exists():
        return {k: tuple(v) for k, v in json.loads(frozen.read_text())["items"].items()}
    img_dir, gt_dir = out / "data/images", out / "data/gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    items, group_of, note = {}, {}, ""
    if name == "bd3":
        for p in scorer("bd3").manifest():
            items[p["id"]] = [str(EVAL / "bd3" / p["image"]), str(EVAL / "bd3" / p["gt"]), p["fold"],
                              p["folder"] == "minor_crack"]
            group_of[p["id"]] = p["group"]
        note = "manifest folds (grouped by near-duplicates across folders)"
    elif name == "cubit":
        sc = scorer("cubit")
        for k, _split, imp, lbp in sc.pairs(CUBIT):
            im = cv2.resize(cv2.imread(str(imp)), None, fx=sc.SCALE, fy=sc.SCALE, interpolation=cv2.INTER_AREA)
            crack, ign = sc.rasterise(lbp, *im.shape[:2])
            gt = np.where(ign, IGNORE, crack).astype(np.uint8)
            cv2.imwrite(str(img_dir / f"{k}.png"), im)
            cv2.imwrite(str(gt_dir / f"{k}.png"), gt)
            items[k] = [str(img_dir / f"{k}.png"), str(gt_dir / f"{k}.png"), None, k.startswith("CR")]
        ids = sorted(items)
        group_of, links = near_duplicates.groups(ids, [items[i][0] for i in ids])
        note = f"near-duplicate groups (cos >= {near_duplicates.COS} or pHash <= {near_duplicates.HAMMING}), {len(links)} links"
    elif name == "buildcrack":
        sc = scorer("buildcrack")
        for p in sorted((BUILDCRACK / "images").glob("*.png")):
            gt = (cv2.imread(str(BUILDCRACK / "labels" / p.name), 0) > 0).astype(np.uint8)
            cv2.imwrite(str(gt_dir / p.name), gt)
            items[p.stem] = [str(p), str(gt_dir / p.name), None, bool(gt.any())]
            group_of[p.stem] = sc.source_of(p.stem)
        note = "source photo (numeric prefix or DJI id)"
    if name != "bd3":
        fold_of = balanced_folds(sorted(items), group_of, {k: v[3] for k, v in items.items()})
        for k in items:
            items[k][2] = fold_of[k]
    sizes = collections.Counter(group_of.values())
    write(frozen, {"dataset": name, "grouping": note, "groups": len(sizes), "largest_group": max(sizes.values()),
                   "fold_sizes": collections.Counter(v[2] for v in items.values()),
                   "fold_positive": collections.Counter(v[2] for v in items.values() if v[3]),
                   "group_of": group_of, "items": items})
    return {k: tuple(v) for k, v in items.items()}


def run(name):
    out = OUT / name
    items = prepare(name, out)
    for seed in SEEDS:
        pred_dir = out / f"pred_s{seed}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(FOLDS):
            held = [k for k, v in items.items() if v[2] == fold]
            if all((pred_dir / f"{k}.png").exists() for k in held):
                continue
            ck = out / f"runs/s{seed}_f{fold}/last.pt"
            if not ck.exists():
                # Real flag off: these label maps are read unchanged, not binarised.
                train = [(k, v[0], v[1], False) for k, v in items.items() if v[2] != fold]
                fit(train, ck.parent, seed, syn_envelope=0, syn_camera=False)
            model = load_model(ck)
            for k in held:
                img = cv2.cvtColor(cv2.imread(items[k][0]), cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(pred_dir / f"{k}.png"), predict(model, img, SCALE).astype(np.uint8))
            del model
            print(name, f"s{seed}", f"f{fold}", "done", flush=True)
    return score(name, items, out)


def mean_sd(rows):
    return {m: [float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows], ddof=1))] for m in rows[0]}


def score(name, items, out):
    sc = scorer(name)
    preds = {s: {k: cv2.imread(str(out / f"pred_s{s}/{k}.png"), 0) for k in items} for s in SEEDS}
    gts = {k: cv2.imread(items[k][1], 0) for k in items}
    res = {"dataset": name, "model": "U-Net + ResNet-34, real-only grouped 5-fold CV", "seeds": list(SEEDS),
           "protocol": {"tolerance_px": TOLERANCE_PX, "test_scale": SCALE}, "images": len(items)}
    if name == "bd3":
        per = [sc.score(out / f"pred_s{s}") for s in SEEDS]
        res["results"] = {"minor_crack": mean_sd([{m: p[m] for m in ("centreline_f1", "centreline_precision",
                          "centreline_recall", "cldice", "crack_iou")} for p in per]),
                          "plain_false_alarms": float(np.mean([p["plain_false_alarms"] for p in per]))}
    elif name == "cubit":
        cr = [k for k in sorted(items) if k.startswith("CR")]
        sp = [k for k in sorted(items) if k.startswith("SP")]
        rows = {s: {k: sc.counts(preds[s][k] == 1, gts[k] == 1, gts[k] == IGNORE) for k in items} for s in SEEDS}
        alarm = [np.mean([((preds[s][k] == 1) & (gts[k] != IGNORE)).mean() > sc.ALARM_FRACTION for k in sp])
                 for s in SEEDS]
        res["results"] = {"crack_photos_CR": mean_sd([sc.metrics(sum(rows[s][k] for k in cr)) for s in SEEDS]),
                          "all_labelled": mean_sd([sc.metrics(sum(rows[s].values())) for s in SEEDS]),
                          "spalling_photos_with_false_crack": float(np.mean(alarm))}
    else:
        ids = sorted(items)
        empty = [k for k in ids if not items[k][3]]
        per = [sc.metrics(sum(sc.one(preds[s][k] == 1, gts[k]) for k in ids)) for s in SEEDS]
        votes = {k: sum((preds[s][k] == 1).astype(np.uint8) for s in SEEDS) >= len(SEEDS) // 2 + 1 for k in ids}
        res["results"] = {"mean_of_seeds": mean_sd(per),
                          "ensemble": sc.metrics(sum(sc.one(votes[k], gts[k]) for k in ids)),
                          "empty_tile_alarm_rate": float(np.mean([[(preds[s][k] == 1).mean() > sc.ALARM_FRACTION
                                                                  for k in empty] for s in SEEDS]))}
    write(out / "results.json", res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["bd3", "cubit", "buildcrack"])
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    for name in args.datasets:
        if args.prepare_only:
            prepare(name, OUT / name)
            split = json.loads((OUT / name / "split.json").read_text())
            print(name, {k: split[k] for k in ("grouping", "groups", "largest_group", "fold_sizes", "fold_positive")})
            continue
        res = run(name)
        print(json.dumps(res["results"], indent=1), flush=True)
    print("REAL_ONLY_DONE", flush=True)


if __name__ == "__main__":
    main()
