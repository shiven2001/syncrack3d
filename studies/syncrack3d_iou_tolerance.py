"""Crack IoU with boundary tolerance, and where v1.17 misses BuildCrack cracks.

Two questions about the v1.17 U-Net R34 (4,000 walls, seeds 0-2) and the
real-only U-Net references (syncrack3d_real_only):

1. How much of the IoU gap is boundary width? For each benchmark's crack images
   (BD3 minor_crack, CUBIT CR photos with spalling ignored, all BuildCrack tiles),
   pooled over images and averaged over seeds:
   - strict IoU (the scorers' crack_iou);
   - IoU against labels grown by r px (as if the annotators drew wider cracks);
   - tolerant IoU at t px: precision = predicted pixels within t of the label,
     recall = label pixels within t of the prediction, IoU = PR / (P + R - PR).
2. Which BuildCrack cracks are missed? Every label centreline pixel gets a width
   (2 * distance to the label edge - 1, px at tile scale), a contrast
   ((ring - crack) / ring luminance, ring 3-8 px outside the label) and its
   source (DJI drone photo or not). It is found when a prediction lies within
   5 px (the centreline recall rule). Zero-shot is also re-run at test scales
   1, 1.5 (released) and 2 to see whether wide cracks are missed for size.
   v1.17 training label widths (train_a, native 640 px) are the comparison.

Zero-shot BD3 uses the saved pred_v2 masks; CUBIT and BuildCrack are re-inferred
on the real-only prepared images (same 1/3-scale CUBIT and ignore masks).

    python -m studies.syncrack3d_iou_tolerance
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

APP = Path(__file__).resolve().parents[1]
EVAL = APP / "eval"
sys.path.insert(0, str(EVAL))
from infer import TEST_SCALE, load_model, predict  # noqa: E402
from metrics import TOLERANCE_PX, _grow  # noqa: E402

DT = APP / "outputs/domain_transfer"
ZS = DT / "syncrack3d_v117_unet_w4000"
RO = DT / "syncrack3d_real_only_v1"
OUT = DT / "syncrack3d_v117_iou_tolerance_v1"
TRAIN_LABELS = Path("/mnt/nvme_storage/syncrack3d_v1.17.0_dataset/train_a/labels")
SEEDS = (0, 1, 2)
GROW, TOLS = (1, 2), (1, 2, 3)
WIDTH_BINS = (0, 3, 6, 10, 20, 1000)
SCALES = (1.0, TEST_SCALE, 2.0)
IGNORE = 255


def items(name):
    split = json.loads((RO / name / "split.json").read_text())["items"]
    keep = {"bd3": lambda k, v: v[3], "cubit": lambda k, v: k.startswith("CR"), "buildcrack": lambda k, v: True}
    return {k: v for k, v in sorted(split.items()) if keep[name](k, v)}


def zero_shot_dir(name, seed, scale=TEST_SCALE):
    if name == "bd3" and scale == TEST_SCALE:
        return ZS / f"pred_v2/s{seed}"
    return OUT / f"zs_pred/{name}_x{scale:g}/s{seed}"


def infer(name, its, scale=TEST_SCALE):
    for seed in SEEDS:
        out = zero_shot_dir(name, seed, scale)
        if all((out / f"{k}.png").exists() for k in its):
            continue
        out.mkdir(parents=True, exist_ok=True)
        model = load_model(ZS / f"runs/s{seed}/last.pt")
        for k, v in its.items():
            img = cv2.cvtColor(cv2.imread(v[0]), cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(out / f"{k}.png"), predict(model, img, scale).astype(np.uint8))
        del model
        print("inferred", name, f"x{scale:g}", f"s{seed}", flush=True)


def counts(p, g, valid):
    """Strict, grown-label and tolerant pixel counts for one image."""
    p, g = p & valid, g & valid
    row = [(p & g).sum(), (p & ~g).sum(), (g & ~p).sum()]
    for r in GROW:
        gr = _grow(g, r) & valid
        row += [(p & gr).sum(), (p & ~gr).sum(), (gr & ~p).sum()]
    for t in TOLS:
        row += [(p & _grow(g, t)).sum(), p.sum(), (g & _grow(p, t)).sum(), g.sum()]
    return np.array(row, np.int64)


def scores(c):
    iou = lambda tp, fp, fn: tp / max(1, tp + fp + fn)  # noqa: E731
    out = {"iou": iou(*c[:3]), "pixel_precision": c[0] / max(1, c[0] + c[1]), "pixel_recall": c[0] / max(1, c[0] + c[2])}
    for i, r in enumerate(GROW):
        out[f"iou_label_grown_{r}px"] = iou(*c[3 + 3 * i:6 + 3 * i])
    base = 3 + 3 * len(GROW)
    for i, t in enumerate(TOLS):
        a = c[base + 4 * i:base + 4 * i + 4]
        p, r = a[0] / max(1, a[1]), a[2] / max(1, a[3])
        out[f"iou_tolerant_{t}px"] = p * r / max(1e-9, p + r - p * r)
    return {k: float(v) for k, v in out.items()}


def masks(pred_path, gt_path):
    pred, gt = cv2.imread(str(pred_path), 0), cv2.imread(str(gt_path), 0)
    if pred.shape != gt.shape:
        raise ValueError(f"{pred_path}: {pred.shape} != {gt.shape}")
    return pred == 1, gt == 1, gt != IGNORE


def tolerance_table(name, its):
    res = {}
    for model, pred_dir in (("zero_shot_v117", lambda s: zero_shot_dir(name, s)),
                            ("real_only", lambda s: RO / name / f"pred_s{s}")):
        per = [scores(sum(counts(*masks(pred_dir(s) / f"{k}.png", v[1])) for k, v in its.items())) for s in SEEDS]
        res[model] = {m: float(np.mean([p[m] for p in per])) for m in per[0]}
        print(name, model, {m: round(x, 3) for m, x in res[model].items()}, flush=True)
    return res


def label_widths(gt):
    """(skeleton mask, width per skeleton pixel) for a boolean label."""
    sk = skeletonize(gt)
    dist = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 3)
    return sk, np.maximum(1.0, 2 * dist[sk] - 1)


def contrast(img, gt, sk):
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ring = _grow(gt, 8) & ~_grow(gt, 3)
    if not ring.any():
        return np.zeros(sk.sum(), np.float32)
    # Local ring mean: blur the ring luminance and the ring mask over ~25 px.
    num = cv2.blur(np.where(ring, lum, 0), (25, 25))
    den = cv2.blur(ring.astype(np.float32), (25, 25))
    ring_mean = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)
    crack = cv2.blur(np.where(gt, lum, 0), (5, 5)) / np.maximum(cv2.blur(gt.astype(np.float32), (5, 5)), 1e-6)
    c = (ring_mean - crack) / np.maximum(ring_mean, 1.0)
    return np.nan_to_num(c[sk], nan=0.0)


def buildcrack_misses(its):
    rows = {"width": [], "contrast": [], "drone": [], "pred_width": []}
    found = {f"zero_shot_x{s:g}": [] for s in SCALES} | {"real_only": []}
    for k, v in its.items():
        img = cv2.imread(v[0])
        gt = cv2.imread(v[1], 0) == 1
        if not gt.any():
            continue
        sk, w = label_widths(gt)
        rows["width"].append(w)
        rows["contrast"].append(contrast(img, gt, sk))
        rows["drone"].append(np.full(len(w), k.startswith("DJI_")))
        pw = []
        for s in SCALES:
            hits = np.mean([_grow(cv2.imread(str(zero_shot_dir("buildcrack", seed, s) / f"{k}.png"), 0) == 1,
                                  TOLERANCE_PX)[sk] for seed in SEEDS], 0)
            found[f"zero_shot_x{s:g}"].append(hits)
            if s == TEST_SCALE:
                p0 = cv2.imread(str(zero_shot_dir("buildcrack", 0, s) / f"{k}.png"), 0) == 1
                d = cv2.distanceTransform(p0.astype(np.uint8), cv2.DIST_L2, 3)
                pw.append(np.where(p0[sk], np.maximum(1.0, 2 * d[sk] - 1), 0.0))
        found["real_only"].append(np.mean([_grow(cv2.imread(str(RO / f"buildcrack/pred_s{seed}/{k}.png"), 0) == 1,
                                                 TOLERANCE_PX)[sk] for seed in SEEDS], 0))
        rows["pred_width"].append(pw[0])
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    found = {k: np.concatenate(v) for k, v in found.items()}
    n = len(rows["width"])

    def recall(mask):
        return {m: float(f[mask].mean()) for m, f in found.items()} | {"length_share": float(mask.sum() / n)}

    out = {"centreline_px": n, "overall": recall(np.ones(n, bool)), "by_width": [], "by_contrast": [], "by_source": {}}
    for lo, hi in zip(WIDTH_BINS[:-1], WIDTH_BINS[1:]):
        m = (rows["width"] >= lo) & (rows["width"] < hi)
        if m.any():
            hit = m & (rows["pred_width"] > 0)
            out["by_width"].append({"bin_px": f"{lo}-{hi}", **recall(m),
                                    "zero_shot_pred_width_median": float(np.median(rows["pred_width"][hit]))
                                    if hit.any() else None})
    q = np.quantile(rows["contrast"], [0, 1 / 3, 2 / 3, 1])
    for i, name in enumerate(("low", "mid", "high")):
        m = (rows["contrast"] >= q[i]) & (rows["contrast"] <= q[i + 1])
        out["by_contrast"].append({"tertile": name, "contrast_range": [float(q[i]), float(q[i + 1])], **recall(m)})
    for name, m in (("drone", rows["drone"]), ("other", ~rows["drone"])):
        out["by_source"][name] = recall(m) | {"width_median": float(np.median(rows["width"][m]))}
    out["label_width_quantiles"] = {str(p): float(np.quantile(rows["width"], p)) for p in (.1, .25, .5, .75, .9)}
    return out


def train_widths(limit=600):
    files = sorted(TRAIN_LABELS.glob("*.png"))[::max(1, len(sorted(TRAIN_LABELS.glob("*.png"))) // limit)]
    ws = [label_widths(cv2.imread(str(f), 0) == 1)[1] for f in files]
    ws = np.concatenate([w for w in ws if len(w)])
    return {"views": len(files), **{str(p): float(np.quantile(ws, p)) for p in (.1, .25, .5, .75, .9)},
            "share_ge_6px": float((ws >= 6).mean()), "share_ge_10px": float((ws >= 10).mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"tolerance": {}, "protocol": {"seeds": SEEDS, "zero_shot_scale": TEST_SCALE, "real_only_scale": 1.0,
                                            "label_grow_px": GROW, "tolerant_px": TOLS}}
    its = {name: items(name) for name in ("bd3", "cubit", "buildcrack")}
    for name in ("cubit", "buildcrack"):
        infer(name, its[name])
    for s in SCALES:
        infer("buildcrack", its["buildcrack"], s)
    for name in its:
        report["tolerance"][name] = tolerance_table(name, its[name])
    report["buildcrack_misses"] = buildcrack_misses(its["buildcrack"])
    report["train_label_widths_v117"] = train_widths()
    (OUT / "results.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("buildcrack_misses", "train_label_widths_v117")}, indent=1))
    print("IOU_TOLERANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
