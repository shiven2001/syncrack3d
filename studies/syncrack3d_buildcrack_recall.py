"""Which BuildCrack cracks do SynCrack3D models miss, and what did v1.17 lose against v1.8.0?

Every labelled centreline pixel of the 358 BuildCrack tiles is described by
- width: 2 * distance to the label edge - 1 (px at tile scale);
- contrast: (ring - crack) / ring luminance, ring 3-8 px outside the label;
- orientation: from the label's structure tensor (horizontal / diagonal / vertical);
- source: DJI drone photo or not;
and counts as found when a model's prediction (released protocol: 1.5x test-time
upscale) lies within 5 px, the scorers' centreline recall rule. Recall is the
seed mean. Models: v1.8.0 release (5 seeds), v1.17 at 500, 1,000 and 4,000
training walls. Also reports how much length sits in tiles a model misses almost
entirely (tile recall < 0.2), and the recall at a lower crack threshold.

    python -m studies.syncrack3d_buildcrack_recall
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "syncrack3d/eval"))
from infer import TEST_SCALE, _probs, load_model  # noqa: E402
from metrics import TOLERANCE_PX, _grow  # noqa: E402

from studies.syncrack3d_iou_tolerance import contrast, label_widths  # noqa: E402

DT = ROOT / "syncrack3d/outputs/domain_transfer"
RO = DT / "syncrack3d_real_only_v1/buildcrack"
OUT = DT / "syncrack3d_buildcrack_recall_v1"
MODELS = {"v1.8.0": DT / "syncrack3d_facade2_bd3_v1", "v1.17 500 walls": DT / "syncrack3d_v117_unet_w500",
          "v1.17 1000 walls": DT / "syncrack3d_v117_unet_w1000", "v1.17 4000 walls": DT / "syncrack3d_v117_unet_w4000"}
THRESHOLDS = (0.5, 0.3)  # crack + other-damage probability needed to call a pixel crack (0.5 is released)
WIDTH_BINS = (0, 3, 6, 10, 20, 1000)


def crack_prob(model, img, scale=TEST_SCALE):
    """p(crack) where crack beats class 2, as infer.predict, before thresholding."""
    h, w = img.shape[:2]
    up = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    prob = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in _probs(model, up)])
    return np.where(prob[1] >= prob[2], prob[1] + prob[2], 0.0)


def orientation(gt):
    """Crack direction in degrees from horizontal (0-90) at every pixel, from the label's structure tensor."""
    m = cv2.GaussianBlur(gt.astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(m, cv2.CV_32F, 1, 0), cv2.Sobel(m, cv2.CV_32F, 0, 1)
    jxx, jxy, jyy = (cv2.GaussianBlur(a, (0, 0), 5) for a in (gx * gx, gx * gy, gy * gy))
    normal = 0.5 * np.arctan2(2 * jxy, jxx - jyy)  # direction across the crack
    return np.abs(np.degrees(normal + np.pi / 2 - np.pi * (normal > 0)))  # along the crack, folded to 0-90


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = json.loads((RO / "split.json").read_text())["items"]
    tiles = []
    for k, v in sorted(items.items()):
        gt = cv2.imread(v[1], 0) == 1
        if not gt.any():
            continue
        img = cv2.imread(v[0])
        sk, w = label_widths(gt)
        ang = orientation(gt)[sk]
        ang = np.minimum(ang, 180 - ang) if ang.max() > 90 else ang
        tiles.append(dict(key=k, img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), sk=sk, width=w,
                          contrast=contrast(img, gt, sk), angle=ang, drone=k.startswith("DJI_")))
    found = {}
    for name, folder in MODELS.items():
        seeds = sorted(folder.glob("runs/s*/last.pt"))
        per = {t: [] for t in THRESHOLDS}
        for ck in seeds:
            model = load_model(ck)
            hits = {t: [] for t in THRESHOLDS}
            for tile in tiles:
                p = crack_prob(model, tile["img"])
                for t in THRESHOLDS:
                    hits[t].append(_grow(p >= t, TOLERANCE_PX)[tile["sk"]])
            for t in THRESHOLDS:
                per[t].append(hits[t])
            del model
            print(name, ck.parent.name, flush=True)
        found[name] = {t: [np.mean([s[i] for s in per[t]], 0) for i in range(len(tiles))] for t in THRESHOLDS}

    cat = lambda key: np.concatenate([t[key] for t in tiles])  # noqa: E731
    width, con, ang = cat("width"), cat("contrast"), cat("angle")
    drone = np.concatenate([np.full(len(t["width"]), t["drone"]) for t in tiles])
    n = len(width)
    groups = {"all": np.ones(n, bool), "drone tiles": drone, "other tiles": ~drone}
    for lo, hi in zip(WIDTH_BINS[:-1], WIDTH_BINS[1:]):
        groups[f"width {lo}-{hi} px"] = (width >= lo) & (width < hi)
    q = np.quantile(con, [1 / 3, 2 / 3])
    groups.update({"low contrast": con < q[0], "mid contrast": (con >= q[0]) & (con < q[1]),
                   "high contrast": con >= q[1]})
    groups.update({"horizontal (<25 deg)": ang < 25, "diagonal": (ang >= 25) & (ang < 65), "vertical (>65 deg)": ang >= 65})
    report = {"centreline_px": int(n), "contrast_tertiles": q.tolist(), "groups": {}, "missed_tiles": {}}
    for g, m in groups.items():
        row = {"length_share": float(m.mean())}
        for name in MODELS:
            for t in THRESHOLDS:
                row[f"{name} @{t}"] = float(np.concatenate(found[name][t])[m].mean())
        report["groups"][g] = row
    for name in MODELS:
        rec = np.array([f.mean() for f in found[name][0.5]])
        length = np.array([len(f) for f in found[name][0.5]])
        missed = rec < 0.2
        report["missed_tiles"][name] = {"tiles": int(missed.sum()), "of": len(tiles),
                                        "length_share": float(length[missed].sum() / length.sum()),
                                        "drone_share_of_missed_length": float(
                                            length[missed & np.array([t["drone"] for t in tiles])].sum()
                                            / max(1, length[missed].sum()))}
    (OUT / "results.json").write_text(json.dumps(report, indent=1) + "\n")
    names = list(MODELS)
    print(f"{'group':24s} {'share':>6s} " + " ".join(f"{m[:16]:>16s}" for m in names) + "   (threshold 0.5 / 0.3)")
    for g, row in report["groups"].items():
        print(f"{g:24s} {row['length_share']:6.2f} " + " ".join(
            f"{row[f'{m} @0.5']:7.2f}/{row[f'{m} @0.3']:.2f}  " for m in names))
    print(json.dumps(report["missed_tiles"], indent=1))
    print("BUILDCRACK_RECALL_DONE")


if __name__ == "__main__":
    main()
