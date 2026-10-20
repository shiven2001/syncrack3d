"""Score crack predictions on the frozen BD3 crack-only set in this folder.

487 minor_crack photos: pooled centreline F1 at 3 px, clDice and crack IoU.
355 plain photos: false-alarm rate. A plain photo is an alarm when predicted
crack pixels exceed 0.1% of the image. Peeling photos are not in this set.

Predictions are uint8 class maps named <id>.png. Crack is class 1. Class 2
is not crack.

    python eval/bd3/score.py --pred path/to/pred_dir
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import centreline_counts, centreline_scores, cldice_counts

HERE = Path(__file__).resolve().parent
ALARM_FRACTION = 0.001


def manifest():
    data = json.loads((HERE / "manifest.json").read_text())
    photos = data["photos"]
    assert sum(p["folder"] == "minor_crack" for p in photos) == 487
    assert sum(p["folder"] == "plain" for p in photos) == 355
    return photos


def counts(pred, gt):
    p, g = pred == 1, gt == 1
    valid = gt != 255
    tp = int((p & g & valid).sum())
    fp = int((p & ~g & valid).sum())
    fn = int((~p & g & valid).sum())
    return np.r_[tp, fp, fn, centreline_counts(p, g, 3), cldice_counts(p, g)]


def metrics(total):
    cl = centreline_scores(total[3:7])
    t = total[7:]
    tp_ = t[0] / max(1, t[1])
    ts_ = t[2] / max(1, t[3])
    return {
        "centreline_f1_3px": cl["f1"],
        "centreline_precision": cl["precision"],
        "centreline_recall": cl["recall"],
        "cldice": float(2 * tp_ * ts_ / max(1e-9, tp_ + ts_)),
        "crack_iou": float(total[0] / max(1, total[0] + total[1] + total[2])),
    }


def score(pred_dir):
    pred_dir = Path(pred_dir)
    photos = manifest()
    total = np.zeros(11, np.int64)
    alarms = 0
    plain = 0
    for photo in photos:
        path = pred_dir / f"{photo['id']}.png"
        pred = cv2.imread(str(path), 0)
        if pred is None:
            raise FileNotFoundError(path)
        gt = cv2.imread(str(HERE / photo["gt"]), 0)
        if pred.shape != gt.shape:
            raise ValueError(f"{photo['id']}: pred {pred.shape} != gt {gt.shape}")
        if photo["folder"] == "minor_crack":
            total += counts(pred, gt)
        else:
            plain += 1
            alarms += float((pred == 1).mean() > ALARM_FRACTION)
    out = metrics(total)
    out["plain_false_alarms"] = alarms / plain
    out["minor_crack_photos"] = 487
    out["plain_photos"] = plain
    out["groups"] = len({p["group"] for p in photos})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="directory of <id>.png class maps; crack is 1")
    ap.add_argument("--out", help="optional JSON path")
    args = ap.parse_args()
    out = score(args.pred)
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
