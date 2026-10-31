"""Paired BD3 comparison of SynCrack3D crack runs against the first model in MODELS.

Rescores every model's per-seed predictions (PRED/s*/<id>.png, written by
syncrack3d/eval/bd3/predict.py) with syncrack3d/eval/bd3/score.py's counts at its
default tolerance (protocol v2: 5 px), so all runs share one scorer.
Minor-crack metrics pool counts over photos; plain false alarm is the share of
plain photos with predicted crack above 0.1%. Intervals resample duplicate
groups (both folders) and the five seeds, then average the drawn seeds.

    python -m studies.syncrack3d_bd3_compare
"""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BD3 = ROOT / "syncrack3d/eval/bd3"
DT = ROOT / "syncrack3d/outputs/domain_transfer"
MODELS = {"v1.2.0": "syncrack3d_layergate_bd3_v1", "amount": "syncrack3d_amount_bd3_v1",
          "hostcrack": "syncrack3d_hostcrack_bd3_v1", "null": "syncrack3d_straightnull_bd3_v1",
          "straight": "syncrack3d_straight_bd3_v1",
          "tilestone": "syncrack3d_tilestone_bd3_v1"}
OUT = DT / "syncrack3d_bd3_compare_v1.json"
PRED = "pred_v2"  # eval/bd3/predict.py output (protocol v2); "pred" holds the training-time v1 maps
KEYS = ("crack_iou", "centreline_f1", "centreline_precision", "centreline_recall", "plain_false_alarm")
N_BOOT = 1000


def _scorer():
    spec = importlib.util.spec_from_file_location("bd3_score", BD3 / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    sc = _scorer()
    photos = sc.manifest()
    minor = [p for p in photos if p["folder"] == "minor_crack"]
    plain = [p for p in photos if p["folder"] == "plain"]
    groups = sorted({p["group"] for p in photos})
    gidx = {g: i for i, g in enumerate(groups)}
    gts = {p["id"]: cv2.imread(str(BD3 / p["gt"]), 0) for p in minor}
    counts, alarms = {}, {}
    for name, folder in MODELS.items():
        seeds = sorted((DT / folder / PRED).glob("s*"))
        c = np.zeros((len(seeds), len(groups), 11), np.int64)
        a = np.zeros((len(seeds), len(groups), 2))
        for s, d in enumerate(seeds):
            for p in minor:
                c[s, gidx[p["group"]]] += sc.counts(cv2.imread(str(d / f"{p['id']}.png"), 0), gts[p["id"]])
            for p in plain:
                pred = cv2.imread(str(d / f"{p['id']}.png"), 0)
                a[s, gidx[p["group"]]] += [float((pred == 1).mean() > sc.ALARM_FRACTION), 1.0]
        counts[name], alarms[name] = c, a
        print("scored", name, len(seeds), "seeds", flush=True)

    def score(name, seed_idx, group_idx):
        vals = []
        for s in seed_idx:
            m = sc.metrics(counts[name][s, group_idx].sum(0))
            al = alarms[name][s, group_idx].sum(0)
            vals.append([m[k] for k in KEYS[:-1]] + [al[0] / max(1.0, al[1])])
        return np.mean(vals, 0)

    all_s, all_g = np.arange(5), np.arange(len(groups))
    point = {m: score(m, all_s, all_g) for m in MODELS}
    rng = np.random.default_rng(0)
    baseline = list(MODELS)[0]
    boot = {m: [] for m in list(MODELS)[1:]}
    for _ in range(N_BOOT):
        g = rng.integers(0, len(groups), len(groups))
        s = rng.integers(0, 5, 5)
        base = score(baseline, s, g)
        for m in boot:
            boot[m].append(score(m, s, g) - base)
    res = {"photos": {"minor_crack": len(minor), "plain": len(plain)}, "groups": len(groups),
           "mean_of_seeds": {m: dict(zip(KEYS, map(float, v))) for m, v in point.items()},
           "baseline": baseline,
           "minus_baseline": {m: {k: [float(point[m][j] - point[baseline][j]),
                                    *map(float, np.percentile(np.array(b)[:, j], [2.5, 97.5]))]
                                for j, k in enumerate(KEYS)} for m, b in boot.items()}}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'':10}" + "".join(f"{k:>22}" for k in KEYS))
    for m in MODELS:
        print(f"{m:10}" + "".join(f"{res['mean_of_seeds'][m][k]:22.3f}" for k in KEYS))
    for m, d in res["minus_baseline"].items():
        print(f"{m + '-' + baseline:10}" + "".join(f"  {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]" for v in d.values()))


if __name__ == "__main__":
    main()
