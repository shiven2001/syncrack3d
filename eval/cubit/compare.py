"""Paired CUBIT-InSeg comparison of two 1/3-scale crack runs.

Difference B - A, averaged over each model's seeds, with a 95% photo bootstrap
(the same resampled photos for both models). CUBIT has no scene groups in the
local copy, so photos are the unit. Counts come from eval/cubit/score.py.
Older count files that also stored a full-resolution pass are read at the
1/3-scale arrays only.

    python eval/cubit/compare.py --a path/to/a --b path/to/b --out compare.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import metrics

METRICS = ("centreline_f1", "centreline_precision", "centreline_recall", "cldice", "crack_iou")


def protocol(z):
    """(tolerance_px, test_scale); count files written before protocol v2 are (3, 1)."""
    return (int(z["tolerance_px"]), float(z["test_scale"])) if "tolerance_px" in z else (3, 1.0)


def load(path):
    z = np.load(path / "counts.npz")
    seeds = list(z["seeds"])
    missing = [s for s in seeds if f"third_{s}" not in z]
    if missing:
        raise SystemExit(f"{path} has no 1/3-scale counts for {missing}")
    return list(z["ids"]), [z[f"third_{s}"] for s in seeds], protocol(z)


def mean_metrics(per_seed, idx):
    per = [metrics(c[idx].sum(0)) for c in per_seed]
    return {m: float(np.mean([p[m] for p in per])) for m in METRICS}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ids, a_counts, pa = load(args.a)
    ids_b, b_counts, pb = load(args.b)
    if pa != pb:
        raise SystemExit(f"protocols differ: {args.a} {pa} vs {args.b} {pb}; rescore with the same --tolerance/--scale")
    if ids != ids_b:
        raise SystemExit("the two runs do not list the same photos")
    subsets = {
        "crack_photos_CR": np.array([i for i, k in enumerate(ids) if str(k).startswith("CR")]),
        "all_labelled": np.arange(len(ids)),
    }
    rng = np.random.default_rng(0)
    res = {"a": str(args.a), "b": str(args.b), "scale": "1/3", "bootstrap": args.n,
           "protocol": {"tolerance_px": pa[0], "test_scale": pa[1]}, "results": {}}
    for sub, idx in subsets.items():
        ma, mb = mean_metrics(a_counts, idx), mean_metrics(b_counts, idx)
        boot = []
        for _ in range(args.n):
            drawn = rng.choice(idx, len(idx))
            xa, xb = mean_metrics(a_counts, drawn), mean_metrics(b_counts, drawn)
            boot.append([xb[m] - xa[m] for m in METRICS])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        seeds_b = [metrics(c[idx].sum(0)) for c in b_counts]
        seeds_a = [metrics(c[idx].sum(0)) for c in a_counts]
        res["results"][sub] = {
            m: {
                "a": ma[m],
                "b": mb[m],
                "diff": mb[m] - ma[m],
                "ci95": [float(lo[j]), float(hi[j])],
                "b_seeds_above_a_mean": int(sum(s[m] > ma[m] for s in seeds_b)),
                "a_sd": float(np.std([s[m] for s in seeds_a], ddof=1)),
                "b_sd": float(np.std([s[m] for s in seeds_b], ddof=1)),
            }
            for j, m in enumerate(METRICS)
        }
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["results"].items():
        print(k)
        for m, x in v.items():
            print(f"  {m:22s} A {x['a']:.3f}  B {x['b']:.3f}  diff {x['diff']:+.3f} "
                  f"[{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]  B seeds>A {x['b_seeds_above_a_mean']}/{len(seeds_b)}")


if __name__ == "__main__":
    main()
