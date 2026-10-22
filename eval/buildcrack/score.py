"""Score SynCrack3D crack checkpoints on BuildCrack tiles.

358 paired 448 px tiles. Labels are 0/255. Crack is prediction class 1.
Centreline F1 is pooled (protocol v2: 5 px tolerance, 1.5x test-time upscaling). Empty tiles stay in the pool, and a plain
tile is an alarm when predicted crack pixels exceed 0.1%.

Tiles cut from one photo are one bootstrap unit: a numeric prefix, or a DJI
id. DJI frame index and tile offsets are not separate photos. Each bootstrap
draw resamples source images and training seeds, then averages the drawn seed
scores. The vote marks a pixel as crack when a majority of the seeds do, and
its interval resamples source images only.

    python eval/buildcrack/score.py \\
        --model v1.2.0=path/to/checkpoint_dir --out path/to/out
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer import TEST_SCALE, load_model, predict
from metrics import TOLERANCE_PX, centreline_counts, centreline_scores, cldice_counts

DATA = Path(__file__).resolve().parents[3] / "BuildCrack"
KEYS = ("centreline_f1", "centreline_precision", "centreline_recall", "cldice", "crack_iou")
N_BOOT = 1000
ALARM_FRACTION = 0.001


def source_of(stem: str) -> str:
    if stem.startswith("DJI_"):
        return "_".join(stem.split("_")[:2])
    return stem.split("_")[0]


def one(pred, gt, tol=TOLERANCE_PX):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    return np.r_[tp, fp, fn, centreline_counts(p, g, tol), cldice_counts(p, g)].astype(np.int64)


def metrics(total):
    cl = centreline_scores(total[3:7])
    t = total[7:]
    tp_ = t[0] / max(1, t[1])
    ts_ = t[2] / max(1, t[3])
    return {
        "centreline_f1": float(cl["f1"]),
        "centreline_precision": float(cl["precision"]),
        "centreline_recall": float(cl["recall"]),
        "cldice": float(2 * tp_ * ts_ / max(1e-9, tp_ + ts_)),
        "crack_iou": float(total[0] / max(1, total[0] + total[1] + total[2])),
    }


def pooled(counts, idx):
    if len(idx) == 0:
        return metrics(np.zeros(11, np.int64))
    tiles, mult = np.unique(idx, return_counts=True)
    return metrics((counts[tiles] * mult[:, None]).sum(0))


def parse_model(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("--model expects name=directory")
    name, folder = text.split("=", 1)
    if not name or not folder:
        raise argparse.ArgumentTypeError("--model expects name=directory")
    return name, Path(folder)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--model", action="append", type=parse_model, required=True,
                    help="name=directory, directory contains runs/s*/last.pt; repeat for comparisons")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    ap.add_argument("--tolerance", type=int, default=TOLERANCE_PX, help="centreline tolerance in px (v1: 3)")
    ap.add_argument("--scale", type=float, default=TEST_SCALE, help="test-time upscaling (v1: 1)")
    args = ap.parse_args()
    import torch

    names = [name for name, _ in args.model]
    if len(names) != len(set(names)):
        raise SystemExit("model names must be unique")
    stems = sorted(p.stem for p in (args.data / "images").glob("*.png"))
    if not stems:
        raise SystemExit(f"no tiles under {args.data / 'images'}")
    groups = {}
    for i, stem in enumerate(stems):
        groups.setdefault(source_of(stem), []).append(i)
    group_ids = sorted(groups)
    group_index = [np.array(groups[g], np.int32) for g in group_ids]
    images, gts, empty = [], [], []
    for stem in stems:
        images.append(cv2.cvtColor(cv2.imread(str(args.data / "images" / f"{stem}.png")), cv2.COLOR_BGR2RGB))
        gt = cv2.imread(str(args.data / "labels" / f"{stem}.png"), 0) > 0
        gts.append(gt)
        empty.append(not bool(gt.any()))
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    counts, ensemble_counts, saved = {}, {}, {}
    report = {
        "tiles": len(stems),
        "positive_tiles": int((~np.array(empty)).sum()),
        "empty_tiles": int(np.array(empty).sum()),
        "source_images": len(group_ids),
        "bootstrap": args.bootstrap,
        "protocol": {"tolerance_px": args.tolerance, "test_scale": args.scale},
        "grouping": "numeric prefix, or DJI id; tile offsets are not separate photos",
        "models": {},
        "differences": {},
    }
    for name, folder in args.model:
        seeds = sorted(folder.glob("runs/s*/last.pt"))
        if not seeds:
            raise SystemExit(f"no runs/s*/last.pt under {folder}")
        per_seed = []
        votes = [np.zeros(gt.shape, np.uint8) for gt in gts]
        alarms = []
        for ck in seeds:
            model = load_model(ck)
            rows, seed_alarms = [], []
            for i, (img, gt) in enumerate(zip(images, gts)):
                pred = predict(model, img, args.scale) == 1
                votes[i] += pred.astype(np.uint8)
                rows.append(one(pred, gt, args.tolerance))
                if empty[i]:
                    seed_alarms.append(float(pred.mean() > ALARM_FRACTION))
            per_seed.append(np.stack(rows))
            alarms.append(seed_alarms)
            del model
            torch.cuda.empty_cache()
            print(name, ck.parent.name, "f1", round(metrics(per_seed[-1].sum(0))["centreline_f1"], 3), flush=True)
        stacked = np.stack(per_seed)
        counts[name] = stacked
        saved[name] = stacked
        majority = len(seeds) // 2 + 1
        ens = np.stack([one(votes[i] >= majority, gts[i], args.tolerance) for i in range(len(stems))])
        ensemble_counts[name] = ens
        seed_scores = [metrics(stacked[s].sum(0)) for s in range(stacked.shape[0])]
        f1s = [s["centreline_f1"] for s in seed_scores]
        report["models"][name] = {
            "checkpoint": str(folder),
            "seeds": len(seeds),
            "mean_of_seeds": {k: float(np.mean([s[k] for s in seed_scores])) for k in KEYS},
            "seed_f1": f1s,
            "seed_f1_spread": float(max(f1s) - min(f1s)),
            "ensemble": metrics(ens.sum(0)),
            "empty_tile_alarm_rate": float(np.mean(alarms)) if any(empty) else 0.0,
        }
        print(name, "vote f1", round(report["models"][name]["ensemble"]["centreline_f1"], 3), flush=True)

    n_seeds = {report["models"][name]["seeds"] for name in names}
    if len(n_seeds) != 1:
        raise SystemExit(f"models have different seed counts: {n_seeds}")
    n_seed = n_seeds.pop()
    rng = np.random.default_rng(0)

    def mean_of_drawn_seeds(name, seed_draw, tile_idx):
        vals = [pooled(counts[name][s], tile_idx) for s in seed_draw]
        return {k: float(np.mean([v[k] for v in vals])) for k in KEYS}

    for i, left in enumerate(names):
        for right in names[:i]:
            point = {k: report["models"][left]["mean_of_seeds"][k] - report["models"][right]["mean_of_seeds"][k] for k in KEYS}
            ens_point = {k: report["models"][left]["ensemble"][k] - report["models"][right]["ensemble"][k] for k in KEYS}
            boot, ens_boot = [], []
            for _ in range(args.bootstrap):
                drawn_groups = rng.choice(len(group_index), len(group_index), replace=True)
                tile_idx = np.concatenate([group_index[g] for g in drawn_groups])
                seed_draw = rng.choice(n_seed, n_seed, replace=True)
                a = mean_of_drawn_seeds(right, seed_draw, tile_idx)
                b_ = mean_of_drawn_seeds(left, seed_draw, tile_idx)
                boot.append([b_[k] - a[k] for k in KEYS])
                ea, eb = pooled(ensemble_counts[right], tile_idx), pooled(ensemble_counts[left], tile_idx)
                ens_boot.append([eb[k] - ea[k] for k in KEYS])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            elo, ehi = np.percentile(ens_boot, [2.5, 97.5], axis=0)
            report["differences"][f"{left} - {right}"] = {
                "mean_of_seeds": {k: {"diff": point[k], "ci95": [float(lo[j]), float(hi[j])]} for j, k in enumerate(KEYS)},
                "ensemble": {k: {"diff": ens_point[k], "ci95": [float(elo[j]), float(ehi[j])]} for j, k in enumerate(KEYS)},
            }
            print(f"\n{left} - {right}", flush=True)
            for j, k in enumerate(KEYS):
                print(f"  seeds {k:22} {point[k]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}]", flush=True)
                print(f"  vote  {k:22} {ens_point[k]:+.3f} [{elo[j]:+.3f}, {ehi[j]:+.3f}]", flush=True)

    np.savez_compressed(args.out / "counts.npz", stems=np.array(stems), empty=np.array(empty),
                        **{name: saved[name] for name in names})
    (args.out / "results.json").write_text(json.dumps(report, indent=1) + "\n")
    print("BUILDCRACK_DONE", flush=True)


if __name__ == "__main__":
    main()
