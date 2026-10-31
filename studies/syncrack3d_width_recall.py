"""Where do the amount-control and host-crack mixes lose crack recall against v1.2.0?

Every labelled crack centreline pixel gets a width (2 * distance to the label
edge - 1, in px at the scored scale) and a contrast ((ring - crack) / ring
luminance, ring 3-8 px outside the label). Straightness (rms distance from the
best-fit line of centreline pixels within 25 px) comes from straightness.npz
when present. A pixel is found when a seed's
prediction lies within 3 px: the centreline-recall rule of
syncrack3d/eval/metrics.py. Covered pixels also get the predicted width there.

CUBIT crack photos (1/3 scale, spalling ignore as in eval/cubit/score.py) and
BuildCrack tiles are re-inferred for all five seeds; BD3 minor_crack uses the
saved per-seed predictions in each model's pred/ folder.
Intervals resample photos (CUBIT), source images (BuildCrack) or duplicate
groups (BD3) and average the five seeds, as the scorers do.

    python -m studies.syncrack3d_width_recall
"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "syncrack3d/eval"
sys.path.insert(0, str(EVAL))
from infer import load_model, predict  # noqa: E402
from metrics import _grow  # noqa: E402

DT = ROOT / "syncrack3d/outputs/domain_transfer"
MODELS = {"v1.2.0": DT / "syncrack3d_layergate_bd3_v1",
          "amount": DT / "syncrack3d_amount_bd3_v1",
          "hostcrack": DT / "syncrack3d_hostcrack_bd3_v1",
          "null": DT / "syncrack3d_straightnull_bd3_v1",
          "straight": DT / "syncrack3d_straight_bd3_v1",
          "tilestone": DT / "syncrack3d_tilestone_bd3_v1"}
OUT = DT / "syncrack3d_width_recall_v1"
WIDTH_BINS = [0, 2, 3, 4, 6, 10, np.inf]
CONTRAST_BINS = [-np.inf, 0.05, 0.1, 0.2, 0.35, np.inf]
STRAIGHT_BINS = [0, 0.5, 1, 2, 4, np.inf]  # rms px off the best-fit line within 25 px; low = straight
N_BOOT = 1000


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def features(img, gt, valid):
    """Centreline pixels of gt inside valid, with width and contrast."""
    g = gt.astype(np.uint8)
    sk = skeletonize(gt) & valid
    ys, xs = np.nonzero(sk)
    width = 2 * cv2.distanceTransform(g, cv2.DIST_L2, 3)[ys, xs] - 1
    lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ring = (cv2.dilate(g, _disk(8)) > 0) & ~(cv2.dilate(g, _disk(3)) > 0) & valid
    r = ring.astype(np.float32)
    bg = cv2.boxFilter(lum * r, -1, (17, 17), normalize=False) / np.maximum(
        cv2.boxFilter(r, -1, (17, 17), normalize=False), 1e-6)
    gf = g.astype(np.float32)
    cr = cv2.boxFilter(lum * gf, -1, (5, 5), normalize=False) / np.maximum(
        cv2.boxFilter(gf, -1, (5, 5), normalize=False), 1e-6)
    has_ring = cv2.boxFilter(r, -1, (17, 17), normalize=False)[ys, xs] > 0
    contrast = np.where(has_ring, (bg[ys, xs] - cr[ys, xs]) / np.maximum(bg[ys, xs], 1.0), np.nan)
    return ys, xs, width.astype(np.float32), contrast.astype(np.float32)


def straightness(ys, xs, radius=25):
    """Rms distance (px) of nearby centreline pixels from their best-fit line."""
    from scipy.spatial import cKDTree

    pts = np.c_[xs, ys].astype(np.float32)
    res = np.full(len(pts), np.nan, np.float32)
    if len(pts) < 5:
        return res
    for i, nb in enumerate(cKDTree(pts).query_ball_point(pts, radius)):
        if len(nb) < 15:
            continue
        q = pts[nb] - pts[nb].mean(0)
        res[i] = np.sqrt(max(np.linalg.eigvalsh(q.T @ q / len(nb))[0], 0))
    return res


def score_pred(pred, ys, xs):
    hit = _grow(pred, 3)[ys, xs]
    pw = 2 * cv2.distanceTransform(pred.astype(np.uint8), cv2.DIST_L2, 3)[ys, xs] - 1
    return hit, np.where(pred[ys, xs], pw, np.nan).astype(np.float32)


def cubit_and_buildcrack():
    cubit = _module("cubit_score", EVAL / "cubit/score.py")
    build = _module("buildcrack_score", EVAL / "buildcrack/score.py")
    sets = {"cubit": [], "buildcrack": []}
    for k, _split, imp, lbp in cubit.pairs(cubit.CUBIT):
        if not k.startswith("CR"):
            continue
        img = cv2.cvtColor(cv2.imread(str(imp)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(img, None, fx=cubit.SCALE, fy=cubit.SCALE, interpolation=cv2.INTER_AREA)
        gt, ign = cubit.rasterise(lbp, *im.shape[:2])
        valid = ~ign
        sets["cubit"].append({"id": k, "group": k, "img": im, "mask": valid,
                              "f": features(im, (gt > 0) & valid, valid)})
    for p in sorted((build.DATA / "images").glob("*.png")):
        gt = cv2.imread(str(build.DATA / "labels" / p.name), 0) > 0
        if not gt.any():
            continue
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        valid = np.ones(gt.shape, bool)
        sets["buildcrack"].append({"id": p.stem, "group": build.source_of(p.stem), "img": im, "mask": valid,
                                   "f": features(im, gt, valid)})
    return sets


def bd3():
    photos = [p for p in json.loads((EVAL / "bd3/manifest.json").read_text())["photos"]
              if p["folder"] == "minor_crack"]
    out = []
    for p in photos:
        raw = cv2.imread(str(EVAL / "bd3" / p["gt"]), 0)
        img = cv2.cvtColor(cv2.imread(str(EVAL / "bd3" / p["image"])), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, raw.shape[::-1], interpolation=cv2.INTER_AREA)
        valid = raw != 255
        out.append({"id": p["id"], "group": p["group"], "f": features(img, raw == 1, valid)})
    return out


def collect(models):
    """Per-pixel features for every set, and hits for the named models only."""
    import torch

    sets = cubit_and_buildcrack()
    sets["bd3"] = bd3()
    hits = {ds: {m: [] for m in models} for ds in sets}
    pwid = {ds: {m: [] for m in models} for ds in sets}
    for name in models:
        folder = MODELS[name]
        for ck in sorted(folder.glob("runs/s*/last.pt")):
            model = load_model(ck)
            for ds in ("cubit", "buildcrack"):
                h, w = [], []
                for ph in sets[ds]:
                    pred = (predict(model, ph["img"]) == 1) & ph["mask"]
                    a, b = score_pred(pred, *ph["f"][:2])
                    h.append(a)
                    w.append(b)
                hits[ds][name].append(np.concatenate(h))
                pwid[ds][name].append(np.concatenate(w))
            del model
            torch.cuda.empty_cache()
            h, w = [], []
            for ph in sets["bd3"]:
                pred = cv2.imread(str(folder / "pred" / ck.parent.name / f"{ph['id']}.png"), 0) == 1
                a, b = score_pred(pred, *ph["f"][:2])
                h.append(a)
                w.append(b)
            hits["bd3"][name].append(np.concatenate(h))
            pwid["bd3"][name].append(np.concatenate(w))
            print("done", name, ck.parent.name, flush=True)
    arrays = {}
    for ds, photos in sets.items():
        groups = sorted({ph["group"] for ph in photos})
        gidx = {g: i for i, g in enumerate(groups)}
        arrays[f"{ds}_group"] = np.concatenate([np.full(len(ph["f"][0]), gidx[ph["group"]], np.int32) for ph in photos])
        arrays[f"{ds}_width"] = np.concatenate([ph["f"][2] for ph in photos])
        arrays[f"{ds}_contrast"] = np.concatenate([ph["f"][3] for ph in photos])
        arrays[f"{ds}_photos"] = np.array(len(photos))
        for m in models:
            arrays[f"{ds}_hit_{m}"] = np.stack(hits[ds][m])
            arrays[f"{ds}_predwidth_{m}"] = np.stack(pwid[ds][m]).astype(np.float16)
    return arrays


def table(a, ds, key, edges, rng):
    group, val = a[f"{ds}_group"], a[f"{ds}_{key}"]
    ok = ~np.isnan(val)
    b = np.digitize(val[ok], edges) - 1
    g = group[ok]
    n_g, n_b = int(group.max()) + 1, len(edges) - 1
    tot = np.zeros((n_g, n_b))
    np.add.at(tot, (g, b), 1)
    hit = {}
    for m in MODELS:
        h = a[f"{ds}_hit_{m}"][:, ok].astype(float)
        hit[m] = np.zeros((h.shape[0], n_g, n_b))
        for s in range(h.shape[0]):
            np.add.at(hit[m][s], (g, b), h[s])
    draws = rng.integers(0, n_g, (N_BOOT, n_g))
    rows = []
    for j in range(n_b):
        row = {"bin": f"{edges[j]:g}-{edges[j + 1]:g}", "length_share": float(tot[:, j].sum() / tot.sum())}
        for m in MODELS:
            row[m] = float(np.mean(hit[m][:, :, j].sum(1) / max(1, tot[:, j].sum())))
            pw = a[f"{ds}_predwidth_{m}"][:, ok].astype(np.float32)[:, b == j]
            row[f"{m}_predwidth"] = float(np.nanmedian(pw)) if np.isfinite(pw).any() else None
        t_draw = tot[draws, j].sum(1)
        base = (hit["v1.2.0"][:, draws, j].sum(2) / np.maximum(1, t_draw)).mean(0)
        for m in list(MODELS)[1:]:
            d = (hit[m][:, draws, j].sum(2) / np.maximum(1, t_draw)).mean(0) - base
            row[f"{m}_minus_v1.2.0"] = [row[m] - row["v1.2.0"], *map(float, np.percentile(d, [2.5, 97.5]))]
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "pixels.npz"
    a = dict(np.load(cache)) if cache.exists() else {}
    missing = [m for m in MODELS if f"bd3_hit_{m}" not in a]
    if missing:
        new = collect(missing)
        for k, v in new.items():
            if k in a and "_hit_" not in k and "_predwidth_" not in k and not np.array_equal(a[k], v, equal_nan=True):
                raise RuntimeError(f"cached {k} differs from a fresh run")
        a.update(new)
        np.savez_compressed(cache, **a)
    if not (OUT / "straightness.npz").exists():
        sets = cubit_and_buildcrack()
        sets["bd3"] = bd3()
        np.savez_compressed(OUT / "straightness.npz", **{
            f"{ds}_straight": np.concatenate([straightness(*ph["f"][:2]) for ph in photos])
            for ds, photos in sets.items()})
    a.update(np.load(OUT / "straightness.npz"))
    rng = np.random.default_rng(0)
    res = {}
    for ds in ("bd3", "cubit", "buildcrack"):
        res[ds] = {
            "photos": int(a[f"{ds}_photos"]), "centreline_px": int(len(a[f"{ds}_width"])),
            "recall": {m: float(a[f"{ds}_hit_{m}"].mean()) for m in MODELS},
            "gt_width_median": float(np.median(a[f"{ds}_width"])),
            "by_width": table(a, ds, "width", WIDTH_BINS, rng),
            "by_contrast": table(a, ds, "contrast", CONTRAST_BINS, rng),
        }
        if f"{ds}_straight" in a:
            res[ds]["by_straightness"] = table(a, ds, "straight", STRAIGHT_BINS, rng)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    for ds, r in res.items():
        print(f"\n== {ds}: {r['photos']} photos, {r['centreline_px']} centreline px, median gt width "
              f"{r['gt_width_median']:.1f}, recall " + " ".join(f"{m} {v:.3f}" for m, v in r["recall"].items()))
        for key in ("by_width", "by_contrast", "by_straightness"):
            if key not in r:
                continue
            print(f"  {key:11} share  " + " ".join(f"{m[:8]:>8}" for m in MODELS)
                  + "".join(f"   {m[:8] + '-v1.2':>13} [95%]     " for m in list(MODELS)[1:]))
            for row in r[key]:
                diffs = "".join(f"   {row[f'{m}_minus_v1.2.0'][0]:+.3f} [{row[f'{m}_minus_v1.2.0'][1]:+.3f},"
                                f"{row[f'{m}_minus_v1.2.0'][2]:+.3f}]" for m in list(MODELS)[1:])
                print(f"  {row['bin']:11} {row['length_share']:.3f} " + " ".join(f"{row[m]:8.3f}" for m in MODELS) + diffs)


if __name__ == "__main__":
    main()
