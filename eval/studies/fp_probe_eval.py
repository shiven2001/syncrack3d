"""Does the SynCrack3D v1.2.0 crack model make false positives on synthetic walls?

Probe set from fp_probe_render.py (outputs/walls/syncrack3d_fp_probe_v1):
  indoor_default  unseen walls, training-like soft lighting, up to 3 props
  indoor_hard     the same walls, props and cameras, a hard_mix rig per view
  outdoor         unseen facades, sun or overcast, eave/downpipe/corner
Model: syncrack3d_layergate_bd3_v1 (v1.2.0), five seeds, 640 px as trained.

False positive (FP) = predicted crack more than 3 px from labelled crack and
outside the ignore mask. Each FP pixel gets one cause, in this order:
  prop / outdoor_structure / other_scene   it lies on that object, not the wall
  object_edge     wall pixel within 4 px of an object silhouette or corner
  cast_edge       wall pixel within 4 px of a prop/eave/corner/downpipe shadow boundary
  in_cast         inside such a shadow
  wall_surface    anywhere else on the wall (texture, joints, relief, light patches)
A view alarms when FP pixels exceed 0.1% of the image (the BD3 plain-wall rule).
"Consistent" FPs are pixels predicted by at least 3 of 5 seeds.

Light dependence (indoor_default vs indoor_hard, same seed and rig): the share of
consistent hard-light FP pixels that are also FP (within 3 px) under default light.
A low share means the FP comes from the light and a relight removes it.
Note: syncrack3d/syncrack3d/material_diversity.py was edited at 15:17 on
2026-10-18, during indoor_default (wood walls 60% clean instead of always).
indoor_hard and outdoor used the new rule. Only seed 196009 differs between the
indoor batches, and only views with identical camera and defects are paired.

Run after: python eval/studies/fp_probe_exr.py <batch dirs>  (Blender env).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP / "eval"))
from infer import load_model, predict  # noqa: E402
from metrics import centreline_counts, centreline_scores  # noqa: E402

PROBE = APP / "outputs/walls/syncrack3d_fp_probe_v1"
MODEL = APP / "outputs/domain_transfer/syncrack3d_layergate_bd3_v1"
OUT = APP / "outputs/domain_transfer/syncrack3d_fp_probe_v1"
BATCHES = ("indoor_default", "indoor_hard", "outdoor")
CAUSES = ("prop", "outdoor_structure", "other_scene", "object_edge", "cast_edge", "in_cast", "wall_surface")
STEM = re.compile(r"seed_(\d+)_rig(\d+)_")


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def views(batch):
    pack = PROBE / batch / "pack"
    out = []
    for img in sorted((pack / "images").glob("*.png")):
        seed, rig = map(int, STEM.match(img.stem).groups())
        scene = json.loads((PROBE / batch / f"seed_{seed}" / "scene.json").read_text())
        light = scene["view_lighting"][rig]["regime"] if scene.get("view_lighting") else scene["lighting"]["regime"]
        out.append({"batch": batch, "stem": img.stem, "seed": seed, "rig": rig, "image": img,
                    "mask": pack / "masks" / img.name, "ignore": pack / "ignore_masks" / img.name,
                    "maps": PROBE / batch / "shadow_probe" / f"seed_{seed}" / f"{rig}_maps.npz",
                    "light": light, "material": scene["wall_material"]["diverse"]["category"],
                    "clean_wall": bool(scene["wall_material"]["clean_wall"]),
                    "props": len(scene["props"]), "camera": scene["cameras"][rig],
                    "defects": scene["defects_placed"]})
    return out


def regions(v):
    raw = cv2.imread(str(v["mask"]), 0)
    crack = raw == 60
    ign = cv2.imread(str(v["ignore"]), 0) if v["ignore"].exists() else np.zeros_like(raw)
    excluded = cv2.dilate(crack.astype(np.uint8), disk(3)).astype(bool) | ((ign > 0) & ~crack)
    m = np.load(v["maps"])
    idx, cast = m["index"], m["cast"]
    wall = (idx == 1) | (idx == 2)
    silhouette = (cv2.dilate(idx, disk(1)) != cv2.erode(idx, disk(1)))
    near_obj = cv2.dilate(silhouette.astype(np.uint8), disk(4)).astype(bool)
    c8 = cast.astype(np.uint8)
    cast_bd = cv2.dilate(c8, disk(4)).astype(bool) & ~cv2.erode(c8, disk(4)).astype(bool)
    cause = np.full(raw.shape, len(CAUSES) - 1, np.uint8)  # wall_surface
    cause[wall & cast] = CAUSES.index("in_cast")
    cause[wall & cast_bd] = CAUSES.index("cast_edge")
    cause[wall & near_obj] = CAUSES.index("object_edge")
    cause[idx == 0] = CAUSES.index("other_scene")
    cause[idx == 4] = CAUSES.index("outdoor_structure")
    cause[idx == 3] = CAUSES.index("prop")
    return crack, excluded, cause


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    import torch

    out = a.out
    (out / "pred").mkdir(parents=True, exist_ok=True)
    allv = [v for bt in BATCHES for v in views(bt)]
    ckpts = sorted(a.model.glob("runs/s*/last.pt"))
    votes = {v["batch"] + "/" + v["stem"]: np.zeros((640, 640), np.uint8) for v in allv}
    rows = []
    cl = defaultdict(lambda: np.zeros(4))
    for ck in ckpts:
        model = load_model(ck)
        for v in allv:
            key = v["batch"] + "/" + v["stem"]
            img = cv2.cvtColor(cv2.imread(str(v["image"])), cv2.COLOR_BGR2RGB)
            pred = predict(model, img, scale=1) == 1
            votes[key] += pred
            crack, excluded, cause = regions(v)
            fp = pred & ~excluded
            cl[(v["batch"], ck.parent.name)] += centreline_counts(pred, crack, 3)
            rows.append({"batch": v["batch"], "stem": v["stem"], "model_seed": ck.parent.name,
                         "fp_px": int(fp.sum()), "alarm": bool(fp.sum() > 0.001 * fp.size),
                         **{f"fp_{c}": int((fp & (cause == i)).sum()) for i, c in enumerate(CAUSES)},
                         **{f"area_{c}": int(((cause == i) & ~excluded).sum()) for i, c in enumerate(CAUSES)}})
        del model
        torch.cuda.empty_cache()
        print("done", ck.parent.name, flush=True)
    meta = {v["batch"] + "/" + v["stem"]: v for v in allv}
    for k, vote in votes.items():
        cv2.imwrite(str(out / "pred" / (k.replace("/", "__") + ".png")), (vote * 50).astype(np.uint8))

    res = {"model": str(a.model), "seeds": [c.parent.name for c in ckpts], "batches": {}}
    for bt in BATCHES:
        R = [r for r in rows if r["batch"] == bt]
        V = [v for v in allv if v["batch"] == bt]
        per_seed_alarm = [np.mean([r["alarm"] for r in R if r["model_seed"] == s.parent.name]) for s in ckpts]
        fp_tot = {c: sum(r[f"fp_{c}"] for r in R) for c in CAUSES}
        area_tot = {c: sum(r[f"area_{c}"] for r in R) for c in CAUSES}
        tot = max(1, sum(fp_tot.values()))
        base = fp_tot["wall_surface"] / max(1, area_tot["wall_surface"])
        cons, cons_cause = 0, defaultdict(int)
        alarm_light, alarm_mat, alarm_clean = defaultdict(list), defaultdict(list), defaultdict(list)
        for v in V:
            key = bt + "/" + v["stem"]
            crack, excluded, cause = regions(v)
            c = (votes[key] >= 3) & ~excluded
            cons += c.sum()
            for i, name in enumerate(CAUSES):
                cons_cause[name] += int((c & (cause == i)).sum())
            al = [r["alarm"] for r in R if r["stem"] == v["stem"]]
            alarm_light[v["light"]].append(np.mean(al))
            alarm_mat[v["material"]].append(np.mean(al))
            alarm_clean["clean_wall" if v["clean_wall"] else "cracked_wall"].append(np.mean(al))
        cs = [centreline_scores(cl[(bt, s.parent.name)]) for s in ckpts]
        res["batches"][bt] = {
            "views": len(V), "walls": len({v["seed"] for v in V}),
            "clean_wall_views": sum(v["clean_wall"] for v in V),
            "views_with_cast_shadow": int(sum(np.load(v["maps"])["cast"].mean() > 0.005 for v in V)),
            "alarm_rate": [float(np.mean(per_seed_alarm)), float(np.std(per_seed_alarm, ddof=1))],
            "fp_share_by_cause": {c: fp_tot[c] / tot for c in CAUSES},
            "fp_rate_vs_wall_surface": {c: (fp_tot[c] / max(1, area_tot[c])) / max(1e-12, base) for c in CAUSES},
            "area_share_by_cause": {c: area_tot[c] / max(1, sum(area_tot.values())) for c in CAUSES},
            "consistent_fp_px_per_view": float(cons / max(1, len(V))),
            "consistent_fp_share_by_cause": {c: cons_cause[c] / max(1, cons) for c in CAUSES},
            "alarm_by_light": {k: [float(np.mean(x)), len(x)] for k, x in sorted(alarm_light.items())},
            "alarm_by_material": {k: [float(np.mean(x)), len(x)] for k, x in sorted(alarm_mat.items())},
            "alarm_by_wall": {k: [float(np.mean(x)), len(x)] for k, x in sorted(alarm_clean.items())},
            "crack_centreline_f1": float(np.mean([c["f1"] for c in cs])),
            "crack_centreline_precision": float(np.mean([c["precision"] for c in cs])),
            "crack_centreline_recall": float(np.mean([c["recall"] for c in cs])),
        }

    # light dependence on matched indoor views
    d = {v["stem"]: v for v in allv if v["batch"] == "indoor_default"}
    h = {v["stem"]: v for v in allv if v["batch"] == "indoor_hard"}
    # Only views whose camera and defects match: material_diversity.py changed
    # between the two batches (wood clean fraction), which re-drew seed 196009.
    common = sorted(set(d) & set(h))
    pairs = [s for s in common if d[s]["camera"] == h[s]["camera"] and d[s]["defects"] == h[s]["defects"]]
    same_cam = len(pairs)
    hard_fp = shared = 0
    flips = {"alarm_hard_only": 0, "alarm_default_only": 0, "alarm_both": 0, "neither": 0}
    per_pair = []
    for s in pairs:
        _, ex_h, _ = regions(h[s])
        _, ex_d, _ = regions(d[s])
        ch = (votes["indoor_hard/" + s] >= 3) & ~ex_h
        cd = cv2.dilate(((votes["indoor_default/" + s] >= 3) & ~ex_d).astype(np.uint8), disk(3)).astype(bool)
        hard_fp += ch.sum()
        shared += (ch & cd).sum()
        ah = ch.sum() > 0.001 * ch.size
        ad = ((votes["indoor_default/" + s] >= 3) & ~ex_d).sum() > 0.001 * ch.size
        flips["alarm_both" if ah and ad else "alarm_hard_only" if ah else "alarm_default_only" if ad else "neither"] += 1
        per_pair.append((int(ch.sum()), s))
    res["light_dependence"] = {"views_in_both": len(common), "matched_views": same_cam,
                               "hard_consistent_fp_px": int(hard_fp),
                               "share_also_fp_under_default": float(shared / max(1, hard_fp)),
                               "consistent_alarm_flips": flips}
    (out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    with open(out / "views.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    sheets(out, allv, votes, per_pair, d)
    print(json.dumps(res, indent=1))


def overlay(v, vote):
    img = cv2.imread(str(v["image"]))
    crack, excluded, cause = regions(v)
    ov = img.copy()
    ov[crack] = (0, 200, 0)
    colours = {"prop": (255, 0, 255), "outdoor_structure": (255, 0, 255), "other_scene": (255, 0, 255),
               "object_edge": (0, 165, 255), "cast_edge": (0, 0, 255), "in_cast": (0, 0, 160), "wall_surface": (0, 255, 255)}
    fp = (vote >= 3) & ~excluded
    for i, c in enumerate(CAUSES):
        ov[fp & (cause == i)] = colours[c]
    tp = (vote >= 3) & crack
    ov[tp] = (255, 255, 0)
    return img, ov


def sheets(out, allv, votes, per_pair, d):
    by = defaultdict(list)
    for v in allv:
        key = v["batch"] + "/" + v["stem"]
        _, ex, _ = regions(v)
        by[v["batch"]].append((int(((votes[key] >= 3) & ~ex).sum()), v))
    for bt, items in by.items():
        top = sorted(items, key=lambda t: -t[0])[:8]
        rows = []
        for n, v in top:
            img, ov = overlay(v, votes[bt + "/" + v["stem"]])
            row = np.hstack([img, ov])
            cv2.putText(row, f"{v['stem'][:17]} {v['light']} {v['material']} fp={n}", (8, 26), 0, 0.7, (255, 255, 255), 2)
            rows.append(cv2.resize(row, (800, 400)))
        cv2.imwrite(str(out / f"sheet_top_fp_{bt}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    hv = {v["stem"]: v for v in allv if v["batch"] == "indoor_hard"}
    rows = []
    for n, s in sorted(per_pair, reverse=True)[:8]:
        _, od = overlay(d[s], votes["indoor_default/" + s])
        _, oh = overlay(hv[s], votes["indoor_hard/" + s])
        row = np.hstack([od, oh])
        cv2.putText(row, f"{s[:17]} default | {hv[s]['light']}", (8, 26), 0, 0.7, (255, 255, 255), 2)
        rows.append(cv2.resize(row, (800, 400)))
    if rows:
        cv2.imwrite(str(out / "sheet_light_pairs.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
