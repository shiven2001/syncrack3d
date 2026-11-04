"""Cracks on held-out synthetic stone block and large tile walls.

Real benchmarks hardly contain these surfaces, so this measures what adding
their cracks buys: 6 unseen walls (outputs/walls/
syncrack3d_tilestone_test_v1, seeds 176040-176045, legacy profile), never in any
training set. Per surface and model (mean over five seeds): centreline P/R/F1
at 3 px and crack IoU on cracked views, false-positive pixels per view more than
3 px from a labelled crack, and the share of crack-free views that alarm
(predicted crack above 0.1%, the BD3 plain-wall rule). Ignore masks are
excluded, as in training.

    PYTHONPATH=eval python -m studies.syncrack3d_synth_surface_eval
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP / "eval"))
from infer import load_model, predict  # noqa: E402
from metrics import centreline_counts, centreline_scores  # noqa: E402

TEST = APP / "outputs/walls/syncrack3d_tilestone_test_v1"
DT = APP / "outputs/domain_transfer"
MODELS = {"v1.2.0": "syncrack3d_layergate_bd3_v1", "null": "syncrack3d_straightnull_bd3_v1",
          "tilestone": "syncrack3d_tilestone_bd3_v1", "hostcrack": "syncrack3d_hostcrack_bd3_v1"}
OUT = DT / "syncrack3d_synth_surface_v1.json"


def views():
    out = []
    for pack in sorted(TEST.glob("legacy/*/pack")):
        for img in sorted((pack / "images").glob("*.png")):
            raw = cv2.imread(str(pack / "masks" / img.name), 0)
            crack = raw == 60
            ign_path = pack / "ignore_masks" / img.name
            ign = (cv2.imread(str(ign_path), 0) > 0) if ign_path.exists() else np.zeros_like(crack)
            out.append({"surface": pack.parent.name, "image": img, "crack": crack,
                        "valid": ~(ign & ~crack)})
    return out


def main():
    import torch

    vs = views()
    near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    imgs = [cv2.cvtColor(cv2.imread(str(v["image"])), cv2.COLOR_BGR2RGB) for v in vs]
    res = {"views": {s: [int(sum(v["surface"] == s and v["crack"].any() for v in vs)),
                         int(sum(v["surface"] == s and not v["crack"].any() for v in vs))]
                     for s in sorted({v["surface"] for v in vs})},
           "models": {}}
    for name, folder in MODELS.items():
        per_seed = []
        for ck in sorted((DT / folder).glob("runs/s*/last.pt")):
            model = load_model(ck)
            acc = {}
            for v, img in zip(vs, imgs):
                pred = (predict(model, img, scale=1) == 1) & v["valid"]
                a = acc.setdefault(v["surface"], {"cl": np.zeros(4), "iou": np.zeros(3), "fp": [], "alarm": []})
                far = pred & ~(cv2.dilate(v["crack"].astype(np.uint8), near) > 0)
                if v["crack"].any():
                    a["cl"] += centreline_counts(pred, v["crack"], 3)
                    a["iou"] += [(pred & v["crack"]).sum(), (pred & ~v["crack"]).sum(), (~pred & v["crack"]).sum()]
                    a["fp"].append(far.sum())
                else:
                    a["alarm"].append(far.mean() > 0.001)
            per_seed.append({s: {**{k: centreline_scores(a["cl"])[k] for k in ("precision", "recall", "f1")},
                                 "crack_iou": a["iou"][0] / max(1, a["iou"].sum()),
                                 "fp_px_per_cracked_view": float(np.mean(a["fp"])),
                                 "plain_view_alarm": float(np.mean(a["alarm"])) if a["alarm"] else None}
                             for s, a in acc.items()})
            del model
            torch.cuda.empty_cache()
        res["models"][name] = {
            s: {k: ([float(np.mean([p[s][k] for p in per_seed])), float(np.std([p[s][k] for p in per_seed], ddof=1))]
                    if per_seed[0][s][k] is not None else None) for k in per_seed[0][s]}
            for s in per_seed[0]}
        print("done", name, flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("views (cracked, plain):", res["views"])
    for s in res["views"]:
        print(f"\n{s}: {'model':10} {'recall':>13} {'precision':>13} {'F1':>13} {'IoU':>13} {'FP px/view':>14} {'plain alarm':>13}")
        for name, r in res["models"].items():
            m = r[s]
            pa = "-" if m["plain_view_alarm"] is None else f"{m['plain_view_alarm'][0]:.2f}"
            print(f"{'':13}{name:10} " + " ".join(f"{m[k][0]:.3f}±{m[k][1]:.3f}" for k in ("recall", "precision", "f1", "crack_iou"))
                  + f" {m['fp_px_per_cracked_view'][0]:14.0f} {pa:>13}")


if __name__ == "__main__":
    main()
