"""Centreline F1 of v1.3.0 and v1.4.0 at 1-5 px tolerance on BD3, CUBIT and BuildCrack.

Checks whether the 3 px choice changes any conclusion. Same masks and pooling as
syncrack3d/eval (BD3 from the saved per-seed predictions; CUBIT crack photos at
1/3 scale with spalling ignored; all BuildCrack tiles), mean of five seeds.

    python -m studies.syncrack3d_tolerance_check
"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[1]
EVAL = APP / "eval"
sys.path.insert(0, str(EVAL))
from infer import load_model, predict  # noqa: E402
from metrics import centreline_counts, centreline_scores  # noqa: E402

DT = APP / "outputs/domain_transfer"
MODELS = {"v1.3.0": DT / "syncrack3d_tilestone_bd3_v1", "v1.4.0": DT / "syncrack3d_network2_bd3_v1"}
TOLS = (1, 2, 3, 4, 5)
OUT = DT / "syncrack3d_tolerance_check_v1.json"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def counts(pred, gt):
    return np.stack([centreline_counts(pred, gt, t) for t in TOLS])


def main():
    import torch

    bd3, cubit, build = (_module(n, EVAL / f"{n}/score.py") for n in ("bd3", "cubit", "buildcrack"))
    minor = [p for p in bd3.manifest() if p["folder"] == "minor_crack"]
    cub = []
    for k, _s, imp, lbp in cubit.pairs(cubit.CUBIT):
        if k.startswith("CR"):
            im = cv2.resize(cv2.cvtColor(cv2.imread(str(imp)), cv2.COLOR_BGR2RGB), None,
                            fx=cubit.SCALE, fy=cubit.SCALE, interpolation=cv2.INTER_AREA)
            gt, ign = cubit.rasterise(lbp, *im.shape[:2])
            cub.append((im, (gt > 0) & ~ign, ~ign))
    bc = [(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), cv2.imread(str(build.DATA / "labels" / p.name), 0) > 0)
          for p in sorted((build.DATA / "images").glob("*.png"))]
    res = {}
    for name, folder in MODELS.items():
        per = {"BD3": [], "CUBIT": [], "BuildCrack": []}
        for ck in sorted(folder.glob("runs/s*/last.pt")):
            tot = {k: np.zeros((len(TOLS), 4), np.int64) for k in per}
            for p in minor:
                pred = cv2.imread(str(folder / "pred" / ck.parent.name / f"{p['id']}.png"), 0) == 1
                tot["BD3"] += counts(pred, cv2.imread(str(EVAL / "bd3" / p["gt"]), 0) == 1)
            model = load_model(ck)
            for im, gt, valid in cub:
                tot["CUBIT"] += counts((predict(model, im) == 1) & valid, gt)
            for im, gt in bc:
                tot["BuildCrack"] += counts(predict(model, im) == 1, gt)
            del model
            torch.cuda.empty_cache()
            for k in per:
                per[k].append([centreline_scores(tot[k][i]) for i in range(len(TOLS))])
            print("done", name, ck.parent.name, flush=True)
        res[name] = {k: {f"{t}px": {m: float(np.mean([s[i][m] for s in v])) for m in ("precision", "recall", "f1")}
                         for i, t in enumerate(TOLS)} for k, v in per.items()}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    for k in ("BD3", "CUBIT", "BuildCrack"):
        print(f"\n{k}: centreline F1 by tolerance")
        for t in TOLS:
            a, b = res["v1.3.0"][k][f"{t}px"]["f1"], res["v1.4.0"][k][f"{t}px"]["f1"]
            print(f"  {t} px  v1.3.0 {a:.3f}  v1.4.0 {b:.3f}  difference {b - a:+.3f}")


if __name__ == "__main__":
    main()
