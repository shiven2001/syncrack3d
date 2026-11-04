"""SynCrack3D dataset: data-scaling study and three-architecture zero-shot evaluation.

The dataset version comes from $STUDY_TAG: v116 (default; renders since deleted,
models and scores kept), v117 (fixtures-first unified recipe) or v200 (v2.0.0,
Infinigen 2 materials; studies/syncrack3d_v2_compare.sh) or v200np (v2.0.0
without wood-plank walls; syncrack3d_v2_noplanks.sh) or v117lp (v1.17.0 with
--prop-placement loose; syncrack3d_loose_props.sh) or v118fm (v1.18.0 with
--fixture-mount relief; syncrack3d_fixture_mount_250.sh) or v118fc (that plus
--fixture-spacing clear; syncrack3d_fixture_clear_250.sh) or v119fr (v1.19.0 with
every fixture placed by one rule; syncrack3d_fixture_rule_250.sh) or v120mc (that plus
plaster and stucco each one unified category; syncrack3d_material_combine_250.sh) or v120nd
(that without outdoor wall dirt or crack staining; syncrack3d_no_dirt_250.sh) or v120io (that
with indoor walls only; syncrack3d_indoor_only_250.sh) or v122 (the v1.22.0
indoor-only generator; syncrack3d_v122_250.sh) or v122jn (v122 with half the cracks moved to
window, door, vent and access-panel junctions; syncrack3d_v122_junctions_250.sh). Data live in
/mnt/nvme_storage/syncrack3d_<version>_dataset and models in
outputs/domain_transfer/syncrack3d_<tag>_NAME.

Dataset (/mnt/nvme_storage/syncrack3d_v1.16.0_dataset, render.sh): unified recipe,
every wall cracked, one split per seed range so no wall is in two splits:
  train  train_a 910000-910499, train_b 910500-910999, train_c 911000-911999,
         train_d 912000-913999 (rendered last, used only if scaling continues)
  val    901000-901199   synthetic validation: the scaling stopping rule
  test   900000-900299   synthetic test: reported once at the end
Training subsets are nested: the first N train walls in seed order.

Commands
  labels CHUNK                       crack/ignore label maps for one chunk
  train NAME --walls N --arch A --seeds 0 1 [--iters K]
                                     model dir outputs/domain_transfer/syncrack3d_v116_NAME;
                                     iters default max(4000, number of training views)
  eval NAME [--splits val test] [--real]      protocol v2: 5 px tolerance, 1.5x upscale
  summary                            all results -> one JSON

Zero shot: no real image or real label is used for training, model selection or
the dataset-size decision (that uses the synthetic val split only). Real
benchmarks are reported, not tuned on.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[1]
TAG = os.environ.get("STUDY_TAG", "v116")
VERSION = {"v116": "v1.16.0", "v117": "v1.17.0", "v200": "v2.0.0", "v200np": "v2.0.0-noplanks",
           "v117lp": "v1.17.0-looseprops", "v118fm": "v1.18.0-fixturemount",
           "v118fc": "v1.18.0-fixtureclear", "v119fr": "v1.19.0-fixturerule",
           "v120mc": "v1.20.0-matcombine", "v120nd": "v1.20.0-nodirt",
           "v120io": "v1.20.0-indooronly", "v122": "v1.22.0",
           "v122jn": "v1.22.0-junctions", "v125": "v1.25.0"}[TAG]
DATA = Path(f"/mnt/nvme_storage/syncrack3d_{VERSION}_dataset")
DT = APP / "outputs/domain_transfer"
PY = "/home/shiven/anaconda3/bin/python"
TRAIN_CHUNKS = ("train_a", "train_b", "train_c", "train_d")
LR = {"unet_r34": 3e-4, "deeplabv3plus_r50": 3e-4, "segformer_b2": 6e-5}
ENV = {**os.environ, "SYNCRACK3D_CUBIT": "/mnt/nvme_storage/CUBIT-InSeg",
       "SYNCRACK3D_BUILDCRACK": str(APP / "external/BuildCrack")}


def labels(chunk):
    """labels/<stem>.png: 1 crack, 0 background, 255 ignore (ignore mask outside cracks)."""
    pack, out = DATA / chunk / "pack", DATA / chunk / "labels"
    out.mkdir(exist_ok=True)
    n = 0
    for img in sorted((pack / "images").glob("*.png")):
        dest = out / img.name
        if dest.exists():
            continue
        raw = cv2.imread(str(pack / "masks" / img.name), 0)
        if not set(np.unique(raw).tolist()) <= {0, 60, 255}:
            raise ValueError(img.name)
        lab = np.zeros(raw.shape, np.uint8)
        lab[raw == 60] = 1
        ign = pack / "ignore_masks" / img.name
        if ign.exists():
            lab[(cv2.imread(str(ign), 0) > 0) & (lab == 0)] = 255
        cv2.imwrite(str(dest), lab)
        n += 1
    print(chunk, "labels written", n, flush=True)


def train_rows(walls):
    """Views of the first `walls` train walls in seed order, from every rendered train chunk."""
    rows = []
    for chunk in TRAIN_CHUNKS:
        if not (DATA / chunk / "done").exists():
            continue
        labels(chunk)
        for img in sorted((DATA / chunk / "pack" / "images").glob("*.png")):
            rows.append((int(img.name.split("_")[1]), img))
    seeds = sorted({s for s, _ in rows})
    if len(seeds) < walls:
        raise SystemExit(f"only {len(seeds)} train walls rendered, {walls} requested")
    keep = set(seeds[:walls])
    return [(f"{TAG}__{img.stem}", str(img), str(img.parent.parent.parent / "labels" / img.name), False)
            for s, img in rows if s in keep]


def train(args):
    import torch

    from studies.seg_train import train as fit

    rows = train_rows(args.walls)
    iters = args.iters or max(4000, len(rows))
    out = DT / f"syncrack3d_{TAG}_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    cracked = sum(int((cv2.imread(r[2], 0) == 1).any()) for r in rows)
    (out / "train_list.json").write_text(json.dumps(dict(
        dataset=str(DATA), walls=args.walls, views=len(rows), views_with_crack=cracked, arch=args.arch,
        lr=LR[args.arch], iters=iters, batch=8, imgsz=512, seeds=args.seeds,
        first_wall=rows[0][0], generator=f"SynCrack3D {VERSION}" + (", --recipe unified" if not TAG.startswith("v200") else "")
        + (" --prop-placement loose" if TAG == "v117lp" else "") + (" --fixture-mount relief" if TAG == "v118fm" else "")
        + (" (loose props, --fixture-mount relief --fixture-spacing clear)" if TAG == "v118fc" else "")
        + (" (loose props, --fixture-mount relief, one fixture rule)" if TAG == "v119fr" else "")
        + (" (one fixture rule, plaster and stucco combined: 11 categories)" if TAG == "v120mc" else "")
        + (" (combined plaster and stucco, no outdoor dirt or crack staining)" if TAG == "v120nd" else "")
        + (" (v120nd with indoor walls only: 22 outdoor walls replaced by 910250+ indoor walls)" if TAG == "v120io" else "")
        + (" (indoor only; one fixture catalogue with full-length runs)" if TAG == "v122" else "")
        + (" (v122 with --crack-junction-fraction .5)" if TAG == "v122jn" else "")
        + (" (full dataset: junction cracks, ranged fixtures, indoor only)" if TAG == "v125" else "")),
        indent=1) + "\n")
    print(args.name, len(rows), "views", iters, "iters", flush=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    for seed in args.seeds:
        run = out / "runs" / f"s{seed}"
        if not (run / "last.pt").exists():
            fit(rows, run, seed, iters=iters, arch=args.arch, lr=LR[args.arch])
        torch.cuda.empty_cache()


def run(cmd, log):
    with open(log, "a") as fh:
        r = subprocess.run([str(c) for c in cmd], cwd=APP, stdout=fh, stderr=subprocess.STDOUT, env=ENV)
    if r.returncode:
        print("FAILED", " ".join(map(str, cmd[:4])), "see", log, flush=True)
    return r.returncode == 0


def evaluate(args):
    model = DT / f"syncrack3d_{TAG}_{args.name}"
    res = model / "results"
    (res / "logs").mkdir(parents=True, exist_ok=True)
    # Results pool every seed: when seeds were added since, redo the pooled ones.
    seeds = sorted(p.parent.name for p in model.glob("runs/s*/last.pt"))
    stamp = res / "seeds.json"
    if stamp.exists() and json.loads(stamp.read_text()) != seeds:
        import shutil

        for f in list(res.glob("synth_*.json")) + [res / "cubit_tol5", res / "buildcrack_tol5"]:
            shutil.rmtree(f) if f.is_dir() else f.unlink(missing_ok=True)
    stamp.write_text(json.dumps(seeds))
    for split in args.splits:
        if not (DATA / split / "manifest.json").exists():
            run([PY, "eval/synth/manifest.py", DATA / split], res / "logs" / "manifest.log")
        if not (res / f"synth_{split}.json").exists():
            run([PY, "eval/synth/score.py", "--set", DATA / split, "--model", model,
                 "--save", res / f"synth_{split}.json"], res / "logs" / f"synth_{split}.log")
    if args.real:
        tols = (5, 3) if args.tolerance3 else (5,)
        if not (model / "pred_v2").exists() or len(list((model / "pred_v2").glob("s*"))) < len(list(model.glob("runs/s*"))):
            run([PY, "eval/bd3/predict.py", "--model", model], res / "logs" / "bd3_predict.log")
        for s in sorted((model / "pred_v2").glob("s*")):
            for t in tols:
                out = res / f"bd3_{s.name}_tol{t}.json"
                if not out.exists():
                    run([PY, "eval/bd3/score.py", "--pred", s, "--out", out, "--tolerance", t], res / "logs" / "bd3.log")
        for t in tols:
            if not (res / f"cubit_tol{t}" / "results.json").exists():
                run([PY, "eval/cubit/score.py", "--model", model, "--out", res / f"cubit_tol{t}", "--tolerance", t],
                    res / "logs" / f"cubit_tol{t}.log")
            if not (res / f"buildcrack_tol{t}" / "results.json").exists():
                run([PY, "eval/buildcrack/score.py", "--model", f"{args.name}={model}", "--out", res / f"buildcrack_tol{t}",
                     "--tolerance", t], res / "logs" / f"buildcrack_tol{t}.log")
    print("evaluated", args.name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("labels")
    p.add_argument("chunk")
    p = sub.add_parser("train")
    p.add_argument("name")
    p.add_argument("--walls", type=int, required=True)
    p.add_argument("--arch", default="unet_r34", choices=sorted(LR))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--iters", type=int, default=0)
    p = sub.add_parser("eval")
    p.add_argument("name")
    p.add_argument("--splits", nargs="*", default=["val"])
    p.add_argument("--real", action="store_true")
    p.add_argument("--tolerance3", action="store_true")
    args = ap.parse_args()
    {"labels": lambda a: labels(a.chunk), "train": train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
