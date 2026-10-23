"""Build the frozen SynCrack3D synthetic evaluation set.

    python eval/synth/build.py --out OUT [--blender-py PY] [--only legacy major ...]

One held-out set covering every crack recipe of the current generator, plus
crack-free hard negatives. Seeds start at 800000; every training wall so far
used seeds below 200000. Each subset is built with the release recipe
(syncrack3d/recipes.py), exported with labels, and checked with the label
contract. Steps resume: a subset whose walls, pack or check already exist is not
redone.

  legacy     indoor walls, legacy cracks on every wall (--clean-wall-scale 0),
             all materials of --wall-materials diverse; wood and irregular stone
             stay crack-free by design and give plain close-ups
  major      indoor walls, broad open and branched cracks
  network    indoor paint/stucco walls with crack networks (crazing) beside
             indoor structures
  facade     outdoor facades, junction cracks, rain-wash staining
  negatives  crack-free walls, indoor and outdoor, with fixture views (doors,
             sockets, vents, cables, pipes, windows, gutters, ...) and plain
             close-ups

OUT/manifest.json records the code commit, the exact commands, and per view:
subset, seed, wall material, environment, crack pixels and SHA-256 of image and
mask. eval/synth/score.py scores a model on it.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]
SUBSETS = {
    "legacy": ("legacy", "800000-800039", ["--clean-wall-scale", "0"]),
    "major": ("major", "801000-801029", []),
    "network": ("network", "802000-802039", []),
    "facade": ("facade", "803000-803029", []),
    "negatives": ("legacy", "804000-804039", ["--environment", "mixed", "--structures", "auto",
                                             "--structure-views", "2", "--structure-clean-fraction", "1"]),
}

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--blender-py", default=os.environ.get("SYNCRACK3D_BLENDER_PY", sys.executable),
                help="python with bpy 4.2 (default: this interpreter or $SYNCRACK3D_BLENDER_PY)")
ap.add_argument("--only", nargs="+", choices=list(SUBSETS))
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


def run(cmd, log):
    print(">>", " ".join(cmd[1:4]), "...", flush=True)
    with open(log, "a") as fh:
        subprocess.run(cmd, cwd=APP, stdout=fh, stderr=subprocess.STDOUT, check=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=APP, capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=APP,
                            capture_output=True, text=True).stdout.strip())
commands = {}
for name, (recipe, seeds, extra) in SUBSETS.items():
    if args.only and name not in args.only:
        continue
    root = args.out / name
    root.mkdir(exist_ok=True)
    log = root / "build.log"
    build = [args.blender_py, "syncrack3d/build_dataset.py", "--recipe", recipe, "--seeds", seeds,
             "--output", str(root / "walls"), *extra, "--keep-going", "--resume"]
    export = [args.blender_py, "tools/prepare_defect_annotated_dataset.py", "-i", str(root / "walls" / "frames"),
              "-o", str(root / "pack"), "--aov-masks", "auto", "--instance-masks", "auto", "--with-bboxes",
              "--max-negative-fraction", "1"]
    check = [args.blender_py, "tools/check_instance_contract.py", "--pack", str(root / "pack")]
    commands[name] = [" ".join(c[1:]) for c in (build, export, check)]
    run(build, log)
    if not (root / "pack" / "images").exists():
        run(export, log)
    if not (root / "contract.txt").exists():
        out = subprocess.run(check, cwd=APP, capture_output=True, text=True)
        (root / "contract.txt").write_text(out.stdout + out.stderr)
        if out.returncode != 0:
            raise SystemExit(f"{name}: label contract check failed, see {root / 'contract.txt'}")
    print(name, "done", flush=True)

views = []
for name in SUBSETS:
    pack = args.out / name / "pack"
    if not (pack / "images").exists():
        continue
    scenes = {}
    for img in sorted((pack / "images").glob("*.png")):
        seed = int(img.name.split("_")[1])
        if seed not in scenes:
            scenes[seed] = json.loads((args.out / name / "walls" / f"seed_{seed}" / "scene.json").read_text())
        rec = scenes[seed]
        mask = pack / "masks" / img.name
        views.append(dict(
            subset=name, id=img.stem, seed=seed,
            image=str(img.relative_to(args.out)), mask=str(mask.relative_to(args.out)),
            material=rec["wall_material"].get("diverse", {}).get("category", rec["wall_material"].get("generator")),
            environment=rec.get("environment"),
            crack_px=int((cv2.imread(str(mask), 0) == 60).sum()),
            sha256_image=sha(img), sha256_mask=sha(mask)))
summary = {}
for name in SUBSETS:
    v = [x for x in views if x["subset"] == name]
    if v:
        summary[name] = dict(views=len(v), walls=len({x["seed"] for x in v}),
                             cracked_views=sum(x["crack_px"] > 0 for x in v),
                             crack_free_views=sum(x["crack_px"] == 0 for x in v),
                             materials=dict(sorted({m: sum(x["material"] == m for x in v)
                                                    for m in {x["material"] for x in v}}.items())))
old = json.loads((args.out / "manifest.json").read_text()) if (args.out / "manifest.json").exists() else {}
manifest = dict(
    name="SynCrack3D synthetic evaluation set", version="v1", code_commit=commit, code_dirty=dirty,
    subsets={k: dict(recipe=r, seeds=s, extra_flags=e) for k, (r, s, e) in SUBSETS.items()},
    commands={**old.get("commands", {}), **commands},
    mask_encoding="masks/*.png: 60 = crack, 0 = background; ignore_masks/*.png > 0 = ignore unless crack",
    summary=summary, views=views)
(args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(summary, indent=1))
if dirty:
    print("WARNING: built from a working tree with uncommitted changes", flush=True)
