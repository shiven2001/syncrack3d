"""Build the frozen SynCrack3D synthetic evaluation set.

    python eval/synth/build.py --out OUT [--blender-py PY] [--only legacy major ...]

One held-out set from the v2 generator (the `unified` recipe: 15 crack-able
materials drawn uniformly, one clean-wall table), covering every crack style plus
crack-free hard negatives. Each style gets its own subset by forcing
--crack-style-mix, so all styles have enough walls to report separately; the
clean-wall table still leaves about 40% of walls plain, which supplies the
crack-free close-ups each subset's alarm rate is measured on. Seeds start at
810000; every training wall so far used seeds below 200000. Each subset is
exported with labels and checked with the label contract. Steps resume.

  legacy     indoor walls, legacy cracks
  major      indoor walls, broad open and branched cracks
  network    indoor paint/stucco walls with crack networks beside indoor fixtures
  facade     outdoor facades, facade fixtures, junction cracks, rain-wash staining
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

APP = Path(__file__).resolve().parents[2]
SUBSETS = {
    "legacy": ("unified", "810000-810049", ["--crack-style-mix", "legacy=1"]),
    "major": ("unified", "811000-811039", ["--crack-style-mix", "major=1"]),
    "network": ("unified", "812000-812049", ["--crack-style-mix", "network=1"]),
    "facade": ("unified", "813000-813039", ["--crack-style-mix", "facade=1"]),
    "negatives": ("unified", "814000-814039", ["--crack-style-mix", "off", "--environment", "mixed",
                                               "--structures", "auto", "--structure-views", "2",
                                               "--structure-clean-fraction", "1"]),
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


def contract_issues(report):
    """{view id: labelled crack px} for every view the contract check flagged.

    Overlapping crack networks can leave a few pixels holding two instance ids.
    In a view where every instance fell below the evidence floor, nothing is
    labelled and those pixels are already ignored, so the view is excluded and
    recorded; an issue in a labelled view stops the build.
    """
    issues, view, labelled = {}, None, 0
    for line in report.read_text().splitlines():
        if line and not line.startswith(" ") and line.startswith("seed_"):
            view, labelled = line.strip(), 0
        elif line.strip().startswith("labelled "):
            labelled = int(line.split()[1])
        elif line.strip().startswith("ISSUE:") and view:
            issues[view] = labelled
    return issues


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
    bad = contract_issues(root / "contract.txt")
    if any(labelled for labelled in bad.values()):
        raise SystemExit(f"{name}: label contract issues on labelled views, see {root / 'contract.txt'}")
    print(name, "done", f"({len(bad)} unlabelled view(s) with contract issues excluded)" if bad else "", flush=True)

views, excluded = [], []
for name in SUBSETS:
    pack = args.out / name / "pack"
    if not (pack / "images").exists():
        continue
    bad = contract_issues(args.out / name / "contract.txt")
    excluded += [dict(subset=name, id=v, reason="label contract issue (blended instance ids) in a view "
                      "with no labelled crack; pixels already ignored") for v in sorted(bad)]
    scenes = {}
    for img in sorted((pack / "images").glob("*.png")):
        if img.stem in bad:
            continue
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
            crack_style=rec.get("appearance_controls", {}).get("crack_style"),
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
    name="SynCrack3D synthetic evaluation set", version="v2 generator (unified recipe)", code_commit=commit, code_dirty=dirty,
    subsets={k: dict(recipe=r, seeds=s, extra_flags=e) for k, (r, s, e) in SUBSETS.items()},
    commands={**old.get("commands", {}), **commands},
    mask_encoding="masks/*.png: 60 = crack, 0 = background; ignore_masks/*.png > 0 = ignore unless crack",
    summary=summary, excluded=excluded, views=views)
(args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(summary, indent=1))
if dirty:
    print("WARNING: built from a working tree with uncommitted changes", flush=True)
