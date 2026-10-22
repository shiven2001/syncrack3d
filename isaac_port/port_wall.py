"""Port one CWG wall to Isaac Sim end to end, and check it against Blender.

    python isaac_port/port_wall.py --seed 112000 --material-category paint_neutral --out OUT
    python isaac_port/port_wall.py --out OUT --isaac-only        # later, when the GPU is free

Stages (each skipped when its outputs exist, so a crash resumes):
  scene     syncrack3d/build_dataset.py --save-blend (rendered views + scene.blend)
  ref_pack  tools/prepare_defect_annotated_dataset.py on those views (reference masks)
  cams      dump_cams.py
  patches   bake_view_patch.py (colour, roughness, class coverage per close-up),
            bake_patch_height_ortho.py (true displaced relief from Cycles' Position pass),
            bake_patch_normal.py (host bump as a tangent-space normal map),
            exr_to_npy.py (the Isaac env reads .npy; it has no OpenEXR)
  refs      blender_lightref.py (shared light plan + Blender references),
            blender_lightref2.py (isolated lights + rendered Position pass)
  -- Isaac (Isaac Sim python; cores 16-31, nice 10, one GPU job at a time) --
  panels    build_panels.py: one USD panel per close-up with a defect (for the RL rig)
  isaac     isaac_patch_render.py: displaced, labelled meshes rendered RTX + path traced
  checks    validate_relief.py (baked vs rendered relief), compare.py (Blender vs Isaac
            light response per camera), figure.py
Interpreters: --blender-py (bpy 4.2 module), --isaac-py (Isaac Sim 5 / Isaac Lab env).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent
CRACK_COVERAGE = {"DefectCrack": "crack"}


def run(cmd, log, env=None, isaac=False):
    if isaac:
        cmd = ["taskset", "-c", "16-31", "nice", "-n", "10", *cmd]
    print(">>", " ".join(str(c) for c in cmd[:6]), "...", flush=True)
    with open(log, "a") as fh:
        subprocess.run([str(c) for c in cmd], cwd=APP, stdout=fh, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, **(env or {})})


def closeups(root):
    cams = json.loads((root / "cams.json").read_text())["cameras"]
    return [c["name"] for c in cams if "closeup" in c["name"]]


def rig_of(cam):
    return int(cam.split("_")[2])  # CWG_Cam_003_closeup -> render index 3


def blender_stages(a, root):
    import numpy as np

    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    scene = root / "scene"
    if not list(scene.glob("seed_*/scene.blend")):
        from domain_common import common_options

        run([a.blender_py, APP / "syncrack3d/build_dataset.py", "--seeds", f"{a.seed}-{a.seed}", "--output", scene,
             *common_options(a), "--save-blend"], logs / "scene.log")
    blend = next(scene.glob("seed_*/scene.blend"))
    if not (root / "ref_pack" / "curation.json").exists():
        run([a.blender_py, APP / "tools/prepare_defect_annotated_dataset.py", "-i", scene / "frames", "-o",
             root / "ref_pack", "--aov-masks", "on", "--instance-masks", "on", "--with-bboxes",
             "--max-negative-fraction", "1", "--no-progress"], logs / "ref_pack.log")
    if not (root / "cams.json").exists():
        run([a.blender_py, HERE / "dump_cams.py", blend, root / "cams.json"], logs / "cams.log")
    cams = closeups(root)
    pdir = root / "patches"
    if not (pdir / "patch.json").exists():
        run([a.blender_py, HERE / "bake_view_patch.py", blend, pdir, *cams, "--res", a.res], logs / "patch_bake.log")
    if not all((pdir / f"{c}_height_mm.npy").exists() for c in cams):
        run([a.blender_py, HERE / "bake_patch_height_ortho.py", blend, pdir, "--res", a.res, "--dicing", "1.0"],
            logs / "height_ortho.log")
    if not all((pdir / f"{c}_normal.png").exists() for c in cams):
        run([a.blender_py, HERE / "bake_patch_normal.py", blend, pdir, "--res", a.res], logs / "bake_normal.log")
    if not all((pdir / f"{c}_color.npy").exists() for c in cams):
        run([a.blender_py, HERE / "exr_to_npy.py", pdir], logs / "exr_npy.log")  # Isaac env has no OpenEXR
    if not (root / "blender_ref" / "plan.json").exists():
        run([a.blender_py, HERE / "blender_lightref.py", blend, root / "blender_ref"], logs / "blender_ref.log")
    if not all((root / "blender_ref2" / f"{c}_frontal.exr").exists() for c in cams):
        run([a.blender_py, HERE / "blender_lightref2.py", blend, root / "blender_ref" / "plan.json",
             root / "blender_ref2", *cams], logs / "blender_ref2.log")
    # panel spec: one panel per close-up whose patch carries crack coverage
    panels = []
    for c in cams:
        cov = {cls: float((np.load(pdir / f"{c}_{aov}.npy") >= 0.5).mean())
               for aov, cls in CRACK_COVERAGE.items() if (pdir / f"{c}_{aov}.npy").exists()}
        if not cov or max(cov.values()) == 0:
            continue
        kind = max(cov, key=cov.get)
        panels.append({"name": f"{kind}_{a.seed if a.seed is not None else 'wall'}_{c}",
                       "patch_dir": str(pdir), "camera": c})
    (root / "panels_spec.json").write_text(json.dumps({"note": "built by syncrack3d/isaac_port/port_wall.py",
                                                       "panels": panels}, indent=1) + "\n")
    print(f"blender stages done: {len(cams)} close-ups, {len(panels)} panels with defects", flush=True)


def isaac_stages(a, root):
    logs = root / "logs"
    env = {"OMNI_KIT_ACCEPT_EULA": "YES"}
    if not (root / "panels" / "panels.json").exists():
        run([a.isaac_py, HERE / "build_panels.py", "--spec", root / "panels_spec.json", "--out", root / "panels"],
            logs / "panels.log", env, isaac=True)
    if not list((root / "isaac").glob("*")):
        # render at the scene's own resolution so Isaac frames align with the Blender references
        res = json.loads((root / "cams.json").read_text())["resolution"][0]
        run([a.isaac_py, HERE / "isaac_patch_render.py", root / "patches", root / "blender_ref" / "plan.json",
             root / "isaac", "--res", res], logs / "isaac.log", env, isaac=True)


def check_stages(a, root):
    logs = root / "logs"
    specs = [f"{c}:{rig_of(c)}" for c in closeups(root)]
    run([a.blender_py, HERE / "validate_relief.py", root, *specs], logs / "validate_relief.log")
    run([a.blender_py, HERE / "compare.py", root, "isaac"], logs / "compare.log")
    print("checks written:", logs / "validate_relief.log", logs / "compare.log", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--material-category", default="paint_neutral")
    ap.add_argument("--res", default="2048")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="more build_dataset options (last)")
    ap.add_argument("--blender-py", default=os.environ.get("SYNCRACK3D_BLENDER_PY", sys.executable),
                    help="Python with bpy 4.2 (default: $SYNCRACK3D_BLENDER_PY, else this interpreter)")
    ap.add_argument("--isaac-py", default=os.environ.get("SYNCRACK3D_ISAAC_PY"),
                    help="Isaac Sim 5 Python (or $SYNCRACK3D_ISAAC_PY); needed unless --blender-only")
    ap.add_argument("--blender-only", action="store_true")
    ap.add_argument("--isaac-only", action="store_true")
    a = ap.parse_args()
    root = a.out.resolve()
    sys.path.insert(0, str(HERE))
    if not a.isaac_only:
        if a.seed is None:
            ap.error("--seed is required to build the scene")
        blender_stages(a, root)
    if not a.blender_only:
        if not a.isaac_py:
            ap.error("--isaac-py (or $SYNCRACK3D_ISAAC_PY) is required for the Isaac stages; use --blender-only to stop before them")
        isaac_stages(a, root)
        check_stages(a, root)


if __name__ == "__main__":
    main()
