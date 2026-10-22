"""Render embodied-style views of a composed room (Isaac Sim 5 python).

    OMNI_KIT_ACCEPT_EULA=YES isaac-sim-5/python isaac_port/render_room_views.py EXPORT_DIR OUT \
        [--views 60] [--res 640] [--seed 0]

Poses are what a mobile camera could take, not close-ups framed on a crack:
70% look at a uniform point on the test wall from 0.4-2.5 m at up to 35 deg off
its normal, 30% look anywhere (walls, floor, ceiling, furniture). Positions are
inside the room, 0.8-1.8 m high and clear of furniture. Focal 24 mm on a 36 mm
aperture (74 deg).

Per view: <i>_pt.npy and <i>_rt.npy (linear HDR, path traced 64 spp / real-time
after warm-up), <i>_albedo.npy (diffuse albedo, for telling shadow edges from
paint edges), <i>_semantic.npy + labels in views.json, and the pose.
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("export_dir")
ap.add_argument("out")
ap.add_argument("--views", type=int, default=60)
ap.add_argument("--res", type=int, default=640)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shadow-only", action="store_true",
                help="with --hard-light: skip the colour passes and write <i>_shadow.npy, where the probe "
                     "lamps' light on a surface pixel drops below half when furniture and fixtures are present")
ap.add_argument("--hard-light", action="store_true",
                help="dim the room's area lights to 20%% and add two small bright lamps that cast sharp "
                     "shadows: one low at the back (furniture shadows on the walls), one grazing from a "
                     "top corner (fixture and pipe shadows along the test wall)")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.res, "height": args.res})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdSemantics  # noqa: E402

root = Path(args.export_dir)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((root / "scene_meta.json").read_text())
(x0, x1), (y0, y1), (z0, z1) = meta["room"]["x_range"], meta["room"]["y_range"], meta["room"]["z_range"]

omni.usd.get_context().open_stage(str(root / "scene.usda"))
for _ in range(30):
    app.update()
stage = omni.usd.get_context().get_stage()
# Probe-only labels (not saved): tell the test wall from the other walls.
for prim in stage.Traverse():
    path = str(prim.GetPath())
    probe = ("test_wall" if path == "/World/TestWall/wall" else "side_wall" if "Shell_Return" in path
             else "back_wall" if path in ("/World/BackWall", "/World/TestWallBacking") else None)
    if probe and path.count("/") <= 3:
        UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set([probe])
if args.hard_light:
    from pxr import UsdLux

    for prim in stage.Traverse():
        if prim.GetTypeName() in ("DiskLight", "RectLight", "SphereLight"):
            a = UsdLux.LightAPI(prim).GetIntensityAttr()
            a.Set(a.Get() * 0.2)
    for name, pos in (("back", (0.3 * x1, y0 + 0.3, 1.1)), ("graze", (x0 + 0.15, -0.35, z1 - 0.15))):
        lamp = UsdLux.SphereLight.Define(stage, f"/World/ProbeLamp_{name}")
        lamp.CreateRadiusAttr(0.01)
        lamp.CreateIntensityAttr(4.0e6)
        lamp.CreateTreatAsPointAttr(True)
        UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(*pos))
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
boxes = []
for path, cls in meta["labels"].items():
    if cls == "furniture":
        r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        if r.GetMax()[2] - r.GetMin()[2] > 0.05:
            boxes.append((np.array(r.GetMin()), np.array(r.GetMax())))


def free(p, margin=0.25):
    if not (x0 + margin < p[0] < x1 - margin and y0 + margin < p[1] < y1 - margin and z0 + 0.3 < p[2] < z1 - 0.2):
        return False
    return not any(np.all(p > lo - margin) and np.all(p < hi + margin) for lo, hi in boxes)


rng = np.random.default_rng(args.seed)
poses = []
while len(poses) < args.views:
    if rng.random() < 0.7:
        target = np.array([rng.uniform(x0 + 0.1, x1 - 0.1), 0.0, rng.uniform(z0 + 0.1, z1 - 0.1)])
        yaw, pitch = np.radians(rng.uniform(-35, 35)), np.radians(rng.uniform(-35, 35))
        d = np.array([np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
        eye = target + d * rng.uniform(0.4, 2.5)
        kind = "test_wall"
    else:
        eye = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(0.8, 1.8)])
        yaw, pitch = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(-50, 40))
        target = eye + np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
        kind = "anywhere"
    if 0.8 <= eye[2] <= 1.8 and free(eye):
        poses.append(dict(eye=eye.round(4).tolist(), target=target.round(4).tolist(), kind=kind))

cam = UsdGeom.Camera.Define(stage, "/World/ProbeCam")
xform = UsdGeom.Xformable(cam).AddTransformOp()
cam.CreateFocalLengthAttr(24.0)
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
rp = rep.create.render_product("/World/ProbeCam", (args.res, args.res))
hdr = rep.AnnotatorRegistry.get_annotator("HdrColor")
sem = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
alb = rep.AnnotatorRegistry.get_annotator("DiffuseAlbedo")
for a in (hdr, sem, alb):
    a.attach(rp)


def place(p):
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*p["eye"]), Gf.Vec3d(*p["target"]), Gf.Vec3d(0, 0, 1))
    xform.Set(view.GetInverse())


labels = {}
if args.shadow_only:
    if not args.hard_light:
        raise SystemExit("--shadow-only needs --hard-light")
    rep.settings.set_render_rtx_realtime()
    for prim in stage.Traverse():
        if prim.GetTypeName() in ("DiskLight", "RectLight", "DomeLight"):
            UsdGeom.Imageable(prim).MakeInvisible()
    casters = [stage.GetPrimAtPath(pth) for pth, c in meta["labels"].items() if c in ("furniture", "fixture")]

    def lamp_only(i):
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=16, delta_time=0.0)
        return np.asarray(hdr.get_data())[..., :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)

    for i, p in enumerate(poses):
        place(p)
        lit = lamp_only(i)
        for c in casters:
            UsdGeom.Imageable(c).MakeInvisible()
        bare = lamp_only(i)
        for c in casters:
            UsdGeom.Imageable(c).MakeVisible()
        np.save(out / f"{i:03d}_shadow.npy", ((lit < 0.5 * bare) & (bare > np.percentile(bare, 5))).astype(np.uint8))
        print("shadow", i, flush=True)
    app.close()
    raise SystemExit(0)
for mode in ("rt", "pt"):
    if mode == "pt":
        rep.settings.set_render_pathtraced(samples_per_pixel=64)
    else:
        rep.settings.set_render_rtx_realtime()
    for i, p in enumerate(poses):
        place(p)
        # Real-time needs several frames to settle after a camera jump.
        for _ in range(2 if mode == "pt" else 6):
            rep.orchestrator.step(rt_subframes=64 if mode == "pt" else 8, delta_time=0.0)
        np.save(out / f"{i:03d}_{mode}.npy", np.asarray(hdr.get_data())[..., :3].astype(np.float16))
        if mode == "rt":
            s = sem.get_data()
            np.save(out / f"{i:03d}_semantic.npy", np.asarray(s["data"]).astype(np.uint16))
            labels[i] = {k: v.get("class", str(v)) if isinstance(v, dict) else str(v)
                         for k, v in s["info"]["idToLabels"].items()}
            np.save(out / f"{i:03d}_albedo.npy", np.asarray(alb.get_data())[..., :3].astype(np.uint8))
        print(mode, i, flush=True)
(out / "views.json").write_text(json.dumps(dict(export=str(root), res=args.res, hard_light=args.hard_light, focal_mm=24.0, aperture_mm=36.0,
                                                poses=poses, labels=labels), indent=1))
app.close()
