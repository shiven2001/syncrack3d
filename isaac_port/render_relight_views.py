"""Relight a composed room with opposing lamps (Isaac Sim 5 python), for the paper's port figure.

    OMNI_KIT_ACCEPT_EULA=YES isaac-sim-5/python isaac_port/render_relight_views.py EXPORT_DIR OUT \
        [--crack 0] [--res 1024] [--spp 256] [--room-ambient .15] [--crack-ambient .03]

Renders two cameras, each under a left and a right lamp, with nothing changed between the two
frames of a pair except which lamp is on:
  room    a wide view of the room; the lamps stand near the side walls
  crack   a close view of one crack; the lamps rake along the wall from either side of it

The room's own lights stay on, dimmed by the ambient factor, so unlit parts do not go black.
Writes <view>_<left|right>.npy (linear HDR, path traced), <view>_semantic.npy, and views.json with
the poses, lamp positions and the semantic id -> class table. Stage changes are not saved.
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("export_dir")
ap.add_argument("out")
ap.add_argument("--crack", type=int, default=0, help="index into scene_meta.json defects")
ap.add_argument("--res", type=int, default=1024)
ap.add_argument("--spp", type=int, default=256)
ap.add_argument("--room-ambient", type=float, default=0.15)
ap.add_argument("--crack-ambient", type=float, default=0.03)
ap.add_argument("--room-eye", type=float, nargs=3, help="default: a high corner, as compose_room's preview")
ap.add_argument("--room-target", type=float, nargs=3)
ap.add_argument("--room-focal", type=float, default=14.0)
ap.add_argument("--room-lamp-inset", type=float, default=1.2, help="room lamp distance from its side wall, metres")
ap.add_argument("--room-lamp-height", type=float, default=0.45, help="room lamp height, as a fraction of the wall's")
ap.add_argument("--crack-standoff", type=float, default=0.6, help="camera distance from the wall, metres")
ap.add_argument("--crack-frame", type=float, help="wall width in view, metres (default 1.5x the crack size)")
ap.add_argument("--crack-center", type=float, nargs=2, help="wall x, z to centre on (default the crack centre)")
ap.add_argument("--rake-offset", type=float, default=1.2, help="lamp distance along the wall from the crack, metres")
ap.add_argument("--rake-standoff", type=float, default=0.3, help="lamp distance from the wall, metres")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.res, "height": args.res})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

# Streamed textures are still at low mips when a headless render reads its frames: the 0.625 mm wall
# texture renders at about 1.5-1.75x less sharp than with streaming off.
carb.settings.get_settings().set("/rtx-transient/resourcemanager/enableTextureStreaming", False)

root, out = Path(args.export_dir), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((root / "scene_meta.json").read_text())
(x0, x1), (y0, y1), (z0, z1) = meta["room"]["x_range"], meta["room"]["y_range"], meta["room"]["z_range"]
W, H, depth = x1 - x0, z1 - z0, y1 - y0
defect = meta["defects"][args.crack]
cx, _, cz = defect["center_world"]
if args.crack_center:
    cx, cz = args.crack_center
frame = args.crack_frame or 1.5 * defect["size_m"]

omni.usd.get_context().open_stage(str(root / "scene.usda"))
for _ in range(30):
    app.update()
stage = omni.usd.get_context().get_stage()

room_lights = [(UsdLux.LightAPI(p).GetIntensityAttr(), UsdLux.LightAPI(p).GetIntensityAttr().Get())
               for p in stage.Traverse() if p.GetTypeName() in ("DiskLight", "RectLight", "SphereLight")]
print("room lights", len(room_lights), flush=True)


def lamp(name, pos, intensity):
    light = UsdLux.SphereLight.Define(stage, f"/World/FigureLamps/{name}")
    light.CreateRadiusAttr(0.01)
    light.CreateIntensityAttr(intensity)
    light.CreateTreatAsPointAttr(True)  # sharp shadows, and no lamp visible in frame
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdGeom.Imageable(light).MakeInvisible()
    return light


def camera(name, eye, target, focal_mm):
    cam = UsdGeom.Camera.Define(stage, f"/World/FigureCams/{name}")
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())
    cam.CreateFocalLengthAttr(focal_mm)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    return str(cam.GetPath())


room_eye = args.room_eye or [x1 - 0.3, y0 + 0.3, z1 - 0.3]
room_target = args.room_target or [x0 + W / 3, 0.0, H * 0.35]
crack_eye = [cx + 0.1, -args.crack_standoff, cz]
views = dict(
    room=dict(eye=room_eye, target=room_target, focal_mm=args.room_focal, ambient=args.room_ambient,
              lamps=dict(left=[x0 + args.room_lamp_inset, y0 + depth / 2, H * args.room_lamp_height],
                         right=[x1 - args.room_lamp_inset, y0 + depth / 2, H * args.room_lamp_height]),
              intensity=4.0e6),
    crack=dict(eye=crack_eye, target=[cx, 0.0, cz], focal_mm=36.0 * args.crack_standoff / frame,
               ambient=args.crack_ambient,
               lamps=dict(left=[cx - args.rake_offset, -args.rake_standoff, cz],
                          right=[cx + args.rake_offset, -args.rake_standoff, cz]),
               intensity=4.0e6),
)
for vname, v in views.items():
    v["camera"] = camera(vname, v["eye"], v["target"], v["focal_mm"])
    v["lamp_prims"] = {side: lamp(f"{vname}_{side}", pos, v["intensity"]) for side, pos in v["lamps"].items()}
for _ in range(10):
    app.update()

rep.settings.set_render_pathtraced(samples_per_pixel=args.spp)
record = dict(export=str(root), res=args.res, spp=args.spp, crack=args.crack, crack_center=[cx, cz],
              crack_frame_m=frame, views={})
for vname, v in views.items():
    for attr, base in room_lights:
        attr.Set(base * v["ambient"])
    rp = rep.create.render_product(v["camera"], (args.res, args.res))
    hdr = rep.AnnotatorRegistry.get_annotator("HdrColor")
    sem = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
    hdr.attach(rp)
    sem.attach(rp)
    id_to_label = None
    for side in ("left", "right"):
        UsdGeom.Imageable(v["lamp_prims"][side]).MakeVisible()
        # The first step can still hold the previous lamp's accumulation.
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=args.spp, delta_time=0.0)
        img = np.asarray(hdr.get_data())[..., :3].astype(np.float32)
        np.save(out / f"{vname}_{side}.npy", img)
        s = sem.get_data()
        np.save(out / f"{vname}_semantic.npy", np.asarray(s["data"]).astype(np.uint16))
        id_to_label = {k: (val.get("class", str(val)) if isinstance(val, dict) else str(val))
                       for k, val in s["info"]["idToLabels"].items()}
        UsdGeom.Imageable(v["lamp_prims"][side]).MakeInvisible()
        print(vname, side, "mean", float(img.mean()), flush=True)
    hdr.detach()
    sem.detach()
    rp.destroy()
    record["views"][vname] = dict(eye=v["eye"], target=v["target"], focal_mm=v["focal_mm"], ambient=v["ambient"],
                                  lamps=v["lamps"], id_to_label=id_to_label)
(out / "views.json").write_text(json.dumps(record, indent=1))
app.close()
