"""Measure each view patch's displaced surface by rendering it, not by baking.

    python bake_patch_height_ortho.py scene.blend patches_dir [--res 2048] [--dicing 1.0] [--hide-prefix PeelFlap]

The shader-height bake under-reports relief (validate_relief.py: shape right,
amplitude ~1/23), so this renders an orthographic camera square-on to each
patch with Cycles' Position pass: the world y of the surface Cycles actually
displaces. Writes <cam>_height_mm.npy in the same format as the bake (relief in
mm along the outward normal, +50 offset, 0 where the ray missed the wall).
"""
import json
import math
import sys
from pathlib import Path

import bpy
import Imath
import numpy as np
import OpenEXR

argv = sys.argv[1:]
opts = {"--res": 2048, "--dicing": 1.0, "--hide-prefix": ""}
for k in list(opts):
    if k in argv:
        i = argv.index(k); opts[k] = type(opts[k])(argv[i + 1]); del argv[i:i + 2]
blend, pdir = argv[0], Path(argv[1])
info = json.loads((pdir / "patch.json").read_text())
bpy.ops.wm.open_mainfile(filepath=blend)
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.get_devices(); prefs.compute_device_type = "OPTIX"; prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "OPTIX"
scene = bpy.context.scene
scene.cycles.device = "GPU"
scene.cycles.samples = 4
scene.cycles.use_denoising = False
scene.cycles.dicing_rate = opts["--dicing"]
scene.render.resolution_x = scene.render.resolution_y = opts["--res"]
scene.render.resolution_percentage = 100
scene.render.filter_size = 0.01
scene.use_nodes = False
scene.render.use_compositing = False
scene.view_layers[0].use_pass_position = True
scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
scene.render.image_settings.color_depth = "32"
cam_data = bpy.data.cameras.new("height_cam")
cam_data.type = "ORTHO"
cam = bpy.data.objects.new("height_cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
# Overhanging geometry (peel flaps) cannot be a height field; it is exported as
# meshes separately (dump_meshes.py) and hidden here.
if opts["--hide-prefix"]:
    for o in bpy.data.objects:
        if o.name.startswith(opts["--hide-prefix"]):
            o.hide_render = True
for p in info["patches"]:
    cx, _, cz = p["center"]
    cam.location = (cx, -0.5, cz)
    cam.rotation_euler = (math.pi / 2, 0, 0)  # looks along +Y; image up = +Z, right = +X
    cam_data.ortho_scale = p["size_m"]
    cam_data.clip_start, cam_data.clip_end = 0.01, 2.0
    tmp = pdir / f"{p['camera']}_position.exr"
    scene.render.filepath = str(tmp)
    bpy.ops.render.render(write_still=True)
    f = OpenEXR.InputFile(str(tmp))
    dw = f.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    key = next(k for k in f.header()["channels"] if k.endswith("Position.Y"))
    y = np.frombuffer(f.channel(key, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(h, w)
    rel0 = -(y - np.median(y)) * 1000.0
    ok = (rel0 > -6) & (rel0 < 8)  # physical defect relief is within -4.2..+5.5 mm
    relief = rel0 - np.median(rel0[ok])
    out = np.where(ok, relief + 50.0, 0.0).astype(np.float32)
    np.save(pdir / f"{p['camera']}_height_mm.npy", out)
    print(p["camera"], "relief mm min %.3f max %.3f, miss %.1f%%" % (relief[ok].min(), relief[ok].max(), 100 * (~ok).mean()))
