"""Add a frontal camera whose wall footprint covers every placed defect.

    python add_panel_camera.py scene.blend scene.json out.blend [--margin 0.10] [--name CWG_Cam_900_wallpanel]

bake_view_patch.py bakes the wall region a camera sees, so this camera turns the
per-close-up port into one wall-scale panel: a square region around all of the
wall's defects plus `margin` metres on each side (bake_view_patch adds 10% more).
The wall front is y = 0 with outward normal -Y; defect (u, v) is world (x, z).
"""
import json
import math
import sys
from pathlib import Path

import bpy

argv = sys.argv[1:]
opts = {"--margin": 0.10, "--name": "CWG_Cam_900_wallpanel"}
for k in list(opts):
    if k in argv:
        i = argv.index(k); opts[k] = type(opts[k])(argv[i + 1]); del argv[i:i + 2]
blend, scene_json, out = argv[0], Path(argv[1]), argv[2]
defects = json.loads(scene_json.read_text())["defects_placed"]
xs = [d["u_m"] + s * d["size_m"] / 2 for d in defects for s in (-1, 1)]
zs = [d["v_m"] + s * d["size_m"] / 2 for d in defects for s in (-1, 1)]
cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
size = max(max(xs) - min(xs), max(zs) - min(zs)) + 2 * opts["--margin"]

bpy.ops.wm.open_mainfile(filepath=blend)
dist = 1.0
cam_data = bpy.data.cameras.new(opts["--name"])
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam_data.lens = cam_data.sensor_width * dist / size  # footprint width = size at dist
cam = bpy.data.objects.new(opts["--name"], cam_data)
bpy.context.scene.collection.objects.link(cam)
cam.location = (cx, -dist, cz)
cam.rotation_euler = (math.pi / 2, 0.0, 0.0)  # looks along +Y at the wall, up = +Z
bpy.ops.wm.save_as_mainfile(filepath=out)
print(json.dumps({"camera": opts["--name"], "center_xz": [cx, cz], "footprint_m": size, "defects": len(defects)}))
