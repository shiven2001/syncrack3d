"""Blender reference renders under controlled lights, and the shared light plan.

For each close-up camera: target = where its optical axis meets the wall front
(y = 0). Lights: raking left/right/top at ~10 degrees to the surface, and one
frontal light beside the camera. Writes plan.json (consumed by the Isaac script)
and renders <cam>_<cond>.png with plain Cycles (Standard view, no compositor),
all other lights off and a black world, so both renderers see the same light.

    python blender_lightref.py scene.blend out_dir
"""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

blend, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=blend)
scene = bpy.context.scene
# Fresh bpy defaults to CPU rendering regardless of the saved scene setting.
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.get_devices()
prefs.compute_device_type = "OPTIX"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "OPTIX"
scene.cycles.device = "GPU"
assert any(d.use for d in prefs.devices), "no OptiX device"

RAKE_DIST, RAKE_OUT, POWER_W = 0.8, 0.15, 40.0


def target_of(cam):
    origin = cam.matrix_world.translation
    forward = (cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
    t = -origin.y / forward.y
    return origin + forward * t


plan = []
for cam in sorted((o for o in bpy.data.objects if o.type == "CAMERA" and "closeup" in o.name), key=lambda o: o.name):
    tgt = target_of(cam)
    loc = cam.matrix_world.translation
    side = (loc - tgt)
    conds = {
        "rake_left": tgt + Vector((-RAKE_DIST, -RAKE_OUT, 0)),
        "rake_right": tgt + Vector((RAKE_DIST, -RAKE_OUT, 0)),
        "rake_top": tgt + Vector((0, -RAKE_OUT, RAKE_DIST)),
        "frontal": loc + Vector((0.05, 0, 0.05)),
    }
    plan.append(dict(camera=cam.name, target=list(tgt), camera_distance_m=side.length,
                     matrix_world=[list(r) for r in cam.matrix_world], focal_mm=cam.data.lens,
                     sensor_width_mm=cam.data.sensor_width,
                     lights={k: list(v) for k, v in conds.items()}))
(out / "plan.json").write_text(json.dumps(dict(power_w=POWER_W, radius_m=0.02, views=plan), indent=1))

# Isolate the test light: other lights off, black world, no compositor.
for o in bpy.data.objects:
    if o.type == "LIGHT":
        o.hide_render = True
if scene.world is None:
    scene.world = bpy.data.worlds.new("black")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get("Background")
if bg is not None:
    bg.inputs["Strength"].default_value = 0.0
scene.use_nodes = False
scene.render.use_compositing = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = 0.0
scene.cycles.samples = 64
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
light_data = bpy.data.lights.new("port_test_light", "POINT")
light_data.energy = POWER_W
light_data.shadow_soft_size = 0.02
light = bpy.data.objects.new("port_test_light", light_data)
scene.collection.objects.link(light)
for view in plan:
    scene.camera = bpy.data.objects[view["camera"]]
    for cond, pos in view["lights"].items():
        light.location = pos
        scene.render.filepath = str(out / f"{view['camera']}_{cond}.exr")
        bpy.ops.render.render(write_still=True)
print("rendered", len(plan), "views x", len(plan[0]["lights"]), "lights")
