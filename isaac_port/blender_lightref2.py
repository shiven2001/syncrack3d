"""Isolated-light Blender references plus the rendered surface position.

    python blender_lightref2.py scene.blend plan.json out_dir CAM [CAM ...]

Like blender_lightref.py, but every emitting object (lights, emissive meshes,
world) is switched off so only the planned point light remains, and each render
is a multilayer EXR that includes Cycles' Position pass: the world position of
the displaced surface actually rendered, used to check the baked relief.
"""
import json
import sys
from pathlib import Path

import bpy

blend, plan_path, out, cams = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
out.mkdir(parents=True, exist_ok=True)
plan = json.loads(Path(plan_path).read_text())
bpy.ops.wm.open_mainfile(filepath=blend)
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.get_devices(); prefs.compute_device_type = "OPTIX"; prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "OPTIX"
scene = bpy.context.scene
scene.cycles.device = "GPU"


def emits(mat):
    if mat is None or not mat.use_nodes:
        return False
    for n in mat.node_tree.nodes:
        if n.bl_idname == "ShaderNodeEmission":
            s = n.inputs["Strength"]
            if s.is_linked or s.default_value > 0:
                return True
        if n.bl_idname == "ShaderNodeBsdfPrincipled":
            s = n.inputs.get("Emission Strength")
            if s is not None and (s.is_linked or s.default_value > 0):
                c = n.inputs.get("Emission Color")
                if c is None or c.is_linked or any(v > 0 for v in c.default_value[:3]):
                    return True
    return False


hidden = []
for o in bpy.data.objects:
    if o.type == "LIGHT" or (o.type == "MESH" and any(emits(m) for m in o.data.materials)):
        if not o.hide_render:
            o.hide_render = True
            hidden.append(o.name)
print("switched off:", hidden)
if scene.world is not None and scene.world.use_nodes:
    for n in scene.world.node_tree.nodes:
        if n.bl_idname == "ShaderNodeBackground":
            n.inputs["Strength"].default_value = 0.0
scene.use_nodes = False
scene.render.use_compositing = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = 0.0
scene.cycles.samples = 64
scene.view_layers[0].use_pass_position = True
scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
scene.render.image_settings.color_depth = "32"
ld = bpy.data.lights.new("port_test_light", "POINT")
ld.energy = plan["power_w"]
ld.shadow_soft_size = plan["radius_m"]
light = bpy.data.objects.new("port_test_light", ld)
scene.collection.objects.link(light)
views = {v["camera"]: v for v in plan["views"]}
for cam in cams:
    scene.camera = bpy.data.objects[cam]
    for cond, pos in views[cam]["lights"].items():
        light.location = pos
        scene.render.filepath = str(out / f"{cam}_{cond}.exr")
        bpy.ops.render.render(write_still=True)
(out / "switched_off.json").write_text(json.dumps(hidden))
print("done")
