"""Bake the wall's shading normals (its bump) into each view patch as a normal map.

    python bake_patch_normal.py scene.blend patches_dir [--res 2048]

Recreates the planes recorded in patches_dir/patch.json and bakes a
tangent-space NORMAL map, selected-to-active from the wall. Displacement is not
part of a bake, so this holds exactly what the height bake misses: the host's
bump detail (and any bump on the defects). Writes <cam>_normal.png (8-bit,
standard OpenGL-style tangent-space encoding).
"""
import json
import sys
from pathlib import Path

import bpy

argv = sys.argv[1:]
res = 2048
if "--res" in argv:
    i = argv.index("--res"); res = int(argv[i + 1]); del argv[i:i + 2]
blend, pdir = argv[0], Path(argv[1])
info = json.loads((pdir / "patch.json").read_text())
bpy.ops.wm.open_mainfile(filepath=blend)
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.get_devices(); prefs.compute_device_type = "OPTIX"; prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "OPTIX"
scene = bpy.context.scene
scene.cycles.device = "GPU"
scene.cycles.samples = 8
scene.render.bake.margin = 2
wall = next(o for o in bpy.data.objects if o.type == "MESH" and "main.wall" in o.name)
for p in info["patches"]:
    cx, _, cz = p["center"]
    bpy.ops.mesh.primitive_plane_add(size=p["size_m"], location=(cx, -0.002, cz), rotation=(1.5707963, 0, 0))
    patch = bpy.context.object
    mat = bpy.data.materials.new(f"n_{p['camera']}"); mat.use_nodes = True
    patch.data.materials.append(mat)
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.new(f"{p['camera']}_normal", res, res)
    img.colorspace_settings.name = "Non-Color"
    node.image = img
    mat.node_tree.nodes.active = node
    for o in bpy.data.objects:
        o.select_set(False)
    wall.select_set(True); patch.select_set(True)
    bpy.context.view_layer.objects.active = patch
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", use_selected_to_active=True,
                        cage_extrusion=0.01, max_ray_distance=0.02)
    img.filepath_raw = str(pdir / f"{p['camera']}_normal.png")
    img.file_format = "PNG"
    img.save()
    print("baked normal", p["camera"])
