"""Dump meshes whose names start with a prefix (e.g. PeelFlap) in world space.

    python dump_meshes.py scene.blend out.npz PeelFlap

Evaluated geometry (modifiers applied), triangulated. Keys: names, and for each
name <name>__points (N, 3) float32, <name>__tris (M, 3) int32, <name>__instance.
"""
import sys

import bpy
import numpy as np

blend, out, prefix = sys.argv[1:4]
bpy.ops.wm.open_mainfile(filepath=blend)
dg = bpy.context.evaluated_depsgraph_get()
data, names = {}, []
for o in bpy.data.objects:
    if o.type != "MESH" or not o.name.startswith(prefix) or o.hide_render:
        continue
    ev = o.evaluated_get(dg)
    mesh = ev.to_mesh()
    mesh.calc_loop_triangles()
    mw = o.matrix_world
    pts = np.array([tuple(mw @ v.co) for v in mesh.vertices], np.float32)
    tris = np.array([tuple(t.vertices) for t in mesh.loop_triangles], np.int32)
    data[f"{o.name}__points"], data[f"{o.name}__tris"] = pts, tris
    data[f"{o.name}__instance"] = np.array(int(o.name.split("_")[1]))
    names.append(o.name)
    ev.to_mesh_clear()
np.savez(out, names=np.array(names), **data)
print("dumped", names)
