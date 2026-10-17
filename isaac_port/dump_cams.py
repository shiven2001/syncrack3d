"""Dump camera matrices/intrinsics and light summary from a CWG scene.blend."""
import json, sys
import bpy
blend, out = sys.argv[1], sys.argv[2]
bpy.ops.wm.open_mainfile(filepath=blend)
s = bpy.context.scene
cams = []
for o in sorted((o for o in bpy.data.objects if o.type == "CAMERA"), key=lambda o: o.name):
    cams.append(dict(name=o.name, matrix_world=[list(r) for r in o.matrix_world],
                     focal_mm=o.data.lens, sensor_width_mm=o.data.sensor_width,
                     sensor_fit=o.data.sensor_fit, clip=[o.data.clip_start, o.data.clip_end]))
lights = [dict(name=o.name, type=o.data.type, energy=o.data.energy, loc=list(o.matrix_world.translation))
          for o in bpy.data.objects if o.type == "LIGHT"]
meshes = [dict(name=o.name, verts=len(o.data.vertices), mats=[m.name for m in o.data.materials if m],
               hidden=o.hide_render) for o in bpy.data.objects if o.type == "MESH"]
wall = next(o for o in bpy.data.objects if o.type == "MESH" and "main.wall" in o.name)
json.dump(dict(resolution=[s.render.resolution_x, s.render.resolution_y], cameras=cams, lights=lights,
               meshes=meshes, wall=dict(name=wall.name, dims=list(wall.dimensions),
               cycles_subdiv=getattr(wall.cycles, "use_adaptive_subdivision", None),
               modifiers=[m.type for m in wall.modifiers])), open(out, "w"), indent=1)
print("cameras", len(cams), "lights", len(lights), "meshes", len(meshes), "wall", wall.name, list(wall.dimensions), [m.type for m in wall.modifiers])
