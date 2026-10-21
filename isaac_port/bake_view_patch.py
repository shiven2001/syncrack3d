"""Bake the wall region a close-up camera sees into high-resolution maps.

    python bake_view_patch.py scene.blend out_dir CAM [CAM ...] [--res 2048]

For each camera: the footprint of its frustum on the wall front (y = 0) plus a
10% margin becomes a target plane 2 mm in front of the wall. Emission bakes
(selected-to-active, evaluated on the wall itself, so its object-space fields are
exact) write float EXRs: height_mm (displacement along the outward normal, in
mm), base colour, roughness, and the class coverage AOVs. patch.json records the
plane's centre, axes and size.
"""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv[1:]
res = 2048
if "--res" in argv:
    i = argv.index("--res"); res = int(argv[i + 1]); del argv[i:i + 2]
blend, out, cams = argv[0], Path(argv[1]), argv[2:]
out.mkdir(parents=True, exist_ok=True)
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
mat = wall.data.materials[0]
nt = mat.node_tree
mout = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output)
disp = mout.inputs["Displacement"].links[0].from_node
assert disp.bl_idname == "ShaderNodeDisplacement", disp.bl_idname
h_link = disp.inputs["Height"].links[0].from_socket
mid, scale = disp.inputs["Midlevel"].default_value, disp.inputs["Scale"].default_value
bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")


def src(sock):
    return sock.links[0].from_socket if sock.is_linked else None


def src_or_const(sock):
    """The linked socket, or a node holding the unlinked input's constant (e.g. a fixed stucco roughness)."""
    if sock.is_linked:
        return src(sock)
    if sock.type == "RGBA":
        node = nt.nodes.new("ShaderNodeRGB")
        node.outputs[0].default_value = sock.default_value
    else:
        node = nt.nodes.new("ShaderNodeValue")
        node.outputs[0].default_value = sock.default_value
    return node.outputs[0]


aovs = {n.aov_name: src(n.inputs["Value"]) for n in nt.nodes
        if n.bl_idname == "ShaderNodeOutputAOV" and n.aov_name.startswith("Defect")
        and n.aov_name not in ("DefectInstance", "DefectInstanceSquared")}
surface_link = mout.inputs["Surface"].links[0].from_socket


def math(op, a, b):
    node = nt.nodes.new("ShaderNodeMath"); node.operation = op
    nt.links.new(a, node.inputs[0]) if not isinstance(a, float) else None
    if isinstance(a, float):
        node.inputs[0].default_value = a
    if isinstance(b, float):
        node.inputs[1].default_value = b
    else:
        nt.links.new(b, node.inputs[1])
    return node.outputs[0]


# height in mm along the outward normal, + 50 so the emission stays positive
height_mm = math("ADD", math("MULTIPLY", math("SUBTRACT", h_link, mid), scale * 1000.0), 50.0)
passes = {"height_mm": height_mm, "roughness": src_or_const(bsdf.inputs["Roughness"])}
color_src = src_or_const(bsdf.inputs["Base Color"])
passes["color"] = color_src
for name, sock in aovs.items():
    if sock is not None:
        passes[name] = sock

emit = nt.nodes.new("ShaderNodeEmission")
emit.inputs["Strength"].default_value = 1.0
# The wall box's camera-facing plane is y = 0 with outward normal -Y.
plan = []
for cam_name in cams:
    cam = bpy.data.objects[cam_name]
    mw = cam.matrix_world
    half = cam.data.sensor_width / (2 * cam.data.lens)
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            d = (mw.to_3x3() @ Vector((sx * half, sy * half, -1))).normalized()
            t = -mw.translation.y / d.y
            corners.append(mw.translation + d * t)
    xs, zs = [c.x for c in corners], [c.z for c in corners]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    size = max(max(xs) - min(xs), max(zs) - min(zs)) * 1.1
    bpy.ops.mesh.primitive_plane_add(size=size, location=(cx, -0.002, cz), rotation=(1.5707963, 0, 0))
    patch = bpy.context.object
    patch.name = f"patch_{cam_name}"
    pmat = bpy.data.materials.new(f"patch_mat_{cam_name}"); pmat.use_nodes = True
    patch.data.materials.append(pmat)
    img_node = pmat.node_tree.nodes.new("ShaderNodeTexImage")
    pmat.node_tree.nodes.active = img_node
    for pname, sock in passes.items():
        img = bpy.data.images.new(f"{cam_name}_{pname}", res, res, float_buffer=True)
        img_node.image = img
        nt.links.new(sock, emit.inputs["Color"])
        nt.links.new(emit.outputs[0], mout.inputs["Surface"])
        for o in bpy.data.objects:
            o.select_set(False)
        wall.select_set(True); patch.select_set(True)
        bpy.context.view_layer.objects.active = patch
        bpy.ops.object.bake(type="EMIT", use_selected_to_active=True, cage_extrusion=0.01,
                            max_ray_distance=0.02)
        img.filepath_raw = str(out / f"{cam_name}_{pname}.exr")
        img.file_format = "OPEN_EXR"
        img.save()
    nt.links.new(surface_link, mout.inputs["Surface"])
    # plane UV (0,0) is at local (-s/2, -s/2); local X -> world +X, local Y -> world +Z
    plan.append(dict(camera=cam_name, center=[cx, 0.0, cz], size_m=size, res=res,
                     u_axis=[1, 0, 0], v_axis=[0, 0, 1], outward_normal=[0, -1, 0],
                     height_offset_mm=50.0, passes=list(passes)))
(out / "patch.json").write_text(json.dumps(dict(midlevel=mid, scale=scale, patches=plan), indent=1))
print("baked", [p["camera"] for p in plan], "passes", list(passes))
