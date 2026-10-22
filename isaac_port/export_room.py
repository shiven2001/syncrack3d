"""Export a whole SynCrack3D room for Isaac Sim (Blender stage).

    python isaac_port/export_room.py scene_dir out_dir [--px-per-m 1600] [--prop-res 1024]

scene_dir holds build_dataset.py --save-blend output (scene.blend, scene.json,
defect_instances.json). Unlike port_wall.py, which bakes one patch per close-up,
this keeps the full room and no cameras: an embodied agent has to find the
cracks itself, so nothing may be posed at them.

Writes to out_dir:
  wall/          the test wall's front face baked over its full width and height
                 (row 0 = top edge, column 0 = -X): height_mm.npy (displacement
                 along the outward normal; Cycles bakes leave displacement out, so
                 it is the shader's own height field), color.npy (linear),
                 roughness.npy, crack.npy (class coverage), bump_normal.png
                 (tangent-space host bump), wall.json (frame and resolution)
  room.usdc      everything else - floor, ceiling, side returns, fixtures, props
                 and lights - with procedural materials baked to textures/
compose_room.py (Isaac Sim python) then builds the displaced wall and scene.usda.
"""
import json
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np

argv = sys.argv[1:]
opts = {"--px-per-m": 1600.0, "--prop-res": 1024, "--max-px": 8192}
for k in list(opts):
    if k in argv:
        i = argv.index(k); opts[k] = type(opts[k])(argv[i + 1]); del argv[i:i + 2]
scene_dir, out = Path(argv[0]), Path(argv[1])
(out / "wall").mkdir(parents=True, exist_ok=True)
record = json.loads((scene_dir / "scene.json").read_text())
W, H = record["spec"]["width_m"], record["spec"]["height_m"]

bpy.ops.wm.open_mainfile(filepath=str(scene_dir / "scene.blend"))
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.get_devices(); prefs.compute_device_type = "OPTIX"; prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "OPTIX"
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 8
scene.render.bake.margin = 4


def select_only(*objs, active=None):
    bpy.context.view_layer.update()
    for o in bpy.context.view_layer.objects:
        if o is not None:
            o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


def pixels(img, channels):
    """Image pixels as (rows, cols, channels), row 0 = top."""
    a = np.empty(img.size[0] * img.size[1] * 4, np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(img.size[1], img.size[0], 4)[::-1, :, :channels]


# ---------------------------------------------------------------- test wall
wall = next(o for o in bpy.data.objects if o.type == "MESH" and "main.wall" in o.name)
corners = np.array([wall.matrix_world @ v.co for v in wall.data.vertices])
assert np.allclose([corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 2].min(),
                    corners[:, 2].max()], [-W / 2, W / 2, 0.0, 0.0, H], atol=1e-3), corners
ppm = min(opts["--px-per-m"], opts["--max-px"] / max(W, H))
nx, nz = int(round(W * ppm)), int(round(H * ppm))

mat = wall.data.materials[0]
nt = mat.node_tree
mout = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output)
disp_out = mout.inputs["Displacement"].links[0].from_socket
bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
surface_link = mout.inputs["Surface"].links[0].from_socket


def source(sock, tree):
    """The socket feeding `sock`, or a node holding its unlinked constant."""
    if sock.is_linked:
        return sock.links[0].from_socket
    node = tree.nodes.new("ShaderNodeRGB" if sock.type == "RGBA" else "ShaderNodeValue")
    node.outputs[0].default_value = sock.default_value
    return node.outputs[0]


def math_node(op, a, b):
    node = nt.nodes.new("ShaderNodeMath"); node.operation = op
    nt.links.new(a, node.inputs[0])
    node.inputs[1].default_value = b
    return node.outputs[0]


def upstream(sock, seen):
    for link in sock.links:
        node = link.from_node
        if node.name not in seen:
            seen.add(node.name)
            yield node
            for i in node.inputs:
                yield from upstream(i, seen)


# Height = the displacement vector the material outputs (several displacement
# nodes may be summed, e.g. tile relief plus cracks) projected on the surface
# normal, in world units: exactly the relief Cycles renders.
spaces = {n.space for n in upstream(mout.inputs["Displacement"], set()) if n.bl_idname == "ShaderNodeDisplacement"}
assert len(spaces) <= 1, spaces
vec = disp_out
if spaces == {"OBJECT"}:
    vt = nt.nodes.new("ShaderNodeVectorTransform")
    vt.vector_type, vt.convert_from, vt.convert_to = "VECTOR", "OBJECT", "WORLD"
    nt.links.new(vec, vt.inputs[0])
    vec = vt.outputs[0]
geo = nt.nodes.new("ShaderNodeNewGeometry")
dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
nt.links.new(vec, dot.inputs[0]); nt.links.new(geo.outputs["Normal"], dot.inputs[1])
passes = {
    # +50 keeps the emission positive; removed again below
    "height_mm": math_node("ADD", math_node("MULTIPLY", dot.outputs["Value"], 1000.0), 50.0),
    "color": source(bsdf.inputs["Base Color"], nt),
    "roughness": source(bsdf.inputs["Roughness"], nt),
}
crack_aov = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputAOV" and n.aov_name == "DefectCrack"), None)
if crack_aov is not None and crack_aov.inputs["Value"].is_linked:
    passes["crack"] = crack_aov.inputs["Value"].links[0].from_socket

# Bake target: a plane 2 mm in front of the wall, UV (0,0) at (-W/2, z=0).
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, -0.002, H / 2), rotation=(1.5707963, 0.0, 0.0))
target = bpy.context.object
target.scale = (W, H, 1.0)
bpy.ops.object.transform_apply(scale=True)
tmat = bpy.data.materials.new("wall_bake_target"); tmat.use_nodes = True
target.data.materials.append(tmat)
img_node = tmat.node_tree.nodes.new("ShaderNodeTexImage")
tmat.node_tree.nodes.active = img_node
emit = nt.nodes.new("ShaderNodeEmission")
maps = {}
for name, sock in passes.items():
    img = bpy.data.images.new(f"wall_{name}", nx, nz, float_buffer=True)
    img_node.image = img
    nt.links.new(sock, emit.inputs["Color"])
    nt.links.new(emit.outputs[0], mout.inputs["Surface"])
    select_only(wall, target, active=target)
    bpy.ops.object.bake(type="EMIT", use_selected_to_active=True, cage_extrusion=0.01, max_ray_distance=0.02)
    channels = 3 if name == "color" else 1
    maps[name] = pixels(img, channels)
    bpy.data.images.remove(img)
nt.links.new(surface_link, mout.inputs["Surface"])
# Host bump: a NORMAL bake holds what the height field does not.
img = bpy.data.images.new("wall_bump", nx, nz, float_buffer=False)
img.colorspace_settings.name = "Non-Color"
img_node.image = img
select_only(wall, target, active=target)
bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", use_selected_to_active=True,
                    cage_extrusion=0.01, max_ray_distance=0.02)
img.filepath_raw = str(out / "wall" / "bump_normal.png"); img.file_format = "PNG"; img.save()
# PNG rows are already top-first on disk

hit = maps["height_mm"][..., 0] > 1.0
np.save(out / "wall" / "height_mm.npy", np.where(hit, maps["height_mm"][..., 0] - 50.0, np.nan).astype(np.float32))
np.save(out / "wall" / "color.npy", maps["color"].astype(np.float16))
np.save(out / "wall" / "roughness.npy", maps["roughness"][..., 0].astype(np.float16))
np.save(out / "wall" / "crack.npy", (maps["crack"][..., 0] if "crack" in maps else np.zeros((nz, nx))).astype(np.float16))
(out / "wall" / "wall.json").write_text(json.dumps(dict(
    width_m=W, height_m=H, px_per_m=ppm, shape=[nz, nx], wall_object=wall.name,
    frame="front face y = 0, outward normal -Y; column 0 at x = -W/2, row 0 at z = H; "
          "height_mm is displacement along the outward normal (cracks negative)",
    hit_fraction=round(float(hit.mean()), 5)), indent=1))
print(f"wall baked: {nx} x {nz} px ({1000 / ppm:.2f} mm/px), hit {hit.mean():.4f}", flush=True)

# ---------------------------------------------------------------- the rest of the room
# No cameras (nothing may be posed at a crack), no hidden crack cards, and the
# wall itself is rebuilt by compose_room.py from the maps above.
for o in list(bpy.data.objects):
    if o in (wall, target) or o.type == "CAMERA" or o.name.startswith("CWG_Defect_") or o.hide_render:
        bpy.data.objects.remove(o, do_unlink=True)

curves = [o for o in bpy.data.objects if o.type == "CURVE"]
if curves:
    select_only(*curves)
    bpy.ops.object.convert(target="MESH")


def is_plain(m):
    """A single unlinked Principled BSDF exports as-is to UsdPreviewSurface."""
    if m is None or not m.use_nodes:
        return True
    p = next((n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    o = next((n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output), None)
    return (p is not None and o is not None and o.inputs["Surface"].is_linked
            and o.inputs["Surface"].links[0].from_node == p and not o.inputs["Displacement"].is_linked
            and not any(s.is_linked for s in p.inputs))


tex_dir = out / "textures"
tex_dir.mkdir(exist_ok=True)
baked = []
for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
    mats = [s.material for s in obj.material_slots]
    if all(is_plain(m) for m in mats):
        continue
    select_only(obj)
    bpy.ops.object.convert(target="MESH")  # apply modifiers so the bake UVs match the export
    me = obj.data
    old_uvs = [uv.name for uv in me.uv_layers]
    bake_uv = me.uv_layers.new(name="bake")
    me.uv_layers.active = bake_uv  # bake target; shaders keep reading the render-active map
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    area = sum(p.area for p in me.polygons) * max(obj.scale) ** 2
    res = opts["--prop-res"] * (2 if area > 2.0 else 1)
    stem = obj.name.replace(".", "_")
    images = {}
    for kind in ("color", "roughness", "metallic", "normal"):
        img = bpy.data.images.new(f"{stem}_{kind}", res, res, float_buffer=False)
        if kind != "color":
            img.colorspace_settings.name = "Non-Color"
        restore = []
        for m in mats:
            if m is None or not m.use_nodes:
                continue
            t = m.node_tree
            node = t.nodes.new("ShaderNodeTexImage"); node.image = img; t.nodes.active = node
            mo = next(n for n in t.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output)
            p = next((n for n in t.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
            if kind != "normal":
                key = {"color": "Base Color", "roughness": "Roughness", "metallic": "Metallic"}[kind]
                if p is None:
                    val = t.nodes.new("ShaderNodeRGB" if kind == "color" else "ShaderNodeValue")
                    val.outputs[0].default_value = (0.5, 0.5, 0.5, 1) if kind == "color" else 0.5 if kind == "roughness" else 0.0
                    src = val.outputs[0]
                else:
                    src = source(p.inputs[key], t)
                prev = mo.inputs["Surface"].links[0].from_socket if mo.inputs["Surface"].is_linked else None
                e = t.nodes.new("ShaderNodeEmission")
                t.links.new(src, e.inputs["Color"])
                t.links.new(e.outputs[0], mo.inputs["Surface"])
                restore.append((t, mo, prev))
        if kind == "normal":
            bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT")
        else:
            bpy.ops.object.bake(type="EMIT")
        for t, mo, prev in restore:
            if prev is not None:
                t.links.new(prev, mo.inputs["Surface"])
        img.filepath_raw = str(tex_dir / f"{stem}_{kind}.png"); img.file_format = "PNG"; img.save()
        images[kind] = img
    for name in old_uvs:
        me.uv_layers.remove(me.uv_layers[name])
    me.uv_layers["bake"].active_render = True
    # One baked material replaces every slot.
    new = bpy.data.materials.new(f"{stem}_baked"); new.use_nodes = True
    t = new.node_tree
    p = next(n for n in t.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    for kind, sock in (("color", "Base Color"), ("roughness", "Roughness"), ("metallic", "Metallic")):
        node = t.nodes.new("ShaderNodeTexImage"); node.image = images[kind]
        t.links.new(node.outputs["Color"], p.inputs[sock])
    node = t.nodes.new("ShaderNodeTexImage"); node.image = images["normal"]
    nmap = t.nodes.new("ShaderNodeNormalMap")
    t.links.new(node.outputs["Color"], nmap.inputs["Color"]); t.links.new(nmap.outputs[0], p.inputs["Normal"])
    me.materials.clear(); me.materials.append(new)
    for poly in me.polygons:
        poly.material_index = 0
    baked.append(obj.name)
    print("baked", obj.name, res, flush=True)

bpy.ops.wm.usd_export(
    filepath=str(out / "room.usdc"), selected_objects_only=False, visible_objects_only=True,
    export_cameras=False, export_lights=True, export_materials=True, generate_preview_surface=True,
    export_textures=False, relative_paths=True, export_uvmaps=True, rename_uvmaps=True, export_normals=True,
    evaluation_mode="RENDER", root_prim_path="/Room", export_custom_properties=True,
    convert_world_material=False, triangulate_meshes=False)
shutil.copy(scene_dir / "scene.json", out / "scene.json")
shutil.copy(scene_dir / "defect_instances.json", out / "defect_instances.json")
(out / "export.json").write_text(json.dumps(dict(source=str(scene_dir), baked_props=baked,
                                                 wall_px_per_m=ppm, prop_res=opts["--prop-res"]), indent=1))
print("exported", out / "room.usdc", flush=True)
