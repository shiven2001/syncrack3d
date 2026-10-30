# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.

"""Small mesh helpers for wall-mounted procedural fixtures."""

import math

import bpy
from mathutils import Vector

from cwg_infinigen.core.util import blender as butil


def box(size, location=(0.0, 0.0, 0.0), name="box"):
    """Axis-aligned box of world size ``(sx, sy, sz)`` centered at ``location``."""
    sx, sy, sz = size
    obj = butil.spawn_cube(size=1, location=location, scale=(sx, sy, sz), name=name)
    butil.apply_transform(obj, loc=True)
    return obj


def cylinder(radius, depth, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), name="cyl"):
    obj = butil.spawn_cylinder(radius=radius, depth=depth, location=location, name=name)
    obj.rotation_euler = rotation
    butil.apply_transform(obj, loc=True)
    return obj


def shade_smooth(obj):
    mesh = obj.data
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = 0.7
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def solid_material(name, color, roughness=0.35, metallic=0.0):
    """Predictable Principled material; ``color`` is RGB or RGBA in 0–1."""
    if len(color) == 3:
        color = (*color, 1.0)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = next(
            n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"
        )
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = metallic
    return mat


def plastic_material(name, color, roughness=0.35, sheen_scale=None, variation=0.12):
    """Moulded ABS fascia: roughness breaks up over the surface instead of sitting
    at one value, which is what makes a flat `solid_material` panel read as CAD."""
    if len(color) == 3:
        color = (*color, 1.0)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color

    # Fine noise -> roughness, so highlights crawl the way they do on real moulded
    # plastic rather than forming one clean specular sheet.
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = (
        sheen_scale if sheen_scale is not None else 180.0
    )
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.6
    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.0
    ramp.inputs["From Max"].default_value = 1.0
    ramp.inputs["To Min"].default_value = max(0.04, roughness - variation)
    ramp.inputs["To Max"].default_value = min(1.0, roughness + variation)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], bsdf.inputs["Roughness"])

    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.42
    return mat


def rounded_box(size, location=(0.0, 0.0, 0.0), radius=0.004, segments=3, name="box"):
    """Box with eased arrises - moulded parts have no truly sharp edges."""
    obj = box(size, location=location, name=name)
    butil.modify_mesh(obj, "BEVEL", width=radius, segments=segments, limit_method="ANGLE")
    shade_smooth(obj)
    return obj


def glass_material(name, roughness=0.035, ior=1.48):
    """Clear architectural glass for shower screens."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "HASHED"
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    glass = nt.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Roughness"].default_value = roughness
    glass.inputs["IOR"].default_value = ior
    glass.inputs["Color"].default_value = (0.88, 0.94, 0.95, 1.0)
    nt.links.new(glass.outputs[0], out.inputs["Surface"])
    return mat


def mirror_material(name, roughness=0.02):
    """Front-silvered cabinet / medicine-cabinet mirror."""
    return solid_material(name, (0.90, 0.91, 0.93), roughness=roughness, metallic=1.0)


def assign(obj, mat):
    obj.active_material = mat
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj


def shift_min_x_to_zero(obj):
    """Put the back face on x=0 so flush_wall can snap against the wall."""
    xs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_x = min(v.x for v in xs)
    obj.location.x -= min_x
    butil.apply_transform(obj, loc=True)
    return obj


def disable_shadow_rays(mat):
    """Printed marks should not drop a contact shadow on the fascia."""
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    surface = [lnk for lnk in nt.links if lnk.to_node == out and lnk.to_socket.name == "Surface"]
    if not surface:
        return mat
    from_socket = surface[0].from_socket
    nt.links.remove(surface[0])
    light_path = nt.nodes.new("ShaderNodeLightPath")
    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(light_path.outputs["Is Shadow Ray"], mix.inputs["Fac"])
    nt.links.new(from_socket, mix.inputs[1])
    nt.links.new(transparent.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    return mat


def brand_lettering(
    text,
    size,
    location,
    rotation=(math.pi / 2.0, 0.0, math.pi / 2.0),
    extrude=0.0014,
    name="brand",
    seat_x=None,
):
    """Filled appliance brand, facing +X the way oven ``nodegroup_text`` does.

    ``seat_x`` puts the back of the letters on that plane so they read as a
    print on the fascia instead of floating type with a shadow halo.
    """
    bpy.ops.object.text_add(location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    curve = obj.data
    curve.body = str(text)
    curve.size = float(size)
    curve.extrude = float(extrude)
    curve.offset = 0.0
    curve.bevel_depth = 0.0
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    with butil.SelectObjects(obj):
        bpy.ops.object.convert(target="MESH")
    butil.apply_transform(obj, loc=True)
    if seat_x is not None and obj.data.vertices:
        xs = [v.co.x for v in obj.data.vertices]
        dx = float(seat_x) - min(xs)
        for v in obj.data.vertices:
            v.co.x += dx
        obj.data.update()
    return obj


def brushed_stainless(
    name,
    color_lo=(0.50, 0.51, 0.52),
    color_hi=(0.74, 0.75, 0.76),
    grain=(48.0, 48.0, 1.0),
    roughness=(0.34, 0.50),
):
    """Vertical-grain brushed steel for fridge doors and similar faces.

    ``appliance_steel()`` is a flat metallic Principled and reads as chrome
    under indoor lighting. Real top-freezer skins are anisotropic and duller.
    Grain defaults to vary along Y so a +X door face gets vertical brush lines.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = grain
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 22.0
    noise.inputs["Detail"].default_value = 16.0
    noise.inputs["Distortion"].default_value = 0.8
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.22
    ramp.color_ramp.elements[0].color = (*color_lo, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (*color_hi, 1.0)
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["From Min"].default_value = 0.0
    rough.inputs["From Max"].default_value = 1.0
    rough.inputs["To Min"].default_value = roughness[0]
    rough.inputs["To Max"].default_value = roughness[1]
    nt.links.new(tex.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(noise.outputs["Fac"], rough.inputs["Value"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.78
    if "Anisotropic" in bsdf.inputs:
        bsdf.inputs["Anisotropic"].default_value = 0.55
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.45
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


_STEEL_NAME = "ApplianceSteel"


def appliance_steel():
    """Brushed stainless for appliance faces, created once per blend file.

    Hoods, sinks and mixers used to sample the general metals list one by one,
    so a kitchen came out with a hammered-bronze basin under a brushed tap
    beside a galvanized hood. Appliance faces are neutral steel and they match.
    """
    mat = bpy.data.materials.get(_STEEL_NAME)
    if mat is None:
        mat = solid_material(
            _STEEL_NAME,
            (0.78, 0.79, 0.80),
            roughness=0.28,
            metallic=1.0,
        )
        mat.name = _STEEL_NAME
    return mat
