"""Small, controlled substrate shaders compatible with defect injection.

Every host shader keeps a linked Base Color and a Principled BSDF.  Those are
the two assumptions used when crack injection adds grime, roughness changes,
displacement, and label AOVs.

The stand-in shaders intentionally contain no room-scale dirt or baked
ambient occlusion. Their micro-relief is a bump node. Infinigen wall
materials carry a real displacement socket; ``enable_true_displacement``
turns that into geometry on every host, including walls with no defect.
"""

from __future__ import annotations

import bpy

from syncrack3d.spec import MaterialSpec


def _scaled_color(color, scale):
    return tuple(max(0.0, min(1.0, c * scale)) for c in color[:3]) + (1.0,)


def _principled_input(bsdf, *names):
    for name in names:
        socket = bsdf.inputs.get(name)
        if socket is not None:
            return socket
    return None


def _paint_or_concrete_nodes(nodes, links, texcoord, spec, bsdf):
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "CWG_Fine_Substrate_Variation"
    noise.noise_dimensions = "3D"
    noise.inputs["Scale"].default_value = spec.texture_scale
    noise.inputs["Detail"].default_value = 4.0 if spec.finish == "paint" else 7.0
    noise.inputs["Roughness"].default_value = 0.55 if spec.finish == "paint" else 0.72
    links.new(texcoord.outputs["Object"], noise.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = "CWG_Controlled_Albedo_Range"
    variation = spec.color_variation
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = _scaled_color(spec.base_color, 1.0 - variation)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = _scaled_color(spec.base_color, 1.0 + variation)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    rough_map = nodes.new("ShaderNodeMapRange")
    rough_map.name = "CWG_Controlled_Roughness_Range"
    rough_map.inputs["From Min"].default_value = 0.0
    rough_map.inputs["From Max"].default_value = 1.0
    rough_map.inputs["To Min"].default_value = max(0.0, spec.roughness - 0.035)
    rough_map.inputs["To Max"].default_value = min(1.0, spec.roughness + 0.035)
    links.new(noise.outputs["Fac"], rough_map.inputs["Value"])
    links.new(rough_map.outputs["Result"], bsdf.inputs["Roughness"])
    return noise


def _masonry_nodes(nodes, links, texcoord, spec, bsdf):
    brick = nodes.new("ShaderNodeTexBrick")
    brick.name = "CWG_Masonry_Pattern"
    brick.inputs["Color1"].default_value = _scaled_color(spec.base_color, 0.88)
    brick.inputs["Color2"].default_value = _scaled_color(spec.base_color, 1.10)
    brick.inputs["Mortar"].default_value = (0.32, 0.30, 0.27, 1.0)
    brick.inputs["Scale"].default_value = spec.texture_scale
    brick.inputs["Mortar Size"].default_value = 0.018
    brick.inputs["Mortar Smooth"].default_value = 0.02
    links.new(texcoord.outputs["Object"], brick.inputs["Vector"])
    links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = spec.roughness
    return brick


def _multiscale_paint(nodes, links, texcoord, spec, bsdf, fine_normal):
    """Add mesoscopic roller/plaster variation over fine paint, without labels."""
    axes = nodes.new("ShaderNodeVectorMath")
    axes.name = "CWG_Paint_Directional_Scale"
    axes.operation = "MULTIPLY"
    links.new(texcoord.outputs["Object"], axes.inputs[0])
    axes.inputs[1].default_value = spec.roller_axis_scale
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "CWG_Paint_Meso_Texture"
    noise.inputs["Scale"].default_value = spec.meso_scale
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = .6
    links.new(axes.outputs["Vector"], noise.inputs["Vector"])
    tone = nodes.new("ShaderNodeMapRange")
    tone.inputs["To Min"].default_value = .3
    tone.inputs["To Max"].default_value = 1.7
    links.new(noise.outputs["Fac"], tone.inputs["Value"])
    paint = bsdf.inputs["Base Color"].links[0].from_socket
    mix = nodes.new("ShaderNodeMixRGB")
    mix.name = "CWG_Paint_Meso_Albedo"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = spec.meso_color_variation
    links.new(paint, mix.inputs["Color1"])
    links.new(tone.outputs["Result"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.name = "CWG_Paint_Meso_Bump"
    bump.inputs["Strength"].default_value = .3
    bump.inputs["Distance"].default_value = spec.meso_bump_distance_m
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(fine_normal, bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def make_substrate_material(spec: MaterialSpec, index: int) -> bpy.types.Material:
    """Create one unique injection-ready host material."""
    mat = bpy.data.materials.new(f"CWG_{spec.finish}_{index:03d}")
    mat.use_nodes = True
    mat["cwg_material"] = True
    mat["surface_finish"] = spec.finish
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.name = "CWG_Material_Output"
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.name = "CWG_Principled"
    specular = _principled_input(bsdf, "Specular IOR Level", "Specular")
    if specular is not None:
        specular.default_value = 0.20 if spec.finish == "paint" else 0.14

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.name = "CWG_Metric_Object_Coordinates"
    if spec.finish == "masonry":
        texture = _masonry_nodes(nodes, links, texcoord, spec, bsdf)
        bump_source = texture.outputs["Fac"]
    else:
        texture = _paint_or_concrete_nodes(nodes, links, texcoord, spec, bsdf)
        bump_source = texture.outputs["Fac"]

    bump = nodes.new("ShaderNodeBump")
    bump.name = "CWG_Micro_Bump_Only"
    bump.inputs["Strength"].default_value = spec.micro_bump_strength
    bump.inputs["Distance"].default_value = spec.micro_bump_distance_m
    links.new(bump_source, bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    if spec.finish == "paint" and spec.paint_profile != "baseline":
        _multiscale_paint(nodes, links, texcoord, spec, bsdf, bump.outputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def enable_true_displacement(wall, dicing_rate: float = 2.0) -> None:
    """Displace every host as real geometry, whether or not it has a defect.

    Blender materials default to bump-only. Defect injection was the only
    step that switched a host to true displacement and adaptive subdivision,
    so a clean wall rendered its joints as a normal map: no depth and no
    cast shadows, while the same wall with a crack became geometry.
    """
    for slot in wall.material_slots:
        material = slot.material
        if material is not None and hasattr(material, "displacement_method"):
            material.displacement_method = "DISPLACEMENT"
    if not any(mod.type == "SUBSURF" for mod in wall.modifiers):
        modifier = wall.modifiers.new("wall_adaptive", "SUBSURF")
        modifier.subdivision_type = "SIMPLE"
    wall.cycles.use_adaptive_subdivision = True
    wall.cycles.dicing_rate = dicing_rate


def make_solid_material(name: str, color, roughness=0.42, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat
