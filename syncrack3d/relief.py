"""How far a material's displacement stands out of its surface.

Bakes the height along the surface normal (metres) onto a flat patch with
metre UVs, in a scene of its own so the scene being built is not synced or
changed. Used per wall at build time (material_diversity.wall_relief) and by
tools/measure_wall_relief.py.
"""

from __future__ import annotations

import bpy
import numpy as np

_OFFSET = 1.0  # keeps the baked value positive


def bake_height(material, size_m: float = 2.0, res: int = 500) -> np.ndarray:
    """Outward height in metres at res x res texels over a size_m square of UV space."""
    scene = bpy.data.scenes.new("CWG_ReliefBake")
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    half = size_m * 0.5
    mesh = bpy.data.meshes.new("CWG_ReliefPatch")
    mesh.from_pydata([(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)], [], [(0, 1, 2, 3)])
    uv = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (co.x, co.y)  # metres, like the wall's (x, z)
    patch = bpy.data.objects.new("CWG_ReliefPatch", mesh)
    scene.collection.objects.link(patch)
    image = bpy.data.images.new("CWG_ReliefHeight", res, res, float_buffer=True)
    probe = material.copy()
    try:
        tree = probe.node_tree
        out = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output)
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        dot = tree.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        tree.links.new(out.inputs["Displacement"].links[0].from_socket, dot.inputs[0])
        tree.links.new(geometry.outputs["Normal"], dot.inputs[1])
        add = tree.nodes.new("ShaderNodeMath")
        add.inputs[1].default_value = _OFFSET
        tree.links.new(dot.outputs["Value"], add.inputs[0])
        emit = tree.nodes.new("ShaderNodeEmission")
        tree.links.new(add.outputs["Value"], emit.inputs["Color"])
        tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        target = tree.nodes.new("ShaderNodeTexImage")
        target.image = image
        tree.nodes.active = target
        mesh.materials.append(probe)
        view_layer = scene.view_layers[0]
        patch.select_set(True, view_layer=view_layer)
        view_layer.objects.active = patch
        with bpy.context.temp_override(scene=scene, view_layer=view_layer, object=patch,
                                       active_object=patch, selected_objects=[patch]):
            bpy.ops.object.bake(type="EMIT")
        pixels = np.empty(res * res * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        return pixels.reshape(-1, 4)[:, 0] - _OFFSET
    finally:
        bpy.data.objects.remove(patch)
        bpy.data.meshes.remove(mesh)
        bpy.data.images.remove(image)
        bpy.data.materials.remove(probe)
        bpy.data.scenes.remove(scene)


def outward_relief(material, percentile: float = 99.9, **kwargs) -> float:
    """Metres the material's surface stands out of its plane (0 when it only recesses)."""
    return max(0.0, float(np.percentile(bake_height(material, **kwargs), percentile)))
