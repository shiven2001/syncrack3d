"""Heavy-texture painted stucco: knockdown, sand-float, skip-trowel, popcorn, heavy orange-peel.

Added because 64% of BD3 crack photos have more fine-texture contrast than 90%
of CWG6-8 synthetic training walls, and cracks on rough stucco were missed by
every synthetic model. Painted (finish "paint", via "plaster" in the class
name), so cracks and peeling can be injected. Same structure as Plaster that
injection relies on: a linked Principled Base Color and a Displacement node into
the Material Output. Paint colours come from plaster_paint_params, so the CWG
colour categories apply. Relief is real displacement, 0.5-6 mm. UV in metres.
"""

import numpy as np
from numpy.random import uniform

from cwg_infinigen.assets.materials.ceramic.plaster import _sample_plaster_pair, plaster_paint_params
from cwg_infinigen.core import surface
from cwg_infinigen.core.nodes.node_info import Nodes
from cwg_infinigen.core.nodes.node_utils import build_color_ramp
from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler

STYLES = ("knockdown", "sand_float", "skip_trowel", "popcorn", "heavy_orange_peel")


def _height(nw, uv, style):
    """Relief field in [0, 1] and its displacement depth in metres."""
    if style == "knockdown":
        n = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={
            "Scale": uniform(25, 60), "Detail": 3, "Distortion": uniform(0.2, 0.6)})
        t = uniform(0.45, 0.58)
        blobs = build_color_ramp(nw, n.outputs["Fac"], [t, t + 0.04], [(0, 0, 0, 1), (1, 1, 1, 1)])
        fine = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(300, 600), "Detail": 2})
        h = nw.scalar_add(nw.scalar_multiply(blobs, 0.85), nw.scalar_multiply(fine.outputs["Fac"], 0.15))
        return h, uniform(0.0015, 0.0050)
    if style == "sand_float":
        grit = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(800, 1600), "Detail": 2})
        swirl = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(20, 50), "Detail": 4})
        h = nw.scalar_add(nw.scalar_multiply(grit.outputs["Fac"], 0.75),
                          nw.scalar_multiply(swirl.outputs["Fac"], 0.25))
        return h, uniform(0.0005, 0.0016)
    if style == "skip_trowel":
        n = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={
            "Scale": uniform(6, 15), "Detail": 4, "Distortion": uniform(1.0, 3.0)})
        t = uniform(0.45, 0.6)
        patches = build_color_ramp(nw, n.outputs["Fac"], [t, t + 0.08], [(0, 0, 0, 1), (1, 1, 1, 1)])
        streak_uv = nw.new_node(Nodes.Mapping, [uv], input_kwargs={"Scale": (1.0, uniform(3, 6), 1.0)})
        streak = nw.new_node(Nodes.NoiseTexture, [streak_uv], input_kwargs={"Scale": uniform(25, 60), "Detail": 3})
        h = nw.scalar_add(nw.scalar_multiply(patches, 0.8), nw.scalar_multiply(streak.outputs["Fac"], 0.2))
        return h, uniform(0.0012, 0.0035)
    if style == "popcorn":
        cells = nw.new_node(Nodes.VoronoiTexture, [uv], input_kwargs={
            "Scale": uniform(150, 350), "Randomness": 1.0})
        bumps = build_color_ramp(nw, cells.outputs["Distance"], [0.0, uniform(0.45, 0.6)],
                                 [(1, 1, 1, 1), (0, 0, 0, 1)])
        fine = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(400, 900), "Detail": 2})
        h = nw.scalar_add(nw.scalar_multiply(bumps, 0.8), nw.scalar_multiply(fine.outputs["Fac"], 0.2))
        return h, uniform(0.0025, 0.0060)
    peel = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(120, 300), "Detail": 2})
    h = build_color_ramp(nw, peel.outputs["Fac"], [0.3, 0.7], [(0, 0, 0, 1), (1, 1, 1, 1)])
    return h, uniform(0.0006, 0.0020)


def shader_stucco(nw: NodeWrangler, plaster_colored=None, style=None, **kwargs):
    front, back = _sample_plaster_pair(plaster_colored)
    style = style or STYLES[int(np.random.randint(len(STYLES)))]
    uv = nw.new_node(Nodes.UVMap)
    height, depth = _height(nw, uv, style)
    mottle = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(3, 10), "Detail": 6})
    base = build_color_ramp(nw, mottle.outputs["Fac"], [uniform(0.2, 0.35), 1.0], [back, front])
    # Paint pools slightly darker in the low spots of heavy texture.
    shade = build_color_ramp(nw, height, [0.0, 1.0],
                             [(uniform(0.78, 0.9),) * 3 + (1,), (1.0, 1.0, 1.0, 1.0)])
    color = nw.new_node(Nodes.MixRGB, [uniform(0.4, 0.9), base, shade], attrs={"blend_type": "MULTIPLY"})
    displacement = nw.new_node(Nodes.Displacement,
                               input_kwargs={"Height": height, "Scale": depth, "Midlevel": 0.0})
    bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
        "Base Color": color, "Roughness": uniform(0.6, 0.92), "Specular IOR Level": uniform(0.1, 0.2)})
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": bsdf, "Displacement": displacement})


class StuccoPlaster:
    shader = shader_stucco

    def generate(self, plaster_colored=None, **kwargs):
        if plaster_colored is None:
            plaster_colored = uniform() < plaster_paint_params()["colored_prob"]
        return surface.shaderfunc_to_material(shader_stucco, plaster_colored, **kwargs)

    __call__ = generate
