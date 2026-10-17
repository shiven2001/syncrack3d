"""Clean-wall negative materials: large-format wall tile and irregular stone.

CWG9's remaining BD3 plain-wall false alarms were (1) large plain wall tiles
whose thin grout lines read as cracks, and (2) stone walls with dark, uneven
joints. Infinigen's Tile draws small patterned/hexagonal tiles and StoneBlock
draws regular rows, so neither covers these. UV in metres (CWG metric UVs).
"""

import numpy as np
from numpy.random import uniform

from cwg_infinigen.core import surface
from cwg_infinigen.core.nodes.node_info import Nodes
from cwg_infinigen.core.nodes.node_utils import build_color_ramp
from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
from cwg_infinigen.core.util.color import hsv2rgba
from cwg_infinigen.core.util.random import log_uniform


def shader_large_tile(nw: NodeWrangler, **kwargs):
    """20-60 cm rectangular or square tiles, 1.5-4 mm grout, semi-gloss glaze."""
    uv = nw.new_node(Nodes.UVMap)
    if uniform() < 0.5:
        uv = nw.new_node(Nodes.Mapping, [uv], input_kwargs={"Rotation": (0.0, 0.0, 1.5708)})
    row = uniform(0.20, 0.60)
    hue, sat = uniform(0.0, 1.0), uniform(0.0, 0.10)
    value = uniform(0.45, 0.90)
    tile_a = hsv2rgba(hue, sat, value)
    tile_b = hsv2rgba(hue, sat, value * uniform(0.94, 1.0))
    grout = hsv2rgba(hue, sat * 0.5, value * uniform(0.35, 1.05))
    tiles = nw.new_node(
        Nodes.BrickTexture,
        [uv, tile_a, tile_b, grout],
        input_kwargs={
            "Scale": 1,
            "Row Height": row,
            "Brick Width": row * float(np.random.choice([1.0, 1.0, 2.0, 1.5])),
            "Mortar Size": uniform(0.0015, 0.004),
            "Mortar Smooth": 0.1,
            "Bias": 0.0,
        },
        attrs={"offset": float(np.random.choice([0.0, 0.0, 0.5])), "offset_frequency": 2,
               "squash": 1.0, "squash_frequency": 1},
    )
    glaze = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(2, 8), "Detail": 3})
    color = nw.new_node(
        Nodes.MixRGB, [uniform(0.02, 0.08), tiles.outputs["Color"], glaze.outputs["Color"]],
        attrs={"blend_type": "OVERLAY"},
    )
    # Fac is 1 in the grout. 1-2 mm of recess; the old 0.5-1.5 mm read as flat
    # once the joint was a real shadow rather than a colour line.
    height = nw.scalar_multiply(nw.scalar_sub2(1.0, tiles.outputs["Fac"]), 1.0)
    displacement = nw.new_node(
        Nodes.Displacement, input_kwargs={"Height": height, "Scale": uniform(0.001, 0.002), "Midlevel": 0.0}
    )
    bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": color, "Roughness": uniform(0.08, 0.45)},
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": bsdf, "Displacement": displacement})


def shader_irregular_stone(nw: NodeWrangler, **kwargs):
    """Cobble-like stones of mixed size and shape from Voronoi cells, dark or light joints."""
    uv = nw.new_node(Nodes.UVMap)
    stretch = uniform(1.0, 2.2)
    uv = nw.new_node(
        Nodes.Mapping, [uv],
        input_kwargs={"Rotation": (0.0, 0.0, float(np.random.choice([0.0, 1.5708]))),
                      "Scale": (1.0, stretch, 1.0)},
    )
    size = uniform(0.06, 0.16)  # typical stone diameter, metres
    cells = nw.new_node(
        Nodes.VoronoiTexture, [uv],
        input_kwargs={"Scale": 1.0 / size, "Randomness": uniform(0.6, 1.0)},
        attrs={"feature": "DISTANCE_TO_EDGE"},
    )
    stones = nw.new_node(
        Nodes.VoronoiTexture, [uv], input_kwargs={"Scale": 1.0 / size, "Randomness": 1.0},
    )
    edge = cells.outputs["Distance"]
    joint = build_color_ramp(nw, edge, [0.0, uniform(0.04, 0.10)], [(1, 1, 1, 1), (0, 0, 0, 1)])
    hue, sat = uniform(0.03, 0.62), uniform(0.0, 0.08)
    value = uniform(0.10, 0.40)
    stone_col = build_color_ramp(
        nw, stones.outputs["Color"],
        [0.0, 1.0], [hsv2rgba(hue, sat, value * 0.6), hsv2rgba(hue, sat, min(value * 1.5, 0.65))],
    )
    speckle = nw.new_node(Nodes.VoronoiTexture, [uv], input_kwargs={"Scale": uniform(400, 900), "Randomness": 1.0})
    speckle = build_color_ramp(nw, speckle.outputs["Distance"], [0.15, 0.55],
                               [(0.4, 0.4, 0.4, 1), (1.4, 1.4, 1.4, 1)])
    stone_col = nw.new_node(Nodes.MixRGB, [uniform(0.5, 0.9), stone_col, speckle],
                            attrs={"blend_type": "MULTIPLY"})
    mortar = hsv2rgba(hue, sat * 0.5, value * float(np.random.choice([uniform(0.25, 0.6), uniform(1.2, 1.8)])))
    color = nw.new_node(Nodes.MixRGB, [joint, stone_col, mortar], attrs={"blend_type": "MIX"})
    face = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(30, 80), "Detail": 10})
    height = nw.scalar_add(
        nw.scalar_multiply(nw.scalar_sub2(1.0, joint), uniform(0.5, 0.9)),
        nw.scalar_multiply(face.outputs["Fac"], uniform(0.5, 1.0)),
    )
    displacement = nw.new_node(
        Nodes.Displacement, input_kwargs={"Height": height, "Scale": uniform(0.006, 0.016), "Midlevel": 0.0}
    )
    bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={"Base Color": color, "Roughness": uniform(0.7, 0.95)})
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": bsdf, "Displacement": displacement})


class LargeTile:
    shader = shader_large_tile

    def generate(self):
        return surface.shaderfunc_to_material(shader_large_tile)

    __call__ = generate


class IrregularStone:
    shader = shader_irregular_stone

    def generate(self):
        return surface.shaderfunc_to_material(shader_irregular_stone)

    __call__ = generate
