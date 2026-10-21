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


def _grout_material(kind, t, tile_value):
    """A joint that is not a darker copy of the tile: cement, plaster, or resin."""
    if kind == "concrete":
        value, sat, rough = 0.18 + 0.12 * t, 0.04, 0.88 + 0.06 * t
        hue = 0.07
    elif kind == "plaster":
        value, sat, rough = 0.86 + 0.08 * t, 0.03, 0.72 + 0.10 * t
        hue = 0.09
    else:
        value, sat, rough = 0.48 + 0.16 * t, 0.12, 0.30 + 0.16 * t
        hue = 0.10
    if abs(value - tile_value) < 0.18:
        value = 0.30 if tile_value > 0.55 else 0.86
    return hsv2rgba(hue, sat, value), rough


def shader_large_tile(nw: NodeWrangler, **kwargs):
    """20-60 cm tiles, 1.5-4 mm grout. Glaze can be neutral or a real colour.

    The random draws are the same calls, in the same order, as the gray-only
    shader, so the crack sampler that continues from here is unchanged.
    """
    uv = nw.new_node(Nodes.UVMap)
    if uniform() < 0.5:
        uv = nw.new_node(Nodes.Mapping, [uv], input_kwargs={"Rotation": (0.0, 0.0, 1.5708)})
    row = uniform(0.20, 0.60)
    hue = uniform(0.0, 1.0)
    # Was uniform(0, 0.10), which could only make a gray. The same one draw
    # now keeps some walls neutral and lets the rest be a coloured glaze.
    sat_u = uniform(0.0, 1.0)
    value = uniform(0.45, 0.90)
    vary = uniform(0.94, 1.0)
    grout_u = (float(uniform(0.35, 1.05)) - 0.35) / 0.70
    width = float(np.random.choice([1.0, 1.0, 2.0, 1.5]))
    mortar = uniform(0.0015, 0.004)
    offset = float(np.random.choice([0.0, 0.0, 0.5]))
    glaze_scale = uniform(2, 8)
    glaze_amt = uniform(0.02, 0.08)
    disp_scale = uniform(0.001, 0.002)
    tile_rough = uniform(0.08, 0.45)

    # One draw, four glazes: white, cream, a gray, or a colour.
    if sat_u < 0.12:
        sat = 0.012
        value = 0.90 + (value - 0.45) / 0.45 * 0.06
        hue = 0.0
    elif sat_u < 0.28:
        sat = 0.05 + (sat_u - 0.12) / 0.16 * 0.05
        value = 0.82 + (value - 0.45) / 0.45 * 0.10
        hue = 0.07 + (hue % 1.0) * 0.04
    elif sat_u < 0.42:
        sat = (sat_u - 0.28) / 0.14 * 0.06
    else:
        sat = 0.22 + (sat_u - 0.42) / 0.58 * 0.40
    span = (vary - 0.94) / 0.06
    tile_a = hsv2rgba(hue, sat, value)
    tile_b = hsv2rgba(
        (hue + (span - 0.5) * 0.03) % 1.0,
        min(sat * (0.75 + 0.45 * span), 0.7),
        value * (0.66 + 0.34 * span),
    )
    if grout_u < 0.34:
        kind = "concrete"
    elif grout_u < 0.67:
        kind = "plaster"
    else:
        kind = "resin"
    grout, grout_rough = _grout_material(kind, min(max(grout_u, 0.0), 1.0), value)

    tiles = nw.new_node(
        Nodes.BrickTexture,
        [uv, tile_a, tile_b, grout],
        input_kwargs={
            "Scale": 1,
            "Row Height": row,
            "Brick Width": row * width,
            "Mortar Size": mortar,
            "Mortar Smooth": 0.1,
            "Bias": 0.0,
        },
        attrs={"offset": offset, "offset_frequency": 2,
               "squash": 1.0, "squash_frequency": 1},
    )
    joint = tiles.outputs["Fac"]  # 1 in the grout
    glaze = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": glaze_scale, "Detail": 3})
    tile_color = nw.new_node(
        Nodes.MixRGB, [glaze_amt, tiles.outputs["Color"], glaze.outputs["Color"]],
        attrs={"blend_type": "OVERLAY"},
    )
    grain = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": 140.0, "Detail": 6})
    grout_shade = nw.scalar_add(0.86, nw.scalar_multiply(grain.outputs["Fac"], 0.22))
    grout_color = nw.new_node(
        Nodes.MixRGB, [1.0, grout, grout_shade], attrs={"blend_type": "MULTIPLY"}
    )
    # Fac is 1 in the grout, so the glaze stays on the face and the joint
    # material is the second colour.
    color = nw.new_node(Nodes.MixRGB, [joint, tile_color, grout_color])
    roughness = nw.scalar_add(
        nw.scalar_multiply(nw.scalar_sub2(1.0, joint), tile_rough),
        nw.scalar_multiply(joint, grout_rough),
    )
    # Fac is 1 in the grout. 1-2 mm of recess; the old 0.5-1.5 mm read as flat
    # once the joint was a real shadow rather than a colour line.
    height = nw.scalar_multiply(nw.scalar_sub2(1.0, joint), 1.0)
    displacement = nw.new_node(
        Nodes.Displacement, input_kwargs={"Height": height, "Scale": disp_scale, "Midlevel": 0.0}
    )
    bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": color, "Roughness": roughness},
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
