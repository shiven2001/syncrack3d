"""Grey split-face stone block cladding with recessed mortar joints.

Added as a clean-wall (negative) material: every plain-wall false alarm of the
CWG7/CWG8 synthetic-only models on BD3 was granite or stone block cladding,
whose mortar joints were mistaken for cracks, and the existing Brick material
only draws saturated red clay brick. Grey to warm-grey blocks, per-block tone
variation, irregular staggered rows, granite speckle, rough faces and joints
set back as real displacement. Uses UV in metres (CWG sets metric wall UVs).
"""

from numpy.random import uniform

from cwg_infinigen.core import surface
from cwg_infinigen.core.nodes.node_info import Nodes
from cwg_infinigen.core.nodes.node_utils import build_color_ramp
from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
from cwg_infinigen.core.util.color import hsv2rgba
from cwg_infinigen.core.util.random import log_uniform


def shader_stone_block(nw: NodeWrangler, **kwargs):
    uv_map = nw.new_node(Nodes.UVMap)
    # Half the walls run in vertical stacked columns, as many BD3 claddings do.
    vertical = uniform() < 0.5
    uv = nw.new_node(
        Nodes.Mapping, [uv_map],
        input_kwargs={"Rotation": (0.0, 0.0, 1.5708 if vertical else 0.0)},
    )
    # Distort joints slightly so blocks are not ruler-straight.
    wobble = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(4, 10), "Detail": 2})
    uv = nw.new_node(
        Nodes.MixRGB, [uniform(0.004, 0.015), uv, wobble.outputs["Color"]],
        attrs={"blend_type": "LINEAR_LIGHT"},
    )
    row = uniform(0.05, 0.13)  # block height, metres
    hue = uniform(0.05, 0.62)
    sat = uniform(0.0, 0.07)
    value = uniform(0.10, 0.40)
    block_a = hsv2rgba(hue, sat, value)
    block_b = hsv2rgba(hue + uniform(-0.03, 0.03), sat, min(value * uniform(0.6, 1.5), 0.65))
    # Joints: darker than, similar to, or a little lighter than the blocks.
    mortar = hsv2rgba(hue, sat * 0.5, value * uniform(0.5, 1.4))

    bricks = nw.new_node(
        Nodes.BrickTexture,
        [uv, block_a, block_b, mortar],
        input_kwargs={
            "Scale": 1,
            "Row Height": row,
            "Brick Width": row * log_uniform(0.8, 2.2),
            "Mortar Size": row * uniform(0.03, 0.08),
            "Mortar Smooth": uniform(0.05, 0.3),
            "Bias": uniform(-0.5, 0.5),
        },
        attrs={
            "offset": uniform(0.3, 0.7),
            "offset_frequency": 2,
            "squash": uniform(0.6, 1.4),
            "squash_frequency": 2,
        },
    )
    joint = bricks.outputs["Fac"]  # 1 in the mortar

    # Granite speckle and coarse mottling inside each block.
    speckle = nw.new_node(
        Nodes.VoronoiTexture,
        [uv_map],
        input_kwargs={"Scale": uniform(350, 900), "Randomness": 1.0},
    )
    speckle = build_color_ramp(
        nw, speckle.outputs["Distance"], [uniform(0.10, 0.25), uniform(0.45, 0.65)],
        [(0.35, 0.35, 0.35, 1), (1.45, 1.45, 1.45, 1)],
    )
    color = nw.new_node(
        Nodes.MixRGB, [uniform(0.6, 0.95), bricks.outputs["Color"], speckle],
        attrs={"blend_type": "MULTIPLY"},
    )
    mottle = nw.new_node(Nodes.MusgraveTexture, [uv_map], input_kwargs={"Scale": uniform(3, 9)})
    color = nw.new_node(
        Nodes.MixRGB, [nw.scalar_multiply(uniform(0.15, 0.35), mottle), color, (0.1, 0.1, 0.1, 1)],
        attrs={"blend_type": "MIX"},
    )

    # Split-face relief: rough block faces stand proud of recessed joints.
    face = nw.new_node(
        Nodes.NoiseTexture,
        [uv_map],
        input_kwargs={"Scale": uniform(30, 90), "Detail": uniform(8, 14), "Distortion": uniform(0.5, 1.5)},
    )
    height = nw.scalar_add(
        nw.scalar_multiply(nw.scalar_sub2(1.0, joint), uniform(0.5, 0.9)),
        nw.scalar_multiply(face.outputs["Fac"], uniform(0.6, 1.0)),
    )
    displacement = nw.new_node(
        Nodes.Displacement,
        input_kwargs={"Height": height, "Scale": uniform(0.006, 0.018), "Midlevel": 0.0},
    )
    roughness = nw.build_float_curve(
        nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 40}), [(0, 0.7), (1, 0.95)]
    )
    bsdf = nw.new_node(
        Nodes.PrincipledBSDF, input_kwargs={"Base Color": color, "Roughness": roughness}
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": bsdf, "Displacement": displacement})


class StoneBlock:
    shader = shader_stone_block

    def generate(self):
        return surface.shaderfunc_to_material(shader_stone_block)

    __call__ = generate
