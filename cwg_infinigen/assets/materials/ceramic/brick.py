# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

import numpy as np
from numpy.random import uniform

from cwg_infinigen.assets.utils.object import new_plane
from cwg_infinigen.core import surface
from cwg_infinigen.core.nodes.node_info import Nodes
from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
from cwg_infinigen.core.util import blender as butil
from cwg_infinigen.core.util.color import hsv2rgba
from cwg_infinigen.core.util.random import log_uniform


def shader_brick(nw: NodeWrangler, height=None, **kwargs):
    if height is None:
        height = log_uniform(0.07, 0.12)
    uv_map = nw.new_node(Nodes.UVMap)

    front_color, back_color = [
        hsv2rgba(uniform(0, 0.05), uniform(0.8, 1), log_uniform(0.02, 0.5))
        for _ in range(2)
    ]
    mortar_color = hsv2rgba(uniform(0, 0.05), uniform(0.2, 0.5), log_uniform(0.02, 0.8))
    dark_color = hsv2rgba(uniform(0, 0.05), uniform(0.8, 1), log_uniform(0.005, 0.02))
    noise = nw.new_node(
        Nodes.NoiseTexture,
        [uv_map],
        input_kwargs={"Scale": uniform(40, 50), "Detail": uniform(15, 20)},
    )
    bricks = nw.new_node(
        Nodes.BrickTexture,
        [uv_map, front_color, back_color, mortar_color],
        input_kwargs={
            "Scale": 1,
            "Row Height": height,
            "Brick Width": height * log_uniform(1.2, 2.5),
            "Mortar Size": height * log_uniform(0.04, 0.08),
            "Mortar Smooth": noise,
        },
    )
    color = bricks.outputs["Color"]
    # Fac is 1 in the joint and 0 on the brick. Relief has to come from this
    # mask; driving it from Color made the recess follow how dark the clay was.
    mortar = bricks.outputs["Fac"]
    noise = nw.new_node(
        Nodes.MusgraveTexture, [uv_map], input_kwargs={"Scale": uniform(2, 5)}
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [nw.scalar_multiply(log_uniform(0.5, 1.0), noise), color, dark_color],
        attrs={"blend_type": "DARKEN"},
    )

    roughness = nw.build_float_curve(
        nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50}),
        [(0, 0.5), (1, 1.0)],
    )

    musgrave_disp = nw.new_node(Nodes.MusgraveTexture, input_kwargs={"Scale": 50})
    noise_disp = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Scale": uniform(60, 100),
            "Detail": uniform(8, 14),
            "Distortion": uniform(1, 4),
        },
    )
    # Same three draws as the old colour/musgrave/noise weights, so the crack
    # RNG that continues from this shader is unchanged. Real mortar is recessed
    # 3-10 mm; the face grain is under about 1 mm and does not come from colour.
    recess = uniform(0.003, 0.010)
    face = nw.scalar_add(
        nw.scalar_multiply(musgrave_disp, uniform(0.0002, 0.0008)),
        nw.scalar_multiply(noise_disp.outputs["Fac"], uniform(0.0003, 0.0012)),
    )
    height_m = nw.scalar_add(
        nw.scalar_multiply(nw.scalar_sub2(1.0, mortar), recess),
        face,
    )
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Roughness": roughness, "Base Color": color},
    )
    displacement = nw.new_node(
        Nodes.Displacement,
        # Midlevel defaults to 0.5, which displaces the surface by -0.5*Scale
        # along its normal regardless of Height. Height is already metres.
        input_kwargs={"Height": height_m, "Scale": 1.0, "Midlevel": 0.0},
    )
    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf, "Displacement": displacement},
    )


class Brick:
    shader = shader_brick

    def generate(self):
        return surface.shaderfunc_to_material(shader_brick)

    __call__ = generate


def make_sphere():
    obj = new_plane()
    obj.rotation_euler[0] = np.pi / 2
    butil.apply_transform(obj, True)
    return obj
