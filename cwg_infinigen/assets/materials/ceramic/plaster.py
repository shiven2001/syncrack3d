# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from typing import Iterable

import gin
from numpy.random import choice, uniform

from cwg_infinigen.assets.materials.utils import common
from cwg_infinigen.assets.utils.uv import unwrap_normal
from cwg_infinigen.core import surface
from cwg_infinigen.core.nodes.node_info import Nodes
from cwg_infinigen.core.nodes.node_utils import build_color_ramp
from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
from cwg_infinigen.core.util.color import hsv2rgba
from cwg_infinigen.core.util.random import log_uniform


# Builder-grade pre-occupancy paints (linear-ish RGB).
_PREOCCUPANCY_PAINTS = (
    (0.96, 0.96, 0.94),  # bright white
    (0.92, 0.92, 0.90),  # off-white
    (0.88, 0.88, 0.86),  # eggshell
    (0.82, 0.82, 0.81),  # light grey
    (0.74, 0.74, 0.73),  # mid grey
    (0.86, 0.84, 0.80),  # warm greige
    (0.90, 0.89, 0.86),  # warm white
)


@gin.configurable
def plaster_paint_params(
    colored_prob=0.0,
    max_saturation=0.05,
    value_min=0.74,
    value_max=0.95,
    hue_min=0.04,
    hue_max=0.14,
    sat_min=0.03,
):
    """Neutral plaster defaults for unoccupied interiors. Override via gin.

    hue_min/hue_max/sat_min bound the colored branch (defaults: the original warm band).
    """
    return {
        "colored_prob": colored_prob,
        "max_saturation": max_saturation,
        "value_min": value_min,
        "value_max": value_max,
        "hue_min": hue_min,
        "hue_max": hue_max,
        "sat_min": sat_min,
    }


def _sample_plaster_pair(plaster_colored):
    params = plaster_paint_params()
    if plaster_colored is None:
        plaster_colored = uniform() < params["colored_prob"]

    if plaster_colored:
        hue = uniform(params["hue_min"], params["hue_max"]) % 1.0
        sat = uniform(params["sat_min"], max(params["sat_min"] + 0.01, params["max_saturation"] + 0.06))
        front_value = uniform(params["value_min"], params["value_max"])
        back_value = front_value * uniform(0.88, 0.98)
        front = hsv2rgba(hue, sat, front_value)
        back = hsv2rgba(hue + uniform(-0.02, 0.02), sat * uniform(0.6, 1.0), back_value)
        return front, back

    base = _PREOCCUPANCY_PAINTS[int(choice(len(_PREOCCUPANCY_PAINTS)))]
    jitter = uniform(-0.02, 0.02, size=3)
    front_rgb = tuple(float(min(1.0, max(0.45, c + j))) for c, j in zip(base, jitter))
    shade = uniform(0.90, 0.97)
    back_rgb = tuple(float(min(1.0, max(0.40, c * shade))) for c in front_rgb)
    front = (*front_rgb, 1.0)
    back = (*back_rgb, 1.0)
    return front, back


def shader_plaster(nw: NodeWrangler, plaster_colored=None, **kwargs):
    """Eggshell paint: mottled plaster, orange-peel bump, corner AO."""
    front_color, back_color = _sample_plaster_pair(plaster_colored)

    uv_map = nw.new_node(Nodes.UVMap)
    obj_coord = nw.new_node(Nodes.TextureCoord).outputs["Object"]

    musgrave = nw.new_node(
        Nodes.MusgraveTexture,
        [uv_map],
        input_kwargs={"Scale": log_uniform(8, 18), "Detail": log_uniform(10, 20), "Dimension": 0},
    )
    noise = nw.new_node(
        Nodes.NoiseTexture,
        [uv_map],
        input_kwargs={
            "Scale": log_uniform(4, 10),
            "Detail": log_uniform(10, 18),
            "Distortion": log_uniform(0.4, 1.5),
        },
    )
    noise = build_color_ramp(
        nw, noise, [0, uniform(0.35, 0.55)], [(0, 0, 0, 1), (1, 1, 1, 1)]
    )
    mottling = nw.new_node(
        Nodes.MixRGB, [musgrave, noise], attrs={"blend_type": "DIFFERENCE"}
    )
    base_color = build_color_ramp(
        nw, mottling, [uniform(0.15, 0.35), 1], [back_color, front_color]
    )

    # Room-scale mottling, the dirt layer and the material-level ambient
    # occlusion all used to sit here. Removed: together they were painting
    # metre-scale grey/brown patches across clean plaster and darkening every
    # corner and window reveal.
    #
    # Measured on one generated wall, the three were:
    #
    #   large_mottle  Noise at object scale 0.28-0.75 (0.703 here), mixed
    #                 10-20% toward a darkened base - patches several metres
    #                 across, which read as smudging rather than as plaster.
    #   dirt          a second object-scale Noise (0.45-1.6, 0.562 here) mixed
    #                 8-18% toward (0.50, 0.48, 0.45) - a grime wash over the
    #                 whole surface.
    #   ao            a shader Ambient Occlusion node at 0.28-0.55 m radius
    #                 (0.513 here) mixed 22-38% toward (0.40, 0.38, 0.35).
    #
    # The AO term is the one worth naming carefully, because it is not a
    # texture: a half-metre occlusion radius darkens anything within half a
    # metre of another surface, so every wall corner, window reveal and door
    # jamb picked up a broad dark band. That is the dark streaking seen at
    # openings, and it is baked into the material rather than being lighting,
    # so it appears in the viewport and in every render regardless of where the
    # lamps are.
    #
    # It is also wrong for this dataset specifically. Defect visibility here is
    # supposed to be a function of geometry and light direction; a fixed
    # albedo-space darkening near corners adds contrast that no light can
    # remove and that no real wall has.
    #
    # Fine-scale plaster texture below (`peel`, the micro noise and the
    # displacement) is kept - that is real surface, not painted-on dirt.

    peel = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Scale": uniform(180, 320),
            "Detail": log_uniform(8, 14),
            "Distortion": log_uniform(0.5, 2.0),
        },
    )
    roller = nw.new_node(
        Nodes.MusgraveTexture, input_kwargs={"Scale": uniform(28, 70), "Detail": 5}
    )
    disp_height = nw.scalar_add(
        nw.scalar_multiply(peel.outputs["Fac"], log_uniform(0.45, 0.9)),
        nw.scalar_multiply(roller, log_uniform(0.22, 0.5)),
    )
    displacement = nw.new_node(
        Nodes.Displacement,
        input_kwargs={
            "Scale": log_uniform(0.0022, 0.0050),
            "Height": disp_height,
            # Midlevel defaults to 0.5, so a height of ~0 displaces the whole
            # surface by -0.5*Scale along its normal. Fine as a bump map, but
            # real geometry under displacement_mode="BOTH" - it crumpled small
            # metal parts (taps, handles) into blobs.
            "Midlevel": 0.0,
        },
    )

    roughness = nw.build_float_curve(
        nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": uniform(12, 36)}),
        [(0, uniform(0.55, 0.66)), (1, uniform(0.74, 0.90))],
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": base_color,
            "Roughness": roughness,
            "Specular IOR Level": uniform(0.12, 0.22),
        },
    )

    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf, "Displacement": displacement},
    )


class Plaster:
    shader = shader_plaster

    def apply(self, obj, selection=None, plaster_colored=None, **kwargs):
        if plaster_colored is None:
            plaster_colored = uniform() < plaster_paint_params()["colored_prob"]
        for o in obj if isinstance(obj, Iterable) else [obj]:
            unwrap_normal(o, selection)
        common.apply(
            obj, shader_plaster, selection, plaster_colored=plaster_colored, **kwargs
        )

    def generate(self, plaster_colored=None, **kwargs):
        if plaster_colored is None:
            plaster_colored = uniform() < plaster_paint_params()["colored_prob"]
        return surface.shaderfunc_to_material(shader_plaster, plaster_colored, **kwargs)

    __call__ = generate
