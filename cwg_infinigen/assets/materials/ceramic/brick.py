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


def _detail_rng(*values):
    """Extra texture draws that do not advance the global crack RNG."""
    acc = 2166136261
    for value in values:
        acc ^= int(abs(float(value)) * 1_000_000) & 0xFFFFFFFF
        acc = (acc * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(acc or 1)


def _bump(nw, height, distance, normal=None, strength=0.45):
    kwargs = {"Height": height, "Distance": distance, "Strength": strength}
    if normal is not None:
        kwargs["Normal"] = normal
    return nw.new_node(Nodes.Bump, input_kwargs=kwargs, strict=False)


def _amount(value, rng, low, high):
    """A pinned amount, or a draw from none (low) up to a light amount (high)."""
    if value is None:
        return float(rng.uniform(low, high))
    return float(np.clip(value, 0.0, 1.0))


def shader_brick(nw: NodeWrangler, height=None, dirt=None, speckle=None, bump=None, **kwargs):
    if height is None:
        height = log_uniform(0.07, 0.12)
    uv_map = nw.new_node(Nodes.UVMap)

    # Red clay. The two values are pushed apart so neighbouring bricks differ.
    # Same six draws as the old front/back colours.
    h1, s1, v1 = uniform(0.0, 0.015), uniform(0.65, 0.90), log_uniform(0.04, 0.14)
    h2, s2, v2 = uniform(0.0, 0.015), uniform(0.65, 0.90), log_uniform(0.04, 0.14)
    if v1 <= v2:
        v1 *= 0.55
    else:
        v2 *= 0.55
    front_color, back_color = hsv2rgba(h1, s1, v1), hsv2rgba(h2, s2, v2)
    # Slightly cool gray, so it stays gray under the warm wall light.
    mortar_color = hsv2rgba(uniform(0.55, 0.65), uniform(0.04, 0.12), log_uniform(0.16, 0.30))
    dark_color = hsv2rgba(uniform(0.0, 0.02), uniform(0.45, 0.75), log_uniform(0.05, 0.12))
    noise = nw.new_node(
        Nodes.NoiseTexture,
        [uv_map],
        input_kwargs={"Scale": uniform(40, 50), "Detail": uniform(15, 20)},
    )
    brick_width = height * log_uniform(1.2, 2.5)
    mortar_size = height * log_uniform(0.04, 0.08)
    # Same draw order as the previous shader, so the crack sampler that
    # continues from this material is unchanged.
    grain_scale = uniform(2, 5)
    darken = log_uniform(0.5, 1.0)
    noise_scale = uniform(60, 100)
    noise_detail = uniform(8, 14)
    noise_distort = uniform(1, 4)
    recess = uniform(0.003, 0.010)
    grain_w = uniform(0.0002, 0.0008)
    pit_w = uniform(0.0003, 0.0012)
    detail = _detail_rng(
        height, brick_width, mortar_size, darken, grain_scale, noise_scale,
        noise_detail, noise_distort, recess, grain_w, pit_w,
    )

    # Low-frequency warp so courses are not a perfect grid. Linear-light on
    # the UV is just a signed noise offset; the amplitude stays under a
    # couple of centimetres so joints do not melt.
    warp = nw.new_node(
        Nodes.NoiseTexture,
        [uv_map],
        input_kwargs={"Scale": float(detail.uniform(1.2, 2.6)), "Detail": 2.0},
    )
    warp_centered = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: warp.outputs["Color"], 1: (0.5, 0.5, 0.5)},
        attrs={"operation": "SUBTRACT"},
    )
    warp_scaled = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: warp_centered, "Scale": float(detail.uniform(0.003, 0.010))},
        attrs={"operation": "SCALE"},
    )
    uv = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: uv_map, 1: warp_scaled},
        attrs={"operation": "ADD"},
    )

    bricks = nw.new_node(
        Nodes.BrickTexture,
        [uv, front_color, back_color, mortar_color],
        input_kwargs={
            "Scale": 1,
            "Row Height": height,
            "Brick Width": brick_width,
            "Mortar Size": mortar_size,
            "Mortar Smooth": noise,
        },
    )
    # One joint system only. A second brick colour stamped a smaller rectangle
    # inside each face, because our mortar is lighter than the clay and Darken
    # then keeps the inner clay. Size variation comes from the UV warp.
    color = bricks.outputs["Color"]
    # Fac is 1 in the joint and 0 on the brick. Relief has to come from this
    # mask; driving it from Color made the recess follow how dark the clay was.
    mortar = bricks.outputs["Fac"]
    grain = nw.new_node(
        Nodes.MusgraveTexture, [uv], input_kwargs={"Scale": grain_scale}
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [nw.scalar_multiply(nw.scalar_multiply(darken, 0.18), grain), color, dark_color],
        attrs={"blend_type": "DARKEN"},
    )
    # Face grit: fine clay specks. 0 leaves the colour flat; the random draw
    # goes up to a light coat.
    speckle_amt = _amount(speckle, detail, 0.0, 0.45)
    grit = nw.new_node(
        Nodes.NoiseTexture,
        [uv],
        input_kwargs={"Scale": float(detail.uniform(180, 320)), "Detail": 8.0},
    )
    grit_shade = nw.scalar_add(
        nw.scalar_sub2(1.0, speckle_amt * 1.15),
        nw.scalar_multiply(grit.outputs["Fac"], speckle_amt * 1.60),
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [1.0, color, grit_shade],
        attrs={"blend_type": "MULTIPLY"},
    )
    # Patchy grime, heavier in the joints. 0 is clean. The random draw stops
    # at a light scatter, about the reference photo, not a coated wall.
    dirt_amt = _amount(dirt, detail, 0.0, 0.55)
    # Fine specks across the courses, gathered into dirtier patches.
    dirt_fine = nw.new_node(
        Nodes.NoiseTexture,
        [uv],
        input_kwargs={"Scale": float(detail.uniform(90.0, 170.0)), "Detail": 4.0},
    )
    specks = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": dirt_fine.outputs["Fac"],
            "From Min": 0.55,
            "From Max": 0.72,
            "To Min": 0.0,
            "To Max": 1.0,
        },
    )
    dirt_noise = nw.new_node(
        Nodes.NoiseTexture,
        [uv],
        input_kwargs={"Scale": float(detail.uniform(4.0, 9.0)), "Detail": 3.0},
    )
    gathered = nw.scalar_multiply(
        specks,
        nw.scalar_add(0.25, nw.scalar_multiply(dirt_noise.outputs["Fac"], 0.75)),
    )
    in_joint = nw.scalar_multiply(mortar, float(detail.uniform(0.70, 1.0)))
    dirt_mask = nw.scalar_max2(gathered, in_joint)
    dirt_color = hsv2rgba(
        float(detail.uniform(0.08, 0.30)),
        float(detail.uniform(0.30, 0.60)),
        float(detail.uniform(0.04, 0.10)),
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [1.0, color, nw.scalar_sub2(1.0, nw.scalar_multiply(dirt_amt * 0.95, dirt_mask))],
        attrs={"blend_type": "MULTIPLY"},
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [nw.scalar_multiply(dirt_amt * 0.85, dirt_mask), color, dirt_color],
        attrs={"blend_type": "MIX"},
    )
    face_rough = nw.build_float_curve(
        nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50}),
        [(0, 0.70), (1, 0.90)],
    )
    # Mortar is rougher than the clay face.
    roughness = nw.scalar_add(
        nw.scalar_multiply(nw.scalar_sub2(1.0, mortar), face_rough),
        nw.scalar_multiply(mortar, 0.94),
    )

    musgrave_disp = nw.new_node(Nodes.MusgraveTexture, input_kwargs={"Scale": 50})
    noise_disp = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Scale": noise_scale,
            "Detail": noise_detail,
            "Distortion": noise_distort,
        },
    )
    # Real mortar is recessed 3-10 mm. Face grain stays under about 1 mm.
    # Crack injection adds its own groove onto this height; these bumps are
    # shading only and do not move that displacement.
    face = nw.scalar_add(
        nw.scalar_multiply(musgrave_disp, grain_w),
        nw.scalar_multiply(noise_disp.outputs["Fac"], pit_w),
    )
    height_m = nw.scalar_add(
        nw.scalar_multiply(nw.scalar_sub2(1.0, mortar), recess),
        face,
    )
    chunks = nw.new_node(
        Nodes.NoiseTexture, [uv], input_kwargs={"Scale": 8.0, "Detail": 2.0}
    )
    bump_amt = _amount(bump, detail, 0.25, 1.0)
    normal = _bump(nw, mortar, 0.0012, strength=0.50 * bump_amt)
    normal = _bump(nw, noise_disp.outputs["Fac"], 0.0008, normal.outputs["Normal"], 0.70 * bump_amt)
    normal = _bump(nw, chunks.outputs["Fac"], 0.0016, normal.outputs["Normal"], 0.40 * bump_amt)
    normal = _bump(nw, grit.outputs["Fac"], 0.0005, normal.outputs["Normal"], 0.55 * bump_amt)
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Roughness": roughness, "Base Color": color, "Normal": normal},
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
    # None draws per wall: dirt and speckle from 0 (none) to a light amount,
    # bump from a little shading to full. Pass a number in 0..1 to pin one.
    dirt = None
    speckle = None
    bump = None

    def generate(self, dirt=None, speckle=None, bump=None):
        return surface.shaderfunc_to_material(
            shader_brick,
            dirt=self.dirt if dirt is None else dirt,
            speckle=self.speckle if speckle is None else speckle,
            bump=self.bump if bump is None else bump,
        )

    __call__ = generate


def make_sphere():
    obj = new_plane()
    obj.rotation_euler[0] = np.pi / 2
    butil.apply_transform(obj, True)
    return obj
