"""Exterior masonry walls: cobble/fieldstone in mortar, angular rubble, sandstone ashlar.

Infinigen's terrain stones (terrain/cobble_stone.py, stone.py, sandstone.py) are
landscape-scale and displace through a geometry modifier, so they cannot carry
injected defects on a CWG wall. These are wall-scale shader versions with the
contract defect injection needs (StoneBlock's): metric UVs, a linked Base Color
and real shader displacement. CobbleWall ports the terrain cobblestone's
Voronoi-cell construction (flat stones rising out of recessed joints, per-stone
tone from the cell id); RubbleStone and SandstoneAshlar follow the same pattern
for angular coursed rubble and bedded sandstone blocks. Variation over realism.
"""

import numpy as np
from numpy.random import choice, uniform

from cwg_infinigen.core import surface
from cwg_infinigen.core.nodes.node_info import Nodes
from cwg_infinigen.core.nodes.node_utils import build_color_ramp
from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
from cwg_infinigen.core.util.color import hsv2rgba


def _math(nw, op, a, b=None, clamp=False):
    inputs = [a] if b is None else [a, b]
    return nw.new_node(Nodes.Math, inputs, attrs={"operation": op, "use_clamp": clamp})


def _ramp(nw, fac, lo, hi, interp="LINEAR"):
    """Map fac from [lo, hi] to [0, 1], clamped."""
    return nw.new_node(Nodes.MapRange, input_kwargs={"Value": fac, "From Min": lo, "From Max": hi},
                       attrs={"interpolation_type": interp, "clamp": True})


def _warped_uv(nw, amount, scale, stretch=(1.0, 1.0)):
    uv = nw.new_node(Nodes.UVMap)
    uv = nw.new_node(Nodes.Mapping, [uv], input_kwargs={"Scale": (stretch[0], stretch[1], 1.0)})
    wob = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": scale, "Detail": 3})
    return nw.new_node(Nodes.MixRGB, [amount, uv, wob.outputs["Color"]], attrs={"blend_type": "LINEAR_LIGHT"})


def _stone_palette():
    kind = str(choice(["granite", "fieldstone", "river", "limestone"], p=[0.3, 0.3, 0.2, 0.2]))
    if kind == "granite":
        hue, sat, val = uniform(0.0, 1.0), uniform(0.0, 0.06), uniform(0.22, 0.50)
    elif kind == "fieldstone":
        hue, sat, val = uniform(0.05, 0.11), uniform(0.06, 0.18), uniform(0.18, 0.45)
    elif kind == "river":
        hue, sat, val = uniform(0.52, 0.62), uniform(0.04, 0.14), uniform(0.38, 0.62)
    else:
        hue, sat, val = uniform(0.09, 0.14), uniform(0.04, 0.12), uniform(0.42, 0.62)
    dark = hsv2rgba(hue, sat, val * uniform(0.40, 0.70))
    light = hsv2rgba(hue + uniform(-0.05, 0.05), sat * uniform(0.5, 1.4), min(val * uniform(1.3, 1.9), 0.88))
    return dark, light


def _mortar(dry=False):
    if dry:  # deep dry-stone joint: shadowed earth, no mortar face
        return hsv2rgba(uniform(0.05, 0.10), uniform(0.1, 0.3), uniform(0.03, 0.10))
    # Neutral and darker than the stones. A warm 0.30–0.70 mortar
    # survived white balance as pink cement.
    return hsv2rgba(uniform(0.05, 0.09), uniform(0.0, 0.025), uniform(0.06, 0.16))


def _detail_rng(*values):
    """Extra texture draws that do not advance the global crack RNG."""
    acc = 2166136261
    for value in values:
        acc ^= int(abs(float(value)) * 1_000_000) & 0xFFFFFFFF
        acc = (acc * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(acc or 1)


def _amount(value, rng, low, high):
    if value is None:
        return float(rng.uniform(low, high))
    return float(np.clip(value, 0.0, 1.0))


def _bump(nw, height, distance, normal=None, strength=0.45):
    kwargs = {"Height": height, "Distance": distance, "Strength": strength}
    if normal is not None:
        kwargs["Normal"] = normal
    return nw.new_node(Nodes.Bump, input_kwargs=kwargs, strict=False)


def _finish(nw, color, height, disp_scale, rough_lo=0.65, rough_hi=0.95, normal=None, roughness=None):
    displacement = nw.new_node(Nodes.Displacement,
                               input_kwargs={"Height": height, "Scale": disp_scale, "Midlevel": 0.0})
    # This draw stays even when a caller passes its own roughness, so the crack
    # stream that continues after the shader is unchanged.
    rough_noise = nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": uniform(20, 60)})
    if roughness is None:
        roughness = nw.build_float_curve(rough_noise, [(0, rough_lo), (1, rough_hi)])
    bsdf_kwargs = {"Base Color": color, "Roughness": roughness}
    if normal is not None:
        bsdf_kwargs["Normal"] = normal
    bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs=bsdf_kwargs)
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": bsdf, "Displacement": displacement})


def shader_cobble_wall(nw: NodeWrangler, dirt=None, speckle=None, bump=None, moss=None, **kwargs):
    """Fieldstones set in mortar. Faces are flat; the joint is the real displacement.

    The old height was a radial dome inside each Voronoi cell, which rendered as
    a circular blob on every stone. Those draws are still taken, in the same
    order, so the crack sampler that continues from this shader is unchanged.
    """
    size = uniform(0.05, 0.10)  # stone size, metres
    warp_amt = uniform(0.01, 0.04)
    warp_scale = uniform(3, 8)
    stretch = uniform(0.7, 1.0)
    rnd = uniform(0.6, 1.0)
    joint_w = uniform(0.018, 0.040)  # mortar half-width in cell units
    uniform(0.45, 0.65)  # was the dome radius
    uniform(0.5, 0.8)  # was the dome power
    base = uniform(0.3, 0.5)
    dark, light = _stone_palette()
    grain_scale = uniform(150, 400)
    overlay = uniform(0.28, 0.50)
    dry = uniform() < 0.25
    mortar = _mortar(dry=dry)
    mortar = tuple(c * uniform(0.55, 0.9) for c in mortar[:3]) + (1.0,)
    lump_freq = uniform(2, 4)
    stone_h = uniform(0.7, 1.0)
    lump_amp = uniform(0.15, 0.35)
    detail = _detail_rng(size, warp_amt, warp_scale, stretch, rnd, joint_w, base, grain_scale,
                         overlay, lump_freq, stone_h, lump_amp)

    uv = _warped_uv(nw, warp_amt, warp_scale, stretch=(stretch, 1.0))
    cells = dict(input_kwargs={"Scale": 1.0 / size, "Randomness": rnd})
    edge = nw.new_node(Nodes.VoronoiTexture, [uv], **cells, attrs={"feature": "DISTANCE_TO_EDGE"})
    cell = nw.new_node(Nodes.VoronoiTexture, [uv], **cells)
    # Short bevel at the joint. The face stays a plateau, so the centre of a
    # stone is not a circle.
    rim = _ramp(nw, edge.outputs["Distance"], joint_w * 0.30, joint_w + 0.045, "SMOOTHSTEP")
    tone = nw.new_node(Nodes.SeparateColor, [cell.outputs["Color"]]).outputs[0]
    stone = nw.new_node(Nodes.MixRGB, [tone, dark, light])
    grain = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": grain_scale, "Detail": 8})
    stone = nw.new_node(
        Nodes.MixRGB, [overlay * 0.35, stone, grain.outputs["Color"]], attrs={"blend_type": "OVERLAY"}
    )
    joint = _ramp(nw, edge.outputs["Distance"], joint_w * 0.6, joint_w)
    # The mortar bed is nearly black, which read as an empty hole. Pull it
    # toward soil so the joint is dirt, and the stones stay the gray faces.
    soil = hsv2rgba(0.07, 0.42, 0.22)
    bed = nw.new_node(Nodes.MixRGB, [0.85, mortar, soil])
    color = nw.new_node(Nodes.MixRGB, [joint, bed, stone])

    speckle_amt = _amount(speckle, detail, 0.0, 0.45)
    grit = nw.new_node(
        Nodes.NoiseTexture, [uv], input_kwargs={"Scale": float(detail.uniform(180, 360)), "Detail": 8}
    )
    grit_shade = nw.scalar_add(
        nw.scalar_sub2(1.0, speckle_amt * 1.05),
        nw.scalar_multiply(grit.outputs["Fac"], speckle_amt * 1.45),
    )
    # Grain on the stone face. The joint keeps an even mortar colour.
    grit_shade = nw.new_node(Nodes.MixRGB, [joint, (1.0, 1.0, 1.0, 1.0), grit_shade])
    color = nw.new_node(Nodes.MixRGB, [1.0, color, grit_shade], attrs={"blend_type": "MULTIPLY"})

    # Earth in the joint and along the stone edge, from none to a light fill.
    dirt_amt = _amount(dirt, detail, 0.0, 0.55)
    crevice = nw.scalar_sub2(1.0, joint)
    edge_band = nw.scalar_multiply(nw.scalar_sub2(1.0, rim), 0.85)
    # The joint is already soil. Extra dirt stains the stone edge, and only
    # darkens the bed a little, so the crack stays the dark line.
    dirt_mask = nw.scalar_max2(edge_band, nw.scalar_multiply(crevice, 0.35))
    dirt_color = hsv2rgba(
        float(detail.uniform(0.05, 0.11)),
        float(detail.uniform(0.25, 0.50)),
        float(detail.uniform(0.04, 0.10)),
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [1.0, color, nw.scalar_sub2(1.0, nw.scalar_multiply(dirt_amt * 0.90, dirt_mask))],
        attrs={"blend_type": "MULTIPLY"},
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [nw.scalar_multiply(dirt_amt * 0.80, dirt_mask), color, dirt_color],
        attrs={"blend_type": "MIX"},
    )
    moss_amt = _amount(moss, detail, 0.0, 0.35)
    moss_n = nw.new_node(
        Nodes.NoiseTexture, [uv], input_kwargs={"Scale": float(detail.uniform(12, 28)), "Detail": 4}
    )
    moss_mask = _math(
        nw, "MULTIPLY", nw.scalar_max2(crevice, nw.scalar_multiply(edge_band, 0.8)),
        _ramp(nw, moss_n.outputs["Fac"], 0.48, 0.70),
        clamp=True,
    )
    moss_color = hsv2rgba(
        float(detail.uniform(0.22, 0.38)),
        float(detail.uniform(0.25, 0.55)),
        float(detail.uniform(0.08, 0.18)),
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [nw.scalar_multiply(moss_amt, moss_mask), color, moss_color],
        attrs={"blend_type": "MIX"},
    )

    face_rough = nw.build_float_curve(
        nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": float(detail.uniform(30, 70))}),
        [(0, 0.72), (1, 0.92)],
    )
    roughness = nw.scalar_add(
        nw.scalar_multiply(joint, face_rough),
        nw.scalar_multiply(crevice, 0.96),
    )

    # Real relief is the joint step, 1–3 cm, plus under a millimetre of face
    # grain. Crack injection adds its groove onto this height.
    lumps = nw.new_node(
        Nodes.NoiseTexture, [uv], input_kwargs={"Scale": (1.0 / size) * lump_freq, "Detail": 6}
    )
    tilt = nw.scalar_add(0.82, nw.scalar_multiply(tone, 0.28))
    plateau = base + (1.0 - base) * 0.90
    height = nw.scalar_add(
        nw.scalar_multiply(nw.scalar_multiply(rim, tilt), plateau * stone_h),
        nw.scalar_multiply(_math(nw, "MULTIPLY", lumps.outputs["Fac"], rim), lump_amp * 0.08),
    )
    pits = nw.new_node(
        Nodes.NoiseTexture, [uv], input_kwargs={"Scale": float(detail.uniform(40, 90)), "Detail": 5}
    )
    bump_amt = _amount(bump, detail, 0.25, 1.0)
    normal = _bump(nw, rim, 0.0012, strength=0.45 * bump_amt)
    normal = _bump(nw, grit.outputs["Fac"], 0.0006, normal.outputs["Normal"], 0.65 * bump_amt)
    normal = _bump(nw, pits.outputs["Fac"], 0.0010, normal.outputs["Normal"], 0.40 * bump_amt)
    normal = _bump(nw, crevice, 0.0008, normal.outputs["Normal"], 0.35 * bump_amt)
    _finish(nw, color, height, uniform(0.010, 0.030), normal=normal, roughness=roughness)


def shader_rubble_stone(nw: NodeWrangler, **kwargs):
    """Angular, flat-faced rubble or coursed fieldstone (terrain stone, wall scale)."""
    size = uniform(0.07, 0.14)
    coursed = uniform() < 0.6  # stones elongated along rough courses
    uv = _warped_uv(nw, uniform(0.01, 0.03), uniform(2, 6),
                    stretch=(uniform(0.45, 0.75), 1.0) if coursed else (1.0, 1.0))
    cells = dict(input_kwargs={"Scale": 1.0 / size, "Randomness": uniform(0.75, 1.0)})
    edge = nw.new_node(Nodes.VoronoiTexture, [uv], **cells, attrs={"feature": "DISTANCE_TO_EDGE"})
    cell = nw.new_node(Nodes.VoronoiTexture, [uv], **cells)
    dry = uniform() < 0.35
    joint_w = uniform(0.02, 0.045) if dry else uniform(0.025, 0.055)
    # flat face with a short chipped bevel; dry joints are deep and dark
    face = _ramp(nw, edge.outputs["Distance"], joint_w * 0.4, joint_w + uniform(0.012, 0.030), "SMOOTHSTEP")
    chips = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(25, 70), "Detail": 5,
                                                                "Roughness": 0.75, "Distortion": uniform(0.3, 1.2)})
    tilt = nw.new_node(Nodes.SeparateColor, [cell.outputs["Color"]]).outputs[1]  # per-stone face offset
    dark, light = _stone_palette()
    stone = nw.new_node(Nodes.MixRGB, [nw.new_node(Nodes.SeparateColor, [cell.outputs["Color"]]).outputs[0], dark, light])
    grain = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(200, 500), "Detail": 6})
    stone = nw.new_node(Nodes.MixRGB, [uniform(0.08, 0.20), stone, grain.outputs["Color"]], attrs={"blend_type": "OVERLAY"})
    lichen = _ramp(nw, nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(4, 12)}).outputs["Fac"],
                   uniform(0.62, 0.75), 0.85)
    stone = nw.new_node(Nodes.MixRGB, [nw.scalar_multiply(lichen, uniform(0.0, 0.5)), stone,
                                       hsv2rgba(uniform(0.12, 0.25), uniform(0.2, 0.4), uniform(0.3, 0.5))])
    joint = _ramp(nw, edge.outputs["Distance"], joint_w * 0.5, joint_w)
    # Dry mortar is nearly black and read as an empty hole. Soil keeps the
    # joint a different material from the stone face.
    mortar = _mortar(dry)
    soil = hsv2rgba(0.07, 0.32, 0.18 if dry else 0.24)
    bed = nw.new_node(Nodes.MixRGB, [0.80, mortar, soil])
    color = nw.new_node(Nodes.MixRGB, [joint, bed, stone])
    # The chip noise used to displace the face by several millimetres and
    # came out as fingerprints. The joint step stays; the chips are shading.
    chip_amp = uniform(0.15, 0.35)
    height = nw.scalar_multiply(face, nw.scalar_add(0.75, nw.scalar_multiply(tilt, 0.25)))
    height = nw.scalar_add(
        height, nw.scalar_multiply(_math(nw, "MULTIPLY", chips.outputs["Fac"], face), chip_amp * 0.10)
    )
    normal = _bump(nw, chips.outputs["Fac"], 0.0010, strength=0.45)
    normal = _bump(nw, face, 0.0008, normal.outputs["Normal"], 0.30)
    _finish(nw, color, height, uniform(0.012, 0.035) if dry else uniform(0.008, 0.022), normal=normal)


def shader_sandstone_ashlar(nw: NodeWrangler, dirt=None, speckle=None, **kwargs):
    """Bedded sandstone blocks with weathered arrises and erosion pits (terrain sandstone, wall scale)."""
    uv_map = nw.new_node(Nodes.UVMap)
    uv = _warped_uv(nw, uniform(0.003, 0.012), uniform(3, 8))
    row = uniform(0.18, 0.40)
    hue = uniform(0.04, 0.12)
    sat = uniform(0.20, 0.50)
    val = uniform(0.35, 0.72)
    a = hsv2rgba(hue, sat, val)
    b = hsv2rgba(hue + uniform(-0.02, 0.02), sat * uniform(0.7, 1.2), min(val * uniform(0.75, 1.25), 0.85))
    mortar = hsv2rgba(hue, sat * 0.4, min(val * uniform(0.8, 1.3), 0.9))
    bricks = nw.new_node(
        Nodes.BrickTexture, [uv, a, b, mortar],
        input_kwargs={"Scale": 1, "Row Height": row, "Brick Width": row * uniform(1.4, 3.0),
                      "Mortar Size": row * uniform(0.02, 0.05), "Mortar Smooth": uniform(0.3, 0.8),
                      "Bias": uniform(-0.4, 0.4)},
        attrs={"offset": uniform(0.35, 0.65), "offset_frequency": 2},
    )
    joint = bricks.outputs["Fac"]
    # bedding: thin horizontal laminae, gently wavy, varying the tone
    bed = nw.new_node(Nodes.WaveTexture, [uv_map],
                      input_kwargs={"Scale": uniform(3, 9), "Distortion": uniform(6, 14), "Detail": 6,
                                    "Detail Scale": uniform(1, 3)},
                      attrs={"wave_type": "BANDS", "bands_direction": "Y"})
    k = uniform(0.04, 0.12)  # bedding contrast: a faint tone change, not stripes
    bed = build_color_ramp(nw, bed.outputs["Fac"], [0.3, 0.7],
                           [(1 - k, 1 - k, 1 - k * 1.1, 1), (1 + k, 1 + k * 0.95, 1 + k * 0.9, 1)])
    color = nw.new_node(Nodes.MixRGB, [uniform(0.4, 0.8), bricks.outputs["Color"], bed], attrs={"blend_type": "MULTIPLY"})
    # erosion: irregular honeycomb-weathering blotches, not round holes
    pits = nw.new_node(Nodes.NoiseTexture, [uv_map], input_kwargs={"Scale": uniform(25, 60), "Detail": 3,
                                                                    "Roughness": 0.5, "Distortion": uniform(0.0, 0.4)})
    thr = uniform(0.28, 0.36)
    pit = _ramp(nw, pits.outputs["Fac"], thr, thr + 0.10)
    # weathering comes in patches (a damp course, one corner), not evenly everywhere
    patch = _ramp(nw, nw.new_node(Nodes.NoiseTexture, [uv_map], input_kwargs={"Scale": uniform(1.5, 4)}).outputs["Fac"],
                  uniform(0.45, 0.6), 0.75)
    pit = nw.scalar_sub2(1.0, _math(nw, "MULTIPLY", nw.scalar_sub2(1.0, pit), patch))
    color = nw.new_node(Nodes.MixRGB, [nw.scalar_multiply(nw.scalar_sub2(1.0, pit), uniform(0.15, 0.35)), color,
                                       hsv2rgba(hue, sat * 1.1, val * uniform(0.65, 0.8))])
    sand_scale = uniform(400, 900)
    sand_amt = uniform(0.1, 0.25)
    block_h = uniform(0.6, 0.9)
    pit_h = uniform(0.15, 0.35)
    sand = nw.new_node(Nodes.NoiseTexture, [uv_map], input_kwargs={"Scale": sand_scale, "Detail": 4})
    color = nw.new_node(Nodes.MixRGB, [sand_amt, color, sand.outputs["Color"]], attrs={"blend_type": "OVERLAY"})
    # Dirt and face grit, from none to a light amount. Not part of the height.
    detail = _detail_rng(row, hue, sat, val, k, thr, sand_scale, sand_amt, block_h, pit_h)
    speckle_amt = _amount(speckle, detail, 0.0, 0.35)
    grit = nw.new_node(
        Nodes.NoiseTexture, [uv_map], input_kwargs={"Scale": float(detail.uniform(160, 320)), "Detail": 6}
    )
    grit_shade = nw.scalar_add(
        nw.scalar_sub2(1.0, speckle_amt * 0.85),
        nw.scalar_multiply(grit.outputs["Fac"], speckle_amt * 1.20),
    )
    color = nw.new_node(Nodes.MixRGB, [1.0, color, grit_shade], attrs={"blend_type": "MULTIPLY"})
    dirt_amt = _amount(dirt, detail, 0.0, 0.45)
    stain = nw.new_node(
        Nodes.NoiseTexture, [uv_map], input_kwargs={"Scale": float(detail.uniform(5.0, 12.0)), "Detail": 4}
    )
    patches = _ramp(nw, stain.outputs["Fac"], 0.42, 0.68)
    dirt_mask = nw.scalar_max2(joint, patches)
    dirt_color = hsv2rgba(
        float(detail.uniform(0.05, 0.12)),
        float(detail.uniform(0.25, 0.50)),
        float(detail.uniform(0.06, 0.14)),
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [1.0, color, nw.scalar_sub2(1.0, nw.scalar_multiply(dirt_amt * 0.95, dirt_mask))],
        attrs={"blend_type": "MULTIPLY"},
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [nw.scalar_multiply(dirt_amt * 0.70, dirt_mask), color, dirt_color],
        attrs={"blend_type": "MIX"},
    )
    # weathered blocks: rounded arrises (soft mortar transition), pitted faces
    height = nw.scalar_add(nw.scalar_multiply(nw.scalar_sub2(1.0, joint), block_h),
                           nw.scalar_multiply(pit, pit_h))
    _finish(nw, color, height, uniform(0.006, 0.018), 0.75, 0.97)


class CobbleWall:
    shader = shader_cobble_wall
    # None draws per wall: dirt and speckle from 0 to a light amount, moss the
    # same, bump from a little shading to full. Pass a number in 0..1 to pin one.
    dirt = None
    speckle = None
    bump = None
    moss = None

    def generate(self, dirt=None, speckle=None, bump=None, moss=None):
        return surface.shaderfunc_to_material(
            shader_cobble_wall,
            dirt=self.dirt if dirt is None else dirt,
            speckle=self.speckle if speckle is None else speckle,
            bump=self.bump if bump is None else bump,
            moss=self.moss if moss is None else moss,
        )

    __call__ = generate


class RubbleStone:
    shader = shader_rubble_stone

    def generate(self):
        return surface.shaderfunc_to_material(shader_rubble_stone)

    __call__ = generate


class SandstoneAshlar:
    shader = shader_sandstone_ashlar
    # None draws per wall, from none to a light amount.
    dirt = None
    speckle = None

    def generate(self, dirt=None, speckle=None):
        return surface.shaderfunc_to_material(
            shader_sandstone_ashlar,
            dirt=self.dirt if dirt is None else dirt,
            speckle=self.speckle if speckle is None else speckle,
        )

    __call__ = generate
