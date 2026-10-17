"""Exterior masonry walls: cobble/fieldstone in mortar, angular rubble, sandstone ashlar.

Infinigen's terrain stones (terrain/cobble_stone.py, stone.py, sandstone.py) are
landscape-scale and displace through a geometry modifier, so they cannot carry
injected defects on a CWG wall. These are wall-scale shader versions with the
contract defect injection needs (StoneBlock's): metric UVs, a linked Base Color
and real shader displacement. CobbleWall ports the terrain cobblestone's
Voronoi-cell construction (rounded stones rising out of recessed joints, per-stone
tone from the cell id); RubbleStone and SandstoneAshlar follow the same pattern
for angular coursed rubble and bedded sandstone blocks. Variation over realism.
"""

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


def _finish(nw, color, height, disp_scale, rough_lo=0.65, rough_hi=0.95):
    displacement = nw.new_node(Nodes.Displacement,
                               input_kwargs={"Height": height, "Scale": disp_scale, "Midlevel": 0.0})
    roughness = nw.build_float_curve(
        nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": uniform(20, 60)}), [(0, rough_lo), (1, rough_hi)])
    bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={"Base Color": color, "Roughness": roughness})
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": bsdf, "Displacement": displacement})


def shader_cobble_wall(nw: NodeWrangler, **kwargs):
    """Rounded cobbles / fieldstones set in mortar (terrain cobblestone, wall scale)."""
    size = uniform(0.05, 0.10)  # stone size, metres
    uv = _warped_uv(nw, uniform(0.01, 0.04), uniform(3, 8), stretch=(uniform(0.7, 1.0), 1.0))
    rnd = uniform(0.6, 1.0)
    cells = dict(input_kwargs={"Scale": 1.0 / size, "Randomness": rnd})
    edge = nw.new_node(Nodes.VoronoiTexture, [uv], **cells, attrs={"feature": "DISTANCE_TO_EDGE"})
    cell = nw.new_node(Nodes.VoronoiTexture, [uv], **cells)
    joint_w = uniform(0.018, 0.040)  # mortar half-width in cell units
    # rounded cap: 1 at the stone's centre falling to its edge (F1 distance), cut
    # off where the mortar bed starts; the bed sits below every stone
    r = uniform(0.45, 0.65)
    cap = _math(nw, "SUBTRACT", 1.0, _math(nw, "POWER", _math(nw, "DIVIDE", cell.outputs["Distance"], r, clamp=True), 2.0),
                clamp=True)
    cap = _math(nw, "POWER", cap, uniform(0.5, 0.8))
    rim = _ramp(nw, edge.outputs["Distance"], joint_w * 0.6, joint_w + 0.012, "SMOOTHSTEP")
    # every stone stands on a base above the mortar bed, so its rim is a step
    # with its own shadow, not a slope down to the joint
    base = uniform(0.3, 0.5)
    dome = _math(nw, "MULTIPLY", nw.scalar_add(base, nw.scalar_multiply(cap, 1.0 - base)), rim)
    dark, light = _stone_palette()
    stone = nw.new_node(Nodes.MixRGB, [nw.new_node(Nodes.SeparateColor, [cell.outputs["Color"]]).outputs[0], dark, light])
    grain = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": uniform(150, 400), "Detail": 8})
    stone = nw.new_node(Nodes.MixRGB, [uniform(0.28, 0.50), stone, grain.outputs["Color"]], attrs={"blend_type": "OVERLAY"})
    mortar = _mortar(dry=uniform() < 0.25)
    mortar = tuple(c * uniform(0.55, 0.9) for c in mortar[:3]) + (1.0,)  # bed in shadow under the stones
    joint = _ramp(nw, edge.outputs["Distance"], joint_w * 0.6, joint_w)
    color = nw.new_node(Nodes.MixRGB, [joint, mortar, stone])
    lumps = nw.new_node(Nodes.NoiseTexture, [uv], input_kwargs={"Scale": 1.0 / size * uniform(2, 4), "Detail": 6})
    height = nw.scalar_add(nw.scalar_multiply(dome, uniform(0.7, 1.0)),
                           nw.scalar_multiply(_math(nw, "MULTIPLY", lumps.outputs["Fac"], dome), uniform(0.15, 0.35)))
    _finish(nw, color, height, uniform(0.010, 0.030))


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
    color = nw.new_node(Nodes.MixRGB, [joint, _mortar(dry), stone])
    height = nw.scalar_add(nw.scalar_multiply(face, nw.scalar_add(0.75, nw.scalar_multiply(tilt, 0.25))),
                           nw.scalar_multiply(_math(nw, "MULTIPLY", chips.outputs["Fac"], face), uniform(0.15, 0.35)))
    _finish(nw, color, height, uniform(0.012, 0.035) if dry else uniform(0.008, 0.022))


def shader_sandstone_ashlar(nw: NodeWrangler, **kwargs):
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
    sand = nw.new_node(Nodes.NoiseTexture, [uv_map], input_kwargs={"Scale": uniform(400, 900), "Detail": 4})
    color = nw.new_node(Nodes.MixRGB, [uniform(0.1, 0.25), color, sand.outputs["Color"]], attrs={"blend_type": "OVERLAY"})
    # weathered blocks: rounded arrises (soft mortar transition), pitted faces
    height = nw.scalar_add(nw.scalar_multiply(nw.scalar_sub2(1.0, joint), uniform(0.6, 0.9)),
                           nw.scalar_multiply(pit, uniform(0.15, 0.35)))
    _finish(nw, color, height, uniform(0.006, 0.018), 0.75, 0.97)


class CobbleWall:
    shader = shader_cobble_wall

    def generate(self):
        return surface.shaderfunc_to_material(shader_cobble_wall)

    __call__ = generate


class RubbleStone:
    shader = shader_rubble_stone

    def generate(self):
        return surface.shaderfunc_to_material(shader_rubble_stone)

    __call__ = generate


class SandstoneAshlar:
    shader = shader_sandstone_ashlar

    def generate(self):
        return surface.shaderfunc_to_material(shader_sandstone_ashlar)

    __call__ = generate
