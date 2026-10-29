"""Painted stucco in the Infinigen 2 style: heavy hand-applied texture, real displacement.

Infinigen 2 has no stucco; its paint relief is roller strokes, orange peel and
splatter of a few millimetres. These are the Infinigen 1 stucco finishes
(knockdown, sand float, skip trowel, popcorn, heavy orange peel) rebuilt the
way Infinigen 2's ``paint`` is: procfunc height layers on domain-warped noise
and voronoi, a long-wavelength undulation because hand-applied walls are never
flat, and one displacement with midlevel 0, so relief is geometry when the
wall is diced. The coat is a ``paint_color_rand`` colour, pooling darker in the
low spots, with peaks a little smoother where the trowel or roller passed.

Imported through the Infinigen 2 bridge after ``use_vendored_infinigen2``.
"""

from __future__ import annotations

import numpy as np
import procfunc as pf

from infinigen2.shaders.base_materials.paint import paint_color_rand
from infinigen2.shaders.util.coord import coord_warp

STYLES = ("knockdown", "sand_float", "skip_trowel", "popcorn", "orange_peel")


def _knockdown(rng, vector):
    """Splatter knocked flat by a trowel: wide plateaus, sharp rims, grit."""
    size = pf.random.uniform(rng, 0.02, 0.05)
    warped = coord_warp(vector=vector, size=size * 2.0, strength=pf.random.uniform(rng, 0.6, 1.2))
    blobs = pf.nodes.texture.noise(vector=warped.vector, scale=1.0 / size, detail=2.0, noise_dimensions="2D")
    # Islands of splat on a smooth base, with rounded shoulders: a lower cut or a
    # sharper edge leaves the base as dark, jagged pits that read as cracks.
    cut = pf.random.uniform(rng, 0.54, 0.6)
    plateau = pf.nodes.math.map_range(value=blobs.fac, from_min=cut, from_max=cut + 0.08,
                                      interpolation_type="SMOOTHSTEP")
    # The knocked-down tops are not perfectly flat.
    top = pf.nodes.texture.noise(vector=vector, scale=1.0 / (size * 0.4), detail=2.0, noise_dimensions="2D")
    grit = pf.nodes.texture.noise(vector=vector, scale=pf.random.uniform(rng, 350.0, 650.0), detail=1.0,
                                  noise_dimensions="2D")
    height = plateau * (0.8 + 0.2 * top.fac) + 0.12 * grit.fac
    return height, pf.random.uniform(rng, 0.0015, 0.005)


def _sand_float(rng, vector):
    """Sand grains in the coat, dragged into soft swirls by a float."""
    grain = pf.nodes.texture.voronoi(vector=vector, scale=pf.random.uniform(rng, 350.0, 700.0),
                                     randomness=1.0, voronoi_dimensions="2D")
    grains = pf.nodes.math.map_range(value=grain.distance, from_min=0.55, from_max=0.0,
                                     interpolation_type="SMOOTHSTEP")
    swirl_warp = coord_warp(vector=vector, size=pf.random.uniform(rng, 0.04, 0.09), strength=1.5)
    swirl = pf.nodes.texture.noise(vector=swirl_warp.vector, scale=pf.random.uniform(rng, 20.0, 45.0),
                                   detail=4.0, noise_dimensions="2D")
    height = 0.7 * grains + 0.3 * pf.nodes.math.map_range(value=swirl.fac, from_min=0.3, from_max=0.7)
    return height, pf.random.uniform(rng, 0.0008, 0.002)


def _skip_trowel(rng, vector):
    """Islands of plaster left by a skipping trowel, with drag streaks in one direction."""
    size = pf.random.uniform(rng, 0.07, 0.15)
    warped = coord_warp(vector=vector, size=size, strength=pf.random.uniform(rng, 1.0, 2.0))
    patches = pf.nodes.texture.noise(vector=warped.vector, scale=1.0 / size, detail=4.0, noise_dimensions="2D")
    cut = pf.random.uniform(rng, 0.47, 0.58)
    islands = pf.nodes.math.map_range(value=patches.fac, from_min=cut, from_max=cut + 0.1,
                                      interpolation_type="SMOOTHSTEP")
    stroke = pf.nodes.math.vector_rotate_axis_angle(vector, angle=pf.random.uniform(rng, 0.0, 3.1416))
    streak = pf.nodes.texture.noise(vector=stroke * (1.0, pf.random.uniform(rng, 4.0, 8.0), 1.0),
                                    scale=pf.random.uniform(rng, 25.0, 55.0), detail=3.0, noise_dimensions="2D")
    height = islands * (0.75 + 0.25 * streak.fac) + 0.08 * streak.fac
    return height, pf.random.uniform(rng, 0.0012, 0.0035)


def _popcorn(rng, vector):
    """Clustered rounded nodules: two sizes of domed cells."""
    warped = coord_warp(vector=vector, size=0.01, strength=0.6)
    big = pf.nodes.texture.voronoi(vector=warped.vector, scale=pf.random.uniform(rng, 120.0, 260.0),
                                   randomness=1.0, voronoi_dimensions="2D")
    small = pf.nodes.texture.voronoi(vector=vector, scale=pf.random.uniform(rng, 380.0, 700.0),
                                     randomness=1.0, voronoi_dimensions="2D")
    domes = pf.nodes.math.float_curve(
        factor=1.0, value=pf.nodes.math.map_range(value=big.distance, from_min=0.62, from_max=0.0),
        curve=np.array([[0.0, 0.0], [0.35, 0.62], [1.0, 1.0]]),
    )
    beads = pf.nodes.math.map_range(value=small.distance, from_min=0.6, from_max=0.0)
    height = 0.75 * domes + 0.25 * beads
    return height, pf.random.uniform(rng, 0.0025, 0.006)


def _orange_peel(rng, vector):
    """Heavy roller orange peel: dense soft dimples."""
    warped = coord_warp(vector=vector, size=0.01, strength=0.5)
    peel = pf.nodes.texture.noise(vector=warped.vector, scale=pf.random.uniform(rng, 120.0, 300.0),
                                  detail=2.0, noise_dimensions="2D")
    height = pf.nodes.math.map_range(value=peel.fac, from_min=0.32, from_max=0.68,
                                     interpolation_type="SMOOTHSTEP")
    return height, pf.random.uniform(rng, 0.0006, 0.002)


_TEXTURES = {
    "knockdown": _knockdown, "sand_float": _sand_float, "skip_trowel": _skip_trowel,
    "popcorn": _popcorn, "orange_peel": _orange_peel,
}


def stucco_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector], style: str | None = None,
                base_color: pf.Color | None = None) -> pf.Material:
    """One painted stucco wall. ``style`` is one of ``STYLES`` (drawn when None)."""
    rng_style, rng_texture, rng_color, rng_finish = rng.spawn(4)
    if style is None:
        style = pf.control.choice(rng_style, [(name, 1.0) for name in STYLES])
    texture, depth = _TEXTURES[style](rng_texture, vector)
    texture = pf.nodes.math.clamp(texture)

    # Hand-applied coats wander a millimetre or two over tens of centimetres.
    swell = pf.nodes.texture.noise(vector=vector, scale=1.0 / pf.random.uniform(rng_finish, 0.4, 1.2),
                                   detail=2.0, noise_dimensions="2D")
    undulation = swell.fac * pf.random.uniform(rng_finish, 0.0005, 0.002)

    if base_color is None:
        base_color = paint_color_rand(rng_color, saturation_power=0.9)  # as paint_wall_rand
    mottle = pf.nodes.texture.noise(vector=vector, scale=1.0 / pf.random.uniform(rng_finish, 0.1, 0.35),
                                    detail=5.0, noise_dimensions="2D")
    color = pf.nodes.color.mix_rgb(blend_type="MULTIPLY", factor=mottle.fac * pf.random.uniform(rng_finish, 0.05, 0.14),
                                   a=base_color, b=pf.Color((0.5, 0.5, 0.5)))
    # Paint pools a little darker in the hollows of the texture.
    color = pf.nodes.color.mix_rgb(blend_type="MULTIPLY",
                                   factor=(1.0 - texture) * pf.random.uniform(rng_finish, 0.08, 0.22),
                                   a=color, b=pf.Color((0.55, 0.55, 0.55)))
    rough_lo = pf.random.uniform(rng_finish, 0.55, 0.72)
    roughness = pf.nodes.math.map_range(value=texture, to_min=rough_lo + 0.18, to_max=rough_lo)
    surface = pf.nodes.shader.principled_bsdf(
        base_color=color,
        roughness=roughness,
        specular_ior_level=pf.random.uniform(rng_finish, 0.2, 0.4),
        ior=1.45,
        subsurface_anisotropy=0.0,
    )
    displacement = pf.nodes.shader.displacement(height=texture * depth + undulation, midlevel=0.0)
    return pf.Material(surface=surface, displacement=displacement)
