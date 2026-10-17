"""Opt-in multiscale paint variation, independent of placement/camera RNG.

These are hypotheses inspired by development references, not fitted material
measurements or validated transfer improvements. No real textures are copied.
"""

import random
from dataclasses import replace

PAINT_PROFILES = ("baseline", "smooth", "roller", "rough", "mixed")


def adapt_paint_material(material, seed, profile="baseline"):
    if profile not in PAINT_PROFILES:
        raise ValueError(f"Unknown paint profile: {profile}")
    if profile == "baseline" or material.finish != "paint":
        return material
    rng = random.Random(int(seed) ^ 0x5041494E54)
    if profile == "mixed":
        profile = rng.choices(("smooth", "roller", "rough"), (0.3, 0.45, 0.25))[0]
    ranges = {
        "smooth": (
            (280, 650),
            (0.00004, 0.00012),
            (3, 9),
            (0.00002, 0.00008),
            (0.015, 0.05),
        ),
        "roller": (
            (180, 450),
            (0.00010, 0.00030),
            (5, 14),
            (0.00010, 0.00040),
            (0.025, 0.08),
        ),
        "rough": (
            (90, 260),
            (0.00020, 0.00065),
            (12, 30),
            (0.00020, 0.00080),
            (0.035, 0.10),
        ),
    }
    fine, micro_depth, meso, meso_depth, color = ranges[profile]
    axes = (
        (rng.uniform(5, 12), 1.0, rng.uniform(0.35, 0.8))
        if profile == "roller"
        else (1.0, 1.0, 1.0)
    )
    return replace(
        material,
        paint_profile=profile,
        texture_scale=rng.uniform(*fine),
        micro_bump_distance_m=rng.uniform(*micro_depth),
        micro_bump_strength=rng.uniform(0.12, 0.30),
        meso_scale=rng.uniform(*meso),
        meso_bump_distance_m=rng.uniform(*meso_depth),
        meso_color_variation=rng.uniform(*color),
        roller_axis_scale=axes,
    )
