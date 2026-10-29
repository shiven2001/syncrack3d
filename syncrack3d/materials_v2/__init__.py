"""Procedural Infinigen 2 wall materials.

Each call rebuilds the vendored shader for that category and seed. Nothing is
read from a frozen ``.blend``.
"""

from __future__ import annotations

import bpy

from syncrack3d.infinigen2_bridge import INFINIGEN2_CATEGORIES, realize_material


def generate(category: str, seed: int, overlay_fraction: float = 0.0) -> bpy.types.Material:
    """Build a new wall material for ``category`` / ``seed``."""
    return realize_material(category, seed, overlay_fraction)


def apply_wall_material(wall, seed: int, category: str, overlay_fraction: float = 0.0) -> tuple[str, str, dict]:
    """Assign a freshly realized Infinigen 2 material onto a CWG wall."""
    from syncrack3d.infinigen2_bridge import material_has_injection_albedo_hook
    from syncrack3d.material_diversity import _metric_uv

    material = generate(category, seed, overlay_fraction)
    finish = INFINIGEN2_CATEGORIES[category]
    source = "infinigen2_procedural"
    _metric_uv(wall)
    wall.data.materials.clear()
    wall.data.materials.append(material)
    wall["surface_finish"] = finish
    wall["cwg_material_generator"] = f"infinigen2:{category}"
    wall["cwg_material_category"] = category
    wall["cwg_material_source"] = source
    record = dict(
        category=category,
        generator=f"infinigen2:{category}",
        finish=finish,
        pattern_feeds_rotated=0,
        paint_params=None,
        source=source,
        overlay=material.get("cwg_overlay", "none"),
        albedo_hook=material_has_injection_albedo_hook(material),
        displacement_method=material.displacement_method,
    )
    return f"infinigen2:{category}", finish, record
