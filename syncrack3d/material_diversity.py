"""Wall materials: 15 crack-able Infinigen surface categories, drawn uniformly.

Painted plaster and stucco in neutral, colour and mid-tone coats, concrete,
brick, wall tile, large tile, marble, stone block, cobble, rubble stone and
sandstone ashlar. Wood and irregular stone are not in the palette: cracks on
them hurt real-photo transfer. ``--material-categories`` restricts the draw to
a subset (material-diversity ablation).

A wall can stay crack-free (a clean negative) with a per-category probability,
``CLEAN_FRACTION`` x ``--clean-wall-scale``. The default scale is 0, so every
wall is cracked; crack-free views still come from context views.
``--clean-wall-uniform`` uses one probability for every category instead (the
same per-wall draw), for the clean-wall policy ablation.

Every choice uses its own seed-derived RNG, so a scene's material is
reproducible from its seed and does not move any other random stream.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

# category -> (finish note, generator choices [(module, class, weight)]). The order is
# the uniform draw's order: changing it changes every wall's material.
CATEGORIES = {
    "paint_neutral": ("paint", [("ceramic", "Plaster", 1.0)]),
    "paint_colour": ("paint", [("ceramic", "Plaster", 1.0)]),
    "concrete": ("concrete", [("ceramic", "Concrete", 1.0)]),
    "brick": ("masonry", [("ceramic", "Brick", 1.0)]),
    "tile": ("tile", [("wood", "NonWoodTile", 1.0)]),
    "marble": ("stone", [("ceramic", "Marble", 1.0), ("ceramic", "MarbleRegular", 0.5),
                         ("ceramic", "MarbleVoronoi", 0.5)]),
    # Mid-tone coloured coats (tan, ochre, blue, green...).
    "paint_colour_mid": ("paint", [("ceramic", "Plaster", 1.0)]),
    # Grey split-face stone block cladding; its mortar joints are a common false alarm.
    "stone_block": ("stone", [("ceramic", "StoneBlock", 1.0)]),
    # Large plain wall tile with grout lines.
    "large_tile": ("tile", [("ceramic", "LargeTile", 1.0)]),
    # Heavy-texture painted stucco (knockdown, sand-float, skip-trowel, popcorn,
    # heavy orange-peel): real crack walls are often far rougher than smooth plaster.
    "stucco_neutral": ("paint", [("ceramic", "StuccoPlaster", 1.0)]),
    "stucco_colour": ("paint", [("ceramic", "StuccoPlaster", 1.0)]),
    "stucco_colour_mid": ("paint", [("ceramic", "StuccoPlaster", 1.0)]),
    # Exterior masonry (wall-scale ports of Infinigen's terrain cobblestone/stone/
    # sandstone, ceramic/exterior_stone.py): rounded cobbles in mortar, angular
    # coursed or dry rubble, bedded sandstone ashlar.
    "cobble_wall": ("stone", [("ceramic", "CobbleWall", 1.0)]),
    "rubble_stone": ("stone", [("ceramic", "RubbleStone", 1.0)]),
    "sandstone": ("stone", [("ceramic", "SandstoneAshlar", 1.0)]),
}
PAINTED = ("Plaster", "StuccoPlaster")  # generators that read plaster_paint_params

# Probability, at --clean-wall-scale 1, that a wall of this category carries no cracks.
CLEAN_FRACTION = {
    "paint_neutral": 0.15, "paint_colour": 0.15, "concrete": 0.35, "brick": 0.6, "tile": 0.6, "marble": 0.6,
    "paint_colour_mid": 0.15, "stone_block": 0.6, "large_tile": 0.6, "stucco_neutral": 0.15,
    "stucco_colour": 0.15, "stucco_colour_mid": 0.15, "cobble_wall": 0.6, "rubble_stone": 0.6, "sandstone": 0.6,
}
PAINTED_CATEGORIES = ("paint_neutral", "paint_colour", "paint_colour_mid",
                      "stucco_neutral", "stucco_colour", "stucco_colour_mid")

# Plaster paint parameters (gin: plaster_paint_params). Neutral = Infinigen's
# defaults; colour = any hue, up to medium saturation, light to mid-dark value.
PAINT_PARAMS = {
    "paint_neutral": dict(colored_prob=0.0, max_saturation=0.05, value_min=0.74, value_max=0.95,
                          hue_min=0.04, hue_max=0.14, sat_min=0.03),
    "paint_colour": dict(colored_prob=1.0, max_saturation=0.45, value_min=0.35, value_max=0.95,
                         hue_min=0.0, hue_max=1.0, sat_min=0.15),
    # Mid-tone coloured coats (tan, ochre, blue, green...): what peels to a
    # lighter white/cream plaster in the real BD3 photos.
    "paint_colour_mid": dict(colored_prob=1.0, max_saturation=0.45, value_min=0.30, value_max=0.78,
                             hue_min=0.0, hue_max=1.0, sat_min=0.15),
    "stucco_neutral": dict(colored_prob=0.0, max_saturation=0.05, value_min=0.74, value_max=0.95,
                           hue_min=0.04, hue_max=0.14, sat_min=0.03),
    "stucco_colour_mid": dict(colored_prob=1.0, max_saturation=0.45, value_min=0.25, value_max=0.65,
                              hue_min=0.0, hue_max=1.0, sat_min=0.15),
    "stucco_colour": dict(colored_prob=1.0, max_saturation=0.45, value_min=0.30, value_max=0.90,
                          hue_min=0.0, hue_max=1.0, sat_min=0.15),
}


def _rng(seed: int, salt: int) -> random.Random:
    return random.Random((int(seed) * 1_000_003) ^ salt)


def sample_category(seed: int, forced: str | None = None, allowed=None) -> str:
    """A forced category, else a uniform draw from `allowed` (default: every category)."""
    for name in ([forced] if forced else []) + list(allowed or []):
        if name not in CATEGORIES:
            raise ValueError(f"unknown material category {name}; choose from {sorted(CATEGORIES)}")
    if forced:
        return forced
    return _rng(seed, 0x554E4946).choice(tuple(allowed) if allowed else tuple(CATEGORIES))


# How far a wall's displaced relief stands out of its plane, so fixtures and props can
# mount on it rather than sink into it. Measured on 10 rendered walls per category
# (99.99th percentile of the surface position, cracks included): (largest relief in metres,
# relief / displacement-node scale or None when the material bakes height into its texture).
# Both include a 10-15% margin. Paint and stucco relief is mostly crack coating edges.
_PLASTER, _STUCCO = (0.0048, 0.99), (0.0062, None)
RELIEF_M = {
    "paint_neutral": _PLASTER, "paint_colour": _PLASTER, "paint_colour_mid": _PLASTER,
    "stucco_neutral": _STUCCO, "stucco_colour": _STUCCO, "stucco_colour_mid": _STUCCO,
    "concrete": (0.001, None), "tile": (0.001, None), "large_tile": (0.0031, None),
    "marble": (0.0086, None), "brick": (0.0135, None),
    "cobble_wall": (0.0339, 1.13), "rubble_stone": (0.0394, 1.12),
    "stone_block": (0.0273, 1.62), "sandstone": (0.0230, 1.34),
}


def wall_relief(wall, category: str) -> float:
    """Metres a wall's relief can stand out of its plane: its displacement scale times the
    category's ratio, capped at the category maximum."""
    cap, ratio = RELIEF_M[category]
    scales = [n.inputs["Scale"].default_value for slot in wall.material_slots
              if slot.material is not None and slot.material.use_nodes
              for n in slot.material.node_tree.nodes if n.type == "DISPLACEMENT"]
    return cap if ratio is None or not scales else min(cap, max(scales) * ratio)


def is_clean_wall(seed: int, category: str, scale: float = 0.0, uniform: float | None = None) -> bool:
    """Whether this wall gets no cracks.

    `uniform` replaces the per-category table with one probability for every
    category. It compares against the same per-wall draw, so walls cracked
    under both policies are identical scenes.
    """
    if uniform is not None:
        return _rng(seed, 0x434C4E5F).random() < uniform
    if scale == 0:
        return False
    return _rng(seed, 0x434C4E5F).random() < min(1.0, CLEAN_FRACTION[category] * scale)


# Patterned Infinigen shaders lay their pattern in the shader's XY plane (they
# were written for floors). The CWG wall face is the XZ plane, so tiles came out
# as endless vertical strips and brick lost its mortar courses. These categories
# get their coordinates rotated so the pattern lies on the wall face.
# Only floor-designed shaders are rotated; InteriorWood/plywood are made for
# upright surfaces (doors, shelves) and keep their own grain direction.
ORIENT_CLASSES = {"NonWoodTile", "StaggeredWoodTile", "SquareWoodTile", "CrossedWoodTile"}


def _metric_uv(wall) -> None:
    """UVs in metres on the wall face (u = x, v = z).

    UV-driven shaders (Brick: rows 0.07-0.12 units) assume Infinigen's metric
    room UVs; CWG's cube-cross layout squeezed the 5 m face into a quarter of
    UV space and made bricks ~1 m wide. Injection uses object coordinates, not
    UVs, so defect placement is unaffected.
    """
    mesh = wall.data
    uv = mesh.uv_layers.active or mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv = (co.x, co.z)


def _orient_pattern_to_wall(material) -> int:
    """Rotate every object/generated/position coordinate feed by -90 deg about X.

    (x, y, z) -> (x, z, -y): the shader's XY becomes the wall's XZ. Only this
    material's own tree (and private copies of its node groups) is touched;
    defect injection adds its own coordinate nodes later and is unaffected.
    """
    import math

    feeds = {"ShaderNodeTexCoord": ("Object", "Generated"), "ShaderNodeNewGeometry": ("Position",)}
    count = 0

    def fix(tree):
        nonlocal count
        for node in list(tree.nodes):
            if node.bl_idname == "ShaderNodeGroup" and node.node_tree is not None:
                node.node_tree = node.node_tree.copy()
                fix(node.node_tree)
                continue
            for out_name in feeds.get(node.bl_idname, ()):
                out = node.outputs.get(out_name)
                if out is None or not out.links:
                    continue
                targets = [link.to_socket for link in out.links]
                mapping = tree.nodes.new("ShaderNodeMapping")
                mapping.vector_type = "POINT"
                mapping.inputs["Rotation"].default_value = (-math.pi / 2, 0.0, 0.0)
                tree.links.new(out, mapping.inputs["Vector"])
                for sock in targets:
                    tree.links.new(mapping.outputs["Vector"], sock)
                count += 1

    fix(material.node_tree)
    return count


def apply_wall_material(wall, seed: int, forced: str | None = None, allowed=None) -> tuple[str, str, dict]:
    """Shade `wall` with a sampled material. Returns (generator, finish, record)."""
    import importlib

    import gin

    from cwg_infinigen.core import surface
    from cwg_infinigen.core.constraints.example_solver.room.decorate import _surface_finish

    category = sample_category(seed, forced, allowed)
    _, choices = CATEGORIES[category]
    mod, cls, _w = _rng(seed, 0x47454E5F).choices(choices, weights=[c[2] for c in choices], k=1)[0]
    generator = getattr(importlib.import_module(f"cwg_infinigen.assets.materials.{mod}"), cls)()
    if cls in PAINTED:
        import cwg_infinigen.assets.materials.ceramic.plaster  # noqa: F401  registers the configurable

        with gin.unlock_config():
            for key, value in PAINT_PARAMS[category].items():
                gin.bind_parameter(f"plaster_paint_params.{key}", value)
    # Infinigen materials draw through numpy's global RNG, which the driver has
    # already seeded from the scene seed.
    material = generator.generate()
    oriented = _orient_pattern_to_wall(material) if cls in ORIENT_CLASSES else 0
    _metric_uv(wall)
    wall.data.materials.clear()
    surface.assign_material(wall, material)
    finish = _surface_finish(generator)
    wall["surface_finish"] = finish
    wall["cwg_material_generator"] = cls
    wall["cwg_material_category"] = category
    record = dict(category=category, generator=cls, finish=finish, pattern_feeds_rotated=oriented,
                  paint_params=PAINT_PARAMS.get(category) if cls in PAINTED else None)
    return cls, finish, record


def virtual_targets(seed: int, spec, n: int = 4, size_range_m=(0.10, 0.45)) -> list:
    """Close-up targets on a defect-free wall, shaped like defect placements.

    `sample_camera_poses` frames placements; on a clean wall these stand in so
    negatives are close-ups of wall texture (as real clean photos are), not only
    distant context views. They are never injected and carry no labels.
    """
    r = _rng(seed, 0x56545247)
    margin = 0.22
    out = []
    for i in range(n):
        out.append(SimpleNamespace(
            index=1000 + i,
            u_m=r.uniform(-spec.width_m / 2 + margin, spec.width_m / 2 - margin),
            v_m=r.uniform(margin, spec.height_m - margin),
            size_m=r.uniform(*size_range_m),
            object_name=f"CWG_CleanTarget_{i:02d}",
        ))
    return out
