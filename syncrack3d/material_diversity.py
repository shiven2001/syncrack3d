"""CWGv5 opt-in wall material diversity (`--wall-materials diverse`).

The default Infinigen wall palette is ~90% near-neutral pre-occupancy plaster
(eggshell/greys/greige; even its "coloured" branch is pale orange). The CWG4
real-photo review found the transfer damage concentrated exactly where that
palette is silent: coloured and rough walls flooded with false positives,
grout and mortar joints read as cracks. This module samples a much wider set
of Infinigen surface materials per wall. Non-paint substrates are still mostly
plain walls, so grout, mortar and wood grain also appear with no crack.

Wood and irregular stone are never cracked, even with --clean-wall-scale 0:
cracks on them cost real-photo transfer (v1.3.0 host-crack study). Stone block
and large tile are plain at the default scale and cracked only in
--clean-wall-scale 0 batches, as in v1.3.0.

Every choice uses its own seed-derived RNG, so enabling this does not change
any other mode, and a scene's material is reproducible from its seed.

`--wall-materials unified` (v2, the `unified` recipe) is the palette the
released training data was actually built from: those batches forced one
category at a time in near-equal shares, so the weights above never shaped
them. It draws the 15 crack-able categories uniformly, leaves out wood and
irregular stone entirely, and uses one clean-wall table for every wall, in
which stone block and large tile join the other patterned masonry at 0.60.
`diverse` is kept unchanged so the v1.x releases rebuild exactly.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

# category -> (weight, finish note, generator choices [(module, class, weight)])
CATEGORIES = {
    "paint_neutral": (0.22, "paint", [("ceramic", "Plaster", 1.0)]),
    "paint_colour": (0.22, "paint", [("ceramic", "Plaster", 1.0)]),
    "concrete": (0.12, "concrete", [("ceramic", "Concrete", 1.0)]),
    "brick": (0.10, "masonry", [("ceramic", "Brick", 1.0)]),
    "tile": (0.14, "tile", [("wood", "NonWoodTile", 1.0)]),
    "marble": (0.10, "stone", [("ceramic", "Marble", 1.0), ("ceramic", "MarbleRegular", 0.5),
                               ("ceramic", "MarbleVoronoi", 0.5)]),
    "wood": (0.10, "timber", [("wood", "InteriorWood", 1.0), ("wood", "WhitePlywood", 0.4),
                              ("wood", "BlondePlywood", 0.4), ("wood", "StaggeredWoodTile", 0.5),
                              ("wood", "SquareWoodTile", 0.3), ("wood", "CrossedWoodTile", 0.3)]),
    # Weight 0 and listed last: only used when forced (--material-category), so
    # random category draws for every existing seed are unchanged.
    "paint_colour_mid": (0.0, "paint", [("ceramic", "Plaster", 1.0)]),
    # Grey split-face stone block cladding: mortar joints were every BD3
    # plain-wall false alarm. Weight 0, forced only.
    "stone_block": (0.0, "stone", [("ceramic", "StoneBlock", 1.0)]),
    # CWG9's residual false alarms: large plain wall tile grout, irregular stone
    # joints. Weight 0, forced only.
    "large_tile": (0.0, "tile", [("ceramic", "LargeTile", 1.0)]),
    "irregular_stone": (0.0, "stone", [("ceramic", "IrregularStone", 1.0)]),
    # Heavy-texture painted stucco (knockdown, sand-float, skip-trowel, popcorn,
    # heavy orange-peel): most BD3 crack walls are far rougher than CWG plaster.
    "stucco_neutral": (0.0, "paint", [("ceramic", "StuccoPlaster", 1.0)]),
    "stucco_colour": (0.0, "paint", [("ceramic", "StuccoPlaster", 1.0)]),
    "stucco_colour_mid": (0.0, "paint", [("ceramic", "StuccoPlaster", 1.0)]),
    # Exterior masonry (wall-scale ports of Infinigen's terrain cobblestone/stone/
    # sandstone, ceramic/exterior_stone.py): rounded cobbles in mortar, angular
    # coursed or dry rubble, bedded sandstone ashlar. Cracks only; forced only.
    "cobble_wall": (0.0, "stone", [("ceramic", "CobbleWall", 1.0)]),
    "rubble_stone": (0.0, "stone", [("ceramic", "RubbleStone", 1.0)]),
    "sandstone": (0.0, "stone", [("ceramic", "SandstoneAshlar", 1.0)]),
}
PAINTED = ("Plaster", "StuccoPlaster")  # generators that read plaster_paint_params

# Probability that a wall of this category carries NO defects (a negative wall).
CLEAN_FRACTION = {
    "paint_neutral": 0.15, "paint_colour": 0.15, "concrete": 0.35,
    "brick": 0.6, "tile": 0.6, "marble": 0.6, "wood": 1.0, "paint_colour_mid": 0.15,
    "stone_block": 1.0, "large_tile": 1.0, "irregular_stone": 1.0,
    "stucco_neutral": 0.15, "stucco_colour": 0.15, "stucco_colour_mid": 0.15,
    "cobble_wall": 0.6, "rubble_stone": 0.6, "sandstone": 0.6,
}
# Kept as plain walls even under --clean-wall-scale 0.
NEVER_CRACKED = ("wood", "irregular_stone")

# v2 (`unified`): drawn uniformly, no wood or irregular stone; one clean table.
UNIFIED_CATEGORIES = tuple(c for c in CATEGORIES if c not in NEVER_CRACKED)
UNIFIED_CLEAN_FRACTION = {**{c: CLEAN_FRACTION[c] for c in UNIFIED_CATEGORIES},
                          "stone_block": 0.6, "large_tile": 0.6}
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


def sample_category(seed: int, forced: str | None = None, palette: str = "diverse") -> str:
    if forced:
        if forced not in CATEGORIES:
            raise ValueError(f"unknown material category {forced}; choose from {sorted(CATEGORIES)}")
        return forced
    if palette == "unified":
        return _rng(seed, 0x554E4946).choice(UNIFIED_CATEGORIES)
    names = list(CATEGORIES)
    return _rng(seed, 0x4D41545F).choices(names, weights=[CATEGORIES[n][0] for n in names], k=1)[0]


def is_clean_wall(seed: int, category: str, scale: float = 1.0, palette: str = "diverse",
                  uniform: float | None = None) -> bool:
    """Whether this wall gets no cracks.

    `uniform` replaces the per-category table with one probability for every
    category (the clean-policy study). It compares against the same per-wall
    draw, so walls cracked under both policies are identical scenes.
    """
    table = UNIFIED_CLEAN_FRACTION if palette == "unified" and category in UNIFIED_CLEAN_FRACTION else CLEAN_FRACTION
    if category in NEVER_CRACKED:
        return True  # plain negatives only, at any scale
    if uniform is not None:
        return _rng(seed, 0x434C4E5F).random() < uniform
    if scale == 0:
        return False  # explicit "no clean walls": lets defects go on any texture
    if table[category] >= 1.0:
        return True  # a fraction of 1 keeps every wall of that category plain
    p = min(1.0, table[category] * scale)
    return _rng(seed, 0x434C4E5F).random() < p


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


def apply_diverse_wall_material(wall, seed: int, forced: str | None = None,
                                palette: str = "diverse") -> tuple[str, str, dict]:
    """Shade `wall` with a sampled material. Returns (generator, finish, record)."""
    import importlib

    import gin

    from cwg_infinigen.core import surface
    from cwg_infinigen.core.constraints.example_solver.room.decorate import _surface_finish

    category = sample_category(seed, forced, palette)
    _, _, choices = CATEGORIES[category]
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
