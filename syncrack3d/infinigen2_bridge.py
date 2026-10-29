"""Infinigen 2 shaders -> Blender materials, by surface role.

Walls (the inspection wall and the room's side walls) draw from the leaves of
Infinigen's wall picker; floors, ceilings and outdoor ground use Infinigen's
floor, ceiling and terrain pickers. Leaves that draw a crack or flake mask are
left out, so the only cracks in frame are the injected, labelled ones.

Realizes the full procfunc graph via ``Material.item()`` for every seed.
Shaders and ``procfunc`` are vendored under ``syncrack3d/vendor``; runtime does
not import a pip-installed ``infinigen`` package. The graph is rebuilt per
seed, so color, scale, and sometimes topology change with the seed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
import numpy as np

VENDOR = Path(__file__).resolve().parent / "vendor"

# The leaves of Infinigen 2's wall picker (``wall_material_rand``), plus masonry,
# cobble and our stucco.
# (finish for crack-coating policy, module, function, extra kwargs).
# Left out: ``paint_flaked`` (a ``cracks_rand`` mask) and ``wall_flaked_tile``
# (a ``flake_mask_rand`` mask). A detector reads their cracked and flaked coats
# as damage. ``bricks_paint`` cracks its coat half the time; a constant mask of
# 1 keeps the whole coat, which is its own no-mask branch.
_FL = "infinigen2.shaders.functionality_lists"
_WALL_SPECS = {
    "paint_wall": ("paint", _FL, "paint_wall_rand", {}),
    "paint_patterned": ("paint", _FL, "paint_patterned_rand", {}),
    "paint_art": ("paint", _FL, "_art_patterned_paint_rand", {}),
    "wall_wood_planks": ("wood", "infinigen2.shaders.composites.wood_planks", "wood_planks_rand", {}),
    "wall_concrete": ("concrete", "infinigen2.shaders.base_materials.concrete", "concrete_rand", {}),
    "wall_stone": ("stone", "infinigen2.shaders.base_materials.stone_smooth", "stone_smooth_rand", {}),
    "wall_gravel_concrete": ("concrete", "infinigen2.shaders.base_materials.gravel_concrete", "gravel_concrete_rand", {}),
    "wall_granite": ("stone", "infinigen2.shaders.base_materials.granite", "granite_rand", {}),
    "wall_bricks": ("masonry", "infinigen2.shaders.composites.bricks", "bricks_rand", {}),
    "bricks_paint": ("paint", "infinigen2.shaders.composites.bricks", "bricks_paint_rand", {"mask": 1.0}),
    "wall_bricks_pristine": ("masonry", "infinigen2.shaders.composites.bricks", "bricks_pristine_rand", {}),
    "tile_indoor_wall": ("tile", "infinigen2.shaders.composites.tiles", "tile_indoor_wall_rand", {}),
    # Not in Infinigen's wall picker; added for the stone-block and cobble walls
    # the Infinigen 1 palette had.
    "bricks_masonry": ("masonry", "infinigen2.shaders.composites.bricks", "bricks_masonry_rand", {}),
    "cobble_stone": ("stone", "infinigen2.shaders.base_materials.cobble_stone", "cobble_stone_rand", {}),
    # Painted stucco, ours in the Infinigen 2 style (materials_v2/stucco.py).
    "stucco_knockdown": ("paint", "syncrack3d.materials_v2.stucco", "stucco_rand", {"style": "knockdown"}),
    "stucco_sand_float": ("paint", "syncrack3d.materials_v2.stucco", "stucco_rand", {"style": "sand_float"}),
    "stucco_skip_trowel": ("paint", "syncrack3d.materials_v2.stucco", "stucco_rand", {"style": "skip_trowel"}),
    "stucco_popcorn": ("paint", "syncrack3d.materials_v2.stucco", "stucco_rand", {"style": "popcorn"}),
    "stucco_orange_peel": ("paint", "syncrack3d.materials_v2.stucco", "stucco_rand", {"style": "orange_peel"}),
}
INFINIGEN2_CATEGORIES = {name: spec[0] for name, spec in _WALL_SPECS.items()}

# wall_material_rand's UV rotation: horizontal 60%, vertical 30%, 45 degrees 10%.
_WALL_ROTATIONS = (
    (0.0, 3.0), (math.pi / 2, 0.75), (-math.pi / 2, 0.75), (math.pi / 4, 0.25), (-math.pi / 4, 0.25),
)
# Laid in courses: always horizontal. Brick, block and tile walls are not laid
# at 45 degrees, and vertical courses are rare.
_COURSED = frozenset({
    "wall_bricks", "bricks_paint", "wall_bricks_pristine", "bricks_masonry", "cobble_stone", "tile_indoor_wall",
})

# terrain_material_rand without ground_cracked and stone, whose crack network
# and dark wandering veins a detector would read as cracks, and without mud and
# ice, which render as standing water rather than ground in front of a
# building. The other leaves keep Infinigen's equal weights.
_TERRAIN_LEAVES = (
    ("dirt", "dirt_rand"), ("granite", "granite_rand"), ("gravel_concrete", "gravel_concrete_rand"),
    ("sand", "sand_rand"), ("sandstone", "sandstone_rand"), ("soil", "soil_rand"),
    ("stone_smooth", "stone_smooth_rand"),
)

# Indoor ground tile pattern unit in metres (mean, std, low, high). Infinigen
# draws clip_gaussian(0.25, 0.7, 0.05, 1.5), which on metre UVs often gave
# half-metre and larger tiles. Hexagon and diamond masks halve their input, so
# those tiles span twice the unit: 0.1-0.3 m keeps squares at 0.1-0.3 m and
# hexagons at 0.2-0.6 m.
_FLOOR_TILE_M = (0.2, 0.08, 0.1, 0.3)

# Infinigen's own role pickers. Their rotations and overlays (scratches and
# splats on floors) stay inside these functions; we do not re-roll them.
# The ceiling picker is Infinigen's; floor and terrain are the copies below.
_PICKERS = {
    "floor": (__name__, "_floor_material_rand"),
    "ceiling": (_FL, "ceiling_material_rand"),
    "terrain": (__name__, "_terrain_material_rand"),
}
_PICKER_SALT = {"floor": 0x00F10002, "ceiling": 0x0CE10003, "terrain": 0x7E220001}

# Prop part roles -> Infinigen 2 object pickers (occlusion.py picks the role
# from the v1 material's name). furniture is the copy below without cracked
# paint; the rest have no crack leaves.
_PROP_PICKERS = {
    "furniture": (__name__, "_furniture_material_rand"),
    "cabinet": (_FL, "cabinet_material_rand"),
    "table_top": (_FL, "table_top_material_rand"),
    "upholstery": (_FL, "fabric_sturdy_rand"),
    "fabric": (_FL, "fabric_general_rand"),
    "glass": (_FL, "glass_material_rand"),
    "decorative": (_FL, "decorative_material_rand"),
    "steel": (__name__, "_steel_material_rand"),
    "plastic": ("infinigen2.shaders.base_materials.plastic", "plastic_rand"),
    "ceramic": ("infinigen2.shaders.base_materials.ceramic", "ceramic_rand"),
}


def available() -> bool:
    return (VENDOR / "procfunc" / "__init__.py").is_file() and (
        VENDOR / "infinigen2" / "shaders" / "functionality_lists.py"
    ).is_file()


def use_vendored_infinigen2() -> None:
    """Import ``procfunc`` and ``infinigen2`` from ``syncrack3d/vendor``."""
    if not available():
        raise ImportError(f"vendored Infinigen 2 shaders missing under {VENDOR}")
    vendor = str(VENDOR)
    if vendor in sys.path:
        sys.path.remove(vendor)
    sys.path.insert(0, vendor)

    def _ours(module) -> bool:
        path = getattr(module, "__file__", None) or ""
        return f"{Path('syncrack3d') / 'vendor'}" in path or str(VENDOR) in path

    for name in list(sys.modules):
        if name in {"infinigen2", "procfunc"} or name.startswith(("infinigen2.", "procfunc.")):
            if not _ours(sys.modules[name]):
                del sys.modules[name]


def _vector_uv():
    import procfunc as pf

    return pf.nodes.shader.coord().uv


def _build(category: str, rng, vector):
    import importlib

    _finish, module, function, kwargs = _WALL_SPECS[category]
    return getattr(importlib.import_module(module), function)(rng, vector, **kwargs)


def _floor_tile_rand(rng, vector):
    import procfunc as pf
    from infinigen2.shaders.composites import tiles

    rng_size, rng_tile = rng.spawn(2)
    size = pf.random.clip_gaussian(rng_size, *_FLOOR_TILE_M)
    return tiles.tile_indoor_ground_rand(rng_tile, vector, scale=1.0 / size)


def _floor_material_rand(rng, vector):
    """``floor_material_rand`` with the tile edge from ``_FLOOR_TILE_M``.

    Same branches, weights and RNG spawns, so wood, carpet and concrete floors
    are the ones Infinigen's picker would draw.
    """
    import procfunc as pf
    from infinigen2.shaders import functionality_lists as fl
    from infinigen2.shaders.base_materials import carpet, concrete
    from infinigen2.shaders.composites import wood_planks

    rng_uv, rng_layerable, rng_choice, rng_func = rng.spawn(4)
    vector = fl.uv_maybe_rotate(rng_uv, vector)
    layerable = pf.control.choice(
        rng_layerable,
        [(concrete.concrete_rand, 1.0), (wood_planks.wood_planks_rand, 3.0), (carpet.carpet_rand, 2.0)],
    )
    func = pf.control.choice(
        rng_choice,
        [
            (lambda r, v: layerable(r, v), 2.0),
            (_floor_tile_rand, 2.7),
            (lambda r, v: fl._layer_rand(r, v, layerable(r, v)), 2.0),
        ],
    )
    return func(rng_func, vector)


def _furniture_material_rand(rng, vector):
    """``furniture_material_rand`` without ``paint_cracked_overlay_rand``: cracked
    paint on a chair or desk would be an unlabelled crack in frame."""
    import procfunc as pf
    from infinigen2.shaders import functionality_lists as fl
    from infinigen2.shaders.composites.scratches_overlay import scratches_overlay_rand
    from infinigen2.shaders.composites.splats_overlay import splats_overlay_rand

    rng_uv, rng_choice, rng_mat, rng_wear_choice, rng_wear = rng.spawn(5)
    vector = fl.uv_maybe_rotate_90(rng_uv, vector)
    material = fl._furniture_material_func_rand(rng_choice)(rng_mat, vector)
    wear = pf.control.choice(
        rng_wear_choice,
        [(lambda r, v, m: m, 3.0), (scratches_overlay_rand, 1.0), (splats_overlay_rand, 1.0)],
    )
    return wear(rng_wear, vector, material)


def _steel_material_rand(rng, vector):
    """``metal_brushed_linear_rand`` in near-grey: brushed stainless or chrome.

    Its own colour draw keeps to copper, brass and gold hues.
    """
    import procfunc as pf
    from infinigen2.shaders.base_materials import metal_brushed

    rng_color, rng_metal = rng.spawn(2)
    color = pf.color.hsv_color(
        hue=pf.random.uniform(rng_color, 0.0, 1.0),
        saturation=pf.random.uniform(rng_color, 0.0, 0.04),
        value=pf.random.clip_gaussian(rng_color, 0.55, 0.15, 0.3, 0.85),
    )
    return metal_brushed.metal_brushed_linear_rand(rng_metal, vector, base_color=color)


def _terrain_material_rand(rng, vector):
    """``terrain_material_rand`` minus ``ground_cracked``, ``stone``, ``mud`` and ``ice``."""
    import importlib

    import procfunc as pf

    rng_choice, rng_material = rng.spawn(2)
    leaves = [
        (getattr(importlib.import_module(f"infinigen2.shaders.base_materials.{module}"), function), 1.0)
        for module, function in _TERRAIN_LEAVES
    ]
    return pf.control.choice(rng_choice, leaves)(rng_material, vector)


def _assert_displacement_fidelity(material: bpy.types.Material) -> None:
    """Keep a linked Displacement socket so crack height can be added.

    Most of these shaders already displace. A few are flat on purpose
    (``granite_smooth``, and some terrazzo and leather variants) and leave the
    socket empty. A zero-height Displacement node fills that gap without
    changing a shader that already links one.
    """
    out = next(
        (n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"),
        None,
    )
    if out is None:
        raise RuntimeError(f"{material.name}: missing Material Output")
    if not out.inputs["Surface"].is_linked:
        raise RuntimeError(f"{material.name}: Material Output Surface is unlinked")
    if not out.inputs["Displacement"].is_linked:
        tree = material.node_tree
        disp = tree.nodes.new("ShaderNodeDisplacement")
        disp.inputs["Height"].default_value = 0.0
        disp.inputs["Midlevel"].default_value = 0.0
        disp.inputs["Scale"].default_value = 1.0
        tree.links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
    material.displacement_method = "DISPLACEMENT"


def _realize(module_name: str, function: str, rng, name: str) -> bpy.types.Material:
    import importlib

    pf_mat = getattr(importlib.import_module(module_name), function)(rng, _vector_uv())
    material = pf_mat.item().copy()
    material.name = name
    _assert_displacement_fidelity(material)
    return material


def realize_picker(kind: str, seed: int) -> bpy.types.Material:
    """Build a floor, ceiling or terrain material from Infinigen's own picker.

    ``floor`` is ``floor_material_rand`` (wood planks, carpet, concrete, indoor
    ground tile, sometimes with a scratch or splat layer) with floor-sized
    tiles. ``ceiling`` is ``ceiling_material_rand`` (light paint, concrete,
    wood planks, wall tile). ``terrain`` is ``terrain_material_rand`` minus the
    leaves noted at ``_TERRAIN_LEAVES``. The Displacement socket is kept and
    the method is true displacement, including a zero-height node when a
    variant is flat.
    """
    use_vendored_infinigen2()
    if kind not in _PICKERS:
        raise ValueError(f"unknown Infinigen 2 picker {kind!r}; choose from {sorted(_PICKERS)}")
    rng = np.random.default_rng((int(seed) * 1_000_003) ^ _PICKER_SALT[kind])
    return _realize(*_PICKERS[kind], rng, f"I2_{kind}_{int(seed):08d}")


def realize_prop_material(role: str, seed: int) -> bpy.types.Material:
    """Build a prop part's material from the Infinigen 2 object picker for ``role``.

    Bump, not true displacement: props are not subdivided, and their relief is
    too small to see at the distances they appear in frame.
    """
    use_vendored_infinigen2()
    if role not in _PROP_PICKERS:
        raise ValueError(f"unknown prop role {role!r}; choose from {sorted(_PROP_PICKERS)}")
    rng = np.random.default_rng((int(seed) * 1_000_003) ^ 0x50524F50)
    material = _realize(*_PROP_PICKERS[role], rng, f"I2_prop_{role}_{int(seed):08d}")
    material.displacement_method = "BUMP"
    material["cwg_prop_role"] = role
    return material


# Fixture finishes without a colour of their own choosing: (module, function).
_FIXTURE_LEAVES = {
    "wood": ("infinigen2.shaders.base_materials.wood_grain", "wood_grain_rand"),
    "painted_wood": (_FL, "wood_painted_rand"),
    "concrete": ("infinigen2.shaders.base_materials.concrete", "concrete_rand"),
    "stone": ("infinigen2.shaders.base_materials.stone_smooth", "stone_smooth_rand"),
}


def realize_fixture_material(kind: str, color, seed: int, name: str) -> bpy.types.Material:
    """A wall fixture's finish from Infinigen 2.

    ``plastic`` and ``metal`` take a fixed ``color`` (plastic or brushed metal),
    so fittings of one finish match; the seed only moves the grain. ``wood``,
    ``painted_wood``, ``concrete`` and ``stone`` draw their own colour.
    Object coordinates, since fixture boxes and curves have no metre UVs. Bump.
    """
    import importlib

    use_vendored_infinigen2()
    if kind not in ("plastic", "metal", *_FIXTURE_LEAVES):
        raise ValueError(f"unknown fixture finish {kind!r}; choose plastic, metal or {sorted(_FIXTURE_LEAVES)}")
    import procfunc as pf
    from infinigen2.shaders.base_materials import metal_brushed, plastic

    rng = np.random.default_rng((int(seed) * 1_000_003) ^ 0x46495854)
    vector = pf.nodes.shader.coord().object
    if kind in _FIXTURE_LEAVES:
        module, function = _FIXTURE_LEAVES[kind]
        pf_mat = getattr(importlib.import_module(module), function)(rng, vector)
    else:
        base = pf.Color(tuple(float(c) for c in color[:3]))
        if kind == "metal":
            pf_mat = metal_brushed.metal_brushed_linear_rand(rng, vector, base_color=base)
        else:
            pf_mat = plastic.plastic_rand(rng, vector, base_color=base)
    material = pf_mat.item().copy()
    material.name = name
    _assert_displacement_fidelity(material)
    material.displacement_method = "BUMP"
    return material


def realize_material(category: str, seed: int, overlay_fraction: float = 0.0) -> bpy.types.Material:
    """Build a Blender material from an Infinigen 2 wall category and wall seed.

    The UV rotation is the wall picker's, except coursed walls (``_COURSED``),
    which stay horizontal. With probability ``overlay_fraction``
    the leaf gets a splats overlay, the picker's layered branch. The picker can
    also add a scratches layer. Here it never does: thin scratch lines on the
    inspection wall look like hairline cracks.
    ``material["cwg_overlay"]`` records the overlay used.
    """
    use_vendored_infinigen2()
    if category not in _WALL_SPECS:
        raise ValueError(
            f"unknown Infinigen 2 category {category!r}; "
            f"choose from {sorted(_WALL_SPECS)}"
        )
    import procfunc as pf
    from infinigen2.shaders.composites.splats_overlay import splats_overlay_rand
    from infinigen2.shaders.functionality_lists import uv_maybe_rotate

    # Independent stream so other wall RNGs stay put.
    rng = np.random.default_rng((int(seed) * 1_000_003) ^ 0x49320001)
    rng_uv, rng_material, rng_layer = rng.spawn(3)
    rotation = pf.control.choice(rng_uv, list(_WALL_ROTATIONS))
    if category in _COURSED:
        rotation = 0.0  # drawn anyway, so the other draws do not move
    vector = uv_maybe_rotate(rng_uv, _vector_uv(), rotation_z=rotation)
    pf_mat = _build(category, rng_material, vector)
    overlay = "splats" if rng_layer.random() < overlay_fraction else "none"
    if overlay == "splats":
        pf_mat = splats_overlay_rand(rng_layer, vector, pf_mat)
    material = pf_mat.item()
    material = material.copy()
    material.name = f"I2_{category}_{int(seed):08d}"
    material["cwg_overlay"] = overlay
    _assert_displacement_fidelity(material)
    return material


def material_has_injection_albedo_hook(material: bpy.types.Material) -> bool:
    """True if crack albedo weathering can attach (top-level Principled)."""
    bsdf = next(
        (n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"),
        None,
    )
    return bsdf is not None and bsdf.inputs["Base Color"].is_linked
