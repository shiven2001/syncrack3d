"""A minimal room around the controlled wall.

The flat module on its own renders as a slab floating in sky. That is not a
harmless simplification: a pulled-back frame shows the wall's silhouette against
a blue gradient and a black void, which is not a photograph of anything, and the
close-ups lose the bounce light that gives a real interior its soft fill. A
detector trained on that learns an illumination model no room has.

This adds the smallest context that makes the wall read as indoors:

* a **floor**, always - it is what props stand on and where bounce comes from;
* a **ceiling**, usually - it closes the light off so the sky stops leaking in;
* **return walls** at one or both ends, sometimes - they give the corner and the
  falloff that a flat lit plane otherwise never shows.

None of these is an injection host. They are not passed in ``host_by_name``, so
``resolve_host`` cannot bind a defect to them, and they carry no defect material
so ``classify_plane`` ignores them.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import bpy

from syncrack3d.materials import make_solid_material

SHELL_COLLECTION = "CWG_SHELL"

# Interior finishes that are not the wall under test. Kept duller and more
# varied than the wall palette so the surface being inspected is not
# systematically the brightest thing in frame.
_FLOOR_COLORS = (
    (0.28, 0.22, 0.16, 1.0),
    (0.42, 0.36, 0.30, 1.0),
    (0.20, 0.20, 0.21, 1.0),
    (0.55, 0.50, 0.44, 1.0),
    (0.34, 0.31, 0.29, 1.0),
)
_CEILING_COLORS = (
    (0.86, 0.86, 0.85, 1.0),
    (0.78, 0.78, 0.77, 1.0),
    (0.91, 0.90, 0.88, 1.0),
)
_RETURN_COLORS = (
    (0.74, 0.74, 0.72, 1.0),
    (0.66, 0.67, 0.68, 1.0),
    (0.80, 0.78, 0.74, 1.0),
)


@dataclass(frozen=True)
class ShellPlan:
    depth_m: float
    has_ceiling: bool
    returns: tuple[str, ...]
    floor_color: tuple
    ceiling_color: tuple
    return_color: tuple

    def to_dict(self) -> dict:
        return asdict(self)


def sample_shell_plan(rng: random.Random) -> ShellPlan:
    returns = []
    if rng.random() < 0.55:
        returns.append(rng.choice(("left", "right")))
        if rng.random() < 0.30:
            returns.append("right" if returns[0] == "left" else "left")
    return ShellPlan(
        depth_m=rng.uniform(2.4, 5.0),
        has_ceiling=rng.random() < 0.8,
        returns=tuple(returns),
        floor_color=rng.choice(_FLOOR_COLORS),
        ceiling_color=rng.choice(_CEILING_COLORS),
        return_color=rng.choice(_RETURN_COLORS),
    )


def _collection(name: str):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _slab(name, dimensions, location, material, collection):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.name = f"{name}.mesh"
    obj.data.materials.append(material)
    obj["cwg_role"] = "shell"
    for old in tuple(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def build_room_shell(spec, plan: ShellPlan) -> list:
    """Floor, optional ceiling and optional return walls in front of the wall.

    The wall's front face is at y = 0 with its normal along -Y, so the room
    occupies negative y. Slabs are given real thickness rather than being
    single-sided planes: a zero-thickness floor lit from a grazing angle leaks
    light through its own backface and produces a bright seam along the wall
    base that looks exactly like a defect.
    """
    collection = _collection(SHELL_COLLECTION)
    thickness = 0.12
    depth = plan.depth_m
    width = spec.width_m
    height = spec.height_m
    built = []

    built.append(
        _slab(
            "CWG_Shell_Floor",
            (width, depth, thickness),
            (0.0, -depth * 0.5, -thickness * 0.5),
            make_solid_material("CWG_Shell_Floor_Mat", plan.floor_color, roughness=0.75),
            collection,
        )
    )

    if plan.has_ceiling:
        built.append(
            _slab(
                "CWG_Shell_Ceiling",
                (width, depth, thickness),
                (0.0, -depth * 0.5, height + thickness * 0.5),
                make_solid_material(
                    "CWG_Shell_Ceiling_Mat", plan.ceiling_color, roughness=0.85
                ),
                collection,
            )
        )

    for side in plan.returns:
        sign = -1.0 if side == "left" else 1.0
        built.append(
            _slab(
                f"CWG_Shell_Return_{side}",
                (thickness, depth, height),
                (sign * (width * 0.5 + thickness * 0.5), -depth * 0.5, height * 0.5),
                make_solid_material(
                    f"CWG_Shell_Return_{side}_Mat", plan.return_color, roughness=0.8
                ),
                collection,
            )
        )
    return built


# --------------------------------------------------------------------------- matched finishes
# --shell-finish matched. The plain shell above gives every room grey side walls
# beside a coloured or tiled test wall, so the one textured surface in frame is
# always the one with cracks. Here the side walls usually carry the test wall's
# own finish (a crack-free copy made before injection), otherwise a different
# paint; the floor gets an Infinigen floor material. The ceiling stays plain.
SHELL_FINISHES = ("plain", "matched")
MATCH_PROBABILITY = 0.7
FLOOR_GENERATORS = (
    ("wood", "HardwoodFloor"), ("wood", "SquareWoodTile"), ("wood", "StaggeredWoodTile"),
    ("wood", "CrossedWoodTile"), ("wood", "HexagonWoodTile"), ("wood", "NonWoodTile"),
    ("ceramic", "Concrete"),
)


def _face_frame(obj, angle: float) -> None:
    """Put a return wall in the test wall's local frame, rotated about Z.

    Mesh data is counter-rotated, so the world geometry is unchanged; the inner
    face then lies in local XZ like the test wall's, and object-coordinate
    textures and metric UVs (u = local x, v = z) map onto it the same way.
    """
    from mathutils import Matrix

    obj.data.transform(Matrix.Rotation(-angle, 4, "Z"))
    obj.rotation_euler = (0.0, 0.0, angle)


def _metric_uv_xy(obj) -> None:
    mesh = obj.data
    uv = mesh.uv_layers.active or mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv = (co.x, co.y)


def apply_matched_finish(built, wall, seed: int) -> dict:
    """Give the side walls and floor real finishes. Returns the record.

    Draws from its own streams and restores numpy's global state and the
    test wall's paint parameters afterwards, so the wall, cracks, cameras and
    lights come out exactly as with the plain shell.
    """
    import importlib
    import math

    import gin
    import numpy as np

    from cwg_infinigen.core import surface
    from cwg_infinigen.core.util.math import int_hash
    from syncrack3d.material_diversity import PAINT_PARAMS, _metric_uv, apply_diverse_wall_material

    r = random.Random(int_hash((seed, "shell_finish")))
    np_state, py_state = np.random.get_state(), random.getstate()
    np.random.seed(int_hash((seed, "shell_finish_np")) % (2**32))
    record = {"finish": "matched"}
    try:
        returns = [o for o in built if o.name.startswith("CWG_Shell_Return_")]
        for obj in returns:
            _face_frame(obj, math.pi / 2 if obj.name.endswith("left") else -math.pi / 2)
            _metric_uv(obj)
        if returns:
            if r.random() < MATCH_PROBABILITY and wall.data.materials:
                material = wall.data.materials[0].copy()  # before injection: no cracks
                material.name = "CWG_Shell_Return_Matched"
                record["returns"] = {"source": "test_wall", "category": wall.get("cwg_material_category")}
            else:
                category = r.choice(("paint_neutral", "paint_colour", "paint_colour_mid"))
                apply_diverse_wall_material(returns[0], r.randrange(1, 2**31), forced=category)
                material = returns[0].data.materials[0]
                record["returns"] = {"source": "paint", "category": category}
            # Side walls are never photographed at crack distance: bump, no subdivision.
            material.displacement_method = "BUMP"
            for obj in returns:
                obj.data.materials.clear()
                obj.data.materials.append(material)
        floor = next(o for o in built if o.name == "CWG_Shell_Floor")
        module, cls = r.choice(FLOOR_GENERATORS)
        generator = getattr(importlib.import_module(f"cwg_infinigen.assets.materials.{module}"), cls)()
        floor_material = generator.generate()
        floor_material.displacement_method = "BUMP"
        _metric_uv_xy(floor)
        floor.data.materials.clear()
        surface.assign_material(floor, floor_material)
        record["floor"] = cls
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        category = wall.get("cwg_material_category")
        if category in PAINT_PARAMS:  # the side-wall paint rebinds these; give the wall's back
            with gin.unlock_config():
                for key, value in PAINT_PARAMS[category].items():
                    gin.bind_parameter(f"plaster_paint_params.{key}", value)
    return record
