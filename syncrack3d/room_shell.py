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
# always the one with cracks. Here the side walls always carry the test wall's
# own finish (a crack-free copy made before injection), as one room's walls
# do. Either shell gets Infinigen 2's floor and ceiling pickers. Every finished
# surface is displaced as geometry.
SHELL_FINISHES = ("plain", "matched")


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


def apply_matched_finish(built, wall, seed: int, dicing_rate: float = 2.0) -> dict:
    """Give the side walls the test wall's finish, and the floor and ceiling
    theirs (apply_floor_ceiling). Returns the record. All are displaced as geometry.

    Draws nothing from the shared random streams, so the wall, cracks, cameras
    and lights come out exactly as with the plain shell.
    """
    import math

    from syncrack3d.material_diversity import _metric_uv
    from syncrack3d.materials import enable_true_displacement

    record = {"finish": "matched"}
    returns = [o for o in built if o.name.startswith("CWG_Shell_Return_")]
    if returns and wall.data.materials:
        material = wall.data.materials[0].copy()  # before injection: no cracks
        material.name = "CWG_Shell_Return_Matched"
        for obj in returns:
            _face_frame(obj, math.pi / 2 if obj.name.endswith("left") else -math.pi / 2)
            _metric_uv(obj)
            obj.data.materials.clear()
            obj.data.materials.append(material)
            enable_true_displacement(obj, dicing_rate, modifier_name="return_adaptive")
        record["returns"] = {"source": "test_wall", "category": wall.get("cwg_material_category"),
                             "displacement": material.displacement_method}
    record.update(apply_floor_ceiling(built, seed, dicing_rate))
    return record


def apply_floor_ceiling(built, seed: int, dicing_rate: float = 2.0) -> dict:
    """Infinigen 2 floor and ceiling (when the shell has one). Returns their records."""
    record = {"floor": apply_floor_finish(next(o for o in built if o.name == "CWG_Shell_Floor"), seed, dicing_rate)}
    ceiling = apply_ceiling_finish(built, seed, dicing_rate)
    if ceiling is not None:
        record["ceiling"] = ceiling
    return record


def _apply_picker(obj, kind: str, seed: int, dicing_rate: float) -> dict:
    """Shade a horizontal slab with an Infinigen 2 picker and displace it as geometry.

    Horizontal UVs are metres in x and y. Adaptive subdivision turns the
    shader's Displacement socket into real relief.
    """
    from cwg_infinigen.core.util.math import int_hash
    from syncrack3d.infinigen2_bridge import realize_picker
    from syncrack3d.materials import enable_true_displacement

    material = realize_picker(kind, int_hash((int(seed), f"{kind}_picker")) % (2**31))
    obj.data.materials.clear()
    obj.data.materials.append(material)
    _metric_uv_xy(obj)
    enable_true_displacement(obj, dicing_rate, modifier_name=f"{kind}_adaptive")
    obj["cwg_material_generator"] = f"infinigen2:{kind}"
    obj["cwg_material_source"] = "infinigen2_procedural"
    return {
        "picker": f"{kind}_material_rand",
        "generator": f"infinigen2:{kind}",
        "displacement": material.displacement_method,
    }


def apply_floor_finish(floor, seed: int, dicing_rate: float = 2.0) -> dict:
    """Cover the floor with Infinigen's floor picker: wood planks, carpet,
    concrete or indoor ground tile 0.1-0.6 m across (by shape), sometimes with a scratch or
    splat layer."""
    return _apply_picker(floor, "floor", seed, dicing_rate)


def apply_ceiling_finish(built, seed: int, dicing_rate: float = 2.0) -> dict | None:
    """Cover the ceiling with Infinigen's ceiling picker: light paint, concrete,
    wood planks or wall tile. No-op when the shell has no ceiling; returns the
    record, or None."""
    ceiling = next((obj for obj in built if obj.name == "CWG_Shell_Ceiling"), None)
    if ceiling is None:
        return None
    return _apply_picker(ceiling, "ceiling", seed, dicing_rate)
