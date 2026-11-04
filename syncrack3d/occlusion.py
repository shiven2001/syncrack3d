"""Infinigen props in front of the wall.

Why a bare wall is not enough. A detector trained only on clean plaster with a
defect in the middle learns "anomaly on flat surface", and then fires on every
socket, skirting joint, pipe bracket and furniture edge in a real photo. The
apartment pipeline supplied those distractors for free; a flat-wall generator
has to put them back deliberately, or it trades one narrow distribution for a
narrower one.

Props do three jobs here, all of which matter more than looking furnished:

* **Occlusion.** A defect partly hidden behind an object is the normal case in a
  real inspection photo and never occurs in a clean synthetic close-up.
* **Cast shadow.** A hard shadow edge across plaster is the single most common
  false positive for a crack detector. The model needs to see them labelled
  background.
* **Scale and clutter cues.** Something of known size in frame is what stops the
  network keying on absolute pixel width, which is the cue that does not survive
  a change of camera.

Factories are resolved lazily and individually. ``infinigen.assets.objects``
cannot be imported as a package here - it pulls a circular chain through
table_decorations - so each prop names its own submodule and a failure to import
one is skipped rather than fatal.

A prop stands anywhere on the floor in front of the wall, from `WALL_GAP_M`
past the wall's displaced relief out to `MAX_STANDOFF_M`, turned any way, clear
of everything already there. It may hide at most `MAX_COVER` of any crack
token, seen face-on.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import bpy
import numpy as np

PROP_COLLECTION = "CWG_PROPS"

# (module, factory); every prop is placed by the same rule, _place_loose
_PROP_CATALOG = (
    ("cwg_infinigen.assets.objects.shelves.cell_shelf", "CellShelfFactory"),
    ("cwg_infinigen.assets.objects.shelves.large_shelf", "LargeShelfFactory"),
    ("cwg_infinigen.assets.objects.lamp", "FloorLampFactory"),
    ("cwg_infinigen.assets.objects.tables.dining_table", "TableDiningFactory"),
    ("cwg_infinigen.assets.objects.tableware.bottle", "BottleFactory"),
    ("cwg_infinigen.assets.objects.tableware.food_box", "FoodBoxFactory"),
)


@dataclass(frozen=True)
class PropPlacement:
    index: int
    factory: str
    factory_seed: int
    location: tuple[float, float, float]
    yaw_deg: float
    object_name: str

    def to_dict(self) -> dict:
        return asdict(self)


def _load(module_name: str, class_name: str):
    import importlib

    try:
        import bpy  # noqa: F401
        import bmesh  # noqa: F401

        return getattr(importlib.import_module(module_name), class_name)
    except Exception:
        return None


WALL_GAP_M = 0.02  # nearest a prop stands to the wall, beyond its displaced relief
# Furthest a loose prop's back stands from the wall: the cameras stand about 0.4-2 m out,
# so a prop further back is behind them or fills the view.
MAX_STANDOFF_M = 0.9
MAX_COVER = 0.5  # largest share of a crack token's face a loose prop may hide
U_TRIES = 40


def _world_bounds(obj):
    """World-space (min, max) of an object and its children after modifiers."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for part in (obj, *obj.children_recursive):
        if part.type not in ("MESH", "CURVE"):
            continue
        evaluated = part.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        if len(mesh.vertices):
            co = np.empty(len(mesh.vertices) * 3)
            mesh.vertices.foreach_get("co", co)
            m = np.array(evaluated.matrix_world)
            world = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
            lo, hi = np.minimum(lo, world.min(0)), np.maximum(hi, world.max(0))
        evaluated.to_mesh_clear()
    return None if not np.isfinite(lo).all() else (lo, hi)


def _overlap(a, b, gap: float) -> bool:
    return all(a[0][i] < b[1][i] + gap and b[0][i] < a[1][i] + gap for i in range(3))


def _delete(obj) -> None:
    for part in (*obj.children_recursive, obj):
        bpy.data.objects.remove(part, do_unlink=True)


def _shade_as_bump(root) -> None:
    """Prop relief stays a bump, the same rule as the side walls.

    A material that already feeds the displacement socket keeps that height and
    is set to bump, so it does not become real geometry. A flat shader gets a
    fine bump of its own.
    """
    seen = set()
    for part in (root, *getattr(root, "children_recursive", ())):
        data = getattr(part, "data", None)
        materials = getattr(data, "materials", None)
        if not materials:
            continue
        for mat in materials:
            if mat is None or mat in seen or not mat.use_nodes:
                continue
            seen.add(mat)
            mat.displacement_method = "BUMP"
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
            if output is not None and output.inputs["Displacement"].is_linked:
                continue
            bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None or bsdf.inputs["Normal"].is_linked:
                continue
            coord = nodes.new("ShaderNodeTexCoord")
            noise = nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = 160.0
            noise.inputs["Detail"].default_value = 6.0
            links.new(coord.outputs["Object"], noise.inputs["Vector"])
            bump = nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 0.3
            bump.inputs["Distance"].default_value = 0.00035
            links.new(noise.outputs["Fac"], bump.inputs["Height"])
            links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _collection(name: str):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _cover(box, defect) -> float:
    """Share of a crack token's face-on square that a prop's x-z extent hides."""
    half = defect.size_m * 0.5
    dx = min(box[1][0], defect.u_m + half) - max(box[0][0], defect.u_m - half)
    dz = min(box[1][2], defect.v_m + half) - max(box[0][2], defect.v_m - half)
    return max(0.0, dx) * max(0.0, dz) / max(1e-9, (2 * half) ** 2)


class _Room:
    """The free space in front of the wall and everything already in it."""

    def __init__(self, spec, floor_depth_m, ceiling_m, side_gap_m, obstacles, defects, relief_m=0.0):
        self.relief = relief_m  # displaced wall relief standing out of the plane
        self.half_w = spec.width_m * 0.5 - side_gap_m
        self.depth = math.inf if floor_depth_m is None else floor_depth_m - side_gap_m
        self.ceiling = (getattr(spec, "height_m", math.inf) if ceiling_m is None else ceiling_m) - side_gap_m
        self.gap = side_gap_m
        self.obstacles = list(obstacles)
        self.defects = list(defects)
        self.placed = []  # (lo, hi) of every prop placed so far

    def blockers(self, box):
        return [b for b in (*self.obstacles, *self.placed) if _overlap(box, b, self.gap)]

    def fits(self, box) -> bool:
        lo, hi = box
        return (lo[0] >= -self.half_w and hi[0] <= self.half_w and lo[1] >= -self.depth and lo[2] >= -1e-6
                and hi[2] <= self.ceiling
                and all(_cover(box, d) <= MAX_COVER for d in self.defects)
                and not self.blockers(box))


def _place_loose(obj, rng, room) -> bool:
    """Turn a spawned prop and move it to a clear spot on the floor by its measured bounds; False if none."""
    near = WALL_GAP_M + room.relief
    for _ in range(U_TRIES):
        # a new turn each try: a long prop at an angle often fits where it would not square on
        obj.rotation_euler = (0.0, 0.0, rng.uniform(0.0, 2.0 * math.pi))
        bpy.context.view_layer.update()
        bounds = _world_bounds(obj)
        if bounds is None:
            return False
        lo0, hi0 = bounds
        size = hi0 - lo0
        if size[0] > 2 * room.half_w:
            continue
        u = rng.uniform(-room.half_w + size[0] / 2, room.half_w - size[0] / 2)
        y = -rng.uniform(near, max(near, min(MAX_STANDOFF_M, room.depth - size[1])))
        # back face at y, centred on u, bottom on the floor
        shift = np.array([u - (lo0[0] + hi0[0]) / 2, y - hi0[1], -lo0[2]])
        lo, hi = lo0 + shift, hi0 + shift
        # Step out from the wall past whatever is in the way (skirting, a fixture,
        # another prop); a spot that would need more than MAX_STANDOFF_M is redrawn.
        for _ in range(len(room.obstacles) + len(room.placed) + 1):
            hits = room.blockers((lo, hi))
            if not hits:
                break
            step = min(b[0][1] for b in hits) - room.gap - hi[1]
            if step >= 0 or hi[1] + step < -MAX_STANDOFF_M:
                break
            lo, hi, shift = lo + [0, step, 0], hi + [0, step, 0], shift + [0, step, 0]
        if room.fits((lo, hi)):
            obj.location = tuple(float(a + d) for a, d in zip(obj.location, shift))
            bpy.context.view_layer.update()
            room.placed.append((lo, hi))
            return True
    return False


def sample_props(
    rng: random.Random,
    spec,
    placements,
    *,
    count_range: tuple[int, int] = (0, 3),
    floor_depth_m: float | None = None,
    side_gap_m: float = 0.02,
    obstacles=(),
    relief_m: float = 0.0,
) -> list[PropPlacement]:
    """Place props anywhere on the floor in front of the wall.

    Each prop is turned at random and set down by its measured bounds: on the
    floor, between the side walls, its back from `WALL_GAP_M` past `relief_m`
    out to `MAX_STANDOFF_M`. It may hide at most `MAX_COVER` of a crack token.
    `obstacles` are world boxes already in the room (cameras, lights, fixtures,
    skirting). Overlap steps the prop out from the wall; a prop that cannot fit
    is dropped. Partial occlusion of a crack is wanted. A prop that hid most of
    a token face-on would leave a close-up of furniture while the label still
    claimed the crack was visible.
    """
    count = rng.randint(*count_range)
    if count <= 0:
        return []

    collection = _collection(PROP_COLLECTION)
    results: list[PropPlacement] = []
    room = _Room(spec, floor_depth_m, None, side_gap_m, obstacles, placements, relief_m)

    for index in range(count):
        module_name, class_name = rng.choice(_PROP_CATALOG)
        factory_cls = _load(module_name, class_name)
        if factory_cls is None:
            continue
        factory_seed = rng.randrange(1, 2**31)
        try:
            obj = factory_cls(factory_seed).spawn_asset(i=index)
        except Exception:
            # One prop failing to build is not worth losing the scene over.
            continue
        obj.parent = None
        obj.name = f"CWG_Prop_{index:02d}_{class_name}"
        obj["cwg_role"] = "prop"
        _shade_as_bump(obj)
        for old in tuple(obj.users_collection):
            old.objects.unlink(obj)
        collection.objects.link(obj)
        if not _place_loose(obj, rng, room):
            _delete(obj)
            continue
        results.append(PropPlacement(
            index=index, factory=class_name, factory_seed=factory_seed,
            location=tuple(round(float(v), 5) for v in obj.location),
            yaw_deg=round(math.degrees(obj.rotation_euler.z), 2), object_name=obj.name))
    return results
