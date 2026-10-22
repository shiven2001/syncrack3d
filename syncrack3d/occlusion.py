"""Infinigen props in front of the controlled wall.

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
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import bpy
import numpy as np

PROP_COLLECTION = "CWG_PROPS"

# (module, factory, footprint hint in metres, stands-on-floor)
_PROP_CATALOG = (
    ("cwg_infinigen.assets.objects.shelves.cell_shelf", "CellShelfFactory", 0.9, True),
    ("cwg_infinigen.assets.objects.shelves.large_shelf", "LargeShelfFactory", 1.1, True),
    ("cwg_infinigen.assets.objects.lamp", "FloorLampFactory", 0.4, True),
    ("cwg_infinigen.assets.objects.tables.table_top", "TableTopFactory", 1.0, True),
    ("cwg_infinigen.assets.objects.elements.rug", "RugFactory", 1.6, True),
    ("cwg_infinigen.assets.objects.tableware.bottle", "BottleFactory", 0.12, False),
    ("cwg_infinigen.assets.objects.tableware.food_box", "FoodBoxFactory", 0.2, False),
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


FLOOR_COVERINGS = frozenset({"RugFactory"})


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


def _overlap_xy(a, b, gap: float) -> bool:
    return all(a[0][i] < b[1][i] + gap and b[0][i] < a[1][i] + gap for i in (0, 1))


def _overlap(a, b, gap: float) -> bool:
    return all(a[0][i] < b[1][i] + gap and b[0][i] < a[1][i] + gap for i in range(3))


def _delete(obj) -> None:
    for part in (*obj.children_recursive, obj):
        bpy.data.objects.remove(part, do_unlink=True)


def settle_props(props, *, floor_depth_m: float, gap_m: float = 0.02,
                 camera_margin_m: float = 0.05) -> list[PropPlacement]:
    """Clear props of wall fixtures and cameras, both placed after the props.

    A prop overlapping a structure (a window sill, utility box, door handle) is
    pushed out from the wall until it clears it, or dropped if that leaves the
    floor. A prop whose bounds hold a camera is dropped: the camera would render
    the inside of a shelf. Neither step draws random numbers.
    """
    from dataclasses import replace

    blockers = [b for o in bpy.data.objects if o.get("cwg_role") == "structure"
                and o.type in ("MESH", "CURVE") and (b := _world_bounds(o)) is not None]
    cameras = [o.matrix_world.translation for o in bpy.data.objects if o.type == "CAMERA"]
    kept = []
    for prop in props:
        obj = bpy.data.objects.get(prop.object_name)
        if obj is None:
            continue
        bounds = _world_bounds(obj)
        if bounds is None:
            _delete(obj)
            continue
        lo, hi = bounds
        hits = [b for b in blockers if _overlap((lo, hi), b, gap_m)]
        if hits:
            dy = min(b[0][1] for b in hits) - gap_m - hi[1]
            lo, hi = lo + (0.0, dy, 0.0), hi + (0.0, dy, 0.0)
            if lo[1] < -floor_depth_m + gap_m or any(_overlap((lo, hi), b, gap_m) for b in blockers):
                _delete(obj)
                continue
            obj.location.y += dy
        if any(all(lo[i] - camera_margin_m <= c[i] <= hi[i] + camera_margin_m for i in range(3)) for c in cameras):
            _delete(obj)
            continue
        bpy.context.view_layer.update()
        kept.append(replace(prop, location=tuple(round(float(v), 5) for v in obj.location)))
    return kept


def _collection(name: str):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _place_measured(obj, rug, u, standoff, half_w, floor_depth_m, side_gap_m, placed) -> bool:
    """Move a spawned prop by its measured bounds; False (and deleted) if it does not fit."""
    # The catalogue footprint is only a hint. Real assets are bigger, turn about
    # off-centre origins, and some have their origin at mid-height, so
    # placing by the hint pushed them through the wall, the side returns
    # and the floor. Re-place by the measured bounds: the standoff becomes
    # the true gap to the wall. All draws are done by now, so moving,
    # shrinking or dropping a prop leaves the RNG stream unchanged.
    room_x = 2.0 * (half_w - side_gap_m)
    room_y = (math.inf if floor_depth_m is None else floor_depth_m) - standoff - side_gap_m
    bounds = _world_bounds(obj)
    if bounds is not None and rug:
        # Infinigen sizes rugs to a whole room (3-7 m): fit the floor in front.
        extent = bounds[1] - bounds[0]
        fit = min(1.0, 0.98 * room_x / extent[0], 0.98 * room_y / extent[1])
        if fit < 1.0:
            obj.scale = (obj.scale[0] * fit, obj.scale[1] * fit, obj.scale[2])
            bpy.context.view_layer.update()
            bounds = _world_bounds(obj)
    if bounds is None or bounds[1][0] - bounds[0][0] > room_x:
        _delete(obj)
        return False
    lo, hi = bounds
    shift = np.array([u - (lo[0] + hi[0]) * 0.5, -standoff - hi[1], -lo[2]])
    # keep the whole prop on the floor between the side returns
    shift[0] += max(0.0, -half_w + side_gap_m - (lo[0] + shift[0]))
    shift[0] -= max(0.0, hi[0] + shift[0] - (half_w - side_gap_m))
    lo, hi = lo + shift, hi + shift
    # Furniture may stand on a rug; any other overlap is interpenetration,
    # so step out from the wall past the props in the way.
    for _ in range(len(placed) + 1):
        hits = [b for b, other_rug in placed if other_rug == rug and _overlap_xy((lo, hi), b, side_gap_m)]
        if not hits:
            break
        step = np.array([0.0, min(b[0][1] for b in hits) - side_gap_m - hi[1], 0.0])
        lo, hi, shift = lo + step, hi + step, shift + step
    if hits or lo[1] < -standoff - room_y:
        _delete(obj)
        return False
    placed.append(((lo, hi), rug))
    obj.location = tuple(float(a + d) for a, d in zip(obj.location, shift))
    bpy.context.view_layer.update()
    return True


def sample_props(
    rng: random.Random,
    spec,
    placements,
    *,
    count_range: tuple[int, int] = (0, 3),
    min_defect_clearance_m: float = 0.28,
    standoff_range_m: tuple[float, float] = (0.15, 0.9),
    floor_depth_m: float | None = None,
    side_gap_m: float = 0.02,
    measured: bool = True,
) -> list[PropPlacement]:
    """Place a few props in front of the wall.

    Each prop rests on the floor, `standoff` clear of the wall, inside the
    wall's width and `floor_depth_m`, and apart from the other props. A prop that
    cannot fit is dropped. `measured=False` keeps the v1.0-v1.8 placement by
    the catalogue footprint, which let props clip; it exists to rebuild those
    releases exactly.

    `min_defect_clearance_m` keeps a prop from sitting directly on top of a
    defect in plan. Partial occlusion is wanted; a prop centred on the defect
    would hide it completely while the label still claims it is visible, which
    is the one failure mode that actively teaches the detector something false.
    Total occlusion is better handled by letting the *camera* be blocked, where
    the coverage AOV drops to zero and the exporter already holds the frame out.
    """
    count = rng.randint(*count_range)
    if count <= 0:
        return []

    collection = _collection(PROP_COLLECTION)
    results: list[PropPlacement] = []
    placed: list = []
    half_w = spec.width_m * 0.5

    for index in range(count):
        module_name, class_name, footprint, _on_floor = rng.choice(_PROP_CATALOG)
        factory_cls = _load(module_name, class_name)
        if factory_cls is None:
            continue

        for _attempt in range(30):
            u = rng.uniform(-half_w + 0.2, half_w - 0.2)
            if all(
                abs(u - p.u_m) >= min_defect_clearance_m + footprint * 0.5
                for p in placements
            ):
                break
        else:
            continue

        factory_seed = rng.randrange(1, 2**31)
        try:
            obj = factory_cls(factory_seed).spawn_asset(i=index)
        except Exception:
            # One prop failing to build is not worth losing the scene over.
            continue
        obj.parent = None

        standoff = rng.uniform(*standoff_range_m)
        # Everything stands on the floor. Small items used to be lifted to a
        # sampled height on the assumption they were on furniture, but nothing
        # supports them, so they rendered hovering in mid-air beside the wall -
        # an object with no contact shadow is a stronger synthetic tell than the
        # defect it was meant to distract from.
        z = 0.0
        # -Y is out of the wall, so a prop in front sits at negative y.
        obj.location = (u, -(standoff + footprint * 0.5), z)
        obj.rotation_euler = (0.0, 0.0, rng.uniform(0.0, 2.0 * math.pi))
        obj.name = f"CWG_Prop_{index:02d}_{class_name}"
        obj["cwg_role"] = "prop"
        for old in tuple(obj.users_collection):
            old.objects.unlink(obj)
        collection.objects.link(obj)
        bpy.context.view_layer.update()

        if measured and not _place_measured(obj, class_name in FLOOR_COVERINGS, u, standoff, half_w,
                                            floor_depth_m, side_gap_m, placed):
            continue

        results.append(
            PropPlacement(
                index=index,
                factory=class_name,
                factory_seed=factory_seed,
                location=tuple(round(float(v), 5) for v in obj.location),
                yaw_deg=round(math.degrees(obj.rotation_euler.z), 2),
                object_name=obj.name,
            )
        )
    return results
