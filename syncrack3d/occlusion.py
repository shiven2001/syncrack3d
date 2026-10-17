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


def _collection(name: str):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def sample_props(
    rng: random.Random,
    spec,
    placements,
    *,
    count_range: tuple[int, int] = (0, 3),
    min_defect_clearance_m: float = 0.28,
    standoff_range_m: tuple[float, float] = (0.15, 0.9),
) -> list[PropPlacement]:
    """Place a few props in front of the wall.

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
