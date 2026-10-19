"""Place defect cards on a controlled wall and inject them into its material.

The apartment pipeline reaches defects through the constraint solver: a card is
a solver placeholder, its position is a by-product of scoring, and the framing a
camera gets is whatever the solve happened to allow. That is the right design
for furnished rooms and the wrong one here. On a controlled wall the position,
size, spacing and class mix are the *experiment*, so they are sampled directly
and recorded in the manifest.

Everything downstream of placement is the existing implementation, unchanged:
``run_controlled_injection`` calls ``run_injection``, which rebuilds each defect
as a field inside the wall's own material, displaces it with Cycles, and emits
the same per-class and per-instance AOVs. The labels this produces are therefore
identical in kind to the apartment set, and the same exporter reads both.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import bpy

from syncrack3d.injection_adapter import (
    DefectBinding,
    run_controlled_injection,
)

# Class -> factory, resolved lazily so importing this module does not pull in
# Blender-heavy asset code until a defect is actually requested.
_FACTORY_PATHS = {
    "crack": ("cwg_infinigen.assets.crack_plane", "CrackPlaneFactory"),
}

DEFECT_CLASSES = tuple(_FACTORY_PATHS)

# The card is built thin in X with its extent in Y/Z (see
# CrackPlaneFactory.create_placeholder). Rotating -90 degrees about Z sends the
# card normal to -Y, which is the wall's front face, and sends the card's own
# Y/Z extents onto world X/Z - wall width and wall height. The in-plane spin is
# applied about X *before* that, so it survives as a spin about the normal.
_CARD_YAW = -math.pi / 2

# Keep cards clear of the wall edges. A defect running off the edge is not
# wrong, but it makes `extent` a lie and the close-up framing meaningless.
_EDGE_MARGIN_M = 0.22

# Sit the card a hair proud of the wall face. It is hidden after injection, so
# this only has to be enough that support validation reads it as on the surface
# rather than inside it.
_STANDOFF_M = 0.002


@dataclass(frozen=True)
class DefectPlacement:
    """One sampled defect, recorded verbatim in the manifest."""

    index: int
    defect_class: str
    factory_seed: int
    u_m: float
    v_m: float
    size_m: float
    spin_deg: float
    object_name: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DefectPlan:
    """What to put on one wall, before anything is built."""

    counts: dict[str, int]
    size_range_m: tuple[float, float]
    min_separation_m: float


def sample_defect_plan(
    rng: random.Random,
    *,
    total_range: tuple[int, int] = (3, 6),
    size_range_m: tuple[float, float] = (0.10, 0.45),
    min_separation_m: float = 0.30,
    substrate_finish: str | None = None,
) -> DefectPlan:
    """Choose how many cracks this wall carries.

    The total is capped at the injection shader budget rather than left open:
    ``run_injection``'s ``max_per_host`` is 6 and everything on this wall shares
    one material, so asking for more only produces over-cap drops.
    ``substrate_finish`` is accepted so older callers keep working; every
    substrate can carry a crack.
    """
    del substrate_finish
    total = rng.randint(*total_range)
    return DefectPlan(
        counts={"crack": total},
        size_range_m=size_range_m,
        min_separation_m=min_separation_m,
    )


def _factory(defect_class: str, factory_seed: int):
    import importlib

    module_name, class_name = _FACTORY_PATHS[defect_class]
    return getattr(importlib.import_module(module_name), class_name)(factory_seed)


def _sample_positions(rng, plan, width_m, height_m, count):
    """Rejection-sample non-overlapping card centres on the wall face.

    Separation is enforced because two cards that overlap inject two fields into
    the same patch of material, and the instance AOV then cannot say which pixel
    belongs to which defect - the labels stay valid but the instance identity
    degrades, which is exactly what the close-up rigs are for.
    """
    half_u = max(width_m * 0.5 - _EDGE_MARGIN_M, 0.05)
    half_v_lo = _EDGE_MARGIN_M
    half_v_hi = max(height_m - _EDGE_MARGIN_M, half_v_lo + 0.05)

    placed: list[tuple[float, float]] = []
    for _ in range(count):
        for _attempt in range(80):
            u = rng.uniform(-half_u, half_u)
            v = rng.uniform(half_v_lo, half_v_hi)
            if all(
                math.hypot(u - pu, v - pv) >= plan.min_separation_m
                for pu, pv in placed
            ):
                placed.append((u, v))
                break
        else:
            # Wall too small or too crowded for the requested count. Stopping is
            # correct: silently packing them closer would defeat the separation
            # rule this function exists to enforce.
            break
    return placed


def build_defects(
    spec,
    wall: bpy.types.Object,
    rng: random.Random,
    plan: DefectPlan | None = None,
    legacy_names: bool = False,
    axis_snap: float = 0.0,
) -> list[DefectPlacement]:
    """Spawn and position defect cards on the wall's front face.

    Cards are left unparented on purpose. ``_world_matrix`` in defect_injection
    falls back to ``matrix_basis`` for unparented objects precisely because the
    depsgraph does not evaluate objects in unevaluated collections, and a parented
    card in that state reports itself at the world origin.
    """
    plan = plan or sample_defect_plan(rng)
    requested = [
        name for name, count in plan.counts.items() for _ in range(count)
    ]
    rng.shuffle(requested)
    positions = _sample_positions(
        rng, plan, spec.width_m, spec.height_m, len(requested)
    )

    placements: list[DefectPlacement] = []
    for index, ((u, v), defect_class) in enumerate(zip(positions, requested)):
        factory_seed = rng.randrange(1, 2**31)
        card = _factory(defect_class, factory_seed).spawn_asset(i=index)
        card.parent = None

        size = rng.uniform(*plan.size_range_m)
        longest = max(card.dimensions) or 1.0
        card.scale = (size / longest,) * 3

        spin = rng.uniform(0.0, 2.0 * math.pi)
        # axis_snap of the cards run along a wall axis, like cracks at panel
        # joints and floor lines. Own RNG, so the scene stream is unchanged.
        snap = random.Random(factory_seed ^ 0x534E4150)
        if snap.random() < axis_snap:
            spin = snap.choice((0.0, 0.5, 1.0, 1.5)) * math.pi + math.radians(snap.gauss(0.0, 2.0))
        card.rotation_euler = (spin, 0.0, _CARD_YAW)
        card.location = (u, -_STANDOFF_M, v)
        # Injection seeds each defect from int_hash(card.name), so the name must
        # differ between scenes: with the index and class alone, every scene's
        # defect k of a class had the same shape - 6 per class across all v1
        # seeds. The per-scene factory seed makes the name, and the shape, unique.
        card.name = (
            f"CWG_Defect_{index:02d}_{defect_class}"
            if legacy_names  # v1 naming, kept only to reproduce it
            else f"CWG_Defect_{index:02d}_{defect_class}_{factory_seed}"
        )
        card["syndefect_surface"] = "wall"
        card["cwg_defect_class"] = defect_class
        bpy.context.view_layer.update()

        placements.append(
            DefectPlacement(
                index=index,
                defect_class=defect_class,
                factory_seed=factory_seed,
                u_m=round(float(u), 6),
                v_m=round(float(v), 6),
                size_m=round(float(size), 6),
                spin_deg=round(math.degrees(spin), 3),
                object_name=card.name,
            )
        )
    return placements


def inject_defects(
    placements: list[DefectPlacement],
    wall: bpy.types.Object,
    *,
    manifest_path=None,
    scene_id: str | None = None,
    scene_seed: int | None = None,
    dicing_rate: float = 2.0,
) -> int:
    """Hand the placed cards to the existing injection implementation."""
    bindings = [
        DefectBinding(defect=bpy.data.objects[p.object_name], host=wall)
        for p in placements
        if p.object_name in bpy.data.objects
    ]
    return run_controlled_injection(
        bindings,
        manifest_path=manifest_path,
        scene_id=scene_id,
        scene_seed=scene_seed,
        dicing_rate=dicing_rate,
    )
