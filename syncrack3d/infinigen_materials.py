"""Use Infinigen's own wall materials on the controlled wall.

The hand-written shaders in ``materials.py`` were built to give the base
generator something deterministic and dependency-free. For a dataset meant to
be compared against the apartment set, they are the wrong choice on three
counts:

* **They are not the surface the apartment defects live on.** Apartment defects
  are injected into ``ceramic.Plaster``. A controlled-wall dataset shaded with a
  different plaster differs from the apartment set in wall appearance *and* in
  defect appearance at once, so any difference in transfer cannot be attributed
  to either.
* **They are narrower.** Three hand-written substrates against Infinigen's
  Plaster / Concrete / Brick / NonWoodTile, each with its own internal sampling.
  Material variety is one of the axes this generator exists to sweep.
* **Injection reads the host shader.** ``_wire_displacement`` *adds* into an
  existing Displacement chain rather than replacing it. A simpler shader
  silently takes different branches, so the injected crack is not the one the
  apartment path produces.

``surface_finish`` is derived with the apartment's own ``_surface_finish``.
"""

from __future__ import annotations

import random

# Room-type wall palettes, sampled per wall. Weights are how often a defect
# dataset should see each kind of interior surface, not how common the room type
# is: plaster dominates because that is what indoor defects are photographed on.
_PALETTE_WEIGHTS = (
    ("wall", 0.42),          # Plaster only
    ("bathroom_wall", 0.16),  # Plaster 8 / NonWoodTile 2
    ("utility_wall", 0.16),   # Plaster 5 / Brick 2 / Concrete 1
    ("balcony_wall", 0.12),   # Plaster 5 / Brick 1
    ("garage_wall", 0.14),    # Concrete 5 / Plaster 3 / Brick 1
)


def apply_infinigen_wall_material(wall, rng: random.Random) -> tuple[str, str]:
    """Shade `wall` with a sampled Infinigen wall material.

    Returns (material_generator_name, surface_finish). The finish is what
    defect substrate QA consumes: "paint" admits all three classes, while
    "concrete", "masonry" and "tile" admit cracks only.
    """
    import numpy as np

    from cwg_infinigen.assets.composition import material_assignments
    from cwg_infinigen.core import surface
    from cwg_infinigen.core.constraints.example_solver.room.decorate import (
        _surface_finish,
    )
    from cwg_infinigen.core.util.random import weighted_sample

    palette_name = rng.choices(
        [name for name, _ in _PALETTE_WEIGHTS],
        weights=[w for _, w in _PALETTE_WEIGHTS],
        k=1,
    )[0]
    palette = getattr(material_assignments, palette_name)

    # weighted_sample draws through numpy, so the scene seed already set by the
    # driver is what makes this reproducible - not `rng`.
    generator_cls = weighted_sample(palette)
    generator = generator_cls()

    # Mirrors room_walls: only a few generators accept the vertical hint, and
    # passing it to one that does not raises.
    kwargs = {}
    if generator.__class__.__name__ not in ("Concrete", "Plaster", "Wood", "Metal"):
        kwargs = {}
    material = generator(**kwargs) if callable(generator) else generator

    # Clear the placeholder shader so the wall carries exactly one material and
    # injection's "one unique Principled material" contract still holds.
    wall.data.materials.clear()
    surface.assign_material(wall, material)

    finish = _surface_finish(generator)
    wall["surface_finish"] = finish
    wall["cwg_material_generator"] = generator.__class__.__name__
    wall["cwg_material_palette"] = palette_name
    del np
    return generator.__class__.__name__, finish
