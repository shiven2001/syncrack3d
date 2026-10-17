"""Pure-Python scene specification and deterministic sampling.

No Blender modules are imported here.  Keeping the sampling contract separate
from scene construction makes seeds testable and lets a future dataset manager
decide what should be generated without starting Blender.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GENERATOR_VERSION = "syncrack3d-1"
MODULE_KINDS = ("flat",)
SUBSTRATES = ("paint", "concrete", "masonry")


_PAINT_PALETTE = (
    (0.78, 0.79, 0.77, 1.0),
    (0.68, 0.70, 0.72, 1.0),
    (0.72, 0.69, 0.64, 1.0),
    (0.58, 0.63, 0.66, 1.0),
    (0.76, 0.72, 0.66, 1.0),
)
_CONCRETE_PALETTE = (
    (0.38, 0.40, 0.41, 1.0),
    (0.46, 0.45, 0.42, 1.0),
    (0.34, 0.36, 0.38, 1.0),
)
_MASONRY_PALETTE = (
    (0.42, 0.19, 0.11, 1.0),
    (0.50, 0.27, 0.17, 1.0),
    (0.34, 0.16, 0.10, 1.0),
)


@dataclass(frozen=True)
class MaterialSpec:
    """Controlled substrate parameters, recorded verbatim in the manifest."""

    finish: str
    base_color: tuple[float, float, float, float]
    roughness: float
    color_variation: float
    texture_scale: float
    micro_bump_strength: float
    micro_bump_distance_m: float
    paint_profile: str = "baseline"
    meso_scale: float = 0.0
    meso_color_variation: float = 0.0
    meso_bump_distance_m: float = 0.0
    roller_axis_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MaterialSpec":
        values = dict(values)
        values["base_color"] = tuple(values["base_color"])
        if "roller_axis_scale" in values:
            values["roller_axis_scale"] = tuple(values["roller_axis_scale"])
        return cls(**values)


@dataclass(frozen=True)
class SceneSpec:
    """Complete non-camera, non-render specification for one surface module."""

    seed: int
    module: str
    width_m: float
    height_m: float
    thickness_m: float
    skirting: bool
    material: MaterialSpec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SceneSpec":
        values = dict(values)
        values["material"] = MaterialSpec.from_dict(values["material"])
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "SceneSpec":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _sample_material(rng: random.Random, substrate: str) -> MaterialSpec:
    if substrate == "paint":
        return MaterialSpec(
            finish="paint",
            base_color=rng.choice(_PAINT_PALETTE),
            roughness=rng.uniform(0.58, 0.82),
            color_variation=rng.uniform(0.010, 0.035),
            texture_scale=rng.uniform(35.0, 75.0),
            micro_bump_strength=rng.uniform(0.05, 0.14),
            micro_bump_distance_m=rng.uniform(0.00008, 0.00022),
        )
    if substrate == "concrete":
        return MaterialSpec(
            finish="concrete",
            base_color=rng.choice(_CONCRETE_PALETTE),
            roughness=rng.uniform(0.70, 0.92),
            color_variation=rng.uniform(0.035, 0.085),
            texture_scale=rng.uniform(4.0, 11.0),
            micro_bump_strength=rng.uniform(0.12, 0.28),
            micro_bump_distance_m=rng.uniform(0.00035, 0.0012),
        )
    if substrate == "masonry":
        return MaterialSpec(
            finish="masonry",
            base_color=rng.choice(_MASONRY_PALETTE),
            roughness=rng.uniform(0.72, 0.94),
            color_variation=rng.uniform(0.03, 0.07),
            texture_scale=rng.uniform(4.5, 7.5),
            micro_bump_strength=rng.uniform(0.18, 0.34),
            micro_bump_distance_m=rng.uniform(0.0008, 0.0020),
        )
    raise ValueError(f"Unsupported substrate: {substrate!r}")


def sample_scene_spec(
    seed: int,
    *,
    module: str | None = None,
    substrate: str | None = None,
    width_m: float | None = None,
    height_m: float | None = None,
    thickness_m: float | None = None,
    skirting: bool | None = None,
) -> SceneSpec:
    """Sample one deterministic module.

    Explicit values override only the corresponding draw.  All other values
    remain deterministic for the seed.
    """
    rng = random.Random(int(seed))
    module = module or rng.choice(MODULE_KINDS)
    if module not in MODULE_KINDS:
        raise ValueError(f"module must be one of {MODULE_KINDS}, got {module!r}")
    if substrate in (None, "random"):
        substrate = rng.choices(SUBSTRATES, weights=(0.72, 0.20, 0.08), k=1)[0]
    if substrate not in SUBSTRATES:
        raise ValueError(f"substrate must be one of {SUBSTRATES}, got {substrate!r}")

    width = float(width_m if width_m is not None else rng.uniform(3.2, 5.4))
    height = float(height_m if height_m is not None else rng.uniform(2.35, 3.05))
    thickness = float(thickness_m if thickness_m is not None else rng.uniform(0.12, 0.22))
    return SceneSpec(
        seed=int(seed),
        module=module,
        width_m=round(width, 6),
        height_m=round(height, 6),
        thickness_m=round(thickness, 6),
        skirting=bool(False if skirting is None else skirting),
        material=_sample_material(rng, substrate),
    )
