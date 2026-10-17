"""Randomised lighting for the controlled wall.

This is the axis the whole project rests on. All three release defects are pure
relief - no pigment anywhere after the grime and crease terms were made optional
- so whether a defect is visible at all is decided by where the light is. The
apartment pipeline sampled that only incidentally, through room layout; here it
is swept on purpose.

Incidence is measured from the wall normal, the same convention as
``syndefect_light_incidence_deg`` in placement/camera.py: 0 degrees is head-on
and flattens a dome, 90 degrees is grazing and rakes across relief. Pipeline
measurements put a blister at 3.1% Weber contrast at 45 degrees against 37.8% at
grazing, so a dataset that samples only one end of that range teaches the
detector only one of the two appearances a real defect has.

Colour temperature is sampled wide on purpose too. Indoor lamps ran at
clip_gaussian(4700, 700, 3500, 6500) K in the apartment generator with no white
balance anywhere, which put a fixed orange cast on every frame. Auto white
balance now corrects the cast; sampling temperature widely here is what stops
the *residual* from being a constant the detector can key on.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import bpy
from mathutils import Vector

CAPTURE_COLLECTION = "CWG_CAPTURE"

# Named regimes rather than one uniform draw, because real rooms are lit by a
# small number of recognisable arrangements and a detector should see all of
# them. The weights are roughly how often an inspection photo looks like each.
LIGHT_REGIMES = ("window_raking", "window_soft", "ceiling_diffuse", "lamp_mixed", "flash")
_REGIME_WEIGHTS = (0.26, 0.18, 0.22, 0.18, 0.16)

# Incidence band per regime, in degrees from the wall normal.
_REGIME_INCIDENCE = {
    "window_raking": (62.0, 86.0),
    "window_soft": (35.0, 62.0),
    "ceiling_diffuse": (8.0, 34.0),
    "lamp_mixed": (25.0, 70.0),
    # A phone flash is on the lens axis, so it is head-on almost by definition.
    # This is the regime in which a blister is nearly invisible, and it belongs
    # in the set for exactly that reason.
    "flash": (0.0, 14.0),
}

_REGIME_KELVIN = {
    "window_raking": (5200.0, 7200.0),
    "window_soft": (5400.0, 7600.0),
    "ceiling_diffuse": (2900.0, 4600.0),
    "lamp_mixed": (2600.0, 3600.0),
    "flash": (5000.0, 6200.0),
}

# Hard, directional sources, opt-in through the "hard_mix" profile. Every
# default source is a 0.4-2.6 m disk at 1.4-3.6 m, which subtends roughly
# 10-40 degrees: a sub-millimetre peel step or crack lip then throws a shadow
# that blurs away within a pixel. A sun patch through a window, a bare bulb and
# an inspector's torch held flat to the wall subtend 0.5-4 degrees and cast the
# crisp edge shadows real inspection photos show.
HARD_REGIMES = ("sun_raking", "bare_bulb", "torch_raking")
_REGIME_INCIDENCE.update({
    "sun_raking": (50.0, 86.0),
    "bare_bulb": (25.0, 80.0),
    # Past ~80 degrees the wall's own paint relief turns to crumpled paper.
    "torch_raking": (60.0, 80.0),
})
_REGIME_KELVIN.update({
    "sun_raking": (4800.0, 6000.0),
    "bare_bulb": (2600.0, 4200.0),
    "torch_raking": (4200.0, 6500.0),
})
# Half the draws are hard sources; the soft regimes keep their share between them.
_HARD_MIX_WEIGHTS = {
    "window_raking": 0.13, "window_soft": 0.09, "ceiling_diffuse": 0.11,
    "lamp_mixed": 0.09, "flash": 0.08,
    "sun_raking": 0.18, "bare_bulb": 0.14, "torch_raking": 0.18,
}
LIGHT_PROFILES = ("default", "hard_mix")


@dataclass(frozen=True)
class LightSpec:
    role: str
    regime: str
    kelvin: float
    power_w: float
    size_m: float
    incidence_deg: float
    azimuth_deg: float
    location: tuple[float, float, float]
    # AREA disk (size_m = diameter), POINT/SPOT (size_m = bulb diameter) or SUN
    # (angle_deg = angular diameter; power_w is irradiance in W/m^2).
    light_type: str = "AREA"
    angle_deg: float = 0.0
    target: tuple[float, float, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LightingPlan:
    regime: str
    world_strength: float
    lights: tuple[LightSpec, ...]

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "world_strength": self.world_strength,
            "lights": [light.to_dict() for light in self.lights],
        }


def _placement(width_m, height_m, incidence_deg, azimuth_deg, distance, target=None):
    """Position for a light at the given incidence from the wall normal.

    The wall's front face lies in the z=height plane at y=0 with its outward
    normal along -Y, so a light at incidence 0 sits straight out in -Y and one
    at 90 degrees lies in the plane of the wall. `azimuth_deg` rotates that
    offset around the normal, which is what decides whether a crack is raked
    along its length (and nearly vanishes) or across it. `target` defaults to
    the wall centre.
    """
    inc = math.radians(incidence_deg)
    azi = math.radians(azimuth_deg)
    # Offset direction in the wall plane, from the normal.
    in_plane = Vector((math.cos(azi), 0.0, math.sin(azi)))
    direction = Vector((0.0, -math.cos(inc), 0.0)) + in_plane * math.sin(inc)
    direction.normalize()
    target = Vector((0.0, 0.0, height_m * 0.5)) if target is None else Vector(target)
    return tuple(target + direction * distance)


def _sample_hard_mix(rng, width_m, height_m, target) -> LightingPlan:
    """Default regimes plus hard sources, with less ambient fill behind them."""
    regime = rng.choices(list(_HARD_MIX_WEIGHTS), weights=list(_HARD_MIX_WEIGHTS.values()), k=1)[0]
    hard = regime in HARD_REGIMES
    incidence = rng.uniform(*_REGIME_INCIDENCE[regime])
    azimuth = rng.uniform(0.0, 360.0)
    kelvin = rng.uniform(*_REGIME_KELVIN[regime])
    span = max(width_m, height_m)
    angle = 0.0
    if regime == "sun_raking":
        kind, size, distance, power = "SUN", 0.0, 3.0, rng.uniform(2.0, 8.0)
        angle = rng.uniform(0.5, 2.0)
    elif regime == "bare_bulb":
        kind, size, distance, power = "POINT", rng.uniform(0.01, 0.05), rng.uniform(0.8, 2.6), rng.uniform(25.0, 120.0)
    elif regime == "torch_raking":
        kind, size, distance, power = "SPOT", rng.uniform(0.01, 0.04), rng.uniform(0.35, 1.2), rng.uniform(5.0, 40.0)
    else:
        kind = "AREA"
        # Same as the default regimes, but the disk can now be as small as a
        # 10 cm shade, not only a softbox-sized window.
        size = rng.uniform(0.15, 0.6) if regime == "flash" else rng.uniform(0.1, 2.6)
        distance = rng.uniform(1.4, 3.6) * (0.6 + 0.4 * span / 3.0)
        power = rng.uniform(40.0, 220.0) * (span / 3.0) ** 2
    key = LightSpec(
        role="key", regime=regime, kelvin=kelvin, power_w=power, size_m=size,
        incidence_deg=incidence, azimuth_deg=azimuth,
        location=_placement(width_m, height_m, incidence, azimuth, distance, target),
        light_type=kind, angle_deg=angle, target=target,
    )
    lights = [key]
    # Hard light is mostly seen alone: a second source would fill the shadow.
    if rng.random() < (0.25 if hard else 0.5):
        fill_regime = rng.choice([r for r in LIGHT_REGIMES if r != regime])
        fill_incidence = rng.uniform(*_REGIME_INCIDENCE[fill_regime])
        fill_azimuth = (azimuth + rng.uniform(90.0, 270.0)) % 360.0
        fill_scale = 100.0 * (span / 3.0) ** 2 if kind != "AREA" else power
        lights.append(LightSpec(
            role="fill", regime=fill_regime, kelvin=rng.uniform(*_REGIME_KELVIN[fill_regime]),
            power_w=fill_scale * (rng.uniform(0.05, 0.25) if hard else rng.uniform(0.12, 0.45)),
            size_m=rng.uniform(0.6, 3.0), incidence_deg=fill_incidence, azimuth_deg=fill_azimuth,
            location=_placement(width_m, height_m, fill_incidence, fill_azimuth, rng.uniform(1.8, 4.2), target),
            target=target,
        ))
    return LightingPlan(
        regime=regime,
        world_strength=rng.uniform(0.01, 0.15) if hard else rng.uniform(0.04, 0.45),
        lights=tuple(lights),
    )


def sample_lighting_plan(
    rng: random.Random, width_m: float, height_m: float, profile: str = "default", target=None
) -> LightingPlan:
    """One dominant regime, optionally with a weaker second source.

    profile="hard_mix" (opt-in) adds sun, bare-bulb and raking-torch sources and
    can aim the rig at `target`; "default" draws exactly as before.
    """
    if profile == "hard_mix":
        return _sample_hard_mix(rng, width_m, height_m, target)
    if profile != "default":
        raise ValueError(f"unknown light profile {profile!r}")
    regime = rng.choices(LIGHT_REGIMES, weights=_REGIME_WEIGHTS, k=1)[0]
    inc_lo, inc_hi = _REGIME_INCIDENCE[regime]
    k_lo, k_hi = _REGIME_KELVIN[regime]
    span = max(width_m, height_m)

    key_distance = rng.uniform(1.4, 3.6) * (0.6 + 0.4 * span / 3.0)
    key = LightSpec(
        role="key",
        regime=regime,
        kelvin=rng.uniform(k_lo, k_hi),
        # Domestic fittings, not film lights. The first version drew
        # 220-1100 W and scaled it by area, reaching 2465 W on a 3.5 m wall and
        # clipping 68-100% of every rendered frame - a wall already at 255
        # cannot show defect contrast at all, which is the same trap LIMITS.md
        # records from the earlier bad measurement. Metering below now sets the
        # exposure, so this range only has to be plausible, not exact.
        power_w=rng.uniform(40.0, 220.0) * (span / 3.0) ** 2,
        # Source size is what separates a hard raking shadow from a soft one,
        # and a defect's relief reads completely differently between them.
        size_m=rng.uniform(0.15, 0.6) if regime == "flash" else rng.uniform(0.4, 2.6),
        incidence_deg=rng.uniform(inc_lo, inc_hi),
        azimuth_deg=rng.uniform(0.0, 360.0),
        location=(0.0, 0.0, 0.0),
    )
    key = LightSpec(
        **{
            **key.to_dict(),
            "location": _placement(
                width_m, height_m, key.incidence_deg, key.azimuth_deg, key_distance
            ),
        }
    )
    lights = [key]

    # A second, weaker source about half the time. Real rooms are rarely lit by
    # one thing, and a single source makes the shadow direction perfectly
    # predictable from the defect's own shading - a cue that does not survive
    # contact with a real photo.
    if rng.random() < 0.5:
        fill_regime = rng.choice([r for r in LIGHT_REGIMES if r != regime])
        f_lo, f_hi = _REGIME_INCIDENCE[fill_regime]
        fk_lo, fk_hi = _REGIME_KELVIN[fill_regime]
        fill_incidence = rng.uniform(f_lo, f_hi)
        fill_azimuth = (key.azimuth_deg + rng.uniform(90.0, 270.0)) % 360.0
        fill_distance = rng.uniform(1.8, 4.2)
        lights.append(
            LightSpec(
                role="fill",
                regime=fill_regime,
                kelvin=rng.uniform(fk_lo, fk_hi),
                power_w=key.power_w * rng.uniform(0.12, 0.45),
                size_m=rng.uniform(0.6, 3.0),
                incidence_deg=fill_incidence,
                azimuth_deg=fill_azimuth,
                location=_placement(
                    width_m, height_m, fill_incidence, fill_azimuth, fill_distance
                ),
            )
        )

    return LightingPlan(
        regime=regime,
        # Ambient floor. Too low and every unlit face is pure black, which no
        # real room is; too high and it washes the relief out.
        world_strength=rng.uniform(0.04, 0.45),
        lights=tuple(lights),
    )


def _blackbody_light(name, spec: LightSpec, collection, target):
    bpy.ops.object.light_add(type=spec.light_type, location=spec.location)
    light = bpy.context.object
    light.name = name
    light.data.name = f"{name}_Data"
    light.data.energy = spec.power_w
    if spec.light_type == "AREA":
        light.data.shape = "DISK"
        light.data.size = spec.size_m
    elif spec.light_type == "SUN":
        light.data.angle = math.radians(spec.angle_deg)
    else:
        light.data.shadow_soft_size = spec.size_m / 2.0
        if spec.light_type == "SPOT":
            light.data.spot_size = math.radians(55.0)
            light.data.spot_blend = 0.35
    light.data.use_nodes = True

    nodes = light.data.node_tree.nodes
    links = light.data.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputLight")
    emission = nodes.new("ShaderNodeEmission")
    blackbody = nodes.new("ShaderNodeBlackbody")
    blackbody.inputs["Temperature"].default_value = spec.kelvin
    links.new(blackbody.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    direction = Vector(target) - light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    light["cwg_role"] = spec.role
    light["cwg_regime"] = spec.regime
    light["cwg_incidence_deg"] = spec.incidence_deg
    light["cwg_kelvin"] = spec.kelvin
    for old in tuple(light.users_collection):
        old.objects.unlink(light)
    collection.objects.link(light)
    return light


def apply_lighting(plan: LightingPlan, height_m: float, collection) -> list:
    """Replace any existing lights with the sampled rig."""
    for obj in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj, do_unlink=True)

    centre = (0.0, 0.0, height_m * 0.5)
    built = [
        _blackbody_light(f"CWG_Light_{spec.role}", spec, collection, spec.target or centre)
        for spec in plan.lights
    ]

    world = bpy.context.scene.world
    if world is not None and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.bl_idname == "ShaderNodeBackground":
                node.inputs["Strength"].default_value = plan.world_strength
    return built
