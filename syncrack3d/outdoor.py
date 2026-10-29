"""Outdoor building environment around the wall (facade-style walls).

Indoor walls get the room shell (room_shell.py). An outdoor wall becomes a
building facade instead: open ground in front (Infinigen 2's terrain picker,
with real displacement), no ceiling, visible sky, sometimes the building's inside corner, an eave
or a downpipe, and weathering on the wall (rain streaks from the top, ground
splash and dirt at the base, algae patches). Light is outdoor light: direct sun
at a sampled elevation and direction (hard shadows), overcast sky, the shaded
side of the building, or low golden-hour sun; one sun per wall, as in reality.

Everything here draws from its own stream (random.Random(seed ^ OUTDOOR_SALT)),
and build_dataset still makes every draw an indoor wall makes, so the same seed
has the same wall, defects and cameras indoors and outdoors. None of these
objects is an injection host; weathering edits the wall's colour before
injection, so defect labels are unchanged.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import bpy
from mathutils import Vector

OUTDOOR_SALT = 0x4F555444
SHELL_COLLECTION = "CWG_SHELL"
# Drawn and discarded. The ground surface is the terrain picker; this choice
# stays so the corner, eave and downpipe keep their existing random stream.
GROUNDS = ("cobble", "concrete", "dirt", "mud", "lawn")
SKIES = ("sun_direct", "overcast", "shade", "golden_hour")


@dataclass(frozen=True)
class OutdoorPlan:
    ground: str
    ground_tone: float
    corner: str | None
    corner_depth_m: float
    eave: bool
    eave_depth_m: float
    downpipe_x_m: float | None
    sky: str
    sun_elevation_deg: float
    sun_azimuth_deg: float  # around the wall normal: 0 = straight out from the wall
    sun_strength: float
    sun_kelvin: float
    sun_angle_deg: float
    world_strength: float
    overcast_grey: float
    rain_streaks: float
    ground_splash: float
    algae: float

    def to_dict(self):
        return asdict(self)


def environment_for(seed: int, mode: str) -> str:
    if mode in ("indoor", "outdoor"):
        return mode
    return "outdoor" if random.Random(seed ^ OUTDOOR_SALT ^ 0x4D4958).random() < 0.5 else "indoor"


def sample_outdoor_plan(seed: int, width_m: float) -> OutdoorPlan:
    r = random.Random(seed ^ OUTDOOR_SALT)
    sky = r.choices(SKIES, weights=(0.45, 0.25, 0.15, 0.15), k=1)[0]
    if sky == "sun_direct":
        elev, azi = r.uniform(10, 65), r.uniform(-80, 80)
        strength, kelvin, world = r.uniform(3.0, 8.0), r.uniform(5200, 6200), r.uniform(0.5, 1.4)
    elif sky == "golden_hour":
        elev, azi = r.uniform(2, 12), r.uniform(-75, 75)
        strength, kelvin, world = r.uniform(2.0, 5.0), r.uniform(2800, 3900), r.uniform(0.3, 0.9)
    elif sky == "shade":  # sun behind the building: the wall is lit only by sky and ground bounce
        elev, azi = r.uniform(15, 60), r.uniform(110, 250)
        strength, kelvin, world = r.uniform(4.0, 8.0), r.uniform(5200, 6200), r.uniform(0.8, 2.0)
    else:  # overcast: no direct sun at all
        elev, azi = 0.0, 0.0
        strength, kelvin, world = 0.0, 6500.0, r.uniform(1.5, 4.0)
    corner = r.choice(("left", "right")) if r.random() < 0.4 else None
    r.choice(GROUNDS)
    return OutdoorPlan(
        ground="terrain", ground_tone=r.uniform(0.0, 1.0),
        corner=corner, corner_depth_m=r.uniform(1.5, 4.0),
        eave=r.random() < 0.5, eave_depth_m=r.uniform(0.3, 0.7),
        downpipe_x_m=(r.choice((-1, 1)) * (width_m * 0.5 - r.uniform(0.15, 0.6))) if r.random() < 0.4 else None,
        sky=sky, sun_elevation_deg=elev, sun_azimuth_deg=azi, sun_strength=strength, sun_kelvin=kelvin,
        sun_angle_deg=r.uniform(0.5, 1.2), world_strength=world, overcast_grey=r.uniform(0.55, 0.85),
        rain_streaks=r.uniform(0.1, 0.35) if r.random() < 0.6 else 0.0,
        ground_splash=r.uniform(0.15, 0.4) if r.random() < 0.6 else 0.0,
        algae=r.uniform(0.15, 0.45) if r.random() < 0.3 else 0.0,
    )


# --------------------------------------------------------------------------- materials


def _principled_material(name, base, roughness=0.8):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Roughness"].default_value = roughness
    return mat, bsdf


# --------------------------------------------------------------------------- geometry


def _collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _slab(name, dims, loc, mat, col, role="shell"):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    obj["cwg_role"] = role
    for old in tuple(obj.users_collection):
        old.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def _facade_slab(name, x0, x1, z0, z1, mat, col, frame=None):
    """Facade piece flush with the wall front (y = 0), world-aligned metric UVs so
    brick courses and tile joints continue across the seam with the test wall."""
    from syncrack3d.material_diversity import _metric_uv

    t = 0.2
    obj = _slab(name, (x1 - x0, t, z1 - z0), ((x0 + x1) * 0.5, t * 0.5, (z0 + z1) * 0.5), mat, col)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    if frame is not None:
        # Share the test wall's object frame: plaster/stucco texture in object
        # coordinates, so a different origin shifted the pattern at the seam.
        from mathutils import Matrix

        obj.data.transform(Matrix.Translation(-Vector(frame)))
        obj.location = frame
    _metric_uv(obj)
    # Same true displacement as the test wall: a bump-only neighbour met the
    # displaced wall along a thin shading seam, a crack-like line at its edge.
    from syncrack3d.materials import enable_true_displacement

    enable_true_displacement(obj, bpy.context.scene.cycles.dicing_rate or 0.75)
    return obj


def facade_extent(plan: OutdoorPlan) -> tuple[float, float]:
    """How far the facade continues sideways (4-9 m) and above the test wall (1-4 m).

    Seeded with an MD5 hash of the plan, not Python's hash(), which is
    randomised per process for strings (PYTHONHASHSEED): the surrounding facade,
    the corner and the eave height - and their shadows in wide views - would
    change from run to run.
    """
    from cwg_infinigen.core.util.math import int_hash

    r = random.Random(int_hash((repr(float(plan.ground_tone)), str(plan.sky))))
    return r.uniform(4.0, 9.0), r.uniform(1.0, 4.0)


def facade_bounds(spec, plan: OutdoorPlan) -> tuple[float, float, float]:
    """(left x, right x, top z) of the facade the test wall is a panel of.

    A side with the inside corner stops at the test wall's edge, where the
    corner wall stands.
    """
    side, top = facade_extent(plan)
    left = -spec.width_m * 0.5 - (0.0 if plan.corner == "left" else side)
    right = spec.width_m * 0.5 + (0.0 if plan.corner == "right" else side)
    return left, right, spec.height_m + top


def build_outdoor(spec, plan: OutdoorPlan, wall=None, seed: int = 0, dicing_rate: float = 0.75) -> list:
    """Ground, the rest of the facade, optional corner, eave and downpipe. Wall front y = 0, normal -Y.

    The test wall is one panel of a larger facade: the same material continues
    left, right and above it (the un-injected original, so defects stay on the
    test wall only), which is what a photographed building front looks like.
    The ground is Infinigen's terrain picker without its crack-like and watery
    leaves (infinigen2_bridge._TERRAIN_LEAVES), displaced as geometry.
    """
    from cwg_infinigen.core.util.math import int_hash
    from syncrack3d.infinigen2_bridge import realize_picker
    from syncrack3d.materials import enable_true_displacement
    from syncrack3d.room_shell import _metric_uv_xy

    col = _collection(SHELL_COLLECTION)
    w, h, t = spec.width_m, spec.height_m, 0.12
    terrain_seed = int_hash((int(seed), "terrain_picker")) % (2**31)
    ground_mat = realize_picker("terrain", terrain_seed)
    ground = _slab("CWG_Outdoor_Ground", (60.0, 40.0, t), (0.0, -19.0, -t * 0.5), ground_mat, col)
    _metric_uv_xy(ground)
    enable_true_displacement(ground, dicing_rate, modifier_name="ground_adaptive")
    ground["cwg_material_generator"] = "infinigen2:terrain"
    ground["cwg_material_source"] = "infinigen2_procedural"
    built = [ground]
    left_x, right_x, facade_top = facade_bounds(spec, plan)
    top = facade_top - h
    facade = wall.data.materials[0] if wall is not None else \
        _principled_material("CWG_Outdoor_Facade", (0.45, 0.43, 0.40, 1.0), 0.85)[0]
    frame = tuple(wall.matrix_world.translation) if wall is not None else None
    if plan.corner != "left":
        built.append(_facade_slab("CWG_Outdoor_Facade_L", left_x, -w * 0.5, 0.0, h + top, facade, col, frame))
    if plan.corner != "right":
        built.append(_facade_slab("CWG_Outdoor_Facade_R", w * 0.5, right_x, 0.0, h + top, facade, col, frame))
    built.append(_facade_slab("CWG_Outdoor_Facade_Top", -w * 0.5, w * 0.5, h, h + top, facade, col, frame))
    if plan.corner:
        from syncrack3d.material_diversity import _metric_uv
        from syncrack3d.room_shell import _face_frame

        x = (-1 if plan.corner == "left" else 1) * (w * 0.5 + t * 0.5)
        corner = _slab("CWG_Outdoor_Corner", (t, plan.corner_depth_m, h + top),
                       (x, -plan.corner_depth_m * 0.5, (h + top) * 0.5), facade, col)
        # The facade shader is displacement-only: without metric UVs in the
        # wall's frame and subdivision, this side wall rendered flat.
        _face_frame(corner, math.pi / 2 if plan.corner == "left" else -math.pi / 2)
        _metric_uv(corner)
        enable_true_displacement(corner, dicing_rate, modifier_name="corner_adaptive")
        built.append(corner)
    h = h + top  # the eave and downpipe belong to the whole facade
    if plan.eave:
        built.append(_slab("CWG_Outdoor_Eave", (right_x - left_x + 0.6, plan.eave_depth_m, 0.15),
                           ((left_x + right_x) * 0.5, -plan.eave_depth_m * 0.5, h + 0.075),
                           _principled_material("CWG_Outdoor_Eave_Mat", (0.2, 0.19, 0.18, 1.0), 0.7)[0], col))
    if plan.downpipe_x_m is not None:
        from cwg_infinigen.core.util.math import int_hash
        from syncrack3d.structures import add_downpipe

        for pipe in add_downpipe(col, "CWG_Outdoor_Downpipe", plan.downpipe_x_m, 0.02, h,
                                 int_hash(("downpipe", repr(float(plan.downpipe_x_m))))):
            pipe["cwg_role"] = "prop"
            built.append(pipe)
    return built


def remove_indoor_props():
    """Indoor furniture is still sampled (to keep the RNG stream) but removed outdoors."""
    for obj in [o for o in bpy.data.objects if o.get("cwg_role") == "prop" and o.name.startswith("CWG_Prop_")]:
        bpy.data.objects.remove(obj, do_unlink=True)


# --------------------------------------------------------------------------- light and sky


def outdoor_lighting(plan: OutdoorPlan, spec):
    """A LightingPlan with one sun (or none, overcast), aimed at the wall centre."""
    from syncrack3d.lighting import LightingPlan, LightSpec

    target = (0.0, 0.0, spec.height_m * 0.5)
    lights = []
    if plan.sky != "overcast":
        e, a = math.radians(plan.sun_elevation_deg), math.radians(plan.sun_azimuth_deg)
        d = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
        incidence = math.degrees(math.acos(max(-1.0, min(1.0, d.dot(Vector((0, -1, 0)))))))
        lights.append(LightSpec(role="key", regime=f"outdoor_{plan.sky}", kelvin=plan.sun_kelvin,
                                power_w=plan.sun_strength, size_m=0.0, incidence_deg=incidence,
                                azimuth_deg=plan.sun_azimuth_deg, location=tuple(Vector(target) + d * 50.0),
                                light_type="SUN", angle_deg=plan.sun_angle_deg, target=target))
    return LightingPlan(regime=f"outdoor_{plan.sky}", world_strength=plan.world_strength, lights=tuple(lights))


def apply_sky(plan: OutdoorPlan):
    """Visible sky: Nishita for sun/shade/golden hour (no disc; the sun lamp is the sun), grey for overcast."""
    world = bpy.context.scene.world
    nt = world.node_tree
    sky = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeTexSky"), None)
    bg = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBackground")
    if plan.sky == "overcast":
        if sky is not None:
            for link in list(bg.inputs["Color"].links):
                nt.links.remove(link)
        g = plan.overcast_grey
        bg.inputs["Color"].default_value = (g, g, g * 1.03, 1.0)
    elif sky is not None:
        sky.sun_disc = False
        sky.sun_elevation = math.radians(max(plan.sun_elevation_deg, 1.0))
        sky.sun_rotation = math.radians(plan.sun_azimuth_deg)
    bg.inputs["Strength"].default_value = plan.world_strength
    world["cwg_environment"] = f"outdoor_{plan.sky}"


# --------------------------------------------------------------------------- weathering


def apply_weathering(wall, plan: OutdoorPlan, height_m: float) -> dict:
    """Rain streaks from the top, splash/dirt at the base, algae patches: background, not defects."""
    if not (plan.rain_streaks or plan.ground_splash or plan.algae):
        return {}
    mat = wall.data.materials[0]
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        return {}
    base = bsdf.inputs["Base Color"]
    if base.is_linked:
        src = base.links[0].from_socket
    else:
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = base.default_value
        src = rgb.outputs[0]
    uv = nt.nodes.new("ShaderNodeUVMap")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])
    v = sep.outputs["Y"]  # metres above the ground (metric wall UVs)

    def rng_(value, a, b):
        m = nt.nodes.new("ShaderNodeMapRange")
        m.clamp = True
        m.inputs["From Min"].default_value, m.inputs["From Max"].default_value = a, b
        nt.links.new(value, m.inputs["Value"])
        return m.outputs["Result"]

    def mul(a, b):
        m = nt.nodes.new("ShaderNodeMath")
        m.operation = "MULTIPLY"
        m.use_clamp = True
        for i, x in enumerate((a, b)):
            if hasattr(x, "is_linked"):
                nt.links.new(x, m.inputs[i])
            else:
                m.inputs[i].default_value = x
        return m.outputs[0]

    def mix(fac, c1, colour):
        m = nt.nodes.new("ShaderNodeMixRGB")
        m.blend_type = "MULTIPLY" if colour is None else "MIX"
        nt.links.new(fac, m.inputs["Fac"])
        nt.links.new(c1, m.inputs["Color1"])
        m.inputs["Color2"].default_value = colour if colour is not None else (0.55, 0.53, 0.5, 1.0)
        return m.outputs["Color"]

    color = src
    if plan.rain_streaks:  # long vertical runs, strongest near the top
        stretch = nt.nodes.new("ShaderNodeMapping")
        stretch.inputs["Scale"].default_value = (8.0, 0.6, 1.0)  # 5-20 cm wide soft runs, not crack-like lines
        nt.links.new(uv.outputs["UV"], stretch.inputs["Vector"])
        streak = nt.nodes.new("ShaderNodeTexNoise")
        streak.inputs["Scale"].default_value = 2.0
        streak.inputs["Detail"].default_value = 4.0
        nt.links.new(stretch.outputs[0], streak.inputs["Vector"])
        fac = mul(mul(rng_(streak.outputs["Fac"], 0.5, 0.75), rng_(v, 0.0, height_m)), plan.rain_streaks)
        color = mix(fac, color, None)
    if plan.ground_splash:  # dirty band at the base with a ragged top edge
        edge = nt.nodes.new("ShaderNodeTexNoise")
        edge.inputs["Scale"].default_value = 6.0
        nt.links.new(uv.outputs["UV"], edge.inputs["Vector"])
        band = rng_(v, 0.55, 0.05)
        fac = mul(mul(band, rng_(edge.outputs["Fac"], 0.3, 0.6)), plan.ground_splash)
        color = mix(fac, color, (0.22, 0.18, 0.13, 1.0))
    if plan.algae:  # green-black patches low on the wall
        patch = nt.nodes.new("ShaderNodeTexNoise")
        patch.inputs["Scale"].default_value = 2.5
        nt.links.new(uv.outputs["UV"], patch.inputs["Vector"])
        fac = mul(mul(rng_(v, 0.9, 0.0), rng_(patch.outputs["Fac"], 0.55, 0.7)), plan.algae)
        color = mix(fac, color, (0.10, 0.14, 0.07, 1.0))
    nt.links.new(color, base)
    return {"rain_streaks": plan.rain_streaks, "ground_splash": plan.ground_splash, "algae": plan.algae}
