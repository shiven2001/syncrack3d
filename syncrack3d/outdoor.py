"""Outdoor building environment around the controlled wall (opt-in --environment).

Indoor walls get the room shell (room_shell.py). An outdoor wall becomes a
building facade instead: open ground in front (paving, concrete, gravel, soil,
grass), no ceiling, visible sky, sometimes the building's inside corner, an eave
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
GROUNDS = ("paving", "concrete", "gravel", "soil", "grass")
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
    return OutdoorPlan(
        ground=r.choice(GROUNDS), ground_tone=r.uniform(0.0, 1.0),
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


def _ground_material(plan: OutdoorPlan):
    """Procedural ground: tone and pattern only; it is context, not a defect host."""
    t = plan.ground_tone
    if plan.ground in ("paving", "concrete"):
        v = 0.25 + 0.3 * t
        base = (v, v * 0.98, v * 0.94, 1.0)
    elif plan.ground == "gravel":
        v = 0.30 + 0.25 * t
        base = (v, v * 0.96, v * 0.9, 1.0)
    elif plan.ground == "soil":
        base = (0.16 + 0.1 * t, 0.11 + 0.07 * t, 0.07 + 0.04 * t, 1.0)
    else:
        base = (0.06 + 0.05 * t, 0.12 + 0.08 * t, 0.04 + 0.03 * t, 1.0)
    mat, bsdf = _principled_material(f"CWG_Ground_{plan.ground}", base, 0.9)
    nt = mat.node_tree
    coord = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = {"paving": 2.0, "concrete": 4.0, "gravel": 60.0, "soil": 8.0, "grass": 30.0}[plan.ground]
    noise.inputs["Detail"].default_value = 8.0
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "OVERLAY"
    mix.inputs["Fac"].default_value = 0.35
    mix.inputs["Color1"].default_value = base
    grey = nt.nodes.new("ShaderNodeValToRGB")  # grey variation only: coloured noise read as water
    nt.links.new(noise.outputs["Fac"], grey.inputs["Fac"])
    nt.links.new(grey.outputs["Color"], mix.inputs["Color2"])
    if plan.ground == "paving":  # square slabs with dark joints
        bricks = nt.nodes.new("ShaderNodeTexBrick")
        bricks.offset = 0.5
        bricks.inputs["Scale"].default_value = 1.0
        bricks.inputs["Row Height"].default_value = 0.4
        bricks.inputs["Brick Width"].default_value = 0.4
        bricks.inputs["Mortar Size"].default_value = 0.01
        nt.links.new(coord.outputs["Object"], bricks.inputs["Vector"])
        joints = nt.nodes.new("ShaderNodeMixRGB")
        joints.inputs["Color2"].default_value = (base[0] * 0.4, base[1] * 0.4, base[2] * 0.4, 1.0)
        nt.links.new(bricks.outputs["Fac"], joints.inputs["Fac"])
        nt.links.new(mix.outputs["Color"], joints.inputs["Color1"])
        nt.links.new(joints.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


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

    Seeded with an MD5 hash of the plan. Until v1.8.0 this used Python's hash(),
    which is randomised per process for strings (PYTHONHASHSEED), so the
    surrounding facade, the corner and the eave height - and their shadows in
    wide views - changed from run to run. Test-wall labels were unaffected.
    """
    from cwg_infinigen.core.util.math import int_hash

    r = random.Random(int_hash((repr(float(plan.ground_tone)), str(plan.sky))))
    return r.uniform(4.0, 9.0), r.uniform(1.0, 4.0)


def build_outdoor(spec, plan: OutdoorPlan, wall=None) -> list:
    """Ground, the rest of the facade, optional corner, eave and downpipe. Wall front y = 0, normal -Y.

    The test wall is one panel of a larger facade: the same material continues
    left, right and above it (the un-injected original, so defects stay on the
    test wall only), which is what a photographed building front looks like.
    """
    col = _collection(SHELL_COLLECTION)
    w, h, t = spec.width_m, spec.height_m, 0.12
    built = [_slab("CWG_Outdoor_Ground", (60.0, 40.0, t), (0.0, -19.0, -t * 0.5), _ground_material(plan), col)]
    side, top = facade_extent(plan)
    facade = wall.data.materials[0] if wall is not None else \
        _principled_material("CWG_Outdoor_Facade", (0.45, 0.43, 0.40, 1.0), 0.85)[0]
    left_x = -w * 0.5 - (0.0 if plan.corner == "left" else side)
    right_x = w * 0.5 + (0.0 if plan.corner == "right" else side)
    frame = tuple(wall.matrix_world.translation) if wall is not None else None
    if plan.corner != "left":
        built.append(_facade_slab("CWG_Outdoor_Facade_L", left_x, -w * 0.5, 0.0, h + top, facade, col, frame))
    if plan.corner != "right":
        built.append(_facade_slab("CWG_Outdoor_Facade_R", w * 0.5, right_x, 0.0, h + top, facade, col, frame))
    built.append(_facade_slab("CWG_Outdoor_Facade_Top", -w * 0.5, w * 0.5, h, h + top, facade, col, frame))
    if plan.corner:
        x = (-1 if plan.corner == "left" else 1) * (w * 0.5 + t * 0.5)
        built.append(_slab("CWG_Outdoor_Corner", (t, plan.corner_depth_m, h + top),
                           (x, -plan.corner_depth_m * 0.5, (h + top) * 0.5), facade, col))
    h = h + top  # the eave and downpipe belong to the whole facade
    if plan.eave:
        built.append(_slab("CWG_Outdoor_Eave", (right_x - left_x + 0.6, plan.eave_depth_m, 0.15),
                           ((left_x + right_x) * 0.5, -plan.eave_depth_m * 0.5, h + 0.075),
                           _principled_material("CWG_Outdoor_Eave_Mat", (0.2, 0.19, 0.18, 1.0), 0.7)[0], col))
    if plan.downpipe_x_m is not None:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.045, depth=h + 0.1, location=(plan.downpipe_x_m, -0.09, (h + 0.1) * 0.5))
        pipe = bpy.context.object
        pipe.name = "CWG_Outdoor_Downpipe"
        pipe.data.materials.append(_principled_material("CWG_Downpipe_Mat", (0.35, 0.35, 0.36, 1.0), 0.45)[0])
        pipe["cwg_role"] = "prop"
        for old in tuple(pipe.users_collection):
            old.objects.unlink(pipe)
        col.objects.link(pipe)
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
