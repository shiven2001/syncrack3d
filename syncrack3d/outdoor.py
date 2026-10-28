"""Outdoor building environment around the wall (facade-style walls).

Indoor walls get the room shell (room_shell.py). An outdoor wall becomes a
building facade instead: open ground in front (cobble, poured concrete,
dirt, mud, or a lawn of grass blades), no ceiling, visible sky, sometimes the building's inside corner, an eave
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


def _ground_seed(plan: OutdoorPlan) -> int:
    from cwg_infinigen.core.util.math import int_hash

    return int_hash(("ground", plan.ground, repr(round(float(plan.ground_tone), 6)))) % (2**32)


def _restore_rng(fn):
    """Infinigen shaders draw from numpy while they build. Put that state back."""
    import numpy as np

    from cwg_infinigen.core.util.math import FixedSeed

    def wrapped(*args, **kwargs):
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            with FixedSeed(kwargs.pop("fixed_seed")):
                return fn(*args, **kwargs)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

    return wrapped


def _bump_only(mat):
    """Shader displacement is for a diced terrain mesh. This slab is a facade ground."""
    mat.displacement_method = "BUMP"
    for node in mat.node_tree.nodes:
        if node.type == "DISPLACEMENT" and "Scale" in node.inputs:
            node.inputs["Scale"].default_value = 0.0
    return mat


def _lawn_bed(name):
    """Green thatch under the blades, with the same one-metre mowing stripes.

    Gaps between blades would otherwise show the red dirt shader. A mown lawn
    reads as green even where a single blade does not cover the soil.
    """
    dark = (0.05, 0.16, 0.035, 1.0)
    mid = (0.10, 0.28, 0.055, 1.0)
    mat, bsdf = _principled_material(name, mid, 0.86)
    nt = mat.node_tree
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    separated = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Position"], separated.inputs["Vector"])
    divided = nt.nodes.new("ShaderNodeMath")
    divided.operation = "DIVIDE"
    divided.inputs[1].default_value = 1.05
    nt.links.new(separated.outputs["X"], divided.inputs[0])
    fract = nt.nodes.new("ShaderNodeMath")
    fract.operation = "FRACT"
    nt.links.new(divided.outputs["Value"], fract.inputs[0])
    stripes = nt.nodes.new("ShaderNodeValToRGB")
    stripes.color_ramp.elements[0].position = 0.38
    stripes.color_ramp.elements[0].color = dark
    stripes.color_ramp.elements[1].position = 0.62
    stripes.color_ramp.elements[1].color = mid
    nt.links.new(fract.outputs["Value"], stripes.inputs["Fac"])
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 40.0
    grain.inputs["Detail"].default_value = 4.0
    nt.links.new(geom.outputs["Position"], grain.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Fac"].default_value = 0.22
    nt.links.new(grain.outputs["Fac"], mix.inputs["Fac"])
    nt.links.new(stripes.outputs["Color"], mix.inputs["Color1"])
    mix.inputs["Color2"].default_value = (0.14, 0.34, 0.07, 1.0)
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _ground_material(plan: OutdoorPlan):
    """Infinigen terrain shaders: cobble, concrete, dirt, mud. A lawn is thatch plus blades.

    Colour comes from their world-space noise. Height does not: dirt and mud geomods
    are landscape-scale, and concrete's displacement node is a millimetre of dice.
    Cobble's geomod is a few centimetres, so the apron uses that one as geometry.
    """
    from cwg_infinigen.core.surface import shaderfunc_to_material

    seed = _ground_seed(plan)
    name = f"CWG_Ground_{plan.ground}_{seed & 0xFFFF:04x}"
    if plan.ground == "lawn":
        return _bump_only(_lawn_bed(name))

    def build():
        if plan.ground == "cobble":
            from cwg_infinigen.assets.materials.terrain.cobble_stone import shader_cobblestone

            return shaderfunc_to_material(shader_cobblestone, random_seed=seed, name=name)
        if plan.ground == "concrete":
            from cwg_infinigen.assets.materials.ceramic.concrete import shader_concrete

            value = 0.42 + 0.22 * plan.ground_tone
            return shaderfunc_to_material(
                shader_concrete, name=name, scale=1.6, seed=float(seed % 997),
                roughness=0.84, crack_amount=0.12, crack_scale=1.4, snake_crack=0.2,
                base_color_hsv=(0.08, 0.045, value),
            )
        if plan.ground == "mud":
            from cwg_infinigen.assets.materials.terrain.mud import shader_mud

            return shaderfunc_to_material(shader_mud, name=name)
        from cwg_infinigen.assets.materials.terrain.dirt import shader_dirt

        return shaderfunc_to_material(shader_dirt, random_seed=seed, name=name)

    return _bump_only(_restore_rng(build)(fixed_seed=seed))


def _adopt(obj, col):
    for old in tuple(obj.users_collection):
        old.objects.unlink(obj)
    col.objects.link(obj)
    obj["cwg_role"] = "shell"
    return obj


def _grid(name, span, depth, spacing, z):
    """A ground apron in front of the wall: x across the facade, y from 0 out to -depth."""
    import bmesh

    nx = max(2, int(round(span / spacing)))
    ny = max(2, int(round(depth / spacing)))
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=nx, y_segments=ny, size=1.0)
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    sx = span / (maxx - minx)
    sy = depth / (maxy - miny)
    for vert in bm.verts:
        vert.co.x = (vert.co.x - minx) * sx - span * 0.5
        vert.co.y = (vert.co.y - miny) * sy - depth
        vert.co.z = z
    bm.normal_update()
    if sum(face.normal.z for face in bm.faces) < 0.0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.normal_update()
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return bpy.data.objects.new(name, mesh)


def _use_displaced_output(ng):
    """The stock geometry-node group passes the mesh through. Keep the node that moves it."""
    outputs = [node for node in ng.nodes if node.bl_idname == "NodeGroupOutput"]
    chosen = None
    for node in outputs:
        sock = node.inputs[0] if node.inputs else None
        if sock and sock.is_linked and sock.links[0].from_node.bl_idname != "NodeGroupInput":
            chosen = node
            break
    if chosen is None:
        return
    for node in outputs:
        node.is_active_output = node is chosen


def _bake_displacement(obj, fn, kwargs, seed):
    from cwg_infinigen.core import surface

    def build():
        return surface.add_geomod(obj, fn, apply=False, input_kwargs=kwargs)

    mod = _restore_rng(build)(fixed_seed=seed)
    _use_displaced_output(mod.node_group)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(deps)
    baked = evaluated.to_mesh().copy()
    evaluated.to_mesh_clear()
    obj.modifiers.clear()
    old = obj.data
    obj.data = baked
    if old.users == 0:
        bpy.data.meshes.remove(old)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    obj.data.update()
    return obj


def geo_relief(nw, height=0.02, scale=5.0, detail=4.0):
    """Centimetre clods. Infinigen's dirt and mud geomods are built for mountains."""
    from cwg_infinigen.core.nodes.node_wrangler import Nodes

    group_in = nw.new_node(Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)])
    position = nw.new_node(Nodes.InputPosition)
    normal = nw.new_node(Nodes.InputNormal)
    coarse = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Vector": position, "Scale": scale, "Detail": detail, "Roughness": 0.55},
    )
    fine = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Vector": position, "Scale": scale * 7.0, "Detail": 3.0, "Roughness": 0.65},
    )
    lifted = nw.new_node(
        Nodes.Math, input_kwargs={0: coarse.outputs["Fac"], 1: 0.42}, attrs={"operation": "SUBTRACT"}
    )
    lifted = nw.new_node(Nodes.Math, input_kwargs={0: lifted, 1: 0.0}, attrs={"operation": "MAXIMUM"})
    lifted = nw.new_node(
        Nodes.Math, input_kwargs={0: lifted, 1: height / 0.58}, attrs={"operation": "MULTIPLY"}
    )
    grain = nw.new_node(
        Nodes.Math, input_kwargs={0: fine.outputs["Fac"], 1: 0.5}, attrs={"operation": "SUBTRACT"}
    )
    grain = nw.new_node(
        Nodes.Math, input_kwargs={0: grain, 1: height * 0.18}, attrs={"operation": "MULTIPLY"}
    )
    total = nw.new_node(Nodes.Math, input_kwargs={0: lifted, 1: grain}, attrs={"operation": "ADD"})
    offset = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: normal, 1: total}, attrs={"operation": "MULTIPLY"}
    )
    moved = nw.new_node(
        Nodes.SetPosition, input_kwargs={"Geometry": group_in, "Offset": offset.outputs["Vector"]}
    )
    nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": moved})


def geo_concrete_ground(nw):
    """A poured slab: a few millimetres of float, and saw cuts about every 3 m."""
    from cwg_infinigen.core.nodes.node_wrangler import Nodes

    group_in = nw.new_node(Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)])
    position = nw.new_node(Nodes.InputPosition)
    normal = nw.new_node(Nodes.InputNormal)
    separated = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})
    pitch = 3.05

    def nearest(component):
        shifted = nw.new_node(
            Nodes.Math, input_kwargs={0: component, 1: 50.0}, attrs={"operation": "ADD"}
        )
        divided = nw.new_node(
            Nodes.Math, input_kwargs={0: shifted, 1: pitch}, attrs={"operation": "DIVIDE"}
        )
        fract = nw.new_node(Nodes.Math, input_kwargs={0: divided}, attrs={"operation": "FRACT"})
        along = nw.new_node(
            Nodes.Math, input_kwargs={0: fract, 1: pitch}, attrs={"operation": "MULTIPLY"}
        )
        other = nw.new_node(
            Nodes.Math, input_kwargs={0: pitch, 1: along}, attrs={"operation": "SUBTRACT"}
        )
        return nw.new_node(Nodes.Math, input_kwargs={0: along, 1: other}, attrs={"operation": "MINIMUM"})

    gap = nw.new_node(
        Nodes.Math,
        input_kwargs={0: nearest(separated.outputs["X"]), 1: nearest(separated.outputs["Y"])},
        attrs={"operation": "MINIMUM"},
    )
    groove = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": gap, "From Min": 0.0, "From Max": 0.012, "To Min": 0.008, "To Max": 0.0},
    )
    waviness = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Vector": position, "Scale": 0.35, "Detail": 2.0, "Roughness": 0.45},
    )
    wave = nw.new_node(
        Nodes.Math, input_kwargs={0: waviness.outputs["Fac"], 1: 0.5}, attrs={"operation": "SUBTRACT"}
    )
    wave = nw.new_node(Nodes.Math, input_kwargs={0: wave, 1: 0.006}, attrs={"operation": "MULTIPLY"})
    grit = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Vector": position, "Scale": 70.0, "Detail": 4.0, "Roughness": 0.7},
    )
    grain = nw.new_node(
        Nodes.Math, input_kwargs={0: grit.outputs["Fac"], 1: 0.5}, attrs={"operation": "SUBTRACT"}
    )
    grain = nw.new_node(Nodes.Math, input_kwargs={0: grain, 1: 0.0015}, attrs={"operation": "MULTIPLY"})
    height = nw.new_node(
        Nodes.Math, input_kwargs={0: wave, 1: grain}, attrs={"operation": "ADD"}
    )
    height = nw.new_node(
        Nodes.Math, input_kwargs={0: height, 1: groove.outputs["Result"]}, attrs={"operation": "SUBTRACT"}
    )
    offset = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: normal, 1: height}, attrs={"operation": "MULTIPLY"}
    )
    moved = nw.new_node(
        Nodes.SetPosition, input_kwargs={"Geometry": group_in, "Offset": offset.outputs["Vector"]}
    )
    nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": moved})


def _grass_material(name, colors):
    """Blade colour from Infinigen's lawn palettes, with mowing stripes a metre apart.

    Stripes and clumps are world position, so instanced blades pick up the stripe
    they stand in instead of sharing one flat green.
    """
    mat, bsdf = _principled_material(name, (*colors[1][:3], 1.0), 0.58)
    bsdf.inputs["Specular IOR Level"].default_value = 0.18
    nt = mat.node_tree
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    separated = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Position"], separated.inputs["Vector"])
    divided = nt.nodes.new("ShaderNodeMath")
    divided.operation = "DIVIDE"
    divided.inputs[1].default_value = 1.05
    nt.links.new(separated.outputs["X"], divided.inputs[0])
    fract = nt.nodes.new("ShaderNodeMath")
    fract.operation = "FRACT"
    nt.links.new(divided.outputs["Value"], fract.inputs[0])
    stripes = nt.nodes.new("ShaderNodeValToRGB")
    stripes.color_ramp.elements[0].position = 0.38
    stripes.color_ramp.elements[0].color = colors[0]
    stripes.color_ramp.elements[1].position = 0.62
    stripes.color_ramp.elements[1].color = colors[1]
    nt.links.new(fract.outputs["Value"], stripes.inputs["Fac"])
    clumps = nt.nodes.new("ShaderNodeTexNoise")
    clumps.inputs["Scale"].default_value = 9.0
    clumps.inputs["Detail"].default_value = 2.0
    nt.links.new(geom.outputs["Position"], clumps.inputs["Vector"])
    clump_color = nt.nodes.new("ShaderNodeMixRGB")
    clump_color.inputs["Fac"].default_value = 0.35
    nt.links.new(clumps.outputs["Fac"], clump_color.inputs["Fac"])
    nt.links.new(stripes.outputs["Color"], clump_color.inputs["Color1"])
    clump_color.inputs["Color2"].default_value = colors[2]
    uv = nt.nodes.new("ShaderNodeTexCoord")
    along = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], along.inputs["Vector"])
    tip = nt.nodes.new("ShaderNodeMixRGB")
    nt.links.new(along.outputs["Y"], tip.inputs["Fac"])
    nt.links.new(clump_color.outputs["Color"], tip.inputs["Color1"])
    tip.inputs["Color2"].default_value = colors[3]
    nt.links.new(tip.outputs["Color"], bsdf.inputs["Base Color"])
    translucent = nt.nodes.new("ShaderNodeBsdfTranslucent")
    nt.links.new(tip.outputs["Color"], translucent.inputs["Color"])
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.28
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(translucent.outputs["BSDF"], mix.inputs[2])
    out = next(node for node in nt.nodes if node.bl_idname == "ShaderNodeOutputMaterial")
    for link in list(out.inputs["Surface"].links):
        nt.links.remove(link)
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def _blade(name, height, width, lean):
    import bmesh

    rows = 6
    bm = bmesh.new()
    for i in range(rows):
        t = i / (rows - 1)
        taper = 1.0 - 0.78 * t
        bm.verts.new((-width * 0.5 * taper, -lean * t * t, height * t))
        bm.verts.new((width * 0.5 * taper, -lean * t * t, height * t))
    bm.verts.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for i in range(rows - 1):
        face = bm.faces.new((
            bm.verts[i * 2], bm.verts[i * 2 + 1],
            bm.verts[i * 2 + 3], bm.verts[i * 2 + 2],
        ))
        for loop in face.loops:
            loop[uv_layer].uv = (0.0 if loop.vert.co.x < 0.0 else 1.0, loop.vert.co.z / height)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return bpy.data.objects.new(name, mesh)


def _rand_float(nodes, lo, hi, seed):
    node = nodes.new("FunctionNodeRandomValue")
    node.data_type = "FLOAT"
    next(sock for sock in node.inputs if sock.name == "Min" and sock.type == "VALUE").default_value = float(lo)
    next(sock for sock in node.inputs if sock.name == "Max" and sock.type == "VALUE").default_value = float(hi)
    node.inputs["Seed"].default_value = int(seed) % 100000
    return next(sock for sock in node.outputs if sock.type == "VALUE" and sock.enabled)


def _lawn(col, span, depth, seed):
    """Instanced bent blades on the apron. A full monocot tuft is a subsurf-9 plant; too heavy to scatter."""
    dark = (0.045, 0.16, 0.035, 1.0)
    mid = (0.09, 0.30, 0.06, 1.0)
    light = (0.16, 0.40, 0.09, 1.0)
    tip = (0.24, 0.46, 0.11, 1.0)
    dry = (0.32, 0.28, 0.10, 1.0)
    green = _grass_material(f"CWG_Grass_{seed & 0xFFFF:04x}", (dark, mid, light, tip))
    tan = _grass_material(f"CWG_GrassDry_{seed & 0xFFFF:04x}", (dry, dry, (0.36, 0.30, 0.10, 1.0), (0.42, 0.34, 0.12, 1.0)))
    # Short wide blades carpet the soil. Taller ones break the line where the wall meets the ground.
    shapes = (
        (0.032, 0.016, 0.026, 4),
        (0.048, 0.013, 0.038, 3),
        (0.07, 0.010, 0.05, 2),
        (0.10, 0.008, 0.06, 1),
    )
    sources = bpy.data.collections.new(f"CWG_GrassSources_{seed & 0xFFFF:04x}")
    blades = []
    for index, (height, width, lean, copies) in enumerate(shapes):
        for copy in range(copies):
            obj = _blade(f"CWG_Blade_{seed & 0xFFFF:04x}_{index}_{copy}", height, width, lean)
            obj.data.materials.append(green)
            sources.objects.link(obj)
            blades.append(obj)
    dry_blade = _blade(f"CWG_Blade_{seed & 0xFFFF:04x}_dry", 0.07, 0.008, 0.02)
    dry_blade.data.materials.append(tan)
    sources.objects.link(dry_blade)

    emitter = _grid("CWG_Outdoor_Lawn", span, depth, spacing=0.5, z=0.02)
    _adopt(emitter, col)
    ng = bpy.data.node_groups.new(f"CWG_Lawn_{seed & 0xFFFF:04x}", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = ng.nodes, ng.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    scatter = nodes.new("GeometryNodeDistributePointsOnFaces")
    scatter.distribute_method = "RANDOM"
    scatter.inputs["Density"].default_value = 2600.0
    scatter.inputs["Seed"].default_value = int(seed) % 100000
    links.new(group_in.outputs["Geometry"], scatter.inputs["Mesh"])
    info = nodes.new("GeometryNodeCollectionInfo")
    info.transform_space = "RELATIVE"
    info.inputs["Separate Children"].default_value = True
    info.inputs["Reset Children"].default_value = True
    info.inputs["Collection"].default_value = sources
    yaw = _rand_float(nodes, 0.0, math.tau, seed)
    lean = _rand_float(nodes, -0.45, 0.45, seed + 17)
    scale = _rand_float(nodes, 0.72, 1.25, seed + 29)
    euler = nodes.new("ShaderNodeCombineXYZ")
    links.new(lean, euler.inputs["X"])
    links.new(yaw, euler.inputs["Z"])
    rotation = nodes.new("FunctionNodeEulerToRotation")
    links.new(euler.outputs["Vector"], rotation.inputs["Euler"])
    sized = nodes.new("ShaderNodeCombineXYZ")
    links.new(scale, sized.inputs["X"])
    links.new(scale, sized.inputs["Y"])
    links.new(scale, sized.inputs["Z"])
    inst = nodes.new("GeometryNodeInstanceOnPoints")
    inst.inputs["Pick Instance"].default_value = True
    links.new(scatter.outputs["Points"], inst.inputs["Points"])
    links.new(info.outputs["Instances"], inst.inputs["Instance"])
    links.new(rotation.outputs["Rotation"], inst.inputs["Rotation"])
    links.new(sized.outputs["Vector"], inst.inputs["Scale"])
    links.new(inst.outputs["Instances"], group_out.inputs["Geometry"])
    group_in.location = (-600, 0)
    group_out.location = (600, 0)
    mod = emitter.modifiers.new("Lawn", "NODES")
    mod.node_group = ng
    mod.show_viewport = True
    mod.show_render = True
    return emitter


def _ground_surface(col, spec, plan: OutdoorPlan, material):
    """The part of the ground a facade camera actually sees, with real relief."""
    seed = _ground_seed(plan)
    span = spec.width_m + 8.0
    depth = 6.0
    kind = "dirt" if plan.ground == "lawn" else plan.ground
    spacing = {"cobble": 0.02, "concrete": 0.014, "dirt": 0.035, "mud": 0.03}[kind]
    apron = _grid("CWG_Outdoor_Apron", span, depth, spacing, z=0.02)
    _adopt(apron, col)
    apron.data.materials.append(material)
    if kind == "cobble":
        from cwg_infinigen.assets.materials.terrain.cobble_stone import geo_cobblestone

        _bake_displacement(apron, geo_cobblestone, {"random_seed": seed}, seed)
    elif kind == "concrete":
        _bake_displacement(apron, geo_concrete_ground, {}, seed)
    elif kind == "mud":
        _bake_displacement(apron, geo_relief, {"height": 0.035, "scale": 2.4, "detail": 3.0}, seed)
    elif plan.ground == "lawn":
        _bake_displacement(apron, geo_relief, {"height": 0.012, "scale": 2.2, "detail": 3.0}, seed)
    else:
        _bake_displacement(apron, geo_relief, {"height": 0.055, "scale": 3.6, "detail": 5.0}, seed)
    if apron.data.materials:
        apron.data.materials[0] = material
    else:
        apron.data.materials.append(material)
    built = [apron]
    if plan.ground == "lawn":
        # Blades only where a facade camera can see them. The thatch continues past that.
        built.append(_lawn(col, spec.width_m + 3.0, 3.2, seed))
    return built




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


def build_outdoor(spec, plan: OutdoorPlan, wall=None) -> list:
    """Ground, the rest of the facade, optional corner, eave and downpipe. Wall front y = 0, normal -Y.

    The test wall is one panel of a larger facade: the same material continues
    left, right and above it (the un-injected original, so defects stay on the
    test wall only), which is what a photographed building front looks like.
    """
    col = _collection(SHELL_COLLECTION)
    w, h, t = spec.width_m, spec.height_m, 0.12
    ground = _ground_material(plan)
    built = [_slab("CWG_Outdoor_Ground", (60.0, 40.0, t), (0.0, -19.0, -t * 0.5), ground, col)]
    built.extend(_ground_surface(col, spec, plan, ground))
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
