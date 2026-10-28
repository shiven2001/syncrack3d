"""Seeded background assemblies on the wall's X/Z face (outward is -Y).

These are occluding geometry, never injection hosts or crack annotations.
Windows are Infinigen casements on the wall face. The host is not cut, so
the glass is an opaque tint and the wall behind it does not show through.
Doors are closed
solid panel leaves on the wall face, with no extra surround
and no wall behind them that can crack (cracks keep out). The indoor electrical
plate is a UK double socket: two rockers together above two Type-G
outlets, in white, black, or satin metal. There is no separate switch. A wall cable or conduit is
one route that reaches a wall edge and may turn once or twice, drawn as
one to three runs side by side. Trunking is one duct, and plumbing
is a pair of pipes; each reaches a wall edge and may turn the same way.
A utility box feeds the same kind of
conduit up or down to that edge. Panel
seams are sealant joints, not simulated fractures.

Painted parts share one off-white enamel and metal parts share one satin,
both from the Infinigen plastic and brushed-metal shaders. A service hatch, a return vent, and a socket plate each pick their own
shell: that enamel, black, or the satin.
About half of the frames, boxes and trims carry a sealant bead where they
meet the wall - thin bright lines right beside where cracks run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

import numpy as np

# auto: indoor or outdoor kinds by environment; facade: openings and downpipes;
# all: every kind (geometry tests); off: none (fixtures ablation).
PROFILES = ("off", "auto", "facade", "all")
KINDS = ("cable", "wall_cable", "conduit", "downpipe", "utility_box", "window",
         "panel_seam", "service_hatch", "vent")
INDOOR_KINDS = ("socket", "door", "skirting", "ceiling_trim",
                "trunking", "plumbing")
OUTDOOR_KINDS = ("wall_light",)
KINDS += INDOOR_KINDS + OUTDOOR_KINDS
LARGE_KINDS = ("utility_box", "window", "door", "vent",
               "service_hatch", "socket", "wall_light")
SALT = 0x53545255


@dataclass(frozen=True)
class StructurePlan:
    index: int
    kind: str
    u_m: float
    v_m: float
    width_m: float
    height_m: float
    seed: int

    def to_dict(self):
        return asdict(self)


def sample_structure_plan(seed, spec, profile="auto", environment="indoor", include_kinds=None):
    """Fixture plans for one wall, from their own stream; planned before any crack."""
    if profile not in PROFILES:
        raise ValueError(f"unknown structure profile: {profile}")
    if profile == "off":
        return []
    r = random.Random(seed ^ SALT)
    if include_kinds is not None:
        if not include_kinds or any(k not in KINDS for k in include_kinds):
            raise ValueError("include_kinds must contain known structure kinds")
        kinds = list(dict.fromkeys(include_kinds))
    elif profile == "all":
        kinds = list(KINDS)
    else:
        indoor = ["wall_cable", "conduit", "utility_box", "window", "panel_seam",
                  "service_hatch", "vent", *INDOOR_KINDS]
        outdoor = ["cable", "wall_cable", "conduit", "downpipe", "utility_box", "window",
                   "panel_seam", "service_hatch", "vent", *OUTDOOR_KINDS]
        pool = (["window", "panel_seam", "downpipe", "service_hatch", "vent", *OUTDOOR_KINDS]
                if profile == "facade" else
                outdoor if environment == "outdoor" else indoor)
        kinds = r.sample(pool, r.randint(2, min(4, len(pool))))
    if include_kinds is None and profile != "facade" and environment == "indoor":
        # Rooms have several switched sockets. The rocker is on the socket plate.
        if "socket" in kinds:
            kinds += ["socket"] * r.randint(0, 2)
    # Floor-anchored doors and boundary trims reserve their footprints first.
    kinds.sort(key=lambda k: 0 if k == "door" else 1 if k in ("skirting", "ceiling_trim") else 2)
    plans = []
    for i, kind in enumerate(kinds):
        w, h = {
            "cable": (spec.width_m - .04, .55),
            "wall_cable": (.09, spec.height_m),
            "conduit": (.07, spec.height_m),
            "downpipe": (.13, spec.height_m),
            "utility_box": (r.uniform(.36, .46), r.uniform(.42, .56)),
            "window": (min(1.35, spec.width_m * .40), min(1.15, spec.height_m * .46)),
            "panel_seam": (spec.width_m, .006),
            "socket": (.146, .086),
            "door": (.94, min(2.08, spec.height_m - .20)),
            "skirting": (spec.width_m, .11), "ceiling_trim": (spec.width_m, .065),
            "vent": (side := r.uniform(.40, .50), side * r.uniform(.88, .98)),
            "trunking": (.08, spec.height_m), "plumbing": (.30, spec.height_m),
            "service_hatch": (r.uniform(.62, .80), min(r.uniform(1.18, 1.50), spec.height_m * .62)),
            "wall_light": (.12, .20),
        }[kind]
        if kind == "window":
            w *= r.uniform(.8, 1.15)
            h *= r.uniform(.8, 1.15)
        for attempt in range(60):
            x = r.uniform(-spec.width_m * .5 + w * .5, spec.width_m * .5 - w * .5)
            z = r.uniform(max(h * .5, .3), spec.height_m - max(h * .5, .15))
            if kind == "door":
                z = h * .5
            elif kind == "socket":
                z = r.uniform(.25, .45)
            elif kind == "window":
                # Sill about 0.9–1.1 m above the floor.
                z = h * .5 + r.uniform(.85, 1.05)
                if z + h * .5 > spec.height_m - .10:
                    z = spec.height_m - .10 - h * .5
            elif kind == "service_hatch":
                # A cabinet door standing above the floor, like a meter cupboard.
                z = h * .5 + r.uniform(.22, .48)
                if z + h * .5 > spec.height_m - .08:
                    z = spec.height_m - .08 - h * .5
            elif kind in ("wall_cable", "conduit", "trunking", "plumbing"):
                z = spec.height_m * .5
            elif kind == "cable":
                x = 0.
                # Ends sit at the wall edges. The sag has to clear a door leaf.
                door_top = max((q.v_m + q.height_m * .5 for q in plans if q.kind == "door"), default=0.)
                z = spec.height_m - .08
                if z - .22 < door_top + .06:
                    continue
            elif kind == "utility_box":
                lo, hi = h * .5 + .45, spec.height_m - h * .5 - .45
                z = r.uniform(lo, hi) if hi > lo + .05 else spec.height_m * .5
            elif kind == "skirting":
                x, z = 0., h * .5
            elif kind == "ceiling_trim":
                x, z = 0., spec.height_m - h * .5
            elif kind == "vent":
                # A return grille just above the skirting.
                z = h * .5 + r.uniform(.18, .30)
            elif kind == "wall_light":
                z = r.uniform(spec.height_m * .60, spec.height_m - h * .5 - .10)
            if kind in ("skirting", "ceiling_trim"):
                break
            if kind in ("trunking", "plumbing", "conduit", "wall_cable", "downpipe") and any(q.kind == "door" and
                    abs(x-q.u_m) < (w+q.width_m)*.5 and z-h*.5 < q.height_m for q in plans):
                continue
            if kind in ("downpipe", "wall_cable", "conduit", "trunking", "plumbing") and any(
                    q.kind in ("downpipe", "wall_cable", "conduit", "trunking", "plumbing") and
                    abs(x - q.u_m) < (w + q.width_m) * .5 + .06 for q in plans):
                continue
            if kind not in LARGE_KINDS:
                break
            if any(p.kind != "panel_seam" and
                   abs(x - p.u_m) < (w + p.width_m) * .5 + .05 and
                   abs(z - p.v_m) < (h + p.height_m) * .5 + .05 for p in plans):
                continue
            break
        else:
            continue  # no physically plausible space: report only built plans
        plans.append(StructurePlan(i, kind, x, z, w, h, r.randrange(2**31)))
    return plans


# Panel door leaf: (module under cwg_infinigen.assets.objects, factory, weight).
# Glazed and louvred leaves are not used: the wall is not cut, so a window or
# slit shows the host surface through the opening.
DOOR_MODELS = (("elements.doors.panel", "PanelDoorFactory", 1.),)
# Sealant beads where these meet the wall: which outline edges get one.
SEALED_EDGES = {"door": "lrt", "window": "lrtb", "utility_box": "lrtb", "vent": "lrtb",
                "service_hatch": "lrtb", "skirting": "t", "ceiling_trim": "b",
                "trunking": "tb"}
# The part whose outline meets the wall and is sealed (not feeds, hoods or louvres).
OUTLINE_PARTS = {"door": ("_casing",), "window": ("_frame",),
                 "skirting": ("_section",), "ceiling_trim": ("_section",), "trunking": ("_section",),
                 "utility_box": ("_housing",), "vent": ("_frame",),
                 "service_hatch": ("_frame",)}
_SEALANT_COLOURS = ((.93, .93, .91), (.88, .87, .82), (.80, .80, .79), (.86, .82, .70))


def _union_bounds(objects):
    from syncrack3d.occlusion import _world_bounds

    boxes = [b for o in objects if (b := _world_bounds(o)) is not None]
    return np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)


def _bead_paths(lo, hi, edges, radius, r, floor_z):
    """Polylines for a sealant bead hugging the x-z outline (lo, hi) on the given edges."""
    off, y = radius * .8, -radius * .6
    left, right, bottom, top = lo[0] - off, hi[0] + off, lo[2] - off, hi[2] + off
    runs = {"l": ((left, max(lo[2], floor_z)), (left, hi[2])), "r": ((right, max(lo[2], floor_z)), (right, hi[2])),
            "t": ((lo[0], top), (hi[0], top)), "b": ((lo[0], bottom), (hi[0], bottom))}
    paths = []
    for e in edges:
        (x0, z0), (x1, z1) = runs[e]
        n = max(2, int(math.hypot(x1 - x0, z1 - z0) / .12) + 1)
        horizontal = e in "tb"
        pts = []
        for t in range(n):
            f = t / (n - 1)
            jitter = r.uniform(-.0008, .0008) if 0 < t < n - 1 else 0.
            pts.append((x0 + (x1 - x0) * f + (0. if horizontal else jitter), y,
                        z0 + (z1 - z0) * f + (jitter if horizontal else 0.)))
        paths.append(pts)
    return paths


def _spawn_model(module, factory, seed, **kwargs):
    """An Infinigen model, or None if it cannot be built here."""
    from syncrack3d.occlusion import _load

    try:
        cls = _load("cwg_infinigen.assets.objects." + module, factory)
        obj = cls(seed, **kwargs).spawn_asset(i=0)
    except Exception:
        return None
    obj.parent = None
    return obj


def _fit_on_wall(obj, x, z, w, h, back_y, turn=True, grow=1.0):
    """Face a model out of the wall (-Y), scale it uniformly into w x h (at most `grow`x),
    centre it at (x, z) with its back at `back_y`."""
    import bpy
    from syncrack3d.occlusion import _world_bounds

    if turn:  # Infinigen models face +X
        obj.rotation_euler = (0.0, 0.0, -math.pi / 2)
    bpy.context.view_layer.update()
    lo, hi = _world_bounds(obj)
    s = min(w / (hi[0] - lo[0]), h / (hi[2] - lo[2]), grow)
    obj.scale = (s, s, s)
    bpy.context.view_layer.update()
    lo, hi = _world_bounds(obj)
    obj.location = (obj.location[0] + x - (lo[0] + hi[0]) / 2, obj.location[1] + back_y - hi[1],
                    obj.location[2] + z - (lo[2] + hi[2]) / 2)
    bpy.context.view_layer.update()
    return obj


def _architrave(name, x0, x1, top, width, proud, material, collection):
    """A door architrave on the wall: two legs and a head, mitred at the top corners.

    One closed profile (back on the wall at y=0, a flat face `proud` out with a
    small quirk on the inner edge and a rounded outer edge) is swept around the
    opening; each profile point keeps its own offset from the opening, so the
    corners meet in 45-degree mitres with no gap or overlap.
    """
    import bmesh
    import bpy

    r = proud * .45
    arc = [(width - r + r * math.sin(a), -proud + r - r * math.cos(a))
           for a in np.linspace(0., math.pi / 2, 7)]
    # (offset out from the opening, y); the arc runs from the face down to the wall
    profile = [(0., 0.), (0., -proud * .7), (.004, -proud * .7), (.004, -proud), *arc, (width, 0.)]
    bm = bmesh.new()
    rows = [[bm.verts.new(co) for co in ((x0 - a, y, 0.), (x0 - a, y, top + a), (x1 + a, y, top + a),
                                          (x1 + a, y, 0.))] for a, y in profile]
    smooth = set(range(4, 4 + len(arc) - 1))
    for i in range(len(rows)):
        nxt = rows[(i + 1) % len(rows)]
        for j in range(3):
            face = bm.faces.new((rows[i][j], rows[i][j + 1], nxt[j + 1], nxt[j]))
            face.smooth = i in smooth
    for j in (0, 3):
        bm.faces.new([row[j] for row in rows])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def _cut_behind(obj, y):
    """Delete a model's geometry behind the plane y (a back-side door handle inside the wall)."""
    import bmesh
    import bpy

    obj.data = obj.data.copy()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    m = obj.matrix_world
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if (m @ v.co).y > y + 1e-4], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    bpy.context.view_layer.update()


def _opaque_glazing(mat):
    """Replace a transmissive window pane with dark tinted glass.

    The wall is not cut, so a clear pane would show the host. A glossy
    gradient stays dark at the sill and lighter toward the sky, and it
    does not transmit.
    """
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.08
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.55
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.35
        bsdf.inputs["Coat Roughness"].default_value = 0.06
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    span = nt.nodes.new("ShaderNodeMapRange")
    span.clamp = True
    span.inputs["From Min"].default_value = 0.5
    span.inputs["From Max"].default_value = 2.5
    span.inputs["To Min"].default_value = 0.0
    span.inputs["To Max"].default_value = 1.0
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (0.04, 0.07, 0.09, 1)
    mix.inputs[7].default_value = (0.14, 0.24, 0.30, 1)
    nt.links.new(geo.outputs["Position"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], span.inputs["Value"])
    nt.links.new(span.outputs["Result"], mix.inputs[0])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"


# One product line. Enamel is the Infinigen rough-plastic shader at a fixed
# off-white; metal is the Infinigen brushed-metal shader at one satin. The
# seed only changes the fine grain, not the colour, so every fixture matches.
_ENAMEL = (0.88, 0.87, 0.84, 1.0)
_CHARCOAL = (0.045, 0.046, 0.048, 1.0)
_SATIN_HSV = (0.0, 0.012, 0.80)


def _fixture_finish(name, role, seed):
    """Shared enamel, charcoal or satin. Displacement stays a bump."""
    from cwg_infinigen.core.surface import shaderfunc_to_material
    from cwg_infinigen.core.util.math import FixedSeed

    with FixedSeed(int(seed) % (2 ** 31)):
        if role == "metal":
            from cwg_infinigen.assets.materials.metal.brushed_metal import (
                shader_brushed_metal,
            )
            mat = shaderfunc_to_material(
                shader_brushed_metal, name=name, scale=1.0,
                base_color_hsv=_SATIN_HSV, seed=float(seed % 997))
        else:
            from cwg_infinigen.assets.materials.plastic.plastic_rough import (
                shader_rough_plastic,
            )
            mat = shaderfunc_to_material(
                shader_rough_plastic, name=name,
                base_color=_ENAMEL if role == "enamel" else _CHARCOAL,
                roughness=0.40 if role == "enamel" else 0.55,
                scale=0.8, seed=float(seed % 997), clear=False,
                displacement_scale=0.0)
    if hasattr(mat, "displacement_method"):
        mat.displacement_method = "BUMP"
    for node in mat.node_tree.nodes:
        if node.type == "DISPLACEMENT" and "Scale" in node.inputs:
            node.inputs["Scale"].default_value = 0.0
    return mat


def _shell_finish(name, seed):
    """One shell for a hatch or a vent: white enamel, black, or satin metal."""
    role = ("enamel", "dark", "metal")[random.Random(int(seed) % (2 ** 31)).randrange(3)]
    mat = _fixture_finish(name, role, seed)
    mat["cwg_shell"] = role
    return mat


def _downpipe_plastic(name, tone):
    """PVC rainwater pipe: sheen, fine speckle, and vertical rain streaks."""
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*tone, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.34
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.46
    coord = nt.nodes.new("ShaderNodeNewGeometry")
    mapping = nt.nodes.new("ShaderNodeMapping")
    # World metres: a hard X/Y scale puts several rain marks around an
    # 80 mm pipe, and a mild Z scale keeps them running downward.
    mapping.inputs["Scale"].default_value = (55.0, 55.0, 1.4)
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 1.0
    grain.inputs["Detail"].default_value = 4.0
    grain.inputs["Roughness"].default_value = 0.45
    nt.links.new(coord.outputs["Position"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], grain.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.50
    ramp.color_ramp.elements[1].position = 0.78
    stain = (tone[0] * 0.55, tone[1] * 0.50, tone[2] * 0.42, 1.0)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (*tone, 1.0)
    mix.inputs[7].default_value = stain
    nt.links.new(grain.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs[0])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["From Min"].default_value = 0.15
    rough.inputs["From Max"].default_value = 0.85
    rough.inputs["To Min"].default_value = 0.22
    rough.inputs["To Max"].default_value = 0.55
    nt.links.new(grain.outputs["Fac"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.04
    bump.inputs["Distance"].default_value = 0.0008
    nt.links.new(grain.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _flexible_conduit(name, tone):
    """Dark flexible conduit: ribs running along a vertical drop."""
    from syncrack3d.materials import make_solid_material

    mat = make_solid_material(name, (*tone, 1), roughness=.4)
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    coord = nt.nodes.new("ShaderNodeNewGeometry")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 38.0
    wave.inputs["Distortion"].default_value = 0.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.45
    bump.inputs["Distance"].default_value = 0.0012
    nt.links.new(coord.outputs["Position"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def add_downpipe(collection, prefix, x, bottom, top, seed):
    """A round PVC downpipe: joint collars, stand-off clips, and an outlet shoe.

    The wall face is y=0 and the pipe stands in front of it. Returns the new
    objects, already in ``collection``.
    """
    import bpy
    from mathutils import Vector

    r = random.Random(seed)
    plastic = _downpipe_plastic(prefix + "_pvc", _ENAMEL[:3])
    metal = _fixture_finish(prefix + "_clip", "metal", seed)
    radius = r.uniform(0.036, 0.048)
    standoff = 0.026
    y = -(radius + standoff)
    made = []

    def _own(obj):
        for old in tuple(obj.users_collection):
            old.objects.unlink(obj)
        collection.objects.link(obj)
        for poly in obj.data.polygons:
            poly.use_smooth = True
        made.append(obj)
        return obj

    def cylinder(label, rad, depth, loc, mat, rotation=None):
        bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=rad, depth=depth, location=loc)
        obj = bpy.context.object
        obj.name = f"{prefix}_{label}"
        if rotation is not None:
            obj.rotation_euler = rotation
            bpy.ops.object.transform_apply(rotation=True)
        obj.data.materials.append(mat)
        return _own(obj)

    shoe_top = bottom + 0.11
    cylinder("run", radius, top - shoe_top, (x, y, (top + shoe_top) / 2), plastic)
    cylinder("head", radius * 1.16, 0.055, (x, y, top - 0.03), plastic)
    length = top - shoe_top
    if length > 2.0:
        joint = shoe_top + length * r.uniform(0.38, 0.55)
        cylinder("coupler", radius * 1.14, 0.07, (x, y, joint), plastic)
    # 112° shoe: one end meets the stack, the other discharges away from the wall.
    pipe_end = Vector((x, y, shoe_top - 0.02))
    outlet = Vector((x, y - 0.11, bottom + 0.01))
    shoe_dir = (outlet - pipe_end).normalized()
    shoe_rot = shoe_dir.to_track_quat("Z", "Y").to_euler()
    cylinder("shoe", radius * 1.06, (outlet - pipe_end).length, tuple((pipe_end + outlet) / 2), plastic, shoe_rot)
    bore = bpy.data.materials.new(prefix + "_bore")
    bore.use_nodes = True
    bore_bsdf = bore.node_tree.nodes["Principled BSDF"]
    bore_bsdf.inputs["Base Color"].default_value = (0.015, 0.015, 0.016, 1.0)
    bore_bsdf.inputs["Roughness"].default_value = 0.55
    cylinder("bore", radius * 0.78, 0.006, tuple(outlet + shoe_dir * 0.004), bore, shoe_rot)
    span_lo, span_hi = shoe_top + 0.22, top - 0.16
    count = max(3, int((span_hi - span_lo) / 0.80) + 1)
    for i in range(count):
        zz = span_lo + (span_hi - span_lo) * i / (count - 1)
        bpy.ops.mesh.primitive_cube_add(location=(x, -0.003, zz))
        plate = bpy.context.object
        plate.name = f"{prefix}_plate_{i}"
        plate.dimensions = (0.058, 0.006, 0.070)
        bpy.ops.object.transform_apply(scale=True)
        plate.data.materials.append(plastic)
        _own(plate)
        arm_depth = standoff - 0.006
        bpy.ops.mesh.primitive_cube_add(location=(x, -(0.006 + standoff) / 2, zz))
        arm = bpy.context.object
        arm.name = f"{prefix}_arm_{i}"
        arm.dimensions = (0.020, arm_depth, 0.016)
        bpy.ops.object.transform_apply(scale=True)
        arm.data.materials.append(plastic)
        _own(arm)
        cylinder("strap", radius * 1.10, 0.016, (x, y, zz), metal)
    return made


def _segment_hits_door(a, b, doors, margin=.07):
    x0, x1 = sorted((a[0], b[0]))
    z0, z1 = sorted((a[1], b[1]))
    for u, v, w, h in doors:
        if x1 < u - w * .5 - margin or x0 > u + w * .5 + margin:
            continue
        if z1 < v - h * .5 - margin or z0 > v + h * .5 + margin:
            continue
        return True
    return False


def _chamfer(pts, cut):
    """Pull each corner back so a tube bends instead of kinking."""
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        cx, cz = pts[i]
        for ax, az in (pts[i - 1], pts[i + 1]):
            dx, dz = ax - cx, az - cz
            length = math.hypot(dx, dz)
            if length < 1e-6:
                out.append((cx, cz))
                continue
            t = min(cut, length * .4)
            out.append((cx + dx * t / length, cz + dz * t / length))
    out.append(pts[-1])
    return out


def _along(pts, step, inset):
    """(x, z, horizontal) sample points along each straight leg."""
    spots = []
    for a, b in zip(pts, pts[1:]):
        dx, dz = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dz)
        if length < inset * 2 + .05:
            continue
        horizontal = abs(dx) >= abs(dz)
        count = max(1, int((length - 2 * inset) / step))
        for k in range(count):
            t = (inset + (length - 2 * inset) * (k + .5) / count) / length
            spots.append((a[0] + dx * t, a[1] + dz * t, horizontal))
    return spots


def _offset_route(pts, dist):
    """Shift an axis-aligned route sideways so a copy stays parallel."""
    if len(pts) < 2 or abs(dist) < 1e-9:
        return [tuple(p) for p in pts]

    def left(a, b):
        dx, dz = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dz) or 1.0
        return (-dz / length, dx / length)

    normals = [left(a, b) for a, b in zip(pts, pts[1:])]
    out = [(pts[0][0] + normals[0][0] * dist, pts[0][1] + normals[0][1] * dist)]
    for i in range(1, len(pts) - 1):
        n0, n1 = normals[i - 1], normals[i]
        dot = n0[0] * n1[0] + n0[1] * n1[1]
        denom = 1.0 + dot
        if abs(denom) < 1e-4:
            ox, oz = n1[0] * dist, n1[1] * dist
        else:
            scale = dist / denom
            ox, oz = (n0[0] + n1[0]) * scale, (n0[1] + n1[1]) * scale
        out.append((pts[i][0] + ox, pts[i][1] + oz))
    n = normals[-1]
    out.append((pts[-1][0] + n[0] * dist, pts[-1][1] + n[1] * dist))
    return out


def _parallel(pts, count, gap):
    """One to three copies of the same route, centred on it."""
    if count <= 1:
        return [list(pts)]
    mid = (count - 1) / 2.0
    return [_offset_route(pts, (i - mid) * gap) for i in range(count)]


def _service_bundle(r, hint_x, wall_w, wall_h, doors, gap, count=None):
    """One edge-to-edge route, drawn as `count` parallel runs (1–3 if omitted)."""
    left, right = -wall_w * .5 + .03, wall_w * .5 - .03
    bottom, top = .02, wall_h - .02
    if count is None:
        count = r.choice((1, 2, 2, 3))
    reach = (count - 1) * gap * .5 + .04

    def fits(pts):
        copies = _parallel(pts, count, gap)
        for route in copies:
            if any(_segment_hits_door(a, b, doors) for a, b in zip(route, route[1:])):
                return None
        return copies

    if right - left < .4 or top - bottom < .5:
        x = min(right - reach, max(left + reach, hint_x))
        pts = [(x, bottom), (x, top)]
        return pts, fits(pts) or [pts]
    for _ in range(16):
        style = r.choice(("up", "up", "across", "elbow", "elbow", "zee"))
        if style == "up":
            lo, hi = left + reach, right - reach
            if hi <= lo + .05:
                continue
            x = r.uniform(lo, hi)
            pts = [(x, bottom), (x, top)]
        elif style == "across":
            door_top = max((v + h * .5 for _, v, _, h in doors), default=0.)
            lo, hi = max(.4, door_top + reach), top - reach
            if hi <= lo + .05:
                continue
            z = r.uniform(lo, hi)
            pts = [(left, z), (right, z)]
        elif style == "elbow":
            x0, x1 = left + .25 + reach, right - .25 - reach
            z0, z1 = bottom + .4 + reach, top - .28 - reach
            if x1 <= x0 + .05 or z1 <= z0 + .05:
                continue
            x, z = r.uniform(x0, x1), r.uniform(z0, z1)
            side = left if r.random() < .5 else right
            end = bottom if r.random() < .5 else top
            pts = [(x, end), (x, z), (side, z)]
        else:
            x0 = left + .12 + reach
            x_mid = (left + right) * .5 - .1
            if x_mid <= x0 + .05:
                continue
            x1 = r.uniform(x0, x_mid)
            x2_hi = right - .12 - reach
            if x2_hi <= x1 + .35:
                continue
            x2 = r.uniform(x1 + .35, x2_hi)
            z0, z1 = bottom + .45 + reach, top - .35 - reach
            if z1 <= z0 + .05:
                continue
            z = r.uniform(z0, z1)
            pts = [(x1, bottom), (x1, z), (x2, z), (x2, top)]
        copies = fits(pts)
        if copies is not None:
            return pts, copies
    x = min(right - reach, max(left + reach, hint_x))
    pts = [(x, bottom), (x, top)]
    return pts, fits(pts) or [pts]


def build_structures(plans, relief_m=0.0, wall_height_m=None, wall_width_m=None):
    import bpy
    from mathutils import Vector
    from syncrack3d.geometry import _collection, _box
    from syncrack3d.materials import make_solid_material
    from syncrack3d.occlusion import _world_bounds

    col = _collection("CWG_STRUCTURES")
    if wall_height_m is None:
        tops = [p.v_m + p.height_m * .5 for p in plans if p.kind in ("wall_cable", "conduit", "downpipe")]
        wall_height_m = max(tops) if tops else 2.6
    if wall_width_m is None:
        spans = [p.width_m for p in plans if p.kind in ("skirting", "ceiling_trim", "panel_seam")]
        reach = max((abs(p.u_m) + p.width_m * .5 for p in plans), default=1.6)
        wall_width_m = max(spans) if spans else max(3.2, reach * 2 + .6)
    records = []
    plans = list(plans)
    old_skirting = bpy.data.objects.get("CWG_Skirting")
    if (old_skirting is not None and any(p.kind == "door" for p in plans)
            and not any(p.kind == "skirting" for p in plans)):
        # Replace the base wall's uninterrupted board with sections around
        # the door, even when skirting was not explicitly sampled.
        plans.append(StructurePlan(max(p.index for p in plans)+1, "skirting",
                                   0., .0425, float(old_skirting.dimensions.x), .085,
                                   next(p.seed for p in plans if p.kind == "door") ^ SALT))
    if any(p.kind == "skirting" for p in plans):
        old = bpy.data.objects.get("CWG_Skirting")
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
    for p in plans:
        r = random.Random(p.seed)
        prefix = f"CWG_Structure_{p.index:02d}_{p.kind}"
        root = bpy.data.objects.new(prefix, None)
        col.objects.link(root)
        objects = []
        fixture_model = None
        paint = _fixture_finish(prefix + '_paint', 'enamel', p.seed)
        dark = _fixture_finish(prefix + '_rubber', 'dark', p.seed ^ 0x51)
        metal = _fixture_finish(prefix + '_metal', 'metal', p.seed ^ 0xA3)

        def tag(obj):
            obj["cwg_role"] = "structure"
            obj["cwg_hard_negative"] = p.kind
            obj["cwg_structure_index"] = p.index
            obj.parent = root
            objects.append(obj)
            return obj

        def box(label, dims, loc, mat=paint, bevel_m=None):
            obj = tag(_box(prefix + '_' + label, dims, loc, mat, col, root))
            bevel = obj.modifiers.new('Soft manufactured edges', 'BEVEL')
            bevel.width = min(.003, min(dims) * .15) if bevel_m is None else bevel_m
            bevel.segments = 2
            return obj

        def tube(label, points, radius, mat=paint, smooth=False):
            curve = bpy.data.curves.new(prefix + '_' + label, 'CURVE')
            curve.dimensions = '3D'
            curve.bevel_depth = radius
            curve.bevel_resolution = 3
            curve.resolution_u = 16
            curve.use_fill_caps = True
            spline = curve.splines.new('BEZIER' if smooth else 'POLY')
            if smooth:
                spline.bezier_points.add(len(points) - 1)
                for point, co in zip(spline.bezier_points, points):
                    point.co = co
                    point.handle_left_type = point.handle_right_type = 'AUTO'
            else:
                spline.points.add(len(points) - 1)
                for point, co in zip(spline.points, points):
                    point.co = (*co, 1)
            obj = bpy.data.objects.new(curve.name, curve)
            col.objects.link(obj)
            obj.data.materials.append(mat)
            return tag(obj)

        x, z, w, h = p.u_m, p.v_m, p.width_m, p.height_m
        if p.kind == 'cable':
            # A suspended span from one edge of the wall to the other.
            radius = r.uniform(.003, .006)
            points = [(x + w * (t / 12 - .5), -.07 - .035 * math.sin(math.pi * t / 12),
                       z - .18 * math.sin(math.pi * t / 12)) for t in range(13)]
            tube('sag', points, radius, dark, True)
            for end in (points[0], points[-1]):
                box('anchor', (.04, .055, .04), (end[0], -.03, end[2]), metal)
        elif p.kind == 'wall_cable':
            doors = [(q.u_m, q.v_m, q.width_m, q.height_m) for q in plans if q.kind == 'door']
            radius = r.uniform(.003, .006)
            gap = radius * 2 + r.uniform(.004, .008)
            centre, routes = _service_bundle(r, x, wall_width_m, wall_height_m, doors, gap)
            span = (len(routes) - 1) * gap + radius * 2 + .006
            for route in routes:
                bent = _chamfer(route, r.uniform(.05, .09))
                tube('wire', [(px, -.016, pz) for px, pz in bent], radius, dark, True)
            for px, pz, horizontal in _along(centre, .55, .14):
                dims = (.012, .02, span) if horizontal else (span, .02, .012)
                box('clip', dims, (px, -.012, pz), metal)
        elif p.kind == 'conduit':
            doors = [(q.u_m, q.v_m, q.width_m, q.height_m) for q in plans if q.kind == 'door']
            radius = r.uniform(.011, .016)
            gap = radius * 2 + r.uniform(.006, .012)
            centre, routes = _service_bundle(r, x, wall_width_m, wall_height_m, doors, gap)
            y = -(radius + .02)
            depth = abs(y) - .008
            span = (len(routes) - 1) * gap + radius * 2 + .012
            for route in routes:
                bent = _chamfer(route, max(.06, radius * 3.5))
                tube('run', [(px, y, pz) for px, pz in bent], radius, paint)
                for px, pz, horizontal in _along(route, .7, .18):
                    if horizontal:
                        tube('collar', [(px - .014, y, pz), (px + .014, y, pz)], radius * 1.08, metal)
                    else:
                        tube('collar', [(px, y, pz - .014), (px, y, pz + .014)], radius * 1.08, metal)
            for px, pz, horizontal in _along(centre, .7, .18):
                dims = (.018, depth, span) if horizontal else (span, depth, .018)
                box('bracket', dims, (px, y * .5 - .004, pz), metal)
        elif p.kind == 'downpipe':
            bottom, top = max(.04, z - h * .5), z + h * .5
            for obj in add_downpipe(col, prefix, x, bottom, top, p.seed):
                tag(obj)
        elif p.kind == 'utility_box':
            # Powder-coated control cabinet. A smoked window shows the lamps,
            # and flexible conduits leave the top or the bottom and run on to
            # the edge of the wall.
            glass = make_solid_material(prefix + '_window', (.03, .04, .045, 1), roughness=.14)
            gbsdf = glass.node_tree.nodes['Principled BSDF']
            if 'Specular IOR Level' in gbsdf.inputs:
                gbsdf.inputs['Specular IOR Level'].default_value = .5
            depth = .11
            box('housing', (w, depth, h), (x, -.004 - depth * .5, z), paint)
            box('cap', (w + .028, depth + .012, .016),
                (x, -.008 - (depth + .012) * .5, z + h * .5 + .004), paint)
            face = -.004 - depth
            box('lid', (w - .028, .01, h - .05), (x, face - .006, z - .006), paint)
            win_w, win_h = w * .62, h * .52
            box('bezel', (win_w + .02, .006, win_h + .02), (x, face - .012, z + .01), dark)
            box('glazing', (win_w, .005, win_h), (x, face - .016, z + .01), glass)
            lamps = ((-.012, .05, (0.15, 0.45, 0.9)), (0., .01, (0.15, 0.72, 0.28)),
                     (.012, -.03, (0.78, 0.12, 0.08)))
            for i, (dx, dz, colour) in enumerate(lamps):
                lamp_mat = make_solid_material(prefix + f'_lamp_{i}', (*colour, 1), roughness=.35)
                bpy.ops.mesh.primitive_cylinder_add(
                    vertices=20, radius=.007, depth=.006,
                    location=(x + dx, face - .02, z + .01 + dz))
                lamp = bpy.context.object
                lamp.rotation_euler.x = math.pi / 2
                bpy.ops.object.transform_apply(rotation=True)
                lamp.name = prefix + f'_lamp_{i}'
                lamp.data.materials.append(lamp_mat)
                for old in tuple(lamp.users_collection):
                    old.objects.unlink(lamp)
                col.objects.link(lamp)
                tag(lamp)
            knob = dark
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=20, radius=.008, depth=.008,
                location=(x, face - .021, z + .01 - .09))
            knob_obj = bpy.context.object
            knob_obj.rotation_euler.x = math.pi / 2
            bpy.ops.object.transform_apply(rotation=True)
            knob_obj.name = prefix + '_knob'
            knob_obj.data.materials.append(knob)
            for old in tuple(knob_obj.users_collection):
                old.objects.unlink(knob_obj)
            col.objects.link(knob_obj)
            tag(knob_obj)
            screw = metal
            for dx in (-w * .42, w * .42):
                for dz in (-h * .40, h * .40):
                    bpy.ops.mesh.primitive_cylinder_add(
                        vertices=16, radius=.0045, depth=.004,
                        location=(x + dx, face - .012, z + dz))
                    head = bpy.context.object
                    head.rotation_euler.x = math.pi / 2
                    bpy.ops.object.transform_apply(rotation=True)
                    head.name = prefix + '_screw'
                    head.data.materials.append(screw)
                    for old in tuple(head.users_collection):
                        old.objects.unlink(head)
                    col.objects.link(head)
                    tag(head)
            up = r.random() < .4
            n_pipes = 3
            conduit_mat = _flexible_conduit(prefix + '_conduit_plastic', _CHARCOAL[:3])
            exit_z = z + h * .5 - .01 if up else z - h * .5 + .01
            end_z = wall_height_m if up else 0.
            sign = 1. if up else -1.
            spacing, pipe_r, pipe_y = .032, .009, -.07
            for i in range(n_pipes):
                xx = x + (i - (n_pipes - 1) * .5) * spacing
                tube('conduit', [(xx, face + .02, exit_z),
                                 (xx, pipe_y, exit_z + sign * .06),
                                 (xx, pipe_y, end_z)], pipe_r, conduit_mat, True)
            band_z = exit_z + sign * min(.28, abs(end_z - exit_z) * .4)
            box('strap', (spacing * (n_pipes - 1) + .03, .012, .016), (x, pipe_y, band_z), metal)
        elif p.kind == 'window':
            # Infinigen casement on the wall face. No wall cutout: the factory
            # glass is swapped for an opaque tint so the host does not show.
            leaf = None
            try:
                from syncrack3d.occlusion import _load

                cls = _load("cwg_infinigen.assets.objects.windows.window", "WindowFactory")
                factory = cls(p.seed % 2**31, curtain=False, shutter=False, open=False)
                leaf = factory.spawn_asset(i=0, dimensions=(w, h, r.uniform(.045, .06)))
            except Exception:
                leaf = None
            if leaf is not None:
                leaf.name = prefix + '_frame'
                for child in list(leaf.children):
                    bpy.data.objects.remove(child, do_unlink=True)
                for old in tuple(leaf.users_collection):
                    old.objects.unlink(leaf)
                col.objects.link(leaf)
                for mat in leaf.data.materials:
                    if mat is not None and mat.node_tree is not None and any(
                            n.type == 'BSDF_TRANSPARENT' for n in mat.node_tree.nodes):
                        _opaque_glazing(mat)
                _fit_on_wall(leaf, x, z, w, h, -.002, turn=False, grow=1.05)
                tag(leaf)
                fixture_model = 'WindowFactory'
            else:
                fw, fd = .048, .04
                fy = -.005 - fd * .5
                frame = metal
                glass = make_solid_material(prefix + '_opaque_glazing', (.10, .16, .20, 1), roughness=.1)
                box('frame_rail', (w, fd, fw), (x, fy, z + h * .5 - fw * .5), frame)
                box('frame_rail', (w, fd, fw), (x, fy, z - h * .5 + fw * .5), frame)
                box('frame_jamb', (fw, fd, h), (x - w * .5 + fw * .5, fy, z), frame)
                box('frame_jamb', (fw, fd, h), (x + w * .5 - fw * .5, fy, z), frame)
                box('pane', (w - 2 * fw + .008, .006, h - 2 * fw + .008),
                    (x, fy - fd * .5 + .012, z), glass)
                fixture_model = 'opaque_glazing_proxy'
            # Projecting sill under the casement, a little wider than the frame,
            # with a metal drip on the front underside. The junction anchors
            # sit just below this bottom edge.
            depth, thick, ear = .18, .08, .08
            bottom = z - h * .5 - .0675
            back = -.006
            box('sill', (w + 2 * ear, depth, thick), (x, back - depth * .5, bottom + thick * .5))
            box('drip_edge', (w + 2 * ear - .01, .014, .016),
                (x, back - depth + .008, bottom + .008), metal)
        elif p.kind == 'panel_seam':
            # Opaque sealant in a narrow construction joint: never a crack field.
            box('horizontal_sealant', (w, .004, h), (x, -.004, z), dark)
            if r.random() < .7:
                box('vertical_sealant', (h, .004, max(.4, z)), (x + w * .22, -.004, max(.4, z) * .5), dark)
        elif p.kind == 'socket':
            # UK double switched socket. Two rockers sit together above
            # two Type-G outlets; a screw is on each side of the plate.
            # The plate and rockers share one shell: white, black, or satin.
            shell = _shell_finish(prefix + '_shell', p.seed)
            thick, back = .008, -.003
            front = back - thick
            box('faceplate', (w, thick, h), (x, back - thick * .5, z), shell)
            # The dark aperture is only a hairline. A wide recess reads as a
            # black frame around the rockers.
            rw, rh, gap, rz = .0144, .0154, .0004, z + .020
            span = rw + gap
            box('switch_recess', (span + rw + .0005, .0006, rh + .0005),
                (x, front - .0003, rz), dark, bevel_m=.00008)
            for sx in (-span * .5, span * .5):
                box('rocker', (rw, .0032, rh), (x + sx, front - .0020, rz), shell, bevel_m=.0001)
            for sx in (-.034, .034):
                box('pin', (.0036, .002, .008), (x + sx, front - .0012, z - .002), dark)
                box('pin', (.0036, .002, .006), (x + sx - .011, front - .0012, z - .016), dark)
                box('pin', (.0036, .002, .006), (x + sx + .011, front - .0012, z - .016), dark)
            for sx in (-w * .5 + .012, w * .5 - .012):
                bpy.ops.mesh.primitive_cylinder_add(
                    vertices=16, radius=.0022, depth=.0016,
                    location=(x + sx, front - .0008, z + .006))
                screw = bpy.context.object
                screw.rotation_euler.x = math.pi / 2
                bpy.ops.object.transform_apply(rotation=True)
                screw.name = prefix + '_screw'
                screw.data.materials.append(dark)
                for old in tuple(screw.users_collection):
                    old.objects.unlink(screw)
                col.objects.link(screw)
                tag(screw)
                box('screw_slot', (.0024, .0004, .0005), (x + sx, front - .0017, z + .006), dark)
            fixture_model = 'double_switched_socket'
        elif p.kind == 'door':
            # A closed door set in its opening: an architrave of the leaf's own
            # wood stands on the wall around it and laps over the leaf's edges,
            # so no gap to the wall shows; the leaf face sits a few mm in front
            # of the wall plane, behind the architrave face. The wall is not cut,
            # so the back of the leaf is removed. The model's 4 cm hinge stile
            # (+x) and the slit beside it go under the architrave leg.
            trim, proud, lap, stile = r.uniform(.065, .09), r.uniform(.024, .03), .015, .045
            x0, x1, top = x - w * .5 + trim, x + w * .5 - trim, h - trim
            leaf_w, leaf_h, undercut = x1 - x0 + lap + stile, top + lap - .008, .008
            leaf_x = (x0 - lap + x1 + stile) * .5
            module, factory, _w = r.choices(DOOR_MODELS, weights=[m[2] for m in DOOR_MODELS], k=1)[0]
            from cwg_infinigen.core.constraints.constraint_language.constants import RoomConstants
            from cwg_infinigen.core.util.math import FixedSeed

            with FixedSeed(p.seed):
                constants = RoomConstants()
                # the model adds half its 4 cm hinge stile beyond door_width
                constants.door_width, constants.door_size, constants.wall_thickness = leaf_w - .02, leaf_h, .14
                leaf = _spawn_model(module, factory, p.seed % 2**31, constants=constants)
            if leaf is not None:
                leaf.name = prefix + '_leaf'
                for old in tuple(leaf.users_collection):
                    old.objects.unlink(leaf)
                col.objects.link(leaf)
                _fit_on_wall(leaf, leaf_x, undercut + leaf_h * .5, leaf_w, leaf_h, 0., turn=False, grow=1.0)
                # Bring the leaf's main face (stiles and rails; raised panels stand
                # a few mm proud of it) to just in front of the wall, and drop the
                # back half and its handle, which would sit inside the wall.
                wood = {i for i, slot in enumerate(leaf.material_slots)
                        if slot.material is not None and 'metal' not in slot.material.name}
                m = np.array(leaf.matrix_world)
                faces = [f for f in leaf.data.polygons if f.material_index in wood]
                ys = np.array([(m[:3, :3] @ np.array(f.center) + m[:3, 3])[1] for f in faces])
                front = np.array([(m[:3, :3] @ np.array(f.normal))[1] < -.9 for f in faces])
                area = np.array([f.area for f in faces])
                order = np.argsort(ys[front])
                cum = np.cumsum(area[front][order])
                face_y = ys[front][order][np.searchsorted(cum, cum[-1] * .5)]  # area-weighted median
                leaf.location.y += -.008 - face_y
                bpy.context.view_layer.update()
                _cut_behind(leaf, -.0005)
                tag(leaf)
                face = leaf.material_slots[max(wood, key=lambda i: sum(
                    f.area for f in leaf.data.polygons if f.material_index == i))].material
                fixture_model = factory
            else:
                face = make_solid_material(prefix+'_leaf', (.22, .13, .065, 1), roughness=.6)
                box('closed_leaf', (leaf_w, .004, leaf_h), (leaf_x, -.006, undercut + leaf_h * .5), face)
                hx, hz = x0+.07, .98
                box('handle_plate', (.035, .012, .12), (hx, -.014, hz), metal)
                tube('handle_stem', [(hx, -.020, hz), (hx, -.065, hz)], .008, metal)
                tube('lever', [(hx, -.065, hz), (hx+.10, -.065, hz)], .008, metal)
                fixture_model = 'closed_door_proxy'
            # Behind the leaf: moulding grooves cut deeper than its offset show wood, not wall.
            box('leaf_backing', (leaf_w, .001, leaf_h), (leaf_x, -.0008, undercut + leaf_h * .5), face)
            tag(_architrave(prefix + '_casing', x0, x1, top, trim, proud, face, col))
        elif p.kind in ('skirting', 'ceiling_trim'):
            # Skirting is a floor board with its bead on the top edge. Ceiling
            # trim is that same profile turned over, so the bead hangs down.
            depth = .018
            # Small open butt joints between boards; split around door jambs.
            cuts = [(x-w*.5, x+w*.5)]
            if p.kind == 'skirting':
                for door in (q for q in plans if q.kind == 'door'):
                    left, right = door.u_m-door.width_m*.5, door.u_m+door.width_m*.5
                    cuts = [(a, b) for lo, hi in cuts for a, b in
                            ((lo, min(hi, left)), (max(lo, right), hi)) if b-a > .01]
            for lo, hi in cuts:
                count = max(1, math.ceil((hi-lo)/1.4))
                length = (hi-lo)/count
                for j in range(count):
                    cx = lo+(j+.5)*length
                    box('joint_backing', (length, .003, h), (cx, -.002, z), dark)
                    box('section', (length-.002, depth, h), (cx, -depth*.5-.004, z))
                    # +z is the top of a skirting board. Ceiling trim flips it.
                    bead = h * .28 if p.kind == 'skirting' else -h * .28
                    box('moulding', (length-.003, .008, h*.18), (cx, -depth-.004, z+bead))
        elif p.kind == 'trunking':
            # One rectangular duct from a wall edge to a wall edge. A turn is
            # a right angle; the incoming length owns the corner.
            doors = [(q.u_m, q.v_m, q.width_m, q.height_m) for q in plans if q.kind == 'door']
            duct, depth, half = .055, .028, .0275
            centre, _routes = _service_bundle(r, x, wall_width_m, wall_height_m, doors, duct, count=1)
            y = -.004 - depth * .5

            def _advance(a, b, dist):
                dx, dz = b[0] - a[0], b[1] - a[1]
                length = math.hypot(dx, dz) or 1.0
                return (a[0] + dx / length * dist, a[1] + dz / length * dist)

            legs = [(a, b) for a, b in zip(centre, centre[1:])
                    if math.hypot(b[0] - a[0], b[1] - a[1]) > .05]
            for i, (a, b) in enumerate(legs):
                span = math.hypot(b[0] - a[0], b[1] - a[1])
                a2 = _advance(a, b, half) if i else a
                b2 = _advance(a, b, span + half) if i < len(legs) - 1 else b
                dx, dz = b2[0] - a2[0], b2[1] - a2[1]
                length = math.hypot(dx, dz)
                if length < .02:
                    continue
                cx, cz = (a2[0] + b2[0]) * .5, (a2[1] + b2[1]) * .5
                horizontal = abs(dx) >= abs(dz)
                dims = (length, depth, duct) if horizontal else (duct, depth, length)
                box('section', dims, (cx, y, cz))
                lid = (max(.01, length - .006), .004, duct - .008) if horizontal else (
                    duct - .008, .004, max(.01, length - .006))
                box('lid', lid, (cx, -depth - .006, cz))
            for px, pz, horizontal in _along(centre, 1.25, .22):
                band = (.008, depth + .006, duct + .008) if horizontal else (duct + .008, depth + .006, .008)
                box('joint', band, (px, y - .003, pz), dark)
        elif p.kind == 'service_hatch':
            # Flush cabinet door: a thin frame, a shadow gap, and one chrome
            # cam lock. The shell is white, black, or satin metal.
            powder = _shell_finish(prefix + '_shell', p.seed)
            chrome = make_solid_material(prefix + '_lock', (.90, .91, .93, 1), roughness=.22)
            chrome.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 1.
            flange, proud = .028, .014
            box('frame', (w, proud, flange), (x, -proud * .5, z + h * .5 - flange * .5), powder)
            box('frame', (w, proud, flange), (x, -proud * .5, z - h * .5 + flange * .5), powder)
            box('frame', (flange, proud, h - 2 * flange), (x - w * .5 + flange * .5, -proud * .5, z), powder)
            box('frame', (flange, proud, h - 2 * flange), (x + w * .5 - flange * .5, -proud * .5, z), powder)
            box('reveal', (w - 2 * flange, .004, h - 2 * flange), (x, -.002, z), dark)
            door_w, door_h, door_t = w - 2 * flange - .008, h - 2 * flange - .008, .006
            door_outer = -proud + .001
            box('door_panel', (door_w, door_t, door_h), (x, door_outer + door_t * .5, z), powder)
            lock_x, lock_z = x - door_w * .36, z + door_h * .04
            # A mesh rose. A short curve bevel disappears in the render.
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=32, radius=.013, depth=.007,
                location=(lock_x, door_outer - .0035, lock_z))
            rose = bpy.context.object
            rose.rotation_euler.x = math.pi / 2
            bpy.ops.object.transform_apply(rotation=True)
            rose.name = prefix + '_lock'
            rose.data.materials.append(chrome)
            for old in tuple(rose.users_collection):
                old.objects.unlink(rose)
            col.objects.link(rose)
            tag(rose)
            box('lock_key', (.002, .0012, .007), (lock_x, door_outer - .0072, lock_z), dark)
        elif p.kind == 'vent':
            # Return grille. A flat frame, a dark opening, and horizontal
            # louvres tilted down toward the room. The shell matches the hatch
            # choices: white, black, or satin metal.
            shell = _shell_finish(prefix + '_shell', p.seed)
            border, proud = .034, .016
            face = -.004 - proud * .5
            box('frame', (w, proud, border), (x, face, z + h * .5 - border * .5), shell)
            box('frame', (w, proud, border), (x, face, z - h * .5 + border * .5), shell)
            box('frame', (border, proud, h - 2 * border),
                (x - w * .5 + border * .5, face, z), shell)
            box('frame', (border, proud, h - 2 * border),
                (x + w * .5 - border * .5, face, z), shell)
            inner_w, inner_h = w - 2 * border, h - 2 * border
            box('recess', (inner_w, .006, inner_h), (x, -.003, z), dark)
            count = max(7, int(inner_h / .030))
            usable = inner_h - .016
            for j in range(count):
                zz = z - usable * .5 + usable * (j + .5) / count
                blade = box('louvre', (inner_w - .008, .022, .007), (x, face - .006, zz), shell)
                blade.rotation_euler.x = math.radians(-22)
        elif p.kind == 'plumbing':
            # A pair of pipes on one edge-to-edge route. They stay parallel
            # through a right-angle bend, with a valve on one of them.
            doors = [(q.u_m, q.v_m, q.width_m, q.height_m) for q in plans if q.kind == 'door']
            radius, gap, y = .011, r.uniform(.14, .22), -.045
            centre, routes = _service_bundle(
                r, x, wall_width_m, wall_height_m, doors, gap, count=2)
            cut = r.uniform(.06, .10)
            for route in routes:
                bent = _chamfer(route, cut)
                tube('pipe', [(px, y, pz) for px, pz in bent], radius, metal, True)
                for px, pz, horizontal in _along(route, .7, .2):
                    dims = (.018, .04, .04) if horizontal else (.04, .04, .018)
                    box('wall_clip', dims, (px, -.026, pz))
                    if horizontal:
                        tube('union', [(px - .016, y, pz), (px + .016, y, pz)], .016, metal)
                    else:
                        tube('union', [(px, y, pz - .016), (px, y, pz + .016)], .016, metal)
            spots = _along(routes[-1], 1.1, .3) or _along(centre, 1.1, .2)
            if spots:
                px, pz, horizontal = spots[len(spots) // 2]
                if horizontal:
                    tube('valve_body', [(px - .03, y, pz), (px + .03, y, pz)], .023, metal)
                else:
                    tube('valve_body', [(px, y, pz - .03), (px, y, pz + .03)], .023, metal)
                tube('valve_stem', [(px, y, pz), (px, y - .05, pz)], .006, metal)
                box('valve_lever', (.10, .014, .016), (px + .04, y - .05, pz), dark)
        elif p.kind == 'wall_light':
            # Matte cylinder, open at both ends, washing the wall above and below.
            body_r, body_h = .05, .18
            y = -.008 - body_r
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=40, radius=body_r, depth=body_h, location=(x, y, z))
            body = bpy.context.object
            body.name = prefix + '_body'
            import bmesh
            bm = bmesh.new()
            bm.from_mesh(body.data)
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if abs(f.normal.z) > .9], context='FACES')
            bm.to_mesh(body.data)
            bm.free()
            body.data.materials.append(dark)
            wall = body.modifiers.new('Rim', 'SOLIDIFY')
            wall.thickness = .004
            wall.offset = -1
            for old in tuple(body.users_collection):
                old.objects.unlink(body)
            col.objects.link(body)
            tag(body)
            glow = make_solid_material(prefix + '_glow', (1, .78, .48, 1), roughness=.3)
            gbsdf = glow.node_tree.nodes['Principled BSDF']
            for key, val in (('Emission Color', (1, .70, .36, 1)), ('Emission Strength', 30.0)):
                if key in gbsdf.inputs:
                    gbsdf.inputs[key].default_value = val
            for sign in (1, -1):
                bpy.ops.mesh.primitive_cylinder_add(
                    vertices=32, radius=body_r - .008, depth=.003,
                    location=(x, y, z + sign * (body_h * .5 - .012)))
                opening = bpy.context.object
                opening.name = prefix + ('_glow_up' if sign > 0 else '_glow_down')
                opening.data.materials.append(glow)
                for old in tuple(opening.users_collection):
                    old.objects.unlink(opening)
                col.objects.link(opening)
                tag(opening)
                bpy.ops.object.light_add(
                    type='SPOT', location=(x, y, z + sign * (body_h * .5 + .012)))
                lamp = bpy.context.object
                lamp.name = prefix + ('_lamp_up' if sign > 0 else '_lamp_down')
                lamp.data.energy = 80
                lamp.data.color = (1.0, .62, .32)
                lamp.data.spot_size = math.radians(100)
                lamp.data.spot_blend = .75
                lamp.data.shadow_soft_size = .04
                aim = Vector((x, .0, z + sign * .55))
                lamp.rotation_euler = (aim - lamp.location).to_track_quat('-Z', 'Y').to_euler()
                lamp['cwg_role'] = 'fixture_light'
                lamp['cwg_hard_negative'] = p.kind
                for old in tuple(lamp.users_collection):
                    old.objects.unlink(lamp)
                col.objects.link(lamp)
                lamp.parent = root
        sealed = p.kind in SEALED_EDGES and r.random() < .5
        if sealed:
            seal = make_solid_material(prefix + '_sealant', (*r.choice(_SEALANT_COLOURS), 1), roughness=r.uniform(.3, .5))
            radius = r.uniform(.0015, .003)
            parts = [o for o in objects if any(t in o.name for t in OUTLINE_PARTS.get(p.kind, ("",)))]
            if p.kind == "trunking":
                outlines = []
                for o in parts:
                    lo, hi = _world_bounds(o)
                    edges = "tb" if hi[0] - lo[0] >= hi[2] - lo[2] else "lr"
                    outlines.append((lo, hi, edges))
            elif p.kind in ("skirting", "ceiling_trim"):
                outlines = [(*_world_bounds(o), SEALED_EDGES[p.kind]) for o in parts]
            else:
                lo, hi = _union_bounds(parts)
                outlines = [(lo, hi, SEALED_EDGES[p.kind])]
            for lo, hi, edges in outlines:
                for points in _bead_paths(lo, hi, edges, radius, r,
                                          floor_z=.12 if p.kind == "door" else lo[2]):
                    tube('sealant', points, radius, seal, True)
        # Mount on the wall's displaced relief (stone and brick stand out of the plane).
        root.location.y -= relief_m
        root['cwg_role'] = 'structure'
        root['cwg_hard_negative'] = p.kind
        records.append(dict(**p.to_dict(), object_name=root.name,
                            objects=[o.name for o in objects], semantic_class='background',
                            dimensions_convention='nominal assembly footprint; hardware may extend',
                            geometry_model=fixture_model or 'surface_geometry', sealant=sealed))
    return records


SERVICE_KINDS = ("wall_cable", "conduit", "trunking", "plumbing", "cable", "downpipe")


def crack_regions(plans):
    """(keep-out, near) wall rectangles (u, v, w, h) for fixtures-first crack placement.

    Keep-out: every large assembly, grown to its built extent (a window's sill
    and lintel, a door's frame), so no crack sits under it. Near: thin service
    runs, beside which some cracks are placed so the service partly covers them.
    """
    keepout, near = [], []
    for p in plans:
        if p.kind == "window":
            keepout.append((p.u_m, p.v_m + 0.02, p.width_m + 0.20, p.height_m + 0.32))
        elif p.kind == "door":
            keepout.append((p.u_m, p.v_m, p.width_m + 0.06, p.height_m + 0.06))
        elif p.kind in LARGE_KINDS:
            keepout.append((p.u_m, p.v_m, p.width_m + 0.04, p.height_m + 0.04))
        elif p.kind in SERVICE_KINDS:
            near.append((p.u_m, p.v_m, p.width_m, p.height_m))
    return keepout, near
