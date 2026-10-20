"""Seeded background assemblies on the wall's X/Z face (outward is -Y).

These are occluding geometry, never injection hosts or crack annotations.
Window panes are opaque recessed glazing proxies within a surface-mounted
frame: the continuous injection host is not boolean-cut. Panel seams are
sealant joints, not simulated fractures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

PROFILES = ("off", "auto", "services", "facade", "indoor", "outdoor", "all")
KINDS = ("cable", "wall_cable", "conduit", "downpipe", "utility_box", "window", "panel_seam", "ledge")
INDOOR_KINDS = ("socket", "switch", "door", "skirting", "ceiling_trim",
                "vent", "access_panel", "trunking", "plumbing")
OUTDOOR_KINDS = ("exterior_vent", "service_hatch", "gutter", "wall_light")
KINDS += INDOOR_KINDS + OUTDOOR_KINDS
LARGE_KINDS = ("utility_box", "window", "ledge", "door", "vent", "access_panel",
               "exterior_vent", "service_hatch", "socket", "switch", "wall_light")
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


def structure_clean_wall(seed, profile, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("structure clean fraction must be 0..1")
    return profile != "off" and random.Random(seed ^ SALT ^ 0x434C).random() < fraction


def sample_structure_plan(seed, spec, placements=(), profile="auto", environment="indoor", include_kinds=None):
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
        indoor = ["wall_cable", "conduit", "utility_box", "window", "panel_seam", *INDOOR_KINDS]
        outdoor = ["cable", "wall_cable", "conduit", "downpipe", "utility_box", "window",
                   "panel_seam", "ledge", *OUTDOOR_KINDS]
        pool = (list(KINDS[:5]) + ["trunking", "plumbing"] if profile == "services" else
                ["window", "panel_seam", "ledge", "downpipe", *OUTDOOR_KINDS] if profile == "facade" else
                indoor if profile == "indoor" else outdoor if profile == "outdoor" else
                outdoor if environment == "outdoor" else indoor)
        kinds = r.sample(pool, r.randint(2, min(4, len(pool))))
    # Floor-anchored doors and boundary trims reserve their footprints first.
    kinds.sort(key=lambda k: 0 if k == "door" else 1 if k in ("skirting", "ceiling_trim", "gutter") else 2)
    plans = []
    for i, kind in enumerate(kinds):
        w, h = {
            "cable": (spec.width_m * .85, .5),
            "wall_cable": (.08, spec.height_m * .8),
            "conduit": (.5, min(1.7, spec.height_m * .75)),
            "downpipe": (.13, spec.height_m),
            "utility_box": (.24, .32),
            "window": (min(1.1, spec.width_m * .35), min(.85, spec.height_m * .32)),
            "panel_seam": (spec.width_m, .006),
            "ledge": (min(1.5, spec.width_m * .5), .16),
            "socket": (.146, .086), "switch": (.086, .086),
            "door": (.94, min(2.08, spec.height_m - .20)),
            "skirting": (spec.width_m, .11), "ceiling_trim": (spec.width_m, .065),
            "vent": (.42, .24), "access_panel": (.45, .50),
            "trunking": (spec.width_m * .7, .055), "plumbing": (.50, .65),
            "exterior_vent": (.42, .32), "service_hatch": (.50, .65),
            "gutter": (spec.width_m, .13), "wall_light": (.16, .27),
        }[kind]
        if kind in ("utility_box", "window", "ledge"):
            w *= r.uniform(.8, 1.15)
            h *= r.uniform(.8, 1.15)
        # Half the thin services pass near a defect. Large assemblies keep
        # defect centres clear, while their frame edges can partially occlude.
        for attempt in range(60):
            x = r.uniform(-spec.width_m * .5 + w * .5, spec.width_m * .5 - w * .5)
            z = r.uniform(max(h * .5, .3), spec.height_m - max(h * .5, .15))
            if placements and attempt == 0 and r.random() < .5:
                target = r.choice(placements)
                x = max(-spec.width_m * .5 + w * .5, min(spec.width_m * .5 - w * .5, target.u_m + r.uniform(-.12, .12)))
                z = max(h * .5, min(spec.height_m - h * .5, target.v_m + r.uniform(-.15, .15)))
            if kind == "door":
                z = h * .5
            elif kind == "socket":
                z = r.uniform(.25, .45)
            elif kind == "switch":
                z = r.uniform(1.05, min(1.35, spec.height_m - .2))
            elif kind == "skirting":
                x, z = 0., h * .5
            elif kind in ("ceiling_trim", "gutter"):
                x, z = 0., spec.height_m - h * .5
            elif kind in ("exterior_vent", "wall_light"):
                z = r.uniform(spec.height_m * .60, spec.height_m - h * .5 - .10)
            if kind in ("skirting", "ceiling_trim", "gutter"):
                break
            if kind in ("trunking", "plumbing", "conduit", "wall_cable", "downpipe", "cable") and any(q.kind == "door" and
                    abs(x-q.u_m) < (w+q.width_m)*.5 and z-h*.5 < q.height_m for q in plans):
                continue
            if kind == "cable" and not placements:
                z = r.uniform(spec.height_m * .55, spec.height_m - .3)
            if kind in ("downpipe", "wall_cable", "conduit") and any(
                    q.kind in ("downpipe", "wall_cable", "conduit") and
                    abs(x - q.u_m) < (w + q.width_m) * .5 + .06 for q in plans):
                continue
            if kind not in LARGE_KINDS:
                break
            if any(abs(x - p.u_m) < w * .5 + .06 and abs(z - p.v_m) < h * .5 + .06 for p in placements):
                continue
            if any(p.kind != "panel_seam" and
                   abs(x - p.u_m) < (w + p.width_m) * .5 + .05 and
                   abs(z - p.v_m) < (h + p.height_m) * .5 + .05 for p in plans):
                continue
            break
        else:
            continue  # no physically plausible space: report only built plans
        plans.append(StructurePlan(i, kind, x, z, w, h, r.randrange(2**31)))
    return plans


def build_structures(plans):
    import bpy
    from syncrack3d.geometry import _collection, _box
    from syncrack3d.materials import make_solid_material

    col = _collection("CWG_STRUCTURES")
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
        light = r.uniform(.38, .75)
        paint = make_solid_material(prefix + '_paint', (light, light * .98, light * .94, 1), roughness=.65)
        dark = make_solid_material(prefix + '_rubber', (.012, .014, .016, 1), roughness=.75)
        metal = make_solid_material(prefix + '_metal', (.16, .17, .18, 1), roughness=.4)
        metal.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = .65

        def tag(obj):
            obj["cwg_role"] = "structure"
            obj["cwg_hard_negative"] = p.kind
            obj["cwg_structure_index"] = p.index
            obj.parent = root
            objects.append(obj)
            return obj

        def box(label, dims, loc, mat=paint):
            obj = tag(_box(prefix + '_' + label, dims, loc, mat, col, root))
            bevel = obj.modifiers.new('Soft manufactured edges', 'BEVEL')
            bevel.width = min(.003, min(dims) * .15)
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
            # A suspended span with gravity sag, wall anchors and a loose return.
            radius = r.uniform(.0025, .006)
            points = [(x + w * (t / 8 - .5), -.065 - .06 * math.sin(math.pi * t / 8),
                       z + .22 - .36 * math.sin(math.pi * t / 8)) for t in range(9)]
            tube('sag', points, radius, dark, True)
            for end in (points[0], points[-1]):
                box('anchor', (.035, .08, .05), (end[0], -.04, end[2]), metal)
            if r.random() < .65:
                tube('loop', [(points[0][0], -.065, z + .22), (points[0][0] + .06, -.07, z - .18),
                              (points[0][0] + .20, -.08, z - .28), (points[0][0] + .24, -.06, z + .20)], radius, dark, True)
        elif p.kind == 'wall_cable':
            radius = r.uniform(.003, .007)
            for bundle in range(r.randint(1, 3)):
                xx = x + bundle * .012
                pts = [(xx + .004 * math.sin(t * 2.1), -.018, z - h * .5 + h * t / 8) for t in range(9)]
                tube('wire', pts, radius, dark, True)
            for t in (.1, .45, .8):
                zz = z - h * .5 + h * t
                box('clip', (.055, .026, .014), (x + .012, -.017, zz), metal)
        elif p.kind in ('conduit', 'downpipe'):
            radius = r.uniform(.009, .015) if p.kind == 'conduit' else r.uniform(.035, .055)
            y = -(radius + .022)
            bottom, top = max(.04, z - h * .5), z + h * .5
            if p.kind == 'conduit':
                # Short bevel elbow into a horizontal run.
                tube('run', [(x - w * .5, y, bottom), (x + w * .3, y, bottom),
                             (x + w * .5, y, bottom + .06), (x + w * .5, y, top)], radius, paint)
                x += w * .5
            else:
                tube('run', [(x, y, bottom), (x, y, top)], radius)
                tube('socket', [(x, y, top * .55 - .035), (x, y, top * .55 + .035)], radius * 1.22)
            for zz in (bottom + .15, top - .15):
                box('bracket', (radius * 2 + .045, abs(y), .018), (x, y * .5, zz), metal)
                tube('collar', [(x, y, zz - .012), (x, y, zz + .012)], radius * 1.10, metal)
        elif p.kind == 'utility_box':
            box('housing', (w, .10, h), (x, -.05, z))
            box('lid', (w * .96, .012, h * .96), (x, -.108, z))
            for dx in (-.39, .39):
                for dz in (-.40, .40):
                    tube('screw', [(x + w * dx, -.113, z + h * dz), (x + w * dx, -.12, z + h * dz)], .004, metal)
            tube('feed', [(x, -.025, z + h * .5), (x, -.025, z + h * .5 + .25)], .009)
        elif p.kind == 'window':
            glass = make_solid_material(prefix + '_opaque_glazing', (.035, .060, .075, 1), roughness=.2)
            # Opaque backing blocks the host AOV, including at oblique views.
            box('backing', (w, .035, h), (x, -.020, z), dark)
            box('pane', (w - .07, .015, h - .07), (x, -.046, z), glass)
            frame_mat = dark if r.random() < .65 else paint
            for dx in (-w * .5, 0, w * .5):
                box('mullion', (.038, .10, h + .04), (x + dx, -.06, z), frame_mat)
            for dz in (-h * .5, h * .5):
                box('rail', (w + .04, .10, .045), (x, -.06, z + dz), frame_mat)
            box('sill', (w + .16, .22, .055), (x, -.10, z - h * .5 - .04))
            box('lintel', (w + .20, .16, .10), (x, -.075, z + h * .5 + .07))
        elif p.kind == 'panel_seam':
            # Opaque sealant in a narrow construction joint: never a crack field.
            box('horizontal_sealant', (w, .004, h), (x, -.004, z), dark)
            if r.random() < .7:
                box('vertical_sealant', (h, .004, max(.4, z)), (x + w * .22, -.004, max(.4, z) * .5), dark)
        elif p.kind == 'ledge':
            box('beam', (w, .32, h), (x, -.155, z))
            box('drip_edge', (w, .015, .018), (x, -.29, z - h * .5), metal)
        elif p.kind in ('socket', 'switch'):
            box('gasket', (w, .007, h), (x, -.004, z), dark)
            box('faceplate', (w-.004, .012, h-.004), (x, -.013, z))
            for dx in (-w*.40, w*.40):
                tube('screw', [(x+dx, -.019, z), (x+dx, -.022, z)], .0025, metal)
                box('screw_slot', (.003, .001, .0007), (x+dx, -.0225, z), dark)
            if p.kind == 'switch':
                box('rocker_gap', (.037, .004, .049), (x, -.021, z), dark)
                rocker = box('rocker', (.033, .007, .045), (x, -.025, z))
                rocker.rotation_euler.x = math.radians(r.choice((-7, 7)))
            else:
                for dx in (-w*.22, w*.22):
                    for xx, zz in ((-.009, -.006), (.009, -.006), (0, .014)):
                        box('socket_aperture', (.004, .002, .010), (x+dx+xx, -.020, z+zz), dark)
        elif p.kind == 'door':
            door_mat = make_solid_material(prefix+'_leaf', (.22, .13, .065, 1), roughness=.6)
            box('shadow_reveal', (w, .018, h), (x, -.012, z), dark)
            box('closed_leaf', (w-.10, .026, h-.06), (x, -.034, z-.015), door_mat)
            for dx in (-w*.5+.025, w*.5-.025):
                box('jamb', (.05, .075, h), (x+dx, -.04, z))
            box('header', (w, .075, .05), (x, -.04, z+h*.5-.025))
            box('threshold', (w-.10, .10, .015), (x, -.045, .009), metal)
            hx, hz = x+w*.30, .98
            box('handle_plate', (.035, .012, .12), (hx, -.055, hz), metal)
            tube('handle_stem', [(hx, -.061, hz), (hx, -.105, hz)], .008, metal)
            tube('lever', [(hx, -.105, hz), (hx-.10, -.105, hz)], .008, metal)
            for zz in (.30, h-.30):
                tube('hinge', [(x-w*.5+.06, -.055, zz), (x-w*.5+.06, -.055, zz+.06)], .007, metal)
        elif p.kind in ('skirting', 'ceiling_trim', 'trunking'):
            depth = .028 if p.kind == 'trunking' else .018
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
                    if p.kind == 'trunking':
                        box('lid', (length-.004, .004, h-.006), (cx, -depth-.006, z))
                    else:
                        box('moulding', (length-.003, .008, h*.18), (cx, -depth-.004, z+h*.28))
        elif p.kind in ('vent', 'exterior_vent', 'access_panel', 'service_hatch'):
            box('frame', (w, .028, h), (x, -.016, z))
            box('recess', (w-.035, .008, h-.035), (x, -.035, z), dark)
            if p.kind in ('vent', 'exterior_vent'):
                count = max(4, int(h/.028))
                for j in range(count):
                    blade = box('louvre', (w-.045, .025, .012),
                                (x, -.055, z-h*.5+.025+(h-.05)*(j+.5)/count), metal)
                    blade.rotation_euler.x = math.radians(-25)
                if p.kind == 'exterior_vent':
                    box('rain_hood', (w+.04, .12, .025), (x, -.06, z+h*.5))
            else:
                box('door_panel', (w-.042, .012, h-.042), (x, -.043, z))
                tube('latch', [(x+w*.28, -.051, z), (x+w*.28, -.057, z)], .01, metal)
                for dz in (-h*.28, h*.28):
                    box('hinge', (.025, .012, .06), (x-w*.38, -.053, z+dz), metal)
            for dx in (-w*.43, w*.43):
                for dz in (-h*.40, h*.40):
                    tube('fastener', [(x+dx, -.031, z+dz), (x+dx, -.036, z+dz)], .003, metal)
        elif p.kind == 'plumbing':
            # Paired exposed service pipes, clamps, union fittings and a valve.
            for dx in (-.12, .12):
                xx = x+dx
                tube('pipe', [(xx, -.045, z-h*.5), (xx, -.045, z+h*.5)], .011, metal)
                for dz in (-h*.32, h*.32):
                    box('wall_clip', (.04, .045, .018), (xx, -.0225, z+dz))
                    tube('union', [(xx, -.045, z+dz-.018), (xx, -.045, z+dz+.018)], .016, metal)
            tube('valve_body', [(x+.12, -.045, z-.03), (x+.12, -.045, z+.03)], .023, metal)
            tube('valve_stem', [(x+.12, -.045, z), (x+.12, -.095, z)], .006, metal)
            box('valve_lever', (.10, .014, .016), (x+.15, -.10, z), dark)
        elif p.kind == 'gutter':
            # Open U channel, end caps, section joints and wall brackets.
            box('base', (w, .14, .012), (x, -.08, z-h*.5+.006))
            box('back', (w, .012, h), (x, -.016, z))
            box('front', (w, .012, h*.75), (x, -.144, z-h*.125))
            for dx in (-w*.5, w*.5):
                box('end_cap', (.012, .14, h), (x+dx, -.08, z))
            for j in range(max(2, int(w/.7))):
                xx=x-w*.5+.15+(w-.30)*j/(max(2, int(w/.7))-1)
                box('bracket', (.022, .16, .025), (xx, -.08, z-h*.5-.012), metal)
        elif p.kind == 'wall_light':
            box('backplate', (w, .025, h), (x, -.014, z), dark)
            diffuser = make_solid_material(prefix+'_diffuser', (.65, .68, .62, 1), roughness=.4)
            box('opaque_diffuser', (w-.025, .07, h-.035), (x, -.055, z), diffuser)
            for dz in (-h*.36, 0, h*.36):
                box('guard', (w, .015, .012), (x, -.098, z+dz), metal)
        root['cwg_role'] = 'structure'
        root['cwg_hard_negative'] = p.kind
        records.append(dict(**p.to_dict(), object_name=root.name,
                            objects=[o.name for o in objects], semantic_class='background',
                            dimensions_convention='nominal assembly footprint; hardware may extend',
                            geometry_model='opaque_glazing_proxy' if p.kind == 'window' else 'closed_door_proxy' if p.kind == 'door' else 'surface_geometry'))
    return records


def structure_camera_targets(records, limit=2):
    from types import SimpleNamespace
    # Focus on services as well as windows; list order is independently shuffled
    # by the caller to avoid always selecting cables in the all profile.
    return [SimpleNamespace(index=r['index'], u_m=r['u_m'], v_m=r['v_m'],
                            size_m=max(.4, min(1.4, r['width_m'], r['height_m']) * 1.3),
                            object_name=r['object_name']) for r in records[:limit]]
