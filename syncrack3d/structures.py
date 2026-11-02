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
KINDS = ("cable", "wall_cable", "conduit", "downpipe", "utility_box", "window", "panel_seam")
# Exclusive to one setting. Vent, access panel, and the full-length runs are shared.
# Outdoor scenes have no exclusive fixture of their own.
INDOOR_KINDS = ("door", "trunking")
OUTDOOR_KINDS = ()
KINDS += INDOOR_KINDS + OUTDOOR_KINDS + ("vent", "access_panel", "plumbing")
LARGE_KINDS = ("utility_box", "window", "door", "vent", "access_panel")
# Full-length runs, down the wall or across it. Trunking is indoor only.
# Outdoors the built run continues across the whole facade; the plan stays on
# the inspection wall so cracks and junctions do too.
SPAN_KINDS = ("wall_cable", "conduit", "downpipe", "plumbing", "trunking", "panel_seam")
SPAN_SECTION = {
    "wall_cable": 0.10, "conduit": 0.10, "downpipe": 0.18, "plumbing": 0.40,
    "trunking": 0.055, "panel_seam": 0.006,
}
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


FIXTURE_GAP_M = .03  # nearest any fixture's built extent comes to another's
TRIES = 60
# Hardware an assembly builds past its nominal w x h box, as build_structures
# makes it: (left, right, below, above) in metres - a window's sill and
# lintel, a conduit's elbow and brackets, a cable's sag, a utility box's feed.
# A test checks these cover the built geometry.
PAD = {
    "cable": (.03, .03, .06, 0.), "wall_cable": (0., 0., 0., 0.), "conduit": (0., 0., 0., 0.),
    "downpipe": (0., 0., 0., 0.), "utility_box": (0., 0., 0., .25), "window": (.10, .10, .07, .12),
    "vent": (.02, .02, 0., .015), "access_panel": (.02, .02, 0., .015),
    "plumbing": (0., 0., 0., 0.),
}
JITTER_KINDS = ("utility_box", "window")  # width and height x U(.8, 1.15)
FLUSH_KINDS = ("panel_seam",)  # flush sealant joints: placed like the rest, but others may cross them
BOUNDARIES = ("floor",)


def fixture_rules(width_m, height_m):
    """kind -> (width, height, anchor) on a width_m x height_m wall.

    The anchor is where the centre height is drawn: None from 0.3 m up to
    0.15 m under the ceiling, (lo, hi) a band, "run" the clear wall height, or
    "floor" standing on the floor (a door)."""
    W, H = width_m, height_m
    return {
        "cable": (W * .85, .5, (H * .55, H - .3)),
        "wall_cable": (SPAN_SECTION["wall_cable"], H, "run"),
        "conduit": (SPAN_SECTION["conduit"], H, "run"),
        "downpipe": (SPAN_SECTION["downpipe"], H, "run"),
        "utility_box": (.24, .32, None),
        "window": (min(1.1, W * .35), min(.85, H * .32), None),
        "panel_seam": (SPAN_SECTION["panel_seam"], H, "run"),
        "door": (.94, min(2.08, H - .20), "floor"),
        "vent": (.56, .18, None),
        "access_panel": (.46, .46, None),
        "trunking": (SPAN_SECTION["trunking"], H, "run"),
        "plumbing": (SPAN_SECTION["plumbing"], H, "run"),
    }


def span_on_host(kind, u, v, w, h, bounds):
    """(u, v, w, h) of a full-length run on bounds (u_lo, u_hi, v_lo, v_hi).

    bounds is None on the inspection wall. Outdoors it is the whole front face,
    past the inspection panel. A sideways run keeps its height; a vertical run
    keeps its place across the wall. Other fixtures are unchanged."""
    if bounds is None or kind not in SPAN_KINDS:
        return u, v, w, h
    u0, u1, v0, v1 = bounds
    if w >= h:
        return (u0 + u1) * .5, v, u1 - u0, h
    return u, (v0 + v1) * .5, w, v1 - v0


def built_extent(kind, u, v, w, h):
    """(u_lo, u_hi, v_lo, v_hi) on the wall covered by an assembly as
    build_structures makes it, hardware included (PAD)."""
    left, right, below, above = PAD.get(kind, (0., 0., 0., 0.))
    return u - w * .5 - left, u + w * .5 + right, v - h * .5 - below, v + h * .5 + above


def _overlaps(a, b, gap):
    return a[0] < b[1] + gap and b[0] < a[1] + gap and a[2] < b[3] + gap and b[2] < a[3] + gap


def sample_structure_plan(seed, spec, placements=(), profile="auto", environment="indoor", include_kinds=None):
    """Plan the wall's fixtures. Every kind is placed by one rule; only its
    size, anchor (fixture_rules) and hardware (PAD) differ.

    Each fixture draws its centre within its anchor so that its nominal box
    and built extent stay on the wall, and is kept only if its built extent is
    FIXTURE_GAP_M clear of every fixture already placed and of the base wall's
    skirting; a large kind also keeps crack centres 6 cm clear. A fixture with
    no room after TRIES draws is dropped. A door goes first. A wire, conduit,
    pipe, plumbing run, trunking or panel seam then spans the clear height, or
    the full width. Panel seams are flush joints: nothing is kept clear of them."""
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
        runs = ("wall_cable", "conduit", "downpipe", "plumbing")
        shared = ("vent", "access_panel", *runs)
        indoor = ["utility_box", "window", "panel_seam", *shared, *INDOOR_KINDS]
        outdoor = ["cable", "utility_box", "window", "panel_seam", *shared, *OUTDOOR_KINDS]
        pool = (list(KINDS[:5]) + ["trunking", "plumbing"] if profile == "services" else
                ["window", "panel_seam", *shared, *OUTDOOR_KINDS] if profile == "facade" else
                indoor if profile == "indoor" else outdoor if profile == "outdoor" else
                outdoor if environment == "outdoor" else indoor)
        kinds = r.sample(pool, r.randint(2, min(4, len(pool))))
    W, H = spec.width_m, spec.height_m
    rules = fixture_rules(W, H)
    kinds.sort(key=lambda k: rules[k][2] not in BOUNDARIES)
    skirting = getattr(spec, "skirting", False)
    obstacles = [(-W * .5, W * .5, 0., .085)] if skirting else []  # the base wall's own skirting
    # Free span for a vertical run; the 0.1 mm hair keeps a run that fills it clear of the trims.
    clear = FIXTURE_GAP_M + 1e-4
    low, high = (.085 + clear if skirting else 0.), H
    plans = []
    for i, kind in enumerate(kinds):
        w, h, anchor = rules[kind]
        if kind in SPAN_KINDS:
            # Full length, floor to ceiling or side to side. The draw is its own
            # call so both directions stay in the seeded stream.
            section = SPAN_SECTION[kind]
            if r.random() < .5:
                w, h, anchor = W, section, "side"
            else:
                w, h, anchor = section, H, "run"
        if kind in JITTER_KINDS:
            w *= r.uniform(.8, 1.15)
            h *= r.uniform(.8, 1.15)
        left, right, below, above = (max(0., p) for p in PAD.get(kind, (0., 0., 0., 0.)))
        if anchor == "run":
            h = min(h, high - low - below - above)
            band = (low + below + h * .5, high - above - h * .5)
        elif anchor == "side":
            band = (low + below + h * .5, high - above - h * .5)
        elif anchor in BOUNDARIES:
            band = (H - h * .5,) * 2 if anchor == "ceiling" else (h * .5,) * 2
        else:
            band = anchor or (max(h * .5, .3), H - max(h * .5, .15))
        u_lo, u_hi = -W * .5 + w * .5 + left, W * .5 - w * .5 - right
        v_lo, v_hi = max(band[0], h * .5 + below), min(band[1], H - h * .5 - above)
        if h <= 0 or u_lo > u_hi + 1e-9 or v_lo > v_hi + 1e-9:
            continue  # no room on this wall
        for attempt in range(TRIES):
            u, v = r.uniform(u_lo, u_hi), r.uniform(v_lo, v_hi)
            if placements and attempt == 0 and r.random() < .5:
                # Half the fixtures first try beside a crack, so some partly cover one.
                target = r.choice(placements)
                u = min(u_hi, max(u_lo, target.u_m + r.uniform(-.12, .12)))
                v = min(v_hi, max(v_lo, target.v_m + r.uniform(-.15, .15)))
            if (kind not in FLUSH_KINDS and anchor not in BOUNDARIES
                    and any(_overlaps(built_extent(kind, u, v, w, h), b, FIXTURE_GAP_M) for b in obstacles)):
                continue
            if kind in LARGE_KINDS and any(abs(u - p.u_m) < w * .5 + .06 and abs(v - p.v_m) < h * .5 + .06
                                           for p in placements):
                continue
            break
        else:
            continue  # no room: report only built plans
        plans.append(StructurePlan(i, kind, u, v, w, h, r.randrange(2**31)))
        if kind not in FLUSH_KINDS:
            obstacles.append(built_extent(kind, u, v, w, h))
    return plans


def build_structures(plans, relief_m=0.0, span_bounds=None, host_height_m=None):
    """Build the planned fixtures. Each assembly stands `relief_m` out from the
    wall plane, on the wall's displaced relief rather than sunk into it."""
    import bpy
    from syncrack3d.geometry import _collection, _box
    from syncrack3d.materials import make_solid_material

    col = _collection("CWG_STRUCTURES")
    records = []
    plans = list(plans)
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

        x, z, w, h = span_on_host(p.kind, p.u_m, p.v_m, p.width_m, p.height_m, span_bounds)
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
        def finish(label, color, roughness=.4, metallic=0., variation=.025, scale=48.):
            mat = make_solid_material(prefix + '_' + label, color, roughness=roughness, metallic=metallic)
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            bsdf = nodes['Principled BSDF']
            noise = nodes.new('ShaderNodeTexNoise')
            noise.inputs['Scale'].default_value = scale
            noise.inputs['Detail'].default_value = 3.0
            noise.inputs['Roughness'].default_value = .55
            links.new(nodes.new('ShaderNodeTexCoord').outputs['Object'], noise.inputs['Vector'])
            ramp = nodes.new('ShaderNodeValToRGB')
            ramp.color_ramp.elements[0].position = .32
            ramp.color_ramp.elements[0].color = tuple(max(0., c - variation) for c in color[:3]) + (1.,)
            ramp.color_ramp.elements[1].position = .68
            ramp.color_ramp.elements[1].color = tuple(min(1., c + variation) for c in color[:3]) + (1.,)
            links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
            links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
            return mat

        if p.kind == 'wall_cable':
            # Insulated live, neutral and earth cables in white plastic clips.
            horizontal = w >= h
            colours = ((.03, .03, .035, 1), (.78, .78, .76, 1), (.16, .45, .2, 1))
            count = 3
            span = w if horizontal else h
            groups = max(3, int(span / .34))
            for bundle in range(count):
                short = (bundle - (count - 1) * .5) * .016
                pts = []
                for step in range(13):
                    t = step / 12
                    sag = .0025 * math.sin(t * math.pi * groups)
                    if horizontal:
                        pts.append((x - w * .5 + w * t, -.013 - sag, z + short))
                    else:
                        pts.append((x + short, -.013 - sag, z - h * .5 + h * t))
                tube('wire', pts, .004, finish(f'insul{bundle}', colours[bundle], .48, variation=.02), True)
            clip = finish('clip', (.93, .93, .91, 1), .5, variation=.01)
            span_clip = count * .016 + .012
            for i in range(1, groups):
                t = i / groups
                if horizontal:
                    box('clip', (.016, .012, span_clip), (x - w * .5 + w * t, -.012, z), clip)
                else:
                    box('clip', (span_clip, .012, .016), (x, -.012, z - h * .5 + h * t), clip)
        elif p.kind == 'conduit':
            # White PVC, or galvanized EMT, with a coupler and saddle straps.
            horizontal = w >= h
            pvc = r.random() < .65
            if pvc:
                body = finish('pvc', (.86, .87, .85, 1), .36, variation=.015)
                strap = finish('strap', (.78, .79, .77, 1), .48, variation=.01)
                radius = .012
            else:
                body = finish('emt', (.62, .63, .61, 1), .3, .9, variation=.03, scale=80.)
                strap = body
                radius = .01
            y = -(radius + .016)
            span = w if horizontal else h

            def at(frac, delta=0.):
                if horizontal:
                    return (x - w * .5 + w * frac + delta, y, z)
                return (x, y, z - h * .5 + h * frac + delta)

            tube('run', [at(0.), at(1.)], radius, body)
            joints = max(2, int(span / 1.5)) if span >= 1.2 else 1
            for i in range(1, joints):
                tube('coupling', [at(i / joints, -.02), at(i / joints, .02)], radius * 1.28, body)
            straps = max(4, int(span / .55))
            band = radius * 2 + .016
            for i in range(straps):
                frac = (i + .5) / straps
                loc = at(frac)
                dims = (.016, abs(y), band) if horizontal else (band, abs(y), .016)
                box('saddle', dims, (loc[0], y * .4, loc[2]), strap)
                front = y - radius - .003
                band_dims = (.014, .004, band) if horizontal else (band, .004, .014)
                box('strap', band_dims, (loc[0], front, loc[2]), strap)
        elif p.kind == 'downpipe':
            # Round PVC rainwater pipe, socket joints and stand-off clips.
            horizontal = w >= h
            tone = r.choice(((.91, .91, .89, 1), (.2, .21, .22, 1), (.58, .6, .61, 1)))
            body = finish('pvc', tone, .26, variation=.018)
            radius = r.uniform(.032, .04)
            y = -(radius + .018)
            span = w if horizontal else h

            def at(frac, delta=0.):
                if horizontal:
                    return (x - w * .5 + w * frac + delta, y, z)
                return (x, y, z - h * .5 + h * frac + delta)

            tube('run', [at(0.), at(1.)], radius, body)
            for frac in ((.5,) if span < 1.6 else (.33, .66)):
                tube('socket', [at(frac, -.028), at(frac, .028)], radius * 1.16, body)
            clips = max(3, int(span / .7))
            shoulder = radius * 2 + .026
            for i in range(1, clips):
                loc = at(i / clips)
                if horizontal:
                    box('backplate', (.022, .01, shoulder), (loc[0], y * .28, z), body)
                    box('strap', (.016, .005, shoulder), (loc[0], y - radius - .003, z), body)
                else:
                    box('backplate', (shoulder, .01, .022), (x, y * .28, loc[2]), body)
                    box('strap', (shoulder, .005, .016), (x, y - radius - .003, loc[2]), body)
        elif p.kind == 'utility_box':
            # Painted cabinet, shut door, and a conduit that runs the full host height.
            colour = r.choice(((.78, .58, .10), (.82, .40, .12), (.32, .52, .28), (.55, .57, .59)))
            cabinet = make_solid_material(prefix + '_cabinet', (*colour, 1), roughness=.45)
            box('housing', (w, .055, h), (x, -.030, z), cabinet)
            box('door', (w - .014, .012, h - .014), (x, -.062, z), cabinet)
            for frac in (.22, .78):
                hz = z - h * .5 + h * frac
                tube('hinge', [(x - w * .5 + .008, -.064, hz - .014),
                               (x - w * .5 + .008, -.064, hz + .014)], .005, metal)
            tube('latch', [(x + w * .30, -.066, z), (x + w * .30, -.082, z)], .007, metal)
            if r.random() < .6:
                face = make_solid_material(prefix + '_meter', (.12, .16, .18, 1), roughness=.25)
                rim = min(w, h) * .26
                tube('meter_rim', [(x - w * .06, -.066, z + .01), (x - w * .06, -.080, z + .01)], rim, metal)
                tube('meter_face', [(x - w * .06, -.074, z + .01), (x - w * .06, -.084, z + .01)], rim * .78, face)
            if span_bounds is not None:
                bottom, top = span_bounds[2], span_bounds[3]
            elif host_height_m:
                bottom, top = 0., host_height_m
            else:
                bottom, top = z - h * .5, z + h * .5 + .25
            y, radius = -.018, .012
            riser_mat = finish('riser', (.84, .85, .83, 1), .34, variation=.015)
            tube('riser', [(x, y, bottom), (x, y, top)], radius, riser_mat)
            tube('hub', [(x, y, z + h * .5 - .012), (x, y, z + h * .5 + .016)], radius * 1.35, riser_mat)
            for z0, z1 in ((z + h * .5 + .04, top), (bottom, z - h * .5 - .02)):
                if z1 - z0 < .15:
                    continue
                box('clip', (.036, .02, .012), (x, -.012, (z0 + z1) * .5), riser_mat)
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
            # One construction joint the full length of the host. Never a crack field.
            box('sealant', (w, .004, h), (x, -.004, z), dark)
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
        elif p.kind == 'trunking':
            # White PVC trunking: channel, snap lid with a centre seam, and screws.
            pvc = finish('trunk', (.92, .92, .9, 1), .4, variation=.012)
            seam = finish('seam', (.72, .72, .7, 1), .5, variation=.01)
            horizontal = w >= h
            lo, hi = (x - w * .5, x + w * .5) if horizontal else (z - h * .5, z + h * .5)
            cross = h if horizontal else w
            depth = min(.022, cross * .55)
            count = max(1, math.ceil((hi - lo) / 2.0))
            length = (hi - lo) / count
            gap = .003 if count > 1 else 0.
            for j in range(count):
                c = lo + (j + .5) * length
                piece = length - gap
                if horizontal:
                    box('section', (piece, depth, cross), (c, -depth * .5 - .003, z), pvc)
                    box('lid', (piece - .002, .004, cross - .016), (c, -depth - .005, z), pvc)
                    box('groove', (piece - .01, .0014, .0035), (c, -depth - .008, z), seam)
                else:
                    box('section', (cross, depth, piece), (x, -depth * .5 - .003, c), pvc)
                    box('lid', (cross - .016, .004, piece - .002), (x, -depth - .005, c), pvc)
                    box('groove', (.0035, .0014, piece - .01), (x, -depth - .008, c), seam)
            screws = max(2, int((hi - lo) / .45))
            for i in range(1, screws):
                c = lo + (hi - lo) * i / screws
                if horizontal:
                    tube('screw', [(c, -depth - .004, z), (c, -depth - .010, z)], .0024, metal)
                else:
                    tube('screw', [(x, -depth - .004, c), (x, -depth - .010, c)], .0024, metal)
        elif p.kind in ('vent', 'access_panel'):
            def rails(border, depth, y, mat, tag):
                box(tag + '_stile_l', (border, depth, h), (x - w * .5 + border * .5, y, z), mat)
                box(tag + '_stile_r', (border, depth, h), (x + w * .5 - border * .5, y, z), mat)
                box(tag + '_rail_b', (w - 2 * border, depth, border), (x, y, z - h * .5 + border * .5), mat)
                box(tag + '_rail_t', (w - 2 * border, depth, border), (x, y, z + h * .5 - border * .5), mat)

            if p.kind == 'vent':
                # Wide register: pale frame, dark opening, thin horizontal slats, one screw each side.
                frame = make_solid_material(prefix + '_frame', (.86, .87, .88, 1), roughness=.42)
                border = min(.02, h * .22, w * .05)
                rails(border, .014, -.014, frame, 'frame')
                open_w, open_h = w - 2 * border - .004, h - 2 * border - .004
                box('cavity', (open_w, .008, open_h), (x, -.006, z), dark)
                count = max(7, int(open_h / .012))
                thick = min(.0035, open_h / count * .42)
                for j in range(count):
                    zz = z - open_h * .5 + open_h * (j + .5) / count
                    box('slat', (open_w - .004, .01, thick), (x, -.016, zz), dark)
                for sx in (x - w * .5 + border * .5, x + w * .5 - border * .5):
                    tube('screw', [(sx, -.018, z), (sx, -.024, z)], .004, metal)
            else:
                # Square metal hatch, door shut in the frame, latch and hinge knuckles.
                silver = make_solid_material(prefix + '_silver', (.70, .72, .74, 1), roughness=.28)
                silver.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = .9
                border = min(.03, w * .09, h * .09)
                rails(border, .012, -.012, silver, 'frame')
                open_w, open_h = w - 2 * border, h - 2 * border
                box('cavity', (open_w + .004, .01, open_h + .004), (x, -.004, z), dark)
                reveal = .004
                door_w, door_h = open_w - reveal, open_h - reveal
                box('door', (door_w, .008, door_h), (x, -.018, z), silver)
                lx = x + door_w * .5 - .02
                tube('latch', [(lx, -.020, z), (lx, -.030, z)], .006, metal)
                hx = x + w * .5 - .007
                for frac in (.22, .5, .78):
                    hz = z - h * .5 + h * frac
                    tube('hinge', [(hx, -.022, hz - .016), (hx, -.022, hz + .016)], .005, silver)
        elif p.kind == 'plumbing':
            # Paired copper or chrome pipes, compression fittings, clips and a stopcock.
            horizontal = w >= h
            if r.random() < .7:
                pipe_mat = finish('copper', (.72, .40, .26, 1), .28, .92, variation=.05, scale=70.)
                fit_mat = finish('brass', (.6, .46, .22, 1), .36, .88, variation=.03, scale=60.)
            else:
                pipe_mat = finish('chrome', (.78, .79, .78, 1), .12, .96, variation=.02, scale=90.)
                fit_mat = pipe_mat
            clip_mat = finish('clip', (.93, .93, .91, 1), .5, variation=.01)
            radius, y = .008, -.034
            for short in (-.09, .09):
                if horizontal:
                    tube('pipe', [(x - w * .5, y, z + short), (x + w * .5, y, z + short)], radius, pipe_mat)
                else:
                    tube('pipe', [(x + short, y, z - h * .5), (x + short, y, z + h * .5)], radius, pipe_mat)
                for frac in (.22, .46, .78):
                    if horizontal:
                        cx = x - w * .5 + w * frac
                        box('clip', (.014, .02, .028), (cx, y * .45, z + short), clip_mat)
                        tube('union', [(cx - .012, y, z + short), (cx + .012, y, z + short)], radius * 1.45, fit_mat)
                    else:
                        cz = z - h * .5 + h * frac
                        box('clip', (.028, .02, .014), (x + short, y * .45, cz), clip_mat)
                        tube('union', [(x + short, y, cz - .012), (x + short, y, cz + .012)], radius * 1.45, fit_mat)
            if horizontal:
                tube('valve_body', [(x - .02, y, z + .09), (x + .02, y, z + .09)], .015, fit_mat)
                tube('valve_stem', [(x, y, z + .09), (x, y - .03, z + .09)], .004, fit_mat)
                box('valve_lever', (.01, .008, .05), (x, y - .034, z + .09), fit_mat)
            else:
                tube('valve_body', [(x + .09, y, z - .02), (x + .09, y, z + .02)], .015, fit_mat)
                tube('valve_stem', [(x + .09, y, z), (x + .09, y - .03, z)], .004, fit_mat)
                box('valve_lever', (.05, .008, .01), (x + .07, y - .034, z), fit_mat)
        root.location.y -= relief_m
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
