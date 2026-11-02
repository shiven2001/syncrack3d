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

PROFILES = ("off", "auto", "services", "all")
KINDS = ("cable", "wall_cable", "conduit", "downpipe", "utility_box", "window", "panel_seam")
INDOOR_KINDS = ("door", "trunking")
KINDS += INDOOR_KINDS + ("vent", "access_panel", "plumbing")
LARGE_KINDS = ("utility_box", "window", "door", "vent", "access_panel")
# Full-length runs, down the wall or across it.
SPAN_KINDS = ("wall_cable", "conduit", "downpipe", "plumbing", "trunking", "panel_seam")
SPAN_SECTION = {
    # Wide enough for the largest parallel bundle (4) plus clips. A scene draws 2, 3 or 4
    # and packs them in the middle, so the mesh is often narrower than this slot.
    "wall_cable": 0.10, "conduit": 0.22, "downpipe": 0.36, "plumbing": 0.40,
    "trunking": 0.20, "panel_seam": 0.012,
}
# Sampled per fixture seed. Linear RGB, then a small jitter and a noise albedo.
_ENAMEL = (
    (.90, .90, .88), (.78, .78, .76), (.62, .63, .65), (.40, .41, .43),
    (.22, .23, .24), (.74, .69, .60), (.52, .40, .28), (.80, .78, .72),
)
_SEALANT = (
    (.94, .94, .92), (.76, .76, .74), (.55, .55, .53), (.30, .30, .31),
    (.12, .12, .13), (.72, .67, .58), (.48, .50, .52),
)
_INSULATION = (
    (.04, .04, .045), (.84, .84, .82), (.48, .48, .49), (.36, .22, .12),
    (.16, .30, .55), (.55, .13, .13), (.16, .42, .20), (.74, .60, .16),
)
_PIPE = (
    (.93, .93, .91), (.78, .79, .80), (.48, .49, .51), (.16, .17, .18),
    (.55, .32, .22), (.62, .30, .22), (.86, .83, .74),
)


def run_count(rng: random.Random) -> int:
    """Parallel wires, conduits, pipes, trunking or plumbing runs in one fixture."""
    return rng.randint(2, 4)


def loose_cable_count(rng: random.Random) -> int:
    """Loose indoor cables sharing one span."""
    return rng.randint(1, 3)
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
    "cable": (0., 0., .06, 0.), "wall_cable": (0., 0., 0., 0.), "conduit": (0., 0., 0., 0.),
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
        "cable": (W, .5, (H * .55, H - .3)),
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


@dataclass(frozen=True)
class WindowTrim:
    """Sill and head as build_structures makes them, from the fixture seed.

    Horns, the sill underside and the head stay inside the window pad, so a
    junction crack can sit on the same edge the mesh uses.
    """
    horn: float
    drop: float
    overhang: float
    rise: float
    projection: float
    profile: str


def window_trim(seed: int) -> WindowTrim:
    r = random.Random(int(seed) ^ 0x57494E44)
    return WindowTrim(
        horn=r.uniform(.045, .098),
        drop=r.uniform(.048, .068),
        overhang=r.uniform(.055, .098),
        rise=r.uniform(.08, .118),
        projection=r.uniform(.14, .22),
        profile=r.choice(("square", "nosing", "nosing", "step")),
    )


def _join_window_masonry() -> None:
    """One mesh per window, so the wall displacement does not split the sill from its nose."""
    import bmesh
    import bpy

    groups = {}
    for obj in bpy.data.objects:
        if obj.get("cwg_wall_finish") and obj.type == "MESH" and obj.data is not None and not obj.hide_render:
            groups.setdefault(obj.parent, []).append(obj)
    for parts in groups.values():
        if len(parts) < 2:
            continue
        bm = bmesh.new()
        for src in parts:
            tmp = bmesh.new()
            tmp.from_mesh(src.data)
            bmesh.ops.transform(tmp, verts=list(tmp.verts), matrix=src.matrix_local)
            copied = {v: bm.verts.new(v.co) for v in tmp.verts}
            bm.verts.ensure_lookup_table()
            for face in tmp.faces:
                try:
                    bm.faces.new([copied[v] for v in face.verts])
                except ValueError:
                    pass
            tmp.free()
        mesh = bpy.data.meshes.new(f"{parts[0].name}_masonry")
        bm.to_mesh(mesh)
        bm.free()
        joined = bpy.data.objects.new(mesh.name, mesh)
        collection = parts[0].users_collection[0] if parts[0].users_collection else bpy.context.scene.collection
        collection.objects.link(joined)
        joined.parent = parts[0].parent
        joined["cwg_wall_finish"] = True
        joined["cwg_role"] = parts[0].get("cwg_role")
        joined["cwg_hard_negative"] = parts[0].get("cwg_hard_negative")
        for src in parts:
            src.hide_render = True


def bind_window_masonry(wall, dicing_rate: float = 2.0) -> None:
    """Give sills and heads the inspection wall's material, as real displacement.

    The crack copy on the host stays private. The sill gets the wall surface,
    subdivided the same way, so the relief is geometry rather than a bump.
    """
    import bpy

    source = next((m for m in getattr(wall.data, "materials", ()) if m is not None and m.use_nodes), None)
    if source is None:
        return
    if source.name.endswith(".defects"):
        original = bpy.data.materials.get(source.name[: -len(".defects")])
        if original is not None and original.use_nodes:
            source = original
    material = source.copy()
    material.name = "CWG_Window_Wall"
    material.displacement_method = "DISPLACEMENT"
    _join_window_masonry()
    for obj in bpy.data.objects:
        if not obj.get("cwg_wall_finish") or obj.hide_render or obj.type != "MESH" or obj.data is None:
            continue
        mesh = obj.data
        mesh.materials.clear()
        mesh.materials.append(material)
        uv = mesh.uv_layers.active or mesh.uv_layers.new(name="UVMap")
        for poly in mesh.polygons:
            for li in poly.loop_indices:
                co = mesh.vertices[mesh.loops[li].vertex_index].co
                uv.data[li].uv = (co.x, co.z)
        if not any(mod.type == "SUBSURF" for mod in obj.modifiers):
            modifier = obj.modifiers.new("wall_adaptive", "SUBSURF")
            modifier.subdivision_type = "SIMPLE"
        obj.cycles.use_adaptive_subdivision = True
        obj.cycles.dicing_rate = dicing_rate


def built_extent(kind, u, v, w, h):
    """(u_lo, u_hi, v_lo, v_hi) on the wall covered by an assembly as
    build_structures makes it, hardware included (PAD)."""
    left, right, below, above = PAD.get(kind, (0., 0., 0., 0.))
    return u - w * .5 - left, u + w * .5 + right, v - h * .5 - below, v + h * .5 + above


def _overlaps(a, b, gap):
    return a[0] < b[1] + gap and b[0] < a[1] + gap and a[2] < b[3] + gap and b[2] < a[3] + gap


def sample_structure_plan(seed, spec, placements=(), profile="auto", include_kinds=None):
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
        pool = (list(KINDS[:5]) + ["trunking", "plumbing"] if profile == "services" else
                ["utility_box", "window", "panel_seam", "cable", *shared, *INDOOR_KINDS])
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


def build_structures(plans, relief_m=0.0, host_height_m=None):
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

        def tube(label, points, radius, mat=paint, smooth=False, resolution=3):
            curve = bpy.data.curves.new(prefix + '_' + label, 'CURVE')
            curve.dimensions = '3D'
            curve.bevel_depth = radius
            curve.bevel_resolution = resolution
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

        def finish(label, color, roughness=.4, metallic=0., variation=.025, scale=48., bump_m=.0004, detail=4.):
            mat = make_solid_material(prefix + '_' + label, color, roughness=roughness, metallic=metallic)
            # Noise lightens and darkens by the same amount, and the scale stays
            # fine. Coarse noise that also ran up toward white was the pale clouds.
            mat.displacement_method = 'BUMP'
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            bsdf = nodes['Principled BSDF']
            coord = nodes.new('ShaderNodeTexCoord')
            noise = nodes.new('ShaderNodeTexNoise')
            noise.inputs['Scale'].default_value = scale
            noise.inputs['Detail'].default_value = detail
            noise.inputs['Roughness'].default_value = .55
            links.new(coord.outputs['Object'], noise.inputs['Vector'])
            ramp = nodes.new('ShaderNodeValToRGB')
            ramp.color_ramp.elements[0].position = .38
            ramp.color_ramp.elements[0].color = tuple(max(0., c - variation) for c in color[:3]) + (1.,)
            ramp.color_ramp.elements[1].position = .62
            ramp.color_ramp.elements[1].color = tuple(min(1., c + variation) for c in color[:3]) + (1.,)
            links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
            links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
            if bump_m > 0:
                tooth = nodes.new('ShaderNodeTexNoise')
                tooth.inputs['Scale'].default_value = scale * r.uniform(2.5, 5.)
                tooth.inputs['Detail'].default_value = 6.0
                links.new(coord.outputs['Object'], tooth.inputs['Vector'])
                bump = nodes.new('ShaderNodeBump')
                bump.inputs['Strength'].default_value = .35
                bump.inputs['Distance'].default_value = bump_m
                links.new(tooth.outputs['Fac'], bump.inputs['Height'])
                links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
            return mat

        def relief(mat, scale=60., bump_m=.00035):
            """Fine bump on a flat fixture material. Glazing is left smooth."""
            mat.displacement_method = 'BUMP'
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            bsdf = nodes['Principled BSDF']
            coord = nodes.new('ShaderNodeTexCoord')
            tooth = nodes.new('ShaderNodeTexNoise')
            tooth.inputs['Scale'].default_value = scale
            tooth.inputs['Detail'].default_value = 6.0
            links.new(coord.outputs['Object'], tooth.inputs['Vector'])
            bump = nodes.new('ShaderNodeBump')
            bump.inputs['Strength'].default_value = .35
            bump.inputs['Distance'].default_value = bump_m
            links.new(tooth.outputs['Fac'], bump.inputs['Height'])
            links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
            return mat

        relief(paint, 70.)
        relief(dark, 90., .00025)
        relief(metal, 120., .00012)

        def grain(mat, strength):
            """Fine vertical grain. Mid-grey leaves the colour; the lines lighten and darken equally."""
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            bsdf = nodes['Principled BSDF']
            link = bsdf.inputs['Base Color'].links[0]
            src = link.from_socket
            links.remove(link)
            mapping = nodes.new('ShaderNodeMapping')
            mapping.inputs['Scale'].default_value = (r.uniform(28., 52.), 1., r.uniform(.7, 1.4))
            links.new(nodes.new('ShaderNodeTexCoord').outputs['Object'], mapping.inputs['Vector'])
            wave = nodes.new('ShaderNodeTexWave')
            wave.wave_type = 'BANDS'
            wave.bands_direction = 'X'
            wave.inputs['Scale'].default_value = 1.
            wave.inputs['Distortion'].default_value = r.uniform(.15, .7)
            if 'Detail' in wave.inputs:
                wave.inputs['Detail'].default_value = 0.
            links.new(mapping.outputs['Vector'], wave.inputs['Vector'])
            bands = nodes.new('ShaderNodeValToRGB')
            bands.color_ramp.elements[0].position = .4
            bands.color_ramp.elements[0].color = (.30, .30, .30, 1)
            bands.color_ramp.elements[1].position = .6
            bands.color_ramp.elements[1].color = (.70, .70, .70, 1)
            links.new(wave.outputs['Fac'], bands.inputs['Fac'])
            mix = nodes.new('ShaderNodeMixRGB')
            mix.blend_type = 'OVERLAY'
            mix.inputs['Fac'].default_value = strength
            links.new(src, mix.inputs['Color1'])
            links.new(bands.outputs['Color'], mix.inputs['Color2'])
            links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
            normal = bsdf.inputs['Normal']
            if normal.is_linked:
                tooth = nodes.new('ShaderNodeBump')
                tooth.inputs['Strength'].default_value = .3
                tooth.inputs['Distance'].default_value = .0002
                links.new(wave.outputs['Fac'], tooth.inputs['Height'])
                links.new(normal.links[0].from_socket, tooth.inputs['Normal'])
                links.new(tooth.outputs['Normal'], normal)

        def jitter(rgb, amount=.03):
            return tuple(min(1., max(0., c + r.uniform(-amount, amount))) for c in rgb)

        def pack(count, size, gap, limit):
            """Centred offsets for `count` items of width `size`. Returns (offsets, size)."""
            size = min(size, limit / max(count, 1))
            if count <= 1:
                return [0.], size
            gap = min(gap, max(0., (limit - count * size) / (count - 1)))
            if count * size + (count - 1) * gap > limit:
                size = (limit - (count - 1) * gap) / count
            step = size + gap
            start = -((count - 1) * step) * .5
            return [start + i * step for i in range(count)], size

        if p.kind == 'cable':
            # Loose indoor span: one to three insulated cables in clips, each a catenary.
            count = loose_cable_count(r)
            radius = r.uniform(.0032, .0055)
            sag = min(r.uniform(.05, .16), h * .42)
            pitch = r.uniform(.008, .016)
            offs = [(i - (count - 1) * .5) * pitch for i in range(count)]
            z_chord = z + min(h * .18, sag * .35)
            y0 = r.uniform(.018, .032)
            # One sag for the bundle, so every clip sits on the cables rather than above the droop.
            sag = min(sag, (z_chord + min(offs)) - (z - h * .5) - radius - .008)

            # The span is the full wall. Inset the ends so the clips and the
            # tube caps stay on the wall rather than past its sides.
            inset = .014

            def hang(t, off):
                droop = sag * math.sin(math.pi * t)
                along = (w - 2 * inset) * t + inset
                return (x - w * .5 + along, -(y0 + .006 * math.sin(math.pi * t)), z_chord + off - droop)

            for i, off in enumerate(offs):
                pts = [hang(step / 15, off) for step in range(16)]
                colour = (*jitter(r.choice(_INSULATION), .025), 1)
                tube('wire', pts, radius * r.uniform(.92, 1.08),
                     finish(f'insul{i}', colour, r.uniform(.42, .62), 0, variation=r.uniform(.015, .04),
                            scale=r.uniform(28, 70)), True)
            clip_h = (offs[-1] - offs[0]) + radius * 2 + .012 if count else .02
            clip = finish('clip', (*jitter(r.choice(_ENAMEL), .02), 1), r.uniform(.4, .6), 0,
                          variation=.015, scale=r.uniform(30, 60))
            groups = max(2, int(w / r.uniform(.32, .5)))
            for i in range(groups + 1):
                t = i / groups
                cx, cy, cz = hang(t, 0.)
                box('clip', (.016, .012, min(clip_h, h * .7)), (cx, cy + .005, cz), clip)
        elif p.kind == 'wall_cable':
            # Two, three or four insulated cables in clips. Colour is drawn per cable.
            horizontal = w >= h
            count = run_count(r)
            cross = h if horizontal else w
            radius = r.uniform(.0032, .0052)
            pitch = min(r.uniform(.012, .022), (cross - 2 * radius - .016) / max(count - 1, 1))
            span = w if horizontal else h
            groups = max(3, int(span / r.uniform(.28, .42)))
            for bundle in range(count):
                short = (bundle - (count - 1) * .5) * pitch
                colour = (*jitter(r.choice(_INSULATION), .02), 1)
                sag_amp = r.uniform(.0015, .0035)
                pts = []
                for step in range(13):
                    t = step / 12
                    sag = sag_amp * math.sin(t * math.pi * groups)
                    if horizontal:
                        pts.append((x - w * .5 + w * t, -.013 - sag, z + short))
                    else:
                        pts.append((x + short, -.013 - sag, z - h * .5 + h * t))
                tube('wire', pts, radius, finish(f'insul{bundle}', colour, r.uniform(.4, .6), 0,
                                                 variation=r.uniform(.015, .035), scale=r.uniform(24, 60)), True)
            clip = finish('clip', (*jitter(r.choice(_ENAMEL), .02), 1), r.uniform(.42, .6), 0, variation=.015)
            span_clip = (count - 1) * pitch + radius * 2 + .01
            for i in range(1, groups):
                t = i / groups
                if horizontal:
                    box('clip', (.016, .012, span_clip), (x - w * .5 + w * t, -.012, z), clip)
                else:
                    box('clip', (span_clip, .012, .016), (x, -.012, z - h * .5 + h * t), clip)
        elif p.kind == 'conduit':
            # Two to four PVC or EMT runs, each with couplers and saddle straps.
            horizontal = w >= h
            cross = h if horizontal else w
            count = run_count(r)
            metal_run = r.random() < .3
            if metal_run:
                rgb, rough, metallic = jitter((.62, .63, .61), .03), r.uniform(.22, .38), r.uniform(.75, .95)
            else:
                rgb = jitter(r.choice(_PIPE[:4] + _PIPE[6:]), .03)
                rough, metallic = r.uniform(.28, .48), 0.
            body = finish('body', (*rgb, 1), rough, metallic, variation=r.uniform(.015, .04),
                          scale=r.uniform(36, 90))
            strap = body if metal_run else finish('strap', (*jitter(tuple(c * .92 for c in rgb), .015), 1),
                                                  r.uniform(.4, .55), 0, variation=.015)
            gap = r.uniform(.008, .014)
            offsets, lane = pack(count, r.uniform(.02, .032), gap, cross)
            radius = max(.006, lane * .38)
            y = -(radius + .016)
            span = w if horizontal else h

            def at(frac, offset, delta=0.):
                if horizontal:
                    return (x - w * .5 + w * frac + delta, y, z + offset)
                return (x + offset, y, z - h * .5 + h * frac + delta)

            joints = max(2, int(span / 1.5)) if span >= 1.2 else 1
            straps = max(4, int(span / r.uniform(.48, .7)))
            band = min(lane * .9, radius * 2 + .012)
            for offset in offsets:
                tube('run', [at(0., offset), at(1., offset)], radius, body)
                for i in range(1, joints):
                    tube('coupling', [at(i / joints, offset, -.018), at(i / joints, offset, .018)],
                         radius * 1.22, body)
                for i in range(straps):
                    loc = at((i + .5) / straps, offset)
                    dims = (.014, abs(y), band) if horizontal else (band, abs(y), .014)
                    box('saddle', dims, (loc[0], y * .4, loc[2]), strap)
                    front = y - radius - .003
                    band_dims = (.012, .004, band) if horizontal else (band, .004, .012)
                    box('strap', band_dims, (loc[0], front, loc[2]), strap)
        elif p.kind == 'downpipe':
            # Two to four pipes, socket joints and stand-off clips. Colour is one family per run.
            horizontal = w >= h
            cross = h if horizontal else w
            count = run_count(r)
            rgb = jitter(r.choice(_PIPE), .03)
            metallic = r.uniform(.45, .75) if r.random() < .2 else 0.
            body = finish('pipe', (*rgb, 1), r.uniform(.2, .42), metallic, variation=r.uniform(.015, .04),
                          scale=r.uniform(24, 70))
            gap = r.uniform(.012, .022)
            offsets, lane = pack(count, r.uniform(.04, .07), gap, cross)
            radius = max(.014, lane * .36)
            y = -(radius + .016)
            span = w if horizontal else h

            def at(frac, offset, delta=0.):
                if horizontal:
                    return (x - w * .5 + w * frac + delta, y, z + offset)
                return (x + offset, y, z - h * .5 + h * frac + delta)

            sockets = (.5,) if span < 1.6 else (.33, .66)
            clips = max(3, int(span / r.uniform(.6, .85)))
            shoulder = min(lane * .92, radius * 2 + .02)
            for offset in offsets:
                tube('run', [at(0., offset), at(1., offset)], radius, body)
                for frac in sockets:
                    tube('socket', [at(frac, offset, -.024), at(frac, offset, .024)], radius * 1.14, body)
                for i in range(1, clips):
                    loc = at(i / clips, offset)
                    if horizontal:
                        box('backplate', (.02, .01, shoulder), (loc[0], y * .28, loc[2]), body)
                        box('strap', (.014, .005, shoulder), (loc[0], y - radius - .003, loc[2]), body)
                    else:
                        box('backplate', (shoulder, .01, .02), (loc[0], y * .28, loc[2]), body)
                        box('strap', (shoulder, .005, .014), (loc[0], y - radius - .003, loc[2]), body)
        elif p.kind == 'utility_box':
            # A shut steel enclosure: hinged door, screwed cover, meter, or louvres.
            # Paint or galvanizing, and a conduit into the top. The feed above the
            # box stays inside the 0.25 m pad unless the host height is known.
            style = r.choice(('hinged', 'hinged', 'screwed', 'meter', 'louver'))
            hinged = style in ('hinged', 'louver') or (style == 'meter' and r.random() < .7)
            enamels = (
                (.78, .58, .10), (.82, .40, .12), (.28, .46, .30), (.18, .32, .26),
                (.78, .79, .80), (.50, .52, .54), (.86, .83, .74), (.90, .89, .86),
                (.20, .21, .23), (.28, .38, .50),
            )
            if r.random() < .62:
                rgb = jitter(r.choice(enamels), .02)
                rough, metalness = r.uniform(.30, .52), 0.
            else:
                rgb = jitter(r.choice(((.72, .74, .76), (.60, .62, .64), (.46, .48, .50))), .012)
                rough, metalness = r.uniform(.20, .40), r.uniform(.55, .9)
            body_mat = finish('body', (*rgb, 1), rough, metalness, variation=r.uniform(.012, .028),
                              scale=r.uniform(45, 100), bump_m=.00028)
            shift = 0. if r.random() < .62 else r.uniform(.05, .12) * r.choice((-1., 1.))
            door_rgb = tuple(min(1., max(.02, c + shift)) for c in rgb)
            door_mat = finish('door', (*door_rgb, 1), r.uniform(.26, .48), metalness * .85,
                              variation=r.uniform(.01, .022), scale=r.uniform(60, 120), bump_m=.00022)
            hw_rgb, hw_rough, hw_met = r.choice((
                ((.74, .75, .77), .22, .85), ((.10, .10, .11), .4, .25), ((.76, .62, .34), .3, .8),
            ))
            hardware = finish('hw', (*jitter(hw_rgb, .01), 1), hw_rough, hw_met,
                              variation=.006, scale=90., bump_m=.0001)
            depth = r.uniform(.062, .094)
            back = .003
            front = -(back + depth)
            body_y = -(back + depth * .5)
            door_t = .009
            door_y = front - .002 - door_t * .5
            inset = r.uniform(.009, .015)
            door_w, door_h = w - 2 * inset, h - 2 * inset
            lip = r.uniform(.007, .012)
            lip_t = .0035
            lip_y = door_y - door_t * .5 - lip_t * .5
            box('housing', (w, depth, h), (x, body_y, z), body_mat)
            box('hood', (w * .98, .014, .009), (x, front - .006, z + h * .5 - .005), body_mat)
            box('gasket', (door_w + .003, .003, door_h + .003), (x, front - .0015, z), dark)
            box('door', (door_w, door_t, door_h), (x, door_y, z), door_mat)
            box('lip_l', (lip, lip_t, door_h), (x - door_w * .5 + lip * .5, lip_y, z), door_mat)
            box('lip_r', (lip, lip_t, door_h), (x + door_w * .5 - lip * .5, lip_y, z), door_mat)
            box('lip_b', (door_w, lip_t, lip), (x, lip_y, z - door_h * .5 + lip * .5), door_mat)
            box('lip_t', (door_w, lip_t, lip), (x, lip_y, z + door_h * .5 - lip * .5), door_mat)
            if hinged:
                side = r.choice((-1., 1.))
                hinge_n = r.choice((2, 3))
                hx = x + side * (w * .5 - .009)
                for i in range(hinge_n):
                    hz = z - h * .32 + h * .64 * i / max(hinge_n - 1, 1)
                    tube('hinge', [(hx, door_y, hz - .013), (hx, door_y, hz + .013)], .005, hardware)
                    box('leaf', (.016, .004, .02), (hx - side * .01, door_y + .002, hz), hardware)
                lx = x - side * (door_w * .5 - lip - .006)
                if r.random() < .55:
                    box('hasp', (.02, .005, .034), (lx, lip_y - .0025, z - door_h * .08), hardware)
                    tube('staple', [(lx, lip_y - .006, z - door_h * .08 - .007),
                                    (lx, lip_y - .006, z - door_h * .08 + .007)], .0024, hardware)
                else:
                    box('lockplate', (.026, .003, .026), (lx, lip_y - .0015, z), hardware)
                    tube('lock', [(lx, lip_y, z), (lx, lip_y - .007, z)], .0055, hardware)
            else:
                for fx in (-1., 1.):
                    for fz in (-1., 1.):
                        sx = x + fx * (door_w * .5 - lip * .6)
                        sz = z + fz * (door_h * .5 - lip * .6)
                        tube('screw', [(sx, lip_y, sz), (sx, lip_y - .004, sz)], .0032, hardware)
            if style == 'meter':
                face = finish('meter', (*jitter((.05, .07, .09), .01), 1), .18, .15,
                              variation=.008, scale=80., bump_m=.00008)
                inner = min(door_w, door_h) * .5 - lip
                rim = min(inner * .62, min(door_w, door_h) * r.uniform(.18, .26))
                mx = x + r.uniform(-door_w, door_w) * .06
                mz = z + door_h * r.uniform(0., .1)
                tube('bezel', [(mx, lip_y, mz), (mx, lip_y - .005, mz)], rim, hardware, resolution=8)
                tube('glass', [(mx, lip_y - .003, mz), (mx, lip_y - .007, mz)], rim * .8, face, resolution=8)
                plate_z = mz - rim - lip * 1.6
                if plate_z - lip > z - door_h * .5:
                    box('plate', (rim * 1.5, .0025, lip * 1.4), (mx, lip_y - .001, plate_z), door_mat)
            elif style == 'louver':
                slots = r.randint(4, 6)
                span = door_h - 4 * lip
                band = span * .62
                pitch = band / slots
                slat_w = door_w - 4 * lip
                for i in range(slots):
                    sz = z + band * .5 - pitch * (i + .5)
                    box('slot', (slat_w, .004, pitch * .42), (x, door_y, sz), dark)
                    box('slat', (slat_w, .003, pitch * .22), (x, lip_y, sz + pitch * .1), door_mat)
            if r.random() < .5:
                knock = r.choice((-1., 1.))
                sz = z + r.uniform(-h * .15, h * .15)
                tube('knock', [(x + knock * (w * .5 - .018), body_y, sz),
                               (x + knock * w * .5, body_y, sz)], r.uniform(.006, .01), body_mat)
            entries = [r.choice((-1., 1.))]
            if w > .22 and r.random() < .35:
                entries.append(-entries[0])
            radius = r.uniform(.008, .013)
            feed_rgb = jitter(r.choice(((.86, .86, .84), (.62, .64, .66), (.55, .56, .58), (.78, .79, .80))), .015)
            feed_met = r.uniform(.45, .8) if feed_rgb[0] < .8 else 0.
            riser_mat = finish('riser', (*feed_rgb, 1), r.uniform(.28, .5), feed_met,
                               variation=r.uniform(.008, .02), scale=r.uniform(30, 70), bump_m=.0002)
            top = z + h * .5
            for entry in entries:
                kx = min(x + w * .5 - .02, max(x - w * .5 + .02, x + entry * w * .28))
                tube('hub', [(kx, body_y, top - .012), (kx, body_y, top + .01)], radius * 1.45, body_mat)
                riser_top = host_height_m if host_height_m else top + .2
                tube('riser', [(kx, body_y, top + .004), (kx, body_y, riser_top)], radius, riser_mat)
                clip_z = top + .045
                clip_ceil = riser_top - .04 if host_height_m else top + .2
                while clip_z < clip_ceil:
                    box('clip', (.028, .01, .01), (kx, body_y * .35, clip_z), riser_mat)
                    clip_z += r.uniform(.22, .32)
            if r.random() < .4:
                px = min(x + w * .5 - .02, max(x - w * .5 + .02, x - entries[0] * w * .22))
                tube('plug', [(px, body_y, z - h * .5), (px, body_y, z - h * .5 + .016)], radius * .9, body_mat)
        elif p.kind == 'window':
            # A framed opening on the solid host. The sill and head are the wall,
            # dressed later with its material. The pane is a pale mirror, so it
            # reads as glass without emitting light onto the frame. A dark
            # backing behind it still hides the host.
            trim = window_trim(p.seed)
            z0, z1 = z - h * .5, z + h * .5
            sill_bot, sill_top = z0 - trim.drop, z0 - .004
            sill_thick = sill_top - sill_bot
            head_top, head_bot = z1 + trim.rise, z1 - .01
            head_thick = head_top - head_bot
            sill_w, head_w = w + 2 * trim.horn, w + 2 * trim.overhang

            def masonry(label, dims, loc):
                obj = box(label, dims, loc, paint)
                obj["cwg_wall_finish"] = True
                return obj

            embed, depth = .006, trim.projection
            back_y, front_y = embed, embed - depth
            if trim.profile == "square":
                masonry('sill', (sill_w, depth, sill_thick), (x, (back_y + front_y) * .5, (sill_bot + sill_top) * .5))
                nose_front = front_y
            else:
                main_d = depth * (.62 if trim.profile == "step" else .7)
                nose_d = depth - main_d
                nose_thick = sill_thick * (.5 if trim.profile == "step" else .62)
                masonry('sill', (sill_w, main_d, sill_thick), (x, back_y - main_d * .5, (sill_bot + sill_top) * .5))
                masonry('nose', (sill_w, nose_d, nose_thick),
                        (x, front_y + nose_d * .5, sill_top - nose_thick * .5))
                nose_front = front_y
            box('drip', (sill_w * .9, .012, .006),
                (x, nose_front + .014, sill_bot + .003), dark)
            head_d = depth * .58
            head_back = .004
            masonry('head', (head_w, head_d, head_thick),
                    (x, head_back - head_d * .5, (head_bot + head_top) * .5))
            if trim.profile != "square":
                lip_d = min(.035, head_d * .4)
                masonry('head_lip', (head_w, lip_d, .016),
                        (x, head_back - head_d - lip_d * .5, head_bot + .008))

            frames = (
                ((.90, .90, .88), 0., .38, False), ((.84, .80, .72), 0., .42, False),
                ((.16, .17, .18), .1, .32, False), ((.07, .07, .08), .2, .28, False),
                ((.62, .40, .20), 0., .45, True), ((.32, .18, .09), 0., .4, True),
                ((.74, .75, .77), .88, .22, False), ((.55, .40, .26), .72, .3, False),
            )
            rgb, metallic, rough, woody = r.choice(frames)
            frame = finish('frame', (*jitter(rgb, .012), 1), rough, metallic,
                           variation=r.uniform(.01, .025), scale=r.uniform(50, 110), bump_m=.00022)
            if woody:
                grain(frame, r.uniform(.4, .65))
            hw_rgb, hw_rough, hw_met = r.choice((
                ((.74, .75, .77), .2, .88), ((.12, .12, .13), .38, .3), ((.76, .62, .34), .28, .85),
            ))
            hardware = finish('hw', (*jitter(hw_rgb, .01), 1), hw_rough, hw_met,
                              variation=.006, scale=80., bump_m=.0001)
            tint = jitter(r.choice(((.93, .95, .97), (.96, .96, .95), (.93, .97, .94), (.97, .95, .91))), .008)
            glass = make_solid_material(prefix + '_glazing', (*tint, 1), roughness=r.uniform(.02, .08), metallic=1.)

            sec = r.uniform(.04, .052)
            fd = r.uniform(.05, .068)
            frame_back = -.012
            frame_y = frame_back - fd * .5
            span0, span1 = sill_top, head_bot
            span_h = span1 - span0
            span_z = (span0 + span1) * .5
            for side in (-1., 1.):
                box('jamb', (sec, fd, span_h), (x + side * w * .5, frame_y, span_z), frame)
            rail_h = sec * .78
            box('sill_rail', (w + sec, fd, rail_h), (x, frame_y, span0 + rail_h * .5), frame)
            head_rail = sec * .84
            box('head_rail', (w + sec, fd, head_rail), (x, frame_y, span1 - head_rail * .5), frame)
            gx0, gx1 = x - w * .5 + sec * .62, x + w * .5 - sec * .62
            gz0, gz1 = span0 + rail_h, span1 - head_rail
            inner_w, inner_h = gx1 - gx0, gz1 - gz0
            style = r.choice(('single', 'twin', 'twin', 'transom', 'grid'))
            bar = r.uniform(.016, .024)
            if style == 'single':
                cells = [(x, (gz0 + gz1) * .5, inner_w, inner_h)]
            elif style == 'twin':
                pw = (inner_w - bar) * .5
                cells = [(gx0 + pw * .5, (gz0 + gz1) * .5, pw, inner_h),
                         (gx1 - pw * .5, (gz0 + gz1) * .5, pw, inner_h)]
                box('mullion', (bar, fd, inner_h), (x, frame_y, (gz0 + gz1) * .5), frame)
            elif style == 'transom':
                top_h = max(.08, inner_h * .3 - bar * .5)
                bot_h = inner_h - top_h - bar
                cells = [(x, gz1 - top_h * .5, inner_w, top_h),
                         (x, gz0 + bot_h * .5, inner_w, bot_h)]
                box('transom', (inner_w, fd, bar), (x, frame_y, gz0 + bot_h + bar * .5), frame)
            else:
                pw, ph = (inner_w - bar) * .5, (inner_h - bar) * .5
                cells = []
                for cx in (gx0 + pw * .5, gx1 - pw * .5):
                    for cz, ch in ((gz0 + ph * .5, ph), (gz1 - ph * .5, ph)):
                        cells.append((cx, cz, pw, ch))
                box('mullion', (bar, fd, inner_h), (x, frame_y, (gz0 + gz1) * .5), frame)
                box('transom', (inner_w, fd, bar), (x, frame_y, (gz0 + gz1) * .5), frame)
            pane_y = frame_y - fd * .5 + .02
            for i, (cx, cz, pw, ph) in enumerate(cells):
                box(f'pane{i}', (max(.02, pw - .004), .006, max(.02, ph - .004)), (cx, pane_y, cz), glass)
            box('backing', (max(.04, inner_w - .02), .012, max(.04, inner_h - .02)),
                (x, pane_y + .012, (gz0 + gz1) * .5), dark)
            hand = r.choice((-1., 1.))
            latch_z = gz0 + (gz1 - gz0) * (.42 if style in ('single', 'twin') else .28)
            lx = x + hand * (w * .5 - sec * .35)
            box('handle_plate', (.012, .008, .055), (lx, frame_y - fd * .5 - .004, latch_z), hardware)
            tube('handle', [(lx, frame_y - fd * .5 - .012, latch_z + .01),
                            (lx - hand * .07, frame_y - fd * .5 - .012, latch_z + .01)], .0045, hardware)
            hinge_x = x - hand * (w * .5 - .004)
            for frac in (.3, .72):
                hz = span0 + span_h * frac
                tube('hinge', [(hinge_x, frame_y, hz - .012), (hinge_x, frame_y, hz + .012)], .0045, hardware)
        elif p.kind == 'panel_seam':
            # One construction joint. Colour, width and roughness are drawn; never a crack field.
            horizontal = w >= h
            colour = (*jitter(r.choice(_SEALANT), .03), 1)
            seal = finish('sealant', colour, r.uniform(.32, .88), 0., variation=r.uniform(.02, .08),
                          scale=r.uniform(18, 80))
            thin = (h if horizontal else w) * r.uniform(.4, 1.)
            proud = r.uniform(.003, .008)
            if horizontal:
                box('sealant', (w, proud, thin), (x, -proud * .5 - .002, z), seal)
            else:
                box('sealant', (thin, proud, h), (x, -proud * .5 - .002, z), seal)
        elif p.kind == 'door':
            # Closed leaf: flush or raised panels. The lining is white, cream,
            # stained, or the same finish as the leaf, never a flat grey casing.
            woods = (
                ((.74, .52, .30), False), ((.60, .36, .16), False), ((.46, .24, .11), False),
                ((.26, .15, .08), False), ((.90, .89, .86), True), ((.80, .74, .62), True),
                ((.11, .10, .09), True),
            )
            leaf_rgb, painted = r.choice(woods)
            leaf_rgb = jitter(leaf_rgb, .015 if painted else .03)
            if painted:
                leaf = finish('leaf', (*leaf_rgb, 1), r.uniform(.28, .5), 0,
                              variation=r.uniform(.015, .03), scale=r.uniform(140, 220),
                              bump_m=.00025, detail=6.)
            else:
                leaf = finish('leaf', (*leaf_rgb, 1), r.uniform(.32, .52), 0,
                              variation=r.uniform(.03, .05), scale=r.uniform(120, 190),
                              bump_m=.00035, detail=8.)
                grain(leaf, r.uniform(.55, .78))
            if r.random() < .45:
                frame_rgb = leaf_rgb
            else:
                frame_rgb = jitter(r.choice(((.93, .92, .89), (.84, .78, .66), (.24, .14, .08), (.07, .07, .07))), .015)
            frame = finish('frame', (*frame_rgb, 1), r.uniform(.28, .48), 0,
                           variation=r.uniform(.018, .035), scale=r.uniform(100, 170),
                           bump_m=.00025, detail=6.)
            if not painted and frame_rgb == leaf_rgb:
                grain(frame, r.uniform(.45, .7))
            mrgb, mrough, mmet = r.choice((
                ((.74, .75, .76), .2, .88), ((.76, .62, .34), .28, .9),
                ((.07, .07, .08), .42, .2), ((.86, .87, .88), .1, .96),
            ))
            hardware = finish('hardware', (*jitter(mrgb, .012), 1), mrough, mmet,
                              variation=.008, scale=80., bump_m=.00012)
            casing = r.uniform(.042, .058)
            box('jamb_l', (casing, .065, h - .006), (x - w * .5 + casing * .5 + .003, -.038, z), frame)
            box('jamb_r', (casing, .065, h - .006), (x + w * .5 - casing * .5 - .003, -.038, z), frame)
            box('head', (w - .006, .065, casing), (x, -.038, z + h * .5 - casing * .5 - .003), frame)
            leaf_w = w - 2 * casing - .01
            leaf_h = h - casing - .016
            leaf_z = z - h * .5 + .01 + leaf_h * .5
            box('reveal', (leaf_w + .008, .01, leaf_h + .006), (x, -.02, leaf_z), dark)
            stile = r.uniform(.072, .10)
            panels = r.choice((2, 3, 3, 3, 4, 0))
            top_rail = r.uniform(.075, .11)
            bot_rail = r.uniform(.13, .18)
            mid = r.uniform(.05, .08)
            inner_h = leaf_h - top_rail - bot_rail - max(panels - 1, 0) * mid
            if panels and inner_h < .3:
                panels = 0
            if panels == 0:
                box('leaf', (leaf_w, .026, leaf_h), (x, -.04, leaf_z), leaf)
            else:
                fracs = {2: (.54, .46), 3: (.42, .2, .38), 4: (.25, .25, .25, .25)}[panels]
                box('stile_l', (stile, .03, leaf_h), (x - leaf_w * .5 + stile * .5, -.042, leaf_z), leaf)
                box('stile_r', (stile, .03, leaf_h), (x + leaf_w * .5 - stile * .5, -.042, leaf_z), leaf)
                inner_w = leaf_w - 2 * stile
                base = leaf_z - leaf_h * .5
                box('rail_bot', (inner_w, .03, bot_rail), (x, -.042, base + bot_rail * .5), leaf)
                cursor = base + bot_rail
                for i, frac in enumerate(fracs):
                    ph = max(.04, inner_h * frac)
                    pc = cursor + ph * .5
                    mould = min(.016, ph * .14, inner_w * .08)
                    box('recess', (inner_w - .006, .01, max(.012, ph - .006)), (x, -.03, pc), leaf)
                    box('field', (max(.02, inner_w - 2 * mould), .018, max(.012, ph - 2 * mould)), (x, -.042, pc), leaf)
                    box('lip_l', (mould, .024, ph), (x - inner_w * .5 + mould * .5, -.048, pc), leaf)
                    box('lip_r', (mould, .024, ph), (x + inner_w * .5 - mould * .5, -.048, pc), leaf)
                    box('lip_b', (inner_w, .024, mould), (x, -.048, pc - ph * .5 + mould * .5), leaf)
                    box('lip_t', (inner_w, .024, mould), (x, -.048, pc + ph * .5 - mould * .5), leaf)
                    cursor += ph
                    if i < panels - 1:
                        box('rail', (inner_w, .03, mid), (x, -.042, cursor + mid * .5), leaf)
                        cursor += mid
                box('rail_top', (inner_w, .03, top_rail), (x, -.042, base + leaf_h - top_rail * .5), leaf)
            hand = r.choice((-1., 1.))
            hx = x + hand * (leaf_w * .5 - max(stile, .08) * .4)
            hz = min(base + r.uniform(.96, 1.08) if panels else leaf_z - leaf_h * .5 + r.uniform(.96, 1.08),
                     leaf_z + leaf_h * .5 - .18)
            box('rose', (.05, .01, .085), (hx, -.056, hz), hardware)
            tube('lever', [(hx, -.064, hz + .008), (hx - hand * .10, -.064, hz + .008)], .007, hardware)
            tube('lock', [(hx, -.054, hz - .055), (hx, -.068, hz - .055)], .005, hardware)
            hinge_x = x - hand * (leaf_w * .5 - .01)
            hinge_n = r.choice((2, 3))
            for i in range(hinge_n):
                hz_h = leaf_z - leaf_h * .5 + leaf_h * (i + 1) / (hinge_n + 1)
                box('hinge', (.016, .008, .065), (hinge_x, -.05, hz_h), hardware)
            box('threshold', (leaf_w, .035, .01), (x, -.028, z - h * .5 + .012), frame)
        elif p.kind == 'trunking':
            # Two to four PVC lids side by side. Colour is one family, with a lid seam and screws.
            rgb = jitter(r.choice(_ENAMEL), .025)
            pvc = finish('trunk', (*rgb, 1), r.uniform(.32, .55), 0, variation=r.uniform(.012, .035),
                         scale=r.uniform(24, 70))
            seam = finish('seam', (*jitter(tuple(c * r.uniform(.7, .85) for c in rgb), .01), 1),
                          r.uniform(.45, .65), 0, variation=.02)
            horizontal = w >= h
            lo, hi = (x - w * .5, x + w * .5) if horizontal else (z - h * .5, z + h * .5)
            cross = h if horizontal else w
            lanes = run_count(r)
            gap = r.uniform(.004, .009)
            offsets, duct = pack(lanes, r.uniform(.016, .032), gap, cross)
            depth = r.uniform(.012, .022)
            pieces = max(1, math.ceil((hi - lo) / 2.0))
            length = (hi - lo) / pieces
            piece_gap = .003 if pieces > 1 else 0.
            for offset in offsets:
                for j in range(pieces):
                    c = lo + (j + .5) * length
                    piece = length - piece_gap
                    if horizontal:
                        box('section', (piece, depth, duct), (c, -depth * .5 - .003, z + offset), pvc)
                        box('lid', (piece - .002, .004, max(.008, duct - .008)),
                            (c, -depth - .005, z + offset), pvc)
                        box('groove', (piece - .01, .0014, min(.0035, duct * .2)),
                            (c, -depth - .008, z + offset), seam)
                    else:
                        box('section', (duct, depth, piece), (x + offset, -depth * .5 - .003, c), pvc)
                        box('lid', (max(.008, duct - .008), .004, piece - .002),
                            (x + offset, -depth - .005, c), pvc)
                        box('groove', (min(.0035, duct * .2), .0014, piece - .01),
                            (x + offset, -depth - .008, c), seam)
            screws = max(2, int((hi - lo) / r.uniform(.4, .55)))
            for offset in offsets:
                for i in range(1, screws):
                    c = lo + (hi - lo) * i / screws
                    if horizontal:
                        tube('screw', [(c, -depth - .004, z + offset), (c, -depth - .010, z + offset)], .0022, metal)
                    else:
                        tube('screw', [(x + offset, -depth - .004, c), (x + offset, -depth - .010, c)], .0022, metal)
        elif p.kind in ('vent', 'access_panel'):
            def rails(border, depth, y, mat, tag):
                box(tag + '_stile_l', (border, depth, h), (x - w * .5 + border * .5, y, z), mat)
                box(tag + '_stile_r', (border, depth, h), (x + w * .5 - border * .5, y, z), mat)
                box(tag + '_rail_b', (w - 2 * border, depth, border), (x, y, z - h * .5 + border * .5), mat)
                box(tag + '_rail_t', (w - 2 * border, depth, border), (x, y, z + h * .5 - border * .5), mat)

            if p.kind == 'vent':
                # Register: drawn frame colour, darker or matching slats, one screw each side.
                rgb = jitter(r.choice(_ENAMEL), .03)
                frame = finish('frame', (*rgb, 1), r.uniform(.3, .58),
                               r.uniform(.15, .4) if r.random() < .25 else 0.,
                               variation=r.uniform(.015, .045), scale=r.uniform(22, 80))
                if r.random() < .3:
                    slat_mat = frame
                else:
                    shade = r.uniform(.12, .4)
                    slat_mat = finish('slat', (*tuple(c * shade for c in rgb), 1),
                                      r.uniform(.4, .7), 0, variation=.03, scale=r.uniform(30, 80))
                border = min(r.uniform(.012, .022), h * .22, w * .08)
                rails(border, .014, -.014, frame, 'frame')
                open_w, open_h = w - 2 * border - .004, h - 2 * border - .004
                box('cavity', (open_w, .008, open_h), (x, -.006, z), dark)
                count = max(5, int(open_h / r.uniform(.009, .016)))
                thick = min(.004, open_h / count * .45)
                for j in range(count):
                    zz = z - open_h * .5 + open_h * (j + .5) / count
                    box('slat', (open_w - .004, .01, thick), (x, -.016, zz), slat_mat)
                for sx in (x - w * .5 + border * .5, x + w * .5 - border * .5):
                    tube('screw', [(sx, -.018, z), (sx, -.024, z)], .004, metal)
            else:
                # Closed hatch. No hinges: a drawn painted or metal face, a latch and corner screws.
                painted = r.random() < .6
                if painted:
                    rgb = jitter(r.choice(_ENAMEL), .03)
                    panel = finish('panel', (*rgb, 1), r.uniform(.32, .6), 0.,
                                   variation=r.uniform(.015, .045), scale=r.uniform(22, 75))
                else:
                    rgb = jitter(r.choice(((.82, .83, .84), (.70, .72, .74), (.52, .53, .55), (.32, .33, .35))), .02)
                    panel = finish('panel', (*rgb, 1), r.uniform(.18, .42), r.uniform(.7, .95),
                                   variation=r.uniform(.02, .05), scale=r.uniform(40, 110))
                border = min(r.uniform(.02, .04), w * .12, h * .12)
                rails(border, .012, -.012, panel, 'frame')
                open_w, open_h = w - 2 * border, h - 2 * border
                box('cavity', (open_w + .004, .01, open_h + .004), (x, -.004, z), dark)
                reveal = .004
                door_w, door_h = open_w - reveal, open_h - reveal
                box('door', (door_w, .008, door_h), (x, -.018, z), panel)
                lx = x + door_w * .5 - .02
                tube('latch', [(lx, -.020, z), (lx, -.030, z)], .006, metal)
                for sx in (x - w * .5 + border * .5, x + w * .5 - border * .5):
                    for sz in (z - h * .5 + border * .5, z + h * .5 - border * .5):
                        tube('screw', [(sx, -.016, sz), (sx, -.024, sz)], .0035, metal)
        elif p.kind == 'plumbing':
            # Two to four pipes. One metal or painted family, fittings, clips and a stopcock.
            horizontal = w >= h
            cross = h if horizontal else w
            count = run_count(r)
            families = (
                ((.72, .42, .26), .28, .92),  # copper
                ((.62, .48, .24), .34, .88),  # brass
                ((.80, .81, .80), .14, .96),  # chrome
                ((.88, .88, .86), .4, 0.),    # white
                ((.14, .14, .15), .42, .2),   # black iron
                ((.64, .66, .67), .3, .72),   # galvanized
            )
            rgb, rough, metallic = r.choice(families)
            rgb = jitter(rgb, .03)
            pipe_mat = finish('pipe', (*rgb, 1), rough, metallic, variation=r.uniform(.02, .05),
                              scale=r.uniform(40, 100))
            fit_mat = pipe_mat if metallic > .5 else finish(
                'fitting', (*jitter(tuple(min(1., c * 1.05) for c in rgb), .02), 1),
                min(.6, rough + .08), metallic, variation=.03)
            clip_mat = finish('clip', (*jitter(r.choice(_ENAMEL), .02), 1), r.uniform(.4, .58), 0, variation=.015)
            gap = r.uniform(.012, .02)
            offsets, lane = pack(count, r.uniform(.018, .028), gap, cross)
            radius = max(.005, lane * .32)
            y = -(radius + .02)
            valve_at = offsets[r.randrange(count)]
            for short in offsets:
                if horizontal:
                    tube('pipe', [(x - w * .5, y, z + short), (x + w * .5, y, z + short)], radius, pipe_mat)
                else:
                    tube('pipe', [(x + short, y, z - h * .5), (x + short, y, z + h * .5)], radius, pipe_mat)
                for frac in (.22, .5, .78):
                    if horizontal:
                        cx = x - w * .5 + w * frac
                        box('clip', (.012, .016, min(lane * .85, .024)), (cx, y * .45, z + short), clip_mat)
                        tube('union', [(cx - .01, y, z + short), (cx + .01, y, z + short)], radius * 1.4, fit_mat)
                    else:
                        cz = z - h * .5 + h * frac
                        box('clip', (min(lane * .85, .024), .016, .012), (x + short, y * .45, cz), clip_mat)
                        tube('union', [(x + short, y, cz - .01), (x + short, y, cz + .01)], radius * 1.4, fit_mat)
            if horizontal:
                tube('valve_body', [(x - .016, y, z + valve_at), (x + .016, y, z + valve_at)], radius * 1.7, fit_mat)
                tube('valve_stem', [(x, y, z + valve_at), (x, y - .028, z + valve_at)], .004, fit_mat)
                box('valve_lever', (.032, .006, min(.01, lane * .5)), (x, y - .032, z + valve_at), fit_mat)
            else:
                tube('valve_body', [(x + valve_at, y, z - .016), (x + valve_at, y, z + .016)], radius * 1.7, fit_mat)
                tube('valve_stem', [(x + valve_at, y, z), (x + valve_at, y - .028, z)], .004, fit_mat)
                box('valve_lever', (min(.01, lane * .5), .006, .032), (x + valve_at, y - .032, z), fit_mat)
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
