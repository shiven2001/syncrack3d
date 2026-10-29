"""Infinigen props in front of the controlled wall.

Why a bare wall is not enough. A detector trained only on clean plaster with a
defect in the middle learns "anomaly on flat surface", and then fires on every
socket, skirting joint, pipe bracket and furniture edge in a real photo. The
apartment pipeline supplied those distractors for free; a flat-wall generator
has to put them back deliberately, or it trades one narrow distribution for a
narrower one.

Props do three jobs here, all of which matter more than looking furnished:

* **Occlusion.** A defect partly hidden behind an object is the normal case in a
  real inspection photo and never occurs in a clean synthetic close-up.
* **Cast shadow and edges.** A hard shadow edge across plaster, a sealant line
  around a frame or a cabinet edge against the wall are the most common false
  positives for a crack detector. The model needs to see them labelled
  background, and to see real cracks running right beside them.
* **Scale and clutter cues.** Something of known size in frame is what stops the
  network keying on absolute pixel width, which is the cue that does not survive
  a change of camera.

Each indoor wall draws a room (kitchen, bathroom, living room, bedroom or
utility) and then props from that room's list, so a toilet does not stand beside
a sofa. The bathroom, kitchen and fixture assets are the vendored Infinigen
versions (stainless and chrome fittings, vanity units). Every prop is
loose: it stands on the floor anywhere in front of the wall, from the wall gap
out to `MAX_STANDOFF_M`, turned any way.
A prop in the way of something steps further from the wall within that band, or
tries another spot.

Geometry is Infinigen 1's; surfaces are Infinigen 2's. Each part's v1 material
is swapped for one from the Infinigen 2 object picker for its role (upholstery,
cabinet, table top, steel, glass...), on metre box-projected UVs, as bump.
Printed labels, plants and soil keep their v1 material.

Every prop is placed by its measured bounds (Infinigen origins are often at a
corner or mid-height) and must clear the wall, the floor, the ceiling, the side
returns, the far edge of the floor, the wall fixtures and skirting, the cameras,
the lights and the other props, or it is dropped. A prop may hide at most half
of any crack token, seen face-on.

Factories are resolved lazily and individually. ``infinigen.assets.objects``
cannot be imported as a package here - it pulls a circular chain through
table_decorations - so each prop names its own submodule and a failure to import
one is skipped rather than fatal.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import bpy
import numpy as np

PROP_COLLECTION = "CWG_PROPS"
_OBJECTS = "cwg_infinigen.assets.objects."


@dataclass(frozen=True)
class Prop:
    module: str
    factory: str
    weight: float = 1.0
    # Real-world (width, depth front to back, height) ranges in metres; a model
    # outside them is scaled in, uniformly when one factor fits all three.
    size_m: tuple = (None, None, None)
    dims: tuple | None = None  # factory `dimensions` (depth, width, height) ranges, when it takes them


_CATALOG = {p.factory: p for p in (
    # kitchen
    Prop("appliances.beverage_fridge", "BeverageFridgeFactory", 1.0,
         size_m=((0.55, 0.75), (0.60, 0.75), (1.5, 1.9))),
    Prop("tables.dining_table", "TableDiningFactory", 1.0, size_m=((0.8, 1.8), (0.7, 1.0), (0.72, 0.78))),
    # bathroom
    Prop("bathroom.toilet", "ToiletFactory", 2.0, size_m=((0.35, 0.45), (0.62, 0.76), (0.72, 0.85))),
    Prop("bathroom.bathtub", "BathtubFactory", 1.5, size_m=((1.5, 1.8), (0.70, 0.85), (0.50, 0.62))),
    Prop("bathroom.vanity", "VanityCabinetFactory", 2.0, size_m=((0.5, 1.2), (0.40, 0.55), (0.80, 0.90))),
    Prop("bathroom.bathroom_sink", "StandingSinkFactory", 1.0, size_m=((0.5, 0.8), (0.42, 0.55), (0.95, 1.1))),
    # living room and bedroom
    Prop("seating.sofa", "SofaFactory", 2.0, size_m=((1.4, 2.4), (0.80, 1.00), (0.75, 0.95))),
    Prop("seating.sofa", "ArmChairFactory", 1.0, size_m=((0.70, 0.95), (0.75, 0.95), (0.75, 1.0))),
    Prop("seating.bed", "BedFactory", 1.5, size_m=((0.9, 1.95), (1.95, 2.2), (0.8, 1.3))),
    Prop("shelves.cell_shelf", "TVStandFactory", 1.5, size_m=((1.2, 2.0), (0.30, 0.45), (0.40, 0.60))),
    Prop("shelves.cell_shelf", "CellShelfFactory", 1.0, size_m=((0.4, 2.0), (0.28, 0.40), (0.4, 1.9))),
    Prop("shelves.large_shelf", "LargeShelfFactory", 1.0, size_m=((0.4, 1.8), (0.25, 0.40), (1.0, 2.0))),
    Prop("shelves.simple_bookcase", "SimpleBookcaseFactory", 1.0,
         size_m=((0.4, 0.9), (0.25, 0.40), (0.7, 1.2))),
    Prop("shelves.single_cabinet", "SingleCabinetFactory", 1.5, size_m=((0.4, 1.0), (0.30, 0.60), (0.9, 2.0))),
    Prop("shelves.simple_desk", "SimpleDeskFactory", 1.0, size_m=((0.9, 1.6), (0.50, 0.75), (0.72, 0.78))),
    # a bedside table: the factory draws a 0.5-0.6 m cube
    Prop("shelves.simple_desk", "SidetableDeskFactory", 1.0, size_m=((0.40, 0.52), (0.35, 0.45), (0.50, 0.62))),
    Prop("tables.dining_table", "CoffeeTableFactory", 1.0, size_m=((0.6, 0.9), (1.0, 1.4), (0.38, 0.48))),
    Prop("tables.dining_table", "SideTableFactory", 1.0, size_m=((0.4, 0.6), (0.4, 0.6), (0.45, 0.60))),
    Prop("seating.chairs.chair", "ChairFactory", 1.0, size_m=((0.42, 0.55), (0.45, 0.60), (0.80, 1.0))),
    Prop("seating.chairs.office_chair", "OfficeChairFactory", 0.5,
         size_m=((0.50, 0.70), (0.50, 0.70), (0.90, 1.25))),
    Prop("lamp", "FloorLampFactory", 1.0, size_m=(None, None, (1.50, 1.80))),
    Prop("tableware.plant_container", "LargePlantContainerFactory", 0.7,
         size_m=((0.3, 1.0), (0.3, 1.0), (0.5, 1.6))),
    Prop("tableware.food_box", "FoodBoxFactory", 0.7),
    Prop("tableware.bottle", "BottleFactory", 0.5),
)}

ROOMS = {
    "kitchen": (0.20, ("BeverageFridgeFactory", "TableDiningFactory", "ChairFactory")),
    "bathroom": (0.20, ("ToiletFactory", "BathtubFactory", "VanityCabinetFactory", "StandingSinkFactory")),
    "living": (0.25, ("SofaFactory", "ArmChairFactory", "TVStandFactory", "CellShelfFactory", "LargeShelfFactory",
                      "SimpleBookcaseFactory", "CoffeeTableFactory", "SideTableFactory", "FloorLampFactory",
                      "LargePlantContainerFactory")),
    "bedroom": (0.20, ("BedFactory", "SingleCabinetFactory", "SimpleDeskFactory", "SidetableDeskFactory",
                       "ChairFactory", "OfficeChairFactory", "FloorLampFactory")),
    "utility": (0.15, ("SingleCabinetFactory", "SimpleBookcaseFactory", "LargeShelfFactory", "CellShelfFactory",
                       "FoodBoxFactory", "BottleFactory")),
}

# Upholstery drawn too vivid for most homes: saturation is damped (not removed).
UPHOLSTERED = frozenset({"SofaFactory", "ArmChairFactory", "BedFactory"})
UPHOLSTERY_SATURATION = (0.55, 0.75)
# v1 material name keywords (lower case) -> Infinigen 2 prop role. First match
# wins; a name that matches none (labels, brand, plant, soil) keeps v1.
PROP_ROLES = (
    (("glass",), "glass"),
    (("lampshade",), "fabric"),
    (("fabric", "knit", "leather", "art_composite"), "upholstery"),
    (("chrome", "steel", "handle", "galvanized"), "steel"),
    (("metal",), "decorative"),
    (("plastic", "gasket", "foot"), "plastic"),
    (("marble", "ceramic", "vanitytop"), "table_top"),
    (("shelves", "vanitybody"), "cabinet"),
    (("wood",), "furniture"),
)
# Props whose factory assigns materials in finalize_assets, which spawning
# does not run: their bare meshes get this role.
BARE_ROLES = {"ToiletFactory": "ceramic"}
# One of each per room: a second toilet or fridge in front of the same wall looks wrong.
ONE_PER_ROOM = frozenset({"ToiletFactory", "BathtubFactory", "VanityCabinetFactory", "StandingSinkFactory",
                          "BeverageFridgeFactory", "BedFactory",
                          "SofaFactory", "TVStandFactory"})
WALL_GAP_M = 0.02  # nearest a prop stands to the wall, beyond its displaced relief
# Furthest a prop's back stands from the wall: the cameras stand about 0.4-2 m out, so a
# prop further back is behind them or fills the view.
MAX_STANDOFF_M = 0.9
MAX_COVER = 0.5  # largest share of a crack token's face a prop may hide
U_TRIES = 40


@dataclass(frozen=True)
class PropPlacement:
    index: int
    factory: str
    factory_seed: int
    location: tuple[float, float, float]
    yaw_deg: float
    object_name: str

    def to_dict(self) -> dict:
        return asdict(self)


def _load(module_name: str, class_name: str):
    import importlib

    try:
        import bpy  # noqa: F401
        import bmesh  # noqa: F401

        # wall_decorations and table_decorations import each other (wall tap -> sink ->
        # kitchen mixer -> wall_decorations.primitives). Entered from the table side first
        # (a sofa does), the cycle fails and the prop was silently skipped; entered from the
        # wall side it resolves.
        importlib.import_module("cwg_infinigen.assets.objects.wall_decorations")
        return getattr(importlib.import_module(module_name), class_name)
    except Exception:
        return None


def _world_bounds(obj):
    """World-space (min, max) of an object and its children after modifiers."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for part in (obj, *obj.children_recursive):
        if part.type not in ("MESH", "CURVE"):
            continue
        evaluated = part.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        if len(mesh.vertices):
            co = np.empty(len(mesh.vertices) * 3)
            mesh.vertices.foreach_get("co", co)
            m = np.array(evaluated.matrix_world)
            world = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
            lo, hi = np.minimum(lo, world.min(0)), np.maximum(hi, world.max(0))
        evaluated.to_mesh_clear()
    return None if not np.isfinite(lo).all() else (lo, hi)


def _overlap(a, b, gap: float) -> bool:
    return all(a[0][i] < b[1][i] + gap and b[0][i] < a[1][i] + gap for i in range(3))


def _delete(obj) -> None:
    for part in (*obj.children_recursive, obj):
        bpy.data.objects.remove(part, do_unlink=True)


def _collection(name: str):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _cover(box, defect) -> float:
    """Share of a crack token's face-on square that a prop's x-z extent hides."""
    half = defect.size_m * 0.5
    dx = min(box[1][0], defect.u_m + half) - max(box[0][0], defect.u_m - half)
    dz = min(box[1][2], defect.v_m + half) - max(box[0][2], defect.v_m - half)
    return max(0.0, dx) * max(0.0, dz) / max(1e-9, (2 * half) ** 2)


class _Room:
    """The free space in front of the wall and everything already in it."""

    def __init__(self, spec, floor_depth_m, ceiling_m, side_gap_m, obstacles, defects, relief_m=0.0):
        self.relief = relief_m  # displaced wall relief standing out of the plane
        self.half_w = spec.width_m * 0.5 - side_gap_m
        self.depth = math.inf if floor_depth_m is None else floor_depth_m - side_gap_m
        self.ceiling = (spec.height_m if ceiling_m is None else ceiling_m) - side_gap_m
        self.gap = side_gap_m
        self.obstacles = list(obstacles)
        self.defects = list(defects)
        self.placed = []  # (lo, hi) of every prop placed so far

    def blockers(self, box):
        return [b for b in (*self.obstacles, *self.placed) if _overlap(box, b, self.gap)]

    def fits(self, box) -> bool:
        lo, hi = box
        return (lo[0] >= -self.half_w and hi[0] <= self.half_w and lo[1] >= -self.depth and lo[2] >= -1e-6
                and hi[2] <= self.ceiling
                and all(_cover(box, d) <= MAX_COVER for d in self.defects)
                and not self.blockers(box))


def _place(obj, rng, room) -> bool:
    """Turn a spawned prop and move it to a clear spot on the floor by its measured bounds; False if none."""
    near = WALL_GAP_M + room.relief
    for _ in range(U_TRIES):
        # a new turn each try: a bed or sofa at an angle often fits where it would not square on
        obj.rotation_euler = (0.0, 0.0, rng.uniform(0.0, 2.0 * math.pi))
        bpy.context.view_layer.update()
        bounds = _world_bounds(obj)
        if bounds is None:
            return False
        lo0, hi0 = bounds
        size = hi0 - lo0
        if size[0] > 2 * room.half_w:
            continue
        u = rng.uniform(-room.half_w + size[0] / 2, room.half_w - size[0] / 2)
        y = -rng.uniform(near, max(near, min(MAX_STANDOFF_M, room.depth - size[1])))
        # back face at y, centred on u, bottom on the floor
        shift = np.array([u - (lo0[0] + hi0[0]) / 2, y - hi0[1], -lo0[2]])
        lo, hi = lo0 + shift, hi0 + shift
        # Step out from the wall past whatever is in the way (skirting, a fixture,
        # another prop); a spot that would need more than MAX_STANDOFF_M is redrawn.
        for _ in range(len(room.obstacles) + len(room.placed) + 1):
            hits = room.blockers((lo, hi))
            if not hits:
                break
            step = min(b[0][1] for b in hits) - room.gap - hi[1]
            if step >= 0 or hi[1] + step < -MAX_STANDOFF_M:
                break
            lo, hi, shift = lo + [0, step, 0], hi + [0, step, 0], shift + [0, step, 0]
        if room.fits((lo, hi)):
            obj.location = tuple(float(a + d) for a, d in zip(obj.location, shift))
            bpy.context.view_layer.update()
            room.placed.append((lo, hi))
            return True
    return False


def _fit_size(obj, prop):
    """Scale a model (still facing +X: depth on x, width on y) into its real-world size ranges."""
    lo, hi = _world_bounds(obj)
    dims = (hi[1] - lo[1], hi[0] - lo[0], hi[2] - lo[2])  # width, depth, height
    ranges = [(i, r) for i, r in enumerate(prop.size_m) if r is not None]
    if not ranges:
        return
    s_lo = max(r[0] / max(dims[i], 1e-6) for i, r in ranges)
    s_hi = min(r[1] / max(dims[i], 1e-6) for i, r in ranges)
    if s_lo <= s_hi:
        scale = [min(max(1.0, s_lo), s_hi)] * 3
    else:  # no single factor fits: bring each axis into its own range
        scale = [1.0, 1.0, 1.0]
        for i, (a, b) in ranges:
            scale[i] = min(max(1.0, a / dims[i]), b / dims[i])
    width, depth, height = scale
    obj.scale = (obj.scale[0] * depth, obj.scale[1] * width, obj.scale[2] * height)
    bpy.context.view_layer.update()


def _prop_role(material) -> str | None:
    name = material.name.lower()
    return next((role for keys, role in PROP_ROLES if any(k in name for k in keys)), None)


def _box_uv(part) -> None:
    """Metre UVs by box projection: each face takes the two axes across its normal.

    Most v1 props have no UVs, and the rest are not in metres; Infinigen 2
    shaders read UVs at metre scale.
    """
    mesh = part.data
    if "CWG_BoxUV" in mesh.uv_layers or not mesh.polygons:
        return
    n_poly, n_loop = len(mesh.polygons), len(mesh.loops)
    normals = np.empty(n_poly * 3)
    mesh.polygons.foreach_get("normal", normals)
    totals = np.empty(n_poly, dtype=np.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    verts = np.empty(n_loop, dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", verts)
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)[verts] * np.array(part.matrix_world.to_scale())
    axis = np.repeat(np.abs(normals.reshape(-1, 3)).argmax(1), totals)
    u = np.where(axis == 0, co[:, 1], co[:, 0])
    v = np.where(axis == 2, co[:, 1], co[:, 2])
    layer = mesh.uv_layers.new(name="CWG_BoxUV")
    if layer is None:  # a mesh already at Blender's UV-layer limit keeps its own
        return
    layer.data.foreach_set("uv", np.stack([u, v], 1).ravel())
    mesh.uv_layers.active = layer
    layer.active_render = True


def _v2_materials(obj, factory_seed: int, bare_role: str | None = None) -> None:
    """Swap each part's v1 material for an Infinigen 2 one of the same role.

    One v2 material per v1 material, so parts that shared a finish still share
    one. A part with no material gets `bare_role`, when given. Seeds come from
    the factory seed, not the prop stream.
    """
    from cwg_infinigen.core.util.math import int_hash
    from syncrack3d.infinigen2_bridge import realize_prop_material

    swapped = {}
    for part in (obj, *obj.children_recursive):
        if part.type != "MESH":
            continue
        if bare_role is not None and not any(slot.material for slot in part.material_slots):
            if "bare" not in swapped:
                swapped["bare"] = realize_prop_material(bare_role, int_hash((int(factory_seed), "bare")) % (2**31))
            _box_uv(part)
            part.data.materials.clear()
            part.data.materials.append(swapped["bare"])
            continue
        for slot in part.material_slots:
            old = slot.material
            role = _prop_role(old) if old is not None else None
            if role is None:
                continue
            if old.name not in swapped:
                seed = int_hash((int(factory_seed), old.name)) % (2**31)
                swapped[old.name] = realize_prop_material(role, seed)
            _box_uv(part)
            slot.material = swapped[old.name]


def _damp_saturation(obj, rng):
    """Calm vivid upholstery: a Hue/Saturation step in front of each fabric's base colour."""
    factor = rng.uniform(*UPHOLSTERY_SATURATION)
    materials = {slot.material for part in (obj, *obj.children_recursive) if part.type == "MESH"
                 for slot in part.material_slots if slot.material is not None and slot.material.use_nodes}
    for mat in materials:
        if mat.get("cwg_prop_role") not in ("upholstery", "fabric"):
            continue  # wood frames and plastic feet keep their colour
        nt = mat.node_tree
        for bsdf in [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"]:
            base = bsdf.inputs["Base Color"]
            hsv = nt.nodes.new("ShaderNodeHueSaturation")
            hsv.inputs["Saturation"].default_value = factor
            if base.links:
                nt.links.new(base.links[0].from_socket, hsv.inputs["Color"])
            else:
                hsv.inputs["Color"].default_value = tuple(base.default_value)
            nt.links.new(hsv.outputs["Color"], base)


def _spawn(prop, rng, index, collection):
    module = prop.module if prop.module.startswith("syncrack3d.") else _OBJECTS + prop.module
    factory_cls = _load(module, prop.factory)
    if factory_cls is None:
        return None, 0
    factory_seed = rng.randrange(1, 2**31)
    kwargs = {}
    if prop.dims is not None:
        kwargs["dimensions"] = tuple(rng.uniform(*r) for r in prop.dims)
    try:
        obj = factory_cls(factory_seed, **kwargs).spawn_asset(i=index)
    except Exception:
        # One prop failing to build is not worth losing the scene over.
        return None, factory_seed
    obj.parent = None
    bpy.context.view_layer.update()
    _fit_size(obj, prop)
    _v2_materials(obj, factory_seed, BARE_ROLES.get(prop.factory))
    if prop.factory in UPHOLSTERED:
        _damp_saturation(obj, rng)
    obj.location = (0.0, -2.0, 0.0)
    obj.name = f"CWG_Prop_{index:02d}_{prop.factory}"
    obj["cwg_role"] = "prop"
    for part in (obj, *obj.children_recursive):
        for old in tuple(part.users_collection):
            old.objects.unlink(part)
        collection.objects.link(part)
    bpy.context.view_layer.update()
    return obj, factory_seed


def _record(obj, prop, index, factory_seed):
    return PropPlacement(index=index, factory=prop.factory, factory_seed=factory_seed,
                         location=tuple(round(float(v), 5) for v in obj.location),
                         yaw_deg=round(math.degrees(obj.rotation_euler.z), 2),
                         object_name=obj.name)


def sample_room(rng: random.Random, allowed=None) -> str:
    names = [n for n in ROOMS if allowed is None or n in allowed]
    return rng.choices(names, weights=[ROOMS[n][0] for n in names], k=1)[0]


def sample_props(
    rng: random.Random,
    spec,
    placements,
    *,
    room: str | None = None,
    rooms=None,
    count_range: tuple[int, int] = (0, 3),
    floor_depth_m: float | None = None,
    ceiling_m: float | None = None,
    side_gap_m: float = 0.02,
    obstacles=(),
    relief_m: float = 0.0,
) -> tuple[str, list[PropPlacement]]:
    """Furnish the floor in front of the test wall as one room.

    Returns (room, placements). The room is `room`, or drawn from `rooms` (all
    when None). `obstacles` are world boxes (lo, hi) placed
    before the props (fixtures, skirting, cameras, lights). `placements` are the
    crack tokens: partial occlusion is wanted, but a prop that hid most of a crack
    face-on would leave a close-up of furniture, so it may cover at most
    `MAX_COVER` of each token. Total occlusion is left to the camera checks and
    the exporter, which holds out frames whose crack coverage drops to zero.
    """
    room = room or sample_room(rng, rooms)
    count = rng.randint(*count_range)
    if count <= 0:
        return room, []

    collection = _collection(PROP_COLLECTION)
    space = _Room(spec, floor_depth_m, ceiling_m, side_gap_m, obstacles, placements, relief_m)
    choices = [_CATALOG[name] for name in ROOMS[room][1]]
    results: list[PropPlacement] = []
    index = 0
    for _ in range(count):
        if not choices:
            break
        prop = rng.choices(choices, weights=[p.weight for p in choices], k=1)[0]
        if prop.factory in ONE_PER_ROOM:
            choices = [p for p in choices if p is not prop]
        obj, factory_seed = _spawn(prop, rng, index, collection)
        if obj is None:
            continue
        if not _place(obj, rng, space):
            _delete(obj)
            continue
        results.append(_record(obj, prop, index, factory_seed))
        index += 1
    return room, results
