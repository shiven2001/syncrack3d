"""Props stay inside the room: off the wall, on the floor, under the ceiling, between
the side returns, clear of fixtures, cameras and each other, whatever their real size,
origin and turn, and never hiding most of a crack."""
import random
from types import SimpleNamespace

import pytest

bpy = pytest.importorskip("bpy")
np = pytest.importorskip("numpy")

from syncrack3d import occlusion

W, H, DEPTH = 3.0, 2.5, 2.5
SPEC = SimpleNamespace(width_m=W, height_m=H)


def _box(name, size, origin_offset=(0.0, 0.0, 0.0)):
    """A box whose origin sits `origin_offset` away from its centre."""
    sx, sy, sz = (s * 0.5 for s in size)
    ox, oy, oz = origin_offset
    verts = [(ox + x * sx, oy + y * sy, oz + z * sz) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _room(monkeypatch, entries):
    """entries: (class name, box size, origin offset); all in one test room."""
    factories, catalog = {}, {}
    for name, size, offset in entries:
        factories[name] = type(name, (), {
            "__init__": lambda self, seed: None,
            "spawn_asset": lambda self, i, size=size, offset=offset, name=name: _box(name, size, offset),
        })
        catalog[name] = occlusion.Prop("m", name)
    monkeypatch.setattr(occlusion, "_CATALOG", catalog)
    monkeypatch.setattr(occlusion, "ROOMS", {"test": (1.0, tuple(catalog))})
    monkeypatch.setattr(occlusion, "_load", lambda module, name: factories[name])


@pytest.fixture(autouse=True)
def empty_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _bounds(prop):
    return occlusion._world_bounds(bpy.data.objects[prop.object_name])


def _inside(lo, hi):
    return (-W / 2 <= lo[0] and hi[0] <= W / 2 and lo[1] >= -DEPTH and hi[1] <= 0.0
            and lo[2] >= -1e-6 and hi[2] <= H)


@pytest.mark.parametrize("seed", range(6))
def test_large_off_centre_props_stay_in_the_room(monkeypatch, seed):
    # a 1.4 m cabinet with its origin at a back corner and mid-height, a chair, a bottle
    _room(monkeypatch, [("Cabinet", (0.5, 1.4, 1.8), (0.2, 0.6, -0.9)),
                        ("Chair", (0.5, 0.5, 0.9), (0.0, 0.0, 0.45)),
                        ("Bottle", (0.1, 0.1, 0.3), (0.0, 0.0, 0.15))])
    _, props = occlusion.sample_props(random.Random(seed), SPEC, [], count_range=(3, 3), floor_depth_m=DEPTH)
    assert props
    for p in props:
        lo, hi = _bounds(p)
        assert _inside(lo, hi)
        assert abs(lo[2]) < 1e-6  # on the floor
        assert -occlusion.MAX_STANDOFF_M - 1e-6 <= hi[1] <= -occlusion.WALL_GAP_M + 1e-6  # off the wall
    boxes = [_bounds(p) for p in props]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not occlusion._overlap(a, b, 0.0)


def test_props_stand_anywhere_and_turn_any_way(monkeypatch):
    _room(monkeypatch, [("Chair", (0.5, 0.5, 0.9), (0.0, 0.0, 0.45))])
    gaps, yaws = [], []
    for seed in range(24):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        _, props = occlusion.sample_props(random.Random(seed), SPEC, [], count_range=(1, 1), floor_depth_m=DEPTH)
        gaps.append(-_bounds(props[0])[1][1])
        yaws.append(props[0].yaw_deg % 360)
    assert min(gaps) < 0.3 and max(gaps) > 0.6 and max(gaps) <= occlusion.MAX_STANDOFF_M + 1e-6
    assert max(yaws) - min(yaws) > 180


@pytest.mark.parametrize("seed", range(4))
def test_props_step_past_cameras_and_fixtures(monkeypatch, seed):
    _room(monkeypatch, [("Shelf", (0.6, 0.4, 1.0), (0.0, 0.0, 0.0))])
    # a fixture or camera box across the whole wall front, 0.1-0.6 m out
    obstacle = (np.array([-W, -0.6, 0.0]), np.array([W, -0.1, 2.0]))
    _, props = occlusion.sample_props(random.Random(seed), SPEC, [], count_range=(1, 2), floor_depth_m=DEPTH,
                                      obstacles=[obstacle])
    assert props
    for p in props:
        lo, hi = _bounds(p)
        assert not occlusion._overlap((lo, hi), obstacle, 0.0)
        assert _inside(lo, hi)
        assert hi[1] >= -occlusion.MAX_STANDOFF_M - 1e-6  # stepped past it, not behind the cameras


def test_props_that_cannot_clear_are_dropped(monkeypatch):
    _room(monkeypatch, [("Shelf", (0.6, 0.4, 1.0), (0.0, 0.0, 0.0))])
    # something filling the whole floor
    block = (np.array([-W, -DEPTH, 0.0]), np.array([W, 0.0, 0.05]))
    _, props = occlusion.sample_props(random.Random(0), SPEC, [], count_range=(1, 1), floor_depth_m=DEPTH,
                                      obstacles=[block])
    assert props == []
    assert not [o for o in bpy.data.objects if o.name.startswith("CWG_Prop")]


@pytest.mark.parametrize("seed", range(4))
def test_props_hide_at_most_half_of_a_crack(monkeypatch, seed):
    _room(monkeypatch, [("Cabinet", (0.5, 1.2, 1.2), (0.0, 0.0, 0.6))])
    cracks = [SimpleNamespace(u_m=u, v_m=0.5, size_m=0.4) for u in (-1.0, 0.0, 1.0)]
    _, props = occlusion.sample_props(random.Random(seed), SPEC, cracks, count_range=(1, 1), floor_depth_m=DEPTH)
    for p in props:
        box = _bounds(p)
        assert all(occlusion._cover(box, c) <= occlusion.MAX_COVER + 1e-9 for c in cracks)


def test_props_stand_clear_of_a_rough_wall_relief(monkeypatch):
    _room(monkeypatch, [("Cabinet", (0.5, 1.0, 1.0), (0.0, 0.0, 0.5))])
    relief = 0.015  # cobble stone stands this far out of the wall plane
    for seed in range(8):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        _, props = occlusion.sample_props(random.Random(seed), SPEC, [], count_range=(1, 1), floor_depth_m=DEPTH,
                                          relief_m=relief)
        assert _bounds(props[0])[1][1] <= -occlusion.WALL_GAP_M - relief + 1e-6


def test_one_toilet_per_room(monkeypatch):
    _room(monkeypatch, [("ToiletFactory", (0.6, 0.4, 0.8), (0.0, 0.0, 0.4)),
                        ("Bottle", (0.1, 0.1, 0.3), (0.0, 0.0, 0.15))])
    for seed in range(8):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        _, props = occlusion.sample_props(random.Random(seed), SPEC, [], count_range=(3, 3), floor_depth_m=DEPTH)
        assert sum(p.factory == "ToiletFactory" for p in props) <= 1


def test_models_are_scaled_into_their_real_size(monkeypatch):
    _room(monkeypatch, [("Bed", (2.7, 2.6, 1.0), (0.0, 0.0, 0.5))])  # depth x, width y, height z
    catalog = dict(occlusion._CATALOG)
    catalog["Bed"] = occlusion.Prop("m", "Bed", 1.0, size_m=((0.9, 1.95), (1.95, 2.2), (0.8, 1.3)))
    monkeypatch.setattr(occlusion, "_CATALOG", catalog)
    _, props = occlusion.sample_props(random.Random(0), SimpleNamespace(width_m=4.0, height_m=H), [],
                                      count_range=(1, 1), floor_depth_m=4.0)
    sx, sy, sz = bpy.data.objects[props[0].object_name].scale  # measured before it is turned
    width, depth, height = 2.6 * sy, 2.7 * sx, 1.0 * sz
    assert 0.9 - 1e-6 <= width <= 1.95 + 1e-6 and 1.95 - 1e-6 <= depth <= 2.2 + 1e-6 and 0.8 - 1e-6 <= height <= 1.3 + 1e-6
