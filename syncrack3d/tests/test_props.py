"""Props stand on the floor, clear of the wall, the side returns, each other,
wall fixtures and cameras - whatever their real size and origin."""
import random
from types import SimpleNamespace

import pytest

bpy = pytest.importorskip("bpy")

from syncrack3d import occlusion

W, DEPTH = 3.0, 2.5


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


def _catalog(monkeypatch, entries):
    """entries: (class name, box size, origin offset)."""
    factories = {}
    for name, size, offset in entries:
        factories[name] = type(name, (), {
            "__init__": lambda self, seed: None,
            "spawn_asset": lambda self, i, size=size, offset=offset, name=name: _box(name, size, offset),
        })
    monkeypatch.setattr(occlusion, "_PROP_CATALOG", tuple(("m", n, 0.3, True) for n, _, _ in entries))
    monkeypatch.setattr(occlusion, "_load", lambda module, name: factories[name])


@pytest.fixture(autouse=True)
def empty_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _bounds(prop):
    return occlusion._world_bounds(bpy.data.objects[prop.object_name])


@pytest.mark.parametrize("seed", range(6))
def test_large_off_centre_props_stay_in_the_room(monkeypatch, seed):
    # a 1.4 m shelf with its origin at a back corner and mid-height
    _catalog(monkeypatch, [("Shelf", (1.4, 0.5, 1.8), (0.6, 0.2, -0.9))])
    props = occlusion.sample_props(random.Random(seed), SimpleNamespace(width_m=W), [],
                                   count_range=(1, 3), floor_depth_m=DEPTH)
    assert props
    for p in props:
        lo, hi = _bounds(p)
        assert hi[1] <= -0.15 + 1e-6          # standoff is the true gap to the wall
        assert abs(lo[2]) < 1e-6              # resting on the floor
        assert -W / 2 <= lo[0] and hi[0] <= W / 2 and lo[1] >= -DEPTH
    boxes = [_bounds(p) for p in props]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not occlusion._overlap_xy(a, b, 0.0)


def test_room_sized_rug_is_shrunk_to_the_floor(monkeypatch):
    _catalog(monkeypatch, [("RugFactory", (5.0, 5.0, 0.01), (0.0, 0.0, 0.0))])
    props = occlusion.sample_props(random.Random(0), SimpleNamespace(width_m=W), [],
                                   count_range=(1, 1), floor_depth_m=DEPTH)
    lo, hi = _bounds(props[0])
    assert hi[1] <= -0.15 + 1e-6 and lo[1] >= -DEPTH and -W / 2 <= lo[0] and hi[0] <= W / 2


def test_settle_pushes_props_off_fixtures_and_drops_props_around_cameras(monkeypatch):
    _catalog(monkeypatch, [("Shelf", (0.6, 0.4, 1.0), (0.0, 0.0, 0.0))])
    props = occlusion.sample_props(random.Random(3), SimpleNamespace(width_m=W), [],
                                   count_range=(2, 2), floor_depth_m=DEPTH)
    assert len(props) == 2
    first, second = (_bounds(p) for p in props)
    # a window sill reaching past the first prop's front face
    sill = _box("sill", (first[1][0] - first[0][0], 0.5, 0.05))
    sill.location = ((first[0][0] + first[1][0]) / 2, -0.25, 0.5)
    sill["cwg_role"] = "structure"
    camera = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.scene.collection.objects.link(camera)
    camera.location = tuple((second[0] + second[1]) / 2)
    bpy.context.view_layer.update()

    kept = occlusion.settle_props(props, floor_depth_m=DEPTH)
    assert [p.object_name for p in kept] == [props[0].object_name]
    assert props[1].object_name not in bpy.data.objects
    assert _bounds(kept[0])[1][1] <= -0.5
    # the record follows the move
    assert kept[0].location[1] == pytest.approx(bpy.data.objects[kept[0].object_name].location.y, abs=1e-5)
