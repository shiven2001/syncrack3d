"""Props stand on the floor, clear of the wall, the side returns, each other,
wall fixtures and cameras - whatever their real size and origin."""
import random
from types import SimpleNamespace

import numpy as np

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
    monkeypatch.setattr(occlusion, "_PROP_CATALOG", tuple(("m", n) for n, _, _ in entries))
    monkeypatch.setattr(occlusion, "_load", lambda module, name: factories[name])


@pytest.fixture(autouse=True)
def empty_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _bounds(prop):
    return occlusion._world_bounds(bpy.data.objects[prop.object_name])


def _loose(monkeypatch, entries, seed, count, **kw):
    _catalog(monkeypatch, entries)
    return occlusion.sample_props(random.Random(seed), SimpleNamespace(width_m=W, height_m=2.6), kw.pop("defects", []),
                                  count_range=(count, count), floor_depth_m=DEPTH, **kw)


@pytest.mark.parametrize("seed", range(6))
def test_loose_props_stand_anywhere_in_the_band_and_turn_any_way(monkeypatch, seed):
    props = _loose(monkeypatch, [("Shelf", (1.4, 0.5, 1.8), (0.6, 0.2, -0.9))], seed, 3, relief_m=0.03)
    assert props
    boxes = [_bounds(p) for p in props]
    for lo, hi in boxes:
        assert -occlusion.MAX_STANDOFF_M - 1e-6 <= hi[1] <= -(occlusion.WALL_GAP_M + 0.03) + 1e-6
        assert abs(lo[2]) < 1e-6
        assert -W / 2 <= lo[0] and hi[0] <= W / 2 and lo[1] >= -DEPTH
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not occlusion._overlap(a, b, 0.0)


def test_loose_props_turn_and_reach_the_wall(monkeypatch):
    yaws, gaps = set(), []
    for seed in range(20):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        for p in _loose(monkeypatch, [("Box", (0.5, 0.4, 0.8), (0.0, 0.0, 0.0))], seed, 1):
            yaws.add(round(p.yaw_deg / 45))
            gaps.append(-_bounds(p)[1][1])
    assert len(yaws) >= 4 and min(gaps) < 0.15  # not only square on, and closer than the old 0.15 m standoff


def test_loose_props_clear_obstacles_or_are_dropped(monkeypatch):
    camera = (np.array([-W / 2, -1.0, 0.0]), np.array([W / 2, -0.03, 2.6]))
    assert _loose(monkeypatch, [("Box", (0.5, 0.4, 0.8), (0.0, 0.0, 0.0))], 0, 2, obstacles=[camera]) == []
    assert not bpy.data.objects.get("CWG_Prop_00_Box")
    lamp = (np.array([-0.3, -0.5, 0.0]), np.array([0.3, -0.2, 2.0]))
    for p in _loose(monkeypatch, [("Box", (0.5, 0.4, 0.8), (0.0, 0.0, 0.0))], 1, 3, obstacles=[lamp]):
        assert not occlusion._overlap(_bounds(p), lamp, 0.0)


def test_loose_props_hide_at_most_half_of_a_crack(monkeypatch):
    crack = SimpleNamespace(u_m=0.0, v_m=0.4, size_m=0.4)
    props = []
    for seed in range(10):
        props += _loose(monkeypatch, [("Box", (1.0, 0.4, 1.2), (0.0, 0.0, 0.0))], seed, 2, defects=[crack])
    assert props
    for p in props:
        assert occlusion._cover(_bounds(p), crack) <= occlusion.MAX_COVER + 1e-9

