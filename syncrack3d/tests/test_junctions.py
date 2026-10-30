"""Junction cracks: off by default, and anchors sit on structure edges with sensible directions."""
import random

from syncrack3d.junctions import _anchors, snap_to_junctions
from syncrack3d.structures import StructurePlan


def _plan(kind, u=0.0, v=1.4, w=1.0, h=1.2):
    return StructurePlan(index=0, kind=kind, u_m=u, v_m=v, width_m=w, height_m=h, seed=1)


def test_off_or_nothing_to_snap_changes_nothing():
    placements = [object()]
    assert snap_to_junctions(placements, [_plan("window")], None, 7, 0.0) == (placements, [])
    assert snap_to_junctions(placements, [], None, 7, 1.0) == (placements, [])
    assert snap_to_junctions([], [_plan("window")], None, 7, 1.0) == ([], [])


def test_window_corners_point_outwards_and_sill_crack_is_horizontal():
    for seed in range(50):
        anchors = {}
        for name, (u, v), angle, mode in _anchors(_plan("window"), random.Random(seed)):
            anchors.setdefault(name, []).append((u, v, angle, mode))
        tops, bots = anchors["window_top_corner"], anchors["window_bottom_corner"]
        right_top, left_top = sorted(tops, key=lambda a: -a[0])
        assert 30 <= right_top[2] <= 60 and 120 <= left_top[2] <= 150  # up and away from the opening
        right_bot, left_bot = sorted(bots, key=lambda a: -a[0])
        assert -60 <= right_bot[2] <= -30 and 210 <= left_bot[2] <= 240  # down and away
        assert all(a[3] == "corner" for a in tops + bots)
        (u, v, angle, mode), = anchors["sill_underside"]
        assert mode == "edge" and abs(angle) <= 4 and v < 1.4 - 0.6  # below the sill
        assert all(t[1] > 1.4 + 0.6 for t in tops)  # above the lintel


def test_door_jambs_are_vertical_and_trims_horizontal():
    for seed in range(20):
        r = random.Random(seed)
        jambs = [a for a in _anchors(_plan("door", v=1.05, w=0.9, h=2.1), r) if a[0] == "door_jamb"]
        assert len(jambs) == 2 and all(abs(a[2] - 90) <= 3 for a in jambs)
        for kind in ("ledge", "ceiling_trim", "skirting"):
            assert all(abs(a[2]) <= 4 and a[3] == "edge" for a in _anchors(_plan(kind, h=0.1), r))
    assert _anchors(_plan("cable"), random.Random(0)) == []
