"""--layout-order fixtures-first: cracks are placed around fixtures planned first."""
import math
import random

import pytest

pytest.importorskip("bpy")

from syncrack3d.defects import DefectPlan, _sample_positions  # noqa: E402
from syncrack3d.structures import StructurePlan, crack_regions  # noqa: E402

PLAN = DefectPlan(counts={"crack": 3}, size_range_m=(0.2, 0.45), min_separation_m=0.30)
W, H = 4.0, 2.6


def _original(rng, plan, width_m, height_m, count):
    """The v1.x sampler, verbatim, to pin the no-fixture behaviour."""
    half_u = max(width_m * 0.5 - 0.22, 0.05)
    lo, hi = 0.22, max(height_m - 0.22, 0.22 + 0.05)
    placed = []
    for _ in range(count):
        for _attempt in range(80):
            u, v = rng.uniform(-half_u, half_u), rng.uniform(lo, hi)
            if all(math.hypot(u - pu, v - pv) >= plan.min_separation_m for pu, pv in placed):
                placed.append((u, v))
                break
        else:
            break
    return placed


def test_without_fixtures_the_draws_are_the_original_ones():
    for seed in range(50):
        a = _sample_positions(random.Random(seed), PLAN, W, H, 3)
        b = _original(random.Random(seed), PLAN, W, H, 3)
        assert a == b


def test_keepout_is_never_entered():
    window = (0.0, 1.4, 1.2, 1.0)
    grow = 0.3 * PLAN.size_range_m[1] + 0.03
    for seed in range(200):
        for u, v in _sample_positions(random.Random(seed), PLAN, W, H, 3, keepout=[window]):
            assert abs(u - window[0]) >= window[2] / 2 + grow or abs(v - window[1]) >= window[3] / 2 + grow


def test_near_service_tokens_sit_beside_the_run():
    pipe = (1.0, 1.3, 0.05, 2.2)  # vertical run
    beside = 0
    for seed in range(200):
        for u, v in _sample_positions(random.Random(seed), PLAN, W, H, 1, near=[pipe], near_fraction=1.0):
            gap = abs(u - pipe[0]) - pipe[2] / 2
            beside += 0.03 - 1e-9 <= gap <= 0.15 + 1e-9
    assert beside == 200


def test_crack_regions_grow_openings_to_their_built_extent():
    plans = [StructurePlan(0, "window", 0.0, 1.5, 1.0, 0.9, 1), StructurePlan(1, "socket", 1.0, 0.4, 0.08, 0.08, 2),
             StructurePlan(2, "conduit", -1.0, 1.2, 0.04, 2.0, 3), StructurePlan(3, "panel_seam", 0.0, 1.0, 3.0, 0.006, 4)]
    keepout, near = crack_regions(plans)
    assert keepout[0] == (0.0, 1.52, 1.2, 1.22) and keepout[1][2] == pytest.approx(0.12)
    assert near == [(-1.0, 1.2, 0.04, 2.0)]
