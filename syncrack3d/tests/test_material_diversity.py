"""Wall materials: uniform draw over 15 categories, subsets, clean-wall policy."""

from collections import Counter
from types import SimpleNamespace

import pytest

from syncrack3d import material_diversity as md


def test_category_draw_is_seed_deterministic_and_uniform():
    assert [md.sample_category(s) for s in range(50)] == [md.sample_category(s) for s in range(50)]
    counts = Counter(md.sample_category(s) for s in range(6000))
    assert set(counts) == set(md.CATEGORIES) and len(md.CATEGORIES) == 15
    assert "wood" not in md.CATEGORIES and "irregular_stone" not in md.CATEGORIES
    for name in md.CATEGORIES:
        assert abs(counts[name] / 6000 - 1 / 15) < 0.02


def test_forced_subset_and_unknown_categories():
    assert md.sample_category(7, "tile") == "tile"
    assert {md.sample_category(s, allowed=["brick", "marble"]) for s in range(200)} == {"brick", "marble"}
    for bad in (dict(forced="carpet"), dict(allowed=["brick", "wood"])):
        with pytest.raises(ValueError):
            md.sample_category(7, **bad)


def test_clean_walls():
    # Default: every wall cracked.
    assert not any(md.is_clean_wall(s, name) for s in range(200) for name in md.CATEGORIES)
    clean_paint = sum(md.is_clean_wall(s, "paint_colour", scale=1.0) for s in range(3000)) / 3000
    assert abs(clean_paint - md.CLEAN_FRACTION["paint_colour"]) < 0.03
    clean_block = sum(md.is_clean_wall(s, "stone_block", scale=1.0) for s in range(3000)) / 3000
    assert abs(clean_block - 0.6) < 0.04
    # The uniform policy compares against the same per-wall draw.
    for s in range(300):
        assert md.is_clean_wall(s, "brick", uniform=0.6) == md.is_clean_wall(s, "tile", scale=1.0)
    assert all(md.is_clean_wall(s, "brick", uniform=1.0) for s in range(50))


def test_only_paint_categories_use_plaster_and_are_coated():
    for name, (finish, choices) in md.CATEGORIES.items():
        uses_plaster = any(cls in md.PAINTED for _, cls, _ in choices)
        assert uses_plaster == (finish == "paint"), name
    assert set(md.PAINTED_CATEGORIES) == {n for n, (f, _) in md.CATEGORIES.items() if f == "paint"}
    assert md.PAINT_PARAMS["paint_neutral"]["colored_prob"] == 0.0
    assert md.PAINT_PARAMS["paint_colour"]["hue_max"] - md.PAINT_PARAMS["paint_colour"]["hue_min"] == 1.0


def test_virtual_targets_stay_on_the_wall():
    spec = SimpleNamespace(width_m=4.0, height_m=2.6)
    targets = md.virtual_targets(3, spec, n=20)
    assert len(targets) == 20 and len({t.object_name for t in targets}) == 20
    for t in targets:
        assert -1.78 <= t.u_m <= 1.78 and 0.22 <= t.v_m <= 2.38
        assert 0.10 <= t.size_m <= 0.45
    assert [(t.u_m, t.v_m) for t in targets] == [(t.u_m, t.v_m) for t in md.virtual_targets(3, spec, n=20)]
