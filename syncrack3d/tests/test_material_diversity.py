"""CWGv5 diverse-material sampling: determinism, physics rules, clean walls."""

from collections import Counter
from types import SimpleNamespace

import pytest

from syncrack3d import material_diversity as md


def test_category_sampling_is_seed_deterministic_and_covers_all():
    assert [md.sample_category(s) for s in range(50)] == [md.sample_category(s) for s in range(50)]
    counts = Counter(md.sample_category(s) for s in range(4000))
    # Weight-0 categories are forced-only (--material-category); random draws skip them.
    assert set(counts) == {n for n, (w, _, _) in md.CATEGORIES.items() if w > 0}
    for name, (weight, _, _) in md.CATEGORIES.items():
        assert abs(counts[name] / 4000 - weight) < 0.03


def test_forced_category_and_unknown_category():
    assert md.sample_category(7, "tile") == "tile"
    with pytest.raises(ValueError):
        md.sample_category(7, "carpet")


def test_wood_is_always_clean_and_paint_mostly_has_defects():
    assert all(md.is_clean_wall(s, "wood") for s in range(200))
    assert all(md.is_clean_wall(s, "wood", scale=5.0) for s in range(200))
    # scale 0 is the explicit "no clean walls" switch, used to put defects on any texture
    assert not any(md.is_clean_wall(s, "wood", scale=0.0) for s in range(200))
    clean_paint = sum(md.is_clean_wall(s, "paint_colour") for s in range(2000)) / 2000
    assert abs(clean_paint - md.CLEAN_FRACTION["paint_colour"]) < 0.03
    assert not any(md.is_clean_wall(s, "tile", scale=0.0) for s in range(200))


def test_only_paint_categories_use_plaster_and_are_coated():
    for name, (_, finish, choices) in md.CATEGORIES.items():
        uses_plaster = any(cls in md.PAINTED for _, cls, _ in choices)
        assert uses_plaster == (finish == "paint"), name
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
