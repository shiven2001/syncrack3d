"""Wall materials: uniform draw over legacy categories, subsets, clean-wall policy."""

from collections import Counter
from types import SimpleNamespace

import pytest

from syncrack3d import material_diversity as md


def test_default_draw_follows_infinigen_wall_picker():
    assert [md.sample_category(s) for s in range(50)] == [md.sample_category(s) for s in range(50)]
    counts = Counter(md.sample_category(s) for s in range(20000))
    assert set(counts) == set(md.DEFAULT_CATEGORIES)
    assert set(md.DEFAULT_CATEGORIES) == md.INFINIGEN2_CATEGORY_NAMES
    assert abs(sum(md.WALL_WEIGHTS.values()) - 1.0) < 1e-9
    for name in md.DEFAULT_CATEGORIES:
        assert abs(counts[name] / 20000 - md.WALL_WEIGHTS[name]) < 0.01, name
    assert max(md.WALL_WEIGHTS, key=md.WALL_WEIGHTS.get) == "paint_wall"
    # Overlays only on the picker's layerable leaves; never on brick, tile or granite.
    assert set(md.OVERLAY_FRACTION) == {"paint_wall", "wall_wood_planks", "wall_concrete",
                                        "wall_stone", "wall_gravel_concrete"}
    assert "paint_neutral" in md.CATEGORIES and "paint_neutral" not in counts
    assert "wood" not in md.CATEGORIES and "irregular_stone" not in md.CATEGORIES


def test_wall_draw_has_no_crack_or_non_wall_shaders():
    # Crack and flake masks would be unlabelled cracks; floor, ground, fabric
    # and object shaders are not walls.
    for name in ("paint_flaked", "wall_flaked_tile", "ground_cracked", "sand", "soil", "dirt", "mud", "ice",
                 "tile_indoor_ground", "tile_outdoor_ground", "carpet", "fabric", "leather", "plastic",
                 "glass_colored", "metal_hammered", "terrazzo"):
        assert name not in md.CATEGORIES, name


def test_forced_subset_and_unknown_categories():
    assert md.sample_category(7, "tile") == "tile"
    assert {md.sample_category(s, allowed=["brick", "marble"]) for s in range(200)} == {"brick", "marble"}
    for bad in (dict(forced="lava"), dict(allowed=["brick", "wood"])):
        with pytest.raises(ValueError):
            md.sample_category(7, **bad)
    assert md.sample_category(0, "paint_wall") == "paint_wall"


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
        if name in md.INFINIGEN2_CATEGORY_NAMES:
            continue  # I2 paint finishes are coated without legacy Plaster gin params
        uses_plaster = any(cls in md.PAINTED for _, cls, _ in choices)
        assert uses_plaster == (finish == "paint"), name
    legacy_paint = {n for n, (f, _) in md.CATEGORIES.items()
                    if f == "paint" and n not in md.INFINIGEN2_CATEGORY_NAMES}
    assert legacy_paint <= set(md.PAINTED_CATEGORIES)
    assert md.INFINIGEN2_CATEGORY_NAMES & set(md.PAINTED_CATEGORIES)
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
