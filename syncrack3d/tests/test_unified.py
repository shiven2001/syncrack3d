"""v2 unified recipe: per-wall crack styles, uniform palette, one clean table."""
import collections
from argparse import Namespace

import pytest

from syncrack3d.material_diversity import (NEVER_CRACKED, PAINTED_CATEGORIES, UNIFIED_CATEGORIES,
                                           is_clean_wall, sample_category)
from syncrack3d.recipes import RECIPES, STYLES, UNIFIED_MIX, parse_style_mix, wall_style

ARGS = Namespace(crack_style_mix=UNIFIED_MIX, material_category=None, crack_profile="standard",
                 structures="off")


def test_style_draw_is_deterministic_and_follows_the_mix():
    assert [wall_style(s, ARGS)[0] for s in range(50)] == [wall_style(s, ARGS)[0] for s in range(50)]
    counts = collections.Counter(wall_style(s, ARGS)[0] for s in range(4000))
    mix = parse_style_mix(UNIFIED_MIX)
    assert mix == {"standard": .72, "wide": .17, "crazing": .11}  # the family table in the paper
    assert set(counts) == set(mix)
    for name, p in mix.items():
        assert abs(counts[name] / 4000 - p) < 0.03


def test_style_sets_its_own_settings_and_leaves_the_rest():
    for seed in range(200):
        name, wall = wall_style(seed, ARGS)
        for key, value in STYLES[name].items():
            assert getattr(wall, key) == value
        assert wall.crack_style == name and ARGS.crack_profile == "standard"
        if name == "crazing":
            assert wall.material_category in PAINTED_CATEGORIES


def test_forced_material_survives_network_style():
    forced = Namespace(**{**vars(ARGS), "material_category": "concrete"})
    assert all(wall_style(s, forced)[1].material_category == "concrete" for s in range(100))


def test_unified_palette_has_no_wood_or_irregular_stone():
    counts = collections.Counter(sample_category(s, None, "unified") for s in range(10000))
    assert set(counts) == set(UNIFIED_CATEGORIES) and not set(counts) & set(NEVER_CRACKED)
    assert all(abs(counts[name] / 10000 - 0.1) < 0.02 for name in UNIFIED_CATEGORIES)


def test_unified_clean_table_cracks_stone_block_and_large_tile():
    for category in ("stone_block", "large_tile"):
        rate = sum(is_clean_wall(s, category, 1.0, "unified") for s in range(3000)) / 3000
        assert abs(rate - 0.6) < 0.04
        assert all(is_clean_wall(s, category, 1.0, "diverse") for s in range(50))  # v1.x unchanged


def _path(seed, mode):
    from cwg_infinigen.assets.defect_fields import adapt_crack_morphology, adapt_crack_path, sample_crack_morphology

    p = sample_crack_morphology(seed, half=0.12)
    return adapt_crack_path(adapt_crack_morphology(p, width_scale=1.3, depth_scale=0.5, fit_path=True), seed, mode)


def test_bounded_path_has_no_pinch_or_shear():
    from cwg_infinigen.assets.defect_fields import ANGULAR_MAX_SLOPE, ANGULAR_MIN_SEGMENT_RADII

    steep = 0
    for seed in range(1500):
        bounded = _path(seed, "angular_bounded")
        knots = bounded["path_knots"]
        assert (knots[0][0], knots[-1][0]) == (-bounded["half_v"], bounded["half_v"])  # still spans the card
        for (v0, x0), (v1, x1) in zip(knots, knots[1:]):
            assert v1 - v0 >= ANGULAR_MIN_SEGMENT_RADII * bounded["w_max"] - 1e-9
            assert abs(x1 - x0) <= ANGULAR_MAX_SLOPE * (v1 - v0) + 1e-9
        old = _path(seed, "angular")
        assert bounded["fork_specs"] == old["fork_specs"]  # the same draws, so the rest of the crack is kept
        steep += any(abs(b[1] - a[1]) > ANGULAR_MAX_SLOPE * (b[0] - a[0]) for a, b in zip(old["path_knots"], old["path_knots"][1:]))
    assert steep > 500  # the unbounded mode is what this guards against


def test_bad_mix_is_rejected():
    for text in ("legacy", "nope=1", "legacy=0", "redraw=1", "facade=1"):
        with pytest.raises(ValueError):
            parse_style_mix(text)
    assert set(RECIPES) == {"unified"}
    flags = RECIPES["unified"]
    assert "--crack-style-mix" in flags and "--wall-materials unified" in flags
    assert "--clean-wall-scale 0" in flags and "--shell-finish matched" in flags
    assert "--layout-order fixtures-first" in flags and "--fixture-mount relief" in flags
    assert "--crack-junction-fraction .5" in flags and "--crack-edge-profile" not in flags
    assert "--prop-placement" not in flags
    assert flags.split().count("--crack-path-mode") == 2  # the later flag wins
    assert flags.endswith("--crack-path-mode angular_bounded")


def test_fixtures_first_puts_fixtures_on_every_indoor_wall():
    ff = Namespace(**{**vars(ARGS), "layout_order": "fixtures-first"})
    for seed in range(300):
        name, wall = wall_style(seed, ff)
        assert wall.structures == "auto" and wall.structure_views == 0
        assert wall_style(seed, ARGS)[1].structures == STYLES[name]["structures"]  # legacy order unchanged
