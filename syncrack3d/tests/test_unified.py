"""v2 unified recipe: per-wall crack styles, uniform palette, one clean table."""
import collections
from argparse import Namespace

import pytest

from syncrack3d.material_diversity import (NEVER_CRACKED, PAINTED_CATEGORIES, UNIFIED_CATEGORIES,
                                           is_clean_wall, sample_category)
from syncrack3d.recipes import REDRAW, RECIPES, STYLES, UNIFIED_MIX, parse_style_mix, wall_style

ARGS = Namespace(crack_style_mix=UNIFIED_MIX, material_category=None, crack_profile="legacy",
                 structures="off")


def test_style_draw_is_deterministic_and_follows_the_mix():
    assert [wall_style(s, ARGS)[0] for s in range(50)] == [wall_style(s, ARGS)[0] for s in range(50)]
    counts = collections.Counter(wall_style(s, ARGS)[0] for s in range(4000))
    mix = parse_style_mix(UNIFIED_MIX)
    kept = 1 - mix.pop(REDRAW)
    assert set(counts) == set(mix)  # a redraw slot never becomes a wall's style
    for name, p in mix.items():
        assert abs(counts[name] / 4000 - p / kept) < 0.03


def test_style_sets_its_own_settings_and_leaves_the_rest():
    for seed in range(200):
        name, wall = wall_style(seed, ARGS)
        for key, value in STYLES[name].items():
            assert getattr(wall, key) == value
        assert wall.crack_style == name and ARGS.crack_profile == "legacy"
        if name == "network":
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


def test_bad_mix_is_rejected():
    for text in ("legacy", "nope=1", "legacy=0", "redraw=1", "facade=1"):
        with pytest.raises(ValueError):
            parse_style_mix(text)
    assert "--crack-style-mix" in RECIPES["unified"] and "--wall-materials unified" in RECIPES["unified"]
    assert "--clean-wall-scale 0" in RECIPES["unified"] and "--shell-finish matched" in RECIPES["unified"]
    assert "--layout-order fixtures-first" in RECIPES["unified"]
    assert "--layout-order" not in RECIPES["unified_v1.16"]
    assert "--prop-placement" not in RECIPES["unified"]
    assert "--fixture-mount relief" in RECIPES["unified"]
    assert "--prop-placement" not in RECIPES["unified_v1.17"]
    assert "--fixture-mount" not in RECIPES["unified_v1.18"]
    assert RECIPES["unified_v1.18"] == RECIPES["unified_v1.17"]


def test_fixtures_first_puts_fixtures_on_every_indoor_wall():
    ff = Namespace(**{**vars(ARGS), "layout_order": "fixtures-first"})
    for seed in range(300):
        name, wall = wall_style(seed, ff)
        assert wall.structures == "auto" and wall.structure_views == 0
        assert wall_style(seed, ARGS)[1].structures == STYLES[name]["structures"]  # legacy order unchanged
