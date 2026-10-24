"""v2 unified recipe: per-wall crack styles, uniform palette, one clean table."""
import collections
from argparse import Namespace

import pytest

from syncrack3d.material_diversity import (NEVER_CRACKED, PAINTED_CATEGORIES, UNIFIED_CATEGORIES,
                                           is_clean_wall, sample_category)
from syncrack3d.recipes import RECIPES, STYLES, UNIFIED_MIX, parse_style_mix, wall_style

ARGS = Namespace(crack_style_mix=UNIFIED_MIX, material_category=None, crack_profile="legacy",
                 structures="off", environment="indoor")


def test_style_draw_is_deterministic_and_follows_the_mix():
    assert [wall_style(s, ARGS)[0] for s in range(50)] == [wall_style(s, ARGS)[0] for s in range(50)]
    counts = collections.Counter(wall_style(s, ARGS)[0] for s in range(4000))
    for name, p in parse_style_mix(UNIFIED_MIX).items():
        assert abs(counts[name] / 4000 - p) < 0.03


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
    drawn = {sample_category(s, None, "unified") for s in range(3000)}
    assert drawn == set(UNIFIED_CATEGORIES) and not drawn & set(NEVER_CRACKED)


def test_unified_clean_table_cracks_stone_block_and_large_tile():
    for category in ("stone_block", "large_tile"):
        rate = sum(is_clean_wall(s, category, 1.0, "unified") for s in range(3000)) / 3000
        assert abs(rate - 0.6) < 0.04
        assert all(is_clean_wall(s, category, 1.0, "diverse") for s in range(50))  # v1.x unchanged


def test_bad_mix_is_rejected():
    for text in ("legacy", "nope=1", "legacy=0"):
        with pytest.raises(ValueError):
            parse_style_mix(text)
    assert "--crack-style-mix" in RECIPES["unified"] and "--wall-materials unified" in RECIPES["unified"]
    assert "--clean-wall-scale 0" in RECIPES["unified"] and "--shell-finish matched" in RECIPES["unified"]
