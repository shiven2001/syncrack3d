"""Per-wall crack styles: deterministic draw, style settings, fixtures, network materials."""
import collections
from argparse import Namespace

import pytest

from syncrack3d.material_diversity import PAINTED_CATEGORIES
from syncrack3d.styles import DEFAULT_MIX, STYLE_DEFAULTS, STYLES, parse_style_mix, wall_style

ARGS = Namespace(crack_style_mix=DEFAULT_MIX, material_categories=None, no_fixtures=False)


def test_style_draw_is_deterministic_and_follows_the_mix():
    assert [wall_style(s, ARGS)[0] for s in range(50)] == [wall_style(s, ARGS)[0] for s in range(50)]
    counts = collections.Counter(wall_style(s, ARGS)[0] for s in range(4000))
    for name, p in parse_style_mix(DEFAULT_MIX).items():
        assert abs(counts[name] / 4000 - p) < 0.03


def test_style_sets_its_own_settings_and_defaults_the_rest():
    for seed in range(200):
        name, wall = wall_style(seed, ARGS)
        for key, value in {**STYLE_DEFAULTS, **STYLES[name]}.items():
            assert getattr(wall, key) == value
        assert wall.crack_style == name and not hasattr(ARGS, "crack_style")
        assert (wall.material_category in PAINTED_CATEGORIES) == (name == "network")


def test_every_wall_has_fixtures_unless_ablated():
    for seed in range(300):
        wall = wall_style(seed, ARGS)[1]
        assert wall.structures == ("facade" if wall.environment == "outdoor" else "auto")
    off = Namespace(**{**vars(ARGS), "no_fixtures": True})
    assert {wall_style(s, off)[1].structures for s in range(100)} == {"off"}


def test_network_walls_respect_a_material_subset():
    concrete = Namespace(**{**vars(ARGS), "material_categories": ["concrete"]})
    assert all(wall_style(s, concrete)[1].material_category is None for s in range(100))
    stucco = Namespace(**{**vars(ARGS), "material_categories": ["concrete", "stucco_colour"]})
    for s in range(300):
        name, wall = wall_style(s, stucco)
        assert wall.material_category == ("stucco_colour" if name == "network" else None)


def test_bad_mix_is_rejected():
    for text in ("legacy", "nope=1", "legacy=0"):
        with pytest.raises(ValueError):
            parse_style_mix(text)
