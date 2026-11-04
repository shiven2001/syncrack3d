"""Coating edges follow the host. Single-material walls stay a groove."""
import pytest
from cwg_infinigen.assets.defect_fields import crack_layer_mode, sample_crack_layer

MASONRY = (
    "Brick", "Concrete", "NonWoodTile", "LargeTile", "Marble", "MarbleRegular",
    "MarbleVoronoi", "StoneBlock", "IrregularStone", "CobbleWall", "RubbleStone",
    "SandstoneAshlar", "InteriorWood",
)


def test_layer_mode_follows_the_host():
    assert crack_layer_mode("Plaster") == "paint"
    assert crack_layer_mode("StuccoPlaster") == "stucco"
    assert crack_layer_mode("", "paint") == "paint"
    assert crack_layer_mode(None, "masonry") == "none"
    for gen in MASONRY:
        assert crack_layer_mode(gen, "paint") == "none"


def test_single_material_hosts_never_grow_a_film():
    for seed in range(40):
        assert sample_crack_layer(seed, "none", 1.0) is None


def test_paint_keeps_a_film_and_stucco_breaks_its_coat():
    for seed in range(40):
        paint = sample_crack_layer(seed, "paint", 1.0)
        stucco = sample_crack_layer(seed, "stucco", 1.0)
        assert paint["mode"] == "paint" and stucco["mode"] == "stucco"
        assert 0.0015 <= paint["margin"] <= 0.004
        assert 0.0002 <= paint["film"] <= 0.0007
        assert paint["lip"] > 0
        assert 0.003 <= stucco["margin"] <= 0.008
        assert 0.001 <= stucco["film"] <= 0.003
        assert stucco["lip"] == 0.0
        assert stucco["dark"] >= 0.62


def test_strong_edges_on_ordinary_and_wide_cracks():
    for seed in range(200):
        for mode in ("paint", "stucco"):
            base = sample_crack_layer(seed, mode, 1.0)
            strong = sample_crack_layer(seed, mode, 1.0, edge_profile="strong")
            assert 0.002 <= strong["lip"] <= 0.005
            factor = strong["margin"] / base["margin"]
            assert 3.0 <= factor <= 5.0
            assert strong["chip_scale"] == pytest.approx(base["chip_scale"] / factor)
            for key in ("film", "chip_thr", "offset", "side", "dark", "core"):
                assert strong[key] == base[key]


def test_strong_edges_on_crazing_stay_small():
    for seed in range(200):
        plaster = sample_crack_layer(seed, "paint", 1.0, edge_profile="strong", network=True)
        stucco = sample_crack_layer(seed, "stucco", 1.0, edge_profile="strong", network=True)
        assert 0.0005 <= plaster["lip"] <= 0.002
        assert 0.0005 <= stucco["lip"] <= 0.001
        for lay in (plaster, stucco):
            assert 0.0004 <= lay["margin"] <= 0.0008
            assert 0.00007 <= lay["film"] <= 0.00014
            assert 300.0 <= lay["chip_scale"] <= 500.0
            assert lay["offset"] == 0.0  # a net has no single path to step across


def test_strong_edges_land_on_the_same_cracks():
    for seed in range(200):
        for mode in ("paint", "stucco", "none"):
            for network in (False, True):
                base = sample_crack_layer(seed, mode, 0.7)
                strong = sample_crack_layer(seed, mode, 0.7, edge_profile="strong", network=network)
                assert (base is None) == (strong is None)


def test_original_edges_are_the_default():
    for seed in range(40):
        assert sample_crack_layer(seed, "paint", 1.0) == sample_crack_layer(seed, "paint", 1.0, edge_profile="original")
    with pytest.raises(ValueError):
        sample_crack_layer(0, "paint", 1.0, edge_profile="nope")


def test_fraction_can_skip_and_rejects_an_unknown_mode():
    skipped = [sample_crack_layer(seed, "paint", 0.0) for seed in range(10)]
    assert skipped == [None] * 10
    with pytest.raises(ValueError):
        sample_crack_layer(0, "brick", 1.0)
