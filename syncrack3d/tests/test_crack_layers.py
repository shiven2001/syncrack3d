"""Coating edges follow the host. Single-material walls stay a groove."""
import pytest
from cwg_infinigen.assets.defect_fields import (
    LIP_KEEPOUT_GROW_M,
    STRONG_MARGIN_PER_WIDTH,
    crack_layer_mode,
    fit_strong_band,
    keepout_in_crack_frame,
    sample_crack_layer,
)

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


def test_strong_band_fits_the_crack_width():
    for seed in range(200):
        for mode in ("paint", "stucco"):
            base = sample_crack_layer(seed, mode, 1.0)
            strong = sample_crack_layer(seed, mode, 1.0, edge_profile="strong")
            hairline = fit_strong_band(strong, 0.0001)
            assert hairline["margin"] == pytest.approx(base["margin"])  # never below the historical band
            assert hairline["chip_scale"] == pytest.approx(base["chip_scale"])
            assert fit_strong_band(strong, 1.0) == strong  # an open crack keeps the whole band
            mid = fit_strong_band(strong, strong["margin"] / STRONG_MARGIN_PER_WIDTH / 2)
            assert base["margin"] <= mid["margin"] <= strong["margin"]
            assert mid["margin"] * mid["chip_scale"] == pytest.approx(base["margin"] * base["chip_scale"])
            for key in ("lip", "film", "chip_thr", "offset", "side", "core"):
                assert mid[key] == strong[key]
    crazing = sample_crack_layer(0, "paint", 1.0, edge_profile="strong", network=True)
    assert fit_strong_band(crazing, 0.0001) == crazing
    assert fit_strong_band(None, 0.001) is None


def test_lip_keepout_rectangles_in_the_crack_frame():
    rects = [(0.10, 0.30, 1.00, 1.20), (2.0, 2.2, 1.0, 1.2)]
    # Crack at (0, 1.1), running along the wall's u: only the near rectangle counts.
    near = keepout_in_crack_frame(rects, (0.0, 1.1), (1.0, 0.0), (0.0, 1.0), reach=0.2)
    assert len(near) == 1
    cu, cv, a, b, hu, hv = near[0]
    assert (cu, cv) == pytest.approx((0.2, 0.0))
    assert (hu, hv) == pytest.approx((0.1 + LIP_KEEPOUT_GROW_M, 0.1 + LIP_KEEPOUT_GROW_M))
    # The same crack turned 90 degrees: its u is the wall's v.
    (cu, cv, a, b, _, _), = keepout_in_crack_frame(rects, (0.0, 1.1), (0.0, 1.0), (-1.0, 0.0), reach=0.2)
    assert (cu, cv) == pytest.approx((0.0, -0.2))
    assert a == pytest.approx((0.0, -1.0)) and b == pytest.approx((1.0, 0.0))
    assert keepout_in_crack_frame(rects, (0.0, 1.1), (1.0, 0.0), (0.0, 1.0), reach=0.01) == []


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
