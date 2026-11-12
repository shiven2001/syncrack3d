"""Coating edges follow the host. Single-material walls stay a groove."""
import pytest
from cwg_infinigen.assets.defect_fields import (
    LIP_KEEPOUT_GROW_M,
    STRONG_MARGIN_PER_WIDTH,
    crack_layer_mode,
    fit_strong_band,
    keepout_in_crack_frame,
    sample_crack_layer,
    scale_crack_edge,
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
        assert 0.0002 <= paint["film"] <= 0.0007
        assert 0.002 <= paint["lip"] <= 0.005
        assert 0.001 <= stucco["film"] <= 0.003
        assert 0.002 <= stucco["lip"] <= 0.005
        assert stucco["dark"] >= 0.62


def test_edges_on_ordinary_and_wide_cracks():
    bands = {"paint": ((0.0015, 0.0040), (120.0, 320.0)), "stucco": ((0.003, 0.008), (70.0, 160.0))}
    for seed in range(200):
        for mode, (margin, chips) in bands.items():
            lay = sample_crack_layer(seed, mode, 1.0)
            assert 0.002 <= lay["lip"] <= 0.005
            assert 3.0 <= lay["factor"] <= 5.0
            assert margin[0] <= lay["margin"] / lay["factor"] <= margin[1]
            assert chips[0] <= lay["chip_scale"] * lay["factor"] <= chips[1]


def test_edges_on_crazing_stay_small():
    for seed in range(200):
        plaster = sample_crack_layer(seed, "paint", 1.0, network=True)
        stucco = sample_crack_layer(seed, "stucco", 1.0, network=True)
        assert 0.0005 <= plaster["lip"] <= 0.002
        assert 0.0005 <= stucco["lip"] <= 0.001
        for lay in (plaster, stucco):
            assert 0.0004 <= lay["margin"] <= 0.0008
            assert 0.00007 <= lay["film"] <= 0.00014
            assert 300.0 <= lay["chip_scale"] <= 500.0
            assert lay["offset"] == 0.0  # a net has no single path to step across


def test_band_fits_the_crack_width():
    for seed in range(200):
        for mode in ("paint", "stucco"):
            lay = sample_crack_layer(seed, mode, 1.0)
            base_margin = lay["margin"] / lay["factor"]
            base_chips = lay["chip_scale"] * lay["factor"]
            hairline = fit_strong_band(lay, 0.0001)
            assert hairline["margin"] == pytest.approx(base_margin)  # never below the historical band
            assert hairline["chip_scale"] == pytest.approx(base_chips)
            assert fit_strong_band(lay, 1.0) == lay  # an open crack keeps the whole band
            mid = fit_strong_band(lay, lay["margin"] / STRONG_MARGIN_PER_WIDTH / 2)
            assert base_margin <= mid["margin"] <= lay["margin"]
            assert mid["margin"] * mid["chip_scale"] == pytest.approx(base_margin * base_chips)
            for key in ("lip", "film", "chip_thr", "offset", "side", "core"):
                assert mid[key] == lay[key]
    crazing = sample_crack_layer(0, "paint", 1.0, network=True)
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


def test_edge_scale_changes_lip_and_chips_together():
    lay = sample_crack_layer(3, "paint", 1.0)
    assert scale_crack_edge(lay, 1) is lay
    assert scale_crack_edge(None, 2) is None
    twice = scale_crack_edge(lay, 2)
    assert twice["lip"] == pytest.approx(lay["lip"] * 2)
    assert twice["film"] == pytest.approx(lay["film"] * 2)
    assert twice["offset"] == pytest.approx(lay["offset"] * 2)
    assert twice["margin"] == pytest.approx(lay["margin"] * 2)
    assert twice["chip_scale"] == pytest.approx(lay["chip_scale"] / 2)
    flat = scale_crack_edge(lay, 0)
    assert (flat["lip"], flat["film"], flat["offset"], flat["margin"]) == (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        scale_crack_edge(lay, -1)


def test_fraction_can_skip_and_rejects_an_unknown_mode():
    skipped = [sample_crack_layer(seed, "paint", 0.0) for seed in range(10)]
    assert skipped == [None] * 10
    with pytest.raises(ValueError):
        sample_crack_layer(0, "brick", 1.0)
