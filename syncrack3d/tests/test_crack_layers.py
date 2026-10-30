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


def test_fraction_can_skip_and_rejects_an_unknown_mode():
    skipped = [sample_crack_layer(seed, "paint", 0.0) for seed in range(10)]
    assert skipped == [None] * 10
    with pytest.raises(ValueError):
        sample_crack_layer(0, "brick", 1.0)
