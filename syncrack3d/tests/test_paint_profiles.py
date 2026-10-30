from dataclasses import replace

import pytest

from syncrack3d.paint_profiles import adapt_paint_material
from syncrack3d.spec import SceneSpec, sample_scene_spec


def test_baseline_and_nonpaint_are_unchanged():
    for finish in ("paint", "concrete", "masonry"):
        material = sample_scene_spec(12, substrate=finish).material
        assert adapt_paint_material(material, 12) is material
        if finish != "paint":
            assert adapt_paint_material(material, 12, "mixed") is material


@pytest.mark.parametrize("profile", ["smooth", "roller", "rough", "mixed"])
def test_paint_profiles_are_deterministic_and_serializable(profile):
    original = sample_scene_spec(12, substrate="paint")
    material = adapt_paint_material(original.material, 12, profile)
    assert material == adapt_paint_material(original.material, 12, profile)
    assert original.material.paint_profile == "baseline"
    assert material.base_color == original.material.base_color
    assert material.meso_scale > 0
    assert 0 < material.meso_bump_distance_m < 0.001
    spec = replace(original, material=material)
    assert SceneSpec.from_dict(spec.to_dict()) == spec


def test_mixed_spans_all_three_profiles():
    material = sample_scene_spec(0, substrate="paint").material
    assert {
        adapt_paint_material(material, s, "mixed").paint_profile for s in range(50)
    } == {"smooth", "roller", "rough"}


def test_invalid_profile_rejected():
    with pytest.raises(ValueError):
        adapt_paint_material(sample_scene_spec(0).material, 0, "unknown")
