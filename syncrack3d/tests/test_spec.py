from syncrack3d.spec import MODULE_KINDS, SUBSTRATES, SceneSpec, sample_scene_spec


def test_sampling_is_deterministic():
    assert sample_scene_spec(31) == sample_scene_spec(31)
    assert sample_scene_spec(31) != sample_scene_spec(32)


def test_simple_wall_and_every_substrate_is_valid():
    assert MODULE_KINDS == ("flat",)
    for index, substrate in enumerate(SUBSTRATES):
        spec = sample_scene_spec(index + 4, module="flat", substrate=substrate)
        assert spec.module == "flat"
        assert spec.material.finish == substrate
        assert spec.width_m > 0
        assert spec.height_m > 0


def test_spec_json_round_trip(tmp_path):
    original = sample_scene_spec(9, module="flat", substrate="paint")
    path = tmp_path / "scene.spec.json"
    original.to_json(path)
    assert SceneSpec.from_json(path) == original


def test_explicit_dimensions_are_preserved():
    spec = sample_scene_spec(
        5,
        module="flat",
        substrate="concrete",
        width_m=4.5,
        height_m=2.8,
        thickness_m=0.18,
        skirting=False,
    )
    assert (spec.width_m, spec.height_m, spec.thickness_m) == (4.5, 2.8, 0.18)
    assert spec.skirting is False
