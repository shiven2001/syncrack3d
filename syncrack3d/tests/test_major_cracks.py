"""Morphology contract: reproducible profiles and shared metric field inputs."""
import numpy as np
import pytest
from cwg_infinigen.assets.defect_fields import sample_crack_morphology


def test_profiles_are_reproducible_and_preserve_rng():
    np.random.seed(42)
    before = np.random.get_state()
    for profile in ('legacy', 'major', 'mixed'):
        for seed in range(30):
            assert sample_crack_morphology(seed, profile=profile) == sample_crack_morphology(seed, profile=profile)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_major_and_mixed_ranges():
    major_count = 0
    for seed in range(100):
        p = sample_crack_morphology(seed, profile='major')
        assert p['style'] in ('major_open', 'major_branched')
        assert .0015 <= p['w_min'] <= .0035
        assert .005 <= p['w_max'] <= .012
        assert .008 <= p['crack_depth'] <= .018
        assert p['fork_specs']
        m = sample_crack_morphology(seed, profile='mixed')
        major_count += m['style'].startswith('major_')
        if not m['style'].startswith('major_'):
            assert m == sample_crack_morphology(seed)
    assert 25 < major_count < 75


def test_invalid_profile():
    with pytest.raises(ValueError):
        sample_crack_morphology(0, profile='unknown')
