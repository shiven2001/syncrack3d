"""Straight path mode: near-straight run, sparse forks, no global RNG use."""
import numpy as np
from cwg_infinigen.assets.defect_fields import (
    adapt_crack_branches, adapt_crack_morphology, adapt_crack_path, sample_crack_morphology)


def _straight(seed, profile='legacy'):
    p = adapt_crack_morphology(sample_crack_morphology(seed, half=.12, profile=profile), fit_path=True)
    return adapt_crack_path(p, seed, 'straight')


def test_straight_is_reproducible_and_preserves_rng():
    np.random.seed(7)
    before = np.random.get_state()
    for seed in range(30):
        assert _straight(seed) == _straight(seed)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])


def test_straight_run_and_forks():
    for profile in ('legacy', 'major'):
        for seed in range(100):
            p = _straight(seed, profile)
            xs = [k[1] for k in p['path_knots']]
            assert 2 <= len(xs) <= 4
            assert max(xs) - min(xs) < .2 * p['half_u']
            assert len(p['fork_specs']) <= 1
            assert p['jag_amp'] <= .006 * p['half_u']
            adapt_crack_branches(p, 'local_curved')  # fitted parent leaves room for branches
