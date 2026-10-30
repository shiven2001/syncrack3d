"""Network morphology must be reproducible, metric and independent of legacy RNG."""
import numpy as np
import pytest
from cwg_infinigen.assets.defect_fields import (
    adapt_crack_morphology, build_crack_field, sample_crack_network,
)


def test_network_rng_and_metric_controls():
    np.random.seed(193)
    before = np.random.get_state()
    networks = [sample_crack_network(seed, .14) for seed in range(30)]
    assert networks == [sample_crack_network(seed, .14) for seed in range(30)]
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert len({p['network']['spacing_m'] for p in networks}) == 30
    for p in networks:
        assert .02 <= p['network']['spacing_m'] <= .06
        assert 0 < p['w_min'] < p['w_max'] < .0004
        assert 0 < p['crack_depth'] < .0002
        wider = adapt_crack_morphology(p, width_scale=2, depth_scale=.5)
        assert wider['network'] == p['network']
        assert wider['w_min'] == 2 * p['w_min']
        assert wider['crack_depth'] == .5 * p['crack_depth']


@pytest.mark.parametrize('spacing', [(0, .1), (.1, .01), (-.1, .1), (.01, float('inf'))])
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        sample_crack_network(0, .14, spacing)


def test_network_builds_real_shader_for_both_surface_axes():
    import bpy
    for axes in [('X', 'Y'), ('X', 'Z')]:
        material = bpy.data.materials.new('network_test')
        try:
            material.use_nodes = True
            nt = material.node_tree
            coords = nt.nodes.new('ShaderNodeTexCoord').outputs['Object']
            field = build_crack_field(nt, coords, sample_crack_network(7, .14), *axes)
            assert field.node.id_data == nt
            cells = [n for n in nt.nodes if n.bl_idname == 'ShaderNodeTexVoronoi']
            assert len(cells) == 1
            assert cells[0].feature == 'DISTANCE_TO_EDGE'
            assert cells[0].voronoi_dimensions == '2D'
            assert all(link.is_valid for link in nt.links)
        finally:
            bpy.data.materials.remove(material)


def test_network_grime_range_only_rescales_staining():
    for seed in range(30):
        base = sample_crack_network(seed, .14)
        dark = sample_crack_network(seed, .14, grime_range=(.8, 1.))
        assert .8 <= dark['network']['grime_strength'] <= 1.
        assert {k: v for k, v in dark['network'].items() if k != 'grime_strength'} == \
            {k: v for k, v in base['network'].items() if k != 'grime_strength'}
        assert {k: v for k, v in dark.items() if k != 'network'} == {k: v for k, v in base.items() if k != 'network'}
    with pytest.raises(ValueError):
        sample_crack_network(0, .14, grime_range=(.6, .2))


def test_organic_edge_keeps_square_draws():
    for seed in range(30):
        square = sample_crack_network(seed, .14)
        organic = sample_crack_network(seed, .14, edge='organic')
        assert 'edge' not in square['network']
        assert organic == sample_crack_network(seed, .14, edge='organic')
        assert {k: v for k, v in organic['network'].items() if k != 'edge'} == square['network']
        assert {k: v for k, v in organic.items() if k != 'network'} == {k: v for k, v in square.items() if k != 'network'}
        e = organic['network']['edge']
        assert 0 < e['inner'] < e['outer'] <= .95 and all(.75 <= a <= 1 for a in e['aspect'])
    with pytest.raises(ValueError):
        sample_crack_network(0, .14, edge='round')
