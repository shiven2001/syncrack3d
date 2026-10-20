"""Background details must reproduce without changing defect RNG or labels."""
import random
from types import SimpleNamespace

import numpy as np
import pytest

from syncrack3d.spec import sample_scene_spec
from syncrack3d.structures import (
    KINDS, build_structures, sample_structure_plan, structure_camera_targets,
    structure_clean_wall,
)


def test_reproducible_independent_plan_and_clean_sampling():
    random.seed(77)
    np.random.seed(77)
    before = random.getstate(), np.random.get_state()
    for seed in range(40):
        spec = sample_scene_spec(seed)
        a = sample_structure_plan(seed, spec, profile='all')
        assert a == sample_structure_plan(seed, spec, profile='all')
        assert sample_structure_plan(seed, spec, profile='off') == []
        assert structure_clean_wall(seed, 'auto', 1)
        assert not structure_clean_wall(seed, 'off', 1)
        assert not structure_clean_wall(seed, 'all', 0)
        for p in a:
            assert abs(p.u_m) + p.width_m * .5 <= spec.width_m * .5 + 1e-6
            assert 0 <= p.v_m <= spec.height_m
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])


def test_large_assemblies_clear_crack_centres_and_services():
    for seed in range(30):
        spec = sample_scene_spec(seed)
        targets = [SimpleNamespace(u_m=0., v_m=spec.height_m * .5)]
        plans = sample_structure_plan(seed, spec, targets, 'all')
        for i, p in enumerate(plans):
            if p.kind not in ('window', 'utility_box', 'ledge'):
                continue
            assert abs(p.u_m) >= p.width_m * .5 + .06 or abs(p.v_m - targets[0].v_m) >= p.height_m * .5 + .06
            for q in plans[:i]:
                if q.kind == "panel_seam":
                    continue
                assert abs(p.u_m - q.u_m) >= (p.width_m + q.width_m) * .5 + .05 or abs(p.v_m - q.v_m) >= (p.height_m + q.height_m) * .5 + .05


def test_geometry_is_background_and_structure_cameras_have_no_crack_target():
    import bpy
    from cwg_infinigen.assets.defect_injection import classify_plane
    from syncrack3d.cameras import sample_camera_poses
    from syncrack3d.build_dataset import clear_scene
    clear_scene()
    try:
        spec = sample_scene_spec(191000)
        from dataclasses import replace
        plans = [replace(sample_structure_plan(191000, spec, include_kinds=[kind])[0], index=i)
                 for i, kind in enumerate(KINDS)]
        records = build_structures(plans)
        assert set(r['kind'] for r in records) == set(KINDS)
        for record in records:
            for name in record['objects']:
                obj = bpy.data.objects[name]
                assert obj.get('cwg_role') == 'structure'
                assert not obj.get('cwg_host')
                assert classify_plane(obj) is None
                for mat in obj.data.materials:
                    assert not any(n.bl_idname == 'ShaderNodeOutputAOV' for n in mat.node_tree.nodes)
        cameras = sample_camera_poses(random.Random(1), spec, structure_camera_targets(records),
                                      bpy.context.scene.collection, index_offset=6,
                                      closeup_role='structure', include_context=False)
        assert [c.index for c in cameras] == [6, 7]
        for c in cameras:
            assert c.target_index is None
            assert 'syndefect_target' not in bpy.data.objects[c.object_name]
    finally:
        clear_scene()


@pytest.mark.parametrize('fraction', [-.1, 1.1, float('nan')])
def test_invalid_clean_fraction(fraction):
    with pytest.raises(ValueError):
        structure_clean_wall(0, 'auto', fraction)


def test_rendered_structures_occlude_crack_aov(tmp_path):
    """A fully labelled wall must lose coverage at opaque structure surfaces."""
    import bpy
    import OpenEXR
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.geometry import _box
    from syncrack3d.materials import make_solid_material
    clear_scene()
    try:
        spec = sample_scene_spec(191000)
        records = build_structures(sample_structure_plan(191000, spec, profile='all'))
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 8
        scene.cycles.use_denoising = False
        scene.render.resolution_x = scene.render.resolution_y = 192
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = True
        for name in ('DefectCrack', 'StructureQA'):
            aov = scene.view_layers[0].aovs.add()
            aov.name, aov.type = name, 'VALUE'
        wall_mat = make_solid_material('QA_labelled_wall', (.5, .5, .5, 1))
        node = wall_mat.node_tree.nodes.new('ShaderNodeOutputAOV')
        node.aov_name = 'DefectCrack'
        node.inputs['Value'].default_value = 1
        _box('QA_wall', (spec.width_m, .1, spec.height_m), (0, .05, spec.height_m * .5),
             wall_mat, scene.collection, None)
        materials = {m for rec in records for name in rec['objects'] for m in bpy.data.objects[name].data.materials}
        for mat in materials:
            node = mat.node_tree.nodes.new('ShaderNodeOutputAOV')
            node.aov_name = 'StructureQA'
            node.inputs['Value'].default_value = 1
        bpy.ops.object.camera_add(location=(0, -7, spec.height_m * .5))
        camera = bpy.context.object
        from mathutils import Vector
        camera.rotation_euler = Vector((0, 1, 0)).to_track_quat('-Z', 'Y').to_euler()
        camera.data.type = 'ORTHO'
        camera.data.ortho_scale = spec.width_m * 1.1
        scene.camera = camera
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        layers = tree.nodes.new('CompositorNodeRLayers')
        for name in ('DefectCrack', 'StructureQA'):
            output = tree.nodes.new('CompositorNodeOutputFile')
            output.base_path = str(tmp_path)
            output.format.file_format = 'OPEN_EXR'
            output.format.color_depth = '32'
            output.file_slots[0].path = name
            tree.links.new(layers.outputs[name], output.inputs[0])
        bpy.ops.render.render()
        arrays = {}
        for name in ('DefectCrack', 'StructureQA'):
            with OpenEXR.File(str(next(tmp_path.glob(name + '*.exr')))) as f:
                channels = f.channels()
                pixels = channels['V'].pixels
                arrays[name] = pixels[..., 0] if pixels.ndim == 3 else pixels
        structure_pixels = arrays['StructureQA'] > .99
        assert structure_pixels.sum() > 100
        assert (arrays['DefectCrack'] > .99).sum() > 1000
        assert arrays['DefectCrack'][structure_pixels].max() < .01
    finally:
        for aov in list(bpy.context.scene.view_layers[0].aovs):
            if aov.name in ('DefectCrack', 'StructureQA'):
                bpy.context.scene.view_layers[0].aovs.remove(aov)
        clear_scene()


def test_clean_scene_has_explicit_empty_annotation_contract(tmp_path):
    import json
    import bpy
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.injection_adapter import run_controlled_injection
    from syncrack3d.render import setup_passes
    clear_scene()
    manifest = tmp_path / 'defect_instances.json'
    assert run_controlled_injection([], manifest_path=manifest, scene_id='clean', scene_seed=2) == 0
    assert json.loads(manifest.read_text())['count'] == 0
    names = setup_passes()
    assert set(names) == {'DefectCrack', 'DefectInstance', 'DefectInstanceCoverage', 'DefectInstanceSquared'}
    assert all(a.type == 'VALUE' for a in bpy.context.view_layer.aovs if a.name in names)


def test_structure_view_curation_keeps_only_true_negatives(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'tools/prepare_defect_annotated_dataset.py'
    spec = importlib.util.spec_from_file_location('structure_export_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = {'camera_role': 'structure', 'instances': [{'label_status': 'not_visible'}]}
    def presence():
        (tmp_path / 'sample.json').write_text(json.dumps(payload))
        return module._frame_defect_presence('sample', tmp_path)
    assert presence() == 'none'
    payload['camera_role'] = 'closeup'
    assert presence() == 'failed_target'
    payload['camera_role'] = 'structure'
    for status in ('visible', 'under_resolution', 'no_contrast'):
        payload['instances'][0]['label_status'] = status
        assert presence() == 'present'
    payload['instances'][0]['label_status'] = 'not_visible'
    payload['ambiguous_pixel_count'] = 5
    assert presence() == 'failed_target'


def test_environment_pools_and_mounting_heights():
    from syncrack3d.structures import INDOOR_KINDS, OUTDOOR_KINDS
    seen = {'indoor': set(), 'outdoor': set()}
    for environment in seen:
        for seed in range(150):
            spec = sample_scene_spec(seed)
            plans = sample_structure_plan(seed, spec, profile='auto', environment=environment)
            seen[environment].update(p.kind for p in plans)
            for p in plans:
                if p.kind == 'socket':
                    assert .25 <= p.v_m <= .45
                elif p.kind == 'switch':
                    assert 1.05 <= p.v_m <= 1.35
                elif p.kind in ('door', 'skirting'):
                    assert p.v_m == p.height_m * .5
                elif p.kind in ('ceiling_trim', 'gutter'):
                    assert p.v_m + p.height_m * .5 == pytest.approx(spec.height_m)
    assert set(INDOOR_KINDS) <= seen['indoor']
    assert set(OUTDOOR_KINDS) <= seen['outdoor']
    assert not set(OUTDOOR_KINDS) & seen['indoor']
    assert not set(INDOOR_KINDS) & seen['outdoor']


def test_skirting_clears_door_opening():
    import bpy
    from syncrack3d.build_dataset import clear_scene
    clear_scene()
    try:
        spec = sample_scene_spec(193000)
        plans = sample_structure_plan(193000, spec, include_kinds=['door', 'skirting'])
        assert {p.kind for p in plans} == {'door', 'skirting'}
        records = build_structures(plans)
        door = next(p for p in plans if p.kind == 'door')
        for rec in records:
            if rec['kind'] == 'skirting':
                for name in rec['objects']:
                    obj = bpy.data.objects[name]
                    assert obj.location.x + obj.dimensions.x*.5 <= door.u_m-door.width_m*.5 + 1e-5 or obj.location.x-obj.dimensions.x*.5 >= door.u_m+door.width_m*.5 - 1e-5
    finally:
        clear_scene()


def test_boundary_fixture_cameras_stay_above_floor_and_below_ceiling():
    import bpy
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.cameras import sample_camera_poses
    clear_scene()
    try:
        spec = sample_scene_spec(193000)
        records = build_structures(sample_structure_plan(193000, spec, include_kinds=['skirting', 'ceiling_trim']))
        cameras = sample_camera_poses(random.Random(9), spec, structure_camera_targets(records),
                                      bpy.context.scene.collection, closeups_per_defect=8,
                                      closeup_role='structure', include_context=False, max_pitch_deg=70)
        for c in cameras:
            assert .07999 <= c.location[2] <= spec.height_m-.07999
            assert -spec.width_m*.5+.07999 <= c.location[0] <= spec.width_m*.5-.07999
    finally:
        clear_scene()
