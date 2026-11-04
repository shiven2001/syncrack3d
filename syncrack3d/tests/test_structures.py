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


def test_large_assemblies_clear_crack_centres():
    from syncrack3d.structures import LARGE_KINDS
    for seed in range(30):
        spec = sample_scene_spec(seed)
        targets = [SimpleNamespace(u_m=0., v_m=spec.height_m * .5)]
        for p in sample_structure_plan(seed, spec, targets, 'all'):
            if p.kind in LARGE_KINDS:
                assert abs(p.u_m) >= p.width_m * .5 + .06 or abs(p.v_m - targets[0].v_m) >= p.height_m * .5 + .06


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
            assert 'syncrack3d_target' not in bpy.data.objects[c.object_name]
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


def test_service_runs_span_either_axis():
    from syncrack3d.structures import SPAN_KINDS, SPAN_SECTION
    spec = SimpleNamespace(width_m=4.0, height_m=2.6, skirting=False)
    seen = {kind: set() for kind in SPAN_KINDS}
    for seed in range(160):
        for plan in sample_structure_plan(seed, spec, profile="auto"):
            if plan.kind in SPAN_KINDS:
                seen[plan.kind].add("side" if abs(plan.width_m - spec.width_m) < 1e-6 else "down")
    for kind, axes in seen.items():
        assert axes == {"side", "down"}, kind
    for kind in SPAN_KINDS:
        axes = set()
        for seed in range(24):
            plan = sample_structure_plan(seed, spec, include_kinds=[kind])[0]
            if abs(plan.width_m - spec.width_m) < 1e-6:
                assert plan.height_m == pytest.approx(SPAN_SECTION[kind])
                axes.add("side")
            else:
                assert plan.height_m == pytest.approx(spec.height_m)
                assert plan.width_m == pytest.approx(SPAN_SECTION[kind])
                axes.add("down")
        assert axes == {"side", "down"}, kind


def test_parallel_and_loose_counts_cover_their_ranges():
    from syncrack3d.structures import loose_cable_count, run_count
    runs, cables = set(), set()
    for seed in range(40):
        runs.add(run_count(random.Random(seed)))
        cables.add(loose_cable_count(random.Random(seed + 1000)))
    assert runs == {2, 3, 4}
    assert cables == {1, 2, 3}


def test_auto_pool_and_mounting_heights():
    from syncrack3d.structures import INDOOR_KINDS
    seen = set()
    for seed in range(150):
        spec = sample_scene_spec(seed)
        plans = sample_structure_plan(seed, spec, profile='auto')
        seen.update(p.kind for p in plans)
        for p in plans:
            if p.kind == 'door':
                assert p.v_m == p.height_m * .5
    assert set(INDOOR_KINDS) <= seen
    assert 'cable' in seen  # loose cable is part of the indoor auto pool


def test_boundary_fixture_cameras_stay_above_floor_and_below_ceiling():
    import bpy
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.cameras import sample_camera_poses
    clear_scene()
    try:
        spec = sample_scene_spec(193000)
        records = build_structures(sample_structure_plan(193000, spec, include_kinds=['door']))
        cameras = sample_camera_poses(random.Random(9), spec, structure_camera_targets(records),
                                      bpy.context.scene.collection, closeups_per_defect=8,
                                      closeup_role='structure', include_context=False, max_pitch_deg=70)
        for c in cameras:
            assert .07999 <= c.location[2] <= spec.height_m-.07999
            assert -spec.width_m*.5+.07999 <= c.location[0] <= spec.width_m*.5-.07999
    finally:
        clear_scene()


def test_fixtures_stand_on_the_wall_relief():
    bpy = pytest.importorskip("bpy")
    from dataclasses import replace

    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.occlusion import _world_bounds
    spec = sample_scene_spec(191000)
    plans = [replace(sample_structure_plan(191000, spec, include_kinds=[kind])[0], index=i)
             for i, kind in enumerate(KINDS)]
    backs = []
    for relief in (0.0, 0.034):
        clear_scene()
        records = build_structures(plans, relief)
        backs.append({r['kind']: _world_bounds(bpy.data.objects[r['object_name']])[1][1] for r in records
                      if _world_bounds(bpy.data.objects[r['object_name']]) is not None})
    for kind, back in backs[0].items():
        assert backs[1][kind] == pytest.approx(back - 0.034, abs=1e-5), kind
        if back <= 1e-5:  # a window sill is built 1 cm into the wall on purpose
            assert backs[1][kind] <= -0.034 + 1e-5, kind  # nothing sunk into the relief


def test_fixtures_stay_clear_of_each_other_and_on_the_wall():
    from syncrack3d.structures import FIXTURE_GAP_M, _overlaps, built_extent
    trims = ("door",)
    for seed in range(200):
        spec = sample_scene_spec(seed)
        targets = [SimpleNamespace(u_m=0., v_m=spec.height_m * .5)]
        for profile, near in (("auto", ()), ("all", ()), ("all", targets)):
            plans = sample_structure_plan(seed, spec, near, profile)
            boxes = [(p.kind, built_extent(p.kind, p.u_m, p.v_m, p.width_m, p.height_m)) for p in plans
                     if p.kind != "panel_seam"]
            base = (-spec.width_m * .5, spec.width_m * .5, 0., .085)
            for i, (a, ea) in enumerate(boxes):
                if a not in trims:
                    assert -spec.width_m * .5 - 1e-9 <= ea[0] and ea[1] <= spec.width_m * .5 + 1e-9, (seed, a)
                    assert -1e-9 <= ea[2] and ea[3] <= spec.height_m + 1e-9, (seed, a)
                    if spec.skirting:
                        assert not _overlaps(ea, base, FIXTURE_GAP_M - 1e-9), (seed, a)
                for b, eb in boxes[:i]:
                    if a in trims and b in trims:
                        continue  # trims meet each other and run under a door's cut
                    assert not _overlaps(ea, eb, FIXTURE_GAP_M - 1e-9), (seed, profile, a, b)


def test_built_extent_covers_the_built_geometry():
    bpy = pytest.importorskip("bpy")
    from dataclasses import replace

    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.occlusion import _world_bounds
    from syncrack3d.structures import built_extent
    for seed in (191000, 191001, 191002):
        spec = sample_scene_spec(seed)
        plans = [replace(sample_structure_plan(seed, spec, include_kinds=[kind])[0], index=i)
                 for i, kind in enumerate(KINDS) if kind not in ("door", "panel_seam")]
        clear_scene()
        for r in build_structures(plans):
            lo, hi = _world_bounds(bpy.data.objects[r['object_name']])
            u0, u1, v0, v1 = built_extent(r['kind'], r['u_m'], r['v_m'], r['width_m'], r['height_m'])
            assert u0 - 1e-3 <= lo[0] and hi[0] <= u1 + 1e-3, (r['kind'], lo[0], hi[0], u0, u1)
            assert v0 - 1e-3 <= lo[2] and hi[2] <= v1 + 1e-3, (r['kind'], lo[2], hi[2], v0, v1)
