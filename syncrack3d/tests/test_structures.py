"""Background details must reproduce without changing defect RNG or labels."""
import random

import numpy as np
import pytest

from syncrack3d.spec import sample_scene_spec
from syncrack3d.structures import KINDS, build_structures, sample_structure_plan


def test_reproducible_independent_plan():
    random.seed(77)
    np.random.seed(77)
    before = random.getstate(), np.random.get_state()
    for seed in range(40):
        spec = sample_scene_spec(seed)
        a = sample_structure_plan(seed, spec, profile='all')
        assert a == sample_structure_plan(seed, spec, profile='all')
        assert sample_structure_plan(seed, spec, profile='off') == []
        for p in a:
            assert abs(p.u_m) + p.width_m * .5 <= spec.width_m * .5 + 1e-6
            assert 0 <= p.v_m <= spec.height_m
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])


def test_no_two_fixtures_collide():
    """Every fixture, runs included, by the one shared rule: what it covers once
    built stays clear of every other fixture at an overlapping depth."""
    from syncrack3d.structures import RULES, _collides, plan_shapes

    for seed in range(40):
        spec = sample_scene_spec(seed)
        for env in ("indoor", "outdoor"):
            for profile in ("all", "auto"):
                plans = sample_structure_plan(seed, spec, profile, env)
                for i, p in enumerate(plans):
                    for q in plans[:i]:
                        assert not _collides(plan_shapes(p, spec), RULES[p.kind].depth,
                                             plan_shapes(q, spec), RULES[q.kind].depth), (seed, env, p, q)


def test_every_kind_has_a_rule():
    from syncrack3d.structures import RULES

    assert set(RULES) == set(KINDS)


def test_windows_sit_at_sill_height_below_the_ceiling():
    from syncrack3d.structures import WINDOW_HEAD_GAP_M, WINDOW_SILL_M

    for seed in range(60):
        spec = sample_scene_spec(seed)
        for p in sample_structure_plan(seed, spec, include_kinds=["window"]):
            sill = p.v_m - p.height_m * .5
            assert WINDOW_SILL_M[0] - 1e-9 <= sill <= WINDOW_SILL_M[1] + 1e-9
            assert p.v_m + p.height_m * .5 <= spec.height_m - WINDOW_HEAD_GAP_M + 1e-9


def test_geometry_is_background():
    import bpy
    from cwg_infinigen.assets.defect_injection import classify_plane
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
    finally:
        clear_scene()


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
                elif p.kind == 'door':
                    assert p.v_m == p.height_m * .5
    assert set(INDOOR_KINDS) <= seen['indoor']
    assert set(OUTDOOR_KINDS) <= seen['outdoor']
    assert not set(OUTDOOR_KINDS) & seen['indoor']
    assert not set(INDOOR_KINDS) & seen['outdoor']


def _build_kinds(seed, kinds):
    from syncrack3d.build_dataset import clear_scene
    clear_scene()
    spec = sample_scene_spec(seed)
    plans = sample_structure_plan(seed, spec, include_kinds=kinds)
    return spec, plans, build_structures(plans)


@pytest.mark.parametrize("seed", (193001, 193002, 193003))
def test_door_leaf_sits_in_its_frame_and_not_in_the_wall(seed):
    import bpy
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.occlusion import _world_bounds
    random.seed(5)
    np.random.seed(5)
    before = random.getstate(), np.random.get_state()
    try:
        _, plans, records = _build_kinds(seed, ['door'])
        assert random.getstate() == before[0]  # models build under their own seeds
        assert np.array_equal(np.random.get_state()[1], before[1][1])
        door, rec = plans[0], records[0]
        assert rec['geometry_model'].endswith('DoorFactory')
        leaf = next(bpy.data.objects[n] for n in rec['objects'] if n.endswith('_leaf'))
        casing = next(bpy.data.objects[n] for n in rec['objects'] if n.endswith('_casing'))
        lo, hi = _world_bounds(leaf)
        assert hi[1] <= 1e-3  # nothing inside the wall
        assert lo[0] >= door.u_m - door.width_m * .5 and hi[0] <= door.u_m + door.width_m * .5
        assert lo[2] >= 0 and hi[2] <= door.height_m + 1e-3
        # the architrave fills the footprint, sits on the wall and stands proud of the leaf face
        c_lo, c_hi = _world_bounds(casing)
        assert c_lo[0] == pytest.approx(door.u_m - door.width_m * .5, abs=1e-4)
        assert c_hi[0] == pytest.approx(door.u_m + door.width_m * .5, abs=1e-4)
        assert c_hi[2] == pytest.approx(door.height_m, abs=1e-4) and abs(c_lo[2]) < 1e-6
        assert c_hi[1] == pytest.approx(0., abs=1e-6) and c_lo[1] <= -.024 + 1e-6
        # the leaf's outward wood faces (not its handle) are set back in the frame, clear of the wall
        m = np.array(leaf.matrix_world)
        front = [(m[:3, :3] @ np.array(f.center) + m[:3, 3])[1] for f in leaf.data.polygons
                 if 'metal' not in leaf.material_slots[f.material_index].material.name
                 and (m[:3, :3] @ np.array(f.normal))[1] < -.9]
        assert c_lo[1] + .01 < np.median(front) < -.002
    finally:
        clear_scene()


def test_socket_plate_sits_on_the_wall():
    import bpy
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.occlusion import _world_bounds
    try:
        for seed in range(193010, 193014):
            _, plans, records = _build_kinds(seed, ['socket'])
            p = plans[0]
            plates = [n for n in records[0]['objects'] if '_faceplate' in n]
            assert plates
            lo, hi = _world_bounds(bpy.data.objects[plates[0]])
            assert hi[1] <= 1e-6
            assert hi[0] - lo[0] <= p.width_m + 1e-3 and hi[2] - lo[2] <= p.height_m + 1e-3
            assert abs((lo[0] + hi[0]) / 2 - p.u_m) < 1e-3
            assert abs((lo[2] + hi[2]) / 2 - p.v_m) < 1e-3
            assert sum('_pin' in n for n in records[0]['objects']) == 6
            names = records[0]['objects']
            assert sum('_rocker' in n for n in names) == 2
            rockers = [bpy.data.objects[n] for n in names if '_rocker' in n]
            recess = next(bpy.data.objects[n] for n in names if '_switch_recess' in n)
            rlo = np.min([_world_bounds(o)[0] for o in rockers], 0)
            rhi = np.max([_world_bounds(o)[1] for o in rockers], 0)
            slo, shi = _world_bounds(recess)
            # The dark aperture stays a hairline around the rockers.
            assert shi[0] - rhi[0] < .0012 and rlo[0] - slo[0] < .0012
            assert shi[2] - rhi[2] < .0012 and rlo[2] - slo[2] < .0012
    finally:
        clear_scene()


def test_hatch_vent_and_socket_shells_vary():
    import bpy
    from syncrack3d.build_dataset import clear_scene
    try:
        part = {'service_hatch': '_frame', 'vent': '_frame', 'socket': '_faceplate'}
        seen = {kind: set() for kind in part}
        for seed in range(24):
            for kind, marker in part.items():
                _, _, records = _build_kinds(seed, [kind])
                piece = next(bpy.data.objects[n] for n in records[0]['objects'] if marker in n)
                seen[kind].add(piece.data.materials[0]['cwg_shell'])
        assert seen == {kind: {'enamel', 'dark', 'metal'} for kind in part}
    finally:
        clear_scene()


@pytest.mark.parametrize("kind", ('service_hatch', 'utility_box', 'vent'))
def test_sealant_beads_hug_the_fixture_outline(kind):
    import bpy
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.occlusion import _world_bounds
    from syncrack3d.structures import OUTLINE_PARTS
    try:
        sealed = 0
        for seed in range(193030, 193046):
            _, plans, records = _build_kinds(seed, [kind])
            rec = records[0]
            parts = [bpy.data.objects[n] for n in rec['objects']]
            beads = [o for o in parts if '_sealant' in o.name]
            assert bool(beads) == rec['sealant']
            if not beads:
                continue
            sealed += 1
            body = [_world_bounds(o) for o in parts if any(t in o.name for t in OUTLINE_PARTS[kind])]
            lo = np.min([b[0] for b in body], 0)
            hi = np.max([b[1] for b in body], 0)
            for b in beads:
                blo, bhi = _world_bounds(b)
                assert bhi[1] <= .0015 and blo[1] >= -.01  # pressed into the wall face, not standing off it
                assert np.all(blo[[0, 2]] >= lo[[0, 2]] - .01) and np.all(bhi[[0, 2]] <= hi[[0, 2]] + .01)
        assert 0 < sealed < 16
    finally:
        clear_scene()


def test_indoor_rooms_get_more_sockets():
    sockets = 0
    for seed in range(400):
        spec = sample_scene_spec(seed)
        plans = sample_structure_plan(seed, spec, profile='auto', environment='indoor')
        sockets = max(sockets, sum(p.kind == 'socket' for p in plans))
        assert not any(p.kind == 'switch' for p in plans)
    assert sockets >= 2


def test_fixtures_mount_on_the_wall_relief():
    import bpy
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.occlusion import _world_bounds
    try:
        clear_scene()
        spec = sample_scene_spec(193050)
        plans = sample_structure_plan(193050, spec, profile='all')
        relief = .014
        for rec in build_structures(plans, relief):
            for name in rec['objects']:
                b = _world_bounds(bpy.data.objects[name])
                if b is not None:  # sealant beads press 1-2 mm into the surface they seal
                    assert b[1][1] <= -relief + .002, name
    finally:
        clear_scene()


def test_route_ends_on_the_wall_edge_run_on_to_the_extent():
    from syncrack3d.structures import _reach_extent

    left, right, top, extent = -1.47, 1.47, 2.58, (-6.0, 1.5, 4.0)
    across = _reach_extent([(left, 1.2), (right, 1.2)], left, right, top, extent)
    assert across == [(-5.97, 1.2), (1.47, 1.2)]  # the right side is the corner: no further
    elbow = _reach_extent([(.3, top), (.3, 1.0), (left, 1.0)], left, right, top, extent)
    assert elbow == [(.3, 3.98), (.3, 1.0), (-5.97, 1.0)]  # inner corner unchanged
    up = [(.2, .02), (.2, top)]
    assert _reach_extent(up, left, right, top, None) == up


def test_outdoor_service_runs_reach_the_facade_edge():
    import bpy
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.occlusion import _world_bounds

    kinds = ["wall_cable", "conduit", "trunking", "plumbing", "utility_box", "downpipe"]
    try:
        for seed in (11, 12, 13):
            clear_scene()
            spec = sample_scene_spec(seed)
            extent = (-spec.width_m * .5 - 4.0, spec.width_m * .5 + 3.0, spec.height_m + 2.0)
            plans = sample_structure_plan(seed, spec, environment="outdoor", include_kinds=kinds)
            for rec in build_structures(plans, 0., spec.height_m, spec.width_m, extent=extent):
                if rec["kind"] not in kinds:
                    continue
                parts = [_world_bounds(bpy.data.objects[n]) for n in rec["objects"]]
                lo = np.min([b[0] for b in parts if b is not None], axis=0)
                hi = np.max([b[1] for b in parts if b is not None], axis=0)
                # Every run leaves the test wall: by a side or over the top, or
                # a utility box's conduit down into the ground.
                into_ground = rec["kind"] == "utility_box" and lo[2] < .01
                assert (lo[0] < extent[0] + .1 or hi[0] > extent[1] - .1
                        or hi[2] > extent[2] - .1 or into_ground), (seed, rec["kind"], lo, hi)
    finally:
        clear_scene()


def test_fixture_finishes_are_infinigen2():
    import bpy
    from syncrack3d.structures import _fixture_finish

    for role in ("enamel", "dark", "metal"):
        mat = _fixture_finish(f"probe_{role}", role, 5)
        assert mat.displacement_method == "BUMP"
        assert not any(n.bl_idname == "ShaderNodeGroup" and n.node_tree and "shader_" in n.node_tree.name
                       for n in mat.node_tree.nodes)
        bpy.data.materials.remove(mat)


def test_fixture_palette_is_per_scene_and_varies():
    from syncrack3d.structures import PALETTE, _Palette

    assert _Palette(5).record() == _Palette(5).record()
    seen = {role: {_Palette(s).record()[role] for s in range(200)} for role in PALETTE}
    for role, finishes in seen.items():
        if role != "sill":
            assert len(finishes) > 1, role  # not one fixed finish for every scene
    assert seen["sill"] == {"wall"}


def test_wall_sill_uses_the_wall_material():
    import bpy
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.geometry import build_scene
    from syncrack3d.structures import _Palette

    try:
        clear_scene()
        spec = sample_scene_spec(4)
        wall = build_scene(spec).hosts[0]
        plans = [p for p in sample_structure_plan(4, spec, include_kinds=["window"])]
        records = build_structures(plans, wall=wall, palette_seed=7)
        sill = next(bpy.data.objects[n] for n in records[0]["objects"] if n.endswith("_sill"))
        mat = sill.data.materials[0]
        assert mat.get("cwg_palette") == "wall"
        assert mat.node_tree.nodes.keys() == wall.data.materials[0].node_tree.nodes.keys()
        assert sill.data.uv_layers.active.name == "CWG_BoxUV"
    finally:
        clear_scene()


def test_routed_runs_do_not_pass_through_other_fixtures():
    import bpy
    from syncrack3d.build_dataset import clear_scene
    from syncrack3d.occlusion import _world_bounds
    from syncrack3d.structures import ROUTED_KINDS

    try:
        for seed in range(6):
            clear_scene()
            spec = sample_scene_spec(seed)
            plans = sample_structure_plan(seed, spec, "all", "indoor")
            recs = build_structures(plans, 0., spec.height_m, spec.width_m)
            solid = [(rec["index"], _world_bounds(bpy.data.objects[n])) for rec in recs
                     if rec["kind"] not in ROUTED_KINDS + ("panel_seam",)
                     for n in rec["objects"] if "sealant" not in n and bpy.data.objects[n].type == "MESH"]
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for rec in recs:
                if rec["kind"] not in ROUTED_KINDS:
                    continue
                for n in rec["objects"]:
                    ev = bpy.data.objects[n].evaluated_get(depsgraph)
                    mesh = ev.to_mesh()
                    pts = np.array([ev.matrix_world @ v.co for v in mesh.vertices])
                    ev.to_mesh_clear()
                    for index, (lo, hi) in solid:
                        inside = np.all((pts > lo + .002) & (pts < hi - .002), axis=1)
                        assert not inside.any(), (seed, n, index)
    finally:
        clear_scene()
