"""Infinigen 2 bridge: wall leaves, floor/ceiling/terrain pickers, real displacement."""

from __future__ import annotations

import random
from pathlib import Path

import bpy
import pytest

from syncrack3d.infinigen2_bridge import INFINIGEN2_CATEGORIES, available, use_vendored_infinigen2
from syncrack3d.material_diversity import apply_wall_material
from syncrack3d.spec import sample_scene_spec


def _wall(seed: int = 0):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    from syncrack3d.geometry import build_scene

    return build_scene(sample_scene_spec(seed, module="flat", substrate="paint")).hosts[0]


def _material_output(mat):
    return next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 5)
    try:
        return tuple(round(float(item), 5) for item in value)
    except (TypeError, ValueError):
        return None


def _graph_signature(mat):
    bits = [len(mat.node_tree.nodes)]
    for node in mat.node_tree.nodes:
        for socket in node.inputs:
            if socket.is_linked:
                continue
            number = _number(getattr(socket, "default_value", None))
            if number is not None:
                bits.append((node.bl_idname, socket.name, number))
    return tuple(bits)


@pytest.mark.skipif(not available(), reason="vendored Infinigen 2 shaders missing")
@pytest.mark.parametrize("category", sorted(INFINIGEN2_CATEGORIES))
def test_procedural_apply_keeps_displacement(category):
    wall = _wall()
    generator, finish, record = apply_wall_material(wall, seed=7, forced=category)
    assert generator.startswith("infinigen2:")
    assert finish == INFINIGEN2_CATEGORIES[category]
    assert record["source"] == "infinigen2_procedural"
    mat = wall.data.materials[0]
    assert mat.displacement_method == "DISPLACEMENT"
    assert _material_output(mat).inputs["Displacement"].is_linked


@pytest.mark.skipif(not available(), reason="vendored Infinigen 2 shaders missing")
def test_each_seed_builds_its_own_graph():
    use_vendored_infinigen2()
    import infinigen2

    assert "syncrack3d/vendor" in infinigen2.__file__.replace("\\", "/")
    from syncrack3d.infinigen2_bridge import realize_material

    first = realize_material("paint_wall", 1)
    again = realize_material("paint_wall", 1)
    other = realize_material("paint_wall", 2)
    assert _graph_signature(first) == _graph_signature(again)
    assert _graph_signature(first) != _graph_signature(other)


@pytest.mark.skipif(not available(), reason="infinigen==2.0.0a3 not installed")
@pytest.mark.parametrize("category", sorted(INFINIGEN2_CATEGORIES))
def test_realize_keeps_displacement(category):
    from syncrack3d.infinigen2_bridge import realize_material

    mat = realize_material(category, seed=3)
    out = _material_output(mat)
    assert out.inputs["Surface"].is_linked
    assert out.inputs["Displacement"].is_linked
    assert mat.displacement_method == "DISPLACEMENT"


@pytest.mark.skipif(not available(), reason="vendored Infinigen 2 shaders missing")
@pytest.mark.parametrize("category", sorted(INFINIGEN2_CATEGORIES))
def test_inject_keeps_host_displacement(category, tmp_path: Path):
    """Crack injection must layer onto I2 displacement, not strip it."""
    import numpy as np

    from syncrack3d.defects import build_defects, inject_defects, sample_defect_plan
    from syncrack3d.materials import enable_true_displacement

    seed = 17
    wall = _wall(seed)
    spec = sample_scene_spec(seed, module="flat", substrate="paint")
    wall["syncrack_displacement_sign"] = 1
    apply_wall_material(wall, seed, forced=category)
    assert _material_output(wall.data.materials[0]).inputs["Displacement"].is_linked

    rng = random.Random(seed)
    np.random.seed(seed % (2**32))
    plan = sample_defect_plan(rng, total_range=(1, 1), size_range_m=(0.18, 0.28))
    placements = build_defects(spec, wall, rng, plan)
    assert placements, "expected one crack card"
    injected = inject_defects(
        placements,
        wall,
        manifest_path=tmp_path / f"{category}_defects.json",
        scene_id=f"i2_{category}",
        scene_seed=seed,
        dicing_rate=2.0,
    )
    assert injected >= 1
    enable_true_displacement(wall, 2.0)

    mat = wall.data.materials[0]
    out = _material_output(mat)
    assert out.inputs["Surface"].is_linked
    assert out.inputs["Displacement"].is_linked
    assert mat.displacement_method == "DISPLACEMENT"
    assert any(n.bl_idname == "ShaderNodeOutputAOV" for n in mat.node_tree.nodes)


@pytest.mark.skipif(not available(), reason="vendored Infinigen 2 shaders missing")
def test_ceiling_picker_displaces():
    """The ceiling is Infinigen's ceiling picker, displaced as geometry, with metre UVs in x,y."""
    from syncrack3d.room_shell import ShellPlan, apply_ceiling_finish, build_room_shell

    bpy.ops.wm.read_factory_settings(use_empty=True)
    spec = sample_scene_spec(3, module="flat", substrate="paint")
    grey = (0.7, 0.7, 0.7, 1.0)
    built = build_room_shell(spec, ShellPlan(3.0, True, (), grey, grey, grey))
    record = apply_ceiling_finish(built, seed=3, dicing_rate=0.75)
    ceiling = next(obj for obj in built if obj.name == "CWG_Shell_Ceiling")
    assert record["picker"] == "ceiling_material_rand"
    assert record["generator"] == "infinigen2:ceiling"
    assert record["displacement"] == "DISPLACEMENT"
    _assert_real_displacement(ceiling, 0.75, "ceiling_adaptive")

    mesh = ceiling.data
    uv = mesh.uv_layers.active
    downward = next(poly for poly in mesh.polygons if poly.normal.z < -0.9)
    for loop_index in downward.loop_indices:
        co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        assert uv.data[loop_index].uv[0] == pytest.approx(co.x, abs=1e-5)
        assert uv.data[loop_index].uv[1] == pytest.approx(co.y, abs=1e-5)

    open_shell = build_room_shell(spec, ShellPlan(3.0, False, (), grey, grey, grey))
    assert apply_ceiling_finish(open_shell, seed=3) is None


@pytest.mark.skipif(not available(), reason="vendored Infinigen 2 shaders missing")
def test_matched_side_walls_displace():
    from syncrack3d.room_shell import ShellPlan, apply_matched_finish, build_room_shell

    for seed in (1, 12):  # side walls always copy the test wall
        wall = _wall(seed)
        apply_wall_material(wall, seed)
        spec = sample_scene_spec(seed, module="flat", substrate="paint")
        grey = (0.5, 0.5, 0.5, 1.0)
        built = build_room_shell(spec, ShellPlan(3.0, True, ("left", "right"), grey, grey, grey))
        record = apply_matched_finish(built, wall, seed, dicing_rate=0.75)
        assert record["returns"]["source"] == "test_wall"
        assert record["returns"]["category"] == wall["cwg_material_category"]
        assert record["returns"]["displacement"] == "DISPLACEMENT"
        assert record["floor"]["generator"] == "infinigen2:floor"
        assert record["ceiling"]["generator"] == "infinigen2:ceiling"
        for obj in built:
            if obj.name.startswith("CWG_Shell_Return_"):
                _assert_real_displacement(obj, 0.75, "return_adaptive")


@pytest.mark.skipif(not available(), reason="vendored Infinigen 2 shaders missing")
def test_no_crack_shaders_in_any_role():
    import importlib

    from syncrack3d import infinigen2_bridge as bridge

    use_vendored_infinigen2()
    for module, function in bridge._TERRAIN_LEAVES:  # every leaf resolves
        getattr(importlib.import_module(f"infinigen2.shaders.base_materials.{module}"), function)
    assert not {"cracked_ground", "stone", "mud", "ice"} & {module for module, _ in bridge._TERRAIN_LEAVES}
    functions = {spec[2] for spec in bridge._WALL_SPECS.values()}
    assert not functions & {"paint_flaked_rand", "wall_flaked_tile_rand", "ground_cracked_rand"}
    assert bridge._WALL_SPECS["bricks_paint"][3] == {"mask": 1.0}  # whole coat, no crack mask


def _displacement_nodes(tree):
    found = []
    for node in tree.nodes:
        if node.bl_idname == "ShaderNodeDisplacement":
            found.append(node)
        elif node.bl_idname == "ShaderNodeGroup" and node.node_tree is not None:
            found.extend(_displacement_nodes(node.node_tree))
    return found


def _assert_real_displacement(obj, dicing_rate, modifier_name):
    mat = obj.data.materials[0]
    out = _material_output(mat)
    assert mat.displacement_method == "DISPLACEMENT"
    assert out.inputs["Surface"].is_linked
    assert out.inputs["Displacement"].is_linked
    for node in _displacement_nodes(mat.node_tree):
        scale = node.inputs["Scale"]
        if not scale.is_linked:
            assert scale.default_value != pytest.approx(0.0)
    modifier = obj.modifiers[modifier_name]
    assert modifier.type == "SUBSURF"
    assert modifier.subdivision_type == "SIMPLE"
    assert obj.cycles.use_adaptive_subdivision
    assert obj.cycles.dicing_rate == pytest.approx(dicing_rate)


@pytest.mark.skipif(not available(), reason="vendored Infinigen 2 shaders missing")
def test_floor_picker_displaces():
    from syncrack3d.room_shell import ShellPlan, apply_floor_finish, build_room_shell

    bpy.ops.wm.read_factory_settings(use_empty=True)
    spec = sample_scene_spec(5, module="flat", substrate="paint")
    grey = (0.4, 0.4, 0.4, 1.0)
    built = build_room_shell(
        spec,
        ShellPlan(3.0, False, (), grey, grey, grey),
    )
    floor = next(obj for obj in built if obj.name == "CWG_Shell_Floor")
    record = apply_floor_finish(floor, seed=5, dicing_rate=0.75)
    assert record["picker"] == "floor_material_rand"
    assert record["generator"] == "infinigen2:floor"
    assert record["displacement"] == "DISPLACEMENT"
    _assert_real_displacement(floor, 0.75, "floor_adaptive")
    mesh = floor.data
    uv = mesh.uv_layers.active
    upward = next(poly for poly in mesh.polygons if poly.normal.z > 0.9)
    for loop_index in upward.loop_indices:
        co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        assert uv.data[loop_index].uv[0] == pytest.approx(co.x, abs=1e-5)
        assert uv.data[loop_index].uv[1] == pytest.approx(co.y, abs=1e-5)


@pytest.mark.skipif(not available(), reason="vendored Infinigen 2 shaders missing")
def test_terrain_picker_displaces():
    from syncrack3d.outdoor import build_outdoor, sample_outdoor_plan

    bpy.ops.wm.read_factory_settings(use_empty=True)
    spec = sample_scene_spec(8, module="flat", substrate="paint")
    plan = sample_outdoor_plan(8, spec.width_m)
    assert plan.ground == "terrain"
    built = build_outdoor(spec, plan, seed=8, dicing_rate=0.75)
    ground = next(obj for obj in built if obj.name == "CWG_Outdoor_Ground")
    assert ground["cwg_material_generator"] == "infinigen2:terrain"
    _assert_real_displacement(ground, 0.75, "ground_adaptive")
    names = {obj.name for obj in bpy.data.objects}
    assert "CWG_Outdoor_Apron" not in names
    assert not any(name.startswith("CWG_Outdoor_Lawn") for name in names)
    mesh = ground.data
    uv = mesh.uv_layers.active
    upward = next(poly for poly in mesh.polygons if poly.normal.z > 0.9)
    for loop_index in upward.loop_indices:
        co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        assert uv.data[loop_index].uv[0] == pytest.approx(co.x, abs=1e-5)
        assert uv.data[loop_index].uv[1] == pytest.approx(co.y, abs=1e-5)


def test_prop_roles_from_v1_names():
    from types import SimpleNamespace

    from syncrack3d.occlusion import _prop_role

    expect = {
        "shader_coarse_knit_fabric": "upholstery", "shader_leather": "upholstery",
        "shader_shelves_black_wood_z": "cabinet", "shader_shelves_white_metallic": "decorative",
        "sofa_leg_wood": "furniture", "shader_marble": "table_top", "VanityTop": "table_top",
        "BathroomChrome": "steel", "FridgeSteel": "steel", "FridgeHandle": "steel", "shader_glass": "glass",
        "shader_colored_glass": "glass", "shader_lampshade": "fabric", "shader_rough_plastic": "plastic",
        "shader_text": None, "FridgeBrand": None, "shader_dirt": None, "shader_monocot": None,
    }
    for name, role in expect.items():
        assert _prop_role(SimpleNamespace(name=name)) == role, name


@pytest.mark.skipif(not available(), reason="vendored Infinigen 2 shaders missing")
@pytest.mark.parametrize("factory", ["SofaFactory", "SingleCabinetFactory", "ToiletFactory", "ChairFactory"])
def test_props_get_v2_materials(factory):
    from syncrack3d import occlusion as oc

    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj, _ = oc._spawn(oc._CATALOG[factory], random.Random(2), 0, oc._collection("P"))
    assert obj is not None
    for part in (obj, *obj.children_recursive):
        if part.type != "MESH":
            continue
        assert part.material_slots
        for slot in part.material_slots:
            mat = slot.material
            assert mat.get("cwg_prop_role"), (part.name, mat.name)
            assert mat.displacement_method == "BUMP"
            assert "paint_cracked" not in mat.name
        assert part.data.uv_layers.active.name == "CWG_BoxUV"


def test_relief_bake_reads_outward_height():
    from syncrack3d.relief import outward_relief

    mat = bpy.data.materials.new("relief_probe")
    mat.use_nodes = True
    tree = mat.node_tree
    out = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    disp = tree.nodes.new("ShaderNodeDisplacement")
    disp.inputs["Height"].default_value = 0.012
    disp.inputs["Midlevel"].default_value = 0.0
    tree.links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
    scenes, materials = len(bpy.data.scenes), len(bpy.data.materials)
    assert outward_relief(mat, res=64) == pytest.approx(0.012, abs=2e-4)
    assert (len(bpy.data.scenes), len(bpy.data.materials)) == (scenes, materials)
