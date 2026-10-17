"""A boolean-free, uninterrupted controlled wall surface."""

from __future__ import annotations

from dataclasses import dataclass

import bpy

from syncrack3d.materials import (
    make_solid_material,
    make_substrate_material,
)
from syncrack3d.spec import GENERATOR_VERSION, SceneSpec


ASSET_COLLECTION = "CWG_HOST_SURFACES"
CONTEXT_COLLECTION = "CWG_CONTEXT"
DEFECT_COLLECTION = "CWG_DEFECT_TOKENS"


@dataclass(frozen=True)
class BuildResult:
    root: bpy.types.Object
    hosts: tuple[bpy.types.Object, ...]
    context: tuple[bpy.types.Object, ...]


def _collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def _move_to_collection(obj, collection):
    for old in tuple(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)


def _box(name, dimensions, location, material, collection, parent):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.name = f"{name}.mesh"
    obj.data.materials.append(material)
    obj.parent = parent
    _move_to_collection(obj, collection)
    return obj


def _make_wall(spec, host_collection, root, material):
    module_id = f"cwg_{spec.seed:08d}_00_main"
    wall = _box(
        f"{module_id}.wall",
        (spec.width_m, spec.thickness_m, spec.height_m),
        (0.0, spec.thickness_m * 0.5, spec.height_m * 0.5),
        material,
        host_collection,
        root,
    )
    wall["cwg_host"] = True
    wall["cwg_module_id"] = module_id
    wall["cwg_panel"] = "main"
    wall["cwg_generator_version"] = GENERATOR_VERSION
    wall["syndefect_surface"] = "wall"
    wall["surface_finish"] = spec.material.finish
    wall["cwg_injection_ready"] = True
    return wall


def _make_optional_skirting(spec, context_collection, root):
    if not spec.skirting:
        return ()
    material = make_solid_material(
        "CWG_Context_Skirting",
        (0.58, 0.59, 0.57, 1.0),
        roughness=0.48,
    )
    skirting = _box(
        "CWG_Skirting",
        (spec.width_m, 0.022, 0.085),
        (0.0, -0.011, 0.0425),
        material,
        context_collection,
        root,
    )
    skirting["cwg_hard_negative"] = "skirting"
    return (skirting,)


def build_scene(spec: SceneSpec) -> BuildResult:
    """Build one simple wall and its material; never create defects or cameras."""
    if spec.module != "flat":
        raise ValueError(f"Unsupported module: {spec.module!r}; only 'flat' is supported")

    host_collection = _collection(ASSET_COLLECTION)
    context_collection = _collection(CONTEXT_COLLECTION)
    _collection(DEFECT_COLLECTION)  # Intentionally empty integration boundary.

    root = bpy.data.objects.new("CWG_ROOT", None)
    root.empty_display_type = "CUBE"
    root.empty_display_size = 0.18
    root["cwg_generator_version"] = GENERATOR_VERSION
    root["cwg_seed"] = spec.seed
    root["cwg_module"] = spec.module
    root["cwg_defects_generated"] = False
    root["cwg_camera_configured"] = False
    root["cwg_lighting_configured"] = False
    root["cwg_render_configured"] = False
    host_collection.objects.link(root)

    material = make_substrate_material(spec.material, 0)
    wall = _make_wall(spec, host_collection, root, material)
    context = _make_optional_skirting(spec, context_collection, root)
    return BuildResult(root=root, hosts=(wall,), context=context)
