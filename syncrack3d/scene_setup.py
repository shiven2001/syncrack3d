"""Deterministic camera, key light, and Nishita world setup."""

from __future__ import annotations

import math
from dataclasses import dataclass

import bpy
from mathutils import Vector

from syncrack3d.spec import SceneSpec


CAPTURE_COLLECTION = "CWG_CAPTURE"


@dataclass(frozen=True)
class CaptureSetup:
    camera: bpy.types.Object
    light: bpy.types.Object
    world: bpy.types.World


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


def _point_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _make_camera(spec: SceneSpec, collection):
    target = (0.0, 0.0, spec.height_m * 0.5)
    # Leave enough margin to contain the full wall in a typical 16:9 frame.
    # Vertical framing is the limiting dimension for a 50 mm lens.
    distance = max(spec.width_m * 1.6, spec.height_m * 2.85, 4.0)
    bpy.ops.object.camera_add(location=(0.0, -distance, target[2]))
    camera = bpy.context.object
    camera.name = "CWG_Camera"
    camera.data.name = "CWG_Camera_Data"
    camera.data.lens = 50.0
    camera.data.sensor_width = 36.0
    camera.data.dof.use_dof = False
    camera["cwg_role"] = "front_wall_camera"
    _point_at(camera, target)
    _move_to_collection(camera, collection)
    bpy.context.scene.camera = camera
    return camera


def _make_area_light(spec: SceneSpec, collection):
    target = (0.0, 0.0, spec.height_m * 0.52)
    location = (
        -spec.width_m * 0.32,
        -max(2.0, spec.width_m * 0.55),
        spec.height_m * 1.15,
    )
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = "CWG_Key_Light"
    light.data.name = "CWG_Key_Light_Data"
    light.data.energy = max(550.0, spec.width_m * spec.height_m * 75.0)
    light.data.shape = "DISK"
    light.data.size = min(3.0, max(1.5, spec.width_m * 0.45))
    light["cwg_role"] = "key_light"
    _point_at(light, target)
    _move_to_collection(light, collection)
    return light


def _make_nishita_world():
    world = bpy.data.worlds.get("CWG_Nishita_World")
    if world is None:
        world = bpy.data.worlds.new("CWG_Nishita_World")
    world.use_nodes = True
    world["cwg_environment"] = "nishita"
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputWorld")
    output.name = "CWG_World_Output"
    background = nodes.new("ShaderNodeBackground")
    background.name = "CWG_Sky_Background"
    background.inputs["Strength"].default_value = 0.35
    sky = nodes.new("ShaderNodeTexSky")
    sky.name = "CWG_Nishita_Sky"
    sky.sky_type = "NISHITA"
    sky.sun_disc = True
    sky.sun_size = math.radians(0.545)
    sky.sun_intensity = 0.8
    sky.sun_elevation = math.radians(35.0)
    sky.sun_rotation = math.radians(135.0)
    sky.altitude = 0.2
    sky.air_density = 1.0
    sky.dust_density = 1.0
    sky.ozone_density = 1.0
    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    bpy.context.scene.world = world
    return world


def configure_capture_scene(spec: SceneSpec, root) -> CaptureSetup:
    """Add exactly one camera, one area light, and a Nishita Sky world."""
    collection = _collection(CAPTURE_COLLECTION)
    camera = _make_camera(spec, collection)
    light = _make_area_light(spec, collection)
    world = _make_nishita_world()
    root["cwg_camera_configured"] = True
    root["cwg_lighting_configured"] = True
    return CaptureSetup(camera=camera, light=light, world=world)
