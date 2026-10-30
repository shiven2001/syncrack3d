"""Manifest and structural validation for generated base scenes."""

from __future__ import annotations

import json
from pathlib import Path

import bpy

from syncrack3d.geometry import BuildResult
from syncrack3d.scene_setup import CaptureSetup
from syncrack3d.spec import GENERATOR_VERSION, SceneSpec


def _vector(values):
    return [round(float(v), 6) for v in values]


def validate_base_scene(build: BuildResult, capture: CaptureSetup) -> dict:
    errors = []
    cameras = [obj.name for obj in bpy.data.objects if obj.type == "CAMERA"]
    lights = [obj.name for obj in bpy.data.objects if obj.type == "LIGHT"]
    if cameras != [capture.camera.name]:
        errors.append(f"expected exactly the configured camera, got: {cameras}")
    if lights != [capture.light.name]:
        errors.append(f"expected exactly the configured light, got: {lights}")
    if bpy.context.scene.camera != capture.camera:
        errors.append("configured camera is not the active scene camera")
    if bpy.context.scene.world != capture.world:
        errors.append("configured Nishita world is not active")
    sky_nodes = [
        node
        for node in capture.world.node_tree.nodes
        if node.bl_idname == "ShaderNodeTexSky" and node.sky_type == "NISHITA"
    ]
    if len(sky_nodes) != 1:
        errors.append("world must contain exactly one Nishita Sky Texture")
    for host in build.hosts:
        if not host.name.endswith(".wall"):
            errors.append(f"host lacks .wall suffix: {host.name}")
        if not host.get("cwg_injection_ready"):
            errors.append(f"host is not marked injection-ready: {host.name}")
        if host.modifiers:
            errors.append(f"host unexpectedly has modifiers: {host.name}")
        if len(host.data.materials) != 1 or host.data.materials[0] is None:
            errors.append(f"host must have exactly one material: {host.name}")
            continue
        mat = host.data.materials[0]
        bsdf = next(
            (n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"),
            None,
        )
        if bsdf is None or not bsdf.inputs["Base Color"].is_linked:
            errors.append(f"host material lacks linked Principled Base Color: {host.name}")
    if errors:
        raise RuntimeError("Controlled wall validation failed: " + "; ".join(errors))
    return {
        "valid": True,
        "host_count": len(build.hosts),
        "context_count": len(build.context),
        "camera_count": 1,
        "light_count": 1,
        "world": capture.world.name,
        "sky_type": "NISHITA",
    }


def write_scene_manifest(
    path: str | Path,
    spec: SceneSpec,
    build: BuildResult,
    capture: CaptureSetup,
    validation,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hosts = []
    for host in build.hosts:
        mat = host.data.materials[0]
        hosts.append(
            {
                "name": host.name,
                "surface": host.get("syndefect_surface"),
                "finish": host.get("surface_finish"),
                "module_id": host.get("cwg_module_id"),
                "panel": host.get("cwg_panel"),
                "dimensions_m": _vector(host.dimensions),
                "location_m": _vector(host.location),
                "material": mat.name,
                "injection_ready": bool(host.get("cwg_injection_ready")),
            }
        )
    payload = {
        "schema": "syndefect3d/controlled_wall_scene/1",
        "generator_version": GENERATOR_VERSION,
        "spec": spec.to_dict(),
        "hosts": hosts,
        "context_objects": [obj.name for obj in build.context],
        "capture_setup": {
            "camera": capture.camera.name,
            "camera_location_m": _vector(capture.camera.location),
            "camera_lens_mm": float(capture.camera.data.lens),
            "light": capture.light.name,
            "light_type": capture.light.data.type,
            "light_location_m": _vector(capture.light.location),
            "light_energy_w": float(capture.light.data.energy),
            "world": capture.world.name,
            "sky_type": "NISHITA",
        },
        "validation": validation,
        "excluded_subsystems": {
            "defect_generation": "not run; CWG_DEFECT_TOKENS is intentionally empty",
            "render_settings": "not configured",
            "compositor": "not configured",
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
