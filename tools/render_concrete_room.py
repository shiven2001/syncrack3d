"""One indoor room whose floor is the concrete generator.

The inspection wall is neutral plaster, both side walls are present, and the
ceiling is the plain off-white. The floor is forced to Infinigen ``Concrete``
rather than drawn from the floor catalogue. Writes
``outputs/concrete_room.png``.

    PYTHONPATH=. /home/shiven/anaconda3/envs/infinigen_v2/bin/python tools/render_concrete_room.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from syncrack3d.geometry import build_scene  # noqa: E402
from syncrack3d.lighting import LightSpec, LightingPlan, apply_lighting  # noqa: E402
from syncrack3d.material_diversity import apply_diverse_wall_material  # noqa: E402
from syncrack3d.materials import enable_true_displacement  # noqa: E402
from syncrack3d.render import configure_cycles  # noqa: E402
from syncrack3d.room_shell import ShellPlan, _metric_uv_xy, apply_matched_finish, build_room_shell  # noqa: E402
from syncrack3d.scene_setup import configure_capture_scene  # noqa: E402
from syncrack3d.spec import sample_scene_spec  # noqa: E402

OUT = REPO / "outputs" / "concrete_room.png"
SEED = 4


def _concrete_floor(floor) -> None:
    from cwg_infinigen.assets.materials.ceramic.concrete import Concrete
    from cwg_infinigen.core import surface
    from cwg_infinigen.core.util.math import int_hash

    np.random.seed(int_hash((SEED, "concrete-room-floor")) % (2**32))
    material = Concrete().generate()
    material.displacement_method = "BUMP"
    floor.data.materials.clear()
    surface.assign_material(floor, material)
    _metric_uv_xy(floor)


def _camera() -> None:
    aim = bpy.data.objects.new("CWG_Room_Aim", None)
    aim.location = (0.0, -0.15, 0.72)
    bpy.context.scene.collection.objects.link(aim)
    cam_data = bpy.data.cameras.new("CWG_Room_Cam")
    cam_data.lens = 24.0
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.05
    cam_data.clip_end = 40.0
    cam = bpy.data.objects.new("CWG_Room_Cam", cam_data)
    cam.location = (0.35, -3.05, 1.45)
    track = cam.constraints.new("TRACK_TO")
    track.target = aim
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam


def _render(path: Path) -> None:
    from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
    from cwg_infinigen.core.rendering.sensor_response import apply_sensor_response

    scene = bpy.context.scene
    scene.render.resolution_x = 1440
    scene.render.resolution_y = 1080
    scene.view_settings.exposure = 0.0
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    rgb = apply_sensor_response(
        NodeWrangler(tree), layers.outputs["Image"], enabled=True,
        auto_exposure=True, ev_compensation=0.1, noise=False,
    )
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = str(path.parent)
    out.format.file_format = "PNG"
    out.format.color_mode = "RGB"
    out.file_slots[0].path = path.stem
    tree.links.new(rgb, out.inputs[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in path.parent.glob(f"{path.stem}*.png"):
        old.unlink()
    bpy.ops.render.render(write_still=False)
    written = sorted(path.parent.glob(f"{path.stem}*.png"), key=lambda item: item.stat().st_mtime)
    if not written:
        raise RuntimeError(f"Cycles wrote no PNG for {path.stem}")
    written[-1].replace(path)


def main() -> None:
    import cwg_infinigen.core.constraints.example_solver.room.decorate  # noqa: F401

    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_cycles(resolution=1080, samples=128, denoise=True, dicing_rate=2.0)
    spec = sample_scene_spec(SEED, width_m=4.2, height_m=2.7, thickness_m=0.16, skirting=False)
    built = build_scene(spec)
    wall = built.hosts[0]
    apply_diverse_wall_material(wall, SEED, forced="paint_neutral", palette="unified")
    enable_true_displacement(wall, 2.0)
    shell = build_room_shell(spec, ShellPlan(
        depth_m=4.4,
        has_ceiling=True,
        returns=("left", "right"),
        floor_color=(0.42, 0.40, 0.38, 1.0),
        ceiling_color=(0.91, 0.90, 0.88, 1.0),
        return_color=(0.74, 0.74, 0.72, 1.0),
    ), SEED)
    apply_matched_finish(shell, wall, SEED)
    floor = next(obj for obj in shell if obj.name == "CWG_Shell_Floor")
    _concrete_floor(floor)
    configure_capture_scene(spec, built.root)
    apply_lighting(LightingPlan(
        regime="ceiling_diffuse",
        world_strength=0.35,
        lights=(
            LightSpec("key", "ceiling_diffuse", 4300.0, 90.0, 2.2, 28.0, 15.0,
                      (0.6, -1.6, 2.35), target=(0.0, -0.6, 0.4)),
            LightSpec("fill", "window_soft", 5800.0, 35.0, 1.6, 42.0, 210.0,
                      (-1.8, -2.4, 1.7), target=(0.3, -0.4, 0.5)),
        ),
    ), spec.height_m, bpy.context.scene.collection)
    _camera()
    _render(OUT)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
