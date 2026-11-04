"""Render a SynCrack3D false-positive probe set with cast-shadow attribution passes.

Runs syncrack3d/build_dataset.py unchanged except that each view also gets two
raw (no compositor, linear EXR) renders just before the dataset render, after
any per-view relight:

  <out>/shadow_probe/seed_<s>/<i>_shadow.exr     all objects cast shadows
  <out>/shadow_probe/seed_<s>/<i>_noshadow.exr   props, downpipe, eave and corner cast no shadow

Both are multilayer EXR with Combined, DiffDir and IndexOB. Object pass index:
1 = test wall, 2 = coplanar facade continuation, 3 = prop or downpipe,
4 = outdoor eave or corner, 0 = everything else (room shell, ground, sky).
The room shell keeps its shadows in both: switching them off opens the room to
the world light and darkens nothing locally, it brightens the whole frame.
Hard light edges from the shell (window patch, bulb terminator) are found from
DiffDir instead.
The dataset frames, labels and RNG streams are exactly those of build_dataset.

Usage (from the syncrack3d app root, Blender-as-python env):
  python /path/studies/fp_probe_render.py <build_dataset args>
"""

import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

import bpy  # noqa: E402

from syncrack3d import build_dataset as bd  # noqa: E402

PROBE_SAMPLES = 32
_original = bd.render_cameras


def _index(obj):
    top = obj
    while top.parent is not None:
        top = top.parent
    if obj.get("syncrack3d_displacement_sign") is not None:
        return 1
    if obj.name.startswith("CWG_Outdoor_Facade_"):
        return 2
    if top.get("cwg_role") == "prop" or top.name.startswith("CWG_Prop_"):
        return 3
    if obj.name in ("CWG_Outdoor_Eave", "CWG_Outdoor_Corner"):
        return 4
    return 0


def _raw_render(path: Path):
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def probe_render_cameras(cameras, out_dir, *, before_each=None, **kw):
    out_dir = Path(out_dir)
    probe_dir = out_dir.parent.parent / "shadow_probe" / out_dir.name
    probe_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    meshes = [o for o in scene.objects if o.type in ("MESH", "CURVE", "SURFACE", "META")]
    walls = [o for o in meshes if _index(o) == 1]
    if len(walls) != 1:
        raise RuntimeError(f"expected one test wall, found {[o.name for o in walls]}")
    for o in meshes:
        o.pass_index = _index(o)
    casters = [o for o in meshes if _index(o) in (3, 4)]

    def probe(index, cam):
        if before_each is not None:
            before_each(index, cam)
        saved = (scene.use_nodes, scene.cycles.samples, scene.render.image_settings.file_format,
                 scene.render.image_settings.color_depth, view_layer.use_pass_object_index,
                 view_layer.use_pass_diffuse_direct)
        scene.use_nodes = False
        scene.cycles.samples = PROBE_SAMPLES
        view_layer.use_pass_object_index = True
        view_layer.use_pass_diffuse_direct = True
        scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
        scene.render.image_settings.color_depth = "32"
        try:
            _raw_render(probe_dir / f"{index}_shadow.exr")
            for o in casters:
                o.visible_shadow = False
            _raw_render(probe_dir / f"{index}_noshadow.exr")
        finally:
            for o in casters:
                o.visible_shadow = True
            (scene.use_nodes, scene.cycles.samples, scene.render.image_settings.file_format) = saved[:3]
            scene.render.image_settings.color_depth = "8"
            scene.render.image_settings.color_mode = "RGB"
            view_layer.use_pass_object_index, view_layer.use_pass_diffuse_direct = saved[4:]

    return _original(cameras, out_dir, before_each=probe, **kw)


bd.render_cameras = probe_render_cameras

if __name__ == "__main__":
    sys.exit(bd.main())
