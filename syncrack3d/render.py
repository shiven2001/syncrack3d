"""Render controlled-wall scenes into the layout the defect exporter reads.

Output layout matches what ``tools/prepare_defect_annotated_dataset.py`` walks
for the apartment set, so one exporter serves both and no dataset code has to
learn a second convention:

    <out>/Image/camera_0/Image_<stem>.png
    <out>/DefectCrack/camera_0/DefectCrack_<stem>.png
    <out>/DefectPaintPeel/...  DefectPaintBubble/...
    <out>/DefectInstance*/camera_0/DefectInstance*_<stem>.exr

The per-class AOVs are written as PNG and the instance triplet as EXR because
that is what the exporter's ``find_defect_aov_maps`` and
``find_instance_aov_maps`` look for - coverage is a display-range quantity, the
instance moments are not.
"""

from __future__ import annotations

import logging
from pathlib import Path

import bpy

logger = logging.getLogger(__name__)

_INSTANCE_AOVS = ("DefectInstance", "DefectInstanceCoverage", "DefectInstanceSquared")


def _enable_gpu(preferred=("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")) -> str | None:
    """Put Cycles on the GPU if there is one.

    Blender defaults `compute_device_type` to NONE, which silently renders on
    the CPU no matter what hardware is present. That is not a small difference
    here: measured at 640x640, CPU rendering cost 22 s/frame on a machine with
    an idle RTX 4090. A dataset run is thousands of frames, so leaving this
    unset is the difference between hours and days.
    """
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        return None
    prefs = prefs.preferences
    # get_devices() populates the list. Reading prefs.devices before calling it
    # returns an empty collection, which looks exactly like "this machine has no
    # GPU" - that mistake sent a 4090 machine back to the CPU silently.
    prefs.get_devices()
    available = {d.type for d in prefs.devices}
    for backend in preferred:
        if backend not in available:
            continue
        prefs.compute_device_type = backend
        prefs.get_devices()
        for device in prefs.devices:
            # Enable the accelerator only. Mixing in the CPU can be slower than
            # the GPU alone once tile sync is counted.
            device.use = device.type == backend
        bpy.context.scene.cycles.device = "GPU"
        logger.info("cycles device: %s", backend)
        return backend
    logger.warning("no GPU backend available; rendering on CPU")
    return None


def configure_cycles(
    *,
    resolution: int = 1280,
    samples: int = 256,
    denoise: bool = True,
    dicing_rate: float = 2.0,
    view_transform: str = "Standard",
):
    """Cycles settings matched to the apartment renders.

    Square at 1280 for the same reason the indoor configs are: the real set is
    square phone photos, and 2x the 640 training input keeps a hairline
    antialiased rather than aliased. EXPERIMENTAL is required - adaptive
    subdivision is what turns the injected displacement into real geometry, and
    without it every defect renders flat while its labels stay perfect.
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    _enable_gpu()
    scene.cycles.feature_set = "EXPERIMENTAL"
    scene.cycles.dicing_rate = dicing_rate
    scene.cycles.samples = samples
    scene.cycles.use_denoising = denoise
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    try:
        scene.view_settings.view_transform = view_transform
    except TypeError:
        logger.warning("view transform %s unavailable; leaving default", view_transform)
    return scene


def setup_passes(view_layer=None) -> list[str]:
    """Enable the defect AOVs on the view layer."""
    from cwg_infinigen.core.rendering.defect_aov import setup_defect_aovs

    return setup_defect_aovs(view_layer=view_layer, enabled=True)


def _file_output(tree, slot_name: str, base_dir: Path, stem: str, fmt: str):
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.base_path = str(base_dir / slot_name / "camera_0")
    node.format.file_format = fmt
    if fmt == "OPEN_EXR":
        node.format.color_depth = "32"
        node.format.color_mode = "RGB"
    node.file_slots.clear()
    node.file_slots.new(f"{slot_name}_{stem}")
    return node


def build_compositor(out_dir: Path, stem: str, aov_names: list[str], exposure_ev=None, awb_strength=1.0):
    """Wire Render Layers -> one File Output per pass.

    Built per frame rather than once, because the file-slot name carries the
    frame stem and the exporter pairs passes by that stem.
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    layers = tree.nodes.new("CompositorNodeRLayers")

    # White balance the RGB, but never the label passes. The apartment renders
    # get this through compositor_postprocessing; this path does not call that,
    # so without it these frames would carry the same fixed orange cast that a
    # neutral wall picked up there - measured at R/B 1.93 against a palette
    # whose own R/B is 1.01-1.07. AOVs are coverage and identity fields, not
    # radiance, so they are wired straight from the render layer.
    from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
    from cwg_infinigen.core.rendering.sensor_response import apply_auto_white_balance

    from cwg_infinigen.core.rendering.sensor_response import apply_sensor_response

    nw = NodeWrangler(tree)
    rgb = apply_auto_white_balance(nw, layers.outputs["Image"], strength=awb_strength)
    # Auto-exposure, shot noise and the full-well clip. Without metering the
    # frame's brightness is whatever the sampled light power happened to be, and
    # the lighting sweep this generator exists for would be a brightness sweep
    # instead of a direction sweep. A phone meters; so does this.
    exposure_kwargs = {} if exposure_ev is None else {"ev_compensation": exposure_ev}
    rgb = apply_sensor_response(nw, rgb, enabled=True, auto_exposure=True, **exposure_kwargs)
    image_out = _file_output(tree, "Image", out_dir, stem, "PNG")
    tree.links.new(rgb, image_out.inputs[0])

    for name in aov_names:
        if name not in layers.outputs:
            logger.warning("AOV %s not present on the render layer; skipping", name)
            continue
        fmt = "OPEN_EXR" if name in _INSTANCE_AOVS else "PNG"
        node = _file_output(tree, name, out_dir, stem, fmt)
        tree.links.new(layers.outputs[name], node.inputs[0])
    return tree


def render_cameras(
    cameras: list[str],
    out_dir: Path,
    *,
    aov_names: list[str] | None = None,
    frame: int = 1,
    exposure_evs: list[float] | None = None,
    awb_strengths: list[float] | None = None,
    before_each=None,
) -> list[str]:
    """Render one frame per named camera. Returns the stems written.

    ``before_each(index, camera)`` runs just before each render, e.g. to relight.
    """
    if exposure_evs is not None and len(exposure_evs) != len(cameras):
        raise ValueError("One exposure value is required per camera")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_set(frame)
    aov_names = aov_names if aov_names is not None else setup_passes()

    stems = []
    for index, cam_name in enumerate(cameras):
        cam = bpy.data.objects.get(cam_name)
        if cam is None or cam.type != "CAMERA":
            logger.warning("camera %s missing; skipping", cam_name)
            continue
        scene.camera = cam
        if before_each is not None:
            before_each(index, cam)
        stem = f"{index}_0_{frame:04d}_0"
        exposure_ev = None if exposure_evs is None else exposure_evs[index]
        awb = 1.0 if awb_strengths is None else awb_strengths[index]
        build_compositor(out_dir, stem, aov_names, exposure_ev=exposure_ev, awb_strength=awb)
        bpy.ops.render.render(write_still=False)
        _canonicalise_names(out_dir, stem)
        _write_index_pair(out_dir, stem, scene)
        stems.append(stem)
        logger.info("rendered %s -> %s", cam_name, stem)
    return stems


def _canonicalise_names(out_dir: Path, stem: str) -> None:
    """Rename Blender's output to the stem the exporter parses.

    A File Output node always appends the frame counter to its slot name, so a
    slot called `Image_0_0_0001_0` lands as `Image_0_0_0001_00001.png`. The
    exporter parses the name into rig/resample/frame/subcam and then *asserts*
    the subcam matches the containing directory, so the extra digits are not a
    cosmetic problem: `..._00001.png` under `camera_0/` raises a subcamera
    mismatch and aborts the whole export. Renaming afterwards is the only way to
    get an exact name out of that node.
    """
    for pass_dir in out_dir.iterdir():
        cam_dir = pass_dir / "camera_0"
        if not cam_dir.is_dir():
            continue
        target_prefix = f"{pass_dir.name}_{stem}"
        for path in cam_dir.iterdir():
            if not path.is_file() or path.stem == target_prefix:
                continue
            if path.stem.startswith(f"{pass_dir.name}_{stem}"):
                path.replace(cam_dir / f"{target_prefix}{path.suffix}")


def _write_index_pair(out_dir: Path, stem: str, scene) -> None:
    """Write the MaterialSegmentation / Materials pair the exporter anchors on.

    Both are empty by construction and that is correct, not a stub. An injected
    defect is a field inside the host's own material, so there is no separate
    material for an index pass to find - the reason labels come from AOVs at
    all. The exporter still discovers frames through this pair, so it has to
    exist; the per-class AOVs beside it carry the actual labels.
    """
    import json

    import numpy as np

    height = scene.render.resolution_y
    width = scene.render.resolution_x
    seg_dir = out_dir / "MaterialSegmentation" / "camera_0"
    mat_dir = out_dir / "Materials" / "camera_0"
    seg_dir.mkdir(parents=True, exist_ok=True)
    mat_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        seg_dir / f"MaterialSegmentation_{stem}.npy",
        np.zeros((height, width), dtype=np.int32),
    )
    (mat_dir / f"Materials_{stem}.json").write_text(json.dumps({}) + "\n")
