"""Render wall scenes into the layout the defect exporter reads.

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

With ``depth=True`` (``build_dataset.py --export-depth``) each view also gets

    <out>/Depth/camera_0/Depth_<stem>.png            uint16, camera Z, 0.1 mm per step
    <out>/WallRelief/camera_0/WallRelief_<stem>.png  uint16, wall offset, 0.01 mm per step
    <out>/Camera/camera_0/Camera_<stem>.json         intrinsics, pose, encodings

from the same render: the depth, position and object-index passes of its first
sample, so they are exact per pixel, not blended across edges, and they come
from the displaced geometry the RGB was rendered from (a crack groove in a
decal or colour-only render is flat here, as it is to a depth camera).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import bpy
import numpy as np

logger = logging.getLogger(__name__)

_INSTANCE_AOVS = ("DefectInstance", "DefectInstanceCoverage", "DefectInstanceSquared")

# Depth and wall-relief encodings (also written into every Camera_<stem>.json).
DEPTH_STEP_M = 1e-4  # Depth PNG value * 0.1 mm = camera Z; 0 = no surface, or beyond 6.5535 m
RELIEF_STEP_MM = 0.01  # WallRelief PNG: (value - 32768) * 0.01 mm; 0 = not the main wall
RELIEF_ZERO = 32768
WALL_PASS_INDEX = 4093  # object index of the main wall in the index pass
_DEPTH_RAW = "_DepthRaw"  # scratch EXRs, removed after each view
_DEPTH_SLOTS = (("Depth", "Z"), ("Position", "P"), ("IndexOB", "Index"))  # EXR is RGB/RGBA only


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


def setup_passes(view_layer=None, depth: bool = False) -> list[str]:
    """Enable the defect AOVs on the view layer, and the depth, position and index passes if `depth`."""
    from cwg_infinigen.core.rendering.defect_aov import setup_defect_aovs

    names = setup_defect_aovs(view_layer=view_layer, enabled=True)
    # A clean scene still needs explicit zero passes for strict AOV export.
    # Unwritten Cycles AOVs evaluate to zero; no defect material is introduced.
    from cwg_infinigen.core.rendering.defect_aov import _ensure_view_layer_aov
    layer = view_layer if view_layer is not None else bpy.context.view_layer
    for name in ("DefectCrack", *_INSTANCE_AOVS):
        _ensure_view_layer_aov(layer, name)
        if name not in names:
            names.append(name)
    if depth:
        layer.use_pass_z = True
        layer.use_pass_position = True
        layer.use_pass_object_index = True
    return names


def mark_wall(wall) -> None:
    """Give the main wall its own object index, so the relief map covers it and nothing else."""
    clash = [o.name for o in bpy.data.objects if o is not wall and o.pass_index == WALL_PASS_INDEX]
    if clash:
        raise RuntimeError(f"object index {WALL_PASS_INDEX} already used by {clash}")
    wall.pass_index = WALL_PASS_INDEX


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


def build_compositor(out_dir: Path, stem: str, aov_names: list[str], exposure_ev=None, awb_strength=1.0,
                     depth: bool = False):
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
    if depth:
        for output, slot in _DEPTH_SLOTS:
            node = tree.nodes.new("CompositorNodeOutputFile")
            node.base_path = str(out_dir / _DEPTH_RAW / "camera_0")
            node.format.file_format = "OPEN_EXR"
            node.format.color_depth = "32"
            node.format.color_mode = "RGB"
            node.file_slots.clear()
            node.file_slots.new(f"{slot}_{stem}")
            tree.links.new(layers.outputs[output], node.inputs[0])
    return tree


def _read_exr(path: Path) -> np.ndarray:
    """(H, W, C) float32, top row first."""
    img = bpy.data.images.load(str(path), check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
        w, h = img.size
        buf = np.empty(w * h * img.channels, np.float32)
        img.pixels.foreach_get(buf)
        return buf.reshape(h, w, img.channels)[::-1]
    finally:
        bpy.data.images.remove(img)


def camera_model(cam, scene) -> dict:
    """Pinhole intrinsics and pose of a Blender camera, in OpenCV conventions.

    K maps camera coordinates (x right, y down, z forward) to pixels whose centres
    are at integer (u, v), u right and v down from the top-left pixel.
    """
    w, h = scene.render.resolution_x, scene.render.resolution_y
    frame = [tuple(v) for v in cam.data.view_frame(scene=scene)]
    xs, ys, d = [v[0] for v in frame], [v[1] for v in frame], -frame[0][2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    k = [[w * d / (x1 - x0), 0.0, w * -x0 / (x1 - x0) - 0.5],
         [0.0, h * d / (y1 - y0), h * y1 / (y1 - y0) - 0.5],
         [0.0, 0.0, 1.0]]
    loc, rot, _ = cam.matrix_world.decompose()  # Cycles ignores camera scale
    r = np.array(rot.to_matrix()) @ np.diag([1.0, -1.0, -1.0])  # Blender camera axes -> OpenCV
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = r, np.array(loc)
    return {"width": w, "height": h, "K": k, "cam_to_world": pose.tolist()}


def write_depth(out_dir: Path, stem: str, cam, scene) -> dict:
    """Turn this view's raw depth, position and index EXRs into the Depth and WallRelief PNGs."""
    import cv2

    raw_dir = out_dir / _DEPTH_RAW / "camera_0"
    raw = {}
    for output, slot in _DEPTH_SLOTS:
        found = sorted(raw_dir.glob(f"{slot}_{stem}*.exr"))
        if len(found) != 1:
            raise RuntimeError(f"expected one {slot} EXR for {stem}, found {found}")
        raw[output] = _read_exr(found[0])
    shutil.rmtree(out_dir / _DEPTH_RAW)
    z = raw["Depth"][..., 0].astype(np.float64)
    position = raw["Position"][..., :3].astype(np.float64)
    index = np.rint(raw["IndexOB"][..., 0]).astype(np.int64)
    hit = z < 1e9  # Cycles writes 1e10 where the ray hits nothing

    model = camera_model(cam, scene)
    pose = np.array(model["cam_to_world"])
    in_cam = (position - pose[:3, 3]) @ pose[:3, :3]  # world -> OpenCV camera
    mismatch = np.abs(in_cam[..., 2] - z)[hit]  # both passes are the first sample's hit

    depth_max = 65535 * DEPTH_STEP_M
    keep = hit & (z <= depth_max)
    depth_png = np.where(keep, np.rint(z / DEPTH_STEP_M), 0).astype(np.uint16)
    wall = hit & (index == WALL_PASS_INDEX)
    offset_mm = -position[..., 1] * 1000.0  # wall face is y = 0, facing the cameras at -y
    relief_png = np.where(wall, np.clip(RELIEF_ZERO + np.rint(offset_mm / RELIEF_STEP_MM), 1, 65535), 0)
    relief_png = relief_png.astype(np.uint16)

    for folder, image in (("Depth", depth_png), ("WallRelief", relief_png)):
        target = out_dir / folder / "camera_0"
        target.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target / f"{folder}_{stem}.png"), image):
            raise RuntimeError(f"could not write {folder} for {stem}")
    on_wall = offset_mm[wall]
    payload = {
        "schema": "syncrack3d/camera/1", "frame_stem": stem, "camera": cam.name, **model,
        "conventions": "OpenCV camera (x right, y down, z forward); world z up, metres; "
                       "the main wall's face is the plane y = 0 with its normal along -y, toward the cameras",
        "depth": {"file": f"Depth/camera_0/Depth_{stem}.png", "dtype": "uint16",
                  "metres": f"value * {DEPTH_STEP_M}", "kind": "camera z (planar), not ray length",
                  "invalid": f"0 = no surface or beyond {depth_max:.4f} m",
                  "source": "Cycles depth pass, first sample: displaced geometry, no blending at edges"},
        "wall_relief": {"file": f"WallRelief/camera_0/WallRelief_{stem}.png", "dtype": "uint16",
                        "millimetres": f"(value - {RELIEF_ZERO}) * {RELIEF_STEP_MM}",
                        "sign": "positive toward the camera, negative into the wall (crack grooves)",
                        "invalid": "0 = not the main wall (props, fixtures, other room surfaces, sky)",
                        "source": "Cycles position pass, first sample, signed offset from the plane y = 0; "
                                  "includes the wall material's own relief as well as the cracks"},
        "stats": {"hit_px": int(hit.sum()), "beyond_depth_range_px": int((hit & ~keep).sum()),
                  "wall_px": int(wall.sum()),
                  "wall_offset_mm_p01_p50_p99": [round(float(np.percentile(on_wall, q)), 3) for q in (1, 50, 99)]
                  if on_wall.size else None,
                  "depth_vs_position_max_mm": round(float(mismatch.max()) * 1000, 4) if mismatch.size else None},
    }
    target = out_dir / "Camera" / "camera_0"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"Camera_{stem}.json").write_text(json.dumps(payload, indent=1) + "\n")
    return payload["stats"]


def render_cameras(
    cameras: list[str],
    out_dir: Path,
    *,
    aov_names: list[str] | None = None,
    frame: int = 1,
    exposure_evs: list[float] | None = None,
    awb_strengths: list[float] | None = None,
    before_each=None,
    depth: bool = False,
) -> list[str]:
    """Render one frame per named camera. Returns the stems written.

    ``before_each(index, camera)`` runs just before each render, e.g. to relight.
    ``depth`` also writes Depth, WallRelief and Camera files (call setup_passes and
    mark_wall with depth first).
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
        build_compositor(out_dir, stem, aov_names, exposure_ev=exposure_ev, awb_strength=awb, depth=depth)
        bpy.ops.render.render(write_still=False)
        if depth:
            write_depth(out_dir, stem, cam, scene)
        _canonicalise_names(out_dir, stem)
        _write_index_pair(out_dir, stem, scene)
        write_view_metadata(out_dir, stem, cam)
        stems.append(stem)
        logger.info("rendered %s -> %s", cam_name, stem)
    return stems


def write_view_metadata(out_dir, stem, camera):
    """Keep structural framing separate from failed defect-target close-ups."""
    folder = Path(out_dir) / "DefectViewMeta" / "camera_0"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "syncrack3d/capture_role/1", "frame_stem": stem,
               "camera_role": camera.get("cwg_role", "context"),
               "structure_target": camera.get("cwg_structure_target"),
               "defect_target": camera.get("syncrack3d_target"), "instances": []}
    (folder / f"DefectViewMeta_{stem}.json").write_text(json.dumps(payload, indent=2))


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
