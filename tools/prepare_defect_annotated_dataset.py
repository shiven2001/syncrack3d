#!/usr/bin/env python3
"""
Prepare U-Net training data for defect segmentation from Infinigen render outputs.

Reads:
  - RGB images: .../Image/camera_*/Image_*_0_0001_0.png
  - Material index maps: .../MaterialSegmentation/camera_*/MaterialSegmentation_*_0_0001_0.npy
  - Material name -> pass_index: .../Materials/camera_*/Materials_*_0_0001_0.json

Layouts supported:
  - <input>/<scene>/frames/Image/... (standard Infinigen)
  - <input>/<scene>/Image/... (no frames/)
  - <input>/<scene>/<rig_dir>/Image/... e.g. all_frames/bathroom01/rig0_rs0/Image/...
  - <input>/<scene>/frames/<rig_dir>/Image/... e.g. new_AIC_data/bed_1/frames/rig0_rs0/Image/...

Maps defect materials to classes:
  - 0: background (non-defect)
  - 1: CrackMaterial_*
  - 2: PaintPeelMaterial_*
  - 3: SpallingMaterial_* and SpallingPlugMaterial_* (same class as spalling)
  - 4: BubbleMaterial_*
  - 5: OpenWiringMaterial_*
  - 6: PaintRunMaterial_*
  - 7: PaintPatchMaterial_*
  - 8: CornerChipMaterial_*
  - 9: TileChipMaterial_*

Outputs (default: segmentation pack only):
  - out_dir/images/<id>.png  (RGB)
  - out_dir/masks/<id>.png   (single-channel uint8: class k -> k * MASK_GRAY_STEP, max <= 180;
                              visible in viewers; distinct from legacy 0,85,170,255)
  - out_dir/splits/train.txt, val.txt, test.txt (~70% / 15% / 15% by SCENE,
    so no scene spans two splits; reproducible with --seed)
  - out_dir/class_names.txt

With ``--with-bboxes`` additionally:
  - out_dir/bboxes/<id>.json (one 2D bbox per defect *material pass* / placed asset plane:
                              union of all pixels with that Blender pass_index; pixel coords
                              {x_min,y_min,x_max,y_max}; fields pass_index, material_name)
  - out_dir/bboxes_yolo/<id>.txt (YOLO-format: class_id xc_n yc_n w_n h_n, all in [0,1])
  - out_dir/annotations_coco.json (aggregate COCO)

  Existing samples are skipped when images/ and masks/ already exist (and when using
  ``--with-bboxes``, bboxes/ and bboxes_yolo/ as well); use --force to regenerate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from imageio import imread, imwrite
from PIL import Image as PILImage
from tqdm import tqdm


# Class IDs for U-Net
CLASS_BACKGROUND = 0
CLASS_CRACK = 1
CLASS_PAINT_PEEL = 2
CLASS_SPALLING = 3
CLASS_PAINT_BUBBLE = 4
CLASS_EXPOSED_WIRING = 5
CLASS_PAINT_RUN = 6
CLASS_PAINT_PATCH = 7
CLASS_CORNER_CHIP = 8
CLASS_TILE_CHIP = 9

DEFECT_PREFIXES = {
    "CrackMaterial": CLASS_CRACK,
    "PaintPeelMaterial": CLASS_PAINT_PEEL,
    # Spalling plug planes use a distinct material name; count as spalling everywhere.
    "SpallingPlugMaterial": CLASS_SPALLING,
    "SpallingMaterial": CLASS_SPALLING,
    "BubbleMaterial": CLASS_PAINT_BUBBLE,
    "OpenWiringMaterial": CLASS_EXPOSED_WIRING,
    "PaintRunMaterial": CLASS_PAINT_RUN,
    "PaintPatchMaterial": CLASS_PAINT_PATCH,
    "CornerChipMaterial": CLASS_CORNER_CHIP,
    "TileChipMaterial": CLASS_TILE_CHIP,
}

CLASS_NAMES = {
    CLASS_BACKGROUND: "background",
    CLASS_CRACK: "crack",
    CLASS_PAINT_PEEL: "paint_peel",
    CLASS_SPALLING: "spalling",
    CLASS_PAINT_BUBBLE: "paint_bubble",
    CLASS_EXPOSED_WIRING: "exposed_wiring",
    CLASS_PAINT_RUN: "paint_run",
    CLASS_PAINT_PATCH: "paint_patch",
    CLASS_CORNER_CHIP: "corner_chip",
    CLASS_TILE_CHIP: "tile_chip",
}

NUM_CLASSES = 1 + max(DEFECT_PREFIXES.values())  # background 0 + defects 1..N
# PNG mask grayscale: class k -> k * step, where step is 180 // (NUM_CLASSES-1)
# so the brightest class stays under the legacy 85/170/255 levels. With the
# legacy ids running to 9 that is 20; the release profile's 4 classes use 60.
MASK_GRAY_STEP = max(1, 180 // (NUM_CLASSES - 1))

# Publication-v1 keeps only the three supported relief classes and maps the
# historical semantic ids 1/2/4 to contiguous training ids.  The original ids
# remain in every instance sidecar for round-trip compatibility.
RELEASE_LEGACY_TO_TRAINING = {0: 0, 1: 1, 2: 2, 4: 3}
RELEASE_TRAINING_TO_LEGACY = {
    value: key for key, value in RELEASE_LEGACY_TO_TRAINING.items()
}
RELEASE_CLASS_NAMES = {0: "background", 1: "crack", 2: "paint_peel", 3: "paint_bubble"}
CLASS_PROFILES = {
    "release": (RELEASE_LEGACY_TO_TRAINING, RELEASE_CLASS_NAMES),
    "legacy": ({class_id: class_id for class_id in CLASS_NAMES}, CLASS_NAMES),
}

INSTANCE_AOV_NAMES = {
    "first_moment": "DefectInstance",
    "coverage": "DefectInstanceCoverage",
    "second_moment": "DefectInstanceSquared",
}
# Below this coverage a defect leaves no trace a person would annotate, so its
# pixels stay background: a human labeller would not mark them and the real test
# set's ground truth does not either. Between here and the hard threshold is the
# marginal band - a few pixels, or low contrast - where neither label is
# defensible: calling it background teaches "defect texture is background", and
# calling it a defect teaches the model to invent evidence. That band is ignored
# instead, excluded from the loss rather than answered wrongly.
UNDER_RESOLUTION_IGNORE_FLOOR = 0.15

INSTANCE_ID_TOLERANCE = 0.05
INSTANCE_VARIANCE_TOLERANCE = 0.01
IGNORE_LABEL = 255

# Drop defect materials with almost no visible pixels in this frame (speckle / grazing).
BBOX_MIN_PIXEL_AREA = 4

# Smallest labelled area, in pixels, that is worth training on as a positive.
# BBOX_MIN_PIXEL_AREA alone is a speckle guard with no notion of evidence: at 4
# it let a wall crack emit a box from 5 pixels scattered as five single-pixel
# fragments. The identity was correct and the box was correct; there was simply
# not enough of the defect on screen to learn from, because the frame was shot
# with an ultrawide from 1.2 m and the crack is sub-pixel almost everywhere.
#
# 64 px is 8x8 at the 1920x1440 export, roughly 3x3 after the usual downscale to
# a 640 training input - about the smallest thing a detector can be asked to
# find. It is deliberately conservative: it removes what is unambiguously
# unusable without discarding legitimately small defects. Measured on seed 3,
# real instances ran 672-3650 px against this case's 5, so the threshold sits in
# a wide gap rather than on a knife edge.
#
# Instances below it become ignore rather than background: there *is* evidence,
# just not enough of it, which is the same argument as the marginal coverage
# band and the same answer.
MIN_INSTANCE_LABEL_PX = 64

# Coverage at which a pixel counts toward a THIN instance's bounding box (not
# its segmentation mask, which keeps the strict 0.5 threshold).
#
# A 1-2 px wide crack is spread across neighbouring pixels by antialiasing, so
# its true extent lives in the 0.15-0.5 coverage band. Boxing only the >=0.5
# pixels gives a handful of disconnected specks and usually no box at all. This
# is applied *after* the mass gate has already decided the instance is real, so
# it widens boxes for defects we know are there rather than inventing them.
THIN_BOX_COVERAGE = 0.15

# Least contrast, in 8-bit code values, that counts as visible evidence.
#
# Size is not evidence. An instance can cover thousands of pixels and be
# indistinguishable from the wall it sits on, because coverage is geometry - it
# says a defect occupies the pixel, not that anything about the pixel looks
# different. Relief only shows when light rakes across it, and indoor lighting
# is mostly diffuse: measured on a real bedroom render, twelve of fourteen
# labelled close-ups had a modulation under 1.5 code value, including one
# emitting a box over 5100 pixels of wall that looks exactly like clean wall.
# Training on those teaches a detector to predict defects from featureless
# plaster, which is worse than dropping the frame.
#
# Visible instances in the same measurements ran +10 to +59, invisible ones
# -1.3 to +0.1, so 2.0 sits in a wide gap. Instances below it are ignored rather
# than made background: the defect is really there, so calling the pixels clean
# wall would be its own lie.
MIN_INSTANCE_CONTRAST_CODES = 2.0


def build_pass_index_to_class(materials_json: dict) -> dict[int, int]:
    """Build mapping from Blender pass_index to class id 0..NUM_CLASSES-1 (0=background)."""
    pass_to_class = {}
    for mat_name, info in materials_json.items():
        pass_index = info["pass_index"]
        cls = CLASS_BACKGROUND
        for prefix, class_id in DEFECT_PREFIXES.items():
            if mat_name.startswith(prefix):
                cls = class_id
                break
        pass_to_class[pass_index] = cls
    return pass_to_class


def material_index_to_label_map(
    index_map: np.ndarray, pass_to_class: dict[int, int]
) -> np.ndarray:
    """Convert (H, W) pass indices to (H, W) class labels 0..NUM_CLASSES-1."""
    index_flat = index_map.ravel().astype(np.int64)
    out = np.zeros_like(index_flat, dtype=np.uint8)
    for pass_idx, cls in pass_to_class.items():
        out[index_flat == pass_idx] = cls
    return out.reshape(index_map.shape)


# AOV pass name -> class id. Mirrors DEFECT_AOV_CLASSES in
# infinigen/core/rendering/defect_aov.py, duplicated here because that module
# imports bpy and this tool runs in a plain interpreter.
DEFECT_AOV_NAMES = {
    "DefectCrack": CLASS_CRACK,
    "DefectPaintPeel": CLASS_PAINT_PEEL,
    "DefectSpalling": CLASS_SPALLING,
    "DefectPaintBubble": CLASS_PAINT_BUBBLE,
    "DefectExposedWiring": CLASS_EXPOSED_WIRING,
    "DefectPaintRun": CLASS_PAINT_RUN,
    "DefectPaintPatch": CLASS_PAINT_PATCH,
    "DefectCornerChip": CLASS_CORNER_CHIP,
    "DefectTileChip": CLASS_TILE_CHIP,
}

# A pixel counts as covered once the defect owns most of it. Coverage is
# antialiased, so this also decides how the mask edge is cut.
#
# 0.5 is the majority-coverage convention and matches the legacy material-index
# mask most closely (IoU 0.84 on a branched hairline crack). It does drop about
# 11% of the thinnest pixels, because a sub-pixel-wide hairline never reaches
# half coverage; lower it toward 0.25 to trace fine branches at the cost of
# precision (recall 0.97, precision 0.79 there). The integrated coverage itself
# is unbiased: total AOV mass matches the material-index pixel count to 0.1%.
AOV_COVERAGE_THRESHOLD = 0.5


def class_profile(profile: str) -> tuple[dict[int, int], dict[int, str]]:
    try:
        return CLASS_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown class profile {profile!r}") from exc


def remap_legacy_labels(
    label: np.ndarray, profile: str = "release"
) -> tuple[np.ndarray, np.ndarray]:
    """Map canonical/legacy semantic ids to a contiguous training taxonomy.

    Returns the mapped labels and an ignore mask.  Unsupported non-background
    classes are ignored, never converted into trustworthy background.
    """
    mapping, _ = class_profile(profile)
    out = np.zeros(label.shape, dtype=np.uint8)
    supported = np.zeros(label.shape, dtype=bool)
    for legacy_id, training_id in mapping.items():
        selected = label == legacy_id
        out[selected] = training_id
        supported |= selected
    return out, ~supported


def legacy_id_for_training(training_id: int, profile: str = "release") -> int:
    mapping, _ = class_profile(profile)
    inverse = {training: legacy for legacy, training in mapping.items()}
    if training_id not in inverse:
        raise ValueError(f"training id {training_id} is not in profile {profile}")
    return inverse[training_id]


def _require_shape(
    array: np.ndarray, shape: tuple[int, int], source: Path | str
) -> None:
    if array.shape != shape:
        raise ValueError(
            f"dimension mismatch for {source}: got {array.shape}, expected {shape}"
        )


def find_defect_aov_maps(npy_path: Path, stem: str) -> dict[int, Path]:
    """Locate per-class AOV coverage PNGs that sit alongside the index map.

    MaterialSegmentation/camera_N/MaterialSegmentation_<stem>.npy
      -> <AovName>/camera_N/<AovName>_<stem>.png
    """
    cam_dir = npy_path.parent
    frames_dir = cam_dir.parent.parent
    found: dict[int, Path] = {}
    for aov_name, class_id in DEFECT_AOV_NAMES.items():
        candidate = frames_dir / aov_name / cam_dir.name / f"{aov_name}_{stem}.png"
        if candidate.is_file():
            found[class_id] = candidate
    return found


def find_instance_aov_maps(npy_path: Path, stem: str) -> dict[str, Path]:
    """Locate the EXR moment triplet used for robust instance decoding."""
    cam_dir = npy_path.parent
    frames_dir = cam_dir.parent.parent
    found = {}
    for role, aov_name in INSTANCE_AOV_NAMES.items():
        path = frames_dir / aov_name / cam_dir.name / f"{aov_name}_{stem}.exr"
        if path.is_file():
            found[role] = path
    if found and set(found) != set(INSTANCE_AOV_NAMES):
        missing = sorted(set(INSTANCE_AOV_NAMES) - set(found))
        raise FileNotFoundError(
            f"incomplete instance AOV triplet next to {npy_path}; missing {missing}"
        )
    return found


def read_scalar_exr(path: Path) -> np.ndarray:
    """Read one float channel without relying on OpenCV's EXR build flags."""
    import Imath
    import OpenEXR

    handle = OpenEXR.InputFile(str(path))
    try:
        header = handle.header()
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        channels = header["channels"]
        channel = "R" if "R" in channels else next(iter(channels))
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        data = np.frombuffer(handle.channel(channel, pixel_type), dtype=np.float32)
        return data.reshape(height, width)
    finally:
        handle.close()


def decode_instance_moments(
    first_moment: np.ndarray,
    coverage: np.ndarray,
    second_moment: np.ndarray,
    valid_ids: set[int] | None = None,
    coverage_threshold: float = AOV_COVERAGE_THRESHOLD,
    id_tolerance: float = INSTANCE_ID_TOLERANCE,
    variance_tolerance: float = INSTANCE_VARIANCE_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode antialiased identity and reject pixels with mixed ownership.

    The output is ``(instance_ids, ambiguous, under_resolution)``.  A boundary
    containing one instance plus background remains decodable after dividing
    by coverage.  A pixel containing samples from multiple instance ids has
    positive conditional variance and is marked ambiguous.
    """
    if not (
        first_moment.shape == coverage.shape == second_moment.shape
        and first_moment.ndim == 2
    ):
        raise ValueError("instance moment arrays must be matching 2D images")
    coverage = np.clip(coverage.astype(np.float64), 0.0, 1.0)
    denom = np.maximum(coverage, np.finfo(np.float32).eps)
    mean = first_moment.astype(np.float64) / denom
    variance = np.maximum(second_moment.astype(np.float64) / denom - mean * mean, 0.0)
    nearest = np.rint(mean)
    supported = coverage >= coverage_threshold
    confident = (
        supported
        & (nearest >= 1)
        & (np.abs(mean - nearest) <= id_tolerance)
        & (variance <= variance_tolerance)
    )
    if valid_ids is not None:
        confident &= np.isin(nearest.astype(np.int64), sorted(valid_ids))
    instance_ids = np.zeros(first_moment.shape, dtype=np.uint32)
    instance_ids[confident] = nearest[confident].astype(np.uint32)
    ambiguous = supported & ~confident
    under_resolution = (coverage > 0.0) & ~supported
    return instance_ids, ambiguous, under_resolution


def decode_soft_instance_ids(
    first_moment: np.ndarray,
    coverage: np.ndarray,
    second_moment: np.ndarray,
    valid_ids: set[int] | None = None,
    id_tolerance: float = INSTANCE_ID_TOLERANCE,
    variance_tolerance: float = INSTANCE_VARIANCE_TOLERANCE,
) -> np.ndarray:
    """Attribute sub-threshold coverage when its conditional id is unambiguous."""
    if not (
        first_moment.shape == coverage.shape == second_moment.shape
        and first_moment.ndim == 2
    ):
        raise ValueError("instance moment arrays must be matching 2D images")
    coverage = np.clip(coverage.astype(np.float64), 0.0, 1.0)
    denom = np.maximum(coverage, np.finfo(np.float32).eps)
    mean = first_moment.astype(np.float64) / denom
    variance = np.maximum(second_moment.astype(np.float64) / denom - mean * mean, 0.0)
    nearest = np.rint(mean)
    confident = (
        (coverage > 0.0)
        & (nearest >= 1)
        & (np.abs(mean - nearest) <= id_tolerance)
        & (variance <= variance_tolerance)
    )
    if valid_ids is not None:
        confident &= np.isin(nearest.astype(np.int64), sorted(valid_ids))
    out = np.zeros(first_moment.shape, dtype=np.uint32)
    out[confident] = nearest[confident].astype(np.uint32)
    return out


def _ancestor_file(path: Path, filename: str, max_levels: int = 5) -> Path | None:
    current = path.parent
    for _ in range(max_levels):
        candidate = current / filename
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def find_defect_manifest(npy_path: Path) -> Path | None:
    return _ancestor_file(npy_path, "defect_instances.json")


def find_view_metadata(npy_path: Path, stem: str) -> Path | None:
    cam_dir = npy_path.parent
    frames_dir = cam_dir.parent.parent
    path = frames_dir / "DefectViewMeta" / cam_dir.name / f"DefectViewMeta_{stem}.json"
    return path if path.is_file() else None


def label_map_from_aovs(
    aov_maps: dict[int, Path],
    shape: tuple[int, int],
    threshold: float = AOV_COVERAGE_THRESHOLD,
) -> np.ndarray:
    """Build a class map from per-class coverage passes.

    Where two classes overlap the higher coverage wins, so a defect painted on
    top of another is labelled by whichever actually owns the pixel rather than
    by dictionary order.
    """
    label = np.zeros(shape, dtype=np.uint8)
    best = np.zeros(shape, dtype=np.float32)
    for class_id, path in sorted(aov_maps.items()):
        img = PILImage.open(path).convert("L")
        cov = np.asarray(img, dtype=np.float32) / 255.0
        _require_shape(cov, shape, path)
        take = (cov >= threshold) & (cov > best)
        label[take] = class_id
        best[take] = cov[take]
    return label


def _class_id_for_material_name(mat_name: str) -> int:
    for prefix, class_id in DEFECT_PREFIXES.items():
        if mat_name.startswith(prefix):
            return class_id
    return CLASS_BACKGROUND


def compute_asset_bboxes_from_material_passes(
    index_map: np.ndarray,
    materials_json: dict,
    min_area: int = BBOX_MIN_PIXEL_AREA,
) -> list[dict]:
    """One bbox per defect *material* (Blender pass_index), i.e. per placed plane asset.

    Infinigen assigns a unique pass_index per material datablock at render time; each
    crack/peel plane typically has its own material, so one tight union bbox matches
    one logical object instead of many CC blobs on a merged class mask.

    Each entry: class_id, class_name, pass_index, material_name, x_min, y_min, x_max,
    y_max, width, height, area (pixel count for that pass in this frame).
    """
    by_pass: dict[int, tuple[int, str]] = {}
    for mat_name, info in materials_json.items():
        cls_int = _class_id_for_material_name(mat_name)
        if cls_int == CLASS_BACKGROUND:
            continue
        pass_index = int(info["pass_index"])
        if pass_index not in by_pass:
            by_pass[pass_index] = (cls_int, mat_name)

    boxes: list[dict] = []
    for pass_index in sorted(by_pass.keys()):
        cls_int, mat_name = by_pass[pass_index]
        mask = index_map == pass_index
        ys, xs = np.where(mask)
        if ys.size < min_area:
            continue
        y_min = int(ys.min())
        y_max = int(ys.max())
        x_min = int(xs.min())
        x_max = int(xs.max())
        boxes.append(
            {
                "class_id": cls_int,
                "class_name": CLASS_NAMES.get(cls_int, f"class_{cls_int}"),
                "pass_index": pass_index,
                "material_name": mat_name,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "width": x_max - x_min + 1,
                "height": y_max - y_min + 1,
                "area": int(ys.size),
            }
        )
    return boxes


def compute_asset_bboxes_from_aovs(
    aov_maps: dict[int, Path],
    shape: tuple[int, int],
    threshold: float = AOV_COVERAGE_THRESHOLD,
    min_area: int = BBOX_MIN_PIXEL_AREA,
) -> list[dict]:
    """One bbox per connected blob in each class's AOV coverage pass.

    The material-index route gets instance separation for free, because each
    placed plane owns a material. An injected defect has no material of its
    own - every crack on a wall writes into that wall's single DefectCrack
    pass - so instances are recovered as connected components instead. The
    solver spaces defects apart, so blobs and instances line up in practice;
    two overlapping defects of one class would merge into a single box, and a
    peel patch whose islands detach can split into several.
    """
    from scipy.ndimage import label as cc_label

    boxes: list[dict] = []
    for class_id, path in sorted(aov_maps.items()):
        img = PILImage.open(path).convert("L")
        cov = np.asarray(img, dtype=np.float32) / 255.0
        _require_shape(cov, shape, path)
        mask = cov >= threshold
        if not mask.any():
            continue
        # Hysteresis, as edge detectors use for the same reason: a hairline
        # crack is sub-pixel wide for stretches, so its coverage dips under the
        # threshold and plain connected components chop one crack into a long
        # blob plus a trail of 4-pixel specks. Connect at half the threshold,
        # keep only blobs that reach the full one, and measure the box from the
        # confident pixels.
        loose = cov >= threshold * 0.5
        blobs, n_blobs = cc_label(loose, structure=np.ones((3, 3), dtype=int))
        for blob_id in range(1, n_blobs + 1):
            blob = blobs == blob_id
            # Keep the blob only if it has a confident core, but measure the box
            # from the loose mask. A mask wants majority coverage per pixel; a
            # box wants to bound the whole object, and a hairline fork at 30%
            # coverage is still part of the crack. Measuring at the strict
            # threshold truncated one test crack's box from 108 to 43 px tall.
            if (mask & blob).sum() < min_area:
                continue
            ys, xs = np.where(blob)
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    "pass_index": -1,
                    "material_name": f"{path.parent.parent.name}#{blob_id}",
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width": x_max - x_min + 1,
                    "height": y_max - y_min + 1,
                    "area": int(ys.size),
                }
            )
    return boxes


def load_manifest(path: Path) -> tuple[dict, dict[int, dict]]:
    payload = json.loads(path.read_text())
    records = payload.get("instances")
    if not isinstance(records, list):
        raise ValueError(f"manifest {path} has no instances list")
    by_id = {}
    for record in records:
        instance_id = int(record["instance_id"])
        if instance_id <= 0 or instance_id in by_id:
            raise ValueError(f"manifest {path} has invalid/duplicate id {instance_id}")
        by_id[instance_id] = record
    return payload, by_id


def instance_contrast_codes(luma, pixels, other_defects):
    """How far an instance stands out from the wall around it, in code values.

    Two statistics, whichever is larger. A crack is uniformly darker than its
    surround, so the difference of means catches it. A blister is a dome - lit on
    one side, shadowed on the other - so its mean matches the wall however
    obvious it looks, and what betrays it is carrying more spread than the wall
    does. Using only one of the two would call one of the classes invisible.

    Other defects are excluded from the surround so a neighbour cannot stand in
    for clean wall.
    """
    from scipy.ndimage import binary_dilation

    if pixels.sum() < 4:
        return None
    ring = (
        binary_dilation(pixels, iterations=25)
        & ~binary_dilation(pixels, iterations=8)
        & ~other_defects
    )
    if ring.sum() < 20:
        return None
    inside, outside = luma[pixels], luma[ring]
    mean_delta = abs(float(inside.mean()) - float(outside.mean()))
    i_lo, i_hi = np.percentile(inside, [5, 95])
    o_lo, o_hi = np.percentile(outside, [5, 95])
    return float(max(mean_delta, (i_hi - i_lo) - (o_hi - o_lo)))


def compute_asset_bboxes_from_instances(
    instance_map: np.ndarray,
    manifest_by_id: dict[int, dict],
    profile: str = "release",
    min_area: int = BBOX_MIN_PIXEL_AREA,
    evidence_by_id: dict[int, float] | None = None,
    min_evidence: float = 0.0,
) -> list[dict]:
    """One visible box per manifest instance, including disconnected pieces.

    `evidence_by_id` carries each instance's coverage mass. When supplied, the
    "is there enough of this to train on" decision is made on mass and the pixel
    count here falls back to being a pure speckle guard, which is all it was
    ever good for on thin structures.
    """
    mapping, names = class_profile(profile)
    boxes = []
    for instance_id in sorted(manifest_by_id):
        record = manifest_by_id[instance_id]
        if not record.get("retained", record.get("render_path") == "injected"):
            continue
        legacy_class_id = int(record["class_id"])
        training_class_id = mapping.get(legacy_class_id)
        if training_class_id in (None, 0):
            continue
        ys, xs = np.where(instance_map == instance_id)
        if ys.size < min_area:
            continue
        if evidence_by_id is not None and min_evidence > 0:
            if float(evidence_by_id.get(instance_id, ys.size)) < min_evidence:
                continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        boxes.append(
            {
                "instance_id": instance_id,
                "class_id": training_class_id,
                "legacy_class_id": legacy_class_id,
                "class_name": names[training_class_id],
                "pass_index": -1,
                "material_name": record.get("host_material"),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "width": x_max - x_min + 1,
                "height": y_max - y_min + 1,
                "area": int(ys.size),
            }
        )
    return boxes


# How a person boxes blisters: by visible group - one dome, or a few touching
# ones - not the whole cluster the generator places as one instance, and not
# every 2 mm bump inside it. On the real indoor_defects photos that carry
# blisters the median is 3 boxes per photo with a median side of 5.6% of the
# image; one box per cluster gave 1 box per frame at 21% of the side, and one
# box per connected bump would give 14 boxes at 0.7%. Pieces of one instance
# closer than the gap merge into one group. Off (0) by default.
BLISTER_GROUP_GAP_FRAC = 0.0


def group_blister_boxes(
    boxes: list[dict],
    instance_map: np.ndarray,
    rgb: np.ndarray,
    gap_frac: float,
    min_px: int = MIN_INSTANCE_LABEL_PX,
    min_contrast: float = MIN_INSTANCE_CONTRAST_CODES,
    min_side_frac: float = 0.0,
) -> list[dict]:
    """Replace each paint_bubble instance box with one box per visible group.

    Each group is gated again on its own pixels, with the same size and contrast
    tests the instance passed: a cluster that is visible as a whole can still
    have faint outlying bumps that nobody would box on their own.
    """
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import label as cc_label

    h, w = instance_map.shape
    gap = max(1, int(round(gap_frac * min(h, w))))
    luma = rgb.astype(np.float64)
    luma = luma.mean(axis=2) if luma.ndim == 3 else luma
    all_defect_px = instance_map > 0
    out = []
    for box in boxes:
        if box.get("class_name") != "paint_bubble":
            out.append(box)
            continue
        pixels = instance_map == box["instance_id"]
        groups, n = cc_label(
            binary_dilation(pixels, iterations=gap), structure=np.ones((3, 3), dtype=int)
        )
        for g in range(1, n + 1):
            member = pixels & (groups == g)
            area = int(member.sum())
            if area < max(int(min_px), BBOX_MIN_PIXEL_AREA):
                continue
            contrast = None
            if min_contrast > 0:
                contrast = instance_contrast_codes(luma, member, all_defect_px & ~member)
                if contrast is not None and contrast < float(min_contrast):
                    continue
            ys, xs = np.nonzero(member)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            side = ((x_max - x_min + 1) * (y_max - y_min + 1)) ** 0.5
            if side < min_side_frac * min(h, w):
                continue
            item = dict(box)
            item.update(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                width=x_max - x_min + 1,
                height=y_max - y_min + 1,
                area=area,
                blister_group=g,
                contrast_codes=None if contrast is None else round(float(contrast), 3),
            )
            out.append(item)
    return out


def remap_boxes(boxes: list[dict], profile: str) -> list[dict]:
    mapping, names = class_profile(profile)
    remapped = []
    for box in boxes:
        legacy_class_id = int(box["class_id"])
        training_class_id = mapping.get(legacy_class_id)
        if training_class_id in (None, 0):
            continue
        item = dict(box)
        item["legacy_class_id"] = legacy_class_id
        item["class_id"] = training_class_id
        item["class_name"] = names[training_class_id]
        remapped.append(item)
    return remapped


def build_instance_sidecar(
    sample_id: str,
    scene_id: str,
    stem: str,
    instance_map: np.ndarray,
    soft_instance_map: np.ndarray,
    coverage: np.ndarray,
    ambiguous: np.ndarray,
    under_resolution: np.ndarray,
    manifest_path: Path,
    manifest_payload: dict,
    manifest_by_id: dict[int, dict],
    view_meta_path: Path | None,
    profile: str,
    coverage_threshold: float = AOV_COVERAGE_THRESHOLD,
    min_area: int = BBOX_MIN_PIXEL_AREA,
    under_resolution_floor: float | None = UNDER_RESOLUTION_IGNORE_FLOOR,
    contrast_by_id: dict[int, float] | None = None,
    demoted_ids: set[int] | None = None,
    under_resolution_ids: set[int] | None = None,
    evidence_by_id: dict[int, float] | None = None,
    min_contrast: float = MIN_INSTANCE_CONTRAST_CODES,
    label_diagnostics: dict[str, int] | None = None,
) -> dict:
    mapping, names = class_profile(profile)
    contrast_by_id = contrast_by_id or {}
    under_resolution_ids = under_resolution_ids or set()
    evidence_by_id = evidence_by_id or {}
    demoted_ids = demoted_ids or set()
    view_by_id = {}
    if view_meta_path is not None:
        view_payload = json.loads(view_meta_path.read_text())
        view_by_id = {
            int(record["instance_id"]): record
            for record in view_payload.get("instances", [])
        }
    records = []
    for instance_id in sorted(manifest_by_id):
        manifest_record = manifest_by_id[instance_id]
        pixels = instance_map == instance_id
        labelled = int(pixels.sum())
        soft_pixels = soft_instance_map == instance_id
        soft_sum = float(coverage[soft_pixels].sum())
        legacy_class_id = int(manifest_record["class_id"])
        training_class_id = mapping.get(legacy_class_id)
        retained = bool(
            manifest_record.get(
                "retained", manifest_record.get("render_path") == "injected"
            )
        )
        if not retained:
            label_status = manifest_record.get("status", "dropped")
        elif instance_id in under_resolution_ids:
            # Too little of it on screen, measured as coverage mass. Reported
            # apart from no_contrast because the two want opposite fixes: this
            # one is a framing/resolution problem, that one is a lighting one.
            label_status = "under_resolution"
        elif instance_id in demoted_ids:
            # Coverage says a defect is here and the image does not show it.
            # Recorded distinctly from under_resolution: that is "too small to
            # resolve", this is "resolved and indistinguishable from the wall",
            # and the second is a lighting problem rather than a sampling one.
            label_status = "no_contrast"
        elif labelled >= min_area:
            label_status = "visible"
        elif labelled or soft_sum > 0.0:
            label_status = "under_resolution"
        else:
            label_status = "not_visible"
        record = {
            "instance_id": instance_id,
            "legacy_class_id": legacy_class_id,
            "training_class_id": training_class_id,
            "class_name": names.get(training_class_id),
            "label_status": label_status,
            "labelled_pixel_count": labelled,
            "labelled_soft_coverage_sum": round(soft_sum, 6),
            "contrast_codes": contrast_by_id.get(instance_id),
            "coverage_mass_px": evidence_by_id.get(instance_id),
            "manifest": manifest_record,
            "view": view_by_id.get(instance_id),
        }
        records.append(record)
    return {
        "schema": "syndefect3d/frame_instances/1",
        "sample_id": sample_id,
        "scene_id": scene_id,
        "frame_stem": stem,
        "class_profile": profile,
        "manifest_schema": manifest_payload.get("schema"),
        "manifest_path": str(manifest_path),
        "view_metadata_path": str(view_meta_path) if view_meta_path else None,
        "hard_coverage_threshold": float(coverage_threshold),
        "min_contrast_codes": float(min_contrast),
        "under_resolution_ignore_floor": (
            None if under_resolution_floor is None else float(under_resolution_floor)
        ),
        "hard_threshold_policy": "syndefect3d/majority_coverage/1",
        # /2: the semantic class is the owner's, and only a cross-class
        # conflict joins blended ids in `ambiguous`. /1 also voided owned rim
        # pixels whose class AOV fell just under the threshold.
        "ambiguity_policy": "conditional_id_variance/2",
        "semantic_label_source": "instance_owner_class",
        "ambiguous_pixel_count": int(ambiguous.sum()),
        **(label_diagnostics or {}),
        "under_resolution_pixel_count": int(under_resolution.sum()),
        "instances": records,
    }


def write_bboxes(
    sample_id: str,
    boxes: list[dict],
    image_h: int,
    image_w: int,
    out_bboxes: Path,
    out_bboxes_yolo: Path,
) -> None:
    out_bboxes.mkdir(parents=True, exist_ok=True)
    out_bboxes_yolo.mkdir(parents=True, exist_ok=True)

    payload = {
        "image": f"{sample_id}.png",
        "width": int(image_w),
        "height": int(image_h),
        "boxes": boxes,
    }
    (out_bboxes / f"{sample_id}.json").write_text(json.dumps(payload, indent=2))

    yolo_lines: list[str] = []
    for b in boxes:
        xc = (b["x_min"] + b["x_max"] + 1) / 2.0 / image_w
        yc = (b["y_min"] + b["y_max"] + 1) / 2.0 / image_h
        bw = b["width"] / image_w
        bh = b["height"] / image_h
        # YOLO class ids are 0-indexed; treat defects 1..5 as 0..4.
        yolo_cls = b["class_id"] - 1
        yolo_lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    (out_bboxes_yolo / f"{sample_id}.txt").write_text(
        ("\n".join(yolo_lines) + "\n") if yolo_lines else ""
    )


def _is_frame_leaf(path: Path) -> bool:
    return (
        (path / "Image").is_dir()
        and (path / "MaterialSegmentation").is_dir()
        and (path / "Materials").is_dir()
    )


def _discover_rig_leaves(
    parent: Path, scene_name: str
) -> list[tuple[Path, Path, Path, str]]:
    """One Infinigen render dir per (rig, resample), named e.g. rig0_rs1.

    The PNG stem is Image_<rig>_<subcam>_<frame>_<id>.png; resample_idx is not
    in the filename, so the leaf folder name has to stay in the sample id or
    rs0 and rs1 overwrite each other.
    """
    results = []
    for leaf in sorted(parent.iterdir()):
        if leaf.is_dir() and _is_frame_leaf(leaf):
            results.extend(_discover_in_frames(leaf, f"{scene_name}_{leaf.name}"))
    return results


def _discover_scene_dir(
    scene_dir: Path, scene_name: str
) -> list[tuple[Path, Path, Path, str]]:
    if _is_frame_leaf(scene_dir):
        return _discover_in_frames(scene_dir, scene_name)
    frames = scene_dir / "frames"
    if frames.is_dir():
        found = (
            _discover_in_frames(frames, scene_name)
            if _is_frame_leaf(frames)
            else []
        )
        if not found:
            found = _discover_rig_leaves(frames, scene_name)
        if found:
            return found
    return _discover_rig_leaves(scene_dir, scene_name)


def discover_scene_frames(root: Path) -> list[tuple[Path, Path, Path, str]]:
    """
    Discover all (image, mask_npy, materials_json, sample_id) under root.
    root can be a single scene dir (e.g. .../room2) or a parent (e.g. .../indoors) containing room1, room2, ...
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return []

    # Single scene: Image at root, or frames/ (flat Image/ or per-rig leaves).
    if _is_frame_leaf(root):
        return _discover_in_frames(root, root.name)
    frames = root / "frames"
    if frames.is_dir():
        found = (
            _discover_in_frames(frames, root.name)
            if _is_frame_leaf(frames)
            else []
        )
        if not found:
            found = _discover_rig_leaves(frames, root.name)
        if found:
            return found

    # Parent of many scenes, or a frames dump whose children are rig leaves
    # (run_bedroom_dataset.py writes out/frames/s{seed}_rig{n}_rs{r}/).
    results = []
    for scene_dir in sorted(root.iterdir()):
        if scene_dir.is_dir():
            results.extend(_discover_scene_dir(scene_dir, scene_dir.name))
    return results


def _discover_in_frames(
    frames_dir: Path, scene_name: str
) -> list[tuple[Path, Path, Path, str]]:
    """Discover triplets inside a frames-like directory.

    Supports both:
      - <scene>/frames/Image/..., MaterialSegmentation/..., Materials/...
      - <scene>/Image/..., MaterialSegmentation/..., Materials/...
    """
    # If frames_dir already is the "frames" folder, its children are Image/, MaterialSegmentation/, Materials/.
    # If frames_dir is the scene dir, it directly contains Image/, MaterialSegmentation/, Materials/.
    if (frames_dir / "Image").is_dir():
        image_dir = frames_dir / "Image"
        seg_dir = frames_dir / "MaterialSegmentation"
        mat_dir = frames_dir / "Materials"
    else:
        image_dir = frames_dir / "frames" / "Image"
        seg_dir = frames_dir / "frames" / "MaterialSegmentation"
        mat_dir = frames_dir / "frames" / "Materials"
    if not image_dir.is_dir() or not seg_dir.is_dir() or not mat_dir.is_dir():
        return []

    # Assume one camera subfolder (camera_0)
    results = []
    for cam_dir in sorted(image_dir.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith("camera_"):
            continue
        subcam = cam_dir.name  # camera_0
        seg_cam = seg_dir / cam_dir.name
        mat_cam = mat_dir / cam_dir.name
        if not seg_cam.is_dir() or not mat_cam.is_dir():
            continue

        # Image_<rig>_<resample>_<frame>_<subcam>.png.  Keep every component in
        # the sample id so animations/resamples cannot overwrite one another.
        pattern = re.compile(r"^Image_(\d+)_(\d+)_(\d{4})_(\d+)\.png$")
        for img_path in sorted(cam_dir.glob("Image_*.png")):
            m = pattern.match(img_path.name)
            if not m:
                continue
            rig, resample, frame, subcam_id = m.groups()
            if cam_dir.name != f"camera_{subcam_id}":
                raise ValueError(
                    f"subcamera mismatch: {img_path} is under {cam_dir.name}"
                )
            stem = f"{rig}_{resample}_{frame}_{subcam_id}"
            npy_path = seg_cam / f"MaterialSegmentation_{stem}.npy"
            json_path = mat_cam / f"Materials_{stem}.json"
            if not npy_path.is_file() or not json_path.is_file():
                continue
            sample_id = (
                f"{scene_name}_rig{rig}_resample{resample}_frame{frame}_{subcam}"
            )
            register_scene_lineage(sample_id, npy_path)
            results.append((img_path, npy_path, json_path, sample_id))
    return results


def realism_postprocess_rgb(
    rgb: np.ndarray,
    saturation_scale: float = 0.35,
    brightness_scale: float = 1.15,
    noise_sigma: float = 4.0,
    seed=None,
) -> np.ndarray:
    """Desaturate, brighten, and add film grain to match phone-photo statistics.

    Seeded from the sample id by the caller. Unseeded, exporting the same frames
    twice produced different pixels, so a pack could not be regenerated and its
    checksums meant nothing - the images would differ while every label stayed
    identical.
    """
    img = rgb.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    gray = img.mean(axis=-1, keepdims=True)
    out = gray + (img - gray) * saturation_scale
    out = np.clip(out * brightness_scale, 0.0, 255.0)
    rng = np.random.default_rng(seed)
    out = np.clip(out + rng.normal(0.0, noise_sigma, out.shape), 0.0, 255.0)
    return out.round().astype(np.uint8)


def process_sample(
    image_path: Path,
    npy_path: Path,
    json_path: Path,
    sample_id: str,
    out_images: Path,
    out_masks: Path,
    out_bboxes: Path,
    out_bboxes_yolo: Path,
    with_bboxes: bool = False,
    realism_postprocess: bool = False,
    saturation_scale: float = 0.35,
    brightness_scale: float = 1.15,
    noise_sigma: float = 4.0,
    aov_masks: str = "auto",
    aov_threshold: float = AOV_COVERAGE_THRESHOLD,
    instance_masks: str = "auto",
    class_profile_name: str = "release",
    out_instance_masks: Path | None = None,
    out_instance_records: Path | None = None,
    out_ignore_masks: Path | None = None,
    under_resolution_floor: float | None = UNDER_RESOLUTION_IGNORE_FLOOR,
    min_instance_px: int = MIN_INSTANCE_LABEL_PX,
    thin_box_coverage: float = THIN_BOX_COVERAGE,
    min_contrast: float = MIN_INSTANCE_CONTRAST_CODES,
    blister_group_gap: float = BLISTER_GROUP_GAP_FRAC,
    blister_min_contrast: float | None = None,
    blister_min_side: float = 0.0,
) -> bool:
    """Build semantic and instance-aware outputs for one rendered frame."""
    try:
        class_profile(class_profile_name)
        rgb = imread(image_path)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        elif rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if realism_postprocess:
            rgb = realism_postprocess_rgb(
                rgb,
                saturation_scale=saturation_scale,
                brightness_scale=brightness_scale,
                noise_sigma=noise_sigma,
                # Deterministic per sample, so re-exporting a pack reproduces it.
                seed=int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16),
            )

        index_map = np.load(npy_path)
        if index_map.ndim == 3:
            index_map = index_map.squeeze()
        if index_map.ndim != 2:
            raise ValueError(f"material index map must be 2D, got {index_map.shape}")
        _require_shape(index_map, rgb.shape[:2], npy_path)

        with open(json_path) as f:
            materials = json.load(f)
        pass_to_class = build_pass_index_to_class(materials)

        # AOV coverage passes are antialiased and survive a defect that shares
        # its host's material, so prefer them when the render produced any.
        aov_maps = {}
        if aov_masks != "off":
            stem = npy_path.stem.replace("MaterialSegmentation_", "", 1)
            aov_maps = find_defect_aov_maps(npy_path, stem)
        if aov_masks == "on" and not aov_maps:
            raise FileNotFoundError(
                f"--aov-masks on, but no defect AOV passes found next to {npy_path}"
            )
        stem = npy_path.stem.replace("MaterialSegmentation_", "", 1)
        manifest_path = find_defect_manifest(npy_path)
        manifest_payload = None
        manifest_by_id = None
        if manifest_path is not None:
            manifest_payload, manifest_by_id = load_manifest(manifest_path)

        if aov_maps:
            legacy_label = label_map_from_aovs(aov_maps, rgb.shape[:2], aov_threshold)
        else:
            legacy_label = material_index_to_label_map(index_map, pass_to_class)

        if aov_masks == "on" and manifest_by_id is not None:
            expected_classes = {
                int(record["class_id"])
                for record in manifest_by_id.values()
                if record.get("retained", record.get("render_path") == "injected")
            }
            missing_classes = sorted(expected_classes - set(aov_maps))
            if missing_classes:
                raise FileNotFoundError(
                    f"missing required class AOVs {missing_classes} for {npy_path}"
                )

        instance_aov_maps = {}
        if instance_masks != "off":
            instance_aov_maps = find_instance_aov_maps(npy_path, stem)
        if instance_masks == "on" and not instance_aov_maps:
            raise FileNotFoundError(
                f"--instance-masks on, but no instance AOV triplet found next to {npy_path}"
            )
        if instance_aov_maps and manifest_by_id is None:
            raise FileNotFoundError(
                f"instance AOVs exist for {npy_path}, but defect_instances.json is missing"
            )

        instance_map = None
        soft_instance_map = None
        ownership_coverage = None
        ambiguous = np.zeros(rgb.shape[:2], dtype=bool)
        under_resolution = np.zeros(rgb.shape[:2], dtype=bool)
        label_diagnostics = {}
        class_without_owner = np.zeros(rgb.shape[:2], dtype=bool)
        if instance_aov_maps:
            moments = {
                role: read_scalar_exr(path) for role, path in instance_aov_maps.items()
            }
            for role, array in moments.items():
                _require_shape(array, rgb.shape[:2], instance_aov_maps[role])
            retained_ids = {
                instance_id
                for instance_id, record in manifest_by_id.items()
                if record.get("retained", record.get("render_path") == "injected")
            }
            ownership_coverage = moments["coverage"]
            instance_map, ambiguous, under_resolution = decode_instance_moments(
                moments["first_moment"],
                ownership_coverage,
                moments["second_moment"],
                valid_ids=retained_ids,
                coverage_threshold=aov_threshold,
            )
            soft_instance_map = decode_soft_instance_ids(
                moments["first_moment"],
                ownership_coverage,
                moments["second_moment"],
                valid_ids=retained_ids,
            )
            blended_ids = ambiguous.copy()
            # One ownership decision drives both the semantic and the instance
            # label. The class AOVs average soft field coverage over samples;
            # the instance coverage is the fraction of samples a field owns. Each
            # thresholded at 0.5 on its own, they disagree on the antialiased
            # rim: on the full CWG3 render set, 16,012 rim pixels had a single,
            # confidently decoded owner and a class AOV just under 0.5, against
            # zero pixels mixing two ids. Voiding those as ambiguous reported an
            # estimator mismatch as identity blending, and left the semantic
            # mask and the instance map describing different shapes. The class
            # comes from the owner's manifest record instead, and the class AOVs
            # only catch a real conflict - an owner of one class where another
            # class's AOV is positive - which stays ambiguous.
            owner_class = np.zeros_like(legacy_label)
            for instance_id in retained_ids:
                owner_class[instance_map == instance_id] = int(
                    manifest_by_id[instance_id]["class_id"]
                )
            owned = instance_map > 0
            class_conflict = owned & (legacy_label != 0) & (legacy_label != owner_class)
            rim_disagreement = owned & (legacy_label == 0)
            # Class evidence with no owner: ignore, since neither label is safe.
            # In practice this is the 0.15-0.5 marginal band, already ignored.
            class_without_owner = ~owned & (legacy_label != 0)
            ambiguous |= class_conflict
            instance_map[class_conflict] = 0
            owner_class[class_conflict] = 0
            legacy_label = owner_class
            label_diagnostics = {
                "blended_id_pixel_count": int(blended_ids.sum()),
                "class_conflict_pixel_count": int(class_conflict.sum()),
                "class_rim_disagreement_pixel_count": int(rim_disagreement.sum()),
                "class_without_owner_pixel_count": int(class_without_owner.sum()),
            }

        label, unsupported = remap_legacy_labels(
            legacy_label, profile=class_profile_name
        )
        ignore = ambiguous | unsupported | class_without_owner
        marginal = np.zeros_like(ignore)
        if under_resolution_floor is not None and ownership_coverage is not None:
            # Partial coverage on the antialiased rim of a well-resolved defect
            # lands here too, which is the usual boundary-ignore band and is
            # what you want: those pixels are genuinely part defect, part wall.
            marginal = under_resolution & (
                ownership_coverage >= float(under_resolution_floor)
            )
            ignore |= marginal
        # Two independent demotion reasons, tracked separately. Merging them is
        # what made a hairline report label_status "no_contrast" when the real
        # cause was "too few solid pixels" - a lighting diagnosis for a sampling
        # problem, which sends anyone reading the sidecar to the wrong fix.
        under_resolution_ids: set[int] = set()
        no_contrast_ids: set[int] = set()
        thin_ids: set[int] = set()
        instance_contrast: dict[int, float] = {}
        instance_evidence: dict[int, float] = {}
        if instance_map is not None and (min_instance_px > 0 or min_contrast > 0):
            # Two independent ways an instance can fail to be a usable target:
            # too little of it on screen, and too little contrast against the
            # wall. Neither implies the other - the worst case measured was a
            # 5100 px instance at 0.0 contrast - so both are checked, and both
            # ignore rather than background, because the defect is genuinely
            # there and calling its pixels clean wall would be its own lie.
            luma = rgb.astype(np.float64)
            luma = luma.mean(axis=2) if luma.ndim == 3 else luma
            all_defect_px = instance_map > 0
            # Evidence is coverage MASS, not the count of majority-coverage
            # pixels.
            #
            # The AOV is an area field, so summing it is the unbiased estimate
            # of how much defect is on screen - the same identity that makes
            # total AOV mass match the material-index pixel count to 0.1%.
            # Counting pixels over 0.5 is a *biased* estimator, and the bias is
            # catastrophic exactly where it matters: a hairline 1-2 px wide
            # almost never owns most of any pixel, so the count collapses to
            # near zero while the crack is plainly visible. Measured on a
            # dining-room ceiling hairline (0.3-0.5 mm at 0.54 m): 2298 pixels
            # of partial coverage, 8 over 0.5. The count said "no evidence"; the
            # mass says several hundred pixels of crack, which is what the image
            # shows and what a detector finds.
            #
            # For a well-resolved blob the two agree, so this only changes the
            # verdict for thin structures. Speckle stays out: isolated pixels
            # carry almost no mass, which is the property the 64 floor was
            # protecting and the reason not to simply lower the coverage gate.
            for inst_id in (int(i) for i in np.unique(instance_map) if i):
                pixels = instance_map == inst_id
                majority_px = int(pixels.sum())
                mass = majority_px
                if soft_instance_map is not None and ownership_coverage is not None:
                    mass = float(
                        ownership_coverage[soft_instance_map == inst_id].sum()
                    )
                evidence = max(float(majority_px), float(mass))
                instance_evidence[inst_id] = round(float(evidence), 2)
                if min_instance_px > 0 and evidence < float(min_instance_px):
                    ignore |= pixels
                    under_resolution_ids.add(inst_id)
                    continue
                if min_contrast > 0:
                    contrast = instance_contrast_codes(
                        luma, pixels, all_defect_px & ~pixels
                    )
                    if contrast is not None:
                        instance_contrast[inst_id] = round(contrast, 3)
                        if contrast < float(min_contrast):
                            ignore |= pixels
                            no_contrast_ids.add(inst_id)
                            continue
                # Passed on evidence but too thin to own 64 majority pixels: its
                # box has to come from the soft footprint or there is no box.
                if majority_px < int(min_instance_px):
                    thin_ids.add(inst_id)
        label[ignore] = 0

        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)
        imwrite(out_images / f"{sample_id}.png", rgb)
        # Spread grayscale: visible in viewers; distinct from legacy 85/170/255 (see class_names.txt)
        _, profile_names = class_profile(class_profile_name)
        mask_step = max(1, 180 // (len(profile_names) - 1))
        mask_gray = np.clip(label.astype(np.uint32) * mask_step, 0, 255).astype(
            np.uint8
        )
        PILImage.fromarray(mask_gray, mode="L").save(
            str(out_masks / f"{sample_id}.png")
        )

        if instance_map is not None:
            root = out_masks.parent
            out_instance_masks = out_instance_masks or root / "instance_masks"
            out_instance_records = out_instance_records or root / "instances"
            out_ignore_masks = out_ignore_masks or root / "ignore_masks"
            out_instance_masks.mkdir(parents=True, exist_ok=True)
            out_instance_records.mkdir(parents=True, exist_ok=True)
            out_ignore_masks.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_instance_masks / f"{sample_id}.npz",
                instance_id=instance_map,
                soft_instance_id=soft_instance_map,
                ownership_coverage=ownership_coverage.astype(np.float16),
                ambiguous=ambiguous,
                under_resolution=under_resolution,
            )
            PILImage.fromarray(ignore.astype(np.uint8) * IGNORE_LABEL, mode="L").save(
                out_ignore_masks / f"{sample_id}.png"
            )
            sidecar = build_instance_sidecar(
                contrast_by_id=instance_contrast,
                demoted_ids=no_contrast_ids,
                under_resolution_ids=under_resolution_ids,
                evidence_by_id=instance_evidence,
                min_contrast=min_contrast,
                sample_id=sample_id,
                scene_id=scene_of(sample_id),
                stem=stem,
                instance_map=instance_map,
                soft_instance_map=soft_instance_map,
                coverage=ownership_coverage,
                ambiguous=ambiguous,
                under_resolution=under_resolution,
                manifest_path=manifest_path,
                manifest_payload=manifest_payload,
                manifest_by_id=manifest_by_id,
                view_meta_path=find_view_metadata(npy_path, stem),
                profile=class_profile_name,
                coverage_threshold=aov_threshold,
                label_diagnostics=label_diagnostics,
            )
            (out_instance_records / f"{sample_id}.json").write_text(
                json.dumps(sidecar, indent=2)
            )

        if with_bboxes:
            h, w = label.shape[:2]
            if instance_map is not None:
                demoted_ids = under_resolution_ids | no_contrast_ids
                boxed_map = instance_map
                if demoted_ids or thin_ids:
                    boxed_map = instance_map.copy()
                    for inst_id in demoted_ids:
                        boxed_map[boxed_map == inst_id] = 0
                    # A hairline that passed the evidence gate still owns almost
                    # no majority-coverage pixels, so its extent has to come from
                    # the soft footprint or it gets no box at all. Only the box
                    # is widened - the segmentation mask keeps the strict 0.5
                    # threshold, and the 0.15-0.5 band stays in the ignore mask,
                    # so nothing here relabels part-wall pixels as defect.
                    if thin_ids and (
                        soft_instance_map is not None and ownership_coverage is not None
                    ):
                        extent = ownership_coverage >= float(thin_box_coverage)
                        for inst_id in thin_ids:
                            boxed_map[extent & (soft_instance_map == inst_id)] = inst_id
                boxes = compute_asset_bboxes_from_instances(
                    boxed_map,
                    manifest_by_id,
                    profile=class_profile_name,
                    min_area=BBOX_MIN_PIXEL_AREA,
                    evidence_by_id=instance_evidence,
                    min_evidence=float(min_instance_px),
                )
                if blister_group_gap > 0:
                    boxes = group_blister_boxes(
                        boxes,
                        boxed_map,
                        rgb,
                        blister_group_gap,
                        min_px=min_instance_px,
                        min_contrast=(
                            min_contrast
                            if blister_min_contrast is None
                            else blister_min_contrast
                        ),
                        min_side_frac=blister_min_side,
                    )
            else:
                boxes = (
                    compute_asset_bboxes_from_aovs(
                        aov_maps, rgb.shape[:2], aov_threshold
                    )
                    if aov_maps
                    else compute_asset_bboxes_from_material_passes(
                        index_map, materials, min_area=BBOX_MIN_PIXEL_AREA
                    )
                )
                boxes = remap_boxes(boxes, class_profile_name)
            write_bboxes(sample_id, boxes, h, w, out_bboxes, out_bboxes_yolo)

        return True
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
        return False


def collect_complete_sample_ids(out_images: Path, out_masks: Path) -> list[str]:
    """IDs that have both images/<id>.png and masks/<id>.png."""
    if not out_images.is_dir() or not out_masks.is_dir():
        return []
    out: list[str] = []
    for p in sorted(out_images.glob("*.png")):
        sid = p.stem
        if (out_masks / f"{sid}.png").is_file():
            out.append(sid)
    return out


# New ids include resample/frame lineage; accept the historical shorter form so
# existing exports continue to split by scene correctly.
_SAMPLE_ID_RE = re.compile(
    r"^(?P<scene>.+)_rig\d+(?:_resample\d+_frame\d+)?_camera_\d+$"
)


# sample id -> scene id, taken from each frame's own defect_instances.json.
# Populated during discovery; explicit lineage always beats parsing a path.
_EXPLICIT_SCENE: dict[str, str] = {}


def scene_of(sample_id: str) -> str:
    """The scene a sample came from, which is the unit a split may divide on.

    Prefers the scene id the manifest recorded at coarse time. Falling back to
    the sample id's shape is a guess about directory layout, and it breaks the
    moment each rig or randomisation pass is rendered into its own folder: the
    parse then reports one scene as many, and near-duplicate frames of the same
    room end up split across train and test, which is a leak that flatters every
    score computed from it.

    An id that parses as nothing is treated as its own scene: that over-groups
    nothing and keeps an unrecognised id out of two splits at once.
    """
    explicit = _EXPLICIT_SCENE.get(sample_id)
    if explicit:
        return explicit
    match = _SAMPLE_ID_RE.match(sample_id)
    return match.group("scene") if match else sample_id


def find_scene_output_dir(npy_path: Path) -> Path | None:
    """Directory that owns scene.blend, if the render tree still sits next to it.

    Manifests stamp scene_id as scene_{seed}, which collides across room types
    that reused the same seed (bed_1 and bath_1 are both scene_1). The output
    folder name is unique per solve and is the unit a split should divide on.
    """
    current = Path(npy_path).resolve().parent
    for _ in range(12):
        if (current / "scene" / "scene.blend").is_file() or (
            current / "scene.blend"
        ).is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def register_scene_lineage(sample_id: str, npy_path: Path) -> str | None:
    """Record the scene a sample belongs to for leak-free splitting."""
    output_dir = find_scene_output_dir(npy_path)
    if output_dir is not None:
        _EXPLICIT_SCENE[sample_id] = output_dir.name
        return output_dir.name
    manifest_path = find_defect_manifest(npy_path)
    if manifest_path is None:
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    scene_id = payload.get("scene_id")
    if scene_id:
        _EXPLICIT_SCENE[sample_id] = str(scene_id)
        return str(scene_id)
    return None


# A frame is worth training on if it shows a defect. Two kinds of frame do not,
# and they need opposite treatment.
#
# A room-trajectory view with no defect in it is a real negative: clean plaster,
# skirting, sockets and seams are exactly the surfaces a real test set is full
# of, and a model that never sees them loses precision on them. Keep some.
#
# A close-up rig aimed at a defect that was dropped, or that renders with no
# contrast, is not a negative - it is a photograph of nothing, framed as though
# something were there. Real inspection photos are never composed that way, so
# these teach the detector a framing that does not occur in the test
# distribution. Drop them outright.
UNUSABLE_LABEL_STATUSES_KEEP = ("visible", "under_resolution")


def _frame_is_positive(sample_id: str, out_bboxes_yolo: Path, out_masks: Path) -> bool:
    yolo = out_bboxes_yolo / f"{sample_id}.txt"
    if yolo.is_file():
        return bool(yolo.read_text().strip())
    mask = out_masks / f"{sample_id}.png"
    if not mask.is_file():
        return False
    return bool(np.asarray(PILImage.open(mask).convert("L")).any())


# Statuses that mean "a defect is genuinely in this image", whether or not it
# earned a box. A frame containing one of these must never be exported as an
# empty negative: the pixels are defect, the label would say background, and a
# correct detection gets scored as a false positive. That is worse than not
# using the frame at all.
DEFECT_PRESENT_STATUSES = ("visible", "under_resolution", "no_contrast")


def _frame_defect_presence(sample_id: str, out_instance_records: Path) -> str:
    """'none' | 'present' | 'failed_target' for one frame.

    'none'          - no instance records: a room view. A legitimate negative.
    'present'       - at least one instance is really in the image.
    'failed_target' - the rig was aimed at defects and every one of them was
                      dropped before it ever reached the shader, so the frame is
                      a close-up of nothing.
    """
    rec = out_instance_records / f"{sample_id}.json"
    if not rec.is_file():
        return "none"
    try:
        payload = json.loads(rec.read_text())
    except Exception:
        return "none"
    instances = payload.get("instances") or []
    if not instances:
        return "none"
    if any(i.get("label_status") in DEFECT_PRESENT_STATUSES for i in instances):
        return "present"
    return "failed_target"


def curate_sample_ids(
    ids: list[str],
    out_bboxes_yolo: Path,
    out_masks: Path,
    out_instance_records: Path,
    max_negative_fraction: float,
    drop_unusable_closeups: bool,
    seed: int,
) -> tuple[list[str], dict]:
    """Keep every positive, cap negatives, drop close-ups of nothing.

    Negatives are thinned rather than removed: `max_negative_fraction` is the
    share of the *returned* set they are allowed to occupy, so 0.15 means one
    negative for roughly every six positives. Selection is a stable hash of the
    sample id with the split seed, so re-running is reproducible and adding new
    scenes does not reshuffle the frames already chosen.
    """
    positives, negatives, unusable, unboxed = [], [], [], []
    for sid in ids:
        presence = _frame_defect_presence(sid, out_instance_records)
        if drop_unusable_closeups and presence == "failed_target":
            unusable.append(sid)
            continue
        if _frame_is_positive(sid, out_bboxes_yolo, out_masks):
            positives.append(sid)
        elif presence == "present":
            # Real defect, no box that met the evidence bar. Excluded from
            # training rather than handed over as background - see
            # DEFECT_PRESENT_STATUSES. The frame stays on disk, and its ignore
            # mask marks the pixels, so a trainer that understands ignore
            # regions can still use it.
            unboxed.append(sid)
        else:
            negatives.append(sid)

    f = max(0.0, min(0.95, float(max_negative_fraction)))
    if positives:
        budget = int(round(f / (1.0 - f) * len(positives))) if f < 1.0 else len(negatives)
    else:
        budget = len(negatives)
    budget = min(budget, len(negatives))

    ranked = sorted(
        negatives, key=lambda sid: int_hash_str(f"{seed}:{sid}")
    )
    kept_neg = ranked[:budget]
    kept = sorted(positives + kept_neg)
    report = {
        "input_frames": len(ids),
        "positives": len(positives),
        "negatives_available": len(negatives),
        "negatives_kept": len(kept_neg),
        "unusable_closeups_dropped": len(unusable),
        "defect_present_but_unboxed_excluded": len(unboxed),
        "kept_frames": len(kept),
        "negative_fraction": (len(kept_neg) / len(kept)) if kept else 0.0,
    }
    return kept, report


def int_hash_str(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def split_train_val_test(
    ids: list[str], seed: int, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[list[str], list[str], list[str]]:
    """Split ~70% / 15% / 15% by SCENE, so no scene spans two splits.

    Splitting on individual sample ids leaks: every camera rig and subcam of one
    scene renders the same room, the same materials and the same lighting, and
    the defect close-up rigs often frame the same defect from slightly different
    positions. Shuffling ids puts near-duplicates of a test frame into train,
    and the reported test score then measures memorisation rather than transfer.
    Grouping by scene is the whole point of the function, so it groups first and
    shuffles scenes, never samples.

    Scenes are assigned greedily to whichever split is furthest below its target
    share of SAMPLES, rather than splitting the scene list 70/15/15, because
    scenes contribute unequal numbers of frames - a scene with 12 defect rigs
    against one with 2 - and splitting scene counts would leave the sample
    counts lopsided.
    """
    if not ids:
        return [], [], []

    by_scene: dict[str, list[str]] = {}
    for sample_id in ids:
        by_scene.setdefault(scene_of(sample_id), []).append(sample_id)

    scenes = sorted(by_scene)
    if len(scenes) < 3:
        print(
            f"WARNING: only {len(scenes)} scene(s) in {len(ids)} sample(s). A "
            "leak-free split needs at least one scene per split, so some splits "
            "will be empty. Render more scenes before trusting any score from "
            "these splits."
        )

    rng = np.random.default_rng(seed)
    order = list(scenes)
    rng.shuffle(order)

    total = len(ids)
    targets = {
        "train": train_frac * total,
        "val": val_frac * total,
        "test": max(0.0, (1.0 - train_frac - val_frac)) * total,
    }
    chosen: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for scene in order:
        # Largest deficit first; ties broken in a fixed order so the result is
        # reproducible for a given seed.
        name = max(
            ("train", "val", "test"),
            key=lambda k: (
                targets[k] - len(chosen[k]),
                -["train", "val", "test"].index(k),
            ),
        )
        chosen[name].extend(by_scene[scene])

    return (
        sorted(chosen["train"]),
        sorted(chosen["val"]),
        sorted(chosen["test"]),
    )


def write_class_names_legend(output_dir: Path, profile: str = "release") -> Path:
    """Write ``class_names.txt`` (short semantic names + decode hints). Shared by full export and single-sample."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legend_path = output_dir / "class_names.txt"
    mapping, names = class_profile(profile)
    mask_step = max(1, 180 // (len(names) - 1))
    half = mask_step // 2
    class_lines = "".join(f"{cid} {name}\n" for cid, name in sorted(names.items()))
    mapping_line = ", ".join(
        f"{legacy}->{training}" for legacy, training in sorted(mapping.items())
    )
    legend_path.write_text(
        class_lines
        + f"# Class profile: {profile}; legacy->training: {mapping_line}\n"
        + f"# Mask PNG: gray level = class_id * {mask_step} (class 0..{len(names) - 1})\n"
        f"# Decode: label = ((mask + {half}) // {mask_step}).clip(0, {len(names) - 1})\n"
        "# Legacy 4-class masks only use {0,85,170,255}: label = (mask // 85).clip(0,3)\n"
        "# Instance masks: instance_masks/<id>.npz with uint32 instance_id, float16\n"
        "#   ownership_coverage, ambiguous, and under_resolution arrays.\n"
        "# 2D bboxes: per-image JSON — one visible box per decoded instance when available;\n"
        "#   boxes[{instance_id, class_id, legacy_class_id, class_name, x_min, y_min, x_max,\n"
        "#   y_max, width, height, area}] (pixel coords, inclusive).\n"
        "# YOLO labels subtract one from contiguous training ids; background is not an object class.\n"
    )
    return legend_path


DEFAULT_INPUT_FOLDER = Path("/mnt/nvme_storage/dataset/all_frames")
DEFAULT_OUTPUT_FOLDER = Path("/mnt/nvme_storage/dataset/defect_segmentation_dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Build defect segmentation dataset from Infinigen frames (RGB + MaterialSegmentation + Materials JSON)."
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        type=Path,
        default=DEFAULT_INPUT_FOLDER,
        help=(
            "Root directory: single scene, or parent of many scenes "
            "(e.g. all_frames with bathroom01/rig0_rs0/Image/...). "
            f"Default: {DEFAULT_INPUT_FOLDER}"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=Path,
        default=DEFAULT_OUTPUT_FOLDER,
        help=(
            "Output directory for images/, masks/, splits/. "
            f"Default: {DEFAULT_OUTPUT_FOLDER}"
        ),
    )
    parser.add_argument(
        "--thin-box-coverage",
        type=float,
        default=THIN_BOX_COVERAGE,
        help=(
            "Coverage at which a pixel counts toward a thin instance's bounding "
            f"box (default {THIN_BOX_COVERAGE}). Segmentation masks are "
            "unaffected and keep the strict 0.5 threshold. Set to 0.5 to "
            "restore the old behaviour, where a hairline got no box."
        ),
    )
    parser.add_argument(
        "--max-negative-fraction",
        type=float,
        default=0.15,
        help=(
            "Share of the exported set allowed to be frames with no defect box "
            "(default 0.15). Negatives are what teach the model that clean "
            "plaster, skirting and sockets are not defects, so some are needed - "
            "but the real test set is phone close-ups OF a defect, so a set that "
            "is mostly empty rooms trains for the wrong distribution. "
            "Positives are never dropped; only negatives are thinned. Use 1.0 to "
            "keep every frame."
        ),
    )
    parser.add_argument(
        "--keep-unusable-closeups",
        action="store_true",
        help=(
            "Keep close-up frames whose every instance was dropped or rendered "
            "with no contrast. Off by default: those are photographs of nothing "
            "composed as though a defect were there, a framing that never occurs "
            "in real inspection photos."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split (default: 42).",
    )
    parser.add_argument(
        "--with-bboxes",
        action="store_true",
        help=(
            "Also write bboxes/, bboxes_yolo/, and annotations_coco.json (detection sidecar). "
            "Default is images/ + masks/ + splits/ + class_names.txt only."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-write outputs even when target files already exist "
            "(needed after changing DEFECT_PREFIXES or bbox rules; otherwise skipped samples stay stale)."
        ),
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Only refresh splits/class_names from existing images/ and masks/ (no input scan).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (e.g. for logs).",
    )
    parser.add_argument(
        "--realism-postprocess",
        action="store_true",
        help=(
            "Desaturate, brighten, and add grain to exported RGB "
            "(fast ablation without re-rendering; see REALISM_IMPROVEMENTS.md)."
        ),
    )
    parser.add_argument(
        "--saturation-scale",
        type=float,
        default=0.35,
        help="HSV saturation multiplier when using --realism-postprocess (default: 0.35).",
    )
    parser.add_argument(
        "--brightness-scale",
        type=float,
        default=1.15,
        help="Value multiplier when using --realism-postprocess (default: 1.15).",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=4.0,
        help="Gaussian grain sigma in 0–255 units when using --realism-postprocess (default: 4).",
    )
    parser.add_argument(
        "--aov-masks",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Use per-class defect AOV coverage passes for masks. 'auto' uses them "
            "when the render produced them and falls back to the material-index "
            "pass otherwise; 'on' requires them; 'off' always uses material index."
        ),
    )
    parser.add_argument(
        "--aov-threshold",
        type=float,
        default=AOV_COVERAGE_THRESHOLD,
        help=(
            "Coverage at which an AOV pixel counts as defect (default: "
            f"{AOV_COVERAGE_THRESHOLD}). Lower it toward 0.25 to keep hairline "
            "branches that never reach half coverage, at the cost of precision."
        ),
    )
    parser.add_argument(
        "--instance-masks",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Decode per-instance masks from the three EXR moment AOVs. 'on' "
            "requires the complete triplet and defect_instances.json."
        ),
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=MIN_INSTANCE_CONTRAST_CODES,
        help=(
            "Least contrast in 8-bit code values for an instance to count as a "
            "visible positive. Below it the instance becomes an ignore region "
            "with no box: coverage says a defect is there, contrast says nothing "
            "in the image shows it. 0 disables."
        ),
    )
    parser.add_argument(
        "--blister-group-gap",
        type=float,
        default=BLISTER_GROUP_GAP_FRAC,
        help=(
            "Box blisters per visible group instead of per generated cluster: "
            "pieces of one blister instance closer than this fraction of the "
            "image's shorter side share a box. 0 (default) keeps one box per "
            "instance. Needs instance masks."
        ),
    )
    parser.add_argument(
        "--blister-min-contrast",
        type=float,
        default=None,
        help=(
            "Contrast gate for each blister group when --blister-group-gap is "
            "set. Defaults to --min-contrast."
        ),
    )
    parser.add_argument(
        "--blister-min-side",
        type=float,
        default=0.0,
        help=(
            "Smallest blister group box, as sqrt(w*h) over the image's shorter "
            "side, when --blister-group-gap is set. 0 disables."
        ),
    )
    parser.add_argument(
        "--under-resolution-floor",
        type=float,
        default=UNDER_RESOLUTION_IGNORE_FLOOR,
        help=(
            "Coverage above which an under-resolved defect becomes an ignore "
            "region instead of background. Below it the defect leaves no trace "
            "a person would annotate and its pixels stay background, matching "
            "what a human labeller (and the real test set) would do. Pass a "
            "negative value to disable and label every under-resolved pixel "
            "background, which is the pre-policy behaviour."
        ),
    )
    parser.add_argument(
        "--min-instance-px",
        type=int,
        default=MIN_INSTANCE_LABEL_PX,
        help=(
            "Smallest labelled area in pixels that is emitted as a positive. "
            "Instances below it become ignore regions with no box - there is "
            "evidence, just not enough to train on. 0 disables."
        ),
    )
    parser.add_argument(
        "--class-profile",
        choices=sorted(CLASS_PROFILES),
        default="release",
        help=(
            "Training taxonomy. 'release' maps legacy ids 1/2/4 to contiguous "
            "ids 1/2/3; 'legacy' preserves all historical ids."
        ),
    )

    args = parser.parse_args()

    input_root = args.input_folder.resolve()
    output_dir = args.output_folder.resolve()
    out_images = output_dir / "images"
    out_masks = output_dir / "masks"
    out_bboxes = output_dir / "bboxes"
    out_bboxes_yolo = output_dir / "bboxes_yolo"
    out_instance_masks = output_dir / "instance_masks"
    out_instance_records = output_dir / "instances"
    out_ignore_masks = output_dir / "ignore_masks"

    samples: list[tuple[Path, Path, Path, str]] = []
    if not args.splits_only:
        samples = discover_scene_frames(input_root)
        if not samples:
            existing = collect_complete_sample_ids(out_images, out_masks)
            if existing:
                print(
                    f"No frame triplets under {input_root}; "
                    f"refreshing splits from {len(existing)} existing image+mask pairs."
                )
            else:
                print(
                    f"No frame triplets found under {input_root} and no existing outputs."
                )
                return 1
        else:
            print(f"Found {len(samples)} samples (input: {input_root}).")
            skipped = 0
            success = 0
            failed = 0
            iterator = (
                samples
                if args.no_progress
                else tqdm(samples, desc="Processing samples", unit="sample")
            )
            for img_path, npy_path, json_path, sample_id in iterator:
                img_out = out_images / f"{sample_id}.png"
                mask_out = out_masks / f"{sample_id}.png"
                bbox_out = out_bboxes / f"{sample_id}.json"
                yolo_out = out_bboxes_yolo / f"{sample_id}.txt"
                complete = img_out.is_file() and mask_out.is_file()
                if args.with_bboxes:
                    complete = complete and bbox_out.is_file() and yolo_out.is_file()
                stem = npy_path.stem.replace("MaterialSegmentation_", "", 1)
                source_has_instances = bool(
                    find_instance_aov_maps(npy_path, stem)
                    if args.instance_masks != "off"
                    else False
                )
                if args.instance_masks == "on" or source_has_instances:
                    complete = (
                        complete
                        and (out_instance_masks / f"{sample_id}.npz").is_file()
                        and (out_instance_records / f"{sample_id}.json").is_file()
                        and (out_ignore_masks / f"{sample_id}.png").is_file()
                    )
                if not args.force and complete:
                    skipped += 1
                    continue
                if process_sample(
                    img_path,
                    npy_path,
                    json_path,
                    sample_id,
                    out_images,
                    out_masks,
                    out_bboxes,
                    out_bboxes_yolo,
                    with_bboxes=args.with_bboxes,
                    realism_postprocess=args.realism_postprocess,
                    saturation_scale=args.saturation_scale,
                    brightness_scale=args.brightness_scale,
                    noise_sigma=args.noise_sigma,
                    aov_masks=args.aov_masks,
                    aov_threshold=args.aov_threshold,
                    instance_masks=args.instance_masks,
                    class_profile_name=args.class_profile,
                    min_instance_px=args.min_instance_px,
                    thin_box_coverage=args.thin_box_coverage,
                    min_contrast=args.min_contrast,
                    blister_group_gap=args.blister_group_gap,
                    blister_min_contrast=args.blister_min_contrast,
                    blister_min_side=args.blister_min_side,
                    under_resolution_floor=(
                        None
                        if args.under_resolution_floor < 0
                        else args.under_resolution_floor
                    ),
                    out_instance_masks=out_instance_masks,
                    out_instance_records=out_instance_records,
                    out_ignore_masks=out_ignore_masks,
                ):
                    success += 1
                else:
                    failed += 1
            print(
                f"Wrote {success} new/updated samples, skipped {skipped} already present "
                f"(of {len(samples)} from input) -> {output_dir}"
            )
            if skipped > 0 and not args.force:
                print(
                    "Hint: skipped rows kept old outputs. "
                    "Use --force to regenerate everything (e.g. after adding SpallingPlugMaterial)."
                )
            if failed:
                print(
                    f"FAILED: {failed} sample(s) had incomplete or invalid ground truth"
                )
                return 1
    else:
        print("--splits-only: skipping input processing.")

    ids = collect_complete_sample_ids(out_images, out_masks)
    if not ids:
        print(f"No complete image+mask pairs under {output_dir}")
        return 1

    ids, curation = curate_sample_ids(
        ids,
        out_bboxes_yolo,
        out_masks,
        out_instance_records,
        max_negative_fraction=args.max_negative_fraction,
        drop_unusable_closeups=not args.keep_unusable_closeups,
        seed=args.seed,
    )
    if not ids:
        print("Every frame was filtered out; relax --max-negative-fraction.")
        return 1
    print(
        f"Curated {curation['input_frames']} -> {curation['kept_frames']} frames: "
        f"{curation['positives']} with a defect, "
        f"{curation['negatives_kept']}/{curation['negatives_available']} negatives kept "
        f"({curation['negative_fraction']*100:.1f}% of the set), "
        f"{curation['unusable_closeups_dropped']} close-up(s) of dropped "
        f"targets discarded, "
        f"{curation['defect_present_but_unboxed_excluded']} frame(s) with a real "
        f"defect but no qualifying box held out of training (not used as negatives)."
    )
    (output_dir / "curation.json").write_text(json.dumps(curation, indent=2) + "\n")

    train_ids, val_ids, test_ids = split_train_val_test(ids, args.seed)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(val_ids) + "\n")
    (splits_dir / "test.txt").write_text("\n".join(test_ids) + "\n")
    n_scenes = {
        name: len({scene_of(i) for i in group})
        for name, group in (
            ("train", train_ids),
            ("val", val_ids),
            ("test", test_ids),
        )
    }
    print(
        f"Wrote splits: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test "
        f"-> {splits_dir}"
    )
    print(
        f"  scenes: {n_scenes['train']} train, {n_scenes['val']} val, "
        f"{n_scenes['test']} test (split by scene; no scene spans two splits)"
    )

    legend_path = write_class_names_legend(output_dir, profile=args.class_profile)
    print(f"Class legend: {legend_path}")

    if args.with_bboxes:
        write_coco_aggregate(
            output_dir, out_bboxes, ids, class_profile_name=args.class_profile
        )

    return 0


def write_coco_aggregate(
    output_dir: Path,
    out_bboxes: Path,
    ids: list[str],
    class_profile_name: str = "release",
) -> None:
    """Combine per-image bbox JSONs into a single COCO-format annotations.json."""
    if not out_bboxes.is_dir():
        return

    images = []
    annotations = []
    ann_id = 1
    for image_id, sid in enumerate(ids, start=1):
        bbox_file = out_bboxes / f"{sid}.json"
        if not bbox_file.is_file():
            continue
        try:
            data = json.loads(bbox_file.read_text())
        except Exception:
            continue
        images.append(
            {
                "id": image_id,
                "file_name": f"{sid}.png",
                "width": data.get("width", 0),
                "height": data.get("height", 0),
            }
        )
        for b in data.get("boxes", []):
            w = b["x_max"] - b["x_min"] + 1
            h = b["y_max"] - b["y_min"] + 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(b["class_id"]),
                    "bbox": [int(b["x_min"]), int(b["y_min"]), int(w), int(h)],
                    "area": int(b.get("area", w * h)),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    _, names = class_profile(class_profile_name)
    categories = [{"id": cid, "name": names[cid]} for cid in sorted(names) if cid != 0]
    coco = {
        "info": {
            "description": "Infinigen defect 2D bboxes (one box per material pass_index / asset)"
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_path = output_dir / "annotations_coco.json"
    coco_path.write_text(json.dumps(coco))
    print(
        f"Wrote COCO aggregate: {len(images)} images, {len(annotations)} boxes -> {coco_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

# Usage:
#   # Defaults: -i /mnt/nvme_storage/dataset/all_frames, -o /mnt/nvme_storage/dataset/defect_segmentation_dataset
#   python prepare_defect_annotated_dataset.py
#
#   python prepare_defect_annotated_dataset.py -i /path/to/all_frames -o /path/to/out
#
# Output: <output-folder>/images/*.png, masks/*.png, splits/train.txt, val.txt, test.txt, class_names.txt
# Splits are grouped by scene: every rig/subcam of one scene lands in the same split.
# Add --with-bboxes for bboxes/, bboxes_yolo/, annotations_coco.json
# Re-run: skips existing pairs unless --force; always rewrites splits from all complete pairs.
# Splits only: python prepare_defect_annotated_dataset.py --splits-only -o <out>
