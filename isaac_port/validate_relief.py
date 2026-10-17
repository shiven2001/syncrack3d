"""Does the baked height match the relief Cycles actually rendered?

    python validate_relief.py <root> CAM:RIG [CAM:RIG ...]

For each camera, reads the rendered world Position pass (blender_ref2), takes the
surface's offset along the wall's outward normal (-Y) relative to the wall's
median, samples the baked height at the same world (x, z), and compares them
inside the dilated defect mask: percentiles, correlation and the least-squares
slope (rendered = slope * baked).
"""
import json
import sys
from pathlib import Path

import cv2
import Imath
import numpy as np
import OpenEXR

root = Path(sys.argv[1])
info = json.loads((root / "patches" / "patch.json").read_text())
patches = {p["camera"]: p for p in info["patches"]}


def channel(path, name):
    f = OpenEXR.InputFile(str(path))
    dw = f.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    key = next(k for k in f.header()["channels"] if k.endswith(name))
    return np.frombuffer(f.channel(key, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(h, w)


result = {}
for spec in sys.argv[2:]:
    cam, rig = spec.split(":")
    exr = root / "blender_ref2" / f"{cam}_frontal.exr"
    x, y, z = (channel(exr, f"Position.{c}") for c in "XYZ")
    p = patches[cam]
    h = np.load(root / "patches" / f"{cam}_height_mm.npy")
    hit = h > 1.0
    base = np.median(h[hit])
    N = h.shape[0]
    s, (cx, _, cz) = p["size_m"], p["center"]
    col = np.clip(((x - (cx - s / 2)) / s) * (N - 1), 0, N - 1).astype(int)
    row = np.clip((1 - (z - (cz - s / 2)) / s) * (N - 1), 0, N - 1).astype(int)
    baked = h[row, col] - base
    valid = hit[row, col] & (np.abs(y) < 0.2)
    rendered = -(y - np.median(y[valid])) * 1000.0
    mask = cv2.imread(str(next((root / "ref_pack" / "masks").glob(f"seed_*_rig{int(rig)}_resample0_frame0001_camera_0.png"))), 0) > 0
    region = (cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & valid
    r, b = rendered[region], baked[region]
    slope = float((r * b).sum() / max((b * b).sum(), 1e-12))
    result[cam] = dict(pixels=int(region.sum()),
                       rendered_mm_p1_p99=[round(float(np.percentile(r, q)), 3) for q in (1, 99)],
                       baked_mm_p1_p99=[round(float(np.percentile(b, q)), 3) for q in (1, 99)],
                       rendered_extreme_mm=round(float(r[np.argmax(np.abs(r))]), 3),
                       baked_extreme_mm=round(float(b[np.argmax(np.abs(b))]), 3),
                       corr=round(float(np.corrcoef(r, b)[0, 1]), 3), slope=round(slope, 3))
(root / "relief_validation.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
