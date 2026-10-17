"""Side-by-side defect crops: Blender vs Isaac (path traced, real-time).

    python figure.py <root> <isaac dir> CAM:RIG:LABEL [...]

Columns: raking light from the left, right and top; then (L - R) / (L + R), the
part of the shading that flips when the light changes side, which only real
relief (geometry or bump) produces; then Isaac's semantic mask over the image.
Every crop uses the same display transform: divide by the crop's median, x0.5,
gamma 2.2. Nothing is stretched per image.
"""
import json
import sys
from pathlib import Path

import cv2
import Imath
import numpy as np
import OpenEXR

root, idir = Path(sys.argv[1]), Path(sys.argv[2])
CONDS = ("rake_left", "rake_right", "rake_top")
T = 300


def exr_rgb(path):
    f = OpenEXR.InputFile(str(path))
    dw = f.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    keys = f.header()["channels"]
    pick = lambda c: next(k for k in keys if k.endswith(f"Combined.{c}"))  # noqa: E731
    return np.stack([np.frombuffer(f.channel(pick(c), Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
                     .reshape(h, w) for c in "RGB"], -1)


def lum(x):
    return x @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def show(x):
    y = np.clip(x / max(np.median(lum(x)), 1e-8) * 0.5, 0, 1) ** (1 / 2.2)
    return cv2.resize(cv2.cvtColor((y * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), (T, T), interpolation=cv2.INTER_AREA)


def flip(l, r):
    a, b = lum(l) / np.median(lum(l)), lum(r) / np.median(lum(r))
    d = (a - b) / np.maximum(a + b, 1e-6)
    d = np.clip(d / 0.3, -1, 1)
    rgb = np.stack([np.clip(-d, 0, 1), np.zeros_like(d), np.clip(d, 0, 1)], -1) * 255  # BGR: blue = lit by left
    g = (1 - np.abs(d))[..., None] * 128
    return cv2.resize((rgb + g).astype(np.uint8), (T, T), interpolation=cv2.INTER_AREA)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
    return img


blocks = []
for spec in sys.argv[3:]:
    cam, rig, name = spec.split(":")
    mask = cv2.imread(str(next((root / "ref_pack" / "masks").glob(f"seed_*_rig{int(rig)}_resample0_frame0001_camera_0.png"))), 0) > 0
    ys, xs = np.where(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.65) + 20
    y0, x0 = max(cy - half, 0), max(cx - half, 0)
    sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
    sources = {"Blender Cycles": {c: exr_rgb(root / "blender_ref2" / f"{cam}_{c}.exr")[sl] for c in CONDS}}
    for mode, title in (("pt", "Isaac path traced"), ("rt", "Isaac RTX real-time")):
        sources[title] = {c: np.load(idir / f"{cam}_{c}_{mode}.npy").astype(np.float32)[sl] for c in CONDS}
    sem = np.load(idir / f"{cam}_semantic.npy")[sl]
    labels = json.loads((idir / f"{cam}_semantic_labels.json").read_text())
    ids = [int(k) for k, v in labels.items() if v.get("class") not in ("BACKGROUND", "UNLABELLED", "wall")]
    rows = []
    for title, imgs in sources.items():
        tiles = [label(show(imgs[c]), f"{title}: {c.replace('_', ' ')}") for c in CONDS]
        tiles.append(label(flip(imgs["rake_left"], imgs["rake_right"]), "shading flip L vs R"))
        if title.startswith("Isaac RTX"):
            base = show(imgs["rake_top"])
            m = cv2.resize(np.isin(sem, ids).astype(np.uint8), (T, T), interpolation=cv2.INTER_NEAREST) > 0
            base[m] = (0.4 * base[m] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            tiles.append(label(base, f"Isaac segmentation: {name}"))
        else:
            tiles.append(np.full((T, T, 3), 255, np.uint8))
        rows.append(np.concatenate(tiles, 1))
    head = np.full((28, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"{name} ({cam}, crop {2 * half}px of 960)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
    blocks.append(np.concatenate([head] + rows, 0))
out = root / "isaac_defect_figure.png"
cv2.imwrite(str(out), np.concatenate(blocks, 0))
print(out)
