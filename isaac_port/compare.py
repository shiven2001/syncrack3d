"""Blender vs Isaac comparison sheets and defect-region statistics.

    python compare.py <root> <isaac out dir name> [<isaac out dir name> ...]

Per close-up camera: a sheet with rows Blender / each Isaac render and columns
the light conditions, each image normalised by its median (wall) luminance so
absolute light units do not matter. Statistics inside the exported defect mask
(dilated 2 px) against a surrounding ring:
  contrast   = std of normalised luminance inside the defect region
  flip       = mean |L(rake_left) - L(rake_right)| inside, i.e. how much the
               defect's shading responds to light direction
  corr       = Pearson correlation of the defect-region luminance with Blender
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import Imath

root = Path(sys.argv[1])
isaac_dirs = sys.argv[2:]
plan = json.loads((root / "blender_ref" / "plan.json").read_text())
conds = list(plan["views"][0]["lights"])


def read_exr(path):
    f = OpenEXR.InputFile(str(path))
    dw = f.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    ch = [np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(h, w)
          for c in "RGB"]
    return np.stack(ch, -1)


def lum(img):
    return img @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def norm(img):
    m = np.median(lum(img))
    return img / max(m, 1e-8) * 0.18


def show(img):
    x = np.clip(norm(img) * 2.0, 0, 1) ** (1 / 2.2)
    return cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


stats = {}
for vi, view in enumerate(plan["views"]):
    cam = view["camera"]
    rig = int(cam.split("_")[2])
    mask_png = next((root / "ref_pack" / "masks").glob(f"seed_*_rig{rig}_resample0_frame0001_camera_0.png"))
    mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE) > 0
    inside = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    sources = {"blender": {c: read_exr(root / "blender_ref" / f"{cam}_{c}.exr") for c in conds}}
    for d in isaac_dirs:
        for mode in ("rt", "pt"):
            files = {c: root / d / f"{cam}_{c}_{mode}.npy" for c in conds}
            if all(f.exists() for f in files.values()):
                sources[f"{d}:{mode}"] = {c: np.load(f).astype(np.float32) for c, f in files.items()}
    rows = []
    for name, imgs in sources.items():
        tiles = []
        for c in conds:
            t = cv2.resize(show(imgs[c]), (320, 320), interpolation=cv2.INTER_AREA)
            cnt, _ = cv2.findContours(cv2.resize(mask.astype(np.uint8), (320, 320)), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(t, cnt, -1, (0, 0, 255), 1)
            cv2.putText(t, f"{name} {c}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
            tiles.append(t)
        rows.append(np.concatenate(tiles, 1))
        L = {c: lum(norm(imgs[c])) for c in conds}
        ref = {c: lum(norm(sources["blender"][c])) for c in conds}
        s = dict(
            contrast={c: round(float(L[c][inside].std()), 4) for c in conds},
            flip=round(float(np.abs(L["rake_left"] - L["rake_right"])[inside].mean()), 4) if inside.any() else None,
            corr={c: round(float(np.corrcoef(L[c][inside], ref[c][inside])[0, 1]), 3) for c in conds}
            if inside.sum() > 10 else None,
        )
        stats.setdefault(cam, {})[name] = s
    cv2.imwrite(str(root / f"sheet_{cam}.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])

(root / "compare_stats.json").write_text(json.dumps(stats, indent=1))
for cam, per in stats.items():
    print(cam)
    for name, s in per.items():
        print(f"  {name:18s} contrast {s['contrast']} flip {s['flip']} corr {s['corr']}")
