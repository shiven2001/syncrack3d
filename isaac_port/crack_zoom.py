"""Zoomed crack crops and the groove's light-side signature, Blender vs Isaac.

    python crack_zoom.py <root> <isaac dir> CAM RIG

Crop: 120 px around the middle of the crack, enlarged 3x, same display
transform for every tile. Signature: along each crack row, find the
groove bottom (darkest pixel near the mask under the top light) in that image,
then mean normalised luminance 1-3 px right of it minus 1-3 px left of it. A groove
lit from the left has its right wall facing the light, so the signature should
be positive under rake_left and negative under rake_right, i.e. flip sign.
"""
import sys
from pathlib import Path

import cv2
import Imath
import numpy as np
import OpenEXR

root, idir, cam, rig = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
CONDS = ("rake_left", "rake_right", "rake_top")


def exr(path):
    f = OpenEXR.InputFile(str(path))
    dw = f.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    ks = f.header()["channels"]
    g = lambda c: np.frombuffer(f.channel(next(k for k in ks if k.endswith("Combined." + c)),  # noqa: E731
                                          Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(h, w)
    return np.stack([g(c) for c in "RGB"], -1)


lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722], np.float32)  # noqa: E731
mask = cv2.imread(str(next((root / "ref_pack" / "masks").glob(f"seed_*_rig{rig}_resample0_frame0001_camera_0.png"))), 0) > 0
ys, xs = np.where(mask)
order = np.argsort(ys)
cy, cx = ys[order[len(order) // 2]], xs[order[len(order) // 2]]
H = 60
sl = (slice(cy - H, cy + H), slice(cx - H, cx + H))
m = mask[sl]
sources = {"Blender Cycles": {c: exr(root / "blender_ref2" / f"{cam}_{c}.exr") for c in CONDS}}
for mode, title in (("pt", "Isaac path traced"), ("rt", "Isaac RTX real-time")):
    sources[title] = {c: np.load(idir / f"{cam}_{c}_{mode}.npy").astype(np.float32) for c in CONDS}

rows, report = [], []
for title, imgs in sources.items():
    tiles, sig = [], {}
    for c in CONDS:
        crop = imgs[c][sl]
        L = lum(crop) / np.median(lum(crop))
        disp = np.clip(crop / np.median(lum(crop)) * 0.5, 0, 1) ** (1 / 2.2)
        t = cv2.resize(cv2.cvtColor((disp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), (360, 360),
                       interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"{title}: {c.replace('_', ' ')}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
        cv2.putText(t, f"{title}: {c.replace('_', ' ')}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        tiles.append(t)
        diffs = []
        # Locate the groove bottom in this image itself (darkest pixel near the
        # mask in the top-lit render), so a 1 px registration offset between
        # renderers cannot fake or hide the flip.
        Lt = lum(imgs["rake_top"][sl])
        for r in range(m.shape[0]):
            cols = np.where(m[r])[0]
            if len(cols) == 0:
                continue
            lo, hi = max(cols.min() - 3, 3), min(cols.max() + 4, m.shape[1] - 4)
            if hi <= lo:
                continue
            cb = lo + int(np.argmin(Lt[r, lo:hi]))
            diffs.append(L[r, cb + 1:cb + 4].mean() - L[r, cb - 3:cb].mean())
        sig[c] = float(np.mean(diffs)) if diffs else float("nan")
    rows.append(np.concatenate(tiles, 1))
    report.append(f"{title:20s} right-minus-left wall: rake_left {sig['rake_left']:+.3f}  "
                  f"rake_right {sig['rake_right']:+.3f}  (flips sign: {np.sign(sig['rake_left']) != np.sign(sig['rake_right'])})")
cv2.imwrite(str(root / "crack_zoom.png"), np.concatenate(rows, 0))
print("\n".join(report))
print(root / "crack_zoom.png")
