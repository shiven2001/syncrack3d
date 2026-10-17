"""Convert a patches dir's baked EXRs to the .npy maps the Isaac side reads.

    python exr_to_npy.py patches_dir

The Isaac Sim environment has no OpenEXR, so build_panels.py and
isaac_patch_render.py read .npy: <cam>_color.npy (H, W, 3) linear RGB, and
(H, W) float32 for roughness and each class coverage (DefectCrack,
DefectPaintPeel, DefectPaintBubble, DefectInstanceCoverage). The shader-bake
height EXR is kept as <cam>_height_mm_shaderbake.npy for reference only; the
true relief <cam>_height_mm.npy comes from bake_patch_height_ortho.py and is
never overwritten here. Run with an interpreter that has OpenEXR (the bpy env).
"""

import sys
from pathlib import Path

import Imath
import numpy as np
import OpenEXR

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)


def read(path, channels):
    f = OpenEXR.InputFile(str(path))
    dw = f.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    have = list(f.header()["channels"])
    out = []
    for c in channels:
        key = next(k for k in have if k == c or k.endswith("." + c))
        out.append(np.frombuffer(f.channel(key, FLOAT), np.float32).reshape(h, w))
    return out[0] if len(out) == 1 else np.stack(out, -1)


def main():
    pdir = Path(sys.argv[1])
    n = 0
    for exr in sorted(pdir.glob("*.exr")):
        stem = exr.stem
        if stem.endswith("_position"):
            continue
        if stem.endswith("_color"):
            arr, dst = read(exr, ["R", "G", "B"]), pdir / f"{stem}.npy"
        elif stem.endswith("_height_mm"):
            arr, dst = read(exr, ["R"]), pdir / f"{stem}_shaderbake.npy"
        else:
            arr, dst = read(exr, ["R"]), pdir / f"{stem}.npy"
        np.save(dst, arr.astype(np.float32))
        n += 1
    print(f"converted {n} EXRs in {pdir}")


if __name__ == "__main__":
    main()
