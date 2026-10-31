"""Read fp_probe_render multilayer EXRs: shadow masks, light edges, object index."""

import cv2
import numpy as np
import OpenEXR
import Imath


def read(path):
    f = OpenEXR.InputFile(str(path))
    dw = f.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    chans = list(f.header()["channels"])

    def ch(suffix):
        name = next(c for c in chans if c.endswith(suffix))
        return np.frombuffer(f.channel(name, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(h, w)

    def lum(p):
        return 0.2126 * ch(f"{p}.R") + 0.7152 * ch(f"{p}.G") + 0.0722 * ch(f"{p}.B")

    return {"lum": lum("Combined"), "direct": lum("DiffDir"), "index": np.round(ch("IndexOB.X")).astype(np.uint8)}


def _blur(x, r):
    return cv2.GaussianBlur(x, (2 * r + 1, 2 * r + 1), 0)


def probe_maps(shadow_exr, noshadow_exr, ratio=0.6):
    """cast: darkened >40% by a prop/downpipe/eave/corner shadow; index: object
    pass index (see fp_probe_render). DiffDir is also in the files, but a light
    edge map from it is not usable: on brick it marks all mortar relief and the
    32-sample noise, not shadow boundaries."""
    s, n = read(shadow_exr), read(noshadow_exr)
    ls, ln = _blur(s["lum"], 2), _blur(n["lum"], 2)
    cast = (ls < ratio * np.maximum(ln, 1e-6)) & (ln > 1e-4)
    return {"cast": cast, "index": s["index"]}


def convert(batch_dir):
    """Write <rig>_maps.npz (cast, index) beside each EXR pair, for envs without OpenEXR."""
    from pathlib import Path

    n = 0
    for sh in sorted(Path(batch_dir).glob("shadow_probe/seed_*/*_shadow.exr")):
        rig = sh.name.split("_")[0]
        out = sh.with_name(f"{rig}_maps.npz")
        if out.exists():
            continue
        m = probe_maps(sh, sh.with_name(f"{rig}_noshadow.exr"))
        np.savez_compressed(out, cast=m["cast"], index=m["index"])
        n += 1
    return n


if __name__ == "__main__":
    import sys

    for d in sys.argv[1:]:
        print(d, convert(d), "converted")
