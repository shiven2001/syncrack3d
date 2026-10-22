"""Run the crack detector on embodied room views and attribute its false alarms.

    python isaac_port/room_false_alarms.py --model MODEL_DIR --out OUT VIEWS_DIR [VIEWS_DIR ...]

VIEWS_DIR is render_room_views.py output. Each linear render goes through the
training camera model (grey-world white balance at strength .25, auto-exposure
metered to 0.18 grey at +0.85 EV - the middle of the release range - sRGB, then
syncrack3d.isp noise and JPEG), then every runs/s*/last.pt under MODEL_DIR
predicts it (protocol v2: 1.5x test-time upscale); a pixel is crack when at
least half the seeds say so. Predictions farther than 5 px from a labelled crack
are false alarms.

Each false-alarm blob gets one cause, checked in this order:
  cast shadow     (with --shadows) at least half its pixels lie on a wall, floor or
                  ceiling within 3 px of a cast-shadow boundary
  object edge     it touches a boundary between semantic classes (wall-floor
                  junction, a fixture or furniture outline)
  shading edge    a strong edge in the lighting (render / albedo, blurred) with
                  little edge in the albedo: cast shadows and light fall-off
  paint/material  a strong albedo edge: grout, marble veins, printed texture
  relief/other    none of these: bump and displacement texture, noise
The thresholds are heuristics; OUT/overlays shows the worst views for checking.
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "eval"))
sys.path.insert(0, str(APP))
from infer import load_model, predict  # noqa: E402

from syncrack3d.isp import process_rgb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("views", nargs="+", type=Path)
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--tolerance-px", type=int, default=5)
ap.add_argument("--min-blob-px", type=int, default=5)
ap.add_argument("--shadows", type=Path, default=None,
                help="render_room_views.py --shadow-only output (one subfolder per room, same poses)")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "overlays").mkdir(exist_ok=True)

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def camera(hdr, seed, awb=0.25, ev=0.85):
    x = hdr.astype(np.float32)
    grey = x.reshape(-1, 3).mean(0)
    x = x * (1 - awb + awb * grey.mean() / np.clip(grey, 1e-6, None))
    x = x * (0.18 * 2 ** ev / max(float((x @ LUM).mean()), 1e-6))
    x = np.clip(x, 0, 1)
    srgb = (np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) * 255).round().astype(np.uint8)
    bgr, _ = process_rgb(cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR), seed)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_linear(u8):
    c = u8.astype(np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def edge(img):
    g = cv2.GaussianBlur(img, (0, 0), 2.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8


models = [load_model(ck) for ck in sorted(args.model.glob("runs/s*/last.pt"))]
if not models:
    raise SystemExit(f"no runs/s*/last.pt under {args.model}")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.tolerance_px + 1,) * 2)
SURFACES = ("test_wall", "side_wall", "back_wall", "wall", "floor", "ceiling")
rows, blobs = [], []
shadow_px = {"pt": [0, 0, 0, 0], "rt": [0, 0, 0, 0]}  # near-outline px, hits; away px, hits
for vdir in args.views:
    info = json.loads((vdir / "views.json").read_text())
    room = Path(info["export"]).name
    for i, pose in enumerate(info["poses"]):
        sem = np.load(vdir / f"{i:03d}_semantic.npy").astype(np.int64)
        names = {int(k): v for k, v in info["labels"][str(i)].items()}
        cls = np.vectorize(lambda k: names.get(k, "unlabelled"))(np.unique(sem))
        lut = dict(zip(np.unique(sem), cls))
        crack_gt = np.isin(sem, [k for k, v in lut.items() if v == "crack"])
        near = cv2.dilate(crack_gt.astype(np.uint8), kernel) > 0
        band = outline = None
        if args.shadows is not None:
            shadow = np.load(args.shadows / room / f"{i:03d}_shadow.npy").astype(bool)
            surface = np.isin(sem, [k for k, v in lut.items() if v in SURFACES])
            shadow &= surface
            boundary = (cv2.dilate(shadow.astype(np.uint8), np.ones((3, 3), np.uint8))
                        != cv2.erode(shadow.astype(np.uint8), np.ones((3, 3), np.uint8)))
            band = (cv2.dilate(boundary.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & surface & ~near
            outline = cv2.dilate((cv2.Laplacian(sem.astype(np.float32), cv2.CV_32F) != 0).astype(np.uint8),
                                 np.ones((11, 11), np.uint8)) > 0
        albedo = np.load(vdir / f"{i:03d}_albedo.npy")
        a_lum = to_linear(albedo) @ LUM
        for mode in ("pt", "rt"):
            hdr = np.load(vdir / f"{i:03d}_{mode}.npy")
            img = camera(hdr, seed=zlib.crc32(f"{room}_{i}_{mode}".encode()))
            votes = sum((predict(m, img) == 1).astype(np.int32) for m in models)
            pred = votes * 2 >= len(models)
            fp = pred & ~near
            shading = np.log(np.clip(hdr.astype(np.float32) @ LUM, 1e-6, None) / np.clip(a_lum, 1e-3, None))
            e_shade, e_alb = edge(shading), edge(np.log(np.clip(a_lum, 1e-3, None)))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(fp.astype(np.uint8), connectivity=8)
            vb = []
            for b in range(1, n):
                if stats[b, cv2.CC_STAT_AREA] < args.min_blob_px:
                    continue
                m = lab == b
                ring = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                classes, counts = np.unique(sem[m], return_counts=True)
                surface = lut[classes[counts.argmax()]]
                if band is not None and band[m].mean() >= 0.5:
                    cause = "cast shadow"
                elif len(np.unique(sem[ring])) > 1:
                    cause = "object edge"
                elif np.percentile(e_shade[ring], 90) > 0.08 and np.percentile(e_alb[ring], 90) < 0.04:
                    cause = "shading edge"
                elif np.percentile(e_alb[ring], 90) >= 0.04:
                    cause = "paint/material"
                else:
                    cause = "relief/other"
                vb.append(dict(room=room, view=i, mode=mode, surface=surface, cause=cause,
                               px=int(stats[b, cv2.CC_STAT_AREA])))
            blobs.extend(vb)
            if band is not None:
                for j, region in ((0, band & outline), (2, band & ~outline)):
                    shadow_px[mode][j] += int(region.sum())
                    shadow_px[mode][j + 1] += int((pred & region).sum())
            visible = int(crack_gt.sum())
            rows.append(dict(room=room, view=i, mode=mode, kind=pose["kind"], crack_px=visible,
                             crack_hit=bool((pred & near).sum() >= 10) if visible >= 30 else None,
                             fp_px=int(fp.sum()), fp_blobs=len(vb),
                             tp_px=int((pred & near).sum())))
            if vb or (visible >= 30 and mode == "pt"):
                over = img.copy()
                over[near] = (0.5 * over[near] + (0, 127, 0)).astype(np.uint8)
                over[fp] = (255, 0, 0)
                over[pred & near] = (255, 255, 0)
                cv2.imwrite(str(args.out / "overlays" / f"{room}_{i:03d}_{mode}.jpg"),
                            cv2.cvtColor(np.concatenate([img, over], 1), cv2.COLOR_RGB2BGR))
        print(room, i, flush=True)

summary = {}
for mode in ("pt", "rt"):
    r = [x for x in rows if x["mode"] == mode]
    b = [x for x in blobs if x["mode"] == mode]
    vis = [x for x in r if x["crack_hit"] is not None]
    summary[mode] = dict(
        views=len(r),
        views_with_false_alarm=round(sum(x["fp_blobs"] > 0 for x in r) / len(r), 3),
        false_alarm_blobs_per_view=round(len(b) / len(r), 3),
        false_alarm_px_per_view=round(sum(x["fp_px"] for x in r) / len(r), 1),
        by_cause={c: sum(x["cause"] == c for x in b) for c in sorted({x["cause"] for x in b})},
        by_surface={s: sum(x["surface"] == s for x in b) for s in sorted({x["surface"] for x in b})},
        by_room={room: round(sum(x["fp_blobs"] > 0 for x in r if x["room"] == room)
                             / max(1, sum(x["room"] == room for x in r)), 3) for room in sorted({x["room"] for x in r})},
        by_pose={k: round(sum(x["fp_blobs"] > 0 for x in r if x["kind"] == k)
                          / max(1, sum(x["kind"] == k for x in r)), 3) for k in ("test_wall", "anywhere")},
        crack_visible_views=len(vis),
        crack_hit_rate=round(sum(x["crack_hit"] for x in vis) / max(1, len(vis)), 3),
        precision_px=round(sum(x["tp_px"] for x in r) / max(1, sum(x["tp_px"] + x["fp_px"] for x in r)), 3))
    if args.shadows is not None:
        n_near, h_near, n_away, h_away = shadow_px[mode]
        summary[mode]["shadow_boundary"] = dict(
            px_near_outline=n_near, crack_rate_near_outline=round(h_near / max(n_near, 1), 4),
            px_away_from_outline=n_away, crack_rate_away_from_outline=round(h_away / max(n_away, 1), 4),
            note="detector crack rate on cast-shadow boundaries; near = within 5 px of an object outline")
(args.out / "views.json").write_text(json.dumps(rows, indent=1))
(args.out / "blobs.json").write_text(json.dumps(blobs, indent=1))
(args.out / "summary.json").write_text(json.dumps(dict(model=str(args.model), tolerance_px=args.tolerance_px,
                                                       summary=summary), indent=1))
print(json.dumps(summary, indent=1))
