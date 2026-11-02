"""Rebuild the v1.20 showcase sheets for seeds 920000, 920009, 920012, 920014.

Each sheet is the wide room or facade view plus the first close-up of the
first two cracks. The walls themselves are the unified recipe at 1024px.

    PYTHONPATH=. python tools/render_showcase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import syncrack3d.build_dataset as build_dataset  # noqa: E402

OUT = REPO / "outputs" / "showcase_v1.20"
WALLS = OUT / "walls"
WIDE = OUT / "wide"
SEEDS = (920000, 920009, 920012, 920014)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BG = (244, 244, 244)


def _wide_camera(record: dict) -> None:
    spec = record["spec"]
    width, height = float(spec["width_m"]), float(spec["height_m"])
    if record["environment"] == "indoor":
        depth = float(record["room_shell"]["depth_m"])
        location = (-0.12 * width, -(depth - 0.30), 0.55 * height)
        aim_at = (0.0, 0.0, 0.40 * height)
        lens = 16.0
    else:
        corner = (record.get("outdoor") or {}).get("corner")
        if corner == "left":
            x = -0.22 * width
        elif corner == "right":
            x = 0.22 * width
        else:
            x = -0.08 * width
        location = (x, -max(3.2, 0.95 * width), 0.42 * height)
        aim_at = (0.0, 0.0, 0.46 * height)
        lens = 24.0
    aim = bpy.data.objects.new("CWG_Showcase_Aim", None)
    aim.location = aim_at
    bpy.context.scene.collection.objects.link(aim)
    data = bpy.data.cameras.new("CWG_Showcase_Cam")
    data.lens = lens
    data.sensor_width = 36.0
    data.clip_start = 0.05
    data.clip_end = 80.0
    cam = bpy.data.objects.new("CWG_Showcase_Cam", data)
    cam.location = location
    track = cam.constraints.new("TRACK_TO")
    track.target = aim
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam


def _render_wide(seed: int, record: dict) -> Path:
    from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
    from cwg_infinigen.core.rendering.sensor_response import (
        apply_auto_white_balance,
        apply_sensor_response,
    )

    context = next((i for i, cam in enumerate(record["cameras"]) if cam["role"] == "context"), 0)
    exposure = record["camera_exposure_ev"][context]
    awb = (record.get("camera_awb_strength") or [1.0])[context]
    _wide_camera(record)
    scene = bpy.context.scene
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1200
    scene.view_settings.exposure = 0.0
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    nw = NodeWrangler(tree)
    rgb = apply_auto_white_balance(nw, layers.outputs["Image"], strength=awb)
    rgb = apply_sensor_response(
        nw, rgb, enabled=True, auto_exposure=True, ev_compensation=exposure,
    )
    path = WIDE / f"seed_{seed}_full.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in path.parent.glob(f"seed_{seed}_full*.png"):
        old.unlink()
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = str(path.parent)
    out.format.file_format = "PNG"
    out.format.color_mode = "RGB"
    out.file_slots[0].path = path.stem
    tree.links.new(rgb, out.inputs[0])
    bpy.ops.render.render(write_still=False)
    written = sorted(path.parent.glob(f"{path.stem}*.png"), key=lambda item: item.stat().st_mtime)
    if not written:
        raise RuntimeError(f"no wide render for seed {seed}")
    written[-1].replace(path)
    return path


def _closeups(seed: int, record: dict) -> list[Path]:
    frames = WALLS / "frames" / f"seed_{seed:05d}" / "Image" / "camera_0"
    chosen = []
    for defect in (0, 1):
        for index, cam in enumerate(record["cameras"]):
            if cam["role"] == "closeup" and cam.get("target_index") == defect:
                path = frames / f"Image_{index}_0_0001_0.png"
                if path.is_file():
                    chosen.append(path)
                break
    if len(chosen) < 2:
        for index, cam in enumerate(record["cameras"]):
            if cam["role"] != "closeup":
                continue
            path = frames / f"Image_{index}_0_0001_0.png"
            if path.is_file() and path not in chosen:
                chosen.append(path)
            if len(chosen) == 2:
                break
    return chosen[:2]


def _caption(draw, image, text, font) -> None:
    x, y = 12, image.height - 12
    draw.text((x, y), text, font=font, fill=(255, 255, 255), anchor="ls",
              stroke_width=2, stroke_fill=(0, 0, 0))


def _sheet(seed: int, record: dict, wide: Path, letters: dict[str, int]) -> Path:
    env = record["environment"]
    letters[env] = letters.get(env, 0) + 1
    letter = chr(ord("A") + letters[env] - 1)
    diverse = (record.get("wall_material") or {}).get("diverse") or {}
    material = diverse.get("paint_band") or diverse.get("category") or "wall"
    kinds = []
    for structure in record.get("structures") or []:
        kind = structure.get("kind")
        if kind and kind not in kinds:
            kinds.append(kind)
    header_font = ImageFont.truetype(FONT, 22)
    caption_font = ImageFont.truetype(FONT, 18)
    sheet = Image.new("RGB", (1690, 960), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 18), f"{env} {letter} | seed {seed} | {material} | fixtures-first",
              font=header_font, fill=(40, 40, 40))
    full = Image.open(wide).convert("RGB").resize((1200, 900), Image.Resampling.LANCZOS)
    _caption(ImageDraw.Draw(full), full, "full view - fixtures: " + ", ".join(kinds), caption_font)
    sheet.paste(full, (10, 50))
    for slot, path in enumerate(_closeups(seed, record)):
        tile = Image.open(path).convert("RGB").resize((440, 440), Image.Resampling.LANCZOS)
        _caption(ImageDraw.Draw(tile), tile, f"close-up, crack {slot}", caption_font)
        sheet.paste(tile, (1230, 50 + slot * 460))
    out = OUT / f"sheet_{seed}.png"
    sheet.save(out)
    return out


def _after_build(seed, args, out_root):
    record = _after_build.original(seed, args, out_root)
    wide = _render_wide(seed, record)
    sheet = _sheet(seed, record, wide, _after_build.letters)
    print(sheet, flush=True)
    return record


def main() -> None:
    WIDE.mkdir(parents=True, exist_ok=True)
    _after_build.original = build_dataset.build_one
    _after_build.letters = {}
    build_dataset.build_one = _after_build
    sys.argv = [
        "build_dataset",
        "--recipe", "unified",
        "--seeds", ",".join(str(seed) for seed in SEEDS),
        "--output", str(WALLS),
        "--resolution", "1024",
        "--samples", "128",
    ]
    raise SystemExit(build_dataset.main())


if __name__ == "__main__":
    main()
