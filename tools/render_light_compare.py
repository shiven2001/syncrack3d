"""One crack, three ways of building it, under a set of lights.

Builds one cracked wall three times with --crack-render albedo (a decal: colour
only), pbr (bump/normal map only) and geometry (real displacement), renders the
same close-up camera under each light of a set, and lays the tiles out as one
sheet. `--lights standard` runs from a raking light across the crack to a dim
room at high ISO; `--lights hard` keeps the raking light as the reference and
adds the conditions that make a crack hard to find: head-on flash, light along
the crack, a cable's shadow beside it, window-blind stripes, overexposure and a
dim room, plus the camera pulled back 2x and 4x; `--lights far` puts those bad
lights on the camera pulled back 4x. `--wall hairline` builds a single thin crack (the legacy recipe at
--crack-width-scale .4 on neutral paint) instead of the unified wall. Each tile
is scored two ways against the crack label rendered with it (the geometry
render's DefectCrack pass at 0.5 coverage, so a pulled-back camera has its own):

* detector F1: the v1.20 U-Net (five seeds, BD3 protocol: 1.5x upscaling,
  centreline F1 at 5 px), averaged over seeds. It was trained on geometry
  renders only, so decal and PBR tiles are new to it.
* crack/texture: how far crack pixels stand from the wall's median brightness,
  over how far the wall's own texture does in a 6-20 px ring around the crack.
  1 means the crack is no more visible than the plaster grain.

    PYTHONPATH=. /home/shiven/anaconda3/bin/python tools/render_light_compare.py \
        [--seed 930032] [--wall hairline] [--lights hard|far]

Writes outputs/light_compare/seed_<seed>[_hairline]/: walls_<mode>/ (the three
built walls), tiles[_<set>]/<mode>_<light>.png, light_compare[_<set>].png
(renders), light_compare[_<set>]_detections.png (seed 0's crack pixels in red)
and scores[_<set>].json. Steps that are done are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "light_compare"  # one folder per seed below this
BPY = "/home/shiven/anaconda3/envs/infinigen_v2/bin/python"
# indoor plaster walls: 930022 has one wide major crack, 930032 thinner legacy cracks
SEED = 930022
MODEL = REPO / "outputs" / "domain_transfer" / "syncrack3d_v120mc_unet_w250"
MODES = (("albedo", "Decal (colour only)"), ("pbr", "PBR (bump map)"), ("geometry", "Geometry (displaced)"))
TILE = 640

# name, caption, key light (incidence from the wall normal, azimuth around it:
# 0 = from the side, 90 = from above), size m, distance m, power W, world strength,
# EV compensation. The dim room is lit so faintly that metering runs out of
# shutter and raises ISO, which is what makes it noisy. Cracks here run roughly vertically, so a light from the side rakes
# across the groove and one from above rakes along it.
LIGHT_SETS = {
    "standard": (
        ("raking_across", "Raking light across the crack (80 deg)", 80, 0, .10, 1.2, 60, .02, 0.0),
        ("window", "Window light (55 deg)", 55, 20, 1.6, 2.4, 160, .05, 0.0),
        ("ceiling", "Soft ceiling light (25 deg)", 25, 90, 1.2, 1.8, 120, .10, 0.0),
        ("raking_along", "Raking light along the crack (80 deg)", 80, 90, .10, 1.2, 60, .02, 0.0),
        ("flash", "Phone flash, head-on (0 deg)", 0, 0, .03, 0.0, 2, .00, 0.0),
        ("dim", "Dim room, high ISO noise", 25, 90, 1.2, 1.8, .03, .00004, 0.0),
    ),
    # Shadow lights are a 1 cm bulb, so the occluder's shadow edge is sharp.
    "hard": (
        ("raking_across", "Reference: raking light across the crack", 80, 0, .10, 1.2, 60, .02, 0.0),
        ("flash", "Phone flash, head-on (0 deg)", 0, 0, .03, 0.0, 2, .00, 0.0),
        ("raking_along", "Grazing light along the crack (85 deg)", 85, 90, .05, 1.2, 60, .02, 0.0),
        ("cable_shadow", "Shadow of a cable beside the crack", 45, 20, .01, 1.0, 40, .02, 0.0),
        ("blind_shadow", "Window-blind shadow stripes", 50, 70, .01, 1.2, 60, .02, 0.0),
        ("overexposed", "Overexposed (+2.5 EV)", 55, 20, 1.6, 2.4, 160, .05, 2.5),
        ("dim", "Dim room, high ISO noise", 25, 90, 1.2, 1.8, .03, .00004, 0.0),
        # same window light, the camera pulled back along its own view line
        ("far2", "Camera 2x farther (window light)", 55, 20, 1.6, 2.4, 160, .05, 0.0),
        ("far4", "Camera 4x farther (window light)", 55, 20, 1.6, 2.4, 160, .05, 0.0),
    ),
    # every column from the camera pulled back 4x, after the close-up reference
    "far": (
        ("window", "Close-up, window light (reference)", 55, 20, 1.6, 2.4, 160, .05, 0.0),
        ("far4_window", "4x farther, window light", 55, 20, 1.6, 2.4, 160, .05, 0.0),
        ("far4_overexposed", "4x farther, overexposed (+2.5 EV)", 55, 20, 1.6, 2.4, 160, .05, 2.5),
        ("far4_overexposed_more", "4x farther, overexposed (+3.5 EV)", 55, 20, 1.6, 2.4, 160, .05, 3.5),
        ("far4_flash", "4x farther, phone flash", 0, 0, .03, 0.0, 2, .00, 0.0),
        ("far4_raking_along", "4x farther, grazing light along the crack", 85, 90, .05, 1.2, 60, .02, 0.0),
        ("far4_blind_shadow", "4x farther, window-blind stripes", 50, 70, .01, 1.2, 60, .02, 0.0),
        ("far4_dim", "4x farther, dim room, high ISO", 25, 90, 1.2, 1.8, .03, .00004, 0.0),
    ),
}
LIGHTS = LIGHT_SETS["standard"]
WALLS = {
    "unified": ["--recipe", "unified"],
    "hairline": ["--recipe", "legacy", "--material-category", "paint_neutral", "--clean-wall-scale", "0",
                 "--crack-width-scale", ".4", "--min-defects", "1", "--max-defects", "1"],
}
WALL = "unified"
SUFFIX = ""  # "" for the standard lights, "_hard" for the hard set


def build_walls() -> None:
    for mode, _ in MODES:
        out = OUT / f"walls_{mode}"
        if (out / f"seed_{SEED}" / "scene.blend").exists():
            continue
        # labels are identical across modes, so only geometry renders its frames (for the crack mask)
        extra = [] if mode == "geometry" else ["--no-render"]
        subprocess.run([BPY, "syncrack3d/build_dataset.py", *WALLS[WALL], "--seeds", str(SEED),
                        "--output", str(out), "--crack-render", mode, "--save-blend", *extra],
                       cwd=REPO, check=True)


def pick_camera() -> tuple[int, str]:
    """The close-up whose frame fill is nearest 1 on the first crack."""
    meta = json.loads((OUT / "walls_geometry" / f"seed_{SEED}" / "scene.json").read_text())
    cams = [c for c in meta["cameras"] if c["role"] == "closeup"]
    cam = min(cams, key=lambda c: abs(c["frame_fill"] - 1.0))
    return cam["index"], cam["object_name"]


def render_tiles(mode: str) -> None:
    """Runs inside Blender: open the wall, then one render per light."""
    import bpy
    from mathutils import Vector

    sys.path.insert(0, str(REPO))
    from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
    from cwg_infinigen.core.rendering.sensor_response import apply_sensor_response
    from syncrack3d.lighting import LightingPlan, LightSpec, _placement, apply_lighting
    from syncrack3d.render import configure_cycles

    scene_dir = OUT / f"walls_{mode}" / f"seed_{SEED}"
    meta = json.loads((scene_dir / "scene.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(scene_dir / "scene.blend"))
    configure_cycles(resolution=TILE, samples=128, dicing_rate=.75)
    scene = bpy.context.scene
    _, cam_name = pick_camera()
    cam = bpy.data.objects[cam_name]
    scene.camera = cam
    scene.frame_set(1)
    # aim every light at the point the camera looks at on the wall (y = 0)
    origin = cam.matrix_world.translation.copy()  # the matrix is live; the camera moves below
    ray = cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    target = tuple(origin + ray * (-origin.y / ray.y))
    collection = bpy.data.collections.get("CWG_CAPTURE") or scene.collection
    tiles = OUT / f"tiles{SUFFIX}"
    tiles.mkdir(parents=True, exist_ok=True)

    home = cam.location.copy()
    for name, _, inc, azi, size, dist, power, world, ev in LIGHTS:
        path = tiles / f"{mode}_{name}.png"
        if path.exists():
            continue
        # "far4" or "far4_<light>": the camera pulled back 4x, under that light
        far = re.match(r"far(\d+)_?(.*)", name)
        back, light = (int(far.group(1)), far.group(2) or "window") if far else (1, name)
        cam.location = home - ray.normalized() * (Vector(target) - origin).length * (back - 1)
        if light == "flash":  # on the lens axis, a few cm above the lens
            location = (cam.location.x, cam.location.y, cam.location.z + .03)
        else:
            location = _placement(meta["spec"]["width_m"], meta["spec"]["height_m"], inc, azi, dist, target)
        point = light == "flash" or light.endswith("_shadow")
        key = LightSpec("key", name, 5500.0, power, size, float(inc), float(azi), tuple(location),
                        light_type="POINT" if point else "AREA", target=target)
        apply_lighting(LightingPlan(name, world, (key,)), meta["spec"]["height_m"], collection)
        occluders = _occluders(light, Vector(location), Vector(target))
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        layers = tree.nodes.new("CompositorNodeRLayers")
        rgb = apply_sensor_response(NodeWrangler(tree), layers.outputs["Image"], enabled=True, auto_exposure=True,
                                    ev_compensation=ev, noise=True, noise_seed=SEED)
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.base_path, node.format.file_format, node.format.color_mode = str(tiles), "PNG", "RGB"
        node.file_slots[0].path = path.stem
        tree.links.new(rgb, node.inputs[0])
        if mode == "geometry":  # the crack label from this camera, shared by all three modes
            label = tree.nodes.new("CompositorNodeOutputFile")
            label.base_path, label.format.file_format, label.format.color_mode = str(tiles), "PNG", "BW"
            label.file_slots[0].path = f"{path.stem}_label"
            tree.links.new(layers.outputs["DefectCrack"], label.inputs[0])
        bpy.ops.render.render(write_still=False)
        for stem, final in ((path.stem, path), (f"{path.stem}_label", tiles / f"{name}_label.png")):
            written = sorted(tiles.glob(f"{stem}0*.png"), key=lambda p: p.stat().st_mtime)
            if written:
                written[-1].replace(final)
        for obj in occluders:
            bpy.data.objects.remove(obj, do_unlink=True)
        print("WROTE", path, flush=True)


def _occluders(name, light, target):
    """Objects between the bulb and the wall that only cast shadows (the camera cannot see them).

    cable_shadow: a 3 mm cable, 20 deg off vertical, a fifth of the way to the
    bulb, placed so its shadow runs about 4 cm beside the crack. blind_shadow:
    1.2 cm slats at a 2.4 cm pitch, a third of the way to the bulb.
    """
    import bpy
    from mathutils import Vector

    made = []
    if name == "cable_shadow":
        at = target + (light - target) * .2 + Vector((.04 * .8, 0.0, 0.0))
        bpy.ops.mesh.primitive_cylinder_add(radius=.0015, depth=1.2, location=at,
                                            rotation=(0.0, math.radians(20.0), 0.0))
        made.append(bpy.context.object)
    elif name == "blind_shadow":
        at = target + (light - target) / 3.0
        for k in range(-12, 13):
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=at + Vector((0.0, 0.0, .024 * k)))
            slat = bpy.context.object
            slat.scale = (.9, .002, .012)
            made.append(slat)
    for obj in made:
        obj.visible_camera = False
        obj.visible_glossy = False
    return made


def crack_mask(shape, name=None):
    """The crack label for one light's tile, or the build's own frame for the camera."""
    import cv2

    tile_label = OUT / f"tiles{SUFFIX}" / f"{name}_label.png"
    if name is not None and tile_label.exists():
        return cv2.resize(cv2.imread(str(tile_label), 0), shape[::-1], interpolation=cv2.INTER_AREA) >= 128
    index, _ = pick_camera()
    frames = OUT / "walls_geometry" / "frames" / f"seed_{SEED}" / "DefectCrack" / "camera_0"
    mask = cv2.imread(str(next(frames.glob(f"DefectCrack_{index}_0_*.png"))), cv2.IMREAD_UNCHANGED)
    mask = mask.max(axis=2) if mask.ndim == 3 else mask
    return cv2.resize(mask, shape[::-1], interpolation=cv2.INTER_AREA) >= 128  # coverage 0.5, as the export


def texture_ratio(img, mask) -> float:
    """Crack pixels' distance from the wall's median brightness, over the wall texture's own (6-20 px ring)."""
    import cv2
    import numpy as np

    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    m = mask.astype("uint8")
    ring = (cv2.dilate(m, np.ones((41, 41), "uint8")) > 0) & ~(cv2.dilate(m, np.ones((13, 13), "uint8")) > 0)
    if not mask.any() or not ring.any():
        return math.nan
    wall = np.median(lum[ring])
    texture = np.abs(lum[ring] - wall).mean()
    return math.nan if texture < .5 else np.abs(lum[mask] - wall).mean() / texture  # nan: a flat (clipped) wall


def detect(images: dict, masks: dict) -> tuple[dict, dict]:
    """Mean centreline F1 over the model's seeds per tile, and seed 0's predictions."""
    import torch

    sys.path.insert(0, str(REPO / "eval"))
    from infer import load_model, predict
    from metrics import TOLERANCE_PX, centreline_counts, centreline_scores

    f1 = {key: [] for key in images}
    first = {}
    for ck in sorted(MODEL.glob("runs/s*/last.pt")):
        model = load_model(ck)
        for key, img in images.items():
            pred = predict(model, img[:, :, ::-1].copy()) == 1  # BGR -> RGB; class 1 is crack
            f1[key].append(centreline_scores(centreline_counts(pred, masks[key], TOLERANCE_PX))["f1"])
            first.setdefault(key, pred)
        del model
        torch.cuda.empty_cache()
    return {key: sum(v) / len(v) for key, v in f1.items()}, first


def _sheet(tiles: dict, captions: dict, path: Path) -> None:
    import cv2
    import numpy as np

    h, w = next(iter(tiles.values())).shape[:2]
    label_w, head_h, cap_h, pad = 230, 70, 62, 8
    sheet = np.full((head_h + len(MODES) * (h + cap_h + pad), label_w + len(LIGHTS) * (w + pad), 3), 255, "uint8")
    font = cv2.FONT_HERSHEY_SIMPLEX
    for j, (_, caption, *_rest) in enumerate(LIGHTS):
        x = label_w + j * (w + pad)
        lines, line = [], ""
        for word in caption.split():  # wrap the column heading to the tile width
            if line and cv2.getTextSize(line + " " + word, font, .7, 2)[0][0] > w - 10:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        lines.append(line)
        for k, text in enumerate(lines):
            cv2.putText(sheet, text, (x + 4, 28 + 28 * k), font, .7, (0, 0, 0), 2, cv2.LINE_AA)
    for i, (mode, label) in enumerate(MODES):
        y = head_h + i * (h + cap_h + pad)
        for k, text in enumerate(label.replace(" (", "\n(").split("\n")):
            cv2.putText(sheet, text, (10, y + h // 2 + 28 * k), font, .75, (0, 0, 0), 2 if k == 0 else 1, cv2.LINE_AA)
        for j, (name, *_rest) in enumerate(LIGHTS):
            x = label_w + j * (w + pad)
            sheet[y:y + h, x:x + w] = tiles[f"{mode}/{name}"]
            cv2.rectangle(sheet, (x - 1, y - 1), (x + w, y + h), (150, 150, 150), 1)  # a blown-out tile stays visible
            for k, (text, colour) in enumerate(captions[f"{mode}/{name}"]):
                cv2.putText(sheet, text, (x + 4, y + h + 26 + 28 * k), font, .7, colour, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet)


def make_sheets() -> Path:
    import cv2
    import numpy as np

    images = {f"{mode}/{name}": cv2.imread(str(OUT / f"tiles{SUFFIX}" / f"{mode}_{name}.png"))
              for mode, _ in MODES for name, *_ in LIGHTS}
    masks = {key: crack_mask(img.shape[:2], key.split("/")[1]) for key, img in images.items()}
    f1, preds = detect(images, masks)
    ratio = {key: texture_ratio(img, masks[key]) for key, img in images.items()}

    def grade(value, good, fair):
        return (0, 130, 0) if value >= good else (0, 130, 230) if value >= fair else (0, 0, 210)

    captions = {key: [(f"detector F1 {f1[key]:.2f}", grade(f1[key], .7, .4)),
                      (f"crack/texture {ratio[key]:.1f}" if ratio[key] == ratio[key] else "crack/texture n/a (flat)",
                       grade(ratio[key], 3, 1.5))] for key in images}
    _sheet(images, captions, OUT / f"light_compare{SUFFIX}.png")
    overlays = {}
    for key, img in images.items():
        shown = img.copy()
        shown[preds[key]] = (0.35 * shown[preds[key]] + 0.65 * np.array([40, 40, 255])).astype("uint8")
        overlays[key] = shown
    _sheet(overlays, captions, OUT / f"light_compare{SUFFIX}_detections.png")
    (OUT / f"scores{SUFFIX}.json").write_text(json.dumps(
        {"model": str(MODEL), "camera": pick_camera()[1],
         "tiles": {key: {"detector_f1": round(f1[key], 3), "crack_texture_ratio": round(ratio[key], 2),
                         "label_px": int(masks[key].sum())}
                   for key in images}}, indent=1) + "\n")
    return OUT / f"light_compare{SUFFIX}.png"


def main() -> None:
    global SEED, OUT, LIGHTS, WALL, SUFFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wall", choices=sorted(WALLS), default=WALL)
    parser.add_argument("--lights", choices=sorted(LIGHT_SETS), default="standard")
    parser.add_argument("--render-mode", help=argparse.SUPPRESS)  # internal: the Blender step for one mode
    args = parser.parse_args()
    SEED, WALL, LIGHTS = args.seed, args.wall, LIGHT_SETS[args.lights]
    SUFFIX = "" if args.lights == "standard" else f"_{args.lights}"
    OUT = OUT / (f"seed_{SEED}" if WALL == "unified" else f"seed_{SEED}_{WALL}")
    if args.render_mode:
        render_tiles(args.render_mode)
        return
    build_walls()
    for mode, _ in MODES:
        if not all((OUT / f"tiles{SUFFIX}" / f"{mode}_{name}.png").exists() for name, *_ in LIGHTS):
            subprocess.run([BPY, __file__, "--seed", str(SEED), "--wall", WALL, "--lights", args.lights,
                            "--render-mode", mode], cwd=REPO, check=True)
    print(make_sheets())


if __name__ == "__main__":
    main()
